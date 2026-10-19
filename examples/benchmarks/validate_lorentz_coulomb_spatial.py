"""Physical Coulomb spatial validation with resolved full Lorentz motion.

Initially monoenergetic 5T beam, pitch .6; compare actual guiding-centre
displacement with an independent finite-Omega backward correlation reference.
Two gyrofrequencies and fixed/adaptive caps separate orbit resolution from
collision integration. No surrogate bath, source or thermal stop is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")
os.environ.setdefault("ESSOS_XLA_CACHE", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np

from essos.collisions import particle_collision_step
from essos.split import solve_split
from .thermal_relaxation_reference import PhysicalRelaxationReference, RelaxationReference, impact_coefficients
from .lorentz_spatial_reference import guiding_center_msd
from .validate_beam_relaxation import physical_native_coefficients, estimate
from .validate_collision_physics import run_batches, work


def solver(method, horizon, times, *, coulomb_log_model="nrl"):
    law = physical_native_coefficients(coulomb_log_model)
    adaptive = method != "fixed"
    def batch(initial, keys, tolerance, cap, omega):
        def one(y0, key):
            def rhs(t, y, args):
                v = y[3:]
                return jnp.concatenate((v, omega*jnp.cross(v, jnp.array([0., 0., 1.]))))
            def kick(t, y, h, dw, args):
                c = law(jnp.linalg.norm(y[3:]), full_orbit=True)
                result = particle_collision_step(y[3:], h, dw, c, tolerance=tolerance)
                return result._replace(y=y.at[3:].set(result.y))
            return solve_split(rhs, kick, y0, t1=horizon, dt0=cap, dtmax=cap,
                ts=jnp.asarray(times), key=key, noise_dim=3, adaptive=adaptive,
                collision_controller="conservative", collision_wiener_norm=lambda w: abs(w[2]),
                method="dopri8", rtol=1e-10, atol=1e-10, max_steps=100_000)
        return jax.vmap(one)(initial, keys)
    return jax.jit(batch)


def plot(report, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    times = np.asarray(report["metadata"]["times"])
    for i, omega in enumerate(report["metadata"]["omegas"]):
        ax = axes[i]
        ax.plot(times, report["references"][str(omega)]["msd"], "k", lw=2, label="Finite-Omega Coulomb reference")
        for row in report["rows"]:
            if row["omega"] != omega:
                continue
            color = "C0" if row["method"] == "fixed" else "C1"
            fine = row["setting"] == "fine"
            value = np.array([r["value"] for r in row["evolution"]])
            half = np.array([r["half95"] for r in row["evolution"]])
            ax.plot(times, value, color=color, ls="-" if fine else ":", label=f"{row['method']}/{row['setting']}")
            if fine:
                ax.fill_between(times, value-half, value+half, color=color, alpha=.1)
            if fine:
                final = row["evolution"][-1]
                x = 2*i+(row["method"] != "fixed")
                axes[2].errorbar(x, 100*final["difference"]/final["reference"],
                    yerr=100*final["half95"]/final["reference"], marker="o", color=color, capsize=4)
        ax.set(title=rf"Full Lorentz: $\Omega/\nu_0={omega:g}$", xlabel=r"Time $t\nu_0$", ylabel=r"$E[|\Delta X_\perp|^2]/\rho_\sigma^2$")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].axhline(0., color="k", ls="--")
    axes[2].set_xticks(range(4), [f"{o:g}/{m}" for o in report["metadata"]["omegas"] for m in ("fixed", "adaptive")], rotation=15)
    axes[2].set(title="Refined final spatial MSD", ylabel="Difference from finite-Omega reference (%)")
    axes[2].grid(alpha=.2)
    family = report['metadata'].get('coulomb_log_model', 'nrl')
    fig.suptitle(f"Coulomb model: {family}; full Lorentz motion; N={report['metadata']['particles']:,}; initially fast beam\n"
        "Two timestep caps and gyrofrequencies; black = independent correlation PDE; shading/bars = marginal 95% MC intervals")
    fig.savefig(output.with_suffix(".png"), dpi=170)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--coulomb-log-model", choices=("nrl", "impact"), default="nrl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    n, batch = args.particles, args.batch_size
    if not 2 <= n <= 2000 or not 2 <= batch <= n or n % batch:
        parser.error("use at most 2000 particles and a dividing batch size")
    times, omegas = np.array([0., .025, .05, .075, .1]), [32., 64.]
    metadata = dict(particles=n, batch_size=batch, seed=7706, times=times.tolist(), omegas=omegas,
        coulomb_log_model=args.coulomb_log_model,
        caps=[.0025, .000625], tolerance=.001,
        scope="Full Lorentz plus actual Coulomb collisions, monoenergetic anisotropic beam, no stopping or source",
        uncertainty="Independent particles; matched configurations share ordered keys, not exact Brownian paths")
    report = dict(complete=False, metadata=metadata, references={}, rows=[])
    if args.output.exists():
        if not args.resume:
            raise ValueError("output exists; use --resume")
        report = json.loads(args.output.read_text())
        report["metadata"].setdefault("coulomb_log_model", "nrl")
        if report["metadata"] != metadata:
            raise ValueError("resume metadata differs")
        report["complete"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    for omega in omegas:
        if str(omega) in report["references"]:
            continue
        cls = RelaxationReference if args.coulomb_log_model == "impact" else PhysicalRelaxationReference
        kwargs = dict(law=impact_coefficients) if args.coulomb_log_model == "impact" else {}
        coarse = guiding_center_msd(cls(nodes=384, **kwargs), times, omega)
        fine = guiding_center_msd(cls(nodes=768, **kwargs), times, omega)
        remote = guiding_center_msd(cls(nodes=960, upper=10., **kwargs), times, omega)
        refinement = float(max(np.max(abs(coarse-fine)), np.max(abs(remote-fine)))/fine[-1])
        if refinement > 3e-4:
            raise RuntimeError(f"finite-Omega spatial reference needs refinement: {omega}, {refinement}")
        report["references"][str(omega)] = dict(msd=fine.tolist(), max_relative_refinement=refinement)
        print(f"Omega={omega}: reference refinement {refinement:.3g}", flush=True)
        save()
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(7706), i))(jnp.arange(n)).reshape((-1, batch))
    velocity = np.sqrt(10.)*np.array([.8, 0., .6])
    for method in ("fixed", "conservative"):
        function, executable = solver(method, .1, times, coulomb_log_model=args.coulomb_log_model), None
        for omega in omegas:
            y0 = np.r_[-np.cross(velocity, [0., 0., 1.])/omega, velocity]
            initial = jnp.broadcast_to(jnp.asarray(y0), (n, 6)).reshape((-1, batch, 6))
            expected = report["references"][str(omega)]["msd"]
            for setting, cap in zip(("coarse", "fine"), metadata["caps"]):
                identifier = f"{method}/{omega}/{setting}"
                if any(r["id"] == identifier for r in report["rows"]):
                    continue
                params = tuple(map(jnp.asarray, (.001, cap, omega)))
                if executable is None:
                    executable = function.lower(initial[0], keys[0], *params).compile()
                print(f"Running {identifier}: N={n}", flush=True)
                output, elapsed = run_batches(executable, list(zip(initial, keys)), params)
                if np.any(output.status != 0):
                    raise RuntimeError("a physical spatial trajectory failed to reach the requested time")
                if method != "fixed" and output.stats.max_accepted_error.max() > 1:
                    raise RuntimeError("accepted spatial collision error exceeds one")
                guiding = omega*output.ys[:, :, :3]+np.cross(output.ys[:, :, 3:], [0., 0., 1.])
                msd = np.sum(guiding[:, :, :2]**2, axis=-1)
                row = dict(id=identifier, method=method, setting=setting, cap=cap, omega=omega,
                    elapsed_seconds=elapsed, **work(output),
                    evolution=[estimate(msd[:, i], target) for i, target in enumerate(expected)])
                report["rows"].append(row)
                save()
                print(json.dumps(dict(id=identifier, final=row["evolution"][-1])), flush=True)
    root = Path(__file__).resolve().parents[2]
    report["source_sha256"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
        "examples/benchmarks/validate_lorentz_coulomb_spatial.py", "examples/benchmarks/lorentz_spatial_reference.py",
        "examples/benchmarks/thermal_relaxation_reference.py", "essos/collisions.py", "essos/split.py", "essos/background_species.py",
        "essos/coulomb.py", "examples/benchmarks/nonrelativistic_collision_reference.py",
        "examples/benchmarks/validate_beam_relaxation.py")}
    report["complete"] = True
    save()
    plot(report, args.output)


if __name__ == "__main__":
    main()
