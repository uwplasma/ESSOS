"""Published QA AE3D/FIRM3D inputs with natural-cubic perturbation profiles.

Preparation checks/replots data without tracing. --run explicitly enables
the existing Boozer adaptive GC solver for 5000 archived markers over 10 ms.
Use --platform cuda for a GPU; --interpolation linear recovers the archived
FIRM3D radial profiles. Long runs require user approval in the shared workspace.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

import numpy as np

from examples.benchmarks.ae3d_firm3d_archive import MANIFEST


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT.parent/"artifacts/fax_integration/ae3d_firm3d_packet"
MASS = 6.69509884346e-27  # Exact paper-pinned FIRM3D: 2*mp+2*mn.
CHARGE = 3.204353268e-19
ENERGY = 5.63966175168e-13
TAG = "QA_35harmonics_scale0_001"


def parse_reference(path):
    """Read complete archived classification tables; missing categories fail."""
    rows = []
    for block in re.split(r"(?=OC tag :)", Path(path).read_text()):
        tag = re.search(r"OC tag :\s*(\S+)", block)
        if tag is None:
            continue
        counts = {}
        for category in ("lost_passing", "lost_barely_trapped", "lost_other_trapped",
                         "confined_passing", "confined_barely_trapped", "confined_other_trapped"):
            value = re.search(rf"^\s*{category}\s+(\d+)\b", block, flags=re.MULTILINE)
            if value is None:
                raise ValueError(f"Missing category {category}")
            counts[category] = int(value[1])
        total = re.search(r"^\s*Total\s+(\d+)\b", block, flags=re.MULTILINE)
        duration = re.search(r"tmax\s*=\s*(\S+)", block)
        if total is None or duration is None or sum(counts.values()) != int(total[1]):
            raise ValueError("Inconsistent archived particle count or duration")
        lost = sum(value for key, value in counts.items() if key.startswith("lost_"))
        label = tag[1]
        amplitude = 0. if label == "saw_zero" else float("0."+label.split("scale0_")[1])
        rows.append(dict(tag=label, amplitude=amplitude, particles=int(total[1]),
                         duration_s=float(duration[1]), lost=lost, loss_fraction=lost/int(total[1]), counts=counts))
    if not rows or len({row["tag"] for row in rows}) != len(rows):
        raise ValueError("Missing or repeated reference cases")
    return rows


def verify_packet(packet):
    manifest = json.loads(MANIFEST.read_text())
    for member in manifest["members"]:
        path = packet/member["name"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run examples.benchmarks.ae3d_firm3d_archive first")
        if hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
            raise ValueError(f"Reference hash mismatch: {member['name']}")
    return packet/"Knyazev_JPP2026"


def equilibrium(path, *, angular_points=48, mode_tolerance=1e-7):
    from netCDF4 import Dataset
    from essos.boozer import BoozerField, psi0_from_vmec

    with Dataset(path) as nc:
        count = int(nc["ns_b"][:])
        s = (np.asarray(nc["jlist"][:])-1.5)/(count-1)
        if not np.array_equal(nc["jlist"][:], np.arange(2, count+1)):
            raise ValueError("The benchmark requires all half-mesh Boozer surfaces")
        return BoozerField.from_booz(s, np.asarray(nc["bmnc_b"][:]).T,
            nc["ixm_b"][:], nc["ixn_b"][:], nc["iota_b"][1:], nc["bvco_b"][1:], nc["buco_b"][1:],
            psi0_from_vmec(nc["phi_b"][-1]), int(nc["nfp_b"][:]), mode_tolerance=mode_tolerance,
            angle_grid=None if angular_points is None else (angular_points, angular_points))


def birth_parameters(data, B):
    """Preserve archived v_parallel AND mu/m, rather than recomputing either."""
    velocity, mu = data[:, 3], data[:, 4]
    speed = np.sqrt(velocity**2+2*mu*B)
    return speed, velocity/speed


def prepare(packet, output, angular_points, mode_tolerance):
    import jax
    import jax.numpy as jnp
    from essos.ae3d_adapter import AE3DMode

    base = verify_packet(packet)
    inputs = base/"simulations_fig4_fig5/QA/all_inputs"
    source = inputs/"ae3d_saw_scaled"/f"{TAG}.npy"
    mode = AE3DMode.from_firm3d_numpy(source)
    references = parse_reference(base/"FIG4-orbit_loss_classification/inputs/QA.txt")
    selected = {row["tag"]: row for row in references if row["tag"] in ("saw_zero", TAG)}
    if len(selected) != 2 or any(row["particles"] != 5000 or row["duration_s"] != .01 for row in selected.values()):
        raise ValueError("The selected published case must have 5000 markers and a 10 ms duration")
    data = np.loadtxt(inputs/"ICs/fusion_ICs_5000.txt")
    if data.shape != (5000, 5) or not np.isfinite(data).all() or np.any(data[:, 4] < 0):
        raise ValueError("Invalid archived initial conditions")
    field = equilibrium(inputs/"boozmn_betaQA.nc", angular_points=angular_points, mode_tolerance=mode_tolerance)
    evaluate = jax.jit(jax.vmap(lambda p: field.modB(*p)))
    B = np.concatenate([np.asarray(evaluate(jnp.asarray(block))) for block in np.array_split(data[:, :3], 20)])
    if not np.all(np.isfinite(B) & (B > 0)):
        raise ValueError("Nonphysical reconstructed birth field")
    expected_B = (2*ENERGY/MASS-data[:, 3]**2)/(2*data[:, 4])
    if not np.all(np.isfinite(expected_B) & (expected_B > 0)):
        raise ValueError("Archived births are inconsistent with the pinned particle constants")
    speed, pitch = birth_parameters(data, B)
    birth_energy = .5*MASS*speed**2
    summary = dict(scope="Published QA AE3D/FIRM3D case; preparation is not a loss-agreement result",
        doi="10.5281/zenodo.21299908", paper="https://arxiv.org/abs/2603.03118",
        firm3d_commit="bc65bab877d88b6b00d6d248f54e5c75d893e6ac", mode_tag=TAG,
        omega_rad_s=float(mode.omega), cyclic_frequency_hz=float(mode.omega/(2*np.pi)),
        frequency_conversion="sqrt(AE3D eigenvalue)*1000 rad/s, following pinned FIRM3D; no extra 2*pi",
        harmonics=len(mode.harmonics), radial_samples=len(mode.s), mode_sha256=mode.source_sha256,
        mass_kg=MASS, charge_C=CHARGE, nominal_energy_J=ENERGY,
        full_reference_particles=5000, full_reference_duration_s=.01,
        birth_state="archived s/theta/zeta/vparallel/mu_per_mass preserved; equal marker weights",
        field=dict(coordinates="same Boozer convention; psi0=-phi_edge/(2*pi); K=0", angular_points=angular_points,
            mode_tolerance=mode_tolerance, retained_modes=len(field.xm),
            interpolation="ESSOS radial sqrt(s) cubic and angular quintic; FIRM3D used cubic 48x48x48",
            max_birth_B_relative_difference=float(np.max(abs(B/expected_B-1))),
            rms_birth_B_relative_difference=float(np.sqrt(np.mean((B/expected_B-1)**2))),
            max_birth_energy_relative_difference=float(np.max(abs(birth_energy/ENERGY-1)))),
        reference=selected, reference_scan=references, cases={})
    arrays = dict(s=mode.s, harmonics=mode.harmonics, potential_volts=mode.potential_volts,
        initial_conditions=data, birth_B_essos=B, birth_B_reference=expected_B,
        birth_speed=speed, birth_pitch=pitch)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"diagnostics.npz", **arrays)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("omega_rad_s","cyclic_frequency_hz","harmonics","field","reference")}, indent=2), flush=True)
    return field, mode, data, arrays, summary


def native_trace(field, data, wave, *, tmax, timestep, tolerance, species=None,
                 n_save=101, seed=42, execution="shared", collision_dtmax=1e-6, stable_keys=False,
                 collision_tolerance=1e-3, moment_tolerance=.005, timing_repeats=0):
    """Preserve archived v_parallel and mu/m, using orbit-only tracing in vacuum."""
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    from essos.background_species import BackgroundSpecies
    from essos.dynamics import Particles
    from essos.split import SplitStatus, trace_split

    points = jnp.asarray(data[:, :3])
    b0 = jax.vmap(lambda p: field.modB(*p))(points)
    speed, pitch = birth_parameters(jnp.asarray(data), b0)
    if species is None and wave is None:
        import time
        from essos.rk import _boozer_batch, _prepare_boozer_gpu
        times=np.linspace(0.,tmax,n_save)
        y0=jnp.column_stack((jnp.sqrt(points[:,0])*jnp.cos(points[:,1]),
            jnp.sqrt(points[:,0])*jnp.sin(points[:,1]),points[:,2],jnp.asarray(data[:,3])))
        mu=jnp.asarray(data[:,4])
        # A common minimum birth speed makes the existing orbit error budget
        # at least as strict as each marker's original normalized budget.
        scale=float(jnp.min(speed))
        solver=(_prepare_boozer_gpu(field,MASS,CHARGE,scale,times,timestep,tolerance,tolerance,100000)
                if execution=='independent' else None)
        function=jax.jit(lambda y,m:_boozer_batch(field,y,m,MASS,CHARGE,scale,jnp.asarray(times),
            timestep,tolerance,tolerance,'dopri8',True,'i',.4,.3,100000,solver))
        started=time.perf_counter();compiled=function.lower(y0,mu).compile()
        compile_seconds=time.perf_counter()-started
        started=time.perf_counter();result=jax.block_until_ready(compiled(y0,mu))
        first_seconds=time.perf_counter()-started
        warm=[]
        for _ in range(timing_repeats):
            started=time.perf_counter();result=jax.block_until_ready(compiled(y0,mu))
            warm.append(time.perf_counter()-started)
        states,loss,failed,_,counts,status=map(np.asarray,result)
        return SimpleNamespace(times=times,states=states,lost=loss>=0,failed=failed>=0,
            axis_stopped=np.zeros(len(data),bool),loss_times=loss,failed_times=failed,
            axis_stop_times=np.full(len(data),-1.),thermalized_times=np.full(len(data),-1.),
            step_stats=SimpleNamespace(accepted=counts[:,1],rejected=counts[:,2],
                orbit_steps=counts[:,1],orbit_rejected=counts[:,2]),
            timings=dict(driver='dedicated Boozer orbit kernel',compile_seconds=compile_seconds,
                         first_execution_seconds=first_seconds,warm_seconds=warm))
    particles = Particles(initial_xyz=points, initial_vparallel_over_v=pitch,
        mass=MASS, charge=CHARGE, energy=.5*MASS*speed**2)
    key = jax.random.key(seed)
    particles.random_keys = (jax.vmap(lambda i:jax.random.fold_in(key,i))(jnp.arange(len(data),dtype=jnp.uint32))
                             if stable_keys else jax.random.split(key,len(data)))
    collisional = species is not None
    if species is None:
        collision_dtmax = tmax
        species = BackgroundSpecies(1,jnp.array([2.]),jnp.array([1.]),jnp.array([0.]),jnp.array([17800.]))
    solution = trace_split(field,particles,species,tmax=tmax,timestep=timestep,n_save=n_save,
        integrator="essos_dopri8",perturbation=wave,rtol=tolerance,atol=tolerance,
        orbit_dt0=timestep,collision_dtmax=collision_dtmax,execution=execution,
        spatial_diffusion=False,adaptive=True,orbit_adaptive=True,
        collision_controller="moments" if collisional else "conservative",
        collision_tolerance=collision_tolerance,moment_tolerance=moment_tolerance)
    y, status, stop = np.asarray(solution.ys), np.asarray(solution.status), np.asarray(solution.t)
    B = np.asarray(jax.jit(jax.vmap(jax.vmap(lambda p:field.modB(*p))))(solution.ys[:,:,:3]))
    current_speed = np.sqrt(np.maximum(y[:,:,3]**2+2*y[:,:,4]*B/MASS,0.))
    event, failed = status == SplitStatus.EVENT, status > SplitStatus.EVENT
    lost = event & (y[:,-1,0] >= 1.-1e-8)
    axis = event & ~lost
    return SimpleNamespace(times=np.asarray(solution.ts),states=np.concatenate((y[:,:,:4],current_speed[:,:,None]),axis=2),
        lost=lost,failed=failed,axis_stopped=axis,loss_times=np.where(lost,stop,-1.),
        failed_times=np.where(failed,stop,-1.),axis_stop_times=np.where(axis,stop,-1.),
        thermalized_times=np.full(len(data),-1.),step_stats=solution.stats)


def run_cases(field, mode, data, arrays, summary, args):
    import jax

    n = args.particles
    indices = np.arange(n)  # A deterministic prefix; never regenerate paper births.
    choices = ("unperturbed", "perturbed") if args.case == "both" else (args.case,)
    summary.update(particles=n, duration_s=args.tmax, tolerance=args.tolerance, method=args.method,
        interpolation=args.interpolation, reference_interpolation="linear",
        platform=jax.default_backend(), devices=[str(device) for device in jax.devices()[:1]],
        device_kind=jax.devices()[0].device_kind, jax_version=jax.__version__,
        full_reference_cohort=(n == 5000 and args.tmax == .01),
        comparison_status="Computed ESSOS counts; agreement requires resolution/tolerance convergence")
    for name in choices:
        wave = mode.as_perturbation(interpolation=args.interpolation) if name == "perturbed" else None
        print(f"Tracing {name}: {n} markers, {args.tmax*1e3:g} ms, {args.method}, "
              f"{args.interpolation if wave is not None else 'no wave'}, "
              f"{summary['device_kind']} ({summary['platform']})", flush=True)
        trace = native_trace(field,data[indices],wave,tmax=args.tmax,
            timestep=min(1e-7,args.tmax/100),tolerance=args.tolerance,execution=args.execution)
        if np.any(trace.failed):
            raise RuntimeError(f"{name}: {np.count_nonzero(trace.failed)} unresolved markers; no valid loss estimate")
        if not np.isfinite(trace.states).all():
            raise RuntimeError(f"{name}: nonfinite saved trajectory; no valid loss estimate")
        hits = np.asarray(trace.lost)
        reference = summary["reference"][TAG if wave is not None else "saw_zero"]
        summary["cases"][name] = dict(lost=int(hits.sum()), particles=n, loss_fraction=float(hits.mean()),
            failed=0, reference_lost=reference["lost"], reference_particles=5000,
            reference_loss_fraction=reference["loss_fraction"],
            comparable_full_cohort=(summary["full_reference_cohort"] and
                                   (wave is None or args.interpolation == "linear")),
            matches_reference_interpolation=(wave is None or args.interpolation == "linear"))
        arrays[f"{name}_loss_times"] = np.asarray(trace.loss_times)
        arrays["times"] = np.asarray(trace.times)
        np.savez_compressed(args.output/"diagnostics.npz", **arrays)
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(f"Completed {name}: {hits.sum()}/{n} LCFS exits", flush=True)


def plot_saved(output, data_prefix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((output/"summary.json" if data_prefix is None else data_prefix.with_suffix(".json")).read_text())
    with np.load(output/"diagnostics.npz" if data_prefix is None else data_prefix.with_suffix(".npz"), allow_pickle=False) as saved:
        data = dict(saved)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    strongest = np.argsort(np.max(abs(data["potential_volts"]), axis=0))[-5:][::-1]
    for i in strongest:
        m, n = data["harmonics"][i]
        axes[0].plot(data["s"], data["potential_volts"][:, i], label=f"m={m}, n={n}")
    axes[0].set(xlabel="s", ylabel="Signed radial potential [V]", title="Archived AE3D mode; five strongest harmonics")
    axes[1].scatter(data["initial_conditions"][:, 0], (data["birth_B_essos"]/data["birth_B_reference"]-1)*100, s=2, alpha=.4)
    axes[1].set(xlabel="Birth s", ylabel="ESSOS / reference birth B − 1 [%]", title="Independent field check from saved v∥ and μ")
    references = (summary["reference"]["saw_zero"], summary["reference"][TAG])
    axes[2].bar([0, 1], [r["loss_fraction"]*100 for r in references], width=.5, color=["C0", "C1"],
                alpha=.5, label="Archived FIRM3D: 5000 markers, 10 ms")
    for index, name in enumerate(("unperturbed", "perturbed")):
        if name in summary["cases"]:
            measured = summary["cases"][name]
            axes[2].plot(index, measured["loss_fraction"]*100, "kx", ms=10,
                label=f"ESSOS: {measured['particles']} markers, {summary['duration_s']*1e3:g} ms" if index == 0 or len(summary["cases"]) == 1 else None)
    axes[2].set(xticks=[0, 1], xticklabels=["Unperturbed", "Perturbed"], ylabel="LCFS loss fraction [%]",
                title="Published loss endpoints" if not summary["cases"] else "Loss comparison: check cohort and duration")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.suptitle(f"QA AE3D–FIRM3D reference: 35 harmonics; ω={summary['omega_rad_s']:.3f} rad/s\n"
                 "Preparation/reference data only — ESSOS losses pending" if not summary["cases"] else
                 f"QA AE3D–FIRM3D: {summary['particles']} archived markers, {summary['duration_s']*1e3:g} ms")
    if summary.get("interpolation", "linear") != "linear" and summary["cases"]:
        fig.suptitle(f"QA interpolation sensitivity: {summary['particles']} markers, {summary['duration_s']*1e3:g} ms\n"
                     "ESSOS natural cubic in s; archived FIRM3D reference uses linear profiles")
    if summary.get("run_status") == "wall_clock_limit_reached":
        missing = ", ".join(summary["incomplete_cases"])
        fig.suptitle(f"QA AE3D–FIRM3D: {summary['particles']} archived markers, {summary['duration_s']*1e3:g} ms\n"
                     f"CPU wall-clock cap reached; incomplete: {missing}")
    for suffix in ("png", "svg"):
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output/f"ae3d_firm3d_loss.{suffix}", dpi=150)
    plt.close(fig)

    if summary["cases"]:
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        for name, color in (("unperturbed", "C0"), ("perturbed", "C1")):
            if name not in summary["cases"]:
                continue
            case = summary["cases"][name]
            exits = np.asarray(data[f"{name}_loss_times"])
            exits = np.sort(exits[exits >= 0])
            ax.step(np.r_[0., exits, summary["duration_s"]]*1e3,
                    np.r_[0., np.arange(1, len(exits)+1), len(exits)]/case["particles"]*100,
                    where="post", color=color,
                    label=f"ESSOS {name}: {case['lost']}/{case['particles']} ({100*case['loss_fraction']:.2f}%)")
            reference = summary["reference"][TAG if name == "perturbed" else "saw_zero"]
            ax.plot(reference["duration_s"]*1e3, reference["loss_fraction"]*100, "D", color=color,
                    label=f"FIRM3D {name}: {100*reference['loss_fraction']:.2f}% (5000 markers)")
        ax.set(xlabel="Time [ms]", ylabel="LCFS loss fraction [%]",
               title=f"QA: {summary['particles']} markers, {summary['duration_s']*1e3:g} ms; "
                     f"{summary.get('interpolation', 'linear')} profiles")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        for suffix in ("png", "svg"):
            fig.savefig(output/f"ae3d_firm3d_loss_history.{suffix}", dpi=150)
        plt.close(fig)


def plot_reported_results(source, output):
    """Replot the reported GPU endpoints, without inventing loss-time data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snapshot = json.loads(source.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), constrained_layout=True)
    for name, ax in zip(("unperturbed", "perturbed"), axes):
        reference = snapshot["reference"][name]
        rows = [("FIRM3D\nlinear", reference)]
        rows += [(f"ESSOS spline\ntol={run['tolerance']:g}", run["cases"][name])
                 for run in snapshot["runs"] if name in run["cases"]]
        bars = ax.bar(np.arange(len(rows)), [100*case["lost"]/case["particles"] for _, case in rows],
                      color=["0.65"]+[f"C{i}" for i in range(len(rows)-1)])
        for bar, (_, case) in zip(bars, rows):
            ax.annotate(f"{case['lost']}/{case['particles']}",
                        (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set(xticks=np.arange(len(rows)), xticklabels=[label for label, _ in rows],
               ylabel="LCFS loss fraction [%]", title=name.capitalize())
        ax.set_ylim(0, max(bar.get_height() for bar in bars)*1.2)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("QA: 5000 archived markers, 10 ms; reported GPU endpoints\nNative FAX modes are not used in these runs")
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output/f"ae3d_firm3d_gpu_results.{suffix}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--output", type=Path, default=ROOT.parent/"artifacts/fax_integration/ae3d_firm3d_qa")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--report-only", action="store_true")
    action.add_argument("--reported-results", type=Path, help="Plot a saved GPU endpoint snapshot; no tracing")
    parser.add_argument("--data", type=Path, help="Saved .json/.npz prefix for --report-only")
    parser.add_argument("--particles", type=int, default=5000)
    parser.add_argument("--case", choices=("both", "unperturbed", "perturbed"), default="both")
    parser.add_argument("--tmax", type=float, default=.01)
    parser.add_argument("--method", choices=("essos_dopri8",), default="essos_dopri8")
    parser.add_argument("--execution", choices=("shared", "independent"), default="shared")
    parser.add_argument("--representation", choices=("fourier","table"), default="fourier")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--interpolation", choices=("linear", "natural-cubic"), default="natural-cubic",
                        help="Natural cubic in s (default); linear reproduces FIRM3D's archived profiles")
    parser.add_argument("--angular-points", type=int, default=48)
    parser.add_argument("--mode-tolerance", type=float, default=1e-7)
    parser.add_argument("--platform", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--wall-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.execution == "independent" and args.representation != "fourier":
        parser.error("Native independent execution requires --representation fourier")
    if args.reported_results:
        if args.data is not None:
            parser.error("--data requires --report-only")
        plot_reported_results(args.reported_results, args.output)
        return
    if args.report_only:
        plot_saved(args.output, args.data)
        return
    if args.data is not None:
        parser.error("--data requires --report-only")
    if not (1 <= args.particles <= 5000 and 0 < args.tmax <= .01 and args.wall_seconds > 0
            and np.isfinite([args.tmax,args.wall_seconds,args.tolerance]).all() and args.tolerance > 0
            and args.angular_points >= 16 and np.isfinite(args.mode_tolerance) and args.mode_tolerance >= 0):
        parser.error("Invalid particle count, duration, resolution, tolerance or wall-clock cap")
    if (args.output/"summary.json").exists():
        parser.error("Choose a fresh --output to preserve results")
    os.environ.update(JAX_PLATFORMS=args.platform, ESSOS_CPU_DEVICES="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.perf_counter()

    def stop():
        print(f"Stopped at {args.wall_seconds:g}-second cap; completed cases remain saved", flush=True)
        os._exit(124)

    deadline = threading.Timer(args.wall_seconds, stop)
    deadline.daemon = True
    deadline.start()
    try:
        field, mode, data, arrays, summary = prepare(args.packet, args.output,
            None if args.representation == "fourier" else args.angular_points,args.mode_tolerance)
        if args.run:
            run_cases(field, mode, data, arrays, summary, args)
        plot_saved(args.output)
        summary["elapsed_seconds"] = time.perf_counter()-started
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(f"Saved reference audit and plot in {args.output}; no tracing" if not args.run else "Saved completed loss cases", flush=True)
    finally:
        deadline.cancel()


if __name__ == "__main__":
    main()
