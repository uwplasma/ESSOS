"""Full-geometry VMEC comparison to Paul (2023), Table 2, QA vac / Equil.

Uses the identical Boozer-angle potential, mapped into VMEC, and the existing
adaptive GC integrator. Defaults to 32 markers/case, 1 ms, CPU, 120 s wall cap.
Run larger samples or GPU jobs explicitly; partial cases are saved immediately.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import tomllib

from examples.benchmarks.paul_conventions import (CONVENTIONS, convert_births, convert_mode,
                                                 load_essos_births, particle_parameters)


def load_fields(wout, *, mode_tolerance=1e-7, boozer_mode_tolerance=1e-6, representation="raw"):
    """Load the same scaled, full equilibrium in both coordinate formulations."""
    import jax.numpy as jnp
    import numpy as np
    from netCDF4 import Dataset
    from booz_xform_jax import Booz_xform
    from essos.boozer import BoozerField, psi0_from_vmec
    from essos.fields import Vmec, VMEC_WOUT_ARRAYS
    from essos.gc_perturbations import VmecBoozerMap

    booz = Booz_xform(verbose=0, mboz=16, nboz=16)
    booz.read_wout(str(wout), flux=False)
    booz.run()
    mapping = VmecBoozerMap.from_booz_xform(booz)
    m, n = np.asarray(booz.xm_b), np.asarray(booz.xn_b)
    bscale = 5.86/np.asarray(booz.bmnc_b)[(m == 0) & (n == 0), 0].item()
    with Dataset(wout) as nc:
        lscale = 1.70/float(nc.variables["Aminor_p"][...])
        phi = float(nc.variables["phi"][-1])
        tables = {key: np.asarray(nc.variables[key][:]) for key in VMEC_WOUT_ARRAYS}
        nfp, ns = int(nc.variables["nfp"][...]), int(nc.variables["ns"][...])
    for key in ("rmnc", "zmns", "Aminor_p"):
        tables[key] *= lscale
    tables["bmnc"] *= bscale
    for key in ("bsubsmns", "bsubumnc", "bsubvmnc"):
        tables[key] *= bscale*lscale
    for key in ("bsupumnc", "bsupvmnc"):
        tables[key] *= bscale/lscale
    tables["gmnc"] *= lscale**3
    field = Vmec.from_arrays(nfp, ns, **{key: jnp.asarray(value) for key, value in tables.items()},
                             ntheta=8, nphi=8, mode_tolerance=mode_tolerance)
    background = BoozerField.from_booz(booz.s_b, np.asarray(booz.bmnc_b)*bscale, m, n,
                 booz.iota, np.asarray(booz.Boozer_G)*bscale*lscale,
                 np.asarray(booz.Boozer_I)*bscale*lscale,
                 psi0_from_vmec(phi)*bscale*lscale**2, nfp,
                 mode_tolerance=boozer_mode_tolerance, angle_grid=None)
    if representation != "raw":
        raise ValueError(f"Unknown equilibrium representation: {representation}")
    return field, background, mapping, dict(B_scale=bscale, length_scale=lscale)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", type=Path, default=root/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
    parser.add_argument("--output", type=Path, default=root/"examples/benchmarks/results/vmec_alfven_paper_benchmark")
    parser.add_argument("--particles", type=int, default=32, help="Markers per case; two cases")
    parser.add_argument("--tmax", type=float, default=1e-3)
    parser.add_argument("--saves", type=int, default=201)
    parser.add_argument("--timestep", type=float, default=5e-9)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--mode-tolerance", type=float, default=1e-7)
    parser.add_argument("--boozer-mode-tolerance", type=float, default=1e-6)
    parser.add_argument("--representation", choices=("raw",), default="raw",
                        help="Full native VMEC field")
    parser.add_argument("--source-convention", choices=CONVENTIONS, default="essos")
    parser.add_argument("--particle-model", choices=CONVENTIONS, default="essos")
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--diagnostic-batch-size", type=int, default=128,
                        help="Bound device memory while evaluating saved kinetic energies")
    parser.add_argument("--wall-seconds", type=float, default=120.)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--births", type=Path, help="Optional earlier Boozer benchmark trajectories.npz")
    parser.add_argument("--resume", action="store_true", help="Reuse completed cases after checking configuration and births")
    parser.add_argument("--report-only", action="store_true", help="Replot completed saved cases without tracing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.report_only:
        report_saved(args.output)
        return
    if args.particles < 2 or args.tmax <= 0 or args.wall_seconds <= 0 or args.diagnostic_batch_size < 1:
        parser.error("Need at least two markers, positive tmax, wall-seconds and diagnostic-batch-size")
    os.environ.update(JAX_PLATFORMS="cuda" if args.platform == "gpu" else "cpu", ESSOS_CPU_DEVICES="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if args.platform == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    def timeout():
        print(f"VMEC Paul benchmark reached its {args.wall_seconds:g}-second wall limit; saved cases remain available", flush=True)
        os._exit(124)

    deadline = threading.Timer(args.wall_seconds, timeout)
    deadline.daemon = True
    deadline.start()
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from essos.dynamics import Particles, Tracing
        from essos.constants import ALPHA_PARTICLE_MASS, ONE_EV
        from essos.gc_perturbations import MappedBoozerPerturbation
        from examples.particle_tracing.analytical_alfven_transport import make_mode
        from examples.benchmarks.sampling import estimate_interval
        reference = tomllib.loads((root/"examples/benchmarks/references/paul2023_vacuum_qa_equilibrium.toml").read_text())
        particle = particle_parameters(args.particle_model, measured_mass=ALPHA_PARTICLE_MASS,
            energy_joule=reference["energy_MeV"]*1e6*ONE_EV)
        # No QS filtering: match the paper's Equil. column, not Perfect QS.
        field, background, mapping, scales = load_fields(args.wout, mode_tolerance=args.mode_tolerance,
                            boozer_mode_tolerance=args.boozer_mode_tolerance, representation=args.representation)
        bscale, lscale = scales["B_scale"], scales["length_scale"]
        rng = np.random.default_rng(args.seed)
        if args.births is None:
            s = rng.uniform(.25, .75, args.particles)
            theta = rng.uniform(0, 2*np.pi, args.particles)
            zeta = rng.uniform(0, 2*np.pi, args.particles)
            pitch = rng.uniform(-1, 1, args.particles)
            s, theta, zeta, pitch = convert_births(s, theta, zeta, pitch, source=args.source_convention)
        else:
            s, theta, zeta, pitch = load_essos_births(args.births, args.particles, source=args.source_convention)
        boozer_births = jnp.asarray(np.stack([s, theta, zeta], axis=1))
        native_births = jax.jit(jax.vmap(mapping.inverse))(boozer_births)
        residual = float(np.max(abs(np.asarray(jax.vmap(mapping)(native_births))-boozer_births)))
        if residual > 1e-10:
            raise AssertionError(f"Birth coordinate map failed: {residual:g}")
        profiles, _ = jax.vmap(background.profiles)(jnp.asarray(s))
        iota, G, I = profiles.T
        B = jax.vmap(background.modB)(jnp.asarray(s), jnp.asarray(theta), jnp.asarray(zeta))
        weights = np.array(abs(background.psi0*(G+iota*I)/B**2))
        weights /= weights.sum()
        config = dict(platform=args.platform, particles_per_case=args.particles, tmax_s=args.tmax, **particle,
                      wall_seconds=args.wall_seconds, tolerance=args.tolerance, mode_tolerance=args.mode_tolerance,
                      boozer_mode_tolerance=args.boozer_mode_tolerance, representation=args.representation,
                      source_convention=args.source_convention, birth_coordinate_convention="essos",
                      saves=args.saves, initial_timestep=args.timestep, batch_size=args.batch_size,
                      max_steps=args.max_steps, seed=args.seed, wout=str(args.wout),
                      wout_sha256=hashlib.sha256(args.wout.read_bytes()).hexdigest(), B_scale=bscale,
                      length_scale=lscale, max_birth_map_residual=residual, reference=reference,
                      wave="Mapped uniform-radial Boozer potential; m=15,n=7; source omega=399000 rad/s",
                      loss_definition="LCFS exit plus singular-wave inner stop at s=1e-8, reported separately")
        saved_results = []
        if args.resume:
            previous = json.loads((args.output/"configuration.json").read_text())
            for key in config.keys()-{"wall_seconds", "max_birth_map_residual"}:
                if key == "wout_sha256" and key not in previous:
                    continue  # Legacy configuration; preserve its existing resume checks.
                if key in particle and key not in previous and args.particle_model == "essos":
                    continue  # Earlier runs used the same measured-mass default at 3.52 MeV.
                if previous.get(key) != config[key]:
                    raise ValueError(f"Cannot resume a different configuration: {key}")
            with np.load(args.output/"births.npz") as saved:
                for key, values in (("birth_s", s), ("birth_theta", theta), ("birth_zeta", zeta),
                                    ("birth_pitch", pitch), ("birth_weights", weights)):
                    np.testing.assert_allclose(saved[key], values, rtol=1e-13, atol=1e-15,
                                               err_msg=f"Cannot resume different {key}")
            saved_results = json.loads((args.output/"progress.json").read_text())["cases"]
        np.savez_compressed(args.output/"births.npz", birth_s=s, birth_theta=theta, birth_zeta=zeta,
                            birth_pitch=pitch, birth_weights=weights, native_births=native_births,
                            metadata_json=json.dumps(config))
        (args.output/"configuration.json").write_text(json.dumps(config, indent=2)+"\n")
        particles = Particles(initial_xyz=native_births, initial_vparallel_over_v=jnp.asarray(pitch),
                              mass=particle["mass_kg"], energy=particle["energy_joule"])
        mode_args = argparse.Namespace(m=15, regular_axis=False, axis_stop_s=1e-8)
        results, curves, energies = [], [], []
        for index, case in enumerate(reference["cases"]):
            if index < len(saved_results):
                with np.load(args.output/f"case_{index}.npz") as saved:
                    if len(saved["status"]) != args.particles or saved["times"][-1] != args.tmax:
                        raise ValueError("Saved case dimensions/duration differ from this run")
                    times = saved["times"]
                    curves.append((saved["loss_curve"], saved["unresolved_curve"]))
                    energies.append(saved["energy_change"])
                results.append(saved_results[index])
                print(f"Restored completed VMEC {case['name']} case", flush=True)
                continue
            source_mode = make_mode(background, mode_args, case["delta_bhat_psi"], case["n"], case["omega_rad_s"])
            mode = convert_mode(source_mode, source=args.source_convention)
            wave = None if index == 0 else MappedBoozerPerturbation(background, mode, mapping)
            print(f"Tracing VMEC {case['name']}: {args.particles} particles, {args.tmax:g} s", flush=True)
            trace = Tracing(field=field, particles=particles, model="GuidingCenterAdaptative", perturbation=wave,
                            maxtime=args.tmax, timestep=args.timestep, times_to_trace=args.saves,
                            rtol=args.tolerance, atol=args.tolerance, max_steps=args.max_steps,
                            progress=False, particle_batch_size=args.batch_size, devices=jax.devices()[:1])
            states, times = np.asarray(trace.trajectories), np.asarray(trace.times)
            stop_times = np.minimum(trace.lcfs_times, trace.axis_stop_times)
            lost = np.isfinite(stop_times)
            final_s = np.where(np.isfinite(trace.lcfs_times), trace.lcfs_states[:, 0], states[:, -1, 0])
            final_s = np.where(trace.axis_stopped, trace.axis_stop_states[:, 0], final_s)
            curve = weights@(stop_times[:, None] <= times[None, :])
            unresolved = weights@(trace.failure_times[:, None] <= times[None, :])
            energy = energy_changes(trace, args.diagnostic_batch_size)
            np.savez_compressed(args.output/f"case_{index}.npz", times=times, states=states,
                                status=trace.status, lcfs_times=trace.lcfs_times, lcfs_states=trace.lcfs_states,
                                axis_stop_times=trace.axis_stop_times, axis_stop_states=trace.axis_stop_states,
                                failure_times=trace.failure_times, energy_change=energy, loss_curve=curve,
                                unresolved_curve=unresolved, final_s=final_s, weights=weights)
            failed = int(trace.failed.sum())
            result = dict(name=case["name"], failed=failed, lcfs_exits=int(trace.boundary_hits.sum()),
                          source_omega_rad_s=float(source_mode.omega), essos_omega_rad_s=float(mode.omega),
                          source_potential_volts=float(source_mode.amplitude), potential_volts=float(mode.amplitude),
                          axis_stops=int(trace.axis_stopped.sum()), weighted_confirmed_loss=float(weights@lost),
                          weighted_unresolved=float(weights@trace.failed),
                          max_relative_kinetic_energy_change=float(np.max(abs(energy[np.isfinite(energy)]))))
            if not failed:
                result["loss_fraction"] = estimate_interval(lost, weights, binary=True)
                result["mean_abs_delta_s"] = estimate_interval(abs(final_s-s), weights)
            if wave is not None:
                mu = states[:, :, 4]
                result["max_relative_mu_change"] = float(np.nanmax(np.where(np.isfinite(mu), abs(mu/mu[:, :1]-1), np.nan)))
            results.append(result)
            curves.append((curve, unresolved))
            energies.append(energy)
            (args.output/"progress.json").write_text(json.dumps(dict(config, cases=results), indent=2)+"\n")
            print(json.dumps(result), flush=True)
        _plot(args.output, times, curves, energies, reference, results, args.particles, args.tmax)
        summary = dict(config, cases=results, elapsed_seconds=time.perf_counter()-started,
                       elapsed_time_scope="current invocation",
                       resumed_cases=[case["name"] for case in saved_results],
                       full_paper_duration=bool(np.isclose(args.tmax, .001, rtol=0, atol=1e-12)),
                       all_orbits_resolved=all(case["failed"] == 0 for case in results),
                       limitations=["Low-resolution input equilibrium; exact paper input unconfirmed; paper uses 5000 markers",
                                    "Sampling intervals exclude equilibrium, interpolation and model errors",
                                    "Uniform-volume importance weights from full Boozer Jacobian",
                                    f"Native field representation: {args.representation}",
                                    "Historical SIMSOPT convention is explicit; paper runtime is unconfirmed",
                                    "The full equilibrium has no exact QS wave-frame invariant"])
        summary["numerical_checks_passed"] = bool(summary["all_orbits_resolved"]
            and results[0]["max_relative_kinetic_energy_change"] < 1e-3
            and results[1]["max_relative_mu_change"] < 1e-12)
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        _report(args.output, summary)
        print(f"Saved comparison to {args.output}; elapsed {summary['elapsed_seconds']:.1f} s", flush=True)
    finally:
        deadline.cancel()


def energy_changes(trace, batch_size):
    """Use the existing energy diagnostic in batches, without changing the trace."""
    import copy
    import numpy as np
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    view = copy.copy(trace)
    energies = []
    for start in range(0, len(trace.trajectories), batch_size):
        view.trajectories = trace.trajectories[start:start+batch_size]
        energies.append(np.asarray(view.energy())/float(trace.particles.energy)-1)
    return np.concatenate(energies)


def report_saved(output):
    """Regenerate plots and the report using saved data; no JAX import or tracing."""
    import numpy as np
    summary = json.loads((output/"summary.json").read_text())
    curves, energies = [], []
    for index in range(len(summary["cases"])):
        with np.load(output/f"case_{index}.npz") as saved:
            times = saved["times"]
            curves.append((saved["loss_curve"], saved["unresolved_curve"]))
            energies.append(saved["energy_change"])
    summary["elapsed_time_scope"] = "current invocation"
    cases = summary["cases"]
    summary["numerical_checks_passed"] = bool(summary["all_orbits_resolved"]
        and cases[0]["max_relative_kinetic_energy_change"] < 1e-3
        and cases[1]["max_relative_mu_change"] < 1e-12)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    _plot(output, times, curves, energies, summary["reference"], cases,
          summary["particles_per_case"], summary["tmax_s"])
    _report(output, summary)


def _report(output, summary):
    lines = ["# Full VMEC / mapped Paul wave", "",
             f"Reference: [Paul et al., QA vac / Equil.]({summary['reference']['source']}).", "",
             f"{summary['particles_per_case']} markers per case; {summary['tmax_s']*1e3:g} ms; "
             f"{summary['elapsed_seconds']:.1f} seconds in the completing invocation.", ""]
    if summary.get("resumed_cases"):
        lines.extend(["Restored completed cases: "+", ".join(summary["resumed_cases"])+". "
                      "The elapsed time excludes the earlier invocation.", ""])
    lines.extend([
             "| Case | Confirmed loss | Unresolved weight | LCFS exits | Inner stops | Paper at 1 ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for case, reference in zip(summary["cases"], summary["reference"]["cases"]):
        lines.append(f"| {case['name']} | {case['weighted_confirmed_loss']:.4f} | "
                     f"{case['weighted_unresolved']:.4f} | {case['lcfs_exits']} | "
                     f"{case['axis_stops']} | {reference['loss_fraction']:.3f} |")
    for case in summary["cases"]:
        if "loss_fraction" in case:
            lo, hi = case["loss_fraction"]["interval_95"]
            lines.append("")
            lines.append(f"{case['name']}: approximate 95% sampling interval [{lo:.4f}, {hi:.4f}].")
    lines.extend(["", f"Full paper duration: {summary['full_paper_duration']}. "
                  f"Numerical checks passed: {summary['numerical_checks_passed']}.", "",
                  "The full VMEC equilibrium retains QS-breaking modes; the older Boozer comparison "
                  "uses the separate Perfect QS reference. This small run does not establish "
                  "a converged reproduction.", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    (output/"comparison.md").write_text("\n".join(lines)+"\n")


def _plot(output, times, curves, energies, reference, results, particles, tmax):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for i, (case, result, (curve, unresolved), energy) in enumerate(zip(reference["cases"], results, curves, energies)):
        axes[0].plot(times*1e3, 100*curve, color=f"C{i}", label=f"VMEC {case['name']}")
        if np.max(unresolved) > 0:
            axes[0].fill_between(times*1e3, 100*curve, 100*(curve+unresolved), color=f"C{i}", alpha=.2)
        axes[0].plot(1., 100*case["loss_fraction"], "x", color=f"C{i}", markersize=9,
                     label=f"Paul Equil. {case['name']}")
        if "loss_fraction" in result:
            lo, hi = np.array(result["loss_fraction"]["interval_95"])*100
            value = result["weighted_confirmed_loss"]*100
            axes[0].errorbar(tmax*1e3, value, yerr=[[value-lo], [hi-value]], fmt=".", color=f"C{i}")
        axes[1].plot(times*1e3, np.nanmax(np.where(np.isfinite(energy), abs(energy), np.nan), axis=0),
                     color=f"C{i}", label=case["name"])
    axes[0].set(xlabel="Time [ms]", ylabel="Weighted loss fraction [%]", title="LCFS exits + wave inner stops")
    axes[1].set(xlabel="Time [ms]", ylabel="Max active-marker |kinetic energy change| / birth energy",
                title="Equilibrium conservation; wave energy exchange")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Mapped Paul wave, full VMEC geometry: {particles} markers per case")
    for extension in ("png", "svg"):
        fig.savefig(output/f"vmec_paul_comparison.{extension}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
