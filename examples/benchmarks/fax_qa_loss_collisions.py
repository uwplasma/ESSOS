"""Six paired QA alpha-loss cases using the saved native FAX mode.

Preparation is the default. --run explicitly enables particle tracing;
--platform cuda selects a GPU. No FAX solve or collision implementation is
added here: all cases use the native adaptive orbit/collision solver.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from examples.benchmarks.ae3d_firm3d_archive import MANIFEST
from examples.benchmarks.ae3d_firm3d_loss import MASS, CHARGE, ENERGY, birth_parameters, equilibrium, native_trace, spatial_geometry
from examples.benchmarks.sampling import estimate_interval
from examples.fieldline_tracing.fax_qa_poincare import ROOT, ARTIFACTS, default_equilibrium


CASES = {
    "unperturbed": (False, False, False, "Unperturbed, collisionless"),
    "unperturbed_collisions": (False, True, False, "Unperturbed + collisions"),
    "perturbed": (True, False, False, "Native FAX, collisionless"),
    "perturbed_collisions": (True, True, False, "Native FAX + collisions"),
    "unperturbed_collisions_spatial": (False, True, True, "Unperturbed + collisions + spatial diffusion"),
    "perturbed_collisions_spatial": (True, True, True, "Native FAX + collisions + spatial diffusion"),
}


def write_summary(output, summary):
    temporary = output/"summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    temporary.replace(output/"summary.json")


def volume_births(field, count, seed, s_min=0., s_max=1.):
    """Nested uniform-flux proposals, isotropic pitch, and physical-volume weights."""
    import jax
    import jax.numpy as jnp

    sample = np.random.default_rng(seed).random((count,4))
    points = np.column_stack((s_min+(s_max-s_min)*sample[:,0],2*np.pi*sample[:,1],2*np.pi*sample[:,2]))
    def measures(point):
        B = field.modB(*point)
        (iota,G,I),_ = field.profiles(point[0])
        return B,jnp.abs(field.psi0*(G+iota*I))/B**2
    B,weights = (np.asarray(v) for v in jax.jit(jax.vmap(measures))(jnp.asarray(points)))
    speed, pitch = np.sqrt(2*ENERGY/MASS), 2*sample[:,3]-1
    data = np.column_stack((points,speed*pitch,speed**2*(1-pitch**2)/(2*B)))
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(weights)&(weights>0)):
        raise ValueError("Generated births require a positive finite field and volume measure")
    return data,weights


def prepare(args):
    import jax
    import jax.numpy as jnp
    from essos.fax_adapter import FaxMode

    snapshot = json.loads(args.wave_summary.read_text())
    recorded = snapshot.get("cases",{}).get("fax",snapshot)
    if "frequency_hz" not in recorded:
        recorded = snapshot
    if hashlib.sha256(args.boozmn.read_bytes()).hexdigest() != snapshot["equilibrium_sha256"]:
        raise ValueError("The equilibrium differs from the magnetic-section comparison")
    field = equilibrium(args.boozmn,angular_points=None if args.representation == "fourier" else args.angular_points,
                        mode_tolerance=args.mode_tolerance)
    if args.birth_source == "volume":
        data,weights = volume_births(field,args.particles,args.seed,args.birth_s_min,args.birth_s_max)
        birth_digest = hashlib.sha256(data.tobytes()).hexdigest()
    else:
        birth_file = args.births or args.boozmn.parent/"ICs/fusion_ICs_5000.txt"
        manifest = json.loads(MANIFEST.read_text())
        birth_record = next(row for row in manifest["members"] if row["name"].endswith("/ICs/fusion_ICs_5000.txt"))
        birth_digest = hashlib.sha256(birth_file.read_bytes()).hexdigest()
        if birth_digest != birth_record["sha256"]:
            raise ValueError("Births differ from the archived 5000-marker cohort")
        archived = np.loadtxt(birth_file)
        if archived.shape != (5000,5) or not np.isfinite(archived).all():
            raise ValueError("Invalid archived birth table")
        data,weights = archived[:args.particles].copy(),np.ones(args.particles)
    packet = FaxMode.load(args.mode, equilibrium=args.boozmn)
    if snapshot.get("mode_sha256") and hashlib.sha256(args.mode.read_bytes()).hexdigest() != snapshot["mode_sha256"]:
        raise ValueError("The mode differs from the saved magnetic section")
    if not np.isclose(packet.frequency_hz, recorded["frequency_hz"], rtol=1e-12, atol=0):
        raise ValueError("The mode differs from the saved native FAX magnetic section")
    phase = (1.+0j if snapshot.get("saved_packet_phase_applied") else
             complex(*snapshot.get("native_global_phase",[1.,0.]))*np.exp(1j*np.deg2rad(snapshot.get("wave_phase_deg",0.))))
    packet = replace(packet, structure=packet.structure*phase)
    wave = packet.as_physical_potential(potential_volts=recorded["potential_volts"], frequency_sign=1)
    packet.save(args.output/"mode.npz")
    B = np.asarray(jax.jit(jax.vmap(lambda p: field.modB(*p)))(jnp.asarray(data[:, :3])))
    speed, pitch = birth_parameters(data, B)
    if (not np.all(np.isfinite(B) & (B > 0)) or not np.all(data[:, 4] >= 0)
            or not np.all(np.isfinite(speed) & (speed > 0)) or not np.all(abs(pitch) <= 1)):
        raise ValueError("Invalid reconstructed particle births")
    births = dict(initial_conditions=data, speed=speed, pitch=pitch, birth_B=B,
                  marker_indices=np.arange(args.particles),weights=weights)
    np.savez_compressed(args.output/"births.npz", **births)
    summary = dict(status="prepared", cases={}, particles=args.particles, duration_s=args.tmax,
        method=args.method, requested_timestep_s=args.timestep,
        collision_dtmax_s=args.collision_dtmax,tolerance=args.tolerance,execution=args.execution,
        adaptive_orbit=True,adaptive_collisions=True,collision_controller="moments",
        collision_tolerance=args.collision_tolerance,moment_tolerance=args.moment_tolerance,
        n_save=args.n_save, collision_seed=args.seed,
        requested_cases=list(CASES) if args.case == "all" else [args.case],
        frequency_hz=packet.frequency_hz, omega_rad_s=float(wave.omega), harmonics=len(packet.harmonics),
        potential_volts=recorded["potential_volts"], native_global_phase=[phase.real,phase.imag],
        magnetic_energy_amplitude=snapshot.get("magnetic_energy_amplitude"),
        amplitude_note="Unchanged voltage from the frozen t=0 magnetic section; wave evolves in physical time",
        mode_source=packet.metadata, equilibrium_sha256=snapshot["equilibrium_sha256"],
        births_sha256=birth_digest,birth_source=args.birth_source,
        birth_selection=("uniform flux-coordinate proposals, isotropic pitch, 3.52 MeV; not the archived fusion source"
                         if args.birth_source == "volume" else "deterministic archive prefix; vparallel and mu/m retained"),
        birth_s_range=[args.birth_s_min,args.birth_s_max] if args.birth_source == "volume" else None,
        weights="physical Boozer volume Jacobian" if args.birth_source == "volume" else "equal markers",
        mass_kg=MASS, charge_C=CHARGE, field_representation=args.representation,
        equilibrium_mode_tolerance=args.mode_tolerance,equilibrium_harmonics=len(field.xm),
        field_angular_points=None if args.representation == "fourier" else args.angular_points,
        field_model="Existing K=0 Boozer equilibrium; natural-cubic physical potential; existing GC RHS",
        collision_operator="Native adaptive Dopri8 and split Coulomb kicks; spatial diffusion selected per case",
        background=dict(species=["electron", "deuterium"], density_each_m3=args.density,
                        temperature_each_eV=args.temperature,
                        source="examples/particle_tracing_collisions/trace_particles_vmec_collisionsMu.py defaults"),
        comparison_scope="Native FAX six-case experiment; no published collisional loss reference or fitted paper endpoints",
        platform=jax.default_backend(), devices=[str(d) for d in jax.devices()[:1]],
        device_kind=jax.devices()[0].device_kind, jax_version=jax.__version__)
    write_summary(args.output, summary)
    return field, wave, births, summary


def summarize_trace(trace, weights=None):
    lost, failed, axis = np.asarray(trace.lost), np.asarray(trace.failed), np.asarray(trace.axis_stopped)
    thermal = np.asarray(trace.thermalized_times) >= 0
    classified = lost.astype(int)+failed.astype(int)+axis.astype(int)+thermal.astype(int)
    if np.any(classified > 1):
        raise RuntimeError("Overlapping particle termination classifications")
    valid = not np.any(failed | axis) and bool(np.isfinite(trace.states).all())
    weights = np.ones(len(lost)) if weights is None else np.asarray(weights)
    estimate = estimate_interval(lost,weights,binary=True)
    result = dict(particles=len(lost), lost=int(lost.sum()), failed=int(failed.sum()),
        axis_stopped=int(axis.sum()), thermalized=int(thermal.sum()),
        confined=int(np.count_nonzero(classified == 0)), valid_loss_estimate=valid,
        observed_exit_fraction=float(lost.mean()), loss_fraction=estimate["value"] if valid else None,
        effective_particles=estimate["effective_particles"])
    result["sampling_interval_95"] = estimate["interval_95"] if valid else None
    if hasattr(trace,"step_stats"):
        result["solver_steps"] = {name:dict(total=int(np.sum(values)),max=int(np.max(values)))
            for name in ("accepted","rejected","orbit_steps","orbit_rejected")
            for values in (np.asarray(getattr(trace.step_stats,name)),)}
    if hasattr(trace,"timings"):result["timings"]=trace.timings
    return result


def run_cases(args, field, wave, births, summary):
    import jax
    import jax.numpy as jnp
    from essos.background_species import BackgroundSpecies
    from essos.constants import ELECTRON_MASS, PROTON_MASS

    species = BackgroundSpecies(2, jnp.array([ELECTRON_MASS/PROTON_MASS, 2.]),
        jnp.array([-1., 1.]), jnp.full(2, args.density), jnp.full(2, args.temperature))
    geometry=spatial_geometry(args.boozmn) if any(CASES[c][2] for c in summary["requested_cases"]) else None
    summary["spatial_geometry_harmonics"]=None if geometry is None else len(geometry.xm)
    for case in summary["requested_cases"]:
        with_wave, with_collisions, with_spatial, label = CASES[case]
        summary.update(status="running", active_case=case, completed_save_intervals=0)
        write_summary(args.output, summary)
        print(f"Tracing {label}: {args.particles} markers, {args.tmax*1e3:g} ms, "
              f"{args.method}, {len(field.xm)} equilibrium harmonics, initial dt={args.timestep:g} s, "
              f"{summary['device_kind']}", flush=True)
        case_started = time.perf_counter()

        trace = native_trace(field,births["initial_conditions"],wave if with_wave else None,
            tmax=args.tmax,timestep=args.timestep,n_save=args.n_save,tolerance=args.tolerance,
            species=species if with_collisions else None,seed=args.seed,execution=args.execution,
            collision_dtmax=args.collision_dtmax,stable_keys=args.birth_source == "volume",
            collision_tolerance=args.collision_tolerance,moment_tolerance=args.moment_tolerance,
            timing_repeats=args.timing_repeats,spatial_diffusion=with_spatial,
            spatial_geometry=geometry if with_spatial else None)
        np.savez_compressed(args.output/f"{case}.npz", times=trace.times, states=trace.states,
            loss_times=trace.loss_times, thermalized_times=trace.thermalized_times,
            failed_times=trace.failed_times, axis_stop_times=trace.axis_stop_times,weights=births["weights"],
            marker_indices=births["marker_indices"])
        summary["cases"][case] = dict(**summarize_trace(trace,births["weights"]),
            perturbed=with_wave,collisions=with_collisions,spatial_diffusion=with_spatial,
            wall_seconds=time.perf_counter()-case_started)
        write_summary(args.output, summary)
        print(f"Completed {case}: {json.dumps(summary['cases'][case])}", flush=True)
    summary.pop("active_case", None)
    summary.pop("completed_save_intervals", None)
    summary["status"] = ("completed" if all(r["valid_loss_estimate"] for r in summary["cases"].values())
                         else "completed_with_unresolved_markers")
    write_summary(args.output, summary)


def plot_saved(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from essos.constants import ELEMENTARY_CHARGE

    plt.rcParams.update({"font.size":13,"font.weight":"bold","axes.labelweight":"bold",
        "axes.titleweight":"bold","axes.linewidth":1.4,"lines.linewidth":2.2,"savefig.dpi":300})
    summary = json.loads((output/"summary.json").read_text())
    if not summary["cases"]:
        print("Prepared inputs only; no particle results to plot", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    curves = []
    for case, (wave, collisions, spatial, label) in CASES.items():
        if case not in summary["cases"]:
            continue
        result = summary["cases"][case]
        with np.load(output/f"{case}.npz", allow_pickle=False) as saved:
            data = dict(saved)
        times = data["times"]
        weights = data.get("weights",np.ones(len(data["loss_times"])))
        weights = weights/weights.sum()
        exits = np.where(data["loss_times"] >= 0,data["loss_times"],np.inf)
        curve = 100*(weights@(exits[:,None] <= times[None,:]))
        color, style = ("C1" if wave else "C0"), (":" if spatial else "-" if collisions else "--")
        suffix = "" if result["valid_loss_estimate"] else " (unresolved markers)"
        curve_label = f"{label}: {curve[-1]:.2f}%"+suffix
        axes[0].step(times*1e3,curve,where="post",color=color,ls=style,label=curve_label)
        curves.append((case,times,curve,color,style,curve_label))
        active = np.ones(data["states"].shape[:2], bool)
        for name in ("loss_times", "thermalized_times", "failed_times", "axis_stop_times"):
            stop = data[name][:, None]
            active &= (stop < 0) | (times[None] < stop)
        energy = .5*summary["mass_kg"]*data["states"][:, :, 4]**2/(1e6*ELEMENTARY_CHARGE)
        denominator = np.sum(active*weights[:,None],axis=0)
        energy = np.divide(np.sum(np.where(active,energy,0.)*weights[:,None],axis=0),denominator,
                           out=np.full(len(times),np.nan),where=denominator>0)
        axes[1].plot(times*1e3, energy, color=color, ls=style, label=label)
    ylabel = "Volume-weighted loss fraction [%]" if summary.get("birth_source") == "volume" else "Loss fraction [%]"
    axes[0].set(xlabel="Time [ms]",ylabel=ylabel,title=f"{len(curves)} paired alpha-loss cases")
    axes[1].set(xlabel="Time [ms]", ylabel="Mean kinetic energy [MeV]", title="Active markers only (changing population)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=9,frameon=False)
    fig.suptitle(f"Native FAX QA: {summary['particles']} identical births per case, "
                 f"{summary['frequency_hz']/1000:.3f} kHz; wave amplitude unchanged")
    for suffix in ("png","svg","pdf"):
        fig.savefig(output/f"fax_qa_loss_collisions.{suffix}",dpi=300)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(8.5,5.5),constrained_layout=True)
    np.savetxt(output/"loss_fraction.csv",np.column_stack((curves[0][1],*[c[2]/100 for c in curves])),
        delimiter=",",header="time_s,"+",".join(c[0] for c in curves),comments="")
    for case,times,curve,color,style,label in curves:
        ax.step(times*1e3,curve,where="post",color=color,ls=style,label=label)
    ax.set(xlabel="Time [ms]",ylabel=ylabel,xlim=(0,summary["duration_s"]*1e3),ylim=(0,None),
        title=f"QA alpha losses: {summary['particles']} particles per case\n"
              f"Native FAX mode, {summary['frequency_hz']/1000:.3f} kHz")
    ax.tick_params(direction="in",width=1.3,length=5)
    ax.grid(alpha=.2)
    ax.legend(fontsize=10,frameon=False)
    for suffix in ("png","svg","pdf"):
        fig.savefig(output/f"fax_qa_loss_fraction.{suffix}",dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--report-only", action="store_true")
    parser.add_argument("--mode", type=Path, default=ARTIFACTS/"qa_saved_mode_35/mode.npz")
    parser.add_argument("--wave-summary", type=Path, default=ARTIFACTS/"qa_saved_mode_35/wave_summary.json")
    parser.add_argument("--boozmn", type=Path, default=default_equilibrium())
    parser.add_argument("--births", type=Path)
    parser.add_argument("--output", type=Path, default=ARTIFACTS/"qa_8192_six_cases_gpu")
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--birth-source", choices=("archive","volume"), default="volume")
    parser.add_argument("--birth-s-min", type=float, default=0.)
    parser.add_argument("--birth-s-max", type=float, default=1.)
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--tmax", type=float, default=.01)
    parser.add_argument("--timestep", type=float, default=1e-8)
    parser.add_argument("--n-save", type=int, default=101)
    parser.add_argument("--method", choices=("essos_dopri8",), default="essos_dopri8")
    parser.add_argument("--execution", choices=("shared", "independent"), help="Default: independent on CUDA, shared on CPU")
    parser.add_argument("--representation", choices=("fourier","table"), default="fourier")
    parser.add_argument("--mode-tolerance", type=float, default=1e-4)
    parser.add_argument("--timing-repeats", type=int, default=0, help="Warm repeats for the unperturbed collisionless orbit kernel")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--collision-dtmax", type=float, default=1e-6)
    parser.add_argument("--collision-tolerance", type=float, default=1e-3)
    parser.add_argument("--moment-tolerance", type=float, default=.01)
    parser.add_argument("--density", type=float, default=4.21e20)
    parser.add_argument("--temperature", type=float, default=17.8e3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--angular-points", type=int, default=48)
    parser.add_argument("--platform", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--wall-seconds", type=float, default=21600.)
    args = parser.parse_args()
    if args.execution is None:
        args.execution="independent" if args.platform=="cuda" else "shared"
    if args.execution == "independent" and args.representation != "fourier":
        parser.error("Native independent execution requires --representation fourier")
    if args.report_only:
        plot_saved(args.output)
        return
    for path in (args.mode,args.wave_summary,args.boozmn):
        if not path.is_file():
            parser.error(f"Missing input: {path}; transfer the saved mode packet or pass explicit input paths")
    if (args.particles < 1 or (args.birth_source == "archive" and args.particles > 5000)
            or (args.birth_source == "volume" and args.births is not None)
            or not 0 <= args.birth_s_min < args.birth_s_max <= 1
            or args.n_save < 2 or args.angular_points < 16
            or not 0 <= args.seed < 2**32 or not 0 < args.tmax <= .01
            or not all(np.isfinite(v) and v > 0 for v in
                       (args.timestep,args.tolerance,args.collision_dtmax,args.collision_tolerance,args.moment_tolerance,
                        args.density,args.temperature,args.wall_seconds))
            or not np.isfinite(args.mode_tolerance) or not 0 < args.mode_tolerance < 1
            or args.timing_repeats < 0):
        parser.error("Invalid particle count, duration, step, background, resolution, seed or wall cap")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh --output to preserve results")
    os.environ.update(JAX_PLATFORMS=args.platform, ESSOS_CPU_DEVICES="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    args.output.mkdir(parents=True, exist_ok=True)
    started, holder = time.perf_counter(), {}

    def stop():
        if "summary" in holder:
            holder["summary"].update(status="wall_clock_limit_reached", elapsed_seconds=time.perf_counter()-started)
            write_summary(args.output, holder["summary"])
        print("Wall cap reached; completed cases remain saved. Use --report-only to plot them.", flush=True)
        os._exit(124)

    timer = threading.Timer(args.wall_seconds, stop)
    timer.daemon = True
    timer.start()
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        field, wave, births, summary = prepare(args)
        holder["summary"] = summary
        if args.run:
            run_cases(args, field, wave, births, summary)
            plot_saved(args.output)
        summary["elapsed_seconds"] = time.perf_counter()-started
        write_summary(args.output, summary)
        print(json.dumps(dict(status=summary["status"], particles=summary["particles"],
            duration_s=summary["duration_s"], method=summary["method"],
            frequency_hz=summary["frequency_hz"], cases=summary["cases"]), indent=2), flush=True)
    except Exception as error:
        if "summary" in holder:
            holder["summary"].update(status="failed", error=str(error), elapsed_seconds=time.perf_counter()-started)
            write_summary(args.output, holder["summary"])
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
