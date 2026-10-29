"""Small FAX -> reduced ideal wave -> existing VMEC GC example.

python -m examples.particle_tracing.fax_mode_particles --compute \
    --interpretation potential_shape --frequency-sign 1

Two particles per case by default; 120-second CPU process cap. This is an
integration demonstration, not a published loss-fraction benchmark. FAX's
scalar envelope is explicitly assigned to Phi; this does not establish its
full ideal-MHD polarization. A mode amplitude is imposed, not predicted.
"""

import argparse
import json
import os
from pathlib import Path
import threading
import time


def plot_saved(output, data_prefix=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_file = output/"summary.json" if data_prefix is None else data_prefix.with_suffix(".json")
    arrays_file = output/"diagnostics.npz" if data_prefix is None else data_prefix.with_suffix(".npz")
    summary = json.loads(summary_file.read_text())
    with np.load(arrays_file) as saved:
        data = dict(saved)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    shape = data["structure"]/np.max(abs(data["structure"]))
    for index, (m, n) in enumerate(data["harmonics"]):
        axes[0, 0].plot(data["s"], abs(shape[:, index]), label=f"m={m}, n={n}")
        axes[0, 1].plot(data["s"], shape[:, index].real, color=f"C{index}", label=f"m={m}, real")
        axes[0, 1].plot(data["s"], shape[:, index].imag, "--", color=f"C{index}", label=f"m={m}, imag")
    for index, label in enumerate(("x", "y", "z")):
        axes[1, 0].plot(data["theta"], data["delta_B_cart"][:, index], label=label)
        axes[1, 1].plot(data["theta"], data["delta_E_cart"][:, index], label=label)
    for case, style in (("unperturbed", "--"), ("perturbed", "-")):
        states, energies = data[f"{case}_states"], data[f"{case}_energy"]
        for index in range(len(states)):
            axes[2, 0].plot(data["times"]*1e6, states[index, :, 0], style, color=f"C{index}",
                            label=case if index == 0 else None)
            axes[2, 1].plot(data["times"]*1e6, energies[index]/energies[index, 0]-1,
                            style, color=f"C{index}")
    axes[0, 0].set(xlabel="s", ylabel="Normalized envelope magnitude", title="Calculated FAX harmonic structure")
    axes[0, 1].set(xlabel="s", ylabel="Normalized complex coefficient", title="Relative phases retained")
    axes[1, 0].set(xlabel="VMEC theta [rad]", ylabel="delta B [T]", title="Cartesian magnetic perturbation")
    axes[1, 1].set(xlabel="VMEC theta [rad]", ylabel="delta E [V/m]", title="Cartesian electric perturbation")
    axes[2, 0].set(xlabel="Time [microseconds]", ylabel="s", title="Existing adaptive VMEC GC")
    axes[2, 1].set(xlabel="Time [microseconds]", ylabel="Relative kinetic-energy change", title="Physical wave work")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in axes.flat[:-1]:
        ax.legend(fontsize=7)
    fig.suptitle(f"FAX reduced envelope assigned to an ideal-wave potential: {summary['frequency_hz']/1e3:.2f} kHz\n"
                 f"{summary['particles_per_case']} particles/case; imposed radial RMS delta B/B={summary['radial_rms']:.2g}")
    for extension in ("png", "svg"):
        fig.savefig(output/f"fax_mode_particles.{extension}", dpi=150)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--compute", action="store_true")
    source.add_argument("--mode", type=Path, help="Previously saved ESSOS/FAX mode.npz; FAX not required")
    source.add_argument("--report-only", action="store_true", help="Redraw diagnostics without tracing or FAX")
    parser.add_argument("--interpretation", choices=("potential_shape",))
    parser.add_argument("--frequency-sign", type=int, choices=(-1, 1))
    parser.add_argument("--wout", type=Path, default=root/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
    parser.add_argument("--output", type=Path, default=root.parent/"artifacts/fax_integration/fax_mode_particles")
    parser.add_argument("--data", type=Path, help="Saved data prefix for --report-only (without .json/.npz)")
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--tmax", type=float, default=2e-7)
    parser.add_argument("--wall-seconds", type=float, default=120.)
    parser.add_argument("--radial-rms", type=float, default=1e-4)
    parser.add_argument("--normalization-s", type=float, default=.4)
    parser.add_argument("--angular-points", type=int, default=16)
    parser.add_argument("--mass-density", type=float, default=3.4e-8, help="Constant FAX background mass density [kg/m^3]")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.report_only:
        plot_saved(args.output, args.data)
        print(f"Redrew saved FAX example in {args.output}; no FAX solve or particle tracing")
        return
    if args.frequency_sign is None or (args.compute and args.interpretation is None):
        parser.error("Select --frequency-sign; --compute also requires --interpretation potential_shape")
    if (args.particles < 1 or not 0 < args.tmax < float("inf") or not 0 < args.wall_seconds < float("inf")
            or not 0 < args.radial_rms < 1 or not 0 < args.normalization_s < 1
            or not 0 < args.mass_density < float("inf") or args.angular_points < 4):
        parser.error("Need positive finite run settings, 0<radial-rms<1, 0<normalization-s<1 and angular-points>=4")
    if (args.output/"summary.json").exists() or (args.compute and (args.output/"mode.npz").exists()):
        parser.error("Select a fresh --output to preserve existing results")
    os.environ.update(JAX_PLATFORMS="cpu", ESSOS_CPU_DEVICES="1", CUDA_VISIBLE_DEVICES="-1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.perf_counter()

    def stop():
        print(f"Stopped at {args.wall_seconds:g}-second CPU cap; completed artifacts remain saved", flush=True)
        os._exit(124)

    deadline = threading.Timer(args.wall_seconds, stop)
    deadline.daemon = True
    deadline.start()
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from netCDF4 import Dataset
        from booz_xform_jax import Booz_xform
        from essos.boozer import BoozerField, psi0_from_vmec
        from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
        from essos.dynamics import Particles, Tracing
        from essos.fields import Vmec
        from essos.fax_adapter import FaxMode
        from essos.gc_perturbations import VmecBoozerMap, MappedBoozerPerturbation
        from essos.mode_reconstruction import perturbed_fields, physical_components, radial_field_rms
        from examples.benchmarks.sampling import estimate_interval

        args.output.mkdir(parents=True, exist_ok=True)
        booz = Booz_xform(verbose=0, mboz=8, nboz=8)
        booz.read_wout(str(args.wout), flux=False)
        booz.run()
        if args.compute:
            from dataclasses import replace
            from fax.boozer import read_boozmn_geometry, slice_boozer_geometry_surfaces
            from fax.vorticity import physical_potential_problem_from_boozer_geometry, solve_physical_potential_eigenmodes

            booz.write_boozmn(str(args.output/"boozmn.nc"))
            geometry = read_boozmn_geometry(args.output/"boozmn.nc")
            indices = np.unique([np.argmin(abs(np.asarray(geometry.s)-s)) for s in np.linspace(.1, .9, 12)])
            geometry = slice_boozer_geometry_surfaces(geometry, indices)
            problem = physical_potential_problem_from_boozer_geometry(geometry, np.array([[3, 2], [4, 2]]),
                mass_density=args.mass_density, theta_points=16, zeta_points=16,
                radial_nodes=np.linspace(0.,1.,12),boundary_condition="axis")
            result = solve_physical_potential_eigenmodes(problem)
            packet = FaxMode.from_result(problem, result, 0, equilibrium=args.wout, interpretation=args.interpretation)
            packet = replace(packet, metadata=dict(packet.metadata, mass_density_kg_m3=args.mass_density,
                mboz=8, nboz=8, radial_surfaces=len(geometry.s), phase_convention="Re exp(i*(m theta-n zeta+omega t))"))
            packet.save(args.output/"mode.npz")
            print(f"Saved FAX mode: {packet.frequency_hz:g} Hz", flush=True)
        else:
            packet = FaxMode.load(args.mode, equilibrium=args.wout)
        with Dataset(args.wout) as nc:
            psi0 = psi0_from_vmec(float(nc.variables["phi"][-1]))
        background = BoozerField.from_booz(booz.s_b, booz.bmnc_b, booz.xm_b, booz.xn_b,
            booz.iota, booz.Boozer_G, booz.Boozer_I, psi0, int(booz.nfp), angle_grid=None)
        field = Vmec(str(args.wout), ntheta=8, nphi=8, mode_tolerance=1e-7)
        mapping = VmecBoozerMap.from_booz_xform(booz)
        mode = packet.as_physical_potential(potential_volts=1., frequency_sign=args.frequency_sign)
        mapped = MappedBoozerPerturbation(background, mode, mapping)
        rms = radial_field_rms(field, mapped, args.normalization_s, float(mode.omega), angular_points=args.angular_points)
        if not np.isfinite(rms) or rms <= 0:
            raise ValueError("No nonzero radial perturbation at the normalization surface")
        voltage = args.radial_rms/rms
        mode = packet.as_physical_potential(potential_volts=voltage,frequency_sign=args.frequency_sign)
        mapped = MappedBoozerPerturbation(background, mode, mapping)
        print(f"Normalized mode at s={args.normalization_s}: amplitude={voltage:g} V", flush=True)

        rng = np.random.default_rng(args.seed)
        births = np.column_stack([rng.uniform(.25, .75, args.particles), rng.uniform(0, 2*np.pi, args.particles),
                                  rng.uniform(0, 2*np.pi, args.particles)])
        pitches = rng.uniform(-1, 1, args.particles)
        particles = Particles(initial_xyz=jnp.asarray(births), initial_vparallel_over_v=jnp.asarray(pitches),
            mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE, energy=FUSION_ALPHA_PARTICLE_ENERGY)
        gyrofrequency = abs(float(particles.charge))/float(particles.mass)*np.asarray(jax.vmap(field.AbsB)(jnp.asarray(births)))
        frequency_to_gyrofrequency = np.abs(float(mode.omega))/gyrofrequency
        weights = np.abs(np.asarray(jax.vmap(field.sqrtg)(jnp.asarray(births))))
        weights /= weights.sum()
        arrays = dict(s=packet.s, harmonics=packet.harmonics, structure=packet.structure,
                      birth_coordinates=births, birth_pitch=pitches, birth_weights=weights)
        cases = []
        for name, perturbation in (("unperturbed", None), ("perturbed", mapped)):
            trace = Tracing(field=field, particles=particles, model="GuidingCenterAdaptative", perturbation=perturbation,
                integrator="essos_dopri8", maxtime=args.tmax, timestep=min(1e-8, .1/abs(float(mode.omega))), times_to_trace=33,
                rtol=1e-8, atol=1e-9, max_steps=4096, progress=False, devices=jax.devices()[:1])
            if trace.failed.any():
                raise RuntimeError("Unresolved trajectories cannot enter a loss comparison")
            states = np.asarray(trace.trajectories)
            energy = np.asarray(trace.energy())
            arrays.update({f"{name}_states": states, f"{name}_energy": energy})
            event = np.isfinite(trace.lcfs_times)
            case = dict(name=name, failed=int(trace.failed.sum()), lcfs_exits=int(event.sum()),
                        loss_fraction=estimate_interval(event, weights, binary=True))
            if perturbation is not None:
                mu = states[:, :, 4]
                case["max_relative_mu_change"] = float(np.nanmax(np.where(np.isfinite(mu), abs(mu/mu[:, :1]-1), np.nan)))
                if case["max_relative_mu_change"] > 1e-12:
                    raise AssertionError("Magnetic moment changed in a collisionless trace")
            else:
                relative_energy = np.where(np.isfinite(energy), abs(energy/float(particles.energy)-1), np.nan)
                case["max_relative_energy_change"] = float(np.nanmax(relative_energy))
                if case["max_relative_energy_change"] > 1e-3:
                    raise AssertionError("Unperturbed trace fails energy conservation")
            cases.append(case)
            np.savez_compressed(args.output/"diagnostics.npz", times=np.asarray(trace.times), **arrays)
            print(f"Completed {name}: {args.particles} particles", flush=True)
        theta = np.linspace(0, 2*np.pi, 65)
        points = jnp.stack([jnp.full(len(theta), args.normalization_s), jnp.asarray(theta), jnp.full(len(theta), .3)], axis=1)
        def components(point):
            b, e = perturbed_fields(field, mapped, point, .37/abs(float(mode.omega)))
            return physical_components(field, point, b, e)
        b, e = jax.jit(jax.vmap(components))(points)
        arrays.update(theta=theta, delta_B_cart=np.asarray(b), delta_E_cart=np.asarray(e), times=np.asarray(trace.times))
        summary = dict(scope="FAX reduced envelope assigned to Phi; imposed ideal-wave polarization; integration demonstration",
            published_loss_benchmark=False, mode_metadata=packet.metadata, frequency_hz=packet.frequency_hz,
            omega_rad_s=float(mode.omega), potential_scale_volts=voltage, radial_interpolation="natural-cubic",
            radial_rms=args.radial_rms, normalization_s=args.normalization_s, angular_points=args.angular_points,
            particles_per_case=args.particles, tmax_s=args.tmax, cases=cases,
            mass_density_kg_m3=packet.metadata.get("mass_density_kg_m3"),
            birth_frequency_to_gyrofrequency=frequency_to_gyrofrequency.tolist(),
            gc_ordering_validated=False,
            wout=str(args.wout), seed=args.seed, platform="cpu", elapsed_seconds_before_plot=time.perf_counter()-started)
        np.savez_compressed(args.output/"diagnostics.npz", **arrays)
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        plot_saved(args.output)
        summary["elapsed_seconds"] = time.perf_counter()-started
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(f"Saved mode, diagnostics and figure in {args.output}; elapsed {time.perf_counter()-started:.1f} s", flush=True)
    finally:
        deadline.cancel()


if __name__ == "__main__":
    main()
