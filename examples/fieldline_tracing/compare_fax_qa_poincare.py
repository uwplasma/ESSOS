"""Compare native FAX reconstruction with an archived or supplied AE3D mode.

Both complete eigenvectors are reconstructed with the same ideal
closure, frozen phase, magnetic-energy amplitude and field-line integrator.
Defaults use the new QA FAX packet and the pinned archived potential.
--ae3d-mode accepts a separately exported matrix-solution packet instead.
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

from examples.fieldline_tracing.fax_qa_poincare import (
    ROOT, ARTIFACTS, default_equilibrium, embedding_samples, magnetic_flux_components,
    plot_saved, trace_sections,
)


def energy_amplitudes(geometry, background, waves, *, radial_points=32, angular_points=64):
    """sqrt(int delta B^2 dV / int B0^2 dV) at the common snapshot t=0.

    Integrate the full s=0..1 volume with Gauss radial nodes and periodic
    angles. Use the full embedding to convert contravariant delta B into
    physical vectors, the signed Boozer Jacobian for curl and |J| for volume.
    Reuse the geometry sampler used by the FAX operator, including its axis
    extension. Fourier matrix products replace point-by-point reconstruction.
    """
    import jax
    import jax.numpy as jnp
    from fax.vorticity import _sample_boozer_geometry

    nodes, weights = np.polynomial.legendre.leggauss(radial_points)
    nodes, weights = (nodes+1)/2, weights/2
    angle = np.arange(angular_points)*2*np.pi/angular_points
    theta, zeta = np.meshgrid(angle, angle, indexing="ij")
    def sample(geo):
        values = _sample_boozer_geometry(geo, jnp.asarray(nodes), jnp.asarray(theta), jnp.asarray(zeta),
                                         jnp.ones_like(geo.s), 1)
        return values[0], values[3]

    B, basis = map(np.asarray, jax.jit(sample)(geometry))
    iota, G, I = np.asarray(jax.vmap(lambda s: background.profiles(s)[0])(jnp.asarray(nodes))).T
    jacobian = background.psi0*(G+iota*I)[:, None]/B**2
    if not np.all(np.isfinite(B) & (B > 0)) or not np.all(np.isfinite(jacobian) & (abs(jacobian) > 0)):
        raise ValueError("Invalid equilibrium volume quadrature")
    volume = abs(jacobian)*weights[:, None]
    evaluators = [jax.jit(jax.vmap(lambda p, wave=wave:
                   magnetic_flux_components(background, wave, p, 0.))) for wave in waves]
    energies, equilibrium_energy = np.zeros(len(waves)), np.sum(B**2*volume)
    for radial_index, s in enumerate(nodes):
        coordinates = np.column_stack([np.full(theta.size, s), theta.ravel(), zeta.ravel()])
        fluxes = [np.asarray(evaluate(jnp.asarray(coordinates))) for evaluate in evaluators]
        for index, flux in enumerate(fluxes):
            physical = np.einsum("pa,pai->pi", flux, basis[radial_index])/jacobian[radial_index, :, None]
            energies[index] += np.sum(np.sum(physical**2, axis=1)*volume[radial_index])
    return np.sqrt(energies/equilibrium_energy)


def profile_matches(spectrum, reference):
    """Identify profiles with one global complex phase, without changing omega."""
    from scipy.interpolate import CubicSpline

    lookup = {tuple(label): index for index, label in enumerate(spectrum["harmonics"])}
    retained = [i for i, label in enumerate(reference.harmonics) if tuple(label) in lookup]
    if not retained:
        raise ValueError("The two modes have no common harmonic labels")
    columns = [lookup[tuple(reference.harmonics[i])] for i in retained]
    profiles = CubicSpline(spectrum["s"], spectrum["mode_structures"], axis=1,
                           bc_type="natural")(reference.s)[..., columns]
    target = -1j*reference.potential_volts[:, retained]  # same complex Phi convention
    s = reference.s
    weight = np.r_[np.diff(s)[0]/2, (s[2:]-s[:-2])/2, np.diff(s)[-1]/2]
    norm = np.sum(weight[:, None]*abs(profiles)**2, axis=(-2, -1))
    target_norm = np.sum(weight[:, None]*abs(target)**2)
    cross = np.sum(weight[:, None]*profiles.conj()*target, axis=(-2, -1))
    overlap = np.clip(abs(cross)/np.sqrt(np.maximum(norm*target_norm, 1e-300)), 0, 1)
    phase = np.divide(cross, abs(cross), out=np.ones_like(cross), where=abs(cross) > 0)
    return phase, overlap, profiles, target


def add_cross_sections(geometry, arrays, cases):
    for case in cases:
        sections = arrays[case]
        R, Z = np.full(sections.shape[:2], np.nan), np.full(sections.shape[:2], np.nan)
        valid = np.isfinite(sections).all(axis=-1)
        points = sections[valid]
        indices = np.flatnonzero(valid)
        for start in range(0, len(points), 256):
            block = points[start:start+256]
            r, z, _, _ = embedding_samples(geometry, block[:, 0], block[:, 1], np.zeros(len(block)))
            R.ravel()[indices[start:start+256]], Z.ravel()[indices[start:start+256]] = r, z
        arrays[case+"_R"], arrays[case+"_Z"] = R, Z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fax-mode", type=Path, default=ARTIFACTS/"fax_qa_poincare/mode.npz")
    parser.add_argument("--ae3d-mode", type=Path, help="Supplied matrix-solution packet; otherwise read archived AE3D potential")
    parser.add_argument("--boozmn", type=Path, default=default_equilibrium())
    parser.add_argument("--energy-amplitude", type=float, default=1e-3)
    parser.add_argument("--energy-radial-points", type=int, default=32)
    parser.add_argument("--energy-angular-points", type=int, default=64)
    parser.add_argument("--lines", type=int, default=24)
    parser.add_argument("--turns", type=int, default=64)
    parser.add_argument("--steps-per-turn", type=int, default=256)
    parser.add_argument("--output", type=Path, default=ARTIFACTS/"fax_qa_matrix_poincare")
    parser.add_argument("--wall-seconds", type=float, default=120.)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--data", type=Path, help="Saved .json/.npz prefix for report-only")
    args = parser.parse_args()
    if args.report_only:
        plot_saved(args.output, args.data)
        return
    if (args.energy_radial_points < 4 or args.energy_angular_points < 16 or args.lines < 2
            or args.turns < 1 or args.steps_per_turn < 32
            or not all(np.isfinite(v) and v > 0 for v in (args.wall_seconds, args.energy_amplitude))):
        parser.error("Invalid resolution, amplitude or wall cap")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh --output")
    if not all(p.is_file() for p in (args.fax_mode,args.boozmn,*(() if args.ae3d_mode is None else (args.ae3d_mode,)))):
        parser.error("Need the QA boozmn and selected mode packets; see docs/fax_modes.rst")
    os.environ.update(JAX_PLATFORMS="cpu", ESSOS_CPU_DEVICES="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary = dict(status="running", cases={})

    def save():
        temporary = args.output/"summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.output/"summary.json")

    def stop():
        summary.update(status="wall_clock_limit_reached", wall_seconds=time.perf_counter()-started)
        save()
        print("Comparison CPU cap reached; completed data saved", flush=True)
        os._exit(124)

    timer = threading.Timer(args.wall_seconds, stop)
    timer.daemon = True
    timer.start()
    try:
        import jax
        import jax.numpy as jnp
        from netCDF4 import Dataset
        from types import SimpleNamespace
        from fax.boozer import read_boozmn_geometry
        from essos.fax_adapter import FaxMode
        from examples.benchmarks.ae3d_firm3d_loss import equilibrium

        jax.config.update("jax_enable_x64", True)
        packets = [FaxMode.load(args.fax_mode,equilibrium=args.boozmn)]
        if args.ae3d_mode is None:
            from essos.ae3d_adapter import AE3DMode
            from examples.benchmarks.ae3d_firm3d_loss import TAG
            archived = AE3DMode.from_firm3d_numpy(args.boozmn.parent/"ae3d_saw_scaled"/f"{TAG}.npy")
            packets.append(FaxMode(archived.s,archived.harmonics,-1j*archived.potential_volts,
                archived.omega/(2*np.pi),dict(interpretation="potential_shape",coordinates="essos_boozer",
                equilibrium_sha256=packets[0].metadata["equilibrium_sha256"],source="archived AE3D potential",
                source_sha256=archived.source_sha256)))
        else:
            packets.append(FaxMode.load(args.ae3d_mode,equilibrium=args.boozmn))
        ae3d_label = "Archived AE3D" if args.ae3d_mode is None else "Supplied AE3D matrix solution"
        reference = packets[1]
        phase, overlap, _, _ = profile_matches(
            dict(s=packets[0].s, harmonics=packets[0].harmonics, mode_structures=packets[0].structure[None]),
            SimpleNamespace(s=reference.s, harmonics=reference.harmonics,
                            potential_volts=np.real(1j*reference.structure)))
        packets[0] = replace(packets[0], structure=packets[0].structure*phase[0])
        geometry = read_boozmn_geometry(args.boozmn)
        with Dataset(args.boozmn) as nc:
            s = (np.asarray(nc["jlist"][:], float)-1.5)/(int(nc["ns_b"][:])-1)
        geometry = replace(geometry, s=jnp.asarray(s))
        background = equilibrium(args.boozmn, angular_points=16)
        units = [p.as_physical_potential(potential_volts=1., frequency_sign=1) for p in packets]
        print("Normalizing the two matrix solutions to the same magnetic energy", flush=True)
        amplitudes = energy_amplitudes(geometry, background, units,
            radial_points=args.energy_radial_points, angular_points=args.energy_angular_points)
        if not np.all(np.isfinite(amplitudes) & (amplitudes > 0)):
            raise ValueError("Cannot normalize a zero or invalid field")
        voltages = args.energy_amplitude/amplitudes
        waves = [p.as_physical_potential(potential_volts=float(v), frequency_sign=1)
                 for p, v in zip(packets, voltages)]
        initial = np.column_stack([np.linspace(.08, .92, args.lines), np.zeros(args.lines)])
        arrays = dict(initial=initial, profile_overlap=overlap)
        summary.update(magnetic_energy_amplitude=args.energy_amplitude,
            normalization="sqrt(int |delta B|^2 dV / int |B0|^2 dV), full volume at common t=0 snapshot",
            energy_quadrature=dict(radial=args.energy_radial_points, angular=args.energy_angular_points),
            profile_overlap=float(overlap[0]), native_global_phase=[float(phase[0].real), float(phase[0].imag)],
            saved_packet_phase_applied=True,
            equilibrium_sha256=packets[0].metadata["equilibrium_sha256"],
            lines=args.lines, turns=args.turns, steps_per_turn=args.steps_per_turn,
            field_model="ESSOS K=0 Boozer; curl(alpha B0); natural-cubic physical Phi; frozen t=0",
            imported_original_AE3D_matrices=False,
            source_note=f"Native FAX versus {ae3d_label}; see packet metadata for matrix provenance",
            figure_stem="fax_qa_matrix_poincare", plot_cases=["fax", "ae3d"],
            plot_titles=[f"FAX matrix construction\n{packets[0].frequency_hz/1000:.3f} kHz",
                         f"{ae3d_label}\n{packets[1].frequency_hz/1000:.3f} kHz"],
            plot_caption=(f"Same QA equilibrium, field-line seeds and frozen phase; full calculated harmonic profiles\n"
                          f"Both fields have magnetic-energy amplitude {args.energy_amplitude:g}; Boozer zeta = 0 mod 2 pi"))
        save()
        for index, case in enumerate(("fax", "ae3d")):
            packet = packets[index]
            packet.save(args.output/f"{case}_mode.npz")
            print(f"Tracing {case}: {len(packet.harmonics)} harmonics, {args.lines} lines, {args.turns} turns", flush=True)
            arrays[case], stopped = trace_sections(background, waves[index], initial,
                turns=args.turns, steps_per_turn=args.steps_per_turn, frozen_time=0.)
            arrays[case+"_stop_codes"] = stopped
            arrays[case+"_s"], arrays[case+"_harmonics"], arrays[case+"_structure"] = packet.s, packet.harmonics, packet.structure
            summary["cases"][case] = dict(frequency_hz=packet.frequency_hz, omega_rad_s=float(waves[index].omega),
                potential_volts=float(voltages[index]), harmonics=len(packet.harmonics),
                source=packet.metadata, completed=int(np.sum(stopped == 0)),
                domain_exits=int(np.sum(stopped == 1)), invalid=int(np.sum(stopped == 2)))
            np.savez_compressed(args.output/"poincare.npz", **arrays)
            save()
        add_cross_sections(geometry, arrays, summary["plot_cases"])
        summary.update(status="completed", wall_seconds=time.perf_counter()-started)
        np.savez_compressed(args.output/"poincare.npz", **arrays)
        save()
        plot_saved(args.output)
        print(json.dumps(dict(status=summary["status"], wall_seconds=summary["wall_seconds"],
            profile_overlap=summary["profile_overlap"], cases={k: {key: v[key] for key in
            ("frequency_hz", "completed", "domain_exits", "invalid")} for k, v in summary["cases"].items()}), indent=2), flush=True)
        if any(row["invalid"] for row in summary["cases"].values()):
            raise RuntimeError("Invalid field lines: inspect saved stop codes")
    except Exception as error:
        summary.update(status="failed", error=str(error), wall_seconds=time.perf_counter()-started)
        save()
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
