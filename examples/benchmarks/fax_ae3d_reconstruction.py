"""Compare a native FAX QA mode and its ESSOS reconstruction with AE3D.

Default: coefficient-reconstruction control only, with a 120-second cap.
--compute: FAX builds and solves its physical-potential matrices, using the
same archived equilibrium, 35-harmonic basis and the paper's flat QA density.
No particle tracing is performed. See docs/fax_modes.rst for claim limits.
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
from scipy.interpolate import CubicSpline

from examples.benchmarks.ae3d_firm3d_loss import PACKET, TAG, equilibrium, verify_packet


ROOT = Path(__file__).resolve().parents[2]
PAPER_DENSITY = 1.88e-6  # kg/m^3; constant QA profile, paper section 3 / table 1.


def align_profiles(computed, reference, s):
    """One complex least-squares gain; never fit individual harmonics.

    Compare the physical Phi profiles with trapezoidal radial weighting.
    An eigenvector's global phase and amplitude are arbitrary; frequency,
    radial shape and all relative harmonic amplitudes/phases stay unchanged.
    """
    computed, reference, s = np.asarray(computed), np.asarray(reference), np.asarray(s)
    if (computed.shape != reference.shape or computed.ndim != 2 or len(s) != len(computed)
            or len(s) < 2 or not np.isfinite(s).all() or np.any(np.diff(s) <= 0)
            or not np.isfinite(computed).all() or not np.isfinite(reference).all()):
        raise ValueError("Need matching finite radial/harmonic profiles and an increasing grid")
    weights = np.r_[np.diff(s)[0]/2, (s[2:]-s[:-2])/2, np.diff(s)[-1]/2,]
    norm_c = float(np.sum(weights[:, None]*abs(computed)**2))
    norm_r = float(np.sum(weights[:, None]*abs(reference)**2))
    if norm_c <= 0 or norm_r <= 0:
        raise ValueError("Cannot compare zero profiles")
    cross = np.sum(weights[:, None]*computed.conj()*reference)
    gain = cross/norm_c
    overlap = float(np.clip(abs(cross)/np.sqrt(norm_c*norm_r), 0., 1.))
    error = float(np.sqrt(np.sum(weights[:, None]*abs(gain*computed-reference)**2)/norm_r))
    return gain, dict(complex_overlap=overlap, aligned_profile_relative_l2=error,
                      gain_real=float(gain.real), gain_imag=float(gain.imag))


def half_mesh_s(boozmn):
    """Use exactly the radial convention of the archived ESSOS loss case."""
    from netCDF4 import Dataset

    with Dataset(boozmn) as nc:
        return (np.asarray(nc["jlist"][:], float)-1.5)/(int(nc["ns_b"][:])-1)


def diagnostic_chart(background):
    """K=0 Boozer diagnostic chart in (u,w,zeta); no GC RHS changes.

    s=u^2+w^2, Bcov=I*dtheta+G*dzeta and sqrt(g_uw)=2*psi0*(G+iota*I)/B^2.
    Fields are compared component by component in this common chart, not
    reported as Cartesian components or an independent AE3D field solution.
    """
    import equinox as eqx
    import jax.numpy as jnp

    class Chart(eqx.Module):
        background: object

        def profiles(self, s):
            return self.background.profiles(s)

        def B_covariant(self, point):
            u, w, _ = point
            s = u*u+w*w
            (_, G, I), _ = self.profiles(s)
            return jnp.array([-I*w/s, I*u/s, G])

        def sqrtg(self, point):
            u, w, zeta = point
            s = u*u+w*w
            (iota, G, I), _ = self.profiles(s)
            B = self.background.modB(s, jnp.arctan2(w, u), zeta)
            return 2*self.background.psi0*(G+iota*I)/(B*B)

        def curl_B(self, point):
            u, w, _ = point
            _, (_, dG, dI) = self.profiles(u*u+w*w)
            return 2*jnp.array([w*dG, -u*dG, dI])/self.sqrtg(point)

    return Chart(background)


def sample_fields(background, wave, omega, count=48):
    import jax
    import jax.numpy as jnp
    from essos.mode_reconstruction import perturbed_fields

    # The same positions and wave phases for every model; its own frequency
    # remains in alpha and the inductive electric field. No axis sampling.
    index = np.arange(count)
    s = .025+.95*(index+.5)/count
    theta = np.mod(index*2.399963229728653, 2*np.pi)
    zeta = np.mod(index*1.2533141373155, 2*np.pi)
    points = jnp.asarray(np.column_stack([np.sqrt(s)*np.cos(theta), np.sqrt(s)*np.sin(theta), zeta]))
    times = jnp.asarray(np.mod(index*.37, 2*np.pi)/omega)
    chart = diagnostic_chart(background)

    def evaluate(point, t):
        values, derivatives = wave.derivatives(chart, point, t)
        B, E = perturbed_fields(chart, wave, point, t)
        return values, derivatives, B, E

    return tuple(np.asarray(x) for x in jax.jit(jax.vmap(evaluate))(points, times))


def compare_fields(computed, reference):
    def relative(a, b):
        return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))

    values, derivatives, B, E = computed
    rv, rd, rB, rE = reference
    return dict(alpha_relative_l2=relative(values[:, 0], rv[:, 0]),
                phi_relative_l2=relative(values[:, 1], rv[:, 1]),
                potential_derivative_relative_l2=[relative(derivatives[:, i], rd[:, i]) for i in range(2)],
                delta_B_component_relative_l2=[relative(B[:, i], rB[:, i]) for i in range(3)],
                delta_E_component_relative_l2=[relative(E[:, i], rE[:, i]) for i in range(3)])


def compute_native(reference, boozmn, args, output):
    import jax.numpy as jnp
    import fax
    from fax.boozer import read_boozmn_geometry
    from fax.vorticity import physical_potential_problem_from_boozer_geometry, solve_physical_potential_eigenmodes
    from essos.fax_adapter import FaxMode

    geometry = read_boozmn_geometry(boozmn)
    actual_s = half_mesh_s(boozmn)
    if len(actual_s) != len(geometry.s):
        raise ValueError("FAX/ESSOS geometry rows disagree")
    shift = float(np.max(abs(np.asarray(geometry.s)-actual_s)))
    # FAX's reader uses (jlist-1)/(ns-1). This packet stores half-mesh
    # coefficients: retain its arrays but use the loss benchmark's half mesh.
    geometry = replace(geometry, s=jnp.asarray(actual_s))
    problem = physical_potential_problem_from_boozer_geometry(geometry, reference.harmonics,
        mass_density=args.mass_density, theta_points=args.angular_points, zeta_points=args.angular_points,
        angular_basis="complex",radial_nodes=np.linspace(0.,1.,args.radial_points),boundary_condition="axis")
    print(f"FAX native assembly/solve: {args.radial_points} nodes, {len(reference.harmonics)} harmonics, "
          f"rho={args.mass_density:g} kg/m^3", flush=True)
    result = solve_physical_potential_eigenmodes(problem)
    profiles = CubicSpline(np.asarray(problem.s), np.asarray(result.mode_structures), axis=1,
                           bc_type="natural")(reference.s)
    candidates = []
    for index, (frequency, residual, omega2) in enumerate(zip(
            np.asarray(result.frequencies_hz), np.asarray(result.residuals), np.asarray(result.omega2))):
        if not (np.isfinite(frequency) and frequency > 0 and np.isfinite(residual)
                and residual <= 1e-7 and omega2 > 0):
            continue
        _, metrics = align_profiles(profiles[index], -1j*reference.potential_volts, reference.s)
        candidates.append(dict(index=index, frequency_hz=float(frequency), residual=float(residual), **metrics))
    if not candidates:
        raise ValueError("No positive, converged native FAX eigenpair")
    candidates.sort(key=lambda item: item["complex_overlap"], reverse=True)
    selected = candidates[0]["index"]  # Explicit shape-based identification, not retuning frequency.
    packet = FaxMode.from_result(problem, result, selected, equilibrium=boozmn,
                                 interpretation="potential_shape", source="fax_native_qa")
    packet = replace(packet, metadata=dict(packet.metadata, density_kg_m3=args.mass_density,
        density_provenance="paper section 3 and table 1: flat QA density" if args.mass_density == PAPER_DENSITY else "user override",
        radial_coordinate="s=(jlist-1.5)/(ns_b-1); same as archived loss benchmark",
        radial_coordinate_reader_shift=shift, radial_points=args.radial_points, angular_points=args.angular_points,
        boundary_note="Regular axis on s=0..1, conducting outer wall",
        operator="physical-potential vorticity",angular_basis="complex",
        mode_selection="largest global phase/amplitude-independent profile overlap with archived mode",
        fax_module=str(Path(fax.__file__).resolve())))
    packet.save(output/"fax_native_mode.npz")
    np.savez_compressed(output/"fax_native_spectrum.npz", frequencies_hz=np.asarray(result.frequencies_hz),
                        residuals=np.asarray(result.residuals), omega2=np.asarray(result.omega2))
    return packet, candidates[:8]


def save_report(output, summary, arrays):
    np.savez_compressed(output/"diagnostics.npz", **arrays)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")


def plot_saved(output, data_prefix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((output/"summary.json" if data_prefix is None else data_prefix.with_suffix(".json")).read_text())
    with np.load(output/"diagnostics.npz" if data_prefix is None else data_prefix.with_suffix(".npz")) as data:
        arrays = dict(data)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    s, reference = arrays["s"], arrays["reference_profiles"]
    strongest = np.argsort(np.max(abs(reference), axis=0))[-5:][::-1]
    for i in strongest:
        label = tuple(map(int, arrays["harmonics"][i]))
        line, = axes[0, 0].plot(s, abs(reference[:, i]), label=f"AE3D {label}")
        if "native_profiles" in arrays:
            axes[0, 0].plot(s, abs(arrays["native_profiles"][:, i]), "--", color=line.get_color())
    axes[0, 0].set(xlabel="s", ylabel="Physical Phi envelope [V]",
                   title="Solid: AE3D; dashed: aligned native FAX" if "native_profiles" in arrays else "Archived AE3D physical potential profiles")
    control = summary["reconstruction_control"]
    errors = [control["phi_relative_l2"], control["alpha_relative_l2"],
              max(control["delta_B_component_relative_l2"]), max(control["delta_E_component_relative_l2"])]
    axes[0, 1].bar(["Phi", "alpha", "delta B", "delta E"], np.maximum(errors, 1e-17))
    axes[0, 1].set(yscale="log", ylabel="Relative L2 error",
                   title="CONTROL: copied AE3D coefficients AND frequency\nExchange/reconstruction only; no native FAX mode")
    axes[0, 1].axhline(1e-9, color="k", ls=":", label="Control acceptance limit")
    axes[0, 1].set_ylim(1e-17, 1e-8)
    axes[0, 1].legend(fontsize=7)
    if "native_profiles" in arrays:
        axes[1, 0].scatter([0, 1], [summary["reference_frequency_hz"], summary["native"]["frequency_hz"]], s=60)
        axes[1, 0].set(xticks=[0, 1], xticklabels=["Archived AE3D", "FAX best shape overlap"],
                       yscale="log", ylabel="Frequency [Hz]", title="Actual frequencies; no frequency rescaling")
        for component, label in enumerate(("u", "w", "zeta")):
            line, = axes[1, 1].plot(arrays["reference_delta_B"][:, component], label=f"AE3D {label}")
            axes[1, 1].plot(arrays["native_delta_B"][:, component], "--", color=line.get_color(), label=f"FAX {label}")
        axes[1, 1].set(xlabel="Sample index", ylabel="Contravariant delta B components", title="Common Boozer chart; own mode frequency")
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(.05, .7, "Reconstruction control complete.\nNative FAX solve has not run.\nUse --compute on the GPU machine.", transform=axes[1, 0].transAxes)
        axes[1, 1].axis("off")
    for ax in axes.flat:
        if ax.axison:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    if "native_profiles" in arrays:
        axes[1, 1].legend(fontsize=7)
    fig.suptitle("QA native FAX screen versus archived AE3D; no particle tracing\n"
                 "The copied-input control is separate from the native FAX calculation")
    for suffix in ("png", "svg"):
        fig.savefig(output/f"fax_ae3d_reconstruction.{suffix}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--compute", action="store_true", help="Assemble and solve native FAX matrices")
    source.add_argument("--mode", type=Path, help="Previously calculated native FAX packet")
    source.add_argument("--report-only", action="store_true")
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--data", type=Path, help="Saved .json/.npz prefix for --report-only")
    parser.add_argument("--output", type=Path, default=ROOT.parent/"artifacts/fax_integration/fax_ae3d_reconstruction")
    parser.add_argument("--platform", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--radial-points", type=int, default=49)
    parser.add_argument("--angular-points", type=int, default=32)
    parser.add_argument("--mass-density", type=float, default=PAPER_DENSITY)
    parser.add_argument("--wall-seconds", type=float, default=120.)
    args = parser.parse_args()
    if args.report_only:
        plot_saved(args.output, args.data)
        return
    if args.data is not None:
        parser.error("--data requires --report-only")
    if (args.radial_points < 4 or args.angular_points < 8 or not np.isfinite(args.mass_density)
            or args.mass_density <= 0 or not np.isfinite(args.wall_seconds) or args.wall_seconds <= 0):
        parser.error("Need >=4 radial points, >=8 angular points, positive finite density and wall cap")
    if (args.output/"summary.json").exists():
        parser.error("Choose a fresh --output to preserve previous results")
    os.environ.update(JAX_PLATFORMS=args.platform, ESSOS_CPU_DEVICES="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    args.output.mkdir(parents=True, exist_ok=True)

    def stop():
        (args.output/"run_status.json").write_text(json.dumps(dict(status="wall_clock_limit_reached", wall_seconds=args.wall_seconds)))
        print("Process cap reached; saved stages remain available", flush=True)
        os._exit(124)

    deadline = threading.Timer(args.wall_seconds, stop)
    deadline.daemon = True
    deadline.start()
    started = time.perf_counter()
    try:
        import jax
        from essos.ae3d_adapter import AE3DMode
        from essos.fax_adapter import FaxMode

        base = verify_packet(args.packet)/"simulations_fig4_fig5/QA/all_inputs"
        boozmn = base/"boozmn_betaQA.nc"
        reference = AE3DMode.from_firm3d_numpy(base/"ae3d_saw_scaled"/f"{TAG}.npy")
        field = equilibrium(boozmn)
        metadata = dict(interpretation="potential_shape", coordinates="essos_boozer",
                        equilibrium_sha256=hashlib.sha256(boozmn.read_bytes()).hexdigest(),
                        source="AE3D coefficients and frequency copied for a control; no native FAX solve")
        control_packet = FaxMode(reference.s, reference.harmonics, -1j*reference.potential_volts,
                                 reference.omega/(2*np.pi), metadata)
        control_packet.save(args.output/"reconstruction_control_mode.npz")
        control_packet = FaxMode.load(args.output/"reconstruction_control_mode.npz", equilibrium=boozmn)
        control = control_packet.as_physical_potential(potential_volts=np.max(abs(reference.potential_volts)), frequency_sign=1)
        reference_fields = sample_fields(field, reference.as_perturbation(), reference.omega)
        control_fields = sample_fields(field, control, reference.omega)
        control_metrics = compare_fields(control_fields, reference_fields)
        errors = [value if isinstance(value, list) else [value] for value in control_metrics.values()]
        passed = max(x for row in errors for x in row) < 1e-9
        summary = dict(scope="Native FAX QA mode/reconstruction screening, separate from the recorded AE3D loss benchmark",
            reference_frequency_hz=reference.omega/(2*np.pi), mode_tag=TAG, harmonics=len(reference.harmonics),
            density_kg_m3=args.mass_density, density_source="https://arxiv.org/html/2603.03118v1#S3.T1",
            platform=jax.default_backend(), device_kind=jax.devices()[0].device_kind,
            reconstruction_control=control_metrics, reconstruction_control_passed=passed,
            reconstruction_control_description="Copies the AE3D coefficients AND frequency; validates interchange/reconstruction, not a native FAX prediction",
            field_components="contravariant delta B and covariant delta E in the same regular Boozer chart; not Cartesian",
            native_mode_calculated=False, particle_tracing=False)
        arrays = dict(s=reference.s, harmonics=reference.harmonics, reference_profiles=-1j*reference.potential_volts,
                      reference_delta_B=reference_fields[2], reference_delta_E=reference_fields[3])
        save_report(args.output, summary, arrays)
        plot_saved(args.output)
        if not passed:
            raise AssertionError(f"Same-coefficient reconstruction control failed: {control_metrics}")
        print("Same-coefficient FAX packet/reconstruction control passed", flush=True)
        if args.compute or args.mode:
            packet, candidates = (compute_native(reference, boozmn, args, args.output) if args.compute else
                                  (FaxMode.load(args.mode, equilibrium=boozmn), []))
            if packet.metadata.get("source") != "fax_native_qa":
                raise ValueError("Use a native QA calculation packet, not a synthetic reconstruction control")
            if not np.array_equal(packet.harmonics, reference.harmonics):
                raise ValueError("Saved mode must use the same ordered 35-harmonic basis")
            if packet.s[0] > reference.s[0] or packet.s[-1] < reference.s[-1]:
                raise ValueError("Mode must enclose the reference grid; extrapolation is not a comparison")
            profiles = CubicSpline(packet.s, packet.structure, axis=0, bc_type="natural")(reference.s)
            gain, metrics = align_profiles(profiles, -1j*reference.potential_volts, reference.s)
            aligned = replace(packet, structure=packet.structure*gain,
                metadata=dict(packet.metadata, amplitude_policy="one global complex least-squares gain against AE3D Phi; not predicted saturation",
                              gain_real=float(gain.real), gain_imag=float(gain.imag)))
            aligned.save(args.output/"fax_aligned_mode.npz")
            wave = aligned.as_physical_potential(potential_volts=np.max(abs(aligned.structure)), frequency_sign=1)
            native_fields = sample_fields(field, wave, 2*np.pi*packet.frequency_hz)
            summary.update(native_mode_calculated=bool(args.compute), native_packet_evaluated=True,
                density_kg_m3=packet.metadata.get("density_kg_m3"),
                native=dict(frequency_hz=packet.frequency_hz,
                    frequency_relative_difference=abs(packet.frequency_hz/summary["reference_frequency_hz"]-1),
                    profile_alignment=metrics, fields=compare_fields(native_fields, reference_fields),
                    metadata=aligned.metadata, top_overlap_candidates=candidates,
                    validation_status="Screening only: native operator equivalence, boundary treatment and radial/angular convergence remain to be established"))
            arrays.update(native_profiles=gain*profiles, native_delta_B=native_fields[2], native_delta_E=native_fields[3])
            print(json.dumps(summary["native"], indent=2), flush=True)
        summary["elapsed_seconds"] = time.perf_counter()-started
        save_report(args.output, summary, arrays)
        plot_saved(args.output)
        print(f"Saved mode/field comparison in {args.output}; no particle tracing", flush=True)
    finally:
        deadline.cancel()


if __name__ == "__main__":
    main()
