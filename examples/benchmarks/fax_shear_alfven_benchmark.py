"""FAX -> reconstructed fields -> existing adaptive GC, against exact results.

Uniform-plasma shear-Alfven verification, not a toroidal alpha-loss prediction.
16 deterministic markers, 64 short trajectories, 120-second CPU process cap.
See docs/fax_modes.rst for the independent orbit and first-exit derivations.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import threading
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/"examples/benchmarks/references/fax_shear_alfven_input.json"


@dataclass(frozen=True)
class SlabCase:
    B0: float
    density: float
    k_parallel: float
    k_perp: float
    delta_b_over_b: float
    wall_x: float
    periods: float
    mass: float
    charge: float
    energy: float

    @property
    def alfven_speed(self):
        return self.B0/np.sqrt(4*np.pi*1e-7*self.density)

    @property
    def omega(self):
        return self.k_parallel*self.alfven_speed

    @property
    def alpha_amplitude(self):
        return self.delta_b_over_b/self.k_perp

    @property
    def potential_volts(self):
        return self.B0*self.alfven_speed*self.alpha_amplitude

    @property
    def excursion(self):
        return self.delta_b_over_b/self.k_parallel

    @property
    def duration(self):
        return self.periods*2*np.pi/self.omega

    @property
    def speed(self):
        return np.sqrt(2*self.energy/self.mass)


def load_case():
    return SlabCase(**json.loads(INPUT.read_text()))


def markers(case):
    """Four velocities x four phases; equal weights, no Monte Carlo sampling."""
    phase = np.tile(np.arange(4)*np.pi/2, 4)
    velocity = np.repeat([-.7*case.speed, .15*case.speed, case.alfven_speed, .7*case.speed], 4)
    points = np.column_stack([np.tile(np.linspace(.730, .734, 4), 4),
                              phase/case.k_perp, np.zeros(16)])
    return points, velocity


def exact_orbit(case, points, velocity, times):
    """NumPy closed form, independent of FAX and the ESSOS RHS/field code."""
    times = np.asarray(times)
    phase0 = case.k_perp*points[:, 1]+case.k_parallel*points[:, 2]
    detuning = case.k_parallel*(velocity-case.alfven_speed)
    result = np.broadcast_to(points[:, None, :], (len(points), len(times), 3)).copy()
    result[:, :, 0] += case.excursion*(np.cos(phase0[:, None]+detuning[:, None]*times)
                                       -np.cos(phase0[:, None]))
    result[:, :, 2] += velocity[:, None]*times
    return result


def exact_first_exit(case, points, velocity):
    """Earliest root of X(t)=wall, in either direction of phase propagation."""
    phase0 = case.k_perp*points[:, 1]+case.k_parallel*points[:, 2]
    detuning = case.k_parallel*(velocity-case.alfven_speed)
    target = np.cos(phase0)+(case.wall_x-points[:, 0])/case.excursion
    angle = np.arccos(np.clip(target, -1., 1.))
    direction = np.where(detuning >= 0, 1., -1.)
    steps = np.mod(direction[:, None]*(np.stack([angle, -angle], axis=1)-phase0[:, None]), 2*np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        exits = steps.min(axis=1)/abs(detuning)
    # Exact comoving cancellation; rounding in Particles' pitch conversion is harmless.
    possible = (abs(target) <= 1) & (abs(detuning) > 1e-12*case.omega)
    exits = np.where(possible & (exits <= case.duration), exits, np.inf)
    return np.where(points[:, 0] >= case.wall_x, 0., exits)


def sampled_first_exit(times, x, wall):
    """Linear first-crossing interpolation of saved, unrestricted trajectories."""
    crossed = x >= wall
    index = crossed.argmax(axis=1)
    before = np.maximum(index-1, 0)
    row = np.arange(len(x))
    denominator = x[row, index]-x[row, before]
    fraction = np.divide(wall-x[row, before], denominator,
                         out=np.zeros(len(x)), where=denominator != 0)
    estimate = times[before]+fraction*(times[index]-times[before])
    return np.where(crossed.any(axis=1), np.where(index == 0, times[0], estimate), np.inf)


def compute_mode(case, output):
    """Solve three actual FAX matrices; the oracle never calls FAX's helper."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from fax.global_modes import constant_radial_block_problem, solve_radial_block_eigenmodes
    from essos.fax_adapter import FaxMode

    arrays, metrics, selected = {}, [], None
    coupling = (.2*case.omega)**2
    for count in (8, 16, 32):
        problem = constant_radial_block_problem(radial_points=count, harmonic_count=1,
            continuum_frequency_hz=case.omega/(2*np.pi), radial_coupling_omega2=coupling,
            boundary_condition="neumann")
        # Re-label the uniform slab's straight-field-line angles explicitly.
        problem = replace(problem, s=.05+.9*problem.s, harmonics=jnp.array([[0, -1]]))
        result = solve_radial_block_eigenmodes(problem)
        exact = case.omega**2+4*coupling/(.9/(count-1))**2*np.sin(np.arange(count)*np.pi/(2*count))**2
        actual = np.asarray(result.omega2)
        arrays[f"spectrum_{count}"] = actual/case.omega**2
        arrays[f"spectrum_exact_{count}"] = exact/case.omega**2
        shape = np.asarray(result.mode_structures[0])
        shape = shape*np.exp(-1j*np.angle(shape.mean()))
        metrics.append(dict(radial_points=count, max_relative_omega2_error=float(np.max(abs(actual/exact-1))),
            frequency_relative_error=float(abs(np.sqrt(actual[0])/case.omega-1)),
            constant_envelope_error=float(np.max(abs(shape/shape.mean()-1))),
            max_eigenpair_residual=float(np.max(np.asarray(result.residuals)))))
        if count == 16:
            packet = FaxMode.from_result(problem, result, 0, equilibrium=INPUT,
                interpretation="potential_shape", source="fax uniform-plasma Neumann fixture")
            selected = replace(packet, structure=shape, metadata=dict(packet.metadata,
                global_phase="mean coefficient real and positive", coordinates_detail="s=X/1m; theta=kyY; zeta=kyY+kzZ",
                equilibrium_kind="uniform Cartesian plasma", radial_coupling_omega2=coupling))
    selected.save(output/"mode.npz")
    # Include the real interchange round trip in the transport benchmark.
    return FaxMode.load(output/"mode.npz", equilibrium=INPUT), arrays, metrics


def reconstructed_wave(case, packet):
    import equinox as eqx
    import jax.numpy as jnp
    from essos.gc_perturbations import MappedBoozerPerturbation

    class UniformField:
        def B_covariant(self, point): return jnp.array([0., 0., case.B0])
        B_contravariant = B_covariant
        def AbsB(self, point): return jnp.asarray(case.B0)
        def dAbsB_by_dX(self, point): return jnp.zeros(3)
        curl_B = curl_b = kappa = dAbsB_by_dX
        def sqrtg(self, point): return jnp.ones(())
        def to_xyz(self, point): return point

    class SlabProfiles(eqx.Module):
        def profiles(self, s):
            G = case.B0/case.k_parallel
            return jnp.array([0., G, -G]), jnp.zeros(3)

    def coordinates(point):
        return jnp.array([point[0], case.k_perp*point[1],
                          case.k_perp*point[1]+case.k_parallel*point[2]])

    mode = packet.as_perturbation(potential_volts=case.potential_volts, frequency_sign=-1, taper_width=.05)
    return UniformField(), MappedBoozerPerturbation(SlabProfiles(), mode, coordinates)


def field_metrics(case, field, wave):
    import jax
    import jax.numpy as jnp
    from essos.dynamics import _guiding_center_velocity
    from essos.electric_field import Electric_field_zero
    from essos.mode_reconstruction import perturbed_fields

    points = np.array([[.70, .8, .3], [.72, 4., 2.], [.76, 8., -3.]])
    t = .17/case.omega
    phase = case.k_perp*points[:, 1]+case.k_parallel*points[:, 2]-case.omega*t
    expected_b = np.zeros_like(points)
    expected_b[:, 0] = -case.B0*case.delta_b_over_b*np.sin(phase)
    expected_e = np.zeros_like(points)
    expected_e[:, 1] = case.potential_volts*case.k_perp*np.sin(phase)
    b, e = jax.jit(jax.vmap(lambda p: perturbed_fields(field, wave, p, t)))(jnp.asarray(points))
    expected_potentials = np.cos(phase)[:, None]*np.array([case.alpha_amplitude, case.potential_volts])
    potentials = jax.vmap(lambda p: wave.potentials(field, p, t))(jnp.asarray(points))
    dx, dv = _guiding_center_velocity(field, Electric_field_zero(), case.charge, case.mass,
        jnp.asarray(points[0]), case.alfven_speed, mu=case.energy/case.B0, t=t, perturbation=wave)
    drift_scale = case.delta_b_over_b*case.alfven_speed
    return dict(potential_relative_error=float(np.max(abs(np.asarray(potentials)-expected_potentials)
                                                  /np.array([case.alpha_amplitude, case.potential_volts]))),
                magnetic_relative_error=float(np.max(abs(np.asarray(b)-expected_b))/(case.B0*case.delta_b_over_b)),
                electric_relative_error=float(np.max(abs(np.asarray(e)-expected_e))/(case.potential_volts*case.k_perp)),
                parallel_electric_relative_error=float(np.max(abs(np.asarray(e)[:, 2]))/(case.potential_volts*case.k_perp)),
                comoving_radial_velocity_normalized=float(abs(dx[0])/drift_scale),
                comoving_parallel_acceleration_normalized=float(abs(dv)/(case.omega*case.speed)),
                # Either deleted contribution leaves this finite error at the same point.
                missing_flutter_or_ExB_velocity_normalized=float(abs(np.sin(phase[0]))))


def run(output):
    import jax
    import jax.numpy as jnp
    from essos.dynamics import Particles, Tracing

    case = load_case()
    output.mkdir(parents=True, exist_ok=True)
    packet, arrays, spectrum = compute_mode(case, output)
    field, wave = reconstructed_wave(case, packet)
    points, velocity = markers(case)
    particles = Particles(initial_xyz=jnp.asarray(points), initial_vparallel_over_v=jnp.asarray(velocity/case.speed),
                          mass=case.mass, charge=case.charge, energy=case.energy)
    velocity = np.asarray(particles.initial_vparallel)
    times = np.linspace(0, case.duration, 257)
    exact = exact_orbit(case, points, velocity, times)
    exits = exact_first_exit(case, points, velocity)
    expected_hits = np.isfinite(exits)
    arrays.update(times=times, initial_points=points, initial_vparallel=velocity, exact_x=exact[:, :, 0],
                  exact_exit_times=exits)
    results, tight = {}, None
    for name, perturbation, tolerance, stopped in (
        ("baseline", None, 1e-10, True), ("loose", wave, 1e-6, False),
        ("tight", wave, 1e-10, False), ("absorbing", wave, 1e-10, True)):
        trace = Tracing(field=field, particles=particles, perturbation=perturbation,
            model="GuidingCenterAdaptative", maxtime=case.duration, timestep=case.duration/256,
            times_to_trace=len(times), rtol=tolerance, atol=tolerance, max_steps=4096,
            stopping_criteria=(lambda t, y, args, **kwargs: case.wall_x-y[0]) if stopped else None,
            devices=jax.devices()[:1], progress=False)
        state = np.asarray(trace.trajectories)
        if not np.isfinite(state).all():
            raise RuntimeError(f"{name}: incomplete or nonfinite trajectories; cannot classify as losses")
        arrays[f"{name}_x"] = state[:, :, 0]
        metrics = dict(rtol=tolerance, atol=tolerance, finite_orbits=len(state))
        if stopped:
            hits = np.asarray(trace.boundary_hits)
            arrays[f"{name}_hits"] = hits
            metrics.update(hits=int(hits.sum()), loss_fraction=float(hits.mean()))
        else:
            metrics.update(max_radial_error_over_excursion=float(np.max(abs(state[:, :, 0]-exact[:, :, 0]))/case.excursion),
                max_transverse_y_error_m=float(np.max(abs(state[:, :, 1]-exact[:, :, 1]))),
                max_parallel_z_error_m=float(np.max(abs(state[:, :, 2]-exact[:, :, 2]))),
                max_vparallel_error_over_speed=float(np.max(abs(state[:, :, 3]-velocity[:, None]))/case.speed),
                max_relative_mu_drift=float(np.max(abs(state[:, :, 4]/state[:, :1, 4]-1))))
            if name == "tight": tight = state
        if name == "baseline":
            metrics["max_radial_error_m"] = float(np.max(abs(state[:, :, 0]-points[:, 0:1])))
        results[name] = metrics
        # Keep completed stages if the wall-clock cap interrupts a later stage.
        np.savez_compressed(output/"diagnostics.npz", **arrays)
        print(f"Completed {name}: 16 markers", flush=True)

    numerical_exits = sampled_first_exit(times, tight[:, :, 0], case.wall_x)
    coarse_exits = sampled_first_exit(times[::2], tight[:, ::2, 0], case.wall_x)
    arrays["numerical_exit_times"] = numerical_exits
    finite = expected_hits & np.isfinite(numerical_exits)
    exit_error = float(np.max(abs(numerical_exits[finite]-exits[finite]))) if finite.any() else float("inf")
    coarse_error = float(np.max(abs(coarse_exits[finite]-exits[finite]))) if finite.any() else float("inf")
    fields = field_metrics(case, field, wave)
    gyro = abs(case.charge)*case.B0/case.mass
    ordering = dict(omega_over_gyro=case.omega/gyro, max_kperp_rho=case.k_perp*case.speed/gyro,
                    kparallel_over_kperp=case.k_parallel/case.k_perp, delta_b_over_b=case.delta_b_over_b)
    comoving = slice(8, 12)
    comoving_error = float(np.max(abs(tight[comoving, :, 0]-points[comoving, 0:1]))/case.excursion)
    gates = dict(
        fax_spectrum=all(m["max_relative_omega2_error"] < 1e-10 and m["max_eigenpair_residual"] < 1e-10 for m in spectrum),
        fax_constant_mode=all(m["constant_envelope_error"] < 1e-10 for m in spectrum),
        field_reconstruction=all(fields[k] < 1e-10 for k in ("potential_relative_error", "magnetic_relative_error", "electric_relative_error")),
        ideal_parallel_electric_field=fields["parallel_electric_relative_error"] < 1e-10,
        comoving_cancellation=fields["comoving_radial_velocity_normalized"] < 1e-10 and comoving_error < 1e-8,
        deletion_control_sensitive=fields["missing_flutter_or_ExB_velocity_normalized"] > .1,
        gc_ordering=ordering["omega_over_gyro"] < .01 and ordering["max_kperp_rho"] < .05,
        tight_exact_orbit=results["tight"]["max_radial_error_over_excursion"] < 1e-6
            and results["tight"]["max_vparallel_error_over_speed"] < 1e-10
            and results["tight"]["max_transverse_y_error_m"] < 1e-8
            and results["tight"]["max_parallel_z_error_m"] < 1e-6
            and results["tight"]["max_relative_mu_drift"] < 1e-13,
        tolerance_convergence=results["tight"]["max_radial_error_over_excursion"]
            < max(1e-10, .1*results["loose"]["max_radial_error_over_excursion"]),
        baseline_no_loss=results["baseline"]["hits"] == 0 and results["baseline"]["max_radial_error_m"] < 1e-12,
        absorbing_loss_mask=bool(np.array_equal(arrays["absorbing_hits"], expected_hits)),
        sampled_loss_mask=bool(np.array_equal(np.isfinite(numerical_exits), expected_hits)),
        exit_times=exit_error < times[1]-times[0],
        exit_interpolation_convergence=exit_error < coarse_error,
        flat_envelope_domain=bool(exact[:, :, 0].min() > .1 and exact[:, :, 0].max() < .9
            and tight[:, :, 0].min() > .1 and tight[:, :, 0].max() < .9),
    )
    gates = {name: bool(value) for name, value in gates.items()}
    summary = dict(scope="Analytic uniform-plasma verification of FAX, reconstruction and reduced GC; absorbing plane, not toroidal confinement",
        passed=all(gates.values()), gates=gates, input_file=str(INPUT.relative_to(ROOT)),
        parameters=asdict(case),
        equilibrium_sha256=packet.metadata["equilibrium_sha256"], frequency_hz=packet.frequency_hz,
        exact_frequency_hz=case.omega/(2*np.pi), potential_volts=case.potential_volts,
        particles=16, trajectories=64, duration_s=case.duration, platform="cpu", spectrum=spectrum, fields=fields,
        ordering=ordering, cases=results, exact_hits=int(expected_hits.sum()), exact_loss_fraction=float(expected_hits.mean()),
        comoving_radial_error_over_excursion=comoving_error, exit_time_error_s=exit_error,
        coarse_exit_time_error_s=coarse_error, save_interval_s=float(times[1]-times[0]))
    np.savez_compressed(output/"diagnostics.npz", **arrays)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    plot_saved(output)
    if not summary["passed"]:
        raise AssertionError(f"Benchmark gates failed: {[k for k,v in gates.items() if not v]}")
    return summary


def plot_saved(output, data_prefix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = data_prefix
    summary = json.loads((output/"summary.json" if prefix is None else prefix.with_suffix(".json")).read_text())
    with np.load(output/"diagnostics.npz" if prefix is None else prefix.with_suffix(".npz"), allow_pickle=False) as saved:
        data = dict(saved)
    case = SlabCase(**summary["parameters"])
    t = data["times"]*1e6
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for count in (8, 16, 32):
        index = np.arange(count)
        axes[0, 0].plot(index, np.sqrt(data[f"spectrum_exact_{count}"]), "-", label=f"Exact discrete, N={count}")
        axes[0, 0].plot(index, np.sqrt(data[f"spectrum_{count}"]), "x", color=f"C{(8,16,32).index(count)}")
    axes[0, 0].set(xlabel="Radial mode index", ylabel="Frequency / shear-Alfvén frequency",
                   title="FAX spectrum (crosses) vs independent formula")
    # All four phase-pi/2 markers: three moving across the wave and one comoving.
    for index, label in zip((1, 5, 9, 13), ("v∥ = −0.7 vα", "v∥ = 0.15 vα", "v∥ = vA", "v∥ = 0.7 vα")):
        color = f"C{index//4}"
        axes[0, 1].plot(t, data["exact_x"][index], color=color, label=label)
        axes[0, 1].plot(t[::8], data["tight_x"][index, ::8], ".", color=color, markersize=3)
    axes[0, 1].axhline(case.wall_x, color="k", linestyle=":", label="Absorbing plane")
    axes[0, 1].set(xlabel="Time [µs]", ylabel="X [m]", title="Exact orbits (lines), adaptive GC (dots)")
    for name, label in (("loose", "rtol = atol = 10⁻⁶"), ("tight", "rtol = atol = 10⁻¹⁰")):
        error = np.max(abs(data[f"{name}_x"]-data["exact_x"]), axis=0)
        axes[1, 0].semilogy(t, np.maximum(error, 1e-15), label=label)
    axes[1, 0].axhline(1e-6*case.excursion, color="k", linestyle=":", label="Tight acceptance limit")
    axes[1, 0].set(xlabel="Time [µs]", ylabel="Max |X − Xexact| [m]", title="Absolute orbit error and tolerance convergence")
    for name, style, label in (("exact_exit_times", "-", "Exact first exits"),
                               ("numerical_exit_times", "--", "ESSOS, interpolated saved trajectories")):
        exits = np.sort(data[name][np.isfinite(data[name])])*1e6
        axes[1, 1].step(np.r_[0, exits, t[-1]], np.r_[0, np.arange(1, len(exits)+1)*100/16, len(exits)*100/16],
                         where="post", linestyle=style, label=label)
    axes[1, 1].plot(t[-1], summary["cases"]["absorbing"]["loss_fraction"]*100, "o", fillstyle="none",
                     markersize=9, label="ESSOS absorbing-event result")
    axes[1, 1].axhline(0, color="C2", linestyle=":", label="Unperturbed: 0/16")
    axes[1, 1].set(xlabel="Time [µs]", ylabel="Equal-weight first-exit fraction [%]", ylim=(-2, 65),
                   title=f"Specified cohort: exact = ESSOS plane exits = {summary['exact_hits']}/16")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("FAX → mode reconstruction → unchanged ESSOS adaptive GC\n"
                 f"Uniform-plasma analytic verification; {summary['frequency_hz']/1e3:.3f} kHz; δB/B = 10⁻³")
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output/f"fax_shear_alfven.{extension}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--data", type=Path, help="Saved .json/.npz prefix for --report-only")
    parser.add_argument("--output", type=Path, default=ROOT/"examples/benchmarks/results/fax_shear_alfven")
    args = parser.parse_args()
    if args.report_only:
        plot_saved(args.output, args.data)
        return
    if args.data is not None:
        parser.error("--data requires --report-only")
    if (args.output/"mode.npz").exists() or (args.output/"summary.json").exists():
        parser.error("Use a fresh --output to preserve existing results")
    os.environ.update(JAX_PLATFORMS="cpu", ESSOS_CPU_DEVICES="1", CUDA_VISIBLE_DEVICES="-1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.perf_counter()

    def stop():
        print("Stopped at 120-second CPU cap; completed artifacts remain saved", flush=True)
        os._exit(124)

    deadline = threading.Timer(120, stop)
    deadline.daemon = True
    deadline.start()
    try:
        summary = run(args.output)
        summary["elapsed_seconds"] = time.perf_counter()-started
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        deadline.cancel()


if __name__ == "__main__":
    main()
