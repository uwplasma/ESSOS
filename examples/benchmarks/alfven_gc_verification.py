"""Small CPU checks of the electromagnetic GC equations, inspired by ORBIT.

Run ``python -m examples.benchmarks.alfven_gc_verification`` from the checkout.
The CLI has a 120 s wall limit including imports and compilation, and traces
18 trajectories in total. It uses the shared native adaptive Dopri8 driver. This
adapts the conservation tests of Bierwage, White & Matsuyama (2022),
https://arxiv.org/abs/2205.11280, to analytic fields; it does not reproduce
their JT-60U equilibrium or numerical figures. No collisions are involved.
"""

import argparse
import json
import os
from pathlib import Path
import threading
import time


if __name__ == "__main__":
    os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="-1",
                      ESSOS_CPU_DEVICES="1", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    _started = time.perf_counter()

    def _stop():
        print("GC verification stopped at its 120-second CPU limit", flush=True)
        os._exit(124)

    _deadline = threading.Timer(120., _stop)
    _deadline.daemon = True
    _deadline.start()

import jax
import jax.numpy as jnp
import numpy as np

from essos.boozer import BoozerField, guiding_center_rhs, trace_boozer
from essos.constants import ALPHA_PARTICLE_MASS as MASS, ALPHA_PARTICLE_CHARGE as CHARGE, ONE_EV
from essos.perturbations import (AlfvenSpectrum, AnalyticalAlfvenMode,
                                 BoozerPerturbation, poloidal_flux)


SPEED = np.sqrt(2*3.5e6*ONE_EV/MASS)
PHASES = np.arange(4)*np.pi/2
OMEGA = 9e5
N = 2
TOLERANCES = (1e-6, 1e-12)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3]/"artifacts/analytical_alfven/gc_verification"


def equilibrium(*, uniform=False):
    """Analytic axisymmetric fixture, with nonconstant current profiles."""
    s = np.linspace(.005, .995, 32)
    if uniform:
        return BoozerField.from_booz(s, np.array([5.+0*s]), [0], [0],
                                     0*s, 40.+0*s, 0*s, -40., 1, angle_grid=None)
    return BoozerField.from_booz(s, np.stack([5.+0*s, -np.sqrt(s)]), [0, 1], [0, 0],
                                 .4+.1*s, 40.+.8*s, .8*s, -40., 1, angle_grid=None)


def spectrum(relative_phase):
    """Adjacent m, common n and omega: a single toroidally rotating pattern."""
    return AlfvenSpectrum(*[
        AnalyticalAlfvenMode(m, N, OMEGA, 3e4, phase=phase,
                             radial_center=.4, radial_width=.5)
        for m, phase in ((3, .3), (4, .3+relative_phase))])


def energy_momentum(field, wave, y, mu, t):
    s = y[0]**2+y[1]**2
    B = field.modB(s, jnp.arctan2(y[1], y[0]), y[2])
    (_, G, _), _ = field.profiles(s)
    alpha, phi = wave.potentials(field, y[:3], t)
    H = MASS*(.5*y[3]**2+mu*B)+CHARGE*phi
    P = G*(MASS*y[3]/B+CHARGE*alpha)-CHARGE*poloidal_flux(field, s)
    return jnp.stack([H, P])


def phase_space_density(field, wave, y, t):
    """GC volume density in (u,w,zeta,v_parallel), up to a constant factor.

    From gamma_theta=q*psi0*s+k*I, gamma_zeta=-q*psi_P+k*G,
    k=M*v_parallel/B+q*alpha, and ds dtheta=2 du dw, the density is
    2*M*q*psi0/B * [G+iota*I+(M*v/(q*B)+alpha)*(G*I'-I*G')/psi0].
    Only the constant 2*M*q*psi0 is omitted below. No radial factor is
    needed in this regular chart. The density itself is time dependent.
    """
    s = y[0]**2+y[1]**2
    B = field.modB(s, jnp.arctan2(y[1], y[0]), y[2])
    (iota, G, I), (_, Gs, Is) = field.profiles(s)
    alpha = wave.potentials(field, y[:3], t)[0]
    return (G+iota*I+(MASS*y[3]/(CHARGE*B)+alpha)*(G*Is-I*Gs)/field.psi0)/B


@jax.jit
def differential_checks(field, wave, y, mu, t):
    """Normalized invariant/Liouville residuals and a missing-term control.

    The control cancels only the inductive acceleration in this diagnostic;
    the production RHS is always used unchanged for all actual trajectories.
    """
    invariant = lambda y, t: energy_momentum(field, wave, y, mu, t)@jnp.array([1., -OMEGA/N])
    density = lambda y, t: phase_space_density(field, wave, y, t)
    rhs = lambda y: guiding_center_rhs(field, y, mu, MASS, CHARGE, t=t, perturbation=wave)
    gradient = jax.grad(invariant, 0)(y, t)
    rate = gradient@rhs(y)+jax.grad(invariant, 1)(y, t)
    source = jax.grad(density, 1)(y, t)
    divergence_terms = jnp.diag(jax.jacfwd(lambda y: density(y, t)*rhs(y))(y))
    volume_rate = source+jnp.sum(divergence_terms)
    volume_scale = jnp.maximum(abs(source)+jnp.sum(abs(divergence_terms)), 1e-30)
    energy_scale = abs(energy_momentum(field, wave, y, mu, t)[0])*OMEGA
    B = field.modB(y[0]**2+y[1]**2, jnp.arctan2(y[1], y[0]), y[2])
    alpha_t = wave.derivatives(field, y[:3], t)[1][0, 3]
    removed_acceleration = CHARGE*B/MASS*alpha_t
    broken_rate = rate+gradient[3]*removed_acceleration
    broken_volume = volume_rate+jax.grad(density, 0)(y, t)[3]*removed_acceleration
    return jnp.array([rate/energy_scale, volume_rate/volume_scale,
                      broken_rate/energy_scale, broken_volume/volume_scale])


class UniformInduction(BoozerPerturbation):
    """Exact nonideal calibration: uniform alpha(t), Phi=0, constant B0."""

    amplitude: float = .003
    omega: float = 7.5e5
    phase: float = .4

    def potentials(self, field, xyz, t):
        alpha = self.amplitude*jnp.sin(self.omega*t+self.phase)
        return jnp.stack([alpha, jnp.zeros_like(alpha)])

    def exact(self, times, v0, zeta0, *, B=5., G=40.):
        times = np.asarray(times)
        k = CHARGE*B/MASS*self.amplitude
        v = v0-k*(np.sin(self.omega*times+self.phase)-np.sin(self.phase))
        zeta = zeta0+B/G*((v0+k*np.sin(self.phase))*times
                         +k/self.omega*(np.cos(self.omega*times+self.phase)-np.cos(self.phase)))
        return v, zeta


def trace_wave(field, wave, tolerance, *, tmax=4e-5, n_save=65):
    s, theta, zeta, pitch = np.array([[.3, .43], [.2, 1.4], [0., .3], [.65, -.7]])
    mu = SPEED**2*(1-pitch**2)/(2*np.asarray(jax.vmap(field.modB)(s, theta, zeta)))
    trace = trace_boozer(field, s, theta, zeta, pitch, speed=SPEED, mass=MASS, charge=CHARGE,
                         tmax=tmax, timestep=2e-7, n_save=n_save, integrator="essos_dopri8",
                         rtol=tolerance, atol=tolerance, perturbation=wave, progress=False,
                         compact=False, devices=jax.devices()[:1])
    check_trace(trace)
    s, theta, zeta, vpar, _ = np.moveaxis(trace.states, -1, 0)
    ys = jnp.stack([np.sqrt(s)*np.cos(theta), np.sqrt(s)*np.sin(theta), zeta, vpar], axis=-1)
    values = jax.vmap(lambda y, mu: jax.vmap(
        lambda state, t: energy_momentum(field, wave, state, mu, t))(y, trace.times))(ys, mu)
    values = np.asarray(values)
    H, P = values[..., 0], values[..., 1]
    K = H-OMEGA/N*P
    # Normalize by initial physical energy, avoiding a nearly zero K denominator.
    residual = (K-K[:, :1])/abs(H[:, :1])
    return trace, H, P, residual


def trace_induction(tolerance, *, tmax=4e-5, n_save=65):
    wave = UniformInduction()
    trace = trace_boozer(equilibrium(uniform=True), [.3], [.2], [.3], [.65],
                         speed=SPEED, mass=MASS, charge=CHARGE, tmax=tmax,
                         timestep=2e-7, n_save=n_save, integrator="essos_dopri8", rtol=tolerance, atol=tolerance,
                         perturbation=wave, progress=False, compact=False, devices=jax.devices()[:1])
    check_trace(trace)
    exact_v, exact_zeta = wave.exact(trace.times, .65*SPEED, .3)
    return trace, exact_v, exact_zeta


def check_trace(trace):
    if (trace.failed.any() or trace.lost.any() or np.any(trace.axis_stop_times >= 0)
            or not np.isfinite(trace.states).all()):
        raise RuntimeError("Verification requires finite, complete, confined trajectories")


def run(output):
    field = equilibrium()
    # Fixed deterministic off-axis points, both directions of parallel velocity.
    points = jnp.array([[.45, .27, -.3, .7*SPEED], [.3, -.4, .7, -.65*SPEED],
                        [-.5, .2, 1.2, .5*SPEED], [.2, .6, -.8, -.8*SPEED]])
    times = jnp.array([0., 1.3e-6, 4.7e-6, 9e-6])
    mu = .5*SPEED**2/(2*5.)
    arrays, records = {}, []
    for phase in PHASES:
        wave = spectrum(phase)
        local = np.asarray(jax.vmap(lambda y, t: differential_checks(field, wave, y, mu, t))(points, times))
        phase_records = []
        for tolerance in TOLERANCES:
            trace, H, P, residual = trace_wave(field, wave, tolerance)
            phase_records.append(dict(tolerance=tolerance,
                                      max_invariant_error=float(np.max(abs(residual))),
                                      endpoint_invariant_error=float(np.max(abs(residual[:, -1]))),
                                      max_relative_energy_exchange=float(np.max(abs((H-H[:, :1])/H[:, :1])))))
            suffix = f"phase{len(records)}_tol{tolerance:g}"
            arrays.update({f"{suffix}_states": trace.states, f"{suffix}_energy": H,
                           f"{suffix}_momentum": P, f"{suffix}_invariant_error": residual})
        arrays[f"phase{len(records)}_local_checks"] = local
        records.append(dict(relative_phase_rad=float(phase),
                            max_local_invariant_residual=float(np.max(abs(local[:, 0]))),
                            max_liouville_residual=float(np.max(abs(local[:, 1]))),
                            missing_induction_invariant_residual=float(np.max(abs(local[:, 2]))),
                            missing_induction_liouville_residual=float(np.max(abs(local[:, 3]))),
                            traces=phase_records))
        print(f"Completed harmonic phase {phase/np.pi:g} pi", flush=True)
    induction = []
    for tolerance in TOLERANCES:
        trace, exact_v, exact_zeta = trace_induction(tolerance)
        velocity_error = float(np.max(abs(trace.states[0, :, 3]-exact_v))/SPEED)
        zeta_error = float(np.max(abs(trace.states[0, :, 2]-exact_zeta)))
        induction.append(dict(tolerance=tolerance, max_velocity_error_over_birth_speed=velocity_error,
                              max_zeta_error_rad=zeta_error))
        arrays[f"induction_tol{tolerance:g}"] = trace.states[0]
    arrays.update(times=trace.times, induction_exact_v=exact_v, induction_exact_zeta=exact_zeta)
    summary = dict(reference="https://arxiv.org/abs/2205.11280", exact_paper_reproduction=False,
                   description="ORBIT-style conservation checks on an analytic axisymmetric fixture",
                   method="essos_dopri8", platform=jax.devices()[0].platform, trajectories=18,
                   particles_per_wave_case=2, simulated_time_s=float(trace.times[-1]),
                   tolerances=list(TOLERANCES), time_limit_s=120,
                   wave_cases=records, uniform_induction=induction)
    # Acceptance limits are absolute checks, not an assumed monotonic error trend
    # once roundoff dominates. The two tolerances are reported for inspection.
    passed = (all(r["max_local_invariant_residual"] < 1e-12 and r["max_liouville_residual"] < 1e-12
                  and r["missing_induction_invariant_residual"] > 1e-6
                  and r["missing_induction_liouville_residual"] > 1e-10
                  and r["traces"][-1]["max_invariant_error"] < 1e-8 for r in records)
              and induction[-1]["max_velocity_error_over_birth_speed"] < 1e-8
              and induction[-1]["max_zeta_error_rad"] < 1e-7)
    summary["passed"] = bool(passed)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"trajectories.npz", **arrays)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    if not passed:
        raise AssertionError(f"GC verification failed; inspect {output/'summary.json'}")
    return summary, arrays


def plot(output, summary, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    times = arrays["times"]*1e6
    phase_labels = [r"$\Delta\phi=0$", r"$\Delta\phi=\pi/2$",
                    r"$\Delta\phi=\pi$", r"$\Delta\phi=3\pi/2$"]
    for index, record in enumerate(summary["wave_cases"]):
        color = f"C{index}"
        label = phase_labels[index]
        for tolerance, style in zip(summary["tolerances"], ("--", "-")):
            suffix = f"phase{index}_tol{tolerance:g}"
            residual = arrays[f"{suffix}_invariant_error"]
            axes[0, 0].semilogy(times, np.maximum(abs(residual).max(axis=0), 1e-16),
                                 style, color=color, label=label if style == "-" else None)
        H = arrays[f"phase{index}_tol{summary['tolerances'][-1]:g}_energy"]
        P = arrays[f"phase{index}_tol{summary['tolerances'][-1]:g}_momentum"]
        for marker in range(H.shape[0]):
            axes[0, 1].plot(OMEGA/N*(P[marker]-P[marker, 0])/(1e3*ONE_EV),
                              (H[marker]-H[marker, 0])/(1e3*ONE_EV), color=color,
                              label=label if marker == 0 else None)
        local = arrays[f"phase{index}_local_checks"]
        axes[1, 0].scatter([index]*len(local), np.maximum(abs(local[:, 1]), 1e-17),
                           color=color, marker="o", label="GC RHS" if index == 0 else None)
        axes[1, 0].scatter([index]*len(local), abs(local[:, 3]), color=color, marker="x",
                           label="Inductive term removed (diagnostic only)" if index == 0 else None)
    axes[0, 0].set(xlabel="Time [microseconds]", ylabel=r"$|K(t)-K(0)|/|H(0)|$",
                   title="Two harmonics: rotating-frame invariant")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].text(.04, .03, f"Solid: tolerance {summary['tolerances'][-1]:g}; dashed: {summary['tolerances'][0]:g}",
                    transform=axes[0, 0].transAxes, fontsize=8)
    limits = axes[0, 1].get_xlim()
    axes[0, 1].plot(limits, limits, "k:", label="Exact exchange relation")
    axes[0, 1].set(xlabel=r"$(\omega/n)\,\Delta P_\zeta$ [keV]", ylabel=r"$\Delta H$ [keV]",
                   title="Physical energy and momentum exchange")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set(yscale="log", xlabel="Relative harmonic phase", ylabel="Normalized Liouville residual",
                   xticks=range(4), xticklabels=["0", "pi/2", "pi", "3pi/2"],
                   title="Phase-space conservation: four points per phase")
    axes[1, 0].legend(fontsize=7)
    dense_times = np.linspace(arrays["times"][0], arrays["times"][-1], 1001)
    # Dense evaluation of the analytic curve only; no additional trajectories.
    dense_v, _ = UniformInduction().exact(dense_times, .65*SPEED, .3)
    axes[1, 1].plot(dense_times*1e6, dense_v/1e6, "k-", label="Exact")
    tight = arrays[f"induction_tol{summary['tolerances'][-1]:g}"]
    axes[1, 1].plot(times[::2], tight[::2, 3]/1e6, "o", ms=3, mfc="none", label=summary["method"])
    error = summary["uniform_induction"][-1]
    axes[1, 1].text(.04, .04, f"max |dv|/v_birth = {error['max_velocity_error_over_birth_speed']:.2e}\n"
                              f"max |dzeta| = {error['max_zeta_error_rad']:.2e} rad",
                    transform=axes[1, 1].transAxes, fontsize=8)
    axes[1, 1].set(xlabel="Time [microseconds]", ylabel="Parallel velocity [million m/s]",
                   title="Uniform inductive field: exact trajectory")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("GC electromagnetic corrections: 18 CPU trajectories; analytic equilibrium\n"
                 "ORBIT-style verification, not a reproduction of the JT-60U figures", fontsize=12)
    for extension in ("png", "svg"):
        fig.savefig(output/f"gc_verification.{extension}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-only", action="store_true", help="Redraw saved data without tracing")
    args = parser.parse_args()
    if args.report_only:
        summary = json.loads((args.output/"summary.json").read_text())
        with np.load(args.output/"trajectories.npz") as data:
            arrays = dict(data)
    else:
        summary, arrays = run(args.output)
        summary["elapsed_seconds_before_plot"] = time.perf_counter()-_started
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    plot(args.output, summary, arrays)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        _deadline.cancel()
