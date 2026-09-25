"""Check the closed-axis response operator against direct coil-field tracing.

A known curl- and divergence-free perturbation (a uniform vertical field) is
added to the fixed coils. Its closed-axis displacement is computed three ways:

1. nonlinear: trace the perturbed closed field line and difference it;
2. exact linear: the periodic solution of the cylindrical variational equation
   on the directly traced vacuum axis;
3. near-axis: the Frenet block operator on the ideal near-axis curve, with the
   ideal gradient and with the actual coil gradient sampled on that curve.

The pressure forcing is then pushed through the exact traced-axis operator to
separate the operator and axis-mismatch effects from the pressure source.

    python axis_operator_check.py --output results/axis_operator_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from shafranov_shift import frenet_axis_response, pressure_axis_response

from scan import load_reference


def cylindrical_rhs(field, extra):
    """Field-line map d(R,Z)/dphi for coils plus a Cartesian field ``extra(x)``."""

    def rhs(phi, RZ):
        R, Z = RZ[0], RZ[1]
        x = jnp.array([R * jnp.cos(phi), R * jnp.sin(phi), Z])
        B = field.B(x) + extra(x)
        BR = B[0] * jnp.cos(phi) + B[1] * jnp.sin(phi)
        Bphi = -B[0] * jnp.sin(phi) + B[1] * jnp.cos(phi)
        return jnp.array([R * BR / Bphi, R * B[2] / Bphi])

    return rhs


def closed_axis(rhs, start, period, phis):
    f = jax.jit(rhs)
    ode = lambda phi, y: np.asarray(f(phi, jnp.asarray(y)))

    def integrate(y0, dense=False):
        return solve_ivp(ode, (0, period), y0, method="DOP853", rtol=1e-12,
                         atol=1e-13, dense_output=dense)

    solved = root(lambda y: integrate(y).y[:, -1] - y, start, tol=1e-13)
    closure = float(np.linalg.norm(integrate(solved.x).y[:, -1] - solved.x))
    track = integrate(solved.x, dense=True).sol(np.mod(phis, period))
    return track, closure


def periodic_linear_response(rhs, axis_sol, forcing, period, phis):
    """Periodic solution of d(dX)/dphi = J dX + forcing on the traced axis."""
    jac = jax.jit(jax.jacfwd(rhs, argnums=1))

    def ode(phi, y):
        X = axis_sol(phi)
        J = np.asarray(jac(phi, jnp.asarray(X)))
        M = y[:4].reshape(2, 2)
        p = y[4:]
        return np.r_[(J @ M).ravel(), J @ p + forcing(phi, X)]

    y0 = np.r_[np.eye(2).ravel(), 0.0, 0.0]
    out = solve_ivp(ode, (0, period), y0, method="DOP853", rtol=1e-11,
                    atol=1e-14, dense_output=True)
    monodromy = out.y[:4, -1].reshape(2, 2)
    start = np.linalg.solve(np.eye(2) - monodromy, out.y[4:, -1])
    samples = out.sol(np.mod(phis, period))
    M = samples[:4].reshape(2, 2, -1)
    return np.einsum("ijn,j->in", M, start) + samples[4:], monodromy


def lab_displacement(solution, u, v):
    geometry = solution.geometry
    phi = np.asarray(solution.phi)
    normal = np.asarray(geometry.normal_cartesian)
    binormal = np.asarray(geometry.binormal_cartesian)
    tangent = np.asarray(geometry.tangent_cartesian)
    eR = np.stack((np.cos(phi), np.sin(phi), 0 * phi), axis=-1)
    ephi = np.stack((-np.sin(phi), np.cos(phi), 0 * phi), axis=-1)
    xi = u[:, None] * normal + v[:, None] * binormal
    xi = xi - tangent * (np.sum(ephi * xi, 1) / np.sum(ephi * tangent, 1))[:, None]
    return np.stack((np.sum(eR * xi, 1), xi[:, 2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=HERE / "reference" / "vacuum_fitted_reference.json")
    parser.add_argument("--segments", type=int, default=480)
    parser.add_argument("--radius", type=float, default=0.0178115,
                        help="flux radius used for the pressure forcing (m)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec, _, field, solution = load_reference(args.reference, args.segments)
    nfp = int(spec["nfp"])
    period = 2 * np.pi / nfp
    phis = np.asarray(solution.phi)
    B0 = float(spec["B0"])
    zero = lambda x: jnp.zeros(3)
    vertical = lambda x: jnp.array([0.0, 0.0, B0])

    rhs0 = cylindrical_rhs(field, zero)
    start = [float(solution.R0[0]), float(solution.Z0[0])]
    X0, closure0 = closed_axis(rhs0, start, period, phis)
    f0 = jax.jit(rhs0)
    axis_ode = solve_ivp(lambda p, y: np.asarray(f0(p, jnp.asarray(y))), (0, period),
                         X0[:, 0], method="DOP853", rtol=1e-12, atol=1e-13,
                         dense_output=True).sol

    def vertical_forcing(phi, X):
        R = X[0]
        x = np.array([R * np.cos(phi), R * np.sin(phi), X[1]])
        B = np.asarray(field.B(jnp.asarray(x)))
        Bphi = -B[0] * np.sin(phi) + B[1] * np.cos(phi)
        BR = B[0] * np.cos(phi) + B[1] * np.sin(phi)
        dB = np.array([0.0, 0.0, B0])
        dBR, dBphi = 0.0, 0.0
        return np.array([R * (dBR / Bphi - BR * dBphi / Bphi**2),
                         R * (dB[2] / Bphi - B[2] * dBphi / Bphi**2)])

    linear, monodromy = periodic_linear_response(rhs0, axis_ode, vertical_forcing,
                                                 period, phis)
    nonlinear = {}
    for eps in (4e-6, 2e-6, 1e-6):
        X, closure = closed_axis(cylindrical_rhs(field, lambda x, e=eps: e * vertical(x)),
                                 X0[:, 0], period, phis)
        nonlinear[eps] = ((X - X0) / eps, closure)
    richardson = 2 * nonlinear[1e-6][0] - nonlinear[2e-6][0]
    scale = np.max(np.abs(richardson))

    # Near-axis Frenet operator on the ideal curve.
    ideal_curve = np.asarray(solution.geometry.position_cartesian)
    delta = np.tile([0.0, 0.0, B0], (len(phis), 1))
    u, v, _ = frenet_axis_response(solution, delta)
    frenet_ideal = lab_displacement(solution, u, v)
    grad = np.asarray(jax.jit(jax.vmap(field.dB_by_dX))(jnp.asarray(ideal_curve)))
    Bcoil = np.asarray(jax.jit(jax.vmap(field.B))(jnp.asarray(ideal_curve)))
    Bt = np.sum(Bcoil * np.asarray(solution.geometry.tangent_cartesian), 1)
    u, v, _ = frenet_axis_response(solution, delta, gradient=grad, tangent_field=Bt)
    frenet_actual = lab_displacement(solution, u, v)

    # Pressure forcing through the exact traced-axis operator. The near-axis
    # plasma field is sampled at the same geometrical phi on the traced axis.
    pressure = pressure_axis_response(solution, args.radius, spec["p2_star"],
                                      check_fft=False)
    normal = np.asarray(solution.geometry.normal_cartesian)
    binormal = np.asarray(solution.geometry.binormal_cartesian)
    Bp = pressure["source_n"][:, None] * normal + pressure["source_b"][:, None] * binormal
    n = len(phis)
    wave = nfp * np.fft.fftfreq(n, d=1 / n)
    coeff = np.fft.fft(Bp, axis=0) / n

    def pressure_forcing(phi, X):
        dB = np.real(np.exp(1j * wave * phi) @ coeff)
        R = X[0]
        x = np.array([R * np.cos(phi), R * np.sin(phi), X[1]])
        B = np.asarray(field.B(jnp.asarray(x)))
        c, s = np.cos(phi), np.sin(phi)
        BR, Bphi = B[0] * c + B[1] * s, -B[0] * s + B[1] * c
        dBR, dBphi = dB[0] * c + dB[1] * s, -dB[0] * s + dB[1] * c
        return np.array([R * (dBR / Bphi - BR * dBphi / Bphi**2),
                         R * (dB[2] / Bphi - B[2] * dBphi / Bphi**2)])

    pressure_traced, _ = periodic_linear_response(rhs0, axis_ode, pressure_forcing,
                                                  period, phis)
    ideal_pressure = np.stack((pressure["delta_R"], pressure["delta_Z"]))
    u, v, _ = frenet_axis_response(solution, Bp, gradient=grad, tangent_field=Bt)
    actual_pressure = lab_displacement(solution, u, v)

    rel = lambda a, b: float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    report = {
        "scope": "closed-axis operator only; the pressure source is the near-axis formula",
        "traced_axis_closure_m": closure0,
        "traced_axis_offset_from_ideal_m": float(np.max(np.hypot(
            X0[0] - np.asarray(solution.R0), X0[1] - np.asarray(solution.Z0)))),
        "monodromy_trace": float(np.trace(monodromy)),
        "monodromy_iota_lab": float(np.arccos(np.clip(np.trace(monodromy) / 2, -1, 1))
                                    / (2 * np.pi) * nfp),
        "vertical_field_response_max_m_per_T": scale / B0,
        "nonlinear_vs_linear_first_order_rel": {
            str(e): rel(val[0], linear) for e, val in nonlinear.items()},
        "richardson_vs_linear_rel": rel(richardson, linear),
        "frenet_ideal_gradient_vs_traced_rel": rel(frenet_ideal, linear),
        "frenet_actual_gradient_vs_traced_rel": rel(frenet_actual, linear),
        "pressure": {
            "radius_m": args.radius,
            "ideal_delta_R0_m": float(ideal_pressure[0, 0]),
            "frenet_actual_gradient_delta_R0_m": float(actual_pressure[0, 0]),
            "traced_operator_delta_R0_m": float(pressure_traced[0, 0]),
            "traced_vs_ideal_rel": rel(pressure_traced, ideal_pressure),
            "frenet_actual_vs_traced_rel": rel(actual_pressure, pressure_traced),
            "ideal_rms_m": float(np.sqrt(np.mean(np.sum(ideal_pressure**2, 0)))),
            "traced_rms_m": float(np.sqrt(np.mean(np.sum(pressure_traced**2, 0)))),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.output.with_suffix(".npz"), phi=phis, traced_axis=X0,
             vertical_linear=linear, vertical_richardson=richardson,
             vertical_frenet_ideal=frenet_ideal, vertical_frenet_actual=frenet_actual,
             pressure_ideal=ideal_pressure, pressure_traced=pressure_traced,
             pressure_frenet_actual=actual_pressure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
