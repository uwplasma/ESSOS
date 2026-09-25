"""Closed axis of the coils plus a frozen near-axis plasma current, at finite alpha.

The first-order pyQSC_JAX current on the vacuum surfaces is held fixed and
scaled by alpha; its free-space field is added to the coil field and the closed
field line is traced directly. This isolates the nonlinearity of the vacuum
field-line map at finite displacement from any change of the current itself,
which is a separate (self-consistent) effect not included here.

    python finite_amplitude_trace.py --output results/finite_amplitude_trace.json
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
from pyqsc_jax.near_axis import near_axis

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from axis_operator_check import closed_axis, cylindrical_rhs
from equilibrium_plasma_field import biot_savart
from scan import load_reference
from source_field_check import near_axis_elements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=HERE / "reference" / "vacuum_fitted_reference.json")
    parser.add_argument("--radius", type=float, default=0.0178115)
    parser.add_argument("--nphi", type=int, default=401)
    parser.add_argument("--half-width", type=float, default=0.012)
    parser.add_argument("--grid-points", type=int, default=25)
    parser.add_argument("--alpha-max", type=float, default=0.17497,
                        help="displacement-screen alpha of the scan manifest")
    parser.add_argument("--multiples", nargs="+", type=float,
                        default=[0.25, 0.5, 1, 2, 4, 6, 8, 10, 12])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec, _, field, solution = load_reference(args.reference, 480)
    period = 2 * np.pi / int(spec["nfp"])
    kw = dict(rc=spec["rc"], zs=spec["zs"], nfp=spec["nfp"], etabar=spec["etabar"],
              order="r3", B0=spec["B0"], B2c=spec["B2c"], I2=0.0, p2=spec["p2_star"])
    # Tabulate the frozen plasma field on a local grid around the vacuum axis,
    # at two toroidal source resolutions, and Richardson-combine (error ~ nphi**-2).
    zero = lambda p: jnp.zeros(3)
    X0, _ = closed_axis(cylindrical_rhs(field, zero), [float(solution.R0[0]), 0.0], period,
                        np.array([0.0]))
    nphi_grid = 25  # odd: no Nyquist mode
    grid_phi = period * np.arange(nphi_grid) / nphi_grid
    axis_grid, _ = closed_axis(cylindrical_rhs(field, zero), X0[:, 0], period, grid_phi)
    offsets = np.linspace(-args.half_width, args.half_width, args.grid_points)
    dR, dZ = np.meshgrid(offsets, offsets, indexing="ij")
    R = axis_grid[0][None, None, :] + dR[..., None]
    Z = axis_grid[1][None, None, :] + dZ[..., None]
    points = np.stack((R * np.cos(grid_phi), R * np.sin(grid_phi), Z), -1).reshape(-1, 3)
    tables = []
    resolutions = (args.nphi, 2 * args.nphi - 1)
    for nphi in resolutions:
        pressure_solution = near_axis(nphi=nphi, **kw).solution
        x, JdV = near_axis_elements(pressure_solution, args.radius)
        tables.append(biot_savart(points, x, JdV, chunk=8).reshape(*R.shape, 3))
        print(f"tabulated nphi={nphi}", flush=True)
    factor = (resolutions[1] / resolutions[0]) ** 2
    table = tables[1] + (tables[1] - tables[0]) / (factor - 1)
    richardson_change = float(np.max(np.abs(tables[1] - table)) / np.max(np.abs(table)))
    # Fourier interpolation in phi and bilinear interpolation in the (R, Z) plane.
    coefficients = np.fft.rfft(table, axis=2) / nphi_grid
    waves = int(spec["nfp"]) * np.arange(coefficients.shape[2])

    def plasma_field(point):
        phi = jnp.arctan2(point[1], point[0])
        Rp = jnp.hypot(point[0], point[1])
        phase = jnp.exp(1j * waves * phi)
        weight = jnp.where(waves == 0, 1.0, 2.0)
        plane = jnp.real(jnp.einsum("ijkc,k->ijc", coefficients, weight * phase))
        axis_R = jnp.real(jnp.sum(weight * phase * axis_coefficients[0]))
        axis_Z = jnp.real(jnp.sum(weight * phase * axis_coefficients[1]))
        u = (Rp - axis_R + args.half_width) / (2 * args.half_width) * (args.grid_points - 1)
        v = (point[2] - axis_Z + args.half_width) / (2 * args.half_width) * (args.grid_points - 1)
        return jax.scipy.ndimage.map_coordinates(plane[..., 0], [u, v], order=1), \
            jax.scipy.ndimage.map_coordinates(plane[..., 1], [u, v], order=1), \
            jax.scipy.ndimage.map_coordinates(plane[..., 2], [u, v], order=1)

    axis_coefficients = jnp.asarray(np.fft.rfft(axis_grid, axis=1) / nphi_grid)
    coefficients = jnp.asarray(coefficients)

    def plasma_vector(point):
        return jnp.stack(plasma_field(point))

    rows = []
    start = X0[:, 0]
    for multiple in args.multiples:
        alpha = multiple * args.alpha_max
        rhs = cylindrical_rhs(field, lambda p, a=alpha: a * plasma_vector(p))
        X, closure = closed_axis(rhs, start, period, np.array([0.0]))
        start = X[:, 0]
        rows.append({"multiple": multiple, "alpha": alpha,
                     "delta_R0_m": float(X[0, 0] - X0[0, 0]),
                     "delta_Z0_m": float(X[1, 0] - X0[1, 0]), "closure_m": closure})
        print(rows[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "scope": "frozen first-order near-axis current; nonlinear vacuum field-line map only",
        "radius_m": args.radius, "nphi": list(resolutions),
        "grid_half_width_m": args.half_width, "grid_points": args.grid_points,
        "richardson_relative_change": richardson_change, "vacuum_R0_m": float(X0[0, 0]),
        "points": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
