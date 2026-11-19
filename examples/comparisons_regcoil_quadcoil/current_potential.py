"""Regularized current-potential solve on a winding surface: ESSOS, REGCOIL and QUADCOIL.

All three codes get the same plasma and winding surfaces, net poloidal current, Fourier basis for the
single-valued potential and regularization, and minimize f_B + lambda f_K, where f_B is the integral of
(B.n)^2 over the plasma and f_K the integral of |K|^2 over the winding surface. Nothing is stored: the
script calls the two codes (``pip install regcoil==0.1.3 quadcoil==0.1.1``).

QA reactor-scale case below (lambda = 1e-14, winding surface one minor radius from the plasma), relative to ESSOS:

                 f_B (32 x 32)   f_B (48 x 48)   f_K (32 x 32)   f_K (48 x 48)
    REGCOIL        -6.2e-3         -7.3e-4          8e-5            6e-6
    QUADCOIL       -1e-8           -6e-8            1e-10           6e-10

ESSOS and QUADCOIL use the same full-torus trapezoid grids and agree to rounding. REGCOIL uses a
different quadrature, and its difference shrinks with resolution (f_B itself moves by 0.4% between the
two grids). Points are per field period; the plasma and winding surfaces use the same grid.
Run time: about 15 s on a laptop CPU, QUADCOIL compilation included.
"""
import os
from time import perf_counter as clock

import jax.numpy as jnp
import numpy as np
import regcoil
from netCDF4 import Dataset
from quadcoil import quadcoil

from essos.constants import mu_0
from essos.current_potential import current_potential_operators, normal_offset_dofs, solve_current_potential
from essos.surfaces import SurfaceRZFourier

# Parameters
LAMBDA = 1e-14       # regularization weight of f_K
MPOL = NTOR = 4      # single-valued current potential modes
ntheta = 32
nphi_per_period = 32

wout = os.path.join(os.path.dirname(__file__), "..", "input_files", "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
plasma = SurfaceRZFourier.from_wout_file(wout, s=1, ntheta=ntheta, nphi=2 * nphi_per_period, close=False)
winding = SurfaceRZFourier.from_wout_file(wout, s=1, ntheta=ntheta, nphi=2 * nphi_per_period, close=False)
winding.dofs = normal_offset_dofs(winding, -jnp.sqrt(plasma.mean_cross_sectional_area() / jnp.pi))  # normals point inward
with Dataset(wout) as dataset:  # net poloidal current linking the plasma, from the covariant B_phi at the edge
    bvco = dataset.variables["bvco"][:]
net_current = float(2 * jnp.pi / mu_0 * (1.5 * bvco[-1] - 0.5 * bvco[-2]))

start = clock()
A, b, C, d = current_potential_operators(plasma, winding, net_current, MPOL, NTOR)
coefficients = solve_current_potential(A, b, C, d, LAMBDA)
results = {"ESSOS": (float(jnp.sum((A @ coefficients + b) ** 2)), float(jnp.sum((C @ coefficients + d) ** 2)), clock() - start)}

start = clock()
regcoil_problem = regcoil.Regcoil(
    regcoil.PlasmaSurface.from_wout(wout, ntheta=ntheta, nzeta=nphi_per_period),
    regcoil.CoilSurface(np.asarray(winding.xm, int), np.asarray(winding.xn, int), np.asarray(winding.rc), np.asarray(winding.zs),
                        nfp=winding.nfp, ntheta=ntheta, nzeta=nphi_per_period),
    mpol_potential=MPOL, ntor_potential=NTOR, net_poloidal_current=net_current)
solution = regcoil_problem.solve(lam=LAMBDA)
results["REGCOIL"] = (solution.f_B, solution.f_K, clock() - start)

# QUADCOIL integrates over one field period, hence the factor nfp below
start = clock()
phi_period = jnp.linspace(0, 1 / plasma.nfp, nphi_per_period, endpoint=False)
theta = jnp.linspace(0, 1, ntheta, endpoint=False)
dofs = lambda surface: jnp.concatenate((surface.rc, surface.zs[1:]))
output = quadcoil(
    nfp=plasma.nfp, stellsym=True, mpol=MPOL, ntor=NTOR,
    plasma_dofs=dofs(plasma), plasma_mpol=plasma.mpol, plasma_ntor=plasma.ntor,
    net_poloidal_current_amperes=net_current, net_toroidal_current_amperes=0.0,
    plasma_quadpoints_phi=phi_period, plasma_quadpoints_theta=theta,
    winding_dofs=dofs(winding), winding_mpol=winding.mpol, winding_ntor=winding.ntor,
    winding_quadpoints_phi=jnp.linspace(0, 1, 2 * nphi_per_period, endpoint=False), winding_quadpoints_theta=theta,
    quadpoints_phi=phi_period, quadpoints_theta=theta,
    objective_name=("f_B", "f_K"), objective_weight=jnp.asarray((1.0, LAMBDA)), objective_unit=(1.0, 1.0),
    metric_name=("f_B", "f_K"), value_only=True, solver="slsqp", convex=True,
    solver_options={"atol": 1e-12, "rtol": 1e-12}, maxiter=1000)[0]
results["QUADCOIL"] = (plasma.nfp * float(output["f_B"]["value"]), plasma.nfp * float(output["f_K"]["value"]), clock() - start)

print(f"{'':9s}{'f_B [T^2 m^2]':>15s}{'f_K [A^2]':>13s}{'time [s]':>10s}{'d f_B':>10s}{'d f_K':>10s}")
for name, (f_b, f_k, seconds) in results.items():
    print(f"{name:9s}{f_b:15.6g}{f_k:13.6g}{seconds:10.2f}{f_b / results['ESSOS'][0] - 1:10.1e}{f_k / results['ESSOS'][1] - 1:10.1e}")
