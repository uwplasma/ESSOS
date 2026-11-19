import os

import jax.numpy as jnp
import pytest

from essos.constants import mu_0
from essos.current_potential import current_potential_operators, normal_offset_dofs, solve_current_potential
from essos.surfaces import SurfaceRZFourier

WOUT = os.path.join(os.path.dirname(__file__), "..", "examples", "input_files", "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")


def _torus(major, minor, n=16):
    return SurfaceRZFourier(rc=jnp.array([major, minor]), zs=jnp.array([0.0, minor]), nfp=1, mpol=1, ntor=0,
                            ntheta=n, nphi=n, close=False)


def test_net_current_on_a_circular_torus_has_analytic_field_and_cost():
    # Poloidal net current G on a torus (major R, minor a) is a toroidal solenoid: B = mu_0 G/(2 pi r) phi-hat
    # inside, tangent to every axisymmetric surface, so B.n = 0; K = G/(2 pi) r_theta/|N| gives
    # f_K = G^2 a / sqrt(R^2 - a^2), since int dtheta/(R + a cos(theta)) = 2 pi/sqrt(R^2 - a^2).
    major, minor, offset, current = 10.0, 1.0, 1.5, 1e8
    plasma = _torus(major, minor)
    winding = _torus(major, minor)
    winding.dofs = normal_offset_dofs(winding, -offset)  # ESSOS normals point inward
    assert jnp.allclose(winding.dofs, _torus(major, minor + offset).dofs)
    A, b, C, d = current_potential_operators(plasma, winding, current, mpol=2, ntor=2)
    assert jnp.abs(b).max() < 1e-9 * mu_0 * current / major
    assert float(jnp.sum(d**2)) == pytest.approx(
        current**2 * (minor + offset) / (major**2 - (minor + offset) ** 2) ** 0.5, rel=1e-12)


def test_solution_matches_regcoil():
    # REGCOIL 0.1.3 (Landreman, Nucl. Fusion 57, 046003), same surfaces and 32 x 32 points per field
    # period, lambda = 1e-14: f_B = 3.9526 T^2 m^2, f_K = 3.82002e16 A^2 (examples/comparisons_regcoil_quadcoil).
    plasma, winding = (SurfaceRZFourier.from_wout_file(WOUT, s=1, ntheta=32, nphi=64, close=False) for _ in range(2))
    winding.dofs = normal_offset_dofs(winding, -jnp.sqrt(plasma.mean_cross_sectional_area() / jnp.pi))
    A, b, C, d = current_potential_operators(plasma, winding, 3.1794057e8, mpol=4, ntor=4)
    c = solve_current_potential(A, b, C, d, 1e-14)
    assert float(jnp.sum((A @ c + b) ** 2)) == pytest.approx(3.9526, rel=1e-2)
    assert float(jnp.sum((C @ c + d) ** 2)) == pytest.approx(3.82002e16, rel=2e-4)
