import numpy as np
import jax
import jax.numpy as jnp
import pytest

simsopt = pytest.importorskip("simsopt")
from simsopt.geo import LinkingNumber

from essos.coils import Coils, CreateEquallySpacedCurves, Curves
from essos.objective_functions import (
    _gauss_linking_integrals_per_pair,
    _linking_numbers_per_pair,
    loss_linkingnumber,
)


def _hopf_link_dofs():
    dofs = np.zeros((2, 3, 3))
    dofs[0, 0, 2] = 1.0
    dofs[0, 1, 1] = 1.0
    dofs[1, 0, 0] = 1.0
    dofs[1, 0, 2] = 1.0
    dofs[1, 2, 1] = 1.0
    return dofs


def _simsopt_cpp_reference(gamma, gamma_dash, dphi, candidates):
    integrals = []
    for i, j in zip(*candidates):
        difference = gamma[i, :, None, :] - gamma[j, None, :, :]
        determinant = np.sum(
            difference
            * np.cross(gamma_dash[i, :, None, :], gamma_dash[j, None, :, :]),
            axis=-1,
        )
        distance = np.linalg.norm(difference, axis=-1)
        integrals.append(
            np.sum(determinant / distance**3) * dphi**2 / (4 * np.pi)
        )
    return np.asarray(integrals)


def _compare_essos_and_simsopt(curves, block_size):
    simsopt_curves = curves.to_simsopt()
    essos_gamma = np.asarray(curves.gamma)
    essos_gamma_dash = np.asarray(curves.gamma_dash)
    simsopt_gamma = np.asarray([curve.gamma() for curve in simsopt_curves])
    simsopt_gamma_dash = np.asarray([curve.gammadash() for curve in simsopt_curves])

    # This first establishes that both packages integrate the same curves,
    # including the ordering and orientation of all symmetry copies.
    np.testing.assert_allclose(essos_gamma, simsopt_gamma, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        essos_gamma_dash, simsopt_gamma_dash, rtol=2e-14, atol=2e-14
    )

    coils = Coils(curves, jnp.ones(curves.n_base_curves, dtype=jnp.float64))
    candidates = np.triu_indices(len(coils), k=1)
    dphi = float(curves.quadpoints[1] - curves.quadpoints[0])
    reference_raw = _simsopt_cpp_reference(
        simsopt_gamma, simsopt_gamma_dash, dphi, candidates
    )
    essos_raw = np.asarray(
        _gauss_linking_integrals_per_pair(coils, candidates, block_size)
    )
    essos_integer = np.asarray(
        _linking_numbers_per_pair(coils, candidates, block_size)
    )

    np.testing.assert_allclose(essos_raw, reference_raw, rtol=2e-13, atol=2e-14)

    simsopt_integer = np.asarray(
        [LinkingNumber([simsopt_curves[i], simsopt_curves[j]]).J() for i, j in zip(*candidates)]
    )
    np.testing.assert_array_equal(essos_integer, simsopt_integer)
    assert float(loss_linkingnumber(coils, candidates, block_size)) == float(
        LinkingNumber(simsopt_curves).J()
    )
    assert float(np.sum(essos_integer)) == float(np.sum(simsopt_integer))
    return coils, simsopt_curves


@pytest.mark.parametrize("block_size", [None, 16])
def test_initial_circular_coils_match_simsopt_pointwise_and_per_pair(block_size):
    curves = CreateEquallySpacedCurves(
        n_curves=2,
        order=2,
        R=1.5,
        r=0.25,
        n_segments=65,
        nfp=3,
        stellsym=True,
    )
    coils, simsopt_curves = _compare_essos_and_simsopt(curves, block_size)
    assert float(loss_linkingnumber(coils, block_size=block_size)) == 0.0
    assert LinkingNumber(simsopt_curves).J() == 0


@pytest.mark.parametrize("block_size", [None, 16])
def test_hopf_link_matches_simsopt_pointwise_and_per_pair(block_size):
    curves = Curves(
        jnp.asarray(_hopf_link_dofs(), dtype=jnp.float64),
        n_segments=97,
        nfp=1,
        stellsym=False,
    )
    coils, simsopt_curves = _compare_essos_and_simsopt(curves, block_size)
    assert float(loss_linkingnumber(coils, block_size=block_size)) == 1.0
    assert LinkingNumber(simsopt_curves).J() == 1


def test_essos_and_simsopt_both_report_zero_derivative():
    dofs = jnp.asarray(_hopf_link_dofs(), dtype=jnp.float64)

    def objective(curve_dofs):
        curves = Curves(curve_dofs, n_segments=65, nfp=1, stellsym=False)
        return loss_linkingnumber(Coils(curves, jnp.ones(2)), block_size=16)

    essos_gradient = np.asarray(jax.grad(objective)(dofs))
    np.testing.assert_array_equal(essos_gradient, np.zeros_like(np.asarray(dofs)))

    simsopt_curves = Curves(dofs, n_segments=65, nfp=1, stellsym=False).to_simsopt()
    simsopt_gradient = np.asarray(LinkingNumber(simsopt_curves).dJ())
    np.testing.assert_array_equal(simsopt_gradient, np.zeros_like(simsopt_gradient))
