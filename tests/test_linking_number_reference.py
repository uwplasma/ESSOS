import numpy as np
import jax
import jax.numpy as jnp

from essos.coils import Coils, CreateEquallySpacedCurves, Curves
from essos.objective_functions import (
    _gauss_linking_integrals_per_pair,
    _linking_numbers_per_pair,
    loss_linkingnumber,
)


def _simsopt_cpp_reference(gamma, gamma_dash, dphi, candidates=None):
    """NumPy translation of SIMSOPT's ``compute_linking_number`` loop."""
    gamma = np.asarray(gamma)
    gamma_dash = np.asarray(gamma_dash)
    if candidates is None:
        candidates = np.triu_indices(len(gamma), k=1)

    integrals = []
    for i, j in zip(*candidates):
        difference = gamma[i, :, None, :] - gamma[j, None, :, :]
        determinant = np.sum(
            difference
            * np.cross(gamma_dash[i, :, None, :], gamma_dash[j, None, :, :]),
            axis=-1,
        )
        distance = np.linalg.norm(difference, axis=-1)
        total = np.sum(determinant / distance**3)
        integrals.append(total * dphi**2 / (4 * np.pi))
    return np.asarray(integrals)


def _coils_from_dofs(dofs, n_segments=97, nfp=1, stellsym=False):
    curves = Curves(
        jnp.asarray(dofs, dtype=jnp.float64),
        n_segments=n_segments,
        nfp=nfp,
        stellsym=stellsym,
    )
    return Coils(curves, jnp.ones(len(dofs), dtype=jnp.float64))


def _hopf_link_dofs(reverse_second=False, swap=False):
    # c0(t) = (cos(2*pi*t), sin(2*pi*t), 0)
    # c1(t) = (1 + cos(2*pi*t), 0, sin(2*pi*t))
    dofs = np.zeros((2, 3, 3))
    dofs[0, 0, 2] = 1.0
    dofs[0, 1, 1] = 1.0
    dofs[1, 0, 0] = 1.0
    dofs[1, 0, 2] = 1.0
    dofs[1, 2, 1] = -1.0 if reverse_second else 1.0
    return dofs[::-1].copy() if swap else dofs


def _assert_matches_reference_for_all_pairs(coils, block_size):
    candidates = np.triu_indices(len(coils), k=1)
    dphi = float(coils.curves.quadpoints[1] - coils.curves.quadpoints[0])
    reference_integrals = _simsopt_cpp_reference(
        coils.gamma, coils.gamma_dash, dphi, candidates
    )
    reference_integer = np.rint(np.abs(reference_integrals))

    actual_integrals = np.asarray(
        _gauss_linking_integrals_per_pair(coils, candidates, block_size)
    )
    actual_integer = np.asarray(
        _linking_numbers_per_pair(coils, candidates, block_size)
    )

    np.testing.assert_allclose(actual_integrals, reference_integrals, rtol=2e-13, atol=2e-14)
    np.testing.assert_array_equal(actual_integer, reference_integer)
    assert float(loss_linkingnumber(coils, candidates, block_size)) == float(
        np.sum(reference_integer)
    )
    return actual_integrals, actual_integer


def test_symmetry_expanded_initial_circular_coils_match_reference_per_pair():
    curves = CreateEquallySpacedCurves(
        n_curves=2,
        order=2,
        R=1.5,
        r=0.25,
        n_segments=65,
        nfp=3,
        stellsym=True,
    )
    coils = Coils(curves, jnp.ones(2, dtype=jnp.float64))

    full_integrals, full_integer = _assert_matches_reference_for_all_pairs(coils, None)
    blocked_integrals, blocked_integer = _assert_matches_reference_for_all_pairs(coils, 16)

    # These are 12 distinct, unlinked circular coils. Their raw quadrature
    # residuals need not be bitwise zero, but every classified pair must be.
    assert len(full_integer) == 66
    assert np.max(np.abs(full_integrals)) < 1e-12
    np.testing.assert_array_equal(full_integer, np.zeros(66))
    np.testing.assert_allclose(blocked_integrals, full_integrals, rtol=2e-13, atol=2e-14)
    np.testing.assert_array_equal(blocked_integer, full_integer)
    assert float(loss_linkingnumber(coils)) == 0.0


def test_hopf_link_matches_reference_and_is_orientation_order_invariant():
    coils = _coils_from_dofs(_hopf_link_dofs())
    reversed_coils = _coils_from_dofs(_hopf_link_dofs(reverse_second=True))
    swapped_coils = _coils_from_dofs(_hopf_link_dofs(swap=True))

    raw, integer = _assert_matches_reference_for_all_pairs(coils, None)
    raw_blocked, integer_blocked = _assert_matches_reference_for_all_pairs(coils, 16)
    raw_reversed, integer_reversed = _assert_matches_reference_for_all_pairs(reversed_coils, 16)
    raw_swapped, integer_swapped = _assert_matches_reference_for_all_pairs(swapped_coils, 16)

    np.testing.assert_allclose(raw_blocked, raw, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(raw_reversed, -raw, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(raw_swapped, raw, rtol=2e-13, atol=2e-14)
    np.testing.assert_array_equal(integer, np.ones(1))
    np.testing.assert_array_equal(integer_blocked, integer)
    np.testing.assert_array_equal(integer_reversed, integer)
    np.testing.assert_array_equal(integer_swapped, integer)
    assert float(loss_linkingnumber(coils)) == 1.0


def test_linking_number_gradient_is_exactly_zero():
    dofs = jnp.asarray(_hopf_link_dofs(), dtype=jnp.float64)

    def objective(curve_dofs):
        return loss_linkingnumber(_coils_from_dofs(curve_dofs), block_size=16)

    gradient = jax.grad(objective)(dofs)
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros_like(np.asarray(dofs)))
