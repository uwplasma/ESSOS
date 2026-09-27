import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.coils import Coils, Curves
from essos.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, ONE_EV
from essos.dynamics import Particles, Tracing, _gc_quantities
from essos.fields import BiotSavart, CombinedField, DipoleField, MagneticField


def _direct_dipole_sum(points, positions, moments):
    """Point-dipole field summed pair by pair, independent of DipoleField."""
    points = np.atleast_2d(np.asarray(points, float))
    out = np.zeros_like(points)
    for pos, mom in zip(np.asarray(positions), np.asarray(moments)):
        r = points - pos
        rn = np.linalg.norm(r, axis=1)[:, None]
        out += 1e-7 * (3.0 * np.sum(mom * r, axis=1)[:, None] * r / rn**5 - mom / rn**3)
    return out


@pytest.fixture
def dipoles():
    rng = np.random.default_rng(0)
    phi = rng.uniform(0.1, 0.6, 7)
    positions = np.stack([1.3 * np.cos(phi), 1.3 * np.sin(phi), rng.uniform(-0.1, 0.1, 7)], axis=1)
    moments = rng.normal(size=(7, 3)) * 50.0
    return jnp.asarray(positions), jnp.asarray(moments)


@pytest.fixture
def coil_field():
    dofs = jnp.zeros((1, 3, 3)).at[0, 0, 2].set(1.0).at[0, 1, 1].set(1.0)
    return BiotSavart(Coils(Curves(dofs, n_segments=64, stellsym=False), jnp.array([2.0e5])))


def test_dipole_field_matches_direct_sum_over_symmetric_copies(dipoles):
    positions, moments = dipoles
    field = DipoleField(positions, moments, jnp.ones(len(positions)), stellsym=True, nfp=3)
    assert field.dipole_positions_full.shape == (2 * 3 * len(positions), 3)
    points = jnp.array([[1.0, 0.1, 0.05], [0.9, -0.3, -0.02], [-0.5, 0.8, 0.1]])
    expected = _direct_dipole_sum(points, field.dipole_positions_full, field.dipole_moments_full)
    np.testing.assert_allclose(field.B(points), expected, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(field.B(points[0]), expected[0], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(field.AbsB(points), np.linalg.norm(expected, axis=1), rtol=1e-10)
    assert field.sqrtg(points[0]) == 1.0
    np.testing.assert_array_equal(field.to_xyz(points[0]), points[0])


def test_dipole_field_is_a_pytree_that_keeps_its_arrays(dipoles):
    positions, moments = dipoles
    field = DipoleField(positions, moments, jnp.ones(len(positions)), stellsym=True, nfp=2)
    leaves, treedef = jax.tree_util.tree_flatten(field)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, DipoleField) and rebuilt.nfp == 2 and rebuilt.n_dipoles == len(positions)
    point = jnp.array([1.0, 0.1, 0.05])
    np.testing.assert_allclose(rebuilt.B(point), field.B(point))
    # The field can be a traced argument, so it is not baked in as a constant.
    leaves[4] = 2 * leaves[4]  # dipole_moments_full
    doubled = jax.tree_util.tree_unflatten(treedef, leaves)
    evaluate = jax.jit(lambda f, x: f.B(x))
    np.testing.assert_allclose(evaluate(doubled, point), 2 * field.B(point), rtol=1e-12)
    np.testing.assert_allclose(evaluate(field, point), field.B(point), rtol=1e-12)


def test_interaction_matrix_reproduces_normal_field(dipoles):
    positions, moments = dipoles
    field = DipoleField(positions, moments, jnp.ones(len(positions)), stellsym=True, nfp=2)
    points = jnp.array([[1.0, 0.1, 0.05], [0.9, -0.3, -0.02], [-0.5, 0.8, 0.1]])
    normals = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.0, 0.0, 1.0]])
    G = field.compute_interaction_matrix(points, normals)
    assert G.shape == (3, len(positions))
    np.testing.assert_allclose(G @ jnp.ones(len(positions)), jnp.sum(field.B(points) * normals, axis=1),
                               rtol=1e-10, atol=1e-14)


def test_dipoles_combine_with_coils_and_trace(coil_field, dipoles):
    positions, moments = dipoles
    dipole_field = DipoleField(positions, moments, jnp.ones(len(positions)), stellsym=True, nfp=2)
    field = CombinedField(coil_field, dipole_field)
    point = jnp.array([1.05, 0.02, 0.01])
    np.testing.assert_allclose(jax.jit(field.B)(point), coil_field.B(point) + dipole_field.B(point), rtol=1e-12)

    # CombinedField does not override gc_quantities, so the generic methods run.
    quantities = _gc_quantities(field, point)
    generic = MagneticField.gc_quantities(field, point)
    for got, want in zip(quantities, generic):
        np.testing.assert_allclose(got, want, rtol=1e-12)

    lines = Tracing(field=field, model='FieldLine', initial_conditions=jnp.array([[0.95, 0.0, 0.0]]),
                    maxtime=1.0, timestep=1e-2, times_to_trace=11)
    assert lines.trajectories.shape == (1, 11, 3) and jnp.all(jnp.isfinite(lines.trajectories))

    particles = Particles(initial_xyz=jnp.array([[0.95, 0.0, 0.0]]), initial_vparallel_over_v=jnp.array([0.3]),
                          charge=ELEMENTARY_CHARGE, mass=ELECTRON_MASS, energy=100 * ONE_EV)
    orbit = Tracing(field=field, particles=particles, model='GuidingCenter',
                    maxtime=1e-8, timestep=1e-10, times_to_trace=5)
    assert orbit.trajectories.shape == (1, 5, 4) and jnp.all(jnp.isfinite(orbit.trajectories))
