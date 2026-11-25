import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest
import jax.numpy as jnp

import essos.objective_functions as objf
from essos.constants import ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from essos.dynamics import Particles, Tracing
from essos.fields import Vmec, VMEC_WOUT_ARRAYS


class DummyCoils:
    def __init__(self):
        self.length = jnp.array([3.0, 4.0])
        self.curvature = jnp.ones((2, 5))

    def copy(self):
        return DummyCoils()


@jax.tree_util.register_pytree_node_class
class DummyField:
    def __init__(self, R0=None, Z0=None, phi=None, B_axis=None, grad_B_axis=None,
                 iota=0.4, elongation=None, r_axis=1.0, z_axis=0.0, coils=None):
        self.R0 = jnp.array([1.0, 1.0]) if R0 is None else R0
        self.Z0 = jnp.array([0.0, 0.0]) if Z0 is None else Z0
        self.phi = jnp.array([0.0, jnp.pi / 2]) if phi is None else phi
        # B_axis matches pyqsc_jax external module shape (3, n); function applies .T
        self.B_axis = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]) if B_axis is None else B_axis
        self.grad_B_axis = jnp.ones((3, 3, 2)) if grad_B_axis is None else grad_B_axis
        self.iota = iota
        self.elongation = jnp.array([1.0, 1.2]) if elongation is None else elongation
        self.r_axis = r_axis
        self.z_axis = z_axis
        self.coils = DummyCoils() if coils is None else coils

    def AbsB(self, points):
        return 5.7 + 0.1 * jnp.sum(points)

    def B(self, points):
        return jnp.array([1.0 + 0.1 * points[0], 0.5, 0.25])

    def dB_by_dX(self, points):
        return jnp.eye(3)

    def B_covariant(self, points):
        return jnp.array([1.0, 0.5, 0.25])

    def copy(self):
        return DummyField()

    def tree_flatten(self):
        children = (self.R0, self.Z0, self.phi, self.B_axis, self.grad_B_axis,
                    self.iota, self.elongation, self.r_axis, self.z_axis)
        return children, {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        (R0, Z0, phi, B_axis, grad_B_axis, iota, elongation, r_axis, z_axis) = children
        return cls(R0=R0, Z0=Z0, phi=phi, B_axis=B_axis, grad_B_axis=grad_B_axis,
                   iota=iota, elongation=elongation, r_axis=r_axis, z_axis=z_axis)


class DummyParticles:
    def __init__(self):
        self.energy = 1.0
        self.mass = 1.0
        self.charge = 1.0
        self.total_speed = 1.0

    def to_full_orbit(self, field):
        return None


class DummyTracing:
    def __init__(self, *args, **kwargs):
        xyz = jnp.array(
            [
                [[1.0, 0.0, 0.0], [1.1, 0.0, 0.05], [1.2, 0.0, 0.1], [1.3, 0.0, 0.15]],
                [[1.0, 0.0, 0.0], [1.05, 0.0, 0.04], [1.1, 0.0, 0.08], [1.15, 0.0, 0.12]],
            ],
            dtype=jnp.float64,
        )
        self.trajectories = xyz
        self.field = kwargs.get("field", DummyField())
        self.loss_fractions = jnp.array([0.1, 0.2, 1.0])
        self.times_to_trace = 4
        self.maxtime = 1e-5


@jax.tree_util.register_pytree_node_class
class DummySurface:
    def __init__(self, gamma=None, unitnormal=None, stellsym=False, nfp=1):
        self.gamma = jnp.zeros((2, 3, 3), dtype=jnp.float64) if gamma is None else gamma
        self.unitnormal = jnp.ones((2, 3, 3), dtype=jnp.float64) if unitnormal is None else unitnormal
        self.stellsym = stellsym
        self.nfp = nfp

    def tree_flatten(self):
        return (self.gamma, self.unitnormal), (self.stellsym, self.nfp)

    @classmethod
    def tree_unflatten(cls, aux, children):
        gamma, unitnormal = children
        stellsym, nfp = aux
        return cls(gamma=gamma, unitnormal=unitnormal, stellsym=stellsym, nfp=nfp)


@jax.tree_util.register_pytree_node_class
class PytreeCoils:
    def __init__(self, gamma, gamma_dash, gamma_dashdash, currents, quadpoints, length, curvature, base_curves, order=1, nfp=1, stellsym=False):
        self.gamma = gamma
        self.gamma_dash = gamma_dash
        self.gamma_dashdash = gamma_dashdash
        self.currents = currents
        self.length = length
        self.curvature = curvature
        self.order = order
        self.nfp = nfp
        self.stellsym = stellsym
        self.curves = SimpleNamespace(quadpoints=quadpoints, curves=base_curves)

    def __len__(self):
        return self.gamma.shape[0]

    def copy(self):
        return PytreeCoils(
            self.gamma,
            self.gamma_dash,
            self.gamma_dashdash,
            self.currents,
            self.curves.quadpoints,
            self.length,
            self.curvature,
            self.curves.curves,
            order=self.order,
            nfp=self.nfp,
            stellsym=self.stellsym,
        )

    def tree_flatten(self):
        children = (
            self.gamma,
            self.gamma_dash,
            self.gamma_dashdash,
            self.currents,
            self.curves.quadpoints,
            self.length,
            self.curvature,
            self.curves.curves,
        )
        aux = {"order": self.order, "nfp": self.nfp, "stellsym": self.stellsym}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, **aux)


@jax.tree_util.register_pytree_node_class
class PytreeSurface:
    def __init__(self, gamma, unitnormal, stellsym=False, nfp=1):
        self.gamma = gamma
        self.unitnormal = unitnormal
        self.stellsym = stellsym
        self.nfp = nfp

    def tree_flatten(self):
        return (self.gamma, self.unitnormal), (self.stellsym, self.nfp)

    @classmethod
    def tree_unflatten(cls, aux, children):
        gamma, unitnormal = children
        stellsym, nfp = aux
        return cls(gamma, unitnormal, stellsym=stellsym, nfp=nfp)


class TestObjectiveFunctions(unittest.TestCase):
    def setUp(self):
        self.field = DummyField()
        self.field_nearaxis = DummyField()
        self.particles = DummyParticles()
        self.surface = DummySurface()
        self.sampler = MagicMock(name="sampler")
        self.keys = jnp.array([0, 1], dtype=jnp.int32)
        self.coils = PytreeCoils(
            gamma=jnp.arange(2 * 5 * 3, dtype=jnp.float64).reshape(2, 5, 3) / 10.0,
            gamma_dash=jnp.ones((2, 5, 3), dtype=jnp.float64),
            gamma_dashdash=jnp.ones((2, 5, 3), dtype=jnp.float64) * 0.1,
            currents=jnp.ones(2, dtype=jnp.float64),
            quadpoints=jnp.linspace(0.0, 1.0, 5),
            length=jnp.array([3.0, 4.0], dtype=jnp.float64),
            curvature=jnp.ones((2, 5), dtype=jnp.float64),
            base_curves=jnp.arange(2 * 3 * 3, dtype=jnp.float64).reshape(2, 3, 3) / 10.0,
        )
        self.pytree_surface = PytreeSurface(
            gamma=jnp.arange(2 * 3 * 3, dtype=jnp.float64).reshape(2, 3, 3) / 20.0,
            unitnormal=jnp.ones((2, 3, 3), dtype=jnp.float64),
        )

    def test_near_axis_losses(self):
        points, B_nearaxis, gradB_nearaxis = objf.near_axis_field_quantities(self.field_nearaxis)
        self.assertEqual(points.shape, (3, 2))
        self.assertEqual(B_nearaxis.shape, (2, 3))
        self.assertEqual(gradB_nearaxis.shape, (2, 3, 3))
        self.assertTrue(jnp.isfinite(objf.loss_B_difference_coils_near_axis(self.field, self.field_nearaxis)))
        self.assertTrue(jnp.isfinite(objf.loss_gradB_difference_coils_near_axis(self.field, self.field_nearaxis)))
        self.assertTrue(jnp.isfinite(objf.loss_iota_near_axis(self.field_nearaxis)))
        self.assertTrue(jnp.isfinite(objf.loss_r0_near_axis(self.field_nearaxis)))

    @patch("essos.objective_functions.Tracing", side_effect=DummyTracing)
    def test_particle_losses(self, tracing):
        self.assertTrue(jnp.isfinite(objf.loss_particle_radial_drift(self.field, self.particles)))
        self.assertTrue(jnp.isfinite(objf.loss_particle_radial_drift_fullorbit(self.field, self.particles)))
        self.assertTrue(jnp.isfinite(objf.loss_particle_alpha_drift(self.field, self.particles)))
        self.assertTrue(jnp.isfinite(objf.loss_particle_gammac(self.field, self.particles)))
        self.assertTrue(jnp.isfinite(objf.loss_particle_rcross_final(self.field, self.particles)))
        self.assertTrue(jnp.isfinite(objf.loss_particle_Br(self.field, self.particles)))
        self.assertTrue(jnp.isfinite(objf.loss_particle_iota(self.field, self.particles)))

    @patch("essos.objective_functions.BdotN_over_B", return_value=jnp.ones((2, 3), dtype=jnp.float64))
    def test_surface_losses(self, bdotn):
        self.assertTrue(jnp.isfinite(objf.normB_axis(self.field)).all())
        self.assertTrue(jnp.isfinite(objf.loss_normB_axis_average(self.field)))
        self.assertTrue(jnp.isfinite(objf.loss_BdotN(self.field, self.surface)))
        self.assertTrue(jnp.isfinite(objf.loss_BdotN_constraint(self.field, self.surface)))

    def test_copy_coils_from_field(self):
        copied = objf.copy_coils_from_field(self.field)
        self.assertIsInstance(copied, DummyCoils)

    @patch("essos.objective_functions.BiotSavart", return_value=DummyField())
    @patch("essos.objective_functions.perturb_curves_systematic")
    @patch("essos.objective_functions.perturb_curves_statistic")
    def test_perturbed_field_from_field(self, statistical, systematic, biot_savart):
        systematic.side_effect = lambda coils, sampler, key=None: coils
        statistical.side_effect = lambda coils, sampler, key=None: coils
        field = objf.perturbed_field_from_field(self.field, 0, self.sampler)
        self.assertIsInstance(field, DummyField)
        systematic.assert_called_once()
        statistical.assert_called_once()

    @patch("essos.objective_functions.BdotN_over_B", return_value=jnp.ones((2, 3), dtype=jnp.float64))
    @patch("essos.objective_functions.perturbed_field_from_field", return_value=DummyField())
    def test_stochastic_surface_losses(self, perturbed_field, bdotn):
        self.assertTrue(jnp.isfinite(objf.loss_bdotn_stochastic(self.field, self.surface, self.sampler, self.keys)))
        self.assertTrue(jnp.isfinite(objf.constraint_bdotn_stochastic(self.field, self.surface, self.sampler, self.keys)))

    def test_coil_length_and_curvature_losses(self):
        self.assertTrue(jnp.isfinite(objf.loss_coil_length(self.coils, max_coil_length=4.0)).all())
        self.assertTrue(jnp.isfinite(objf.loss_coil_curvature(self.coils, max_coil_curvature=2.0)).all())

    def test_compute_candidates(self):
        i_vals, j_vals = objf.compute_candidates(self.coils, min_separation=10.0)
        self.assertEqual(i_vals.ndim, 1)
        self.assertEqual(j_vals.ndim, 1)

    def test_blockwise_losses_with_non_divisible_blocks(self):
        separation = objf.loss_coil_separation(self.coils, 0.5, block_size=3)
        surface_distance = objf.loss_coil_surface_distance(self.coils, self.pytree_surface, 0.5, block_size=4)
        linking = objf.loss_linkingnumber(self.coils, block_size=4)
        self.assertTrue(jnp.isfinite(separation))
        self.assertTrue(jnp.isfinite(surface_distance))
        self.assertTrue(jnp.isfinite(linking))
        self.assertAlmostEqual(float(separation), float(objf.loss_coil_separation.__wrapped__(self.coils, 0.5, block_size=3)))
        self.assertAlmostEqual(float(surface_distance), float(objf.loss_coil_surface_distance.__wrapped__(self.coils, self.pytree_surface, 0.5, block_size=4)))
        self.assertAlmostEqual(float(linking), float(objf.loss_linkingnumber.__wrapped__(self.coils, block_size=4)))

    def test_clearance_loss_gradients_match_finite_difference(self):
        def displaced_coils(displacement):
            gamma = self.coils.gamma.at[0, :, 0].add(displacement)
            return PytreeCoils(
                gamma, self.coils.gamma_dash, self.coils.gamma_dashdash,
                self.coils.currents, self.coils.curves.quadpoints,
                self.coils.length, self.coils.curvature,
                self.coils.curves.curves)

        losses = (
            lambda displacement: objf.loss_coil_separation(
                displaced_coils(displacement), 3.0, block_size=3),
            lambda displacement: objf.loss_coil_surface_distance(
                displaced_coils(displacement), self.pytree_surface, 3.0,
                block_size=4),
        )
        step = 1.0e-5
        for loss in losses:
            exact = jax.grad(loss)(0.0)
            finite_difference = (loss(step) - loss(-step)) / (2 * step)
            self.assertTrue(jnp.isfinite(exact))
            self.assertAlmostEqual(float(exact), float(finite_difference), places=6)

    @patch("essos.objective_functions.BiotSavart_from_gamma")
    def test_loss_lorentz_force_coils(self, biot_savart_from_gamma):
        class DummyBS:
            def B(self, point):
                return jnp.zeros(3)

        biot_savart_from_gamma.return_value = DummyBS()
        force_loss = objf.loss_lorentz_force_coils(self.coils, threshold=1e6, block_size=2, conductor_radius=0.05)
        self.assertTrue(jnp.isfinite(force_loss))
        self.assertAlmostEqual(
            float(force_loss),
            float(objf.loss_lorentz_force_coils.__wrapped__(self.coils, threshold=1e6, block_size=2, conductor_radius=0.05)),
        )

    def test_regularization_helpers(self):
        rc_prime = jnp.ones((10, 3))
        rc_prime_prime = jnp.ones((10, 3))
        gamma = jnp.ones((10, 3)) * 4.0
        gammadash = jnp.ones((10, 3))
        gammadashdash = jnp.ones((10, 3))
        quadpoints = jnp.linspace(0, 1, 10)
        self.assertTrue(jnp.isfinite(objf.B_regularized_singularity_term(rc_prime, rc_prime_prime, 1.0)).all())
        self.assertTrue(jnp.isfinite(objf.B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, 1.0, 1.0)).all())
        self.assertTrue(objf.regularization_circ(2.0) > 0)
        self.assertTrue(jnp.isfinite(objf.regularization_rect(2.0, 1.0)))
        self.assertTrue(jnp.isfinite(objf.rectangular_xsection_k(2.0, 1.0)))
        self.assertTrue(jnp.isfinite(objf.rectangular_xsection_delta(2.0, 1.0)))


if __name__ == "__main__":
    unittest.main()


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _Quadrature:
    quadpoints: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _LoopCoils:
    gamma: jnp.ndarray
    gamma_dash: jnp.ndarray
    gamma_dashdash: jnp.ndarray
    currents: jnp.ndarray
    curves: _Quadrature


def _two_distant_loops(n, radius=1.3, current=2.0e6, separation=1.0e5):
    """Two coaxial circular loops so far apart that their mutual force is negligible."""

    theta = jnp.arange(n) / n
    phase = 2 * jnp.pi * theta
    ring = jnp.stack([radius * jnp.cos(phase), radius * jnp.sin(phase), jnp.zeros(n)], axis=1)
    ring_dash = 2 * jnp.pi * jnp.stack([-radius * jnp.sin(phase), radius * jnp.cos(phase), jnp.zeros(n)], axis=1)
    ring_dashdash = -(2 * jnp.pi) ** 2 * ring.at[:, 2].set(0.0)
    offset = jnp.asarray([0.0, 0.0, separation])
    return _LoopCoils(
        gamma=jnp.stack([ring, ring + offset]),
        gamma_dash=jnp.stack([ring_dash, ring_dash]),
        gamma_dashdash=jnp.stack([ring_dashdash, ring_dashdash]),
        currents=jnp.asarray([current, current]),
        curves=_Quadrature(quadpoints=theta),
    ), radius, current


def test_biot_savart_from_gamma_constructs_and_measures_length():
    from essos.fields import BiotSavart_from_gamma

    coils, radius, _ = _two_distant_loops(64)
    field = BiotSavart_from_gamma(coils.gamma, coils.gamma_dash, coils.gamma_dashdash, coils.currents)
    np.testing.assert_allclose(field.coils_length, 2 * np.pi * radius, rtol=1e-12)


@pytest.mark.parametrize("a_over_r", [0.01, 0.1])
def test_lorentz_force_loss_matches_the_analytic_hoop_force(a_over_r):
    coils, radius, current = _two_distant_loops(128)
    a = a_over_r * radius
    mu0 = 4e-7 * np.pi
    hoop = mu0 * current**2 / (4 * np.pi * radius) * (np.log(8 * radius / a) - 0.75)
    loss = objf.loss_lorentz_force_coils(coils, p=1, threshold=0.0, conductor_radius=a)
    # Two identical loops, threshold 0, p = 1: J = 2 * (force per length).
    np.testing.assert_allclose(float(loss), 2 * hoop, rtol=1e-6)


def test_lorentz_force_loss_is_independent_of_quadrature_resolution():
    values = [
        float(objf.loss_lorentz_force_coils(_two_distant_loops(n)[0], p=2, threshold=1e5, conductor_radius=0.05))
        for n in (64, 128, 256)
    ]
    np.testing.assert_allclose(values, values[0], rtol=1e-8)


def test_lorentz_force_loss_requires_a_conductor_radius():
    with pytest.raises(ValueError, match="conductor_radius"):
        objf.loss_lorentz_force_coils(_two_distant_loops(16)[0])


def test_particle_iota_uses_the_poloidal_angle_gradient(monkeypatch):
    import types
    import essos.objective_functions as of
    B = jnp.array([.3, 1.1, .4])
    xyz = jnp.array([[[1.3, .4, .2], [.9, -.5, -.1]]])
    field = types.SimpleNamespace(r_axis=1.0, z_axis=0.0, B_covariant=lambda x: B)
    monkeypatch.setattr(of, "Tracing", lambda **kw: types.SimpleNamespace(trajectories=xyz, field=field))
    theta = lambda x: jnp.arctan2(x[2], jnp.hypot(x[0], x[1]) - 1.0)
    phi = lambda x: jnp.arctan2(x[1], x[0])
    iota = jnp.array([jax.grad(theta)(x) @ B / (jax.grad(phi)(x) @ B) for x in xyz[0]])
    loss = of.loss_particle_iota(field, None, target_iota=10.)
    np.testing.assert_allclose(loss, jnp.sum(10. - iota), rtol=1e-9)


def test_bdotn_constraint_gradient_is_zero_not_nan_when_satisfied(monkeypatch):
    import essos.objective_functions as of
    monkeypatch.setattr(of, "BdotN_over_B", lambda surface, field: field)
    g = jax.grad(lambda b: of.loss_BdotN_constraint(b, None, target_tol=1e-6))(jnp.array([1e-4, 2e-4]))
    np.testing.assert_array_equal(g, 0.)
    g = jax.grad(lambda b: of.loss_BdotN_constraint(b, None, target_tol=1e-6))(jnp.array([1e-2, 2e-4]))
    assert jnp.isfinite(g).all() and g[0] > 0



# Frozen-orbit drift measure.

R0, A, B0 = 5.0, 1.0, 5.0
LOW_RES_QA = "examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
B_TABLES = ("bmnc", "bsubsmns", "bsubumnc", "bsubvmnc", "bsupumnc", "bsupvmnc")


def circular_tokamak(ns=21, modes=8):
    """Vacuum toroidal field B = B0 R0 / R in a circular torus, as live VMEC tables (sqrt(g) < 0, as in a wout)."""
    s = np.linspace(0, 1, ns)
    half = np.r_[0.0, 0.5 * (s[1:] + s[:-1])]
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)

    def cosine_modes(f, grid):
        r = A * np.sqrt(grid)[:, None]
        c = np.fft.rfft(f(R0 + r * np.cos(theta)), axis=1).real[:, :modes + 1] / 32
        c[:, 0] /= 2
        return jnp.asarray(c)

    zero = jnp.zeros((ns, modes + 1))
    root = jnp.asarray(A * np.sqrt(s))
    return Vmec.from_arrays(
        nfp=1, ns=ns, bmnc=cosine_modes(lambda R: B0 * R0 / R, half), xm=jnp.array([0., 1.]), xn=jnp.zeros(2),
        rmnc=jnp.stack([R0 * jnp.ones(ns), root], 1), zmns=jnp.stack([jnp.zeros(ns), root], 1),
        bsubsmns=zero, bsubumnc=zero, bsubvmnc=zero + B0 * R0 * (jnp.arange(modes + 1) == 0),
        bsupumnc=zero, bsupvmnc=cosine_modes(lambda R: B0 * R0 / R**2, half),
        gmnc=cosine_modes(lambda R: -R * A**2 / 2, half), xm_nyq=jnp.arange(modes + 1.), xn_nyq=jnp.zeros(modes + 1),
        Aminor_p=A, ntheta=8, nphi=8)


def alphas(s, theta, phi, pitch):
    return Particles(initial_xyz=jnp.stack([jnp.asarray(s), jnp.asarray(theta), jnp.asarray(phi)], axis=1),
                     initial_vparallel_over_v=jnp.asarray(pitch))


def scaled_field(field, scale):
    """The same VMEC field with |B| and its components multiplied by ``scale``."""
    arrays = {name: getattr(field, name) for name in VMEC_WOUT_ARRAYS}
    return Vmec.from_arrays(nfp=field.nfp, ns=field.ns, **{**arrays, **{n: arrays[n] * scale for n in B_TABLES}})


@pytest.mark.parametrize("theta0", [0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
def test_radial_drift_rate_matches_vertical_grad_b_drift_in_a_toroidal_field(theta0):
    # In B = B0 R0 / R the guiding centre drifts upward at v_d = E (1 + xi^2) / (q B0 R0), independent of R. From z0 = r sin(theta0)
    # at fixed R, s(t) = s0 + (2 z0 v_d t + v_d^2 t^2) / a^2, whose least-squares slope over [0, W] is (2 z0 v_d + v_d^2 W) / a^2.
    pitch, window, s0 = np.array([0.2, 0.5, -0.5, 0.8]), 2e-6, 0.3
    rate = objf.radial_drift_rate(circular_tokamak(), alphas(np.full(4, s0), np.full(4, theta0), np.zeros(4), pitch),
                                  window=window, num_points=41)
    v_d = FUSION_ALPHA_PARTICLE_ENERGY * (1 + pitch**2) / (ALPHA_PARTICLE_CHARGE * B0 * R0)
    expected = (2 * A * np.sqrt(s0) * np.sin(theta0) * v_d + v_d**2 * window) / A**2
    np.testing.assert_allclose(rate, expected, rtol=1e-5)


def test_radial_drift_rate_is_the_least_squares_slope_of_the_traced_s():
    field = Vmec(LOW_RES_QA)
    rng = np.random.default_rng(1)
    particles = alphas(rng.uniform(0.3, 0.7, 12), rng.uniform(0, 2 * np.pi, 12), rng.uniform(0, np.pi, 12), rng.uniform(-1, 1, 12))
    window, num_points = 2e-5, 41
    rate = objf.radial_drift_rate(field, particles, window=window, num_points=num_points)
    traced = Tracing(field=field, model="GuidingCenterAdaptative", particles=particles, maxtime=window, timestep=1e-8,
                     times_to_trace=num_points, atol=1e-6, rtol=1e-6, stopping_criteria=objf.inside_lcfs,
                     max_steps=256, progress=False).trajectories
    kept = np.asarray(traced[..., 0] < 0.95).all(axis=1)
    slope = np.polyfit(np.linspace(0, window, num_points), np.asarray(traced[..., 0]).T, 1)[0]
    # Quadrature of ds/dt against the kernel versus the fit of the saved samples (orbits that stay inside s < 0.95).
    assert kept.sum() >= 8
    assert np.sqrt(np.mean((rate - slope)[kept]**2) / np.mean(slope[kept]**2)) < 0.05


def test_radial_drift_rate_drops_states_near_the_edge_and_rejects_even_samples():
    field = circular_tokamak()
    outside = alphas([0.97, 0.99], [0.0, 1.0], [0.0, 0.0], [0.3, -0.3])
    np.testing.assert_array_equal(objf.radial_drift_rate(field, outside, window=2e-6, num_points=11), 0.0)
    with pytest.raises(ValueError, match="odd"):
        objf.radial_drift_rate(field, outside, num_points=10)


def test_radial_drift_rate_gradient_flows_through_the_field_at_frozen_orbits():
    # With the orbits frozen, scaling every |B| table by c scales the drift by 1/c exactly in the toroidal field (grad-B and
    # curvature forces are unchanged while q B grows), so d rate / dc = -rate at c = 1.
    field = circular_tokamak()
    particles = alphas(np.full(3, 0.3), [0.0, 2.0, 4.0], np.zeros(3), [0.2, 0.5, -0.5])
    options = dict(window=2e-6, num_points=21, orbit_field=field)
    rate = objf.radial_drift_rate(field, particles, **options)
    gradient = jax.jacfwd(lambda c: objf.radial_drift_rate(scaled_field(field, c), particles, **options))(1.0)
    np.testing.assert_allclose(gradient, -rate, rtol=1e-6)


def test_frozen_drift_loss_gradient_matches_finite_differences_of_the_same_measure():
    field = Vmec(LOW_RES_QA)
    rng = np.random.default_rng(2)
    particles = alphas(rng.uniform(0.3, 0.7, 8), rng.uniform(0, 2 * np.pi, 8), rng.uniform(0, np.pi, 8), rng.uniform(-1, 1, 8))
    loss = jax.jit(lambda c: objf.loss_particle_frozen_drift(scaled_field(field, c), particles, window=2e-5, num_points=41,
                                                            orbit_field=field))
    value, gradient = jax.value_and_grad(loss)(1.0)
    step = 1e-4
    difference = (loss(1 + step) - loss(1 - step)) / (2 * step)
    assert value > 0 and abs(gradient) > 1e-3
    np.testing.assert_allclose(gradient, difference, rtol=1e-6)
