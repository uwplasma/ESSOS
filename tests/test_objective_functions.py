import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

import essos.objective_functions as objf
from essos.dynamics import Tracing


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

    def soft_loss_fraction(self, r_max=0.99, width=0.02):
        return Tracing.soft_loss_fraction(self, r_max=r_max, width=width)


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
        soft_lost_fraction = objf.loss_soft_lost_fraction(self.field, self.particles, r_max=1.2, width=0.05)
        self.assertTrue(0.0 <= soft_lost_fraction <= 1.0)

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

    @patch("essos.objective_functions.Curves.compute_curvature", return_value=jnp.ones(5))
    @patch("essos.objective_functions.BiotSavart_from_gamma")
    def test_loss_lorentz_force_coils(self, biot_savart_from_gamma, compute_curvature):
        class DummyBS:
            def B(self, point):
                return jnp.zeros(3)

        biot_savart_from_gamma.return_value = DummyBS()
        force_loss = objf.loss_lorentz_force_coils(self.coils, threshold=1e6, block_size=2)
        self.assertTrue(jnp.isfinite(force_loss))
        self.assertAlmostEqual(
            float(force_loss),
            float(objf.loss_lorentz_force_coils.__wrapped__(self.coils, threshold=1e6, block_size=2)),
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
