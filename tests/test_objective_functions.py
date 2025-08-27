import unittest
from unittest.mock import MagicMock, patch
import jax.numpy as jnp

import essos.objective_functions as objf

class DummyField:
    def __init__(self):
        self.R0 = jnp.array([1.])
        self.Z0 = jnp.array([0.])
        self.phi = jnp.array([0.])
        self.B_axis = jnp.array([[1., 0., 0.]])
        self.grad_B_axis = jnp.array([[0., 0., 0.]])
        self.r_axis = 1.0
        self.z_axis = 0.0
        self.AbsB = MagicMock(return_value=5.7)
        self.B = MagicMock(return_value=jnp.array([1., 0., 0.]))
        self.dB_by_dX = MagicMock(return_value=jnp.array([0., 0., 0.]))
        self.B_covariant = MagicMock(return_value=jnp.array([1., 0., 0.]))
        self.coils_length = jnp.array([30.])
        self.coils_curvature = jnp.ones((2, 10))
        self.gamma = jnp.zeros((2, 10, 3))
        self.gamma_dash = jnp.ones((2, 10, 3))
        self.gamma_dashdash = jnp.ones((2, 10, 3))
        self.currents = jnp.ones(2)
        self.quadpoints = jnp.linspace(0, 1, 10)
        self.x = jnp.zeros((10))

class DummyCoils(DummyField):
    def __init__(self):
        super().__init__()

class DummyCurves:
    def __init__(self, *args, **kwargs):
        pass

class DummyParticles:
    def __init__(self):
        self.to_full_orbit = MagicMock()
        self.trajectories = jnp.zeros((2, 10, 3))

class DummyTracing:
    def __init__(self, *args, **kwargs):
        self.trajectories = jnp.zeros((2, 10, 3))
        self.field = DummyField()
        self.loss_fraction = 0.1
        self.times_to_trace = 10
        self.maxtime = 1e-5

class DummyVmec:
    def __init__(self):
        self.surface = MagicMock()

class DummySurface:
    def __init__(self):
        self.gamma = jnp.zeros((10, 3))
        self.unitnormal = jnp.ones((10, 3))

def dummy_sampler(*args, **kwargs):
    return 0

def dummy_new_nearaxis_from_x_and_old_nearaxis(x, field_nearaxis):
    class DummyNearAxis:
        elongation = jnp.array([1.])
        iota = 1.0
        x = jnp.array([1.])
        R0 = jnp.array([1.])
        Z0 = jnp.array([0.])
        phi = jnp.array([0.])
        B_axis = jnp.array([[1., 0., 0.]])
        grad_B_axis = jnp.array([[0., 0., 0.]])
    return DummyNearAxis()

class TestObjectiveFunctions(unittest.TestCase):
    def setUp(self):
        self.x = jnp.ones(12)
        self.dofs_curves = jnp.ones((2, 3))
        self.currents_scale = 1.0
        self.nfp = 1
        self.n_segments = 10
        self.stellsym = True
        self.key = 0
        self.sampler = dummy_sampler
        self.field = DummyField()
        self.coils = DummyCoils()
        self.curves = DummyCurves()
        self.particles = DummyParticles()
        self.tracing = DummyTracing()
        self.vmec = DummyVmec()
        self.surface = DummySurface()

    @patch('essos.objective_functions.Curves', return_value=DummyCurves())
    @patch('essos.objective_functions.Coils', return_value=DummyCoils())
    @patch('essos.objective_functions.BiotSavart', return_value=DummyField())
    @patch('essos.objective_functions.perturb_curves_systematic')
    @patch('essos.objective_functions.perturb_curves_statistic')
    def test_perturbed_field_and_coils_from_dofs(self, pcs, pcss, bs, coils, curves):
        objf.pertubred_field_from_dofs(self.x, self.key, self.sampler, self.dofs_curves, self.currents_scale, self.nfp)
        objf.perturbed_coils_from_dofs(self.x, self.key, self.sampler, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.Curves', return_value=DummyCurves())
    @patch('essos.objective_functions.Coils', return_value=DummyCoils())
    @patch('essos.objective_functions.BiotSavart', return_value=DummyField())
    def test_field_and_coils_from_dofs(self, bs, coils, curves):
        objf.field_from_dofs(self.x, self.dofs_curves, self.currents_scale, self.nfp)
        objf.coils_from_dofs(self.x, self.dofs_curves, self.currents_scale, self.nfp)
        objf.curves_from_dofs(self.x, self.dofs_curves, self.nfp)

    @patch('essos.objective_functions.field_from_dofs', return_value=DummyField())
    def test_loss_coil_length_and_curvature(self, ffd):
        objf.loss_coil_length(self.x, self.dofs_curves, self.currents_scale, self.nfp)
        objf.loss_coil_curvature(self.x, self.dofs_curves, self.currents_scale, self.nfp)
        objf.loss_coil_length_new(self.x, self.dofs_curves, self.currents_scale, self.nfp)
        objf.loss_coil_curvature_new(self.x, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.field_from_dofs', return_value=DummyField())
    def test_loss_normB_axis(self, ffd):
        objf.loss_normB_axis(self.x, self.dofs_curves, self.currents_scale, self.nfp)
        objf.loss_normB_axis_average(self.x, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.field_from_dofs', return_value=DummyField())
    def test_loss_particle_functions(self, ffd):
        with patch('essos.objective_functions.Tracing', return_value=self.tracing):
            objf.loss_particle_radial_drift(self.x, self.particles, self.dofs_curves, self.currents_scale, self.nfp)
            objf.loss_particle_alpha_drift(self.x, self.particles, self.dofs_curves, self.currents_scale, self.nfp)
            objf.loss_particle_gamma_c(self.x, self.particles, self.dofs_curves, self.currents_scale, self.nfp)
            objf.loss_particle_r_cross_final(self.x, self.particles, self.dofs_curves, self.currents_scale, self.nfp)
            objf.loss_particle_r_cross_max_constraint(self.x, self.particles, self.dofs_curves, self.currents_scale, self.nfp)
            objf.loss_Br(self.x, self.particles, self.dofs_curves, self.currents_scale, self.nfp)
            objf.loss_iota(self.x, self.particles, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.field_from_dofs', return_value=DummyField())
    def test_loss_lost_fraction(self, ffd):
        with patch('essos.objective_functions.Tracing', return_value=self.tracing):
            objf.loss_lost_fraction(self.field, self.particles)

    def test_normB_axis(self):
        objf.normB_axis(self.field)

    @patch('essos.objective_functions.field_from_dofs', return_value=DummyField())
    @patch('essos.objective_functions.new_nearaxis_from_x_and_old_nearaxis', side_effect=dummy_new_nearaxis_from_x_and_old_nearaxis)
    def test_loss_coils_for_nearaxis_and_loss_coils_and_nearaxis(self, nna, ffd):
        objf.loss_coils_for_nearaxis(self.x, self.field, self.dofs_curves, self.currents_scale, self.nfp)
        objf.loss_coils_and_nearaxis(jnp.ones(13), self.field, self.dofs_curves, self.currents_scale, self.nfp)

    def test_difference_B_gradB_onaxis(self):
        objf.difference_B_gradB_onaxis(self.field, self.field)

    @patch('essos.objective_functions.Curves', return_value=DummyCurves())
    @patch('essos.objective_functions.Coils', return_value=DummyCoils())
    @patch('essos.objective_functions.BiotSavart', return_value=DummyField())
    @patch('essos.objective_functions.BdotN_over_B', return_value=jnp.ones(10))
    def test_loss_bdotn_over_b(self, bdotn, bs, coils, curves):
        objf.loss_bdotn_over_b(self.x, self.vmec, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.field_from_dofs', return_value=DummyField())
    @patch('essos.objective_functions.BdotN_over_B', return_value=jnp.ones(10))
    def test_loss_BdotN(self, bdotn, ffd):
        objf.loss_BdotN(self.x, self.vmec, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.field_from_dofs', return_value=DummyField())
    @patch('essos.objective_functions.BdotN_over_B', return_value=jnp.ones(10))
    def test_loss_BdotN_only(self, bdotn, ffd):
        objf.loss_BdotN_only(self.x, self.vmec, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.field_from_dofs', return_value=DummyField())
    @patch('essos.objective_functions.BdotN_over_B', return_value=jnp.ones(10))
    def test_loss_BdotN_only_constraint(self, bdotn, ffd):
        objf.loss_BdotN_only_constraint(self.x, self.vmec, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.BdotN_over_B', return_value=jnp.ones(10))
    @patch('essos.objective_functions.pertubred_field_from_dofs', return_value=DummyField())
    def test_loss_BdotN_only_stochastic(self, perturbed, bdotn):
        objf.loss_BdotN_only_stochastic(self.x, self.sampler, 2, self.vmec, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.BdotN_over_B', return_value=jnp.ones(10))
    @patch('essos.objective_functions.pertubred_field_from_dofs', return_value=DummyField())
    def test_loss_BdotN_only_constraint_stochastic(self, perturbed, bdotn):
        objf.loss_BdotN_only_constraint_stochastic(self.x, self.sampler, 2, self.vmec, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.coils_from_dofs', return_value=DummyCoils())
    def test_loss_cs_distance_and_array(self, cfd):
        objf.loss_cs_distance(self.x, self.surface, self.dofs_curves, self.currents_scale, self.nfp)
        objf.loss_cs_distance_array(self.x, self.surface, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.coils_from_dofs', return_value=DummyCoils())
    def test_loss_cc_distance_and_array(self, cfd):
        objf.loss_cc_distance(self.x, self.dofs_curves, self.currents_scale, self.nfp)
        objf.loss_cc_distance_array(self.x, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.coils_from_dofs', return_value=DummyCoils())
    def test_loss_linking_mnumber_and_constraint(self, cfd):
        objf.loss_linking_mnumber(self.x, self.dofs_curves, self.currents_scale, self.nfp)
        objf.loss_linking_mnumber_constarint(self.x, self.dofs_curves, self.currents_scale, self.nfp)

    def test_cc_distance_pure(self):
        gamma1 = jnp.ones((10, 3))*3.
        l1 = jnp.ones((10, 3))
        gamma2 = jnp.ones((10, 3))*4.
        l2 = jnp.ones((10, 3))*6.
        objf.cc_distance_pure(gamma1, l1, gamma2, l2, 1.0)

    def test_cs_distance_pure(self):
        gammac = jnp.ones((10, 3))*7.
        lc = jnp.ones((10, 3))
        gammas = jnp.ones((10, 3))*9.
        ns = jnp.ones((10, 3))*10.
        objf.cs_distance_pure(gammac, lc, gammas, ns, 1.0)

    @patch('essos.objective_functions.coils_from_dofs', return_value=DummyCoils())
    def test_loss_lorentz_force_coils(self, cfd):
        objf.loss_lorentz_force_coils(self.x, self.dofs_curves, self.currents_scale, self.nfp)

    @patch('essos.objective_functions.compute_curvature', return_value=1.0)
    @patch('essos.objective_functions.BiotSavart_from_gamma', return_value=MagicMock(B=MagicMock(return_value=jnp.array([1., 0., 0.]))))
    def test_lp_force_pure(self, bsg, cc):
        gamma = jnp.ones((2, 10, 3))*2.
        gamma_dash = jnp.ones((2, 10, 3))*3.
        gamma_dashdash = jnp.ones((2, 10, 3))
        currents = jnp.ones(2)
        quadpoints = jnp.linspace(0, 1, 10)
        objf.lp_force_pure(0, gamma, gamma_dash, gamma_dashdash, currents, quadpoints, 1, 1e6)

    def test_B_regularized_singularity_term(self):
        rc_prime = jnp.ones((10, 3))
        rc_prime_prime = jnp.ones((10, 3))
        objf.B_regularized_singularity_term(rc_prime, rc_prime_prime, 1.0)

    def test_B_regularized_pure(self):
        gamma = jnp.ones((10, 3))*4.
        gammadash = jnp.ones((10, 3))
        gammadashdash = jnp.ones((10, 3))
        quadpoints = jnp.linspace(0, 1, 10)
        current = 1.0
        regularization = 1.0
        objf.B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current, regularization)

    def test_regularization_circ(self):
        self.assertTrue(objf.regularization_circ(2.0) > 0)

    def test_regularization_rect_and_k_and_delta(self):
        a, b = 2.0, 1.0
        objf.regularization_rect(a, b)
        objf.rectangular_xsection_k(a, b)
        objf.rectangular_xsection_delta(a, b)

    def test_linking_number_pure_and_integrand(self):
        gamma1 = jnp.ones((10, 3))*4.
        lc1 = jnp.ones((10, 3))*2.
        gamma2 = jnp.ones((10, 3))*6.
        lc2 = jnp.ones((10, 3))*5.
        dphi = 0.1
        objf.linking_number_pure(gamma1, lc1, gamma2, lc2, dphi)
        r1 = jnp.zeros(3)
        dr1 = jnp.ones(3)
        r2 = jnp.zeros(3)
        dr2 = jnp.ones(3)
        objf.integrand_linking_number(r1, dr1, r2, dr2, dphi, dphi)

if __name__ == "__main__":
    unittest.main()