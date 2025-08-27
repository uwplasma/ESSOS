import unittest
import pytest
import jax
import jax.numpy as jnp
import optax

from essos.augmented_lagrangian import (
    LagrangeMultiplier,
    update_method,
    update_method_squared,
    eq,
    ineq,
    combine,
    total_infeasibility,
    norm_constraints,
    infty_norm_constraints,
    penalty_average,
    Constraint,
    ALM,
    lagrange_update,
    ALM_model_optax,
    ALM_model_jaxopt_lbfgsb,
    ALM_model_jaxopt_LevenbergMarquardt,
    ALM_model_jaxopt_lbfgs,
    ALM_model_optimistix_LevenbergMarquardt,
)

class TestAugmentedLagrangian(unittest.TestCase):

    def test_lagrange_multiplier(self):
        lm = LagrangeMultiplier(value=1.0, penalty=2.0, sq_grad=3.0)
        self.assertEqual(lm.value, 1.0)
        self.assertEqual(lm.penalty, 2.0)
        self.assertEqual(lm.sq_grad, 3.0)

    def test_update_method_all_modes(self):
        params = LagrangeMultiplier(jnp.array([1.]), jnp.array([2.]), jnp.array([0.]))
        updates = LagrangeMultiplier(jnp.array([0.5]), jnp.array([0.]), jnp.array([0.]))
        for mode in [
            'Constant', 'Mu_Monotonic', 'Mu_Conditional_True', 'Mu_Conditional_False',
            'Mu_Tolerance_True', 'Mu_Tolerance_False', 'Mu_Adaptative'
        ]:
            if 'Tolerance' in mode:
                result, eta, omega = update_method(params, updates, 1.0, 1.0, model_mu=mode)
                self.assertIsInstance(result, LagrangeMultiplier)
                self.assertIsInstance(eta, jnp.ndarray)
                self.assertIsInstance(omega, jnp.ndarray)
            else:
                result = update_method(params, updates, 1.0, 1.0, model_mu=mode)
                self.assertIsInstance(result, LagrangeMultiplier)

    def test_update_method_squared_all_modes(self):
        params = LagrangeMultiplier(jnp.array([1.]), jnp.array([2.]), jnp.array([0.]))
        updates = LagrangeMultiplier(jnp.array([0.5]), jnp.array([0.]), jnp.array([0.]))
        for mode in [
            'Constant', 'Mu_Monotonic', 'Mu_Conditional_True', 'Mu_Conditional_False',
            'Mu_Tolerance_True', 'Mu_Tolerance_False', 'Mu_Adaptative'
        ]:
            if 'Tolerance' in mode:
                result, eta, omega = update_method_squared(params, updates, 1.0, 1.0, model_mu=mode)
                self.assertIsInstance(result, LagrangeMultiplier)
                self.assertIsInstance(eta, jnp.ndarray)
                self.assertIsInstance(omega, jnp.ndarray)
            else:
                result = update_method_squared(params, updates, 1.0, 1.0, model_mu=mode)
                self.assertIsInstance(result, LagrangeMultiplier)

    def test_eq_and_ineq_constraint(self):
        def fun(x): return x - 2
        eq_constraint = eq(fun)
        ineq_constraint = ineq(fun)
        params_eq = eq_constraint.init(jnp.array([3.]))
        params_ineq = ineq_constraint.init(jnp.array([3.]))
        eq_constraint.loss(params_eq, jnp.array([3.]))
        ineq_constraint.loss(params_ineq, jnp.array([3.]))

    def test_eq_and_ineq_constraint_squared(self):
        def fun(x): return x - 2
        eq_constraint = eq(fun, model_lagrangian='Squared')
        ineq_constraint = ineq(fun, model_lagrangian='Squared')
        params_eq = eq_constraint.init(jnp.array([3.]))
        params_ineq = ineq_constraint.init(jnp.array([3.]))
        eq_constraint.loss(params_eq, jnp.array([3.]))
        ineq_constraint.loss(params_ineq, jnp.array([3.]))

    def test_combine_constraints(self):
        def fun1(x): return x - 1
        def fun2(x): return x + 1
        c1 = eq(fun1)
        c2 = eq(fun2)
        combined = combine(c1, c2)
        params = combined.init(jnp.array([2.]))
        combined.loss(params, jnp.array([2.]))

    def test_combine_multiple_constraints(self):
        def fun1(x): return x - 1
        def fun2(x): return x + 1
        def fun3(x): return x * 2
        c1 = eq(fun1)
        c2 = eq(fun2)
        c3 = eq(fun3)
        combined = combine(c1, c2, c3)
        params = combined.init(jnp.array([2.]))
        combined.loss(params, jnp.array([2.]))

    def test_total_infeasibility(self):
        tree = {'a': jnp.array([1.0, -2.0]), 'b': jnp.array([3.0])}
        result = total_infeasibility(tree)
        self.assertAlmostEqual(float(result), 6.0)

    def test_norm_constraints(self):
        tree = {'a': jnp.array([3.0, 4.0])}
        result = norm_constraints(tree)
        self.assertAlmostEqual(float(result), 5.0)

    def test_infty_norm_constraints(self):
        tree = {'a': jnp.array([1.0, -5.0, 3.0])}
        result = infty_norm_constraints(tree)
        self.assertAlmostEqual(float(result), 3.0)

    def test_penalty_average(self):
        tree = {'a': LagrangeMultiplier(jnp.array([1.0]), jnp.array([2.0]), jnp.array([0.0]))}
        result = penalty_average(tree)
        self.assertAlmostEqual(float(result), 2.0)

    def test_constraint_namedtuple(self):
        def fun(x): return x - 1
        c = eq(fun)
        self.assertIsInstance(c, Constraint)
        params = c.init(jnp.array([2.]))
        c.loss(params, jnp.array([2.]))

    def test_alm_namedtuple(self):
        def dummy_init(*args, **kwargs): return None
        def dummy_update(*args, **kwargs): return None
        alm = ALM(dummy_init, dummy_update)
        self.assertIsInstance(alm, ALM)
        self.assertTrue(callable(alm.init))
        self.assertTrue(callable(alm.update))

    def test_lagrange_update_gradient_transformation_and_update(self):
        gt = lagrange_update('Standard')
        self.assertTrue(hasattr(gt, 'init'))
        self.assertTrue(hasattr(gt, 'update'))
        # Call init and update with dummy data
        params = {'x': jnp.array([1.0])}
        lagrange_params = LagrangeMultiplier(jnp.array([0.0]), jnp.array([1.0]), jnp.array([0.0]))
        updates = LagrangeMultiplier(jnp.array([-0.5]), jnp.array([1.0]), jnp.array([1.0]))
        state = gt.init(params)
        # eta, omega, etc. are required by update_fn signature
        eta = {'x': jnp.array([0.0])}
        omega = {'x': jnp.array([0.0])}
        gt.update(lagrange_params, updates, state, eta, omega, params=params)
        gt2 = lagrange_update('Squared')
        state2 = gt2.init(params)
        gt2.update(lagrange_params, updates, state2, eta, omega, params=params)

    def test_eq_constraint_init_kwargs(self):
        def fun(x, y=0): return x + y - 2
        constraint = eq(fun)
        params = constraint.init(jnp.array([3.]), y=1)
        self.assertIn('lambda', params)

    def test_ineq_constraint_init_kwargs(self):
        def fun(x, y=0): return x + y - 2
        constraint = ineq(fun)
        params = constraint.init(jnp.array([3.]), y=1)
        self.assertIn('lambda', params)
        self.assertIn('slack', params)

    # ---- ALM model tests ----

    def test_ALM_model_optax_init_and_update(self):
        optimizer = optax.sgd(1e-3)
        def fun(x): return x - 1
        constraint = eq(fun)
        main_params = jnp.array([6.0,2.0])        
        lagrange_params = constraint.init(main_params)
        params = main_params,lagrange_params            
        alm = ALM_model_optax(optimizer, constraint,model_mu='Mu_Conditional')
        self.assertIsInstance(alm, ALM)
        # Call init and update
        state,grad,info = alm.init(params)
        # Simulate a gradient step
        eta = jnp.array([0.0])
        omega = jnp.array([0.0]) 
        alm.update(params, state,grad,info,eta,omega)

    def test_ALM_model_jaxopt_lbfgsb_init_and_update(self):
        def fun(x): return x - 1
        constraint = eq(fun)
        main_params = jnp.array([6.0,2.0])        
        lagrange_params = constraint.init(main_params)
        params = main_params,lagrange_params            
        alm = ALM_model_jaxopt_lbfgsb(constraint)
        self.assertIsInstance(alm, ALM)
        state,grad,info = alm.init(params)
        eta = jnp.array([0.0])
        omega = jnp.array([0.0])  
        alm.update(params, state,grad,info,eta,omega)


    def test_ALM_model_jaxopt_LevenbergMarquardt_init_and_update(self):
        def fun(x): return x - 1
        constraint = eq(fun)
        main_params = jnp.array([6.0,2.0])        
        lagrange_params = constraint.init(main_params)
        params = main_params,lagrange_params        
        alm = ALM_model_jaxopt_LevenbergMarquardt(constraint)
        self.assertIsInstance(alm, ALM)
        state,grad,info = alm.init(params)
        eta = jnp.array([0.0])
        omega =  jnp.array([0.0])
        alm.update(params, state,grad,info,eta,omega)



    def test_ALM_model_jaxopt_lbfgs_init_and_update(self):
        def fun(x): return x - 1
        constraint = eq(fun)
        main_params = jnp.array([6.0,2.0])        
        lagrange_params = constraint.init(main_params)
        params = main_params,lagrange_params            
        alm = ALM_model_jaxopt_lbfgs(constraint)
        self.assertIsInstance(alm, ALM)
        state,grad,info = alm.init(params)
        eta =  jnp.array([0.0])
        omega =  jnp.array([0.0])
        alm.update(params, state,grad,info,eta,omega)


    def test_ALM_model_optimistix_LevenbergMarquardt_init_and_update(self):
        def fun(x): return x - 1
        constraint = eq(fun)
        main_params = jnp.array([6.0,2.0])        
        lagrange_params = constraint.init(main_params)
        params = main_params,lagrange_params            
        alm = ALM_model_optimistix_LevenbergMarquardt(constraint)
        self.assertIsInstance(alm, ALM)
        state,grad,info = alm.init(params)
        eta = jnp.array([0.0])
        omega =  jnp.array([0.0])
        alm.update(params, state,grad,info,eta,omega)


if __name__ == "__main__":
    pytest.main([__file__])