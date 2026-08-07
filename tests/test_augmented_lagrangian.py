import unittest
import pytest
import jax
import jax.numpy as jnp

from essos.augmented_lagrangian import (
    LagrangeMultiplier,
    eq,
    ineq,
    combine,
    SelectiveConstraint,
    total_infeasibility,
    norm_constraints,
    infty_norm_constraints,
    penalty_average,
    BaseConstraint,
    ALM,
    ALM_model_jaxopt_lbfgsb,
)

class TestAugmentedLagrangian(unittest.TestCase):

    def test_lagrange_multiplier(self):
        lm = LagrangeMultiplier(value=1.0, penalty=2.0, omega=4.0, eta=5.0, sq_grad=3.0)
        self.assertEqual(lm.value, 1.0)
        self.assertEqual(lm.penalty, 2.0)
        self.assertEqual(lm.omega, 4.0)
        self.assertEqual(lm.eta, 5.0)
        self.assertEqual(lm.sq_grad, 3.0)

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

    def test_selective_constraint_dependencies_reset_cached_dofs(self):
        selective = SelectiveConstraint(eq(lambda field: field - 1), 'field')
        selective.dependencies = {'field': jnp.array([1.0, 2.0])}
        first = selective.starting_dofs

        selective.dependencies = {'field': jnp.array([3.0, 4.0, 5.0])}
        second = selective.starting_dofs

        self.assertEqual(first.shape[0], 2)
        self.assertEqual(second.shape[0], 3)
        self.assertTrue(jnp.allclose(second, jnp.array([3.0, 4.0, 5.0])))

    def test_composite_constraint_dependencies_reset_cached_dofs(self):
        c1 = SelectiveConstraint(eq(lambda field: field - 1), 'field')
        c2 = SelectiveConstraint(eq(lambda surface: surface + 1), 'surface')
        combined = combine(c1, c2)
        combined.dependencies = {
            'field': jnp.array([1.0, 2.0]),
            'surface': jnp.array([10.0]),
        }
        first = combined.starting_dofs

        combined.dependencies = {
            'field': jnp.array([3.0]),
            'surface': jnp.array([20.0, 30.0]),
        }
        second = combined.starting_dofs

        self.assertEqual(first.shape[0], 3)
        self.assertEqual(second.shape[0], 3)
        self.assertTrue(jnp.allclose(second, jnp.array([3.0, 20.0, 30.0])))

    def test_composite_constraint_set_dependencies_resets_cached_dofs(self):
        c1 = SelectiveConstraint(eq(lambda field: field - 1), 'field')
        combined = combine(c1)
        combined.set_dependencies({'field': jnp.array([1.0, 2.0])})
        first = combined.starting_dofs

        combined.set_dependencies({'field': jnp.array([7.0])})
        second = combined.starting_dofs

        self.assertEqual(first.shape[0], 2)
        self.assertEqual(second.shape[0], 1)
        self.assertTrue(jnp.allclose(second, jnp.array([7.0])))

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
        tree = {'a': LagrangeMultiplier(value=jnp.array([1.0]), penalty=jnp.array([2.0]), omega=jnp.array([0.0]), eta=jnp.array([0.0]), sq_grad=jnp.array([0.0]))}
        result = penalty_average(tree)
        self.assertAlmostEqual(float(result), 2.0)

    def test_constraint_namedtuple(self):
        def fun(x): return x - 1
        c = eq(fun)
        self.assertIsInstance(c, BaseConstraint)
        params = c.init(jnp.array([2.]))
        c.loss(params, jnp.array([2.]))

    def test_alm_namedtuple(self):
        def dummy_init(*args, **kwargs): return None
        def dummy_update(*args, **kwargs): return None
        alm = ALM(dummy_init, dummy_update)
        self.assertIsInstance(alm, ALM)
        self.assertTrue(callable(alm.init))
        self.assertTrue(callable(alm.update))

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

    def test_ALM_model_jaxopt_lbfgsb_init_and_update(self):
        def fun(x): return x - 1
        constraint = eq(fun)
        main_params = jnp.array([6.0,2.0])        
        lagrange_params = constraint.init(main_params)
        params = main_params,lagrange_params            
        alm = ALM_model_jaxopt_lbfgsb(constraint)
        self.assertIsInstance(alm, ALM)
        state,grad,info = alm.init(params)
        eta = jnp.array(1.0)
        omega = jnp.array(1.0)  
        alm.update(params, state,grad,info,eta,omega)


if __name__ == "__main__":
    pytest.main([__file__])
