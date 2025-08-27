import unittest
import pytest
import jax

# ESSOS/essos/test_augmented_lagrangian.py

import jax.numpy as jnp

from essos.augmented_lagrangian import (
    LagrangeMultiplier,
    update_method,
    eq,
    ineq,
    combine,
    total_infeasibility,
    norm_constraints,
    infty_norm_constraints,
    penalty_average,
    Constraint,
    ALM,
)

class TestAugmentedLagrangian(unittest.TestCase):

    def test_lagrange_multiplier(self):
        lm = LagrangeMultiplier(value=1.0, penalty=2.0, sq_grad=3.0)
        self.assertEqual(lm.value, 1.0)
        self.assertEqual(lm.penalty, 2.0)
        self.assertEqual(lm.sq_grad, 3.0)

    def test_update_method_constant(self):
        params = LagrangeMultiplier(jnp.array([1.]), jnp.array([2.]), jnp.array([0.]))
        updates = LagrangeMultiplier(jnp.array([0.5]), jnp.array([0.]), jnp.array([0.]))
        result = update_method(params, updates, 1.0, 1.0, model_mu='Constant')
        self.assertIsInstance(result, LagrangeMultiplier)
        assert jnp.allclose(result.value, updates.value)

    def test_update_method_mu_monotonic(self):
        params = LagrangeMultiplier(jnp.array([1.]), jnp.array([2.]), jnp.array([0.]))
        updates = LagrangeMultiplier(jnp.array([0.5]), jnp.array([0.]), jnp.array([0.]))
        result = update_method(params, updates, 1.0, 1.0, model_mu='Mu_Monotonic')
        self.assertIsInstance(result, LagrangeMultiplier)

    def test_eq_constraint(self):
        def fun(x): return x - 2
        constraint = eq(fun)
        params = constraint.init(jnp.array([3.]))
        loss, inf = constraint.loss(params, jnp.array([3.]))
        self.assertIsInstance(loss, jnp.ndarray)
        self.assertIsInstance(inf, jnp.ndarray)

    def test_ineq_constraint(self):
        def fun(x): return x - 1
        constraint = ineq(fun)
        params = constraint.init(jnp.array([2.]))
        loss, inf = constraint.loss(params, jnp.array([2.]))
        self.assertIsInstance(loss, jnp.ndarray)
        self.assertIsInstance(inf, jnp.ndarray)

    def test_combine_constraints(self):
        def fun1(x): return x - 1
        def fun2(x): return x + 1
        c1 = eq(fun1)
        c2 = eq(fun2)
        combined = combine(c1, c2)
        params = combined.init(jnp.array([2.]))
        loss, inf = combined.loss(params, jnp.array([2.]))
        self.assertIsInstance(loss, jnp.ndarray)
        self.assertIsInstance(inf, tuple)
        self.assertEqual(len(inf), 2)

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
        self.assertAlmostEqual(float(result), 5.0)

    def test_penalty_average(self):
        tree = {'a': LagrangeMultiplier(jnp.array([1.0]), jnp.array([2.0]), jnp.array([0.0]))}
        result = penalty_average(tree)
        self.assertAlmostEqual(float(result), 2.0)

    def test_constraint_namedtuple(self):
        def fun(x): return x - 1
        c = eq(fun)
        self.assertIsInstance(c, Constraint)
        params = c.init(jnp.array([2.]))
        loss, inf = c.loss(params, jnp.array([2.]))
        self.assertIsInstance(loss, jnp.ndarray)

    def test_alm_namedtuple(self):
        def dummy_init(*args, **kwargs): return None
        def dummy_update(*args, **kwargs): return None
        alm = ALM(dummy_init, dummy_update)
        self.assertIsInstance(alm, ALM)
        self.assertTrue(callable(alm.init))
        self.assertTrue(callable(alm.update))

if __name__ == "__main__":
    pytest.main([__file__])