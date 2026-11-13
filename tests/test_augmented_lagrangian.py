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
    ALM_model_lbfgs,
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
        self.assertAlmostEqual(float(result), 5.0)  # infinity norm: largest magnitude

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
        self.assertNotIn('slack', params)

    # ---- ALM model tests ----

    def test_ALM_model_lbfgs_init_and_update(self):
        def fun(x): return x - 1
        constraint = eq(fun)
        main_params = jnp.array([6.0,2.0])        
        lagrange_params = constraint.init(main_params)
        params = main_params,lagrange_params            
        alm = ALM_model_lbfgs(constraint)
        self.assertIsInstance(alm, ALM)
        state,grad,info = alm.init(params)
        alm.update(params, state,grad,info)

    def _solve_ineq(self, h, model_lagrangian, iters=30):
        def loss(x):
            return x - 2.0  # main loss ||x - 2||
        constraint = ineq(h, model_lagrangian=model_lagrangian)
        main_params = jnp.array([0.0])
        params = main_params, constraint.init(main_params)
        alm = ALM_model_lbfgs(
            constraint, loss=loss, model_lagrangian=model_lagrangian)
        state, grad, info = alm.init(params)
        for _ in range(iters):
            params, state, grad, info = alm.update(params, state, grad, info)
        return params

    def test_ALM_ineq_active_constraint(self):
        # min |x-2| s.t. 1 - x >= 0  ->  x = 1
        for model in ('Standard', 'Squared'):
            x, lag = self._solve_ineq(lambda x: 1.0 - x, model)
            self.assertAlmostEqual(float(x[0]), 1.0, places=3)
            self.assertGreater(float(lag['lambda'].value[0]), 0.0)

    def test_ALM_ineq_inactive_constraint(self):
        # min |x-2| s.t. 3 - x >= 0  ->  x = 2, multiplier 0
        for model in ('Standard', 'Squared'):
            x, lag = self._solve_ineq(lambda x: 3.0 - x, model)
            self.assertAlmostEqual(float(x[0]), 2.0, places=3)
            lam = float(lag['lambda'].value[0])
            self.assertAlmostEqual(lam, 0.0, places=6)

    def test_ALM_lbfgsb_unbounded_by_default(self):
        # eq constraint x = 150 must not be clipped by a hidden box
        constraint = eq(lambda x: x - 150.0)
        main_params = jnp.array([0.0])
        params = main_params, constraint.init(main_params)
        alm = ALM_model_lbfgs(constraint)
        state, grad, info = alm.init(params)
        for _ in range(10):
            params, state, grad, info = alm.update(params, state, grad, info)
        self.assertAlmostEqual(float(params[0][0]), 150.0, places=3)

    def test_ALM_lbfgs_respects_bounds(self):
        # eq constraint x = 150 inside the box [0, 100] -> x at the upper bound
        constraint = eq(lambda x: x - 150.0)
        params = jnp.array([10.0]), constraint.init(jnp.array([10.0]))
        alm = ALM_model_lbfgs(constraint, bounds=(0.0, 100.0))
        state, grad, info = alm.init(params)
        for _ in range(5):
            params, state, grad, info = alm.update(params, state, grad, info)
        self.assertLessEqual(float(params[0][0]), 100.0)
        self.assertGreater(float(params[0][0]), 99.0)



@pytest.mark.parametrize("model,mu", [("Standard", "Mu_Tolerance"), ("Squared", "Mu_Adaptative_1"),
                                      ("Standard", "Mu_Adaptative_2"), ("Squared", "Mu_Tolerance")])
def test_alm_recovers_minimum_norm_point_and_multiplier(model, mu):
    """min ||x|| (Standard) or ||x||^2 (Squared) on x0 + x1 = 1 gives x = (1/2, 1/2); the multiplier is the
    objective gradient along the constraint normal, lambda = 1/sqrt(2) or 1 (Nocedal & Wright, Thm 12.1)."""
    constraint = eq(lambda x: x[0] + x[1] - 1.0, model_lagrangian=model)
    x0 = jnp.array([3.0, -1.0])
    params = x0, constraint.init(x0)
    alm = ALM_model_lbfgs(constraint, loss=lambda x: x, model_lagrangian=model, model_mu=mu)
    state, grad, info = alm.init(params)
    for _ in range(25):
        params, state, grad, info = alm.update(params, state, grad, info)
    x, lag = params
    assert jnp.allclose(x, 0.5, atol=2e-3)
    lam = float(jnp.ravel(lag["lambda"].value)[0])
    assert abs(lam - (1 / 2 ** 0.5 if model == "Standard" else 1.0)) < 2e-2
    assert abs(float(total_infeasibility(info[2]))) < 2e-3


def test_constraint_containers_named_dependencies_and_errors():
    from essos.augmented_lagrangian import CompositeConstraint, _multiplier_like
    with pytest.raises(ValueError, match="returned None"):
        _multiplier_like(None, 0.0, 1.0, 1.0, 1.0, 0.0)
    with pytest.raises(TypeError):
        SelectiveConstraint(object(), "a")
    with pytest.raises(ValueError):
        SelectiveConstraint(eq(lambda a: a), )
    with pytest.raises(ValueError):
        combine()
    with pytest.raises(TypeError, match="must be SelectiveConstraint"):
        combine(42)

    sa = SelectiveConstraint(eq(lambda a: a - 1.0), "a")
    sb = SelectiveConstraint(ineq(lambda b, shift: b - shift), "b", shift=0.5)
    with pytest.raises(TypeError):
        sa.dependencies = [1.0]
    with pytest.raises(RuntimeError):
        sa.starting_dofs
    with pytest.raises(KeyError, match="depends on 'a'"):
        sa.init()
    combined = combine(sa, sb)
    assert isinstance(combined, CompositeConstraint) and combined.arg_names == ["a", "b"]
    with pytest.raises(TypeError):
        combined.dependencies = 3
    with pytest.raises(RuntimeError):
        combined.starting_dofs
    deps = {"a": jnp.array([2.0, 3.0]), "b": jnp.array([4.0])}
    combined.dependencies = deps

    # stored-dependency calls, explicit-argument calls and flat dof vectors all give the same Lagrangian terms
    params = combined.init()
    assert set(params[0]) == {"lambda"} and jnp.allclose(params[1]["lambda"].penalty, 1.0)
    loss_stored, info = combined.loss(params)
    # eq: sum(-0 * g + g^2 / 2) with g = a - 1 = (1, 2) -> 2.5; ineq: min(b - 0.5, 0/1) = 0 -> 0
    assert jnp.allclose(loss_stored, 2.5) and jnp.allclose(info[0], jnp.array([1.0, 2.0])) and jnp.allclose(info[1], 0.0)
    flat = combined.starting_dofs
    params_flat = combined.init(flat)  # a flat vector of all dofs is split by name (composite fallback)
    assert jnp.allclose(params_flat[0]["lambda"].penalty, 1.0)
    loss_flat, _ = combined.loss(params, flat)
    assert jnp.allclose(loss_flat, loss_stored)
    assert combined.dofs_to_pytree(flat)[0].shape == (2,)
    assert sa.dofs_to_pytree(jnp.array([2.0, 3.0]))[0].shape == (2,)
    explicit_loss, _ = sa.loss(sa.init(deps["a"]), deps["a"])
    assert jnp.allclose(explicit_loss, 2.5)
    # plain tuples are accepted too
    tup = combine((eq(lambda x: x).init, eq(lambda x: x).loss))
    assert len(tup.init(jnp.ones(2))) == 1


def test_penalty_and_infeasibility_norms_on_multiplier_trees():
    tree = {"c": LagrangeMultiplier(jnp.zeros(2), jnp.array([1.0, 3.0]), jnp.ones(2), jnp.ones(2), jnp.zeros(2))}
    assert jnp.allclose(penalty_average(tree), 2.0)
    infos = {"u": jnp.array([3.0, -4.0]), "v": jnp.array([0.0])}
    assert jnp.allclose(norm_constraints(infos), 5.0)
    assert jnp.allclose(infty_norm_constraints(infos), 4.0)
    assert jnp.allclose(total_infeasibility(infos), 7.0)
