"""Behavioral coverage for the ALM port, with explicit configuration matrices."""
import builtins
import io
import itertools
import pickle
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos import augmented_lagrangian as alm

MODELS = ('Standard', 'Squared')
WEIGHT_MODES = ('element', 'constraint', 'global')


def assert_tree_close(actual, expected):
    leaves, structure = jax.tree_util.tree_flatten(actual)
    reference, other = jax.tree_util.tree_flatten(expected)
    assert structure == other
    for a, b in zip(leaves, reference):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-7)


def checkpoint_problem():
    return alm.combine(alm.ScaledConstraint(alm.eq(lambda x: x - 1), auto_scale=True),
                       alm.ineq(lambda x: 3 - x, use_analytical_penalty=False))


def test_conditional_updates_and_tolerance_schedules():
    for model in MODELS:
        current = jnp.array([.0001, .005, .02, .1])
        previous = jnp.array([.002, .006, .01, .2])
        constraint = alm.eq(lambda x: x, model_lagrangian=model, penalty=2., eta_tol=.001)
        params = constraint.init(current)
        grad = jax.grad(lambda p: constraint.loss(p, current)[0])(params)
        updates = jax.jit(lambda p, g: alm.apply_mu_tolerance_per_constraint(
            p, g, current, previous, model_lagrangian=model, model_mu='Mu_Adaptative_2'))(params, grad)
        np.testing.assert_allclose(updates['lambda'].value, [0, 0, 0, -.2])
        np.testing.assert_allclose(updates['lambda'].penalty, [0, 0, 2, 0])
        updated = jax.tree_util.tree_map(lambda p, delta: p + delta, params, updates)
        np.testing.assert_array_equal(updated['eta_tol'], params['eta_tol'])
        # Multipliers/penalties freeze while inner solve tolerances keep tightening.
        np.testing.assert_array_equal(updated['lambda'].value[:2], params['lambda'].value[:2])
        np.testing.assert_array_equal(updated['lambda'].penalty[:2], params['lambda'].penalty[:2])
        assert jnp.all(updated['lambda'].omega < params['lambda'].omega)
        assert jnp.all(updated['lambda'].eta < params['lambda'].eta)
        # A worsening element in that band wakes up and increases its penalty.
        reactivated = alm.apply_mu_tolerance_per_constraint(
            params, grad, current, previous.at[1].set(.004), model_lagrangian=model,
            model_mu='Mu_Adaptative_2')
        assert reactivated['lambda'].penalty[1] == 2

    constraint = alm.eq(lambda x: x, penalty=2., eta_tol=jnp.array([.1, .001]))
    x = jnp.array([.05, .05])
    params = constraint.init(x)
    grad = jax.grad(lambda p: constraint.loss(p, x)[0])(params)
    updates = alm.apply_mu_tolerance_per_constraint(params, grad, x, x * 2)
    assert updates['lambda'].value[0] == 0
    assert updates['lambda'].value[1] != 0
    assert alm.penalty_average(params) == 2
    global_updates = jax.jit(alm.apply_mu_tolerance_all_constraints)((params,), (grad,), (x,))
    np.testing.assert_array_equal(global_updates[0]['eta_tol'], jnp.zeros(2))
    assert jnp.all(global_updates[0]['lambda'].value != 0)

    for model_mu in ('Mu_Tolerance', 'Mu_Adaptative_1', 'Mu_Adaptative_2'):
        constraint = alm.combine(alm.eq(lambda x: x[:1] - 1, eta_tol=1e-5),
                             alm.ineq(lambda x: x[1:] - 2, eta_tol=1e-4))
        x = jnp.zeros(2)
        params = x, constraint.init(x)
        solver = alm.ALM_model_scipy(constraint, model_mu=model_mu, decrease_tol=.5)
        state, grad, info = solver.init(params)
        for _ in range(3):
            params, state, grad, info = solver.update(params, state, grad, info)
        np.testing.assert_allclose(params[0][0], 1, atol=1e-5)
        assert params[0][1] >= 2 - 1e-4
        np.testing.assert_allclose(params[1][0]['eta_tol'], [1e-5])

    constraint = alm.eq(lambda x: x, penalty=1., eta_tol=1e-4)
    x = jnp.array([1e-6])
    params = constraint.init(x)
    grad = jax.grad(lambda p: constraint.loss(p, x)[0])(params)
    new = jax.tree_util.tree_map(lambda p, delta: p + delta, params,
                                 alm.apply_mu_tolerance_per_constraint(params, grad, x))
    np.testing.assert_allclose(new['lambda'].eta, [.99])
    np.testing.assert_allclose(new['lambda'].omega, [.99])
    np.testing.assert_allclose(new['lambda'].value, [0.])

    constraint = alm.ineq(lambda x: 3 - x)
    x = jnp.zeros(1)
    params = constraint.init(x)
    gradient, residual = jax.grad(lambda p: constraint.loss(p, x), has_aux=True)(params)
    change = alm.apply_mu_tolerance_all_constraints(params, gradient, residual)
    np.testing.assert_allclose(params['lambda'].omega + change['lambda'].omega, [.99])
    np.testing.assert_allclose(params['lambda'].eta + change['lambda'].eta, [.99])

    constraint = alm.eq(lambda x: jnp.exp(x) - 2, eta_tol=1e-10)
    x = jnp.array([1.])
    params = x, constraint.init(x)

    def one_step(threshold):
        solver = alm.ALM_model_scipy(constraint, model_mu='Mu_Adaptative_2',
                                       inner_maxiter=1, decrease_tol=threshold)
        state, grad, info = solver.init(params)
        return solver.update(params, state, grad, info)[0]

    loose = one_step(1.)
    ratio = float(jnp.abs((jnp.exp(loose[0][0]) - 2) / (jnp.exp(x[0]) - 2)))
    assert 0 < ratio < 1
    strict = one_step(ratio / 2)
    np.testing.assert_allclose(strict[0], loose[0])
    np.testing.assert_allclose(loose[1]['lambda'].penalty, [1.])
    np.testing.assert_allclose(strict[1]['lambda'].penalty, [2.])
    defaults = alm.apply_mu_tolerance_all_constraints(params[1])
    np.testing.assert_array_equal(defaults['lambda'].value, 0.)


def test_projected_duals_keep_weights_and_physical_tolerances_fixed():
    for model in MODELS:
        constraint = alm.ineq(lambda x: x, multiplier=1., penalty=2., eta_tol=1e-6,
                          model_lagrangian=model)
        x = jnp.array([.25, 2.])  # feasible inequalities with obsolete multipliers
        params = constraint.init(x)
        grad, residual = jax.grad(lambda p: constraint.loss(p, x), has_aux=True)(params)
        updates = alm.apply_mu_tolerance_per_constraint(params, grad, residual,
            jnp.array([1., 1.]), model_lagrangian=model, model_mu='Mu_Adaptative_2')
        new = jax.tree_util.tree_map(lambda p, delta: p + delta, params, updates)
        np.testing.assert_allclose(new['lambda'].value, [.5, 0.])

    for model, global_update in itertools.product(MODELS, (False, True)):
        constraint = alm.ineq(lambda x: x, multiplier=1., penalty=2., eta_tol=1e-6,
                              model_lagrangian=model)
        x = jnp.array([.1, .25])
        p = {**constraint.init(x), 'weight_g': jnp.full(2, 4.)}
        grad, residual = jax.grad(lambda q: constraint.loss(q, x), has_aux=True)(p)
        np.testing.assert_allclose(grad['weight_g'], 0.)
        np.testing.assert_allclose(residual, [.1, .125])
        if global_update:
            updates = alm.apply_mu_tolerance_all_constraints(p, grad, residual, model_lagrangian=model)
        else:
            updates = alm.apply_mu_tolerance_per_constraint(p, grad, residual, model_lagrangian=model)
        new = jax.tree_util.tree_map(lambda p, delta: p + delta, p, updates)
        np.testing.assert_allclose(new['lambda'].value, [.2, 0.], atol=1e-6)
        np.testing.assert_array_equal(new['weight_g'], p['weight_g'])
        np.testing.assert_array_equal(new['eta_tol'], p['eta_tol'])

    constraint = alm.eq(lambda x: x, eta_tol=1e-4)
    x = jnp.array([1e-5])
    p = {**constraint.init(x), 'weight_g': jnp.array([100.])}
    grad, residual = jax.grad(lambda q: constraint.loss(q, x), has_aux=True)(p)
    np.testing.assert_allclose(residual, x)
    updates = alm.apply_mu_tolerance_per_constraint(p, grad, residual)
    np.testing.assert_array_equal(updates['lambda'].value, jnp.zeros(1))


def test_gradient_weights_use_raw_jacobians_and_finite_norms():
    for mode, batch_size in itertools.product(WEIGHT_MODES, (1, 3)):
        constraints = alm.combine(alm.eq(lambda x: 2 * x[0]), alm.ineq(lambda x: 4 * x[1] + 3))
        x = jnp.zeros(2)
        p = constraints.init(x)
        weights = alm.initialize_constraint_weights(constraints, x, p,
            loss=lambda y: 3 * y[0] + 1, mode=mode, batch_size=batch_size)
        expected = [3 / np.sqrt(20)] * 2 if mode == 'global' else [1.5, .75]
        np.testing.assert_allclose([record['weight_g'] for record in weights], expected, rtol=1e-6)
        # The inequality is inactive in PHR, but its raw gradient is four, not zero.
        assert constraints.loss(p, x)[1][1] == 0
        assert constraints.loss(p, x, _raw=True)[1][1] == 3

    constraint = alm.eq(lambda x: jnp.array([0., 2 * x[0]]))
    x = jnp.zeros(1)
    p = constraint.init(x)
    element = alm.initialize_constraint_weights(constraint, x, p, epsilon=.1)
    block = alm.initialize_constraint_weights(constraint, x, p, mode='constraint', epsilon=.1)
    np.testing.assert_allclose(element['weight_g'], [10., .5])
    np.testing.assert_allclose(block['weight_g'], .5)

    constraint = alm.eq(lambda x: 2 * x)
    x = jnp.zeros(1)
    p = alm.initialize_constraint_weights(constraint, x, constraint.init(x),
                                         loss=lambda y: 1 + 1e-6 * y)
    np.testing.assert_allclose(p['weight_g'], [5e-7], rtol=1e-6)
    zero = alm.initialize_constraint_weights(constraint, x, constraint.init(x), loss=lambda y: y)
    np.testing.assert_allclose(zero['weight_g'], [.5])

    # Source-branch helper APIs retain the same normalization and output trees.
    np.testing.assert_allclose(alm.compute_global_weight_g(constraint, p, x, 3.), 1.5)
    np.testing.assert_allclose(alm.compute_per_constraint_weights_g(constraint, p, x, 3.), [1.5])
    tree = {'x': x}
    values, unpack = alm.compute_element_wise_weights_g(lambda y: 2 * y['x'], 2 * x, tree, 3.)
    np.testing.assert_allclose(values, [1.5])
    np.testing.assert_allclose(unpack(jnp.array([7.])), [7.])
    np.testing.assert_allclose(alm.compute_perturbation_weight_via_jacobian(
        lambda y: 2 * y['x'], 2 * x, tree, 3.), 1.5)

    constraints = alm.combine(
        alm.SelectiveConstraint(alm.eq(lambda field: 2 * field), 'field'),
        alm.SelectiveConstraint(alm.ineq(lambda surface: 3 * surface + 1), 'surface'))
    constraints.dependencies = {'field': jnp.zeros(1), 'surface': jnp.zeros(1)}
    x = constraints.starting_dofs
    p = alm.initialize_constraint_weights(constraints, x, constraints.init(x))
    np.testing.assert_allclose(p[0]['weight_g'], [.5])
    np.testing.assert_allclose(p[1]['weight_g'], [1/3], rtol=1e-6)
    for options in ({'mode': 'invalid'}, {'epsilon': 0}, {'batch_size': 0},
                    {'jacobian_method': 'invalid'}, {'perturbation_mode': 'invalid'}):
        with pytest.raises(ValueError):
            alm.initialize_constraint_weights(constraints, x, constraints.init(x), **options)
    bad = alm.BaseConstraint(lambda x: p, lambda p, x, **kw: (0., (x, x, x)))
    with pytest.raises(ValueError, match='one scalar or array'):
        alm.initialize_constraint_weights(bad, x, p)
    with pytest.raises(ValueError, match='Nonfinite'):
        alm.initialize_constraint_weights(constraint, jnp.zeros(1), constraint.init(jnp.zeros(1)),
                                          loss=lambda y: jnp.nan * y)


def test_seeded_perturbations_and_jacobian_methods():
    p = {'a': jnp.ones(8), 'b': jnp.ones(8), 'zero': jnp.zeros(8), 'integer': 2}
    relative = alm.perturb_params_relative(p, .1, seed=3)
    assert_tree_close(relative, alm.perturb_params_relative(p, .1, seed=3))
    assert not np.allclose(relative['a'], relative['b'])
    np.testing.assert_array_equal(relative['zero'], p['zero'])
    absolute = alm.perturb_params_absolute(p, .1, seed=3)
    assert jnp.linalg.norm(absolute['zero']) > 0
    assert absolute['integer'] == 2
    assert not np.allclose(absolute['a'], alm.perturb_params_absolute(p, .1, seed=4)['a'])

    for mode in WEIGHT_MODES:
        constraint = alm.eq(lambda x: jnp.array([x[0]**2, 3 * x[1]]))
        x = jnp.array([0., 2.])
        options = dict(mode=mode, use_perturbation=True, perturbation_mode='absolute',
                       perturbation_magnitude=.1, seed=4)
        vjp = alm.initialize_constraint_weights(constraint, x, constraint.init(x), **options)
        direct = alm.initialize_constraint_weights(constraint, x, constraint.init(x), jacobian_method='jacrev', **options)
        assert_tree_close(vjp, direct)
        if mode == 'element':
            perturbed = alm.perturb_params_absolute(x, .1, seed=4)
            np.testing.assert_allclose(vjp['weight_g'], [1 / abs(2 * perturbed[0]), 1/3], rtol=2e-6)
            assert vjp['weight_g'][0] < 1e4
    with pytest.raises(ValueError, match='nonnegative'):
        alm.perturb_params_absolute(p, -.1)


def test_padding_and_autoscales_preserve_physical_units():
    for reduction, inequality in itertools.product((jnp.sum, jnp.mean, jnp.max), (False, True)):
        constructor = alm.ineq if inequality else alm.eq
        options = dict(penalty=2., reduction=reduction, eta_tol=jnp.array([.001, .002, .003]))
        plain = constructor(lambda x: x, **options)
        padded = constructor(lambda x: x, pad_output_to_power_of_2=True, **options)
        x = jnp.array([-1., -2., -3.])
        p, q = plain.init(x), padded.init(x)
        assert q['lambda'].value.shape == (4,)
        np.testing.assert_allclose(q['_mask'], [1, 1, 1, 0])
        assert q['eta_tol'][-1] > 0
        np.testing.assert_allclose(plain.loss(p, x)[0], padded.loss(q, x)[0])
        np.testing.assert_allclose(jax.grad(lambda y: plain.loss(p, y)[0])(x),
                                   jax.grad(lambda y: padded.loss(q, y)[0])(x))
        # Dummy entries must not dilute means or the inner stopping tolerance.
        q['lambda'] = q['lambda']._replace(penalty=jnp.array([2., 4., 6., 1000.]),
                                           omega=jnp.array([.1, .2, .3, 1e-12]))
        np.testing.assert_allclose(alm.penalty_average(q), 4.)
        np.testing.assert_allclose(alm.omega_min_from_lagrange_params(q), .1)
        np.testing.assert_allclose(alm.masked_reduction(jnp.array([1., 2., 3., 999.]), 3, jnp.mean), 2.)

    constraint = alm.ScaledConstraint(alm.ineq(lambda x: 100 * x), auto_scale=True, max_scale_factor=.1)
    x = jnp.array([2.])
    params = constraint.init(x)
    np.testing.assert_allclose(constraint._initial_scale, .005)
    # Loss evaluates physical h, while its Lagrangian uses the fixed scale.
    value, residual = constraint.loss(params, jnp.array([-.1]))
    np.testing.assert_allclose(residual, [-10.])
    np.testing.assert_allclose(value, .5 * (.005 * 10)**2, rtol=2e-6)
    nested = alm.ScaledConstraint(alm.ScaledConstraint(alm.eq(lambda y: y), auto_scale=False,
                                  scale_factor=2.), auto_scale=False, scale_factor=3.)
    p = nested.init(jnp.ones(1))
    np.testing.assert_allclose(nested.loss(p, jnp.ones(1))[0], 18.)

    plain = alm.eq(lambda x: x)
    custom = alm.BaseConstraint(plain.init, lambda p, x: plain.loss(p, x))
    scaled = alm.ScaledConstraint(custom, auto_scale=True)
    x = jnp.array([2.])
    params = scaled.init(x)
    value, residual = scaled.loss(params, x)
    np.testing.assert_allclose(value, 1.)
    np.testing.assert_allclose(residual, x)
    elementwise = alm.ScaledConstraint(plain, elementwise_scale=True, max_scale_factor=.1)
    x = jnp.array([0., 2.])
    p = elementwise.init(x)
    np.testing.assert_allclose(p['_auto_scale'], [10., .5])
    np.testing.assert_allclose(elementwise.loss(p, x)[0], .5)
    for constructor in (alm.eq, alm.ineq):
        with pytest.raises(ValueError, match='model_lagrangian'):
            constructor(lambda x: x, model_lagrangian='invalid')
        with pytest.raises(ValueError, match='returned None'):
            constructor(lambda x: None).init(x)
    for options in ({'max_scale_factor': 0}, {'auto_scale': False, 'scale_factor': 0}):
        with pytest.raises(ValueError, match='positive'):
            alm.ScaledConstraint(plain, **options)
    with pytest.raises(TypeError, match='callables'):
        alm.ScaledConstraint(None)
    # Feasibility uses physical h>=0; explicit slacks use the equality h-s=0.
    residual = jnp.array([2., -3.])
    for explicit, expected in ((False, 3.), (True, 5.)):
        c = alm.ineq(lambda x: x, use_analytical_penalty=not explicit)
        np.testing.assert_allclose(alm.total_infeasibility(residual, c.init(residual)), expected)


def test_composition_evaluates_selected_blocks_and_scales_native_dofs_once():
    calls = []
    constraints = alm.combine(alm.eq(lambda x: calls.append('first') or x),
                             alm.eq(lambda x: calls.append('second') or x + 1))
    params = constraints.init(jnp.zeros(1))
    calls.clear()
    constraints.get_single_constraint_loss(0)(params[0], jnp.zeros(1))
    assert calls == ['first']
    with pytest.raises(IndexError):
        constraints.get_single_constraint_loss(-1)

    from examples.coil_optimization.optimize_coils_surface_augmented_lagrangian import initial_configuration
    field, surface, _ = initial_configuration('qs', smoke=True)
    field.coils.curves.scaling_factor = .3
    surface.scaling_factor = .2
    constraints = alm.combine(alm.SelectiveConstraint(alm.eq(lambda f: f.coils.length), 'field'),
                              alm.SelectiveConstraint(alm.eq(lambda s: s.rc), 'surface'))
    constraints.dependencies = {'field': field, 'surface': surface}
    expected = jnp.concatenate((field.dofs, surface.dofs))
    np.testing.assert_allclose(alm._get_constraints_scaled_dofs(constraints), expected)
    np.testing.assert_allclose(alm._get_constraints_scaling_factors(constraints), jnp.ones_like(expected))
    np.testing.assert_allclose(constraints.starting_dofs, expected)
    selective = alm.SelectiveConstraint(alm.eq(lambda x: x), 'x')
    selective.dependencies = {'x': jnp.ones(1)}
    np.testing.assert_allclose(selective.scaled_dofs, [1.])
    np.testing.assert_allclose(selective.starting_dofs, [1.])
    scaled = alm.ScaledConstraint(selective)
    p = scaled.init()
    np.testing.assert_allclose(scaled.loss(p)[0], .5)
    children = {'nested': [scaled]}
    scales = alm.extract_autoscales_from_constraints(children)
    scaled._initial_scale = None
    alm.restore_autoscales_to_constraints(children, scales)
    np.testing.assert_allclose(scaled._initial_scale, 1.)
    alm.restore_autoscales_to_constraints(children, {'unrelated': {}})
    np.testing.assert_allclose(scaled._initial_scale, 1.)
    selective = alm.SelectiveConstraint(alm.ineq(lambda x: x), 'x')
    combined = alm.combine(selective)
    combined.dependencies = {'x': jnp.ones(1)}
    p = combined.init()
    np.testing.assert_allclose(combined.loss(p, _raw=True)[1][0], [1.])
    assert combined.dependencies is selective.dependencies
    combined.clear_cache()
    np.testing.assert_allclose(combined.dofs_to_pytree(jnp.ones(1))[0], [1.])
    np.testing.assert_allclose(combined.loss(combined.init(jnp.ones(1)), jnp.ones(1))[0], 0.)
    p = selective.init(jnp.ones(1))
    np.testing.assert_allclose(selective.loss(p, jnp.ones(1))[0], 0.)
    for wrapper in (selective, combined):
        with pytest.raises(TypeError, match='dictionary'):
            wrapper.dependencies = None
        wrapper.dependencies = {}
        with pytest.raises(RuntimeError, match='dependencies must be set'):
            _ = wrapper.starting_dofs
    with pytest.raises(KeyError, match='depends on'):
        selective.init()
    for construct in (lambda: alm.SelectiveConstraint(None, 'x'), lambda: alm.combine(None)):
        with pytest.raises(TypeError):
            construct()
    for construct in (lambda: alm.SelectiveConstraint(alm.eq(lambda x: x)), alm.combine):
        with pytest.raises(ValueError):
            construct()
    # Both legacy callable tuples and explicit object arguments remain accepted.
    c = alm.eq(lambda obj: obj['x'])
    combined = alm.combine((c.init, c.loss))
    obj = {'x': jnp.ones(1)}
    np.testing.assert_allclose(combined.loss(combined.init(obj), obj)[0], .5)
    combined = alm.combine(alm.SelectiveConstraint(c, 'obj'))
    np.testing.assert_allclose(combined.loss(combined.init(obj), obj)[0], .5)
    # Shared dependencies occupy one DOF block; call-time options override defaults.
    selective = alm.SelectiveConstraint(alm.ineq(lambda x, factor: factor * x,
        use_analytical_penalty=False), 'x', factor=1.)
    combined = alm.combine(selective, selective)
    combined.dependencies = {'x': jnp.ones(1)}
    np.testing.assert_allclose(combined.starting_dofs, [1.])
    p = combined.init(factor=2.)
    np.testing.assert_allclose(p[0]['slack'], [2.])
    np.testing.assert_allclose(combined.loss(p, factor=3., _raw=True)[1][0], [3.])
    np.testing.assert_allclose(combined.loss(p)[0], 1.)


def _check_bounded_weighted_solves(solver_factory):
    for model, (bound, expected, initial) in itertools.product(MODELS, ((1., 1., 0.), (3., 2., 0.), (3., 2., 5.))):
        constraint = alm.ineq(lambda x: bound - x, model_lagrangian=model,
                              use_analytical_penalty=False, eta_tol=1e-5)
        x = jnp.array([initial])
        params = x, constraint.init(x)
        solver = solver_factory(constraint, loss=lambda y: y - 2,
            model_lagrangian=model, model_mu='Mu_Adaptative_2', jit_update=True)
        state, grad, info = solver.init(params)
        for _ in range(25):
            params, state, grad, info = solver.update(params, state, grad, info)
            if float(alm.norm_constraints(info[2])) < 1e-4 and (
                float(info[0]) < 1e-8 or float(jnp.linalg.norm(grad[0])) < 1e-4):
                break
        np.testing.assert_allclose(params[0], [expected], atol=2e-3)
        assert jnp.all(params[1]['slack'] >= 0)
        np.testing.assert_allclose(params[1]['slack'], bound - params[0], atol=2e-3)

    for jit_update in (False, True):
        constraint = alm.ineq(lambda x: 4 * x, eta_tol=1e-5)
        x = jnp.array([.5, .5])
        params = x, constraint.init(x)
        solver = solver_factory(constraint,
            loss=lambda y: jnp.sum((y - jnp.array([10., 4.]))**2),
            bounds=(jnp.zeros(2), jnp.array([2., 3.])), parameter_scale=jnp.array([10., .1]),
            use_per_constraint_weights='element', use_perturbation_weights=True,
            perturbation_seed=7, jit_update=jit_update)
        state, grad, info = solver.init(params)  # same three-value API
        for _ in range(2):
            params, state, grad, info = solver.update(params, state, grad, info)
        np.testing.assert_allclose(params[0], [2., 3.], atol=1e-4)
        assert jnp.all(params[1]['weight_g'] > 0)
        assert_tree_close(params[1]['weight_g'], state.weights[0])

    for explicit, method in itertools.product((False, True), ('vjp', 'jacrev')):
        constraint = alm.combine(alm.eq(lambda x: x - 1, pad_output_to_power_of_2=True, eta_tol=1e-4),
            alm.ineq(lambda x: 4 * x + 3, use_analytical_penalty=not explicit,
                     pad_output_to_power_of_2=True, eta_tol=1e-4))
        x = jnp.zeros(3)
        weights = alm.initialize_constraint_weights(constraint, x, constraint.init(x), jacobian_method=method)
        np.testing.assert_allclose(weights[0]['weight_g'][:3], 1.)
        np.testing.assert_allclose(weights[1]['weight_g'][:3], .25)
        solver = solver_factory(constraint, jit_update=True)
        params = x, weights
        state, grad, info = solver.init(params)
        params, state, grad, info = solver.update(params, state, grad, info)
        for p, residual in zip(params[1], info[2]):
            assert p['lambda'].value.shape == residual.shape == (4,)
            assert jnp.all(jnp.isfinite(residual / p['eta_tol']))
            if 'slack' in p:
                assert jnp.all(p['slack'] >= 0)


def test_scipy_bounded_slacks_and_weighted_convergence():
    constraint = alm.ineq(lambda x: 4 - 4 * x, eta_tol=1e-5)
    x = jnp.zeros(1)
    loss = lambda y: y - 2
    p = alm.initialize_constraint_weights(constraint, x, constraint.init(x), loss=loss)
    np.testing.assert_allclose(p['weight_g'], [.25])
    params = x, p
    solver = alm.ALM_model_scipy(constraint, loss=loss, model_mu='Mu_Adaptative_2')
    state, grad, info = solver.init(params)
    for _ in range(10):
        params, state, grad, info = solver.update(params, state, grad, info)
    np.testing.assert_allclose(params[0], [1.], atol=1e-3)
    np.testing.assert_array_equal(params[1]['weight_g'], p['weight_g'])

    constraint = alm.ineq(lambda x: 3 - x, use_analytical_penalty=False, omega=1e-7)
    x = jnp.zeros(1)
    params = x, constraint.init(x)
    solver = alm.ALM_model_scipy(constraint, method='SLSQP',
        loss=lambda y: 1 + jnp.sum((y - 2)**2), bounds=(jnp.zeros(1), jnp.ones(1)))
    state, grad, info = solver.init(params)
    params, _, _, _ = solver.update(params, state, grad, info)
    np.testing.assert_allclose(params[0], [1.], atol=1e-5)
    np.testing.assert_allclose(params[1]['slack'], [2.], atol=1e-5)

    _check_bounded_weighted_solves(alm.ALM_model_scipy)


def test_scipy_methods_options_and_analytic_gradients(monkeypatch):
    for method in ('L-BFGS-B', 'SLSQP', 'BFGS', 'CG'):
        constraint = alm.ineq(lambda x: x + 10, omega=1e-7)
        x = jnp.zeros(2)
        params = x, constraint.init(x)
        results = []
        solver = alm.ALM_model_scipy(constraint, loss=lambda y: 1 + jnp.sum((y - 2)**2),
                                    method=method, inner_callback=results.append)
        state, grad, info = solver.init(params)
        params, _, _, _ = solver.update(params, state, grad, info)
        np.testing.assert_allclose(params[0], [2., 2.], atol=1e-5)
        assert results[0].success and results[0].njev > 0
        np.testing.assert_allclose(results[0].jac, [0., 0.], atol=1e-5)

    for method, explicit_slack in itertools.product(('BFGS', 'CG'), (False, True)):
        constraint = alm.ineq(lambda x: x, use_analytical_penalty=not explicit_slack)
        x = jnp.ones(1)
        params = x, constraint.init(x)
        solver = alm.ALM_model_scipy(constraint, method=method,
            bounds=None if explicit_slack else (jnp.zeros(1), jnp.ones(1)))
        state, grad, info = solver.init(params)
        with pytest.raises(ValueError, match='cannot enforce'):
            solver.update(params, state, grad, info)

    constraint = alm.ineq(lambda x: x + 10, omega=1e-9)
    x = jnp.zeros(2)
    params = x, constraint.init(x)
    results = []
    solver = alm.ALM_model_scipy(constraint,
        loss=lambda y, target: 1 + jnp.sum((y - target)**2), target=1.,
        inner_options={'gtol': 1e-8, 'maxiter': 20}, inner_callback=results.append)
    state, grad, info = solver.init(params, target=3.)
    params, _, _, _ = solver.update(params, state, grad, info, target=3.)
    np.testing.assert_allclose(params[0], [3., 3.], atol=1e-5)
    assert results[0].success
    for options in ({'method': 'invalid'}, {'model_lagrangian': 'invalid'},
                    {'inner_maxiter': 0}, {'history_size': 0},
                    {'parameter_scale': 0}, {'parameter_scale': jnp.nan}):
        with pytest.raises(ValueError):
            alm.ALM_model_scipy(constraint, **options)
    for loss in (lambda y: jnp.nan * y, lambda y: jnp.sqrt(-jnp.ones_like(y))):
        solver = alm.ALM_model_scipy(constraint, loss=loss)
        with pytest.raises(FloatingPointError, match='objective or gradient'):
            solver.update(params, *solver.init(params))
    # Reject corrupt optimizer results and surface unsuccessful line searches.
    from scipy.optimize import OptimizeResult
    with monkeypatch.context() as patch:
        patch.setattr(alm, 'minimize', lambda *a, **kw: OptimizeResult(x=np.full(2, np.nan)))
        solver = alm.ALM_model_scipy(constraint)
        with pytest.raises(FloatingPointError, match='solution'):
            solver.update(params, *solver.init(params))
        patch.setattr(alm, 'minimize', lambda *a, **kw: OptimizeResult(
            x=np.zeros(2), success=False, status=2, message='line search failed'))
        with pytest.warns(RuntimeWarning, match='line search failed'):
            solver.update(params, *solver.init(params))


def test_zero_norms_and_mixed_precision_keep_updates_finite():
    for model in MODELS:
        constraint = alm.ineq(lambda x: x + 3, model_lagrangian=model)
        x = jnp.zeros(2)
        solver = alm.ALM_model_scipy(constraint, loss=lambda y: y, model_lagrangian=model)
        params = x, constraint.init(x)
        state, grad, info = solver.init(params)
        np.testing.assert_array_equal(grad[0], jnp.zeros(2))
        params, _, grad, info = solver.update(params, state, grad, info)
        assert all(jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves((params, grad, info)))

    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        constraint = alm.eq(lambda x: x)
        x = jnp.ones(1, dtype=jnp.float32)
        params = x, constraint.init(x)
        solver = alm.ALM_model_scipy(constraint, model_mu='Mu_Adaptative_2',
                                    beta=jnp.array(2., dtype=jnp.float64))
        state, grad, info = solver.init(params)
        params, _, _, _ = solver.update(params, state, grad, info)
        assert params[0].dtype == jnp.float32
        assert all(v.dtype == jnp.float32 for v in params[1]['lambda'])
    finally:
        jax.config.update('jax_enable_x64', previous)
    solver = alm.ALM_model_scipy(constraint, use_scaled_main_params=True)
    updated = solver.update(params, *solver.init(params))[0][0]
    np.testing.assert_allclose(updated, jnp.zeros(1), atol=1e-6)


def test_optional_native_backend_uses_shared_alm_without_scipy(monkeypatch, tmp_path):
    for jit_update, model in itertools.product((False, True), MODELS):
        pytest.importorskip('jaxopt')
        def unexpected_scipy(*args, **kwargs):
            raise AssertionError('Native JAXopt backend called SciPy')
        monkeypatch.setattr(alm, 'minimize', unexpected_scipy)
        constraint = alm.ineq(lambda x: x + 3, omega=1e-6)
        x = jnp.zeros(2)
        params = x, constraint.init(x)
        solver = alm.ALM_model_jaxopt_lbfgsb(constraint, loss=lambda y: y,
                                           model_lagrangian=model, jit_update=jit_update)
        state, grad, info = solver.init(params)
        params, _, grad, info = solver.update(params, state, grad, info)
        assert all(jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves((params, grad, info)))
        np.testing.assert_allclose(grad[0], 0.)
        with pytest.raises(ValueError, match='SciPy-only'):
            alm.ALM_model_jaxopt_lbfgsb(constraint, inner_options={'maxls': 40})

    _check_bounded_weighted_solves(alm.ALM_model_jaxopt_lbfgsb)

    for full_object in (False, True):
        _check_checkpoint_continuation(alm.ALM_model_jaxopt_lbfgsb, tmp_path, full_object)


def test_pinned_host_requires_gpu_and_preserves_bounded_weighted_slacks():
    constraint = alm.ineq(lambda x: 4 * x + 3, use_analytical_penalty=False,
                          pad_output_to_power_of_2=True)
    x = jnp.zeros(3)
    params = x, constraint.init(x)
    options = dict(loss=lambda y: 1 + jnp.sum((y - 2)**2),
                   bounds=(jnp.zeros(3), jnp.ones(3)), parameter_scale=2.,
                   use_per_constraint_weights=True, jit_update=True)
    pinned = alm.ALM_model_scipy(constraint, scipy_pinned_host=True, **options)
    state, grad, info = pinned.init(params)
    if jax.devices()[0].platform != 'gpu':
        with pytest.raises(ValueError, match='requires a single GPU'):
            pinned.update(params, state, grad, info)
    else:
        ordinary = alm.ALM_model_scipy(constraint, **options)
        expected = ordinary.update(params, *ordinary.init(params))
        actual = pinned.update(params, state, grad, info)
        for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5)
        assert jnp.all(actual[0][1]['slack'] >= 0)
        assert jnp.all((actual[0][0] >= 0) & (actual[0][0] <= 1))
    with pytest.raises(ValueError, match='SciPy-only'):
        alm.ALM_model_jaxopt_lbfgsb(constraint, scipy_pinned_host=True)


def _check_checkpoint_continuation(solver_factory, tmp_path, full_object):
    constraint = checkpoint_problem()
    x = jnp.array([4.])
    multipliers = alm.initialize_constraint_weights(constraint, x, constraint.init(x))
    params = x, multipliers
    solver = solver_factory(constraint, model_mu='Mu_Adaptative_2', jit_update=True)
    state, grad, info = solver.init(params)
    params, state, grad, info = solver.update(params, state, grad, info)
    filename = tmp_path/'state.pkl'
    alm.save_checkpoint_with_constraints(filename, params, state, grad, .001, .0001, info,
                                         constraint, save_constraint_object=full_object)
    fresh = checkpoint_problem()
    fresh.init(jnp.array([30.]))
    restored = alm.load_and_merge_constraints(filename, None if full_object else fresh)
    assert_tree_close(restored[:3], (params, state, grad))
    restored_constraint = restored[-1]
    assert_tree_close(alm.extract_autoscales_from_constraints(restored_constraint),
                      alm.extract_autoscales_from_constraints(constraint))
    continuation = solver_factory(restored_constraint, model_mu='Mu_Adaptative_2', jit_update=True)
    uninterrupted = solver.update(params, state, grad, info)
    resumed = continuation.update(restored[0], restored[1], restored[2], restored[5])
    assert_tree_close(uninterrupted, resumed)
    kept = alm.restart_optimization_from_file(filename, constraint_object=restored_constraint)
    assert_tree_close(kept[0], params)
    preserved, summary = alm.restart_with_learned_scales(params, state)
    assert_tree_close(preserved, params)
    assert summary['num_constraint_blocks'] == 2 and summary['num_constraints'] == 2
    assert summary['penalty_min'] <= summary['penalty_mean'] <= summary['penalty_max']
    reset = alm.restart_optimization_from_file(filename, constraint_object=restored_constraint,
        alm_object=continuation, keep_learned_scales=False, initial_penalty=2.)
    assert jnp.all(reset[0][1][0]['lambda'].penalty == 2.)


def test_checkpoints_restore_scales_weights_and_exact_continuation(tmp_path):
    for full_object in (False, True):
        _check_checkpoint_continuation(alm.ALM_model_scipy, tmp_path, full_object)
    filename = tmp_path/'state.pkl'
    with pytest.raises(ValueError, match='structure/order'):
        alm.load_checkpoint_for_reconstruction(filename, alm.eq(lambda x: x))
    wrong_shape = alm.combine(alm.ScaledConstraint(alm.eq(lambda x: jnp.tile(x, 2) - 1)),
        alm.ineq(lambda x: 3 - x, use_analytical_penalty=False))
    with pytest.raises(ValueError, match='shapes'):
        alm.load_checkpoint_for_reconstruction(filename, wrong_shape)
    with pytest.raises(ValueError, match='positive penalty'):
        alm.restart_optimization_from_file(filename, keep_learned_scales=False)
    with pytest.raises(ValueError, match='alm_object'):
        alm.restart_optimization_from_file(filename, reinit_alm_state=True)


def test_atomic_checkpoints_and_restarts_keep_actual_tolerances_and_initial_weights(tmp_path):
    solver_factory = alm.ALM_model_scipy
    constraint = alm.eq(lambda x: x - 1)
    params = jnp.zeros(1), constraint.init(jnp.zeros(1))
    solver = solver_factory(constraint, jit_update=True)
    state, grad, info = solver.init(params)
    result = alm.augmented_lagrangian_optimization(params, state, grad, info, solver,
        num_iterations=1, save_every=1, checkpoint_prefix=str(tmp_path/'iteration'))
    filename = tmp_path/'iteration_iter_1.pkl'
    saved = alm.load_optimization_state(filename)
    assert_tree_close(saved[:3], result[:3])
    assert jnp.ndim(saved[3]) == 0 and jnp.ndim(saved[4]) == 0
    original = filename.read_bytes()
    with pytest.raises((pickle.PicklingError, AttributeError)):
        alm.save_optimization_state(filename, lambda x: x, state, grad, 1., 1., info)
    assert filename.read_bytes() == original
    alm.augmented_lagrangian_optimization(params, state, grad, info, solver,
        num_iterations=3, save_every=2, checkpoint_prefix=str(tmp_path/'sparse'))
    assert not (tmp_path/'sparse_iter_1.pkl').exists()
    assert (tmp_path/'sparse_iter_2.pkl').exists() and (tmp_path/'sparse_iter_3.pkl').exists()
    assert_tree_close(alm.load_checkpoint_for_reconstruction(filename, constraint), saved)
    with pytest.raises(ValueError, match='Provide reconstructed constraints'):
        alm.load_and_merge_constraints(filename)
    for options in ({'num_iterations': -1}, {'save_every': 0}):
        with pytest.raises(ValueError, match='num_iterations'):
            alm.augmented_lagrangian_optimization(params, state, grad, info, solver, **options)

    constraint = alm.eq(lambda x: x**2 - 1)
    x = jnp.array([2.])
    params = x, constraint.init(x)
    solver = solver_factory(constraint, use_per_constraint_weights=True,
        use_perturbation_weights=True, perturbation_seed=9, jit_update=True)
    state, grad, info = solver.init(params)
    filename = tmp_path/'initial.pkl'
    alm.save_optimization_state(filename, params, state, grad, 1., 1., info)
    resumed = alm.restart_optimization_from_file(filename, alm_object=solver, reinit_alm_state=True)
    np.testing.assert_array_equal(resumed[0][1]['weight_g'], state.weights[0])
    np.testing.assert_array_equal(resumed[1].weights[0], state.weights[0])


def test_core_imports_and_legacy_checkpoints_do_not_require_optimizer_extras(tmp_path, monkeypatch):
    constraint = alm.eq(lambda x: x - 1)
    x = jnp.zeros(1)
    solver = alm.ALM_model_scipy(constraint)
    params = x, constraint.init(x)
    _, grad, info = solver.init(params)
    filename = tmp_path/'legacy.pkl'
    stream = io.BytesIO()
    pickle.dump(params, stream)
    # Protocol-0 instance of the old zero-field NamedTuple; no Optax import.
    stream.write(b'coptax._src.base\nEmptyState\n)R.')
    for value in (grad, 1., 1., info):
        pickle.dump(value, stream)
    filename.write_bytes(stream.getvalue())
    restored = alm.load_optimization_state(filename)
    assert isinstance(restored[1], alm._EmptyState)
    solver.update(restored[0], restored[1], restored[2], restored[5])

    script = '''
import importlib.abc
import sys
class BlockRemovedDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('optax', 'jaxopt'):
            raise ImportError('Blocked removed dependency: ' + fullname)
sys.meta_path.insert(0, BlockRemovedDependencies())
from essos import augmented_lagrangian as alm
import jax.numpy as jnp
c = alm.eq(lambda x: x - 1)
p = jnp.zeros(1), c.init(jnp.zeros(1))
s = alm.ALM_model_scipy(c)
state, grad, info = s.init(p)
p, _, _, _ = s.update(p, state, grad, info)
assert abs(float(p[0][0]) - 1) < 1e-5
assert 'optax' not in sys.modules and 'jaxopt' not in sys.modules
try:
    alm.ALM_model_jaxopt_lbfgsb(c)
except ImportError as error:
    assert 'essos[jaxopt]' in str(error)
else:
    raise AssertionError('Optional backend should require JAXopt')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    original_import = builtins.__import__
    def without_jaxopt(name, *args, **kwargs):
        if name.split('.')[0] == 'jaxopt':
            raise ImportError('optional dependency unavailable')
        return original_import(name, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', without_jaxopt)
        with pytest.raises(ImportError, match=r'essos\[jaxopt\]'):
            alm.ALM_model_jaxopt_lbfgsb(constraint)
