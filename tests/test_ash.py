"""Independent manufactured/continuum checks; no stellarator physics certificate."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

pytest.importorskip("solvax")

from essos.ash import (
    _bernoulli, evaluate_helium_residence, check_helium_residence,
    restricted_residence, trace_response, convection_from_marker_drift,
)


def cylinder(n=64, *, d=1., v=0., kappa=np.inf, power=1., source=None, cost=None):
    e = jnp.linspace(0, 1, n+1)**power
    return evaluate_helium_residence(
        e, jnp.diff(e**2), 2*e, jnp.ones_like(e)*d, v*e,
        jnp.ones(n) if source is None else source,
        boundary_kappa_m_s=kappa, core_cost=cost)


def test_bernoulli_limits_and_native_derivatives():
    x = jnp.array([-1000., -50., -.001, 0., .001, 50., 1000.])
    np.testing.assert_allclose(_bernoulli(-x)-_bernoulli(x), x, atol=1.e-13)
    assert float(_bernoulli(0.)) == 1.
    assert float(jax.grad(_bernoulli)(0.)) == -.5
    assert np.isfinite(jax.vmap(jax.grad(_bernoulli))(x)).all()


def test_absorbing_robin_and_second_order_convergence():
    errors = []
    for n in (16, 32, 64):
        r = check_helium_residence(cylinder(n))
        errors.append(float(r.mean_residence_s)-.125)
        robin = check_helium_residence(cylinder(n, kappa=.2))
        np.testing.assert_allclose(robin.mean_residence_s-r.mean_residence_s, 2.5, rtol=1.e-11)
    np.testing.assert_allclose(np.array(errors[:-1])/errors[1:], 4., rtol=1.e-9)
    nonuniform = check_helium_residence(cylinder(256, power=1.5))
    np.testing.assert_allclose(nonuniform.mean_residence_s, .125, rtol=5.e-5)


def test_variable_geometry_against_independent_continuum_integral():
    # Spherical W=3r^2, D=1+r, V=2r(1+r), hence integrating factor exp(r^2).
    n = 256
    e = jnp.linspace(0, 1, n+1)
    r = evaluate_helium_residence(e, jnp.diff(e**3), 3*e**2, 1+e,
                                 2*e*(1+e), jnp.ones(n), boundary_kappa_m_s=.8)
    assert bool(r.valid)
    def inner(t):
        return quad(lambda y: 3*y*y*np.exp(y*y), 0, t, epsabs=1.e-12)[0]
    def exact(x):
        return (quad(lambda t: np.exp(-t*t)*inner(t)/(3*t*t*(1+t)), x, 1)[0]
                + np.exp(-1)*inner(1)/(3*.8))
    centers = np.asarray((e[:-1]+e[1:])/2)
    np.testing.assert_allclose(r.occupation_s[::32], [exact(x) for x in centers[::32]], rtol=8.e-5)


def test_weighted_duality_balance_source_scaling_and_location():
    n = 64
    source = 1+jnp.exp(-jnp.linspace(0, 1, n)**2/.1)
    cost = 1+3*jnp.linspace(1, 0, n)
    a = cylinder(n, v=-2., kappa=.4, source=source, cost=cost)
    b = cylinder(n, v=-2., kappa=.4, source=7*source, cost=cost)
    assert bool(a.valid) and bool(b.valid)
    assert float(a.duality_relative_error) < 1.e-11
    assert float(a.balance_relative_error) < 1.e-11
    np.testing.assert_allclose(b.inventory/a.inventory, 7., rtol=1.e-11)
    np.testing.assert_allclose(b.mean_occupation_s, a.mean_occupation_s, rtol=1.e-11)
    assert a.cost > a.inventory
    core, edge = np.zeros(n), np.zeros(n)
    core[4], edge[-5] = 1., 1.
    assert cylinder(n, source=core).mean_residence_s > cylinder(n, source=edge).mean_residence_s
    assert cylinder(n, d=.03, v=.09).mean_residence_s > 10*cylinder(n).mean_residence_s


def test_native_jit_grad_includes_geometry_source_cost_and_boundary():
    n = 18
    def objective(p):
        # Every input category depends on p; finite differences exercise all paths.
        e = jnp.linspace(0, 1, n+1)*p[0]
        s = 1+p[3]*jnp.linspace(0, 1, n)
        return evaluate_helium_residence(
            e, jnp.diff(e**2), 2*e, jnp.ones_like(e)*p[1], p[2]*e,
            s, boundary_kappa_m_s=p[4], core_cost=1+p[5]*jnp.linspace(1, 0, n)).mean_occupation_s
    p = jnp.array([1.1, .8, .3, .2, .6, .4])
    native = jax.jit(jax.grad(objective))(p)
    steps = np.array([1.e-3, 1.e-4, 1.e-5])
    errors = []
    for step in steps:
        fd = np.array([(objective(p.at[i].add(step))-objective(p.at[i].add(-step)))/(2*step)
                       for i in range(len(p))])
        errors.append(float(np.max(np.abs(native-fd)/np.maximum(np.abs(native), 1.e-8))))
        np.testing.assert_allclose(native, fd, rtol=5.e-6, atol=2.e-8)
    assert errors[-1] < .02*errors[0]
    # Absorbing boundary must not introduce inf*0 gradients through W/geometry.
    def absorbing(radius):
        e = jnp.linspace(0, 1, n+1)*radius
        return evaluate_helium_residence(e, jnp.diff(e**2), 2*e, jnp.ones_like(e),
            jnp.zeros_like(e), jnp.ones(n), boundary_kappa_m_s=jnp.inf).mean_residence_s
    np.testing.assert_allclose(jax.grad(absorbing)(1.), 2*absorbing(1.), rtol=1.e-11)
    np.testing.assert_allclose(jax.grad(lambda d: cylinder(16, d=d).mean_residence_s)(1.),
                               -cylinder(16).mean_residence_s, rtol=1.e-11)


@pytest.mark.parametrize('kwargs', [dict(d=0.), dict(d=-1.), dict(d=np.nan),
    dict(kappa=0.), dict(kappa=-1.), dict(kappa=np.nan), dict(source=-np.ones(8)),
    dict(source=np.zeros(8)), dict(cost=-np.ones(8))])
def test_invalid_values_are_explicit_under_jit(kwargs):
    r = jax.jit(lambda: cylinder(8, **kwargs))()
    assert not bool(r.valid)
    assert np.isnan(r.mean_occupation_s)
    with pytest.raises(ArithmeticError):
        check_helium_residence(r)


def test_bad_mesh_shape_and_numerically_closed_escape_rejected():
    with pytest.raises(ValueError):
        cylinder(8, source=np.ones(9))
    e = np.linspace(0, 1, 9)
    r = evaluate_helium_residence(e, np.diff(e**2), 2*e, np.ones(9),
                                 -1.e6*np.ones(9), np.ones(8), boundary_kappa_m_s=np.inf)
    assert not bool(r.valid)
    bad = e.copy(); bad[3] = bad[2]
    r = evaluate_helium_residence(bad, np.diff(e**2), 2*e, np.ones(9),
                                 np.zeros(9), np.ones(8), boundary_kappa_m_s=1.)
    assert not bool(r.valid)


def test_common_horizon_survival_integral_and_weighted_uncertainty():
    r = restricted_residence([1., 2., np.inf], 4., weights=[1., 2., 1.])
    assert bool(r.valid) and not bool(r.no_censoring_in_sample)
    assert float(r.restricted_mean_s) == 2.25
    assert float(r.escaped_fraction) == .75
    np.testing.assert_allclose(r.standard_error_s, np.sqrt(.375*1.1875/.625))
    np.testing.assert_allclose(r.effective_sample_size, 8/3)
    # Integrating empirical survival independently agrees with the restricted mean.
    t = np.array([0., 1., 2., 4.])
    survival = np.array([1., .75, .25])
    assert float(r.restricted_mean_s) == np.diff(t) @ survival
    for times, failed in (([1., 2., np.inf], [False, True, False]),
                           ([1., np.nan, np.inf], None), ([1., 5., np.inf], None),
                           ([1., -np.inf, np.inf], None)):
        bad = restricted_residence(times, 4., failed=failed)
        assert not bool(bad.valid) and np.isnan(bad.restricted_mean_s)
    zero_weight_failure = restricted_residence([1., np.inf], 4., weights=[1., 0.], failed=[False, True])
    assert not bool(zero_weight_failure.valid)


def test_trace_response_and_marker_drift_correction():
    x = jnp.array([-2., 0., 3.])
    r = jax.jit(trace_response)(x, .4-.7*x)
    assert bool(r.valid)
    np.testing.assert_allclose([r.diffusion_m2_s, r.convection_m_s], [.7, .4], atol=1.e-14)
    assert not bool(trace_response(x, .4-.7*x+.1*x*x).valid)
    assert not bool(trace_response(x, .4+.7*x).valid)
    assert not bool(trace_response(jnp.zeros(3), jnp.ones(3)).valid)
    # W=3r^2, D=1+r, V=.4r: A=V+1+(1+r)*2/r.
    radius = jnp.array([.1, .3, .8])
    d = 1+radius
    drift = .4*radius+1+d*2/radius
    np.testing.assert_allclose(convection_from_marker_drift(drift, d, 1., 2/radius), .4*radius, atol=2.e-15)


def test_explicit_coefficient_wout_example_and_provenance(tmp_path):
    import hashlib
    import importlib.util
    import json
    from pathlib import Path
    from netCDF4 import Dataset
    example_path = Path(__file__).parents[1]/'examples/particle_tracing/helium_ash_removal.py'
    spec = importlib.util.spec_from_file_location('helium_example', example_path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    # Minimal WOUT metadata is enough here: no claim of a kinetic/equilibrium run.
    wout = tmp_path/'wout.nc'
    with Dataset(wout, 'w') as nc:
        nc.createVariable('Aminor_p', 'f8').assignValue(1.)
    coefficients, metadata = example.manufactured_response(16)
    coefficients = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in coefficients.items()}
    coefficients['boundary_kappa_m_s'] = None
    metadata['equilibrium_sha256'] = hashlib.sha256(wout.read_bytes()).hexdigest()
    bundle = tmp_path/'response.json'
    bundle.write_text(json.dumps(dict(coefficients=coefficients, metadata=metadata)))
    output = tmp_path/'report.json'
    report = example.main(['--coefficients', str(bundle), '--wout', str(wout), '--output', str(output)])
    assert report['pumped_fraction'] == 'not modeled'
    assert report['thermal_orbit_verification'] == 'not evaluated'
    assert report['metadata']['modeled_boundary_s'] == 1.
    assert json.loads(output.read_text()) == report
    metadata['equilibrium_sha256'] = 'wrong'
    bundle.write_text(json.dumps(dict(coefficients=coefficients, metadata=metadata)))
    with pytest.raises(ValueError, match='hash'):
        example.load_response(bundle, wout)


def test_huge_finite_marker_weights_are_normalized_without_overflow():
    huge = jnp.asarray([1.e308, 1.e308])
    r = jax.jit(lambda w: restricted_residence(jnp.array([1., 2.]), 3., weights=w))(huge)
    assert bool(r.valid)
    np.testing.assert_allclose(r.restricted_mean_s, 1.5)
    np.testing.assert_allclose(r.escaped_fraction, 1.)
    np.testing.assert_allclose(r.standard_error_s, .5)
    np.testing.assert_allclose(r.effective_sample_size, 2.)
    # Large integer counts are converted before reduction, avoiding wraparound.
    counts = jnp.asarray([9_000_000_000_000_000_000]*3, dtype=jnp.int64)
    integer_result = restricted_residence([1., 2., 3.], 4., weights=counts)
    assert bool(integer_result.valid)
    np.testing.assert_allclose(integer_result.restricted_mean_s, 2.)
    np.testing.assert_allclose(integer_result.escaped_fraction, 1.)
    np.testing.assert_allclose(integer_result.effective_sample_size, 3.)


def test_overflowing_inventory_invalidates_otherwise_accurate_weighted_solve():
    # Tiny cost weights keep occupation/cost finite while ordinary inventory
    # overflows. Residual/duality/balance checks alone cannot detect this.
    r = jax.jit(lambda: evaluate_helium_residence(
        jnp.array([0., .5, 1.]), jnp.array([1.e200, 1.e200]),
        jnp.ones(3), jnp.ones(3), jnp.zeros(3), jnp.ones(2),
        boundary_kappa_m_s=jnp.inf, core_cost=jnp.array([1.e-200, 1.e-200])))()
    assert not bool(r.valid)
    assert np.isnan(r.inventory) and np.isnan(r.mean_residence_s)
    assert np.isnan(r.mean_occupation_s)
    assert float(r.forward_residual) == 0.
    assert float(r.duality_relative_error) == 0.
    with pytest.raises(ArithmeticError):
        check_helium_residence(r)
