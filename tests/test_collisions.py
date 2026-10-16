"""Independent physical and numerical checks of the opt-in Cartesian utility."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from essos.collisions import maxwellian_cartesian_coefficients as coefficients
from essos.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
)
from essos.constants import (
    ALPHA_PARTICLE_MASS as MASS,
)
from essos.constants import (
    ELECTRON_MASS,
    EPSILON_0,
    PROTON_MASS,
)
from essos.constants import (
    ELEMENTARY_CHARGE as E,
)

MB = np.array([2 * PROTON_MASS, 3 * PROTON_MASS, ELECTRON_MASS, MASS])
QB = np.array([E, E, -E, CHARGE])
NB = np.array([0.42, 0.42, 0.92, 0.04]) * 1e20
TB = np.full(4, 1000.0)
LOGS = np.array([14.0, 15.0, 16.0, 13.0])
V0 = np.sqrt(1000 * E / MASS)


def evaluate(v, temperature=TB, logs=LOGS, density=NB):
    return coefficients(
        v, MASS, CHARGE, MB, QB, density, temperature, coulomb_logs=logs
    )


def independent_ab(speed, temperature=TB, logs=LOGS):
    """Rosenbluth radial-potential integrals, independent of erf/series helper."""
    vt = np.sqrt(2 * temperature * E / MB)
    gamma = CHARGE**2 * QB**2 * logs / (4 * np.pi * EPSILON_0**2 * MASS**2)
    a = []
    b = []
    for vb, g, n in zip(vt, gamma, NB):
        x = speed / vb
        c = (
            2
            / np.sqrt(np.pi)
            * quad(lambda r: r * r * np.exp(-r * r), 0, x, epsabs=1e-13)[0]
            / x**3
            if x
            else 2 / (3 * np.sqrt(np.pi))
        )
        e = (
            2
            / np.sqrt(np.pi)
            * quad(lambda r: np.exp(-r * r), 0, x, epsabs=1e-13)[0]
            / x
            if x
            else 2 / np.sqrt(np.pi)
        )
        a.append(g * n / vb * c)
        b.append(g * n / vb * (e - c) / 2)
    return np.asarray(a), np.asarray(b)


def test_diffusion_matches_independent_SI_rosenbluth_integrals():
    rng = np.random.default_rng(91773)
    speeds = np.r_[0.0, np.geomspace(1e-10, 12, 48)] * V0
    directions = rng.normal(size=(len(speeds), 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    _A, D, S = jax.jit(jax.vmap(evaluate))(jnp.asarray(directions * speeds[:, None]))
    D, S = map(np.asarray, (D, S))
    parallel = np.einsum("ni,nij,nj->n", directions, D, directions)
    perpendicular = (np.trace(D, axis1=1, axis2=2) - parallel) / 2
    expected = np.array([[a.sum(), b.sum()] for a, b in map(independent_ab, speeds)])
    np.testing.assert_allclose(
        np.column_stack([parallel, perpendicular]), expected, rtol=2e-10
    )
    np.testing.assert_allclose(
        S @ S.swapaxes(1, 2), 2 * D, rtol=3e-12, atol=1e-12 * expected.max()
    )
    assert np.linalg.eigvalsh(D).min() > 0


def test_zero_speed_isotropic_value_and_smooth_velocity_noise():
    out = evaluate([0, 0, 0])
    a, _ = independent_ab(0.0)
    np.testing.assert_allclose(out.diffusion, np.eye(3) * a.sum(), rtol=2e-13)
    np.testing.assert_array_equal(out.ito_drift, np.zeros(3))
    derivative = jax.jacfwd(lambda v: evaluate(v).noise)(jnp.zeros(3))
    np.testing.assert_array_equal(derivative, np.zeros((3, 3, 3)))
    # Crossing q=0 must not introduce an eigenvector-frame discontinuity.
    jumps = []
    for eps in [1e-3, 1e-5, 1e-7]:
        l = evaluate(jnp.array([1.0, 0.3, -eps]) * V0).noise
        r = evaluate(jnp.array([1.0, 0.3, eps]) * V0).noise
        jumps.append(float(jnp.max(abs(l - r))))
    assert jumps[-1] < jumps[0] * 1.01e-4


@pytest.mark.parametrize("mixed", [False, True])
def test_ito_flux_identity_includes_each_regular_log_derivative(mixed):
    temperature = np.array([600.0, 1400.0, 2200.0, 900.0]) if mixed else TB

    def logs(s):
        return jnp.asarray(LOGS) * (
            1 + jnp.array([0.1, 0.3, 0.2, 0.4]) * s / (V0 * V0 + s)
        )

    def actual(v):
        return evaluate(v, temperature, logs)

    rng = np.random.default_rng(5719)
    vv = rng.normal(size=(16, 3)) * V0
    vv = np.vstack([np.zeros(3), vv])
    for v in vv:
        A, D, _ = actual(jnp.asarray(v))
        divergence = jnp.einsum(
            "ijj->i", jax.jacfwd(lambda w: actual(w).diffusion)(jnp.asarray(v))
        )
        a, _ = independent_ab(np.linalg.norm(v), temperature, np.asarray(logs(v @ v)))
        expected_flux = -np.sum(MASS / (temperature * E) * a) * v
        np.testing.assert_allclose(
            (A - divergence) / (V0 * 1e4),
            expected_flux / (V0 * 1e4),
            rtol=3e-11,
            atol=3e-12,
        )
        if not mixed:
            np.testing.assert_allclose(
                (A - divergence + D @ v / V0**2) / (V0 * 1e4), 0, atol=3e-12
            )


def test_species_sum_and_permutation_preserve_SI_operator():
    v = jnp.array([0.4, -0.7, 1.1]) * V0
    total = evaluate(v)
    pairs = [
        coefficients(
            v,
            MASS,
            CHARGE,
            MB[i : i + 1],
            QB[i : i + 1],
            NB[i : i + 1],
            TB[i : i + 1],
            coulomb_logs=LOGS[i],
        )
        for i in range(4)
    ]
    np.testing.assert_allclose(
        sum(x.ito_drift for x in pairs), total.ito_drift, rtol=3e-13
    )
    np.testing.assert_allclose(
        sum(x.diffusion for x in pairs), total.diffusion, rtol=3e-13
    )
    order = np.array([2, 0, 3, 1])
    permuted = coefficients(
        v,
        MASS,
        CHARGE,
        MB[order],
        QB[order],
        NB[order],
        TB[order],
        coulomb_logs=LOGS[order],
    )
    for a, b in zip(total, permuted):
        np.testing.assert_allclose(a, b, rtol=3e-13, atol=1e-12)


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 4.0])
def test_initial_maxwellian_temperature_relaxation_matches_relative_gaussian_SI_rate(
    ratio,
):
    temperature = np.array([600.0, 1400.0, 2200.0, 900.0])
    Ta = ratio * 1000 * E
    gamma = CHARGE**2 * QB**2 * LOGS / (4 * np.pi * EPSILON_0**2 * MASS**2)
    rate = (
        (2 * np.sqrt(2 / np.pi) / 3)
        * (MASS / MB)
        * gamma
        * NB
        / (Ta / MASS + temperature * E / MB) ** 1.5
    )
    expected = np.sum(rate * (temperature / 1000 - ratio))
    compiled = jax.jit(lambda v: evaluate(jnp.array([v, 0.0, 0.0]), temperature))

    def integrand(w):
        v = w * V0
        A, D, _ = compiled(v)
        density = np.sqrt(2 / np.pi) * w * w * np.exp(-w * w / (2 * ratio)) / ratio**1.5
        return (2 / 3) * (v * float(A[0]) + float(jnp.trace(D))) / V0**2 * density

    measured = quad(integrand, 0, 24, epsabs=2e-7, epsrel=2e-10)[0]
    np.testing.assert_allclose(measured, expected, rtol=2e-10, atol=2e-7)


def test_native_parameter_AD_matches_converged_FD_ladder():
    v = jnp.array([0.4, -0.7, 1.1]) * V0

    def objective(p):
        logs = lambda s: (
            jnp.asarray(LOGS)
            * (1 + p * jnp.array([0.1, 0.3, 0.2, 0.4]) * s / (V0 * V0 + s))
        )
        A, D, _ = evaluate(v, jnp.asarray(TB) * p, logs)
        return (v @ A + jnp.trace(D)) / (V0**2 * 1e4)

    derivative = float(jax.jit(jax.grad(objective))(1.1))
    errors = []
    for h in [1e-2, 1e-3, 1e-4, 1e-5]:
        fd = float((objective(1.1 + h) - objective(1.1 - h)) / (2 * h))
        errors.append(abs(fd - derivative) / abs(derivative))
    assert errors[-1] < 1e-7 and errors[-2] < 1e-6 and errors[1] < errors[0] / 80


@pytest.mark.parametrize("logs", [-1.0, 0.0, np.nan, np.inf])
def test_invalid_logs_return_nan(logs):
    assert all(np.isnan(np.asarray(x)).all() for x in evaluate(jnp.ones(3), logs=logs))


@pytest.mark.parametrize(
    "index,value",
    [
        (0, 0.0),
        (0, -MASS),
        (0, np.inf),
        (2, np.full(4, -1.0)),
        (4, np.full(4, -1.0)),
        (5, np.zeros(4)),
    ],
)
def test_invalid_physical_parameters_return_nan(index, value):
    params = [MASS, CHARGE, MB, QB, NB, TB]
    params[index] = value
    assert all(
        np.isnan(np.asarray(x)).all()
        for x in coefficients(jnp.ones(3), *params, coulomb_logs=LOGS)
    )


def test_integer_constant_log_callable_is_a_valid_scalar_policy():
    v = jnp.array([1.0, 2.0, 3.0]) * V0
    np.testing.assert_allclose(
        evaluate(v, logs=lambda _: 15).diffusion,
        evaluate(v, logs=15.0).diffusion,
        rtol=1e-14,
    )


def test_structural_errors_and_zero_collisions_are_explicit():
    with pytest.raises(ValueError):
        evaluate(jnp.ones(2))
    with pytest.raises(ValueError):
        coefficients(jnp.ones(3), MASS, CHARGE, MB, QB, NB[:2], TB, coulomb_logs=LOGS)
    with pytest.raises(ValueError):
        coefficients(jnp.ones(3), [MASS], CHARGE, MB, QB, NB, TB, coulomb_logs=LOGS)
    with pytest.raises(ValueError):
        evaluate(jnp.ones(3), logs=jnp.ones(3))
    for v in [jnp.zeros(3), jnp.array([1.0, 2.0, 3.0]) * V0]:
        assert all(np.all(np.asarray(x) == 0) for x in evaluate(v, density=np.zeros(4)))
    # Velocity derivatives at zero density are finite. Density derivatives of
    # sqrt(n), and charge derivatives of |q|, are NOT claimed at that boundary.
    derivative = jax.jacfwd(lambda v: evaluate(v, density=np.zeros(4)).noise)(
        jnp.zeros(3)
    )
    assert np.isfinite(np.asarray(derivative)).all()


def test_float32_SI_evaluation_remains_finite_without_charge_underflow():
    for speed in [0.0, 0.1, 1.0, 10.0]:
        args = [
            jnp.asarray(x, dtype=jnp.float32)
            for x in [
                jnp.array([0.6, 0.0, 0.8]) * speed * V0,
                MASS,
                CHARGE,
                MB,
                QB,
                NB,
                TB,
            ]
        ]
        actual = coefficients(*args, coulomb_logs=jnp.asarray(LOGS, dtype=jnp.float32))
        expected = evaluate(jnp.array([0.6, 0.0, 0.8]) * speed * V0)
        assert all(np.isfinite(np.asarray(x)).all() for x in actual)
        np.testing.assert_allclose(
            actual.diffusion,
            expected.diffusion,
            rtol=2e-5,
            atol=1e-5 * float(jnp.max(expected.diffusion)),
        )
