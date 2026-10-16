"""SI Cartesian Maxwellian test-particle Landau coefficients.

Explicit pair Coulomb logs are values or a differentiable callable of speed^2
[m^2/s^2]. Temperatures are eV. Returns Ito drift [m/s^2], diffusion D
[m^2/s^3], and a smooth symmetric sigma with sigma@sigma.T=2D. This does not
select a log policy or integrator, or implement transformed GC collisions.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from essos.constants import ELEMENTARY_CHARGE, EPSILON_0


class CartesianCoefficients(NamedTuple):
    ito_drift: jax.Array
    diffusion: jax.Array
    noise: jax.Array


def _shapes(t):
    small = t < 0.0625
    x = jnp.sqrt(jnp.where(small, 1.0, t))
    erf = jax.scipy.special.erf(x)
    direct = (erf - 2 * x / jnp.sqrt(jnp.pi) * jnp.exp(-t)) / (2 * x**3)
    c = (
        2
        / jnp.sqrt(jnp.pi)
        * (
            1 / 3
            + t
            * (
                -1 / 5
                + t
                * (1 / 14 + t * (-1 / 54 + t * (1 / 264 + t * (-1 / 1560 + t / 10800))))
            )
        )
    )
    e = (
        2
        / jnp.sqrt(jnp.pi)
        * (
            1
            + t
            * (
                -1 / 3
                + t
                * (1 / 10 + t * (-1 / 42 + t * (1 / 216 + t * (-1 / 1320 + t / 9360))))
            )
        )
    )
    h = (
        2
        / jnp.sqrt(jnp.pi)
        * (
            -4 / 15
            + t * (4 / 35 + t * (-2 / 63 + t * (2 / 297 + t * (-1 / 858 + t / 5850))))
        )
    )
    return (
        jnp.where(small, c, direct),
        jnp.where(small, e, erf / x),
        jnp.where(small, h, (3 * direct - erf / x) / jnp.where(small, 1.0, t)),
    )


def maxwellian_cartesian_coefficients(
    velocity,
    mass,
    charge,
    bath_mass,
    bath_charge,
    density,
    temperature,
    *,
    coulomb_logs,
):
    """Evaluate native JAX coefficients; invalid numerical inputs return NaNs.

    velocity shape(3,), bath arrays shape(n_species,), scalar test mass/charge.
    Coulomb logs must be positive and finite. A callable takes scalar speed^2
    and returns a pair vector; its regular derivative is INCLUDED in Ito drift.
    Bath masses are kg, charges C, densities m^-3 and temperatures eV.
    The bath is prescribed, isotropic, nonrelativistic and Maxwellian; this is
    a test-particle operator, without a field-particle response. Its particle
    momentum convention follows Hirvijoki et al., Phys. Plasmas 20, 092505
    (2013), Eqs. (1)--(12), doi:10.1063/1.4820951.

    Structural shape mismatches raise ValueError. Zero density/charge means
    zero collisions. The noise is smooth in velocity for a positive active
    bath, including zero velocity. Noise derivatives in density or charge at
    a zero-collision boundary need not exist. Host I/O/conversions are absent
    from numerical evaluation. Float64 is recommended for SI accuracy.
    Noise is generally noncommutative; an integrator must support general
    Ito noise. A Stratonovich solver requires its usual drift conversion.
    """
    v = jnp.asarray(velocity)
    v = v.astype(jnp.result_type(v, float))
    mb, qb, nb, Tb = map(jnp.asarray, (bath_mass, bath_charge, density, temperature))
    if (
        v.shape != (3,)
        or mb.ndim != 1
        or not mb.size
        or any(x.shape != mb.shape for x in (qb, nb, Tb))
    ):
        raise ValueError("Require velocity(3,) and nonempty equal-length bath vectors")
    mass, charge = jnp.asarray(mass), jnp.asarray(charge)
    if mass.ndim != 0 or charge.ndim != 0:
        raise ValueError("Test mass and charge must be scalars")
    s = v @ v
    if callable(coulomb_logs):

        def log_values(speed_squared):
            values = jnp.asarray(coulomb_logs(speed_squared))
            return values.astype(jnp.result_type(values, speed_squared, float))

        logs, slope = jax.jvp(log_values, (s,), (jnp.ones_like(s),))
    else:
        logs = jnp.asarray(coulomb_logs)
        slope = jnp.zeros_like(logs)
    logs = jnp.broadcast_to(logs, mb.shape)
    slope = jnp.broadcast_to(slope, mb.shape)
    temperature_J = Tb * ELEMENTARY_CHARGE
    vt2 = 2 * temperature_J / mb
    vt = jnp.sqrt(vt2)
    c, e, j = _shapes(s / vt2)
    # Keep pair factors independent: no total friction/log derivative shortcut.
    # This ordering avoids squaring SI charges/masses into underflow in float32.
    pref = ((charge / mass) * (qb / (jnp.sqrt(4 * jnp.pi) * EPSILON_0))) ** 2 * nb / vt
    a0 = pref * c
    a = jnp.sum(logs * a0)
    b = jnp.sum(logs * pref * (e - c) / 2)
    h = jnp.sum(logs * pref * j / (2 * vt2))
    alpha = jnp.sum(-(mass + mb) / temperature_J * logs * a0 + 2 * slope * a0)
    outer = jnp.outer(v, v)
    D = b * jnp.eye(3, dtype=b.dtype) + h * outer
    ra = jnp.where(a > 0, jnp.sqrt(2 * jnp.where(a > 0, a, 1.0)), 0.0)
    rb = jnp.where(b > 0, jnp.sqrt(2 * jnp.where(b > 0, b, 1.0)), 0.0)
    denominator = jnp.where(ra + rb > 0, ra + rb, 1.0)
    sigma = rb * jnp.eye(3, dtype=b.dtype) + 2 * h / denominator * outer
    valid = (
        jnp.all(jnp.isfinite(v))
        & jnp.isfinite(s)
        & jnp.isfinite(mass)
        & (mass > 0)
        & jnp.isfinite(charge)
        & jnp.all(jnp.isfinite(mb) & (mb > 0))
        & jnp.all(jnp.isfinite(qb))
        & jnp.all(jnp.isfinite(nb) & (nb >= 0))
        & jnp.all(jnp.isfinite(Tb) & (Tb > 0))
        & jnp.all(jnp.isfinite(logs) & (logs > 0))
        & jnp.all(jnp.isfinite(slope))
    )
    return CartesianCoefficients(
        jnp.where(valid, alpha * v, jnp.nan),
        jnp.where(valid, D, jnp.nan),
        jnp.where(valid, sigma, jnp.nan),
    )
