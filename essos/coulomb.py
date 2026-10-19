"""Weak-coupling Coulomb cutoffs with an explicit effective-speed prescription.

The smooth inner cutoff is Landau--Spitzer GMS-1 (``ls_min_interp``):
https://docs.plasmapy.org/en/stable/api/plasmapy.formulary.collisions.coulomb.Coulomb_logarithm.html

For a test particle at speed v in a stationary Maxwellian bath, use
u_eff**2 = E[|v-v_b|**2] = v**2 + 3 T_b/m_b. Evaluating the logarithm at this
RMS relative speed is an approximation, not an exact Maxwellian average of
the velocity-dependent scattering kernel. Screening is static Debye screening
from the supplied species; dynamic screening is outside this model.
Temperatures below are in eV, masses/charges/densities/speeds in SI.
"""

import jax.numpy as jnp

from .constants import ELEMENTARY_CHARGE, EPSILON_0, HBAR


def debye_inverse_squared(density, temperature, charge):
    """Sum n_s q_s**2/(epsilon_0 T_s) over explicitly supplied screeners.

    Absent species contribute zero, including an absent species with T=0.
    Invalid occupied temperatures or negative densities propagate NaN.
    """
    density, temperature, charge = map(jnp.asarray, (density, temperature, charge))
    active = (density != 0) & (charge != 0)
    energy = jnp.where(active, temperature, 1.)*ELEMENTARY_CHARGE
    terms = jnp.where(active, density*charge**2/(EPSILON_0*energy), 0.)
    valid = (density >= 0) & (~active | (temperature > 0))
    return jnp.sum(jnp.where(valid, terms, jnp.nan))


def impact_parameters(mass_a, charge_a, mass_b, charge_b, speed, temperature_b,
                      screening_length):
    """Return (b_min, b_max), using reduced mass and smooth quantum crossover.

    ``screening_length`` is specified independently of test-particle speed.
    The charge product is |q_a q_b|, including unlike charge magnitudes.
    """
    reduced_mass = mass_a*(mass_b/(mass_a+mass_b))
    relative_speed_squared = speed**2+3*temperature_b*ELEMENTARY_CHARGE/mass_b
    b90 = jnp.abs(charge_a*charge_b)/(4*jnp.pi*EPSILON_0*reduced_mass*relative_speed_squared)
    quantum = HBAR/(2*reduced_mass*jnp.sqrt(relative_speed_squared))
    return jnp.hypot(b90, quantum), screening_length


def impact_coulomb_logarithm(mass_a, charge_a, mass_b, charge_b, speed,
                             temperature_b, density, temperature, charge):
    """Smooth effective-speed Landau--Spitzer log for a specified background.

    A vacuum has zero collision contribution and returns zero. A nonpositive
    occupied-background log returns NaN rather than silently clipping an
    invalid weak-coupling model into a positive collision frequency.
    """
    inverse_squared = debye_inverse_squared(density, temperature, charge)
    # Use a finite inactive value so vacuum derivatives do not contain 0*inf.
    length = 1/jnp.sqrt(jnp.where(inverse_squared == 0, 1., inverse_squared))
    bmin, bmax = impact_parameters(mass_a, charge_a, mass_b, charge_b, speed,
                                  temperature_b, length)
    logarithm = jnp.log(bmax)-jnp.log(bmin)
    occupied = jnp.where(logarithm > 0, logarithm, jnp.nan)
    return jnp.where(inverse_squared == 0, 0., occupied)
