from functools import partial
import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jax import jit
from essos.constants import ELEMENTARY_CHARGE, EPSILON_0, PROTON_MASS, SPEED_OF_LIGHT, HBAR



###This module uses some functions adapted from NEOPAX/JAX-MONKES
JOULE_PER_EV = ELEMENTARY_CHARGE


class BackgroundSpecies():
    """Maxwellian background species for the collision operators.

    ``n_array`` [m^-3] and ``T_array`` [eV] hold one value per species, or,
    with ``radial_grid``, one profile per species sampled on that grid of the
    first coordinate of the traced points (``s`` for a VMEC field), shape
    ``(number_species, len(radial_grid))``. Profiles are interpolated
    linearly and held constant beyond the ends of the grid.

    ``coulomb_log_model="nrl"`` preserves the original pairwise NRL formulas.
    ``"impact"`` selects smooth Landau--Spitzer cutoffs at the RMS relative
    speed, with static Debye screening (see :func:`impact_coulomb_logarithm`). Screening
    uses these background species unless ``screening_species`` supplies a
    separate BackgroundSpecies, e.g. electrons which screen without adding
    electron collisions. No screening electron population is inferred for
    the impact model. Both tracers and both time solvers use this selection.
    """
    def __init__(self, number_species, mass_array, charge_array, n_array, T_array, radial_grid=None,
                 *, coulomb_log_model="nrl", screening_species=None):
        if coulomb_log_model not in ("nrl", "impact"):
            raise ValueError("coulomb_log_model must be 'nrl' or 'impact'")
        if screening_species is not None and coulomb_log_model != "impact":
            raise ValueError("screening_species is only used by the impact Coulomb-log model")
        self.coulomb_log_model = coulomb_log_model
        self.screening_species = screening_species
        self.number_species = number_species
        self.species_indeces = jnp.arange(number_species)
        self.temperature = T_array
        self.density = n_array
        self.radial_grid = None if radial_grid is None else jnp.asarray(radial_grid)
        self.mass=mass_array*PROTON_MASS
        self.charge=charge_array*ELEMENTARY_CHARGE

    def _profile(self, values, species_index, points):
        if self.radial_grid is None:
            return values[species_index]
        return jnp.interp(points[0], self.radial_grid, values[species_index])

    @partial(jit, static_argnames=['self'])
    def get_temperature(self,species_index, points):
        return self._profile(self.temperature, species_index, points)

    @partial(jit, static_argnames=['self'])
    def get_density(self,species_index, points):
        return self._profile(self.density, species_index, points)

    @partial(jit, static_argnames=['self'])
    def get_v_thermal(self,species_index, points):
        m=self.mass[species_index]
        T=self.get_temperature(species_index, points)
        return jnp.sqrt(2*T * JOULE_PER_EV/m)


@partial(jit, static_argnames=['species'])
def gamma_ab(ma: float, ea: float, species_b: int,v: float, points, species: BackgroundSpecies) -> float:
    """Prefactor for pairwise collisionality."""
    lnlambda = coulomb_logarithm(ma, ea, species_b, v, points, species)
    eb = species.charge[species_b]
    return ea**2 * eb**2 * lnlambda / (4 * jnp.pi * EPSILON_0**2 * ma**2)

def dlog_coulomb_dv(ma, ea, species_b, v, points, species):
    """d(ln lnLambda)/dv: the Coulomb logarithm depends on the test-particle speed."""
    ln, slope = jax.jvp(lambda u: coulomb_logarithm(ma, ea, species_b, u, points, species), (v,), (jnp.ones_like(v),))
    # The impact model returns zero in a vacuum, where all rates vanish.
    return slope/jnp.where(ln == 0, 1., ln)


@partial(jit, static_argnames=['species'])
def nu_D_ab(ma: float, ea: float,species_b: int,v:float, points,species: BackgroundSpecies) -> float:
    """Decay rate of the l=1 pitch moment; the angular operator is nu_D L/2."""
    nb = species.get_density(species_b,points)
    vtb = species.get_v_thermal(species_b,points)
    prefactor = gamma_ab(ma,ea, species_b, v,points,species) * nb 
    erf_part = (jax.scipy.special.erf(v / vtb) - chandrasekhar(v / vtb))/ v**3
    return prefactor * erf_part


@partial(jit, static_argnames=['species'])
def d_nu_D_ab(ma: float, ea: float,species_b: int,v:float, points,species: BackgroundSpecies) -> float:
    """Speed derivative of the l=1 deflection rate."""
    nb = species.get_density(species_b,points)
    vtb = species.get_v_thermal(species_b,points)
    prefactor = gamma_ab(ma,ea, species_b, v,points,species) * nb 
    erf_part = (d_erf(v/vtb)-d_chandrasekhar(v/vtb))/vtb/v**3-3.*(jax.scipy.special.erf(v / vtb) - chandrasekhar(v / vtb))/ v**4
    dlog = dlog_coulomb_dv(ma, ea, species_b, v, points, species)
    return prefactor*(erf_part + dlog*(jax.scipy.special.erf(v / vtb) - chandrasekhar(v / vtb))/ v**3)

@partial(jit, static_argnames=['species'])
def nu_par_ab(ma: float, ea: float,species_b: int,v:float, points,species: BackgroundSpecies) -> float:
    """Parallel collision frequency"""
    nb = species.get_density(species_b,points)
    vtb = species.get_v_thermal(species_b,points)
    return (
        2 * gamma_ab(ma,ea, species_b, v,points,species) * nb * chandrasekhar(v / vtb) / v**3
    )

@partial(jit, static_argnames=['species'])
def d_nu_par_ab(ma: float, ea: float,species_b: int,v:float, points,species: BackgroundSpecies):
    """d(Parallel collision frequency)/ dv"""
    nb = species.get_density(species_b,points)
    vtb = species.get_v_thermal(species_b,points)
    return (
        2 *  gamma_ab(ma,ea, species_b,v, points,species) * nb  * (d_chandrasekhar(v / vtb)*v/vtb-3.*chandrasekhar(v / vtb)
                                                                + v*dlog_coulomb_dv(ma, ea, species_b, v, points, species)*chandrasekhar(v / vtb))/ v**4
    )

@partial(jit, static_argnames=['species'])
def nu_s_ab(ma: float, ea: float,species_b: int,v:float, points,species: BackgroundSpecies) -> float:
    """Slowing collision frequency"""
    nb = species.get_density(species_b,points)
    vtb = species.get_v_thermal(species_b,points)
    mb = species.mass[species_b]   
    #Tb = species.get_temperature(species_b,points) 
    Tb = (mb*vtb**2) / 2.  
    return (
        gamma_ab(ma,ea, species_b, v, points,species)* nb * (ma+mb)/Tb * chandrasekhar(v / vtb) /v 
    )*(ma/(ma+mb))


# Smooth Landau-Spitzer GMS-1 (ls_min_interp) impact cutoffs.
# RMS relative speed squared v**2+3*T_b/m_b approximates a Maxwellian average.
# Screening uses the supplied species; dynamic screening is excluded.
# Temperatures are in eV; masses, charges, densities and speeds use SI.
# https://docs.plasmapy.org/en/stable/api/plasmapy.formulary.collisions.coulomb.Coulomb_logarithm.html

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


@partial(jit, static_argnames=['species'])
def coulomb_logarithm(ma:float, ea: float, species_b: int, v: float, points, species: BackgroundSpecies) -> float:
    """Selected pairwise Coulomb logarithm; NRL remains the default."""
    if species.coulomb_log_model == "impact":
        screening = species if species.screening_species is None else species.screening_species
        n = jnp.stack([screening.get_density(i, points) for i in range(screening.number_species)])
        T = jnp.stack([screening.get_temperature(i, points) for i in range(screening.number_species)])
        return impact_coulomb_logarithm(ma, ea, species.mass[species_b], species.charge[species_b],
            v, species.get_temperature(species_b, points), n, T, screening.charge)
    return nrl_coulomb_logarithm(ma, ea, species_b, v, points, species)


def nrl_coulomb_logarithm(ma:float, ea: float, species_b: int, v: float, points, species: BackgroundSpecies) -> float:
    """NRL Plasma Formulary (2019, p. 34) Coulomb logarithm of a test particle
    (mass ``ma``, charge ``ea``, speed ``v``) on background species ``species_b``.

    Electron-electron uses eq. (a) and electron-ion eq. (b), with the density and
    temperature of the background electrons. Ion-ion uses the mixed thermal
    eq. (c) with the test-particle temperature m_a v^2/3, or, when the test
    particle is faster than the rms speed of b, the fast-ion (beam) eq. (d).
    Without background electrons a quasineutral electron density at the
    test-particle temperature is assumed. Densities in cm^-3, temperatures in eV.
    """
    e, mp = ELEMENTARY_CHARGE, PROTON_MASS
    safe = lambda n: jnp.where(n == 0, 1.0, n)  # absent species: rate 0, finite log
    n = jnp.stack([species.get_density(s, points) for s in range(species.number_species)])*1e-6
    T = jnp.stack([species.get_temperature(s, points) for s in range(species.number_species)])
    Z, is_e = jnp.abs(species.charge)/e, species.mass < 0.01*mp
    mb, nb, Tb, Zb, Za = species.mass[species_b], n[species_b], T[species_b], Z[species_b], jnp.abs(ea)/e
    a_e, b_e, Ta = ma < 0.01*mp, is_e[species_b], ma*v**2/(3*e)
    ne = jnp.where(jnp.any(is_e), jnp.sum(jnp.where(is_e, n, 0.)), jnp.sum(jnp.where(is_e, 0., Z*n)))
    Te = jnp.where(jnp.any(is_e), jnp.sum(jnp.where(is_e, T, 0.))/jnp.maximum(jnp.sum(is_e), 1), Ta)
    ee = 23.5 - jnp.log(safe(ne)**0.5*Te**-1.25) - jnp.sqrt(1e-5 + (jnp.log(Te) - 2)**2/16)
    Zi = jnp.where(a_e, Zb, Za)
    ei = jnp.where(Te < 10*Zi**2, 23 - jnp.log(safe(ne)**0.5*Zi*Te**-1.5), 24 - jnp.log(safe(ne)**0.5/Te))
    mua, mub = ma/mp, mb/mp
    na = jnp.sum(jnp.where(jnp.isclose(species.mass, ma, atol=0) & jnp.isclose(species.charge, ea, atol=0), n, 0.))
    screening = safe(nb*Zb**2/Tb + jnp.where(na > 0, na*Za**2/Ta, 0.))
    ii_thermal = 23 - jnp.log(Za*Zb*(mua + mub)/(mua*Tb + mub*Ta)*screening**0.5)
    ii_beam = 43 - jnp.log(Za*Zb*(mua + mub)/(mua*mub*(v/SPEED_OF_LIGHT)**2)*(safe(ne)/Te)**0.5)
    ii = jnp.where(v**2 > 3*Tb*e/mb, ii_beam, ii_thermal)
    return jnp.where(a_e & b_e, ee, jnp.where(a_e | b_e, ei, ii))


def chandrasekhar(x: jax.Array) -> jax.Array:
    """Chandrasekhar function."""
    small = jnp.abs(x) < 0.25
    denominator = jnp.where(small, 1.0, x)
    xs = jnp.where(small, x, 0.0)
    series = 2*xs/jnp.sqrt(jnp.pi)*(1/3 + xs*xs*(-1/5 + xs*xs*(1/14 + xs*xs*(-1/54 + xs*xs*(1/264 + xs*xs*(-1/1560 + xs*xs/10800))))))
    direct = (
        jax.scipy.special.erf(x) - 2 * x / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))
    ) / (2 * denominator**2)
    return jnp.where(small, series, direct)

def d_chandrasekhar(x: jax.Array) -> jax.Array:
    denominator = jnp.where(x == 0, 1.0, x)
    return jnp.where(x == 0, 2/(3*jnp.sqrt(jnp.pi)),
                     2/jnp.sqrt(jnp.pi)*jnp.exp(-(x**2)) - 2/denominator*chandrasekhar(x))
    
    
def d_erf(x: jax.Array) -> jax.Array:
    return 2 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))    
