from functools import partial
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import equinox as eqx
import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jax import jit
import jax.numpy as jnp
from essos.constants import BOLTZMANN, ELEMENTARY_CHARGE, EPSILON_0, HBAR, PROTON_MASS



###This module uses some functions adapted from NEOPAX/JAX-MONKES
JOULE_PER_EV = 11606 * BOLTZMANN
EV_PER_JOULE = 1 / JOULE_PER_EV


class BackgroundSpecies():
    def __init__(self, number_species, mass_array, charge_array, n_array, T_array):
        self.number_species = number_species
        self.species_indeces = jnp.arange(number_species)
        self.temperature = T_array   #Array of lambdas for now
        self.density = n_array         #Array of lambdas for now 
        self.mass=mass_array*PROTON_MASS
        self.charge=charge_array*ELEMENTARY_CHARGE

    @partial(jit, static_argnames=['self'])
    def get_temperature(self,species_index, points):
        x=points[0]
        y=points[1]
        z=points[2]
        return self.temperature[species_index]#(x,y,z)
    

    @partial(jit, static_argnames=['self'])
    def get_density(self,species_index, points):
        x=points[0]
        y=points[1]
        z=points[2]
        return self.density[species_index]#self.density[species_index](x,y,z)
    

    @partial(jit, static_argnames=['self'])
    def get_v_thermal(self,species_index, points):
        m=self.mass[species_index]
        x=points[0]
        y=points[1]
        z=points[2]
        T=self.temperature[species_index]#(x,y,z)
        return jnp.sqrt(2*T * JOULE_PER_EV/m)
 


@partial(jit, static_argnames=['species'])
def gamma_ab(ma: float, ea: float, species_b: int,vth_a: float, points, species: BackgroundSpecies) -> float:
    """Prefactor for pairwise collisionality."""
    lnlambda = coulomb_logarithm(ma, ea, species_b, vth_a, points, species)
    eb = species.charge[species_b]
    return ea**2 * eb**2 * lnlambda / (4 * jnp.pi * EPSILON_0**2 * ma**2)

@partial(jit, static_argnames=['species'])
def nu_D_ab(ma: float, ea: float,species_b: int,v:float, points,species: BackgroundSpecies) -> float:
    """Deflection collision frequency"""
    nb = species.get_density(species_b,points)
    vtb = species.get_v_thermal(species_b,points)
    prefactor = gamma_ab(ma,ea, species_b, v,points,species) * nb 
    erf_part = (jax.scipy.special.erf(v / vtb) - chandrasekhar(v / vtb))/ v**3
    return prefactor * erf_part*2.


@partial(jit, static_argnames=['species'])
def d_nu_D_ab(ma: float, ea: float,species_b: int,v:float, points,species: BackgroundSpecies) -> float:
    """Deflection collision frequency"""
    nb = species.get_density(species_b,points)
    vtb = species.get_v_thermal(species_b,points)
    prefactor = gamma_ab(ma,ea, species_b, v,points,species) * nb 
    erf_part = (d_erf(v/vtb)-d_chandrasekhar(v/vtb))/vtb/v**3-3.*(jax.scipy.special.erf(v / vtb) - chandrasekhar(v / vtb))/ v**4
    return 2.*prefactor*erf_part

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
        2 *  gamma_ab(ma,ea, species_b,v, points,species) * nb  * (d_chandrasekhar(v / vtb)*v/vtb-3.*chandrasekhar(v / vtb))/ v**4
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

@partial(jit, static_argnames=['species'])
def coulomb_logarithm(ma:float, ea: float, species_b: int, vth_a: float, points, species: BackgroundSpecies) -> float:
    """Coulomb logarithm for collisions between species a and b.
    Parameters
    ----------
    maxwellian_a : LocalMaxwellian
        Distribution function of primary species.
    maxwellian_b : LocalMaxwellian
        Distribution function of background species.
    Returns
    -------
    log(lambda) : float
    """
    ##bmin, bmax =   impact_parameter(ma, ea, species_b, vth_a, points, species)
    ##return jnp.log(bmax / bmin)
    #lnL = 25.3 + 1.15*jnp.log10(species.temperature[0,r_index]**2/species.density[0,r_index])  
    lnL = 32.2 + 1.15*jnp.log10(species.get_temperature(0,points)**2/species.get_density(0,points)) 
    #32.2+1.15*alog10(temp(1)**2/density(1))
    return lnL

@partial(jit, static_argnames=['species'])
def impact_parameter(ma:float, ea: float, species_b: int, vth_a: float, points, species: BackgroundSpecies)-> float:
    """Impact parameters for classical Coulomb collision."""
    bmin = jnp.maximum(
        impact_parameter_perp(ma, ea, species_b, vth_a, points, species),
        debroglie_length(ma, species_b, vth_a, points, species),
    )
    bmax = debye_length(points, species)
    return bmin, bmax

@partial(jit, static_argnames=['species'])
def impact_parameter_perp(ma:float, ea: float, species_b: int, vth_a: float, points, species: BackgroundSpecies) -> float:
    """Distance of the closest approach for a 90° Coulomb collision."""
    mb=species.mass[species_b]
    m_reduced = ma*mb / (ma + mb)
    v_th = jnp.sqrt( vth_a * species.get_v_thermal(species_b,points))
    return ( ea*ea / (4 * jnp.pi * EPSILON_0 * m_reduced * v_th**2) )

@partial(jit, static_argnames=['species'])
def debroglie_length(ma:float, species_b: int, vth_a: float, points, species: BackgroundSpecies) -> float:
    """Thermal DeBroglie wavelength."""
    mb=species.mass[species_b]
    m_reduced = ma*mb / (ma + mb)
    v_th = jnp.sqrt( vth_a * species.get_v_thermal(species_b,points))
    return HBAR / (2 * m_reduced * v_th)

@partial(jit, static_argnames=['species'])
def debye_length(points, species: BackgroundSpecies ) -> float:
    """Scale length for charge screening."""
    den = 0
    for m in range(species.number_species):
        den += species.get_density(m,points)/ (species.get_temperature(m,points) * JOULE_PER_EV) * species.charge[m]**2
    #den=jnp.sum(species.density[:,r_index] / (species.temperature[:,r_index] * JOULE_PER_EV) * species.charge[:]**2)
    return jnp.sqrt(EPSILON_0 / den)


def chandrasekhar(x: jax.Array) -> jax.Array:
    """Chandrasekhar function."""
    return (
        jax.scipy.special.erf(x) - 2 * x / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))
    ) / (2 * x**2)

def d_chandrasekhar(x: jax.Array) -> jax.Array:
    return 2 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2)) - 2 / x * chandrasekhar(x)
    
    
def d_erf(x: jax.Array) -> jax.Array:
    return 2 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))    
