"""Slowing down of 10 keV protons in a 1 keV electron bath with the native split solver.

Ions with v_ti << v << v_te lose energy at the Spitzer rate 1/tau_s, tau_s = m_i/(m_e nu_e), and the
mean energy relaxes to 3T/2: <E>(t) = 3T/2 + (E0 - 3T/2) exp(-2 t/tau_s) (Helander & Sigmar, ch. 3).
Run time: about 10 s on a laptop CPU.
"""
import jax.numpy as jnp
import numpy as np
from essos.background_species import BackgroundSpecies, coulomb_logarithm
from essos.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, EPSILON_0, ONE_EV, PROTON_MASS
from essos.dynamics import Particles
from essos.split import trace_split


class UniformField:  # B = 1 T along z
    def B_contravariant(self, x): return jnp.array([0., 0., 1.])
    B_covariant = B_contravariant
    def AbsB(self, x): return 1.
    def dAbsB_by_dX(self, x): return jnp.zeros(3)
    curl_b = kappa = dAbsB_by_dX
    def sqrtg(self, x): return 1.
    def to_xyz(self, x): return x


TEMPERATURE_EV, DENSITY, ENERGY_EV, N_PARTICLES, N_TAU = 1000., 1e20, 10e3, 2000, 6

field = UniformField()
electrons = BackgroundSpecies(1, jnp.array([ELECTRON_MASS/PROTON_MASS]), jnp.array([-1.]), jnp.array([DENSITY]),
                              jnp.array([TEMPERATURE_EV]))
birth = Particles(jnp.tile(jnp.array([1., 0., 0.]), (N_PARTICLES, 1)), mass=PROTON_MASS, charge=ELEMENTARY_CHARGE,
                  energy=ENERGY_EV*ONE_EV, initial_vparallel_over_v=jnp.linspace(-.9, .9, N_PARTICLES), field=field)
kelvin = TEMPERATURE_EV*ONE_EV
log = float(coulomb_logarithm(PROTON_MASS, ELEMENTARY_CHARGE, 0, np.sqrt(2*ENERGY_EV*ONE_EV/PROTON_MASS), jnp.zeros(3), electrons))
nu_e = 4*np.sqrt(2*np.pi)*DENSITY*ELEMENTARY_CHARGE**4*log/(3*(4*np.pi*EPSILON_0)**2*np.sqrt(ELECTRON_MASS)*kelvin**1.5)
rate = ELECTRON_MASS/PROTON_MASS*nu_e
times = jnp.linspace(0., N_TAU/rate, 9)
out = trace_split(field, birth, electrons, tmax=float(times[-1]), timestep=float(times[-1])/300, times=times,
                  orbit=False, adaptive=False, speed_floor=1.)
assert (out.status == 0).all()
mean = np.mean(.5*PROTON_MASS*out.ys[..., 3]**2 + out.ys[..., 4], axis=0)
spitzer = 1.5*kelvin + (ENERGY_EV*ONE_EV - 1.5*kelvin)*np.exp(-2*rate*np.asarray(times))
for t, m, a in zip(np.asarray(times)*rate, mean/ONE_EV, spitzer/ONE_EV):
    print(f"t/tau_s = {t:4.1f}   <E> = {m:8.1f} eV   Spitzer = {a:8.1f} eV")
