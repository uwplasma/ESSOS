"""Research-grade alpha-particle losses in VMEC coordinates: adaptive Dopri8, collisions, energy.

Teaches the slower, more informative loss workflow. Guiding centres move in VMEC
flux coordinates (s, theta, phi) with an error-controlled Dopri8 (``atol``, ``rtol``)
and are lost on reaching s = 1 (or a ``SurfaceClassifier`` boundary, see below).
Two runs from the same births are compared:

* collisionless, with the relative energy error of every orbit as the accuracy check;
* with Monte Carlo collisions on electrons and deuterium (``GuidingCenterCollisionsMuFixed``),
  which also records the energy and position of each lost alpha.

For a seconds-long scan of the same quantity see ``losses_fast_boozer.py``.
Runtime: about 1-2 minutes on a CPU, mostly compilation.
"""
import os
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from essos.fields import Vmec
from essos.constants import (ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY,
                             PROTON_MASS, ELECTRON_MASS)
from essos.dynamics import Tracing, Particles
from essos.background_species import BackgroundSpecies

# Parameters
wout_file = os.path.join(os.path.dirname(__file__), "..", "input_files", "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
nparticles = 24
s0 = 0.6                # birth surface, s = psi/psi_edge
tmax = 1e-4             # horizon [s]
times_to_trace = 500    # saved points per orbit
atol = rtol = 1e-8      # Dopri8 error tolerance (collisionless run)
timestep = 1e-8         # collisional run: fixed step [s]; the collisionless first step is derived from it
T0, n0 = 17.8e3, 4.21e20  # background temperature [eV] and density [m^-3] of electrons and deuterium
seed = 0
# Advanced: Tracing(..., boundary=SurfaceClassifier(vmec.surface, h=0.1)) takes the loss from a surface classifier;
# model='GuidingCenterCollisionsMuAdaptative' controls the collisional step; electric_field=Electric_field_flux(...) adds E x B.

# Equilibrium and births: random VMEC angles and pitch on one surface
vmec = Vmec(wout_file)
rng = np.random.default_rng(seed)
initial_xyz = jnp.array([np.full(nparticles, s0), rng.uniform(0, 2 * np.pi, nparticles),
                         rng.uniform(0, 2 * np.pi / vmec.nfp, nparticles)]).T
pitch = jnp.array(rng.uniform(-1, 1, nparticles))
particles = Particles(initial_xyz=initial_xyz, initial_vparallel_over_v=pitch, mass=ALPHA_PARTICLE_MASS,
                      charge=ALPHA_PARTICLE_CHARGE, energy=FUSION_ALPHA_PARTICLE_ENERGY, field=vmec)

# Collisionless adaptive guiding centre
time0 = time()
free = Tracing(field=vmec, model="GuidingCenterAdaptative", particles=particles, maxtime=tmax, timestep=timestep,
               times_to_trace=times_to_trace, atol=atol, rtol=rtol)
time_free = time() - time0

# Same births with collisions on a thermal electron + deuterium background
species = BackgroundSpecies(number_species=2, mass_array=jnp.array([ELECTRON_MASS / PROTON_MASS, 2.0]),
                            charge_array=jnp.array([-1.0, 1.0]), n_array=jnp.array([n0, n0]),
                            T_array=jnp.array([T0, T0]))
time0 = time()
coll = Tracing(field=vmec, model="GuidingCenterCollisionsMuFixed", particles=particles, maxtime=tmax,
               timestep=timestep, times_to_trace=times_to_trace, species=species, tag_gc=1.0)
time_coll = time() - time0

# Results
energy_error = jnp.abs(free.energy() - particles.energy) / particles.energy
print(f"Collisionless: loss {100 * free.loss_fractions[-1]:.1f}% in {time_free:.1f} s, "
      f"max energy error {float(jnp.nanmax(energy_error)):.1e}")
print(f"Collisions:    loss {100 * coll.loss_fractions[-1]:.1f}% in {time_coll:.1f} s")
lost = np.asarray(coll.lost_times) >= 0
if lost.any():
    print(f"Collisional losses carry on average {float(np.mean(np.asarray(coll.lost_energies)[lost])):.2e} J "
          f"(birth energy {float(particles.energy):.2e} J)")
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].plot(free.times, free.loss_fractions, label="collisionless")
ax[0].plot(coll.times, coll.loss_fractions, "--", label="collisions")
ax[0].set(xlabel="Time (s)", ylabel="Loss fraction")
ax[0].legend()
ax[1].semilogy(free.times[2:], energy_error[:, 2:].T, lw=0.6)
ax[1].set(xlabel="Time (s)", ylabel="Relative energy error (collisionless)")
ax[2].plot(coll.times, coll.trajectories[:, :, 0].T, lw=0.6)
ax[2].set(xlabel="Time (s)", ylabel="s (collisions)")
plt.tight_layout()
plt.show()
