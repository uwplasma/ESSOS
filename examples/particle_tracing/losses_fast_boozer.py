"""Fast alpha-particle losses: Boozer guiding centres, fixed step (seconds).

Teaches the quickest loss estimate in ESSOS, the same one behind ``vmex --trace``:
transform a VMEC equilibrium to Boozer coordinates, keep only the large |B| modes,
and advance guiding centres with a fixed-step RK4 in ``(s, theta, zeta, v_par)``.
A particle is lost the first time it reaches s = 1, so the loss fraction needs
no boundary classifier and only ~100 saved points per orbit.

Use it to scan configurations and to size a study. For adaptive Dopri8, collisions
and energy diagnostics see ``losses_research_vmec.py``.

Needs booz_xform_jax. Runtime: about 15 s on a CPU, mostly compilation.
"""
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from booz_xform_jax import Booz_xform
from essos.boozer import BoozerField, psi0_from_vmec, trace_boozer
from essos.fields import Vmec
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY

# Parameters
wout_file = os.path.join(os.path.dirname(__file__), "..", "input_files", "wout_n3are_R7.75B5.7.nc")  # ARIES-CS
nparticles = 256        # alphas born on one flux surface, uniform in angles and pitch
s0 = 0.25               # birth surface, s = psi/psi_edge
tmax = 2e-3             # horizon [s]
timestep = 1.25e-7      # fixed RK4 step [s]; refine to check convergence
n_save = 101            # saved points per orbit (losses are timed at the step, not at the saves)
mboz = nboz = 32        # Boozer resolution
mode_tolerance = 1e-4   # drop |B| modes below this fraction of the largest one
seed = 0
# Advanced: method="dopri8" (default of trace_boozer), "adaptive8" with tolerance=1e-8, or species=... for collisions

# Equilibrium -> Boozer coordinates -> |B| spline over the kept modes
vmec = Vmec(wout_file)
vmec.nc.set_auto_mask(False)
booz = Booz_xform(verbose=0, mboz=mboz, nboz=nboz)
booz.read_wout(wout_file, flux=False)
field = BoozerField.from_booz_xform(booz, psi0=psi0_from_vmec(vmec.nc.variables["phi"][-1]),
                                    mode_tolerance=mode_tolerance)
print(f"Boozer field with {field.xm.size} modes, nfp = {field.nfp}")

# Births: uniform Boozer angles and pitch v_par/v on one surface, 3.5 MeV alphas
rng = np.random.default_rng(seed)
s = np.full(nparticles, s0)
theta = rng.uniform(0, 2 * np.pi, nparticles)
zeta = rng.uniform(0, 2 * np.pi / field.nfp, nparticles)
pitch = rng.uniform(-1, 1, nparticles)
speed = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)

# Trace; the first call includes JIT compilation
time0 = time()
trace = trace_boozer(field, s, theta, zeta, pitch, speed=speed, mass=ALPHA_PARTICLE_MASS,
                     charge=ALPHA_PARTICLE_CHARGE, tmax=tmax, timestep=timestep, n_save=n_save,
                     method="rk4", progress=False)
elapsed = time() - time0

# Results
lost = trace.lost.mean()
print(f"{nparticles} alphas: loss fraction {100 * lost:.1f}% +- {100 * np.sqrt(lost * (1 - lost) / nparticles):.1f}% "
      f"at {1e3 * tmax:.0f} ms, traced in {elapsed:.1f} s")
print(f"max energy error {trace.energy_error.max():.1e}")
plt.plot(1e3 * trace.times, 100 * trace.loss_fractions())
plt.xlabel("Time (ms)")
plt.ylabel("Loss fraction (%)")
plt.title(f"{nparticles} alphas born at s = {s0}")
plt.tight_layout()
plt.show()
