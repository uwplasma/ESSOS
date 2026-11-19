import os
import numpy as np
from booz_xform_jax import Booz_xform
from netCDF4 import Dataset
from essos.boozer import BoozerField, psi0_from_vmec, trace_boozer
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY

# Loss fraction of 3.5 MeV alphas in a finite-beta QH, with and without the covariant
# radial component K of B in the Boozer guiding-centre equations. Needs booz_xform_jax.
wout_file = os.path.join(os.path.dirname(__file__), "../input_files", "wout_QH_simple_finite_scaled.nc")
nparticles = 1000
tmax = 1e-2
s_birth = 0.5
mboz = 16

booz = Booz_xform(verbose=0, mboz=mboz, nboz=mboz)
booz.read_wout(wout_file, flux=False)
booz.run()
with Dataset(wout_file) as nc:
    psi0 = psi0_from_vmec(nc["phi"][-1])

rng = np.random.default_rng(0)
theta, zeta, pitch = rng.uniform(0, 2*np.pi, nparticles), rng.uniform(0, 2*np.pi, nparticles), rng.uniform(-1, 1, nparticles)
speed = np.sqrt(2*FUSION_ALPHA_PARTICLE_ENERGY/ALPHA_PARTICLE_MASS)

lost = {}
for with_K in (True, False):
    field = BoozerField.from_booz_xform(booz, psi0, with_K=with_K)
    lost[with_K] = trace_boozer(field, np.full(nparticles, s_birth), theta, zeta, pitch, speed=speed,
                                mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE, tmax=tmax,
                                n_save=2, progress=False).lost
    print(f"with_K={with_K}: K kept {field.K is not None}, loss fraction {lost[with_K].mean():.3f} "
          f"+- {np.sqrt(lost[with_K].mean()*(1 - lost[with_K].mean())/nparticles):.3f}")
print(f"orbits that change fate with K: {np.sum(lost[True] != lost[False])} of {nparticles}")
