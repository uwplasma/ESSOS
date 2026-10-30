"""Trace the same guiding centres with each integrator and compare cost and energy error."""
import os
from time import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from diffrax import Dopri5, Tsit5
from essos.fields import BiotSavart
from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, ONE_EV
from essos.dynamics import Tracing, Particles

tmax, tolerance, nparticles = 1e-4, 1e-7, 4
coils = Coils.from_json(os.path.join(os.path.dirname(__file__), '..', 'input_files',
                                     'ESSOS_biot_savart_LandremanPaulQA.json'))
field = BiotSavart(coils)
R0 = jnp.linspace(1.23, 1.27, nparticles)
particles = Particles(initial_xyz=jnp.array([R0, 0 * R0, 0 * R0]).T, mass=ALPHA_PARTICLE_MASS,
                      charge=ALPHA_PARTICLE_CHARGE, energy=4000 * ONE_EV)

# (label, Tracing keywords). Diffrax Dopri8 is the default; any Diffrax solver can be passed with
# solver=, and integrator= selects the native Dormand-Prince solvers in essos/rk.py.
integrators = [("Diffrax Dopri8 (default)", {}),
               ("Diffrax Dopri5", dict(solver=Dopri5())),
               ("Diffrax Tsit5", dict(solver=Tsit5())),
               ("native Dopri5", dict(integrator="essos_rk45")),
               ("native Dopri8", dict(integrator="essos_dopri8"))]

print(f"{'integrator':26s} {'first call s':>12s} {'second call s':>13s} {'max energy error':>17s}")
for label, options in integrators:
    def trace():
        return Tracing(field=field, model='GuidingCenterAdaptative', particles=particles, maxtime=tmax,
                       times_to_trace=100, atol=tolerance, rtol=tolerance, progress=False, **options)
    t0 = time(); trace().trajectories.block_until_ready(); first = time() - t0
    t0 = time(); tracing = trace(); tracing.trajectories.block_until_ready(); run = time() - t0
    error = jnp.max(jnp.abs(tracing.energy() - particles.energy) / particles.energy)
    print(f"{label:26s} {first:12.2f} {run:13.2f} {error:17.2e}")
