import os
number_of_processors_to_use = 4 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import BiotSavart
from essos.coils import Coils_from_json
from essos.constants import PROTON_MASS, ONE_EV
from essos.dynamics import Tracing, Particles

# Input parameters
tmax = 1e-3
nparticles = 4
R0 = jnp.linspace(1.23, 1.27, nparticles)
trace_tolerance = 1e-7
num_steps = 30000
mass=PROTON_MASS
energy=4000*ONE_EV
# pitch angle
angle = .1

# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), 'input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils_from_json(json_file)
field = BiotSavart(coils)

# Initialize particles
Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T

## set the particles to all have the same velocity 
us = jnp.sqrt(1/((jnp.tan(angle*jnp.pi/180)**2)+1))
particles = Particles(initial_xyz=initial_xyz, initial_vparallel_over_v=jnp.ones(len(R0))*us,mass=mass, energy=energy,field=field)

# Trace in ESSOS
time0 = time()
tracing = Tracing(field=field, model='GuidingCenter', particles=particles,
                  maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
print(f"ESSOS tracing took {time()-time0:.2f} seconds")
trajectories = tracing.trajectories

# Plot Poincare plot
from essos.plot import poincare_plot
poincare_plot(tracing, shift = jnp.pi)


## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')