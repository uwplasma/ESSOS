import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import BiotSavart
from essos.coils import Coils_from_json
from essos.constants import PROTON_MASS, ONE_EV
from essos.dynamics import Tracing, Particles

# Input parameters
tmax = 3e-4
nparticles_per_core=6
nparticles = number_of_processors_to_use*nparticles_per_core
R0 = jnp.linspace(1.2, 1.33, nparticles)
trace_tolerance = 1e-7
num_steps = 10000
mass=PROTON_MASS
energy=4000*ONE_EV
# pitch angle = jnp.arctan(jnp.sqrt((v^2/vparallel^2) - 1)) * 180 / jnp.pi, in degrees
angle = 45

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
print(f"v_parallel/v_perp = {us}")
particles = Particles(initial_xyz=initial_xyz, initial_vparallel_over_v=jnp.ones(len(R0))*us,mass=mass, energy=energy,field=field)

# Trace in ESSOS
time0 = time()
tracing = Tracing(field=field, model='GuidingCenterAdaptative', particles=particles,
                  maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance)
print(f"ESSOS tracing took {time()-time0:.2f} seconds")

# Plot results
time0 = time()
plotting_data = tracing.poincare_plot(shifts = [0, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4], show=False)
print(f"Poincare plot took {time()-time0:.2f} seconds")
plt.show()

## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')
