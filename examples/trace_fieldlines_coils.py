import os
number_of_processors_to_use = 12 # Parallelization, this should divide nfieldlines
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import BiotSavart
from essos.coils import Coils_from_json
from essos.dynamics import Tracing

# Input parameters
tmax = 5000
nfieldlines = number_of_processors_to_use
R0 = jnp.linspace(1.21, 1.45, nfieldlines)
trace_tolerance = 1e-9
num_steps = tmax*2

# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), 'input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils_from_json(json_file)
field = BiotSavart(coils)

# Initialize particles
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T

# Trace in ESSOS
time0 = time()
tracing = Tracing(field=field, model='FieldLine', initial_conditions=initial_xyz,
                  maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
print(f"ESSOS tracing took {time()-time0:.2f} seconds")
trajectories = tracing.trajectories

# Plot trajectories
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
coils.plot(ax=ax1, show=False)
tracing.plot(ax=ax1, show=False)
tracing.poincare_plot(ax=ax2, show=False, shifts=[0, jnp.pi/2])#, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4])
plt.tight_layout()
plt.show()

# # Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')