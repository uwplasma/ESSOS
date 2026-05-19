import os
number_of_processors_to_use = 1 # Parallelization, this should divide nfieldlines
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
from jax import block_until_ready
import matplotlib.pyplot as plt
from essos.fields import BiotSavart
from essos.coils import Coils_from_json
from essos.dynamics import Tracing

# Input parameters
tmax = 1000
nfieldlines_per_core=3
nfieldlines = number_of_processors_to_use*nfieldlines_per_core
R0 = jnp.linspace(1.21, 1.4, nfieldlines)
trace_tolerance = 1e-8
num_steps = 6000

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
tracing = block_until_ready(Tracing(field=field, model='FieldLineAdaptative', initial_conditions=initial_xyz,
                  maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance))
print(f"ESSOS tracing took {time()-time0:.2f} seconds")
trajectories = tracing.trajectories

# Plot trajectories
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
#coils.plot(ax=ax1, show=False)
tracing.plot(ax=ax1, show=False)
tracing.poincare_plot(ax=ax2, show=False, shifts=[0, jnp.pi/2])#, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4])
plt.tight_layout()
plt.show()
# # Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')