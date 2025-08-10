import os
number_of_processors_to_use = 1 # Parallelization, this should divide nfieldlines
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import Vmec
from essos.coils import Coils_from_json
from essos.dynamics import Tracing

# Input parameters
tmax = 1.e-6
nfieldlines_per_core=8
nfieldlines = number_of_processors_to_use*nfieldlines_per_core
R0 = jnp.linspace(1.21, 1.4, nfieldlines)
trace_tolerance = 1e-8
num_steps = 400

# Load coils and field
wout_file = os.path.join(os.path.dirname(__file__), 'input_files',"wout_QH_simple_scaled.nc")
vmec = Vmec(wout_file)

# Initialize particles
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T

# Trace in ESSOS
time0 = time()
tracing = Tracing(field=vmec, model='FieldLine', initial_conditions=initial_xyz,
                  maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance,rtol=trace_tolerance)
print(f"ESSOS tracing took {time()-time0:.2f} seconds")
trajectories = tracing.trajectories

# Plot trajectories
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
vmec.surface.plot(ax=ax1, show=False)
tracing.plot(ax=ax1, show=False)
tracing.poincare_plot(ax=ax2, show=False, shifts=[0, jnp.pi/2])#, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4])
plt.tight_layout()
plt.show()

# # Save results in vtk format to analyze in Paraview
tracing.to_vtk('trajectories')
vmec.surface.to_vtk('vmec')