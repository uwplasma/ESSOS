import os
number_of_processors_to_use = 8 # Parallelization, this should divide nfieldlines
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import Vmec
from essos.coils import Coils_from_json
from essos.dynamics import Tracing

# Input parameters
tmax = 4000
nfieldlines = number_of_processors_to_use*1
R0 = jnp.linspace(16, 16.5, nfieldlines)
trace_tolerance = 1e-8
num_steps = tmax

# Load coils and field
wout_file = os.path.join(os.path.dirname(__file__), 'input_files','wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc')
vmec = Vmec(wout_file)

# Initialize particles
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T

# Trace in ESSOS
time0 = time()
tracing = Tracing(field=vmec, model='FieldLine', initial_conditions=initial_xyz,
                  maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
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