import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
from jax import block_until_ready
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from essos.fields import BiotSavart
from essos.coils import Coils_from_json
from essos.constants import PROTON_MASS, ONE_EV, ELEMENTARY_CHARGE
from essos.dynamics import Tracing, Particles
# import integrators
import diffrax

# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), '../examples/input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils_from_json(json_file)
field = BiotSavart(coils)

# Particle parameters
nparticles = number_of_processors_to_use
mass=PROTON_MASS
energy=5000*ONE_EV
cyclotron_frequency = ELEMENTARY_CHARGE*0.3/mass
print("cyclotron period:", 1/cyclotron_frequency)

# Particles initialization
initial_xyz=jnp.array([[1.23, 0, 0]])
particles = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy, initial_vparallel_over_v=[0.9], phase_angle_full_orbit=0)

# Tracing parameters
tmax = 1e-4
dt = 5e-8
num_steps = int(tmax/dt)

fig, ax = plt.subplots(figsize=(9, 6))

for method in ['Tsit5', 'Dopri5', 'Dopri8']:
    energies = []
    tracing_times = []
    for trace_tolerance in [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]:
        time0 = time()
        tracing = Tracing(field=field, model='GuidingCenter', method=getattr(diffrax, method), particles=particles,
                          maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
        block_until_ready(tracing.trajectories)
        tracing_times += [time() - time0]
        
        print(f"Tracing with adaptative {method} and tolerance {trace_tolerance:.0e} took {tracing_times[-1]:.2f} seconds")
        
        energies += [jnp.max(jnp.abs(tracing.energy-particles.energy)/particles.energy)]
    ax.plot(tracing_times, energies, label=f'adaptative {method}', marker='o', markersize=3, linestyle='-')

    energies = []
    tracing_times = []
    for dt in [2e-7, 1e-7, 5e-8, 2.5e-8]:
        num_steps = int(tmax/dt)
        time0 = time()
        tracing = Tracing(field=field, model='GuidingCenter', method=getattr(diffrax, method), particles=particles,
                        stepsize="constant", maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
        block_until_ready(tracing.trajectories)
        tracing_times += [time() - time0]
        
        print(f"Tracing with {method} and step {tmax/num_steps:.2e} took {tracing_times[-1]:.2f} seconds")
        
        energies += [jnp.max(jnp.abs(tracing.energy-particles.energy)/particles.energy)]
    ax.plot(tracing_times, energies, label=f'{method}', marker='o', markersize=4, linestyle='-')


ax.legend()
ax.set_xlabel('Computation time (s)')
ax.set_ylabel('Relative Energy Error')
# ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='x', which='minor', length=0)
yticks = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
ax.set_yticks(yticks)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'gc_integration.pdf'))
plt.savefig(os.path.join(os.path.dirname(__file__), "../../../../UW/article/figures/", 'gc_integration.pdf'))
plt.show()

## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')