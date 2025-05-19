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

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
particles = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy, initial_vparallel_over_v=[0.8])

# Tracing parameters
tmax = 1e-4
dt = 1e-7
num_steps = int(tmax/dt)

fig, ax = plt.subplots(figsize=(9, 6))

for method in ['Tsit5', 'Dopri5', 'Dopri8', 'Kvaerno5']:
    energies = []
    tracing_times = []
    for tolerance in [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]:
        time0 = time()
        tracing = Tracing('GuidingCenter', field, tmax, method=getattr(diffrax, method), timesteps=num_steps,
                          stepsize='adaptive', tol_step_size=tolerance, particles=particles)
        block_until_ready(tracing.trajectories)
        tracing_times += [time() - time0]
        
        print(f"Tracing with adaptive {method} and {tolerance=:.0e} took {tracing_times[-1]:.2f} seconds")
        
        energies += [jnp.max(jnp.abs(tracing.energy-particles.energy)/particles.energy)]
    ax.plot(tracing_times, energies, label=f'{method} adapt', marker='o', markersize=3, linestyle='-')

    if method == 'Kvaerno5': continue

    energies = []
    tracing_times = []
    for dt in [4e-7, 2e-7, 1e-7, 8e-8, 6e-8, 4e-8, 2e-8, 1e-8]:
        num_steps = int(tmax/dt)
        time0 = time()
        tracing = Tracing('GuidingCenter', field, tmax, method=getattr(diffrax, method), 
                          timesteps=num_steps, stepsize="constant", particles=particles)
        block_until_ready(tracing.trajectories)
        tracing_times += [time() - time0]
        
        print(f"Tracing with {method} and {dt=:.2e} took {tracing_times[-1]:.2f} seconds")
        
        energies += [jnp.max(jnp.abs(tracing.energy-particles.energy)/particles.energy)]
    ax.plot(tracing_times, energies, label=f'{method}', marker='o', markersize=4, linestyle='-')


ax.legend(fontsize=15)
ax.set_xlabel('Computation time (s)')
ax.set_ylabel('Relative Energy Error')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-1, 1e2)
ax.set_ylim(1e-16, 1e-4)
plt.grid(axis='x', which='both', linestyle='--', linewidth=0.6)
plt.grid(axis='y', which='major', linestyle='--', linewidth=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gc_integration.pdf'))
plt.savefig(os.path.join(os.path.dirname(__file__), "../../../../UW/article/figures/", 'gc_integration.pdf'))
plt.show()

## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')