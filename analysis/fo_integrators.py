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

# Input parameters
tmax = 1e-4
nparticles = number_of_processors_to_use
R0 = jnp.linspace(1.23, 1.27, nparticles)
trace_tolerance = 1e-12
mass=PROTON_MASS
energy=4000*ONE_EV
cyclotron_frequency = ELEMENTARY_CHARGE*0.3/mass
print("cyclotron period:", 1/cyclotron_frequency)

# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), '../examples/input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils_from_json(json_file)
field = BiotSavart(coils)

# Initialize particles
Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
particles = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy, field=field)

fig, ax = plt.subplots(figsize=(9, 6))

method_names = ['Dopri8', 'Boris']
methods = [getattr(diffrax, method) for method in method_names[:-1]] + ['Boris']
for method_name, method in zip(method_names, methods):
    if method_name != 'Boris':
        starting_dt = 1e-9
        num_steps = int(tmax/starting_dt)
        energies = []
        tracing_times = []
        for trace_tolerance in [1e-8, 1e-10, 1e-12, 1e-14]:
            time0 = time()
            tracing = Tracing(field=field, model='FullOrbit', method=method, particles=particles,
                            maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
            block_until_ready(tracing)
            tracing_times += [time() - time0]
            
            print(f"Tracing with adaptative {method_name} and tolerance {trace_tolerance:.0e} took {tracing_times[-1]:.2f} seconds")
            
            energies += [jnp.mean(jnp.abs(tracing.energy-particles.energy)/particles.energy)]
        ax.plot(tracing_times, energies, label=f'adaptative {method_name}', marker='o', markersize=3, linestyle='-')

    energies = []
    tracing_times = []
    for n_points_in_gyration in [5, 10, 20, 30, 40]:
        dt = 1/(n_points_in_gyration*cyclotron_frequency)
        num_steps = int(tmax/dt)
        time0 = time()
        tracing = Tracing(field=field, model='FullOrbit', method=method, particles=particles,
                        stepsize="constant", maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
        block_until_ready(tracing)
        tracing_times += [time() - time0]
        
        print(f"Tracing with {method_name} and step {tmax/num_steps:.2e} took {tracing_times[-1]:.2f} seconds")
        
        energies += [jnp.mean(jnp.abs(tracing.energy-particles.energy)/particles.energy)]
    ax.plot(tracing_times, energies, label=f'{method_name}', marker='o', markersize=4, linestyle='-')


ax.legend()
ax.set_xlabel('Computation time (s)')
ax.set_ylabel('Relative Energy Error')
# ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='x', which='minor', length=0)
yticks = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
ax.set_yticks(yticks)
ax.set_ylim(top=1e-6)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'fo_integration.pdf'))
plt.savefig(os.path.join(os.path.dirname(__file__), "../../../../UW/article/figures/", 'fo_integration.pdf'))
plt.show()

## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')