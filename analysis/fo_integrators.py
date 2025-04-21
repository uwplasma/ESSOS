import os
number_of_processors_to_use = 1 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from essos.fields import BiotSavart
from essos.coils import Coils_from_json
from essos.constants import PROTON_MASS, ONE_EV
from essos.dynamics import Tracing, Particles
# import integrators
import diffrax

# Input parameters
tmax = 1e-4
nparticles = number_of_processors_to_use
R0 = jnp.linspace(1.23, 1.27, nparticles)
trace_tolerance = 1e-12
num_steps = 5000
mass=PROTON_MASS
energy=4000*ONE_EV

print(f"dt = {tmax/num_steps:.2e}")

# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), '../examples/input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils_from_json(json_file)
field = BiotSavart(coils)

# Initialize particles
Z0 = jnp.zeros(nparticles)
phi0 = jnp.zeros(nparticles)
initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
particles = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy, field=field)

fig, ax = plt.subplots(figsize=(7, 5))

method_names = ['Tsit5', 'Dopri5', 'Dopri8', 'Boris']
methods = [getattr(diffrax, method) for method in method_names[:-1]] + ['Boris']
for method_name, method in zip(method_names, methods):
    if method_name != 'Boris':
        energies = []
        tracing_times = []
        for trace_tolerance in [1e-8, 1e-10, 1e-12, 1e-13, 1e-14]:
            if method_name == 'Dopri8':
                if trace_tolerance == 1e-13:
                    trace_tolerance = 1e-14
                elif trace_tolerance == 1e-14:
                    trace_tolerance = 1e-15
            time0 = time()
            tracing = Tracing(field=field, model='FullOrbit', method=method, particles=particles,
                            maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
            tracing_times += [time() - time0]
            
            print(f"Tracing with adaptative {method_name} and tolerance {trace_tolerance:.0e} took {tracing_times[-1]:.2f} seconds")
            
            energies += [jnp.mean(jnp.abs(tracing.energy-particles.energy)/particles.energy)]
        ax.plot(tracing_times, energies, label=f'adaptative {method_name}', marker='o', markersize=3, linestyle='-')

    energies = []
    tracing_times = []
    for num_steps in [100000, 200000, 300000, 500000, 1000000]:
        if method_name == 'Boris' or method_name == 'Dopri8':
            num_steps //= 10
        time0 = time()
        tracing = Tracing(field=field, model='FullOrbit', method=method, particles=particles,
                        stepsize="constant", maxtime=tmax, timesteps=num_steps, tol_step_size=trace_tolerance)
        tracing_times += [time() - time0]
        
        print(f"Tracing with {method_name} and step {tmax/num_steps:.2e} took {tracing_times[-1]:.2f} seconds")
        
        energies += [jnp.mean(jnp.abs(tracing.energy-particles.energy)/particles.energy)]
    ax.plot(tracing_times, energies, label=f'{method_name}', marker='o', markersize=4, linestyle='-')

from matplotlib.ticker import LogFormatterMathtext

ax.legend()
ax.set_xlabel('Computation time (s)')
ax.set_ylabel('Relative Energy Error')
# ax.set_xscale('log')
ax.set_yscale('log')
# ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.tick_params(axis='x', which='minor', length=0)
yticks = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
ax.set_yticks(yticks)
ax.set_ylim(top=1e-6)
# xticks = [1e-1, 1e-0, 1e1, 1e2]
# ax.set_xticks(xticks)

plt.tight_layout()
plt.savefig(os.path.dirname(__file__) + '/fo_integration.pdf')
plt.show()

## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')