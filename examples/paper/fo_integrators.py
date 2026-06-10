import os
number_of_processors_to_use = 1
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
from jax import block_until_ready
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from essos.fields import BiotSavart
from essos.coils import Coils
from essos.constants import PROTON_MASS, ONE_EV, ELEMENTARY_CHARGE
from essos.dynamics import Tracing, Particles
import diffrax

output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), '../input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils.from_json(json_file)
field = BiotSavart(coils)

# Particle parameters
mass = PROTON_MASS
energy = 5000 * ONE_EV
cyclotron_frequency = ELEMENTARY_CHARGE * 0.3 / mass
print("cyclotron period:", 1 / cyclotron_frequency)

initial_xyz = jnp.array([[1.23, 0, 0]])
particles = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy,
                      initial_vparallel_over_v=[0.8], field=field)

tmax = 1e-4

fig, ax = plt.subplots(figsize=(9, 6))

# Adaptive diffrax solvers: sweep tolerances
diffrax_methods = [('Tsit5', diffrax.Tsit5), ('Dopri5', diffrax.Dopri5), ('Dopri8', diffrax.Dopri8)]
for method_name, method_cls in diffrax_methods:
    energies = []
    tracing_times = []
    for trace_tolerance in [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]:
        t0 = time()
        tracing = Tracing(field=field, model='FullOrbit', particles=particles,
                          maxtime=tmax, timestep=1e-9,
                          atol=trace_tolerance, rtol=trace_tolerance,
                          solver=method_cls())
        block_until_ready(tracing.trajectories)
        tracing_times.append(time() - t0)
        energies.append(jnp.mean(jnp.abs(tracing.energy() - particles.energy) / particles.energy))
        print(f"Tracing with adaptive {method_name} tol={trace_tolerance:.0e} took {tracing_times[-1]:.2f}s")
    ax.plot(tracing_times, energies, label=f'{method_name} adapt', marker='o', markersize=3, linestyle='-')

# Boris (fixed-step symplectic): sweep step sizes
energies = []
tracing_times = []
for n_points_in_gyration in [10, 20, 50, 75, 100, 150, 200]:
    dt = 1 / (n_points_in_gyration * cyclotron_frequency)
    t0 = time()
    tracing = Tracing(field=field, model='FullOrbit_Boris', particles=particles,
                      maxtime=tmax, timestep=dt)
    block_until_ready(tracing.trajectories)
    tracing_times.append(time() - t0)
    energies.append(jnp.mean(jnp.abs(tracing.energy() - particles.energy) / particles.energy))
    print(f"Tracing with Boris step {dt:.2e} took {tracing_times[-1]:.2f}s")
ax.plot(tracing_times, energies, label='Boris', marker='o', markersize=4, linestyle='-')

ax.legend(fontsize=15, loc='upper left')
ax.set_xlabel('Computation time (s)')
ax.set_ylabel('Relative energy error')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-1, 1e2)
ax.set_ylim(1e-16, 1e-4)
plt.grid(axis='x', which='both', linestyle='--', linewidth=0.6)
plt.grid(axis='y', which='major', linestyle='--', linewidth=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fo_integration.pdf'))
