import os
import gc
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

mass = PROTON_MASS
energy = 5000 * ONE_EV
cyclotron_frequency = ELEMENTARY_CHARGE * 0.3 / mass
print("cyclotron period:", 1 / cyclotron_frequency)

initial_xyz = jnp.array([[1.23, 0, 0]])
particles = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy,
                      initial_vparallel_over_v=[0.8], field=field)

tmax = 1e-4

fig, ax = plt.subplots(figsize=(9, 6))
fig_tol, ax_tol = plt.subplots(figsize=(9, 6))
markers = ["o-", "^-", "*-", "s-"]
methods = [('Tsit5', diffrax.Tsit5),
           ('Dopri5', diffrax.Dopri5),
           ('Dopri8', diffrax.Dopri8),
           ('Kvaerno5', diffrax.Kvaerno5)]

for (method_name, method_cls), marker in zip(methods, markers):
    energies = []
    tracing_times = []
    tolerances = [1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]
    for tolerance in tolerances:
        t0 = time()
        tracing = Tracing(field=field, model='GuidingCenter', particles=particles,
                          maxtime=tmax, timestep=1e-7,
                          atol=tolerance, rtol=tolerance,
                          solver=method_cls())
        block_until_ready(tracing.trajectories)
        tracing_times.append(time() - t0)
        energies.append(jnp.max(jnp.abs(tracing.energy() - particles.energy) / particles.energy))
        print(f"Tracing with adaptive {method_name} tol={tolerance:.0e} took {tracing_times[-1]:.2f}s")
    ax.plot(tracing_times, energies, label=f'{method_name} adapt', marker='o', markersize=3)
    ax_tol.plot(tolerances, energies, marker, label=f'{method_name} adapt', clip_on=False, linewidth=2.5)
    gc.collect()

ax.set_xlabel('Computation time (s)')
ax_tol.set_xlabel('Tracing tolerance')
ax.set_xlim(1e-1, 1e2)
ax_tol.set_xlim(tolerances[-1], tolerances[0])

for axis in [ax, ax_tol]:
    axis.legend(fontsize=15)
    axis.set_ylabel('Relative energy error')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_ylim(1e-16, 1e-4)
    axis.grid(axis='x', which='both', linestyle='--', linewidth=0.6)
    axis.grid(axis='y', which='major', linestyle='--', linewidth=0.6)
for figure in [fig, fig_tol]:
    figure.tight_layout()

for spine in ax_tol.spines.values():
    spine.set_zorder(0)

fig.savefig(os.path.join(output_dir, 'gc_integration.pdf'))
fig_tol.savefig(os.path.join(output_dir, 'energy_vs_tol.pdf'))
