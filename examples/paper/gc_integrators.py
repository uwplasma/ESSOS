"""Guiding-centre integrators: accuracy against cost (ESSOS paper).

Teaches how the adaptive solvers (Tsit5, Dopri5, Dopri8 and the implicit Kvaerno5) trade relative energy error
against run time and tolerance for the guiding-centre model.

Runtime: about 5-10 minutes on a CPU (Kvaerno5 adds two slower runs).
"""
import os
import gc
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
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load coils and field
json_file = os.path.join(os.path.dirname(__file__), '../input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
coils = Coils.from_json(json_file)
field = BiotSavart(coils)

# Particle parameters
nparticles = 1
mass=PROTON_MASS
energy=5000*ONE_EV
cyclotron_frequency = ELEMENTARY_CHARGE*0.3/mass
print("cyclotron period:", 1/cyclotron_frequency)

# Particles initialization
initial_xyz=jnp.array([[1.23, 0, 0]])
particles = Particles(initial_xyz=initial_xyz, mass=mass, energy=energy, initial_vparallel_over_v=[0.8], field=field)

# Tracing parameters
tmax = 1e-4

# Two figures: energy error vs computation time, and energy error vs tolerance.
fig, ax = plt.subplots(figsize=(9, 6))
fig_tol, ax_tol = plt.subplots(figsize=(9, 6))
markers = ["o-", "^-", "*-", "s-"]
# For each adaptive solver, sweep the integration tolerance and record both the
# resulting energy error and the wall-clock time. Kvaerno5 is implicit (needs
# lineax >= 0.1.0) and much slower, so it runs at two tolerances only; the
# others are explicit Runge-Kutta methods.
solvers = [('Tsit5', diffrax.Tsit5),
           ('Dopri5', diffrax.Dopri5),
           ('Dopri8', diffrax.Dopri8),
           ('Kvaerno5', diffrax.Kvaerno5)]
for (method, solver_class), marker in zip(solvers, markers):
    energies = []
    tracing_times = []
    tolerances = [1e-7, 1e-8] if method == 'Kvaerno5' else [1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]
    for tolerance in tolerances:
        time0 = time()
        tracing = Tracing(field=field, model='GuidingCenterAdaptative', particles=particles,
                          maxtime=tmax, timestep=1e-7,
                          atol=tolerance, rtol=tolerance,
                          solver=solver_class())
        block_until_ready(tracing.trajectories)
        tracing_times += [time() - time0]

        print(f"Tracing with adaptive {method} and {tolerance=:.0e} took {tracing_times[-1]:.2f} seconds")

        energies += [jnp.max(jnp.abs(tracing.energy()-particles.energy)/particles.energy)]
    ax.plot(tracing_times, energies, label=f'{method} adapt', marker='o', markersize=3)
    ax_tol.plot(tolerances, energies, marker, label=f'{method} adapt', clip_on=False, linewidth=2.5)
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
plt.show()

## Save results in vtk format to analyze in Paraview
# tracing.to_vtk('trajectories')
# coils.to_vtk('coils')
