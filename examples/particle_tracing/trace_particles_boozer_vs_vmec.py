import os
number_of_processors_to_use = 8 # Parallelization, this should divide nparticles
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from booz_xform_jax import Booz_xform
from essos.boozer import BoozerField, trace_boozer
from essos.fields import Vmec
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from essos.dynamics import Tracing, Particles

# Same alphas traced in Boozer coordinates (essos.boozer, from booz_xform_jax)
# and in VMEC coordinates (essos.fields.Vmec) through the ARIES-CS equilibrium.
# Needs booz_xform_jax (pip install booz_xform_jax).
tmax = 1e-4
nparticles = number_of_processors_to_use*16
times_to_trace = 200
j = 69 # half-grid surface of the births, s = (j + 1/2) / (ns - 1)

# Wall time without JIT: subtract the tracing, lowering and compile events JAX reports
compile_time = [0.0]
jax.monitoring.register_event_duration_secs_listener(
    lambda event, duration, **_: compile_time.__setitem__(0, compile_time[0] + duration)
    if event.startswith('/jax/core/compile/') else None)
def run_time(function):
    compile0, time0 = compile_time[0], time()
    result = function()
    return result, time() - time0 - (compile_time[0] - compile0)

# Load the equilibrium and transform it to Boozer coordinates
wout_file = os.path.join(os.path.dirname(__file__), "../input_files", "wout_n3are_R7.75B5.7.nc")
vmec = Vmec(wout_file)
vmec.nc.set_auto_mask(False)
booz = Booz_xform(verbose=0, mboz=32, nboz=32)
booz.read_wout(wout_file, flux=False)
booz.run()
boozer_field = BoozerField.from_booz_xform(booz, psi0=vmec.nc.variables["phi"][-1] / (2*np.pi), mode_tolerance=1e-3)

# Births: uniform Boozer angles and pitch on one surface, mapped to VMEC angles
# with zeta_B = phi + nu and theta_B - iota nu = theta + lambda(theta, phi)
rng = np.random.default_rng(0)
s = np.full(nparticles, (j + 0.5) / (vmec.ns - 1))
theta_b, zeta_b = rng.uniform(0, 2*np.pi, nparticles), rng.uniform(0, 2*np.pi/vmec.nfp, nparticles)
pitch = rng.uniform(-1, 1, nparticles)
nu = np.sin(np.outer(theta_b, booz.xm_b) - np.outer(zeta_b, booz.xn_b)) @ np.asarray(booz.numns_b)[:, j]
phi, theta_star = zeta_b - nu, theta_b - booz.iota[j]*nu
xm, xn, lmns = np.asarray(vmec.xm), np.asarray(vmec.xn), vmec.nc.variables["lmns"][j + 1]
theta = theta_star.copy()
for _ in range(30): # Newton for theta + lambda(theta, phi) = theta*
    angle = np.outer(theta, xm) - np.outer(phi, xn)
    theta -= (theta + np.sin(angle) @ lmns - theta_star) / (1 + np.cos(angle) @ (xm*lmns))

# Trace in Boozer coordinates (fixed-step RK4)
speed = np.sqrt(2*FUSION_ALPHA_PARTICLE_ENERGY/ALPHA_PARTICLE_MASS)
boozer, time_boozer = run_time(lambda: trace_boozer(
    boozer_field, s, theta_b, zeta_b, pitch, speed=speed, mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE, tmax=tmax, timestep=1.25e-7, n_save=times_to_trace))

# Trace in VMEC coordinates (adaptive guiding centre)
particles = Particles(initial_xyz=jnp.array([s, theta, phi]).T, initial_vparallel_over_v=pitch,
                      mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE, energy=FUSION_ALPHA_PARTICLE_ENERGY)
tracing, time_vmec = run_time(lambda: Tracing(field=vmec, model='GuidingCenterAdaptative', particles=particles,
    maxtime=tmax, timestep=1e-8, times_to_trace=times_to_trace, atol=1e-7, rtol=1e-7))

# Loss fractions with binomial errors, and wall times without compilation
for name, lost, run in (("Boozer", boozer.lost.mean(), time_boozer),
                        ("VMEC", float(tracing.loss_fractions[-1]), time_vmec)):
    print(f"{name:7s}: loss fraction {100*lost:5.1f}% ± {100*np.sqrt(lost*(1-lost)/nparticles):.1f}%, "
          f"tracing took {run:.2f} s without JIT")

plt.plot(1e3*boozer.times, 100*boozer.loss_fractions(), label=f'Boozer ({time_boozer:.1f} s)')
plt.plot(1e3*np.asarray(tracing.times), 100*np.asarray(tracing.loss_fractions), '--', label=f'VMEC ({time_vmec:.1f} s)')
plt.xlabel('Time (ms)')
plt.ylabel('Loss fraction (%)')
plt.title(f'{nparticles} alphas in ARIES-CS, s = {s[0]:.3f}')
plt.legend()
plt.tight_layout()
plt.show()
