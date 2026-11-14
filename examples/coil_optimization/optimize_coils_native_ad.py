"""Minimize the mean final minor radius of 8 alpha guiding centers with native reverse AD.

The gradient with respect to the coil Fourier dofs comes from the accepted-step adjoint
(`native_adjoint="accepted"`); L-BFGS-B steps stay within MAX_CHANGE of the starting dofs.
Run time: about 2 min on a laptop CPU, mostly compilation.
"""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, ONE_EV
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart
from essos.particle_losses import mean_final_radius

COILS = Path(__file__).resolve().parents[1]/"input_files"/"ESSOS_biot_savart_LandremanPaulQA.json"
N_PARTICLES, ENERGY_EV, TMAX = 8, 4000., 1e-5
MAX_CHANGE, MAXITER, LENGTH_WEIGHT = 0.01, 5, 1.

base = Coils.from_json(str(COILS))
theta0 = jnp.asarray(base.dofs)
axis = BiotSavart(base)
r_axis, z_axis = float(axis.r_axis), float(axis.z_axis)
length_limit = 1.05*float(jnp.max(base.length))
xyz = jnp.column_stack((jnp.linspace(1.23, 1.27, N_PARTICLES), jnp.zeros((N_PARTICLES, 2))))
pitch = jnp.linspace(0.35, 0.65, N_PARTICLES)


def objective(delta):
    field = BiotSavart(base.with_dofs(theta0 + MAX_CHANGE*delta))
    birth = Particles(xyz, initial_vparallel_over_v=pitch, mass=ALPHA_PARTICLE_MASS,
                      charge=ALPHA_PARTICLE_CHARGE, energy=ENERGY_EV*ONE_EV)
    out = Tracing(field=field, particles=birth, model="GuidingCenterAdaptative", integrator="essos_dopri8",
                  native_adjoint="accepted", rtol=1e-9, atol=1e-9, maxtime=TMAX, timestep=TMAX/2,
                  times_to_trace=32, max_steps=4096, progress=False)
    penalty = jnp.mean(jnp.maximum(field.coils.length - length_limit, 0.))
    return mean_final_radius(out.trajectories, r_axis, z_axis) + LENGTH_WEIGHT*penalty


value_and_grad = jax.jit(jax.value_and_grad(objective))


def fun(x):
    value, gradient = value_and_grad(jnp.asarray(x))
    return float(value), np.asarray(gradient, dtype=float)


result = minimize(fun, np.zeros(theta0.size), jac=True, method="L-BFGS-B",
                  bounds=[(-1., 1.)]*theta0.size, options=dict(maxiter=MAXITER))
print(f"mean final radius: {fun(np.zeros(theta0.size))[0]:.4f} -> {result.fun:.4f} in {result.nit} iterations")
