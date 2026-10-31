"""Shared physics helpers using the production collision maps and Brownian driver."""
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from essos.background_species import chandrasekhar, d_chandrasekhar
from essos.split import CollisionCoefficients, guiding_center_collision_step, particle_collision_step, guiding_center_spatial_diffusion, guiding_center_spatial_increment
from essos.split import solve_split

def constant_log_coefficients(speed, *, full_orbit=False):
    """Nonrelativistic equal-proton Maxwellian coefficients with fixed ln Lambda.

    Velocity is in sigma=sqrt(T/m); time is in 1/nu_s(sigma). Spatial units
    are rho_sigma. This control leaves the production coefficients untouched.
    """
    normalizer = chandrasekhar(jnp.asarray(1 / np.sqrt(2.0)))

    def values(v):
        x = v / jnp.sqrt(2.0)
        g = chandrasekhar(x)
        dp = g / (v * normalizer)
        dt = (jax.scipy.special.erf(x) - g) / (2 * v * normalizer)
        ddp = (d_chandrasekhar(x) / (jnp.sqrt(2.0) * v) - g / v ** 2) / normalizer
        return jnp.array([dp, dt, ddp, -v * dp])
    value, slope = jax.jvp(values, (speed,), (jnp.ones_like(speed),))
    dp, dt, ddp, q = value
    dk = slope[3] + slope[2] + 2 * (slope[0] - slope[1]) / speed - 2 * (dp - dt) / speed ** 2 if full_orbit else jnp.zeros_like(speed)
    return CollisionCoefficients(dp, dt, ddp, q, slope[3], dk)

def thermal_solver(model, method, horizon, times, *, coefficient_law=constant_log_coefficients):
    """Use the actual native collision maps and driver for the analytic control."""
    gc, adaptive = (model == 'gc', method != 'fixed')

    def batch(initial, keys, tolerance, cap, first):

        def one(y0, key):

            def kick(t, y, h, dw, args):
                speed = y[3] if gc else jnp.linalg.norm(y)
                c = coefficient_law(speed, full_orbit=not gc)
                kwargs = dict(tolerance=tolerance)
                if gc:
                    trial = guiding_center_collision_step(speed, y[4], h, dw[-2:], c, speed_floor=0.05 * np.sqrt(2.0), **kwargs)
                    dx = guiding_center_spatial_increment(y[4], dw[:3], c, jnp.array([0.0, 0.0, 1.0]), 1.0)
                    value = y.at[:3].add(dx).at[3:5].set(trial.y)
                    value = value.at[5].add(4 * guiding_center_spatial_diffusion(y[4], c, 1.0) * h)
                    return trial._replace(y=value)
                return particle_collision_step(y, h, dw, c, **kwargs)
            return solve_split(lambda t, y, a: jnp.zeros_like(y), kick, y0, t1=horizon, dt0=first, dtmax=cap, ts=jnp.asarray(times), key=key, noise_dim=5 if gc else 3, adaptive=adaptive, method='dopri5', max_steps=100000)
        return jax.vmap(one)(initial, keys)
    return jax.jit(batch)

def run_batches(executable, batches, parameters):
    outputs = []
    start = perf_counter()
    for inputs in batches:
        outputs.append(jax.device_get(jax.block_until_ready(executable(*inputs, *parameters))))
    output = jax.tree.map(lambda *x: np.concatenate(x, axis=0), *outputs)
    if np.any((output.status != 0) & (output.status != 1)):
        raise RuntimeError(f'invalid physical solve: {np.unique(output.status, return_counts=True)}')
    return (output, perf_counter() - start)

def work(output):
    return dict(accepted_mean=float(output.stats.accepted.mean()), rejected_mean=float(output.stats.rejected.mean()), trials_mean=float(output.stats.steps.mean()), max_accepted_error=float(output.stats.max_accepted_error.max()), peak_pending=int(output.stats.peak_pending.max()))
