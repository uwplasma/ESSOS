"""Production first-passage diagnostic used by validate_production_tolerance."""
import jax
import jax.numpy as jnp
import numpy as np
from essos.background_species import BackgroundSpecies
from essos.constants import PROTON_MASS as M,ELEMENTARY_CHARGE as Q
from essos.split import (solve_split,collision_coefficients,guiding_center_collision_step,particle_collision_step,
    guiding_center_spatial_diffusion,guiding_center_spatial_increment)
from .collision_event_diagnostic import boundary_step_limit


def native_solver(model, controller, adaptive, physical, horizon, threshold, *,
                  spatial=False, times=None, magnetic_field=1., boundary_shift=False,
                  event_interval=None, event_safety=.2, record_steps=False):
    """Native benchmark, optionally testing a discrete-exit boundary correction.

    ``boundary_shift`` tests the Gobet--Menozzi shift at each actual step h;
    it is not enabled in production tracing. The event flag is auxiliary and
    does not enter the velocity/spatial update or the local error estimates.
    ``event_interval`` enables boundary-aware stepping and refinement of
    detected crossing intervals. It is in collision-time units (h*nu0),
    independently of the collision tolerance. No boundary shift is needed.
    """
    if spatial and model != "gc":
        raise ValueError("explicit spatial kicks require the GC model")
    if event_interval is not None:
        if not np.isfinite(event_interval) or event_interval <= 0 or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("event_interval and the thermal threshold must be positive and finite")
        if not np.isfinite(event_safety) or not 0 < event_safety < 1:
            raise ValueError("event_safety must lie strictly between zero and one")
        if boundary_shift:
            raise ValueError("compare event refinement and the experimental shift separately")
    species = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([physical.density]), jnp.array([physical.temperature_ev]),
        coulomb_log_model=physical.coulomb_log_model)
    gc = model == "gc"
    v0 = physical.sigma*np.sqrt(10.)
    y0 = jnp.array([v0, .6]) if gc else jnp.array([.8*v0, 0., .6*v0])
    if spatial:
        # Position, speed, pitch, accumulated 4*D_X*dt. All units are SI.
        y0 = jnp.array([0., 0., 0., v0, .6, 0.])
    physical_size = y0.size
    if boundary_shift:
        y0 = jnp.append(y0, v0/physical.sigma-threshold)
        boundary_diffusion = float(physical.coefficients(threshold*physical.sigma)["parallel"])
    nu = physical.nu
    save_times = jnp.empty((0,)) if times is None else jnp.asarray(times)/nu
    if times is not None:
        # JAX division may use a reciprocal multiplication, differing from
        # Python's horizon/nu by one ulp. A save at the requested horizon must
        # use the exact same SI endpoint as t1, rather than be rejected as >t1.
        save_times = jnp.where(jnp.asarray(times) == horizon, jnp.asarray(horizon/nu), save_times)

    def batch(keys, tolerance, cap, initial):
        def one(key):
            def event_limit(t, y, args):
                speed = y[3] if spatial else (y[0] if gc else jnp.linalg.norm(y))
                c = collision_coefficients(species, M, Q, speed, jnp.zeros(3))
                drift = c.friction+c.parallel_derivative+2*c.parallel/speed
                return boundary_step_limit(speed, threshold*physical.sigma, c.parallel,
                    drift, event_interval/nu, safety=event_safety)

            def kick(t, y, h, dw, args):
                y = y[:physical_size]
                speed = y[3] if spatial else (y[0] if gc else jnp.linalg.norm(y))
                pitch = y[4] if spatial else y[1]
                c = collision_coefficients(species, M, Q, speed, jnp.zeros(3), full_orbit=not gc)
                kwargs = dict(tolerance=tolerance, return_error_model=adaptive and controller == "candidate")
                # Keep driver, Wiener increments and candidate-error metadata
                # in SI units. Only reporting converts to collision time.
                result = (guiding_center_collision_step(speed, pitch, h, dw[-2:], c,
                            speed_floor=.05*np.sqrt(2)*physical.sigma, **kwargs) if gc else
                        particle_collision_step(y, h, dw, c, **kwargs))
                if spatial:
                    omega = Q*magnetic_field/M
                    displacement = guiding_center_spatial_increment(pitch, dw[:3], c, jnp.array([0., 0., 1.]), omega)
                    value = y.at[:3].add(displacement).at[3:5].set(result.y)
                    value = value.at[5].add(4*guiding_center_spatial_diffusion(pitch, c, omega)*h)
                    result = result._replace(y=value)
                if boundary_shift:
                    value = result.y
                    next_speed = value[3] if spatial else (value[0] if gc else jnp.linalg.norm(value))
                    level = (next_speed-threshold*physical.sigma
                        -.5825971579390108*jnp.sqrt(2*boundary_diffusion*h))/physical.sigma
                    result = result._replace(y=jnp.append(value, level))
                return result
            return solve_split(lambda t, y, args: jnp.zeros_like(y), kick, y0,
                t1=horizon/nu, dt0=initial/nu, dtmax=cap/nu, ts=save_times,
                key=key, noise_dim=(5 if spatial else 2) if gc else 3, method="dopri5", adaptive=adaptive,
                collision_controller=controller,
                collision_wiener_norm=(lambda w: jnp.linalg.norm(w[-2:])) if gc else (lambda w: jnp.abs(w[2])),
                event=lambda t, y, args: y[-1] if boundary_shift else
                    (y[3] if spatial else (y[0] if gc else jnp.linalg.norm(y)))/physical.sigma-threshold,
                max_steps=100_000, record_steps=record_steps,
                step_limit=event_limit if event_interval is not None else None,
                event_dtmax=None if event_interval is None else event_interval/nu)
        out = jax.vmap(one)(keys)
        return out._replace(t=out.t*nu)
    return jax.jit(batch)
