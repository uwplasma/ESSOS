"""First-loss objectives with implicit crossing-time sensitivities.

Accepted RK times, collision increments and root-search iterations stay frozen.
The continuous crossing time is differentiated through its level-set equation,
using the time tangent of the same RK polynomial that located the event.
This module does not change the existing accepted-step adjoints.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@jax.custom_jvp
def _implicit_time(time, residual, slope, active, valid):
    return time


@_implicit_time.defjvp
def _implicit_time_jvp(primals, tangents):
    time, residual, slope, active, valid = primals
    _, residual_dot, _, _, _ = tangents
    slope = eqx.error_if(slope, active & ~valid,
        "first-loss AD requires a finite transverse outward continuous crossing; "
        "grazing, unrefined and instantaneous kick events are unsupported")
    safe = jnp.where(active, slope, jnp.ones_like(slope))
    return time, jnp.where(active, -residual_dot/safe, jnp.zeros_like(time))


def first_loss_time(solution, boundary, args=None, *, t0=0., root_atol=1e-7,
                    crossing_speed_floor=0.):
    """Return scalar/batched capped loss times with first-order implicit AD.

    ``boundary(t, state, args)`` must be an *unclamped*, differentiable scalar
    level set, positive inside the chosen confinement region. Supply native
    RK/split output obtained with ``record_event_tangent=True`` and a root-found
    event on this boundary. A clipped-to-zero copy may be used as the forward
    event to catch particles initially outside; this function needs the signed
    level set. Terminal success contributes the horizon with zero sensitivity;
    initial losses contribute t0 with zero sensitivity.
    Use ``adjoint="accepted"`` for reverse AD; ordinary forward solves are
    suitable for cost evaluation and independent finite-difference checks.

    The implicit rule is d(t_hit) = -d(boundary at fixed time) / d(boundary)/dt.
    Its denominator uses ``solution.event_tangent``, not an endpoint RHS
    approximation. Physical boundary parameters in args/closures remain active.
    No controller or rejection derivative is introduced. Failed solves and
    events on another surface raise; unsupported crossing derivatives raise
    during AD, without regularizing or clipping the requested gradient.
    """
    if not np.isfinite(root_atol) or root_atol <= 0:
        raise ValueError("root_atol must be positive and finite")
    if not np.isfinite(crossing_speed_floor) or crossing_speed_floor < 0:
        raise ValueError("crossing_speed_floor must be finite and nonnegative")
    time = lax.stop_gradient(solution.t)
    status = lax.stop_gradient(solution.status)
    state = solution.y
    tangent = lax.stop_gradient(solution.event_tangent)
    if state.shape[:-1] != time.shape or tangent.shape != state.shape or status.shape != time.shape:
        raise ValueError("solution terminal time/state/status/tangent batch shapes disagree")
    time = eqx.error_if(time, jnp.any(status > 1),
        "first-loss objective requires successful or event-terminated traces")
    def one(t, y, velocity, code):
        active = (code == 1) & (t > t0)
        g = jnp.asarray(boundary(t, y, args))
        if g.shape or not jnp.issubdtype(g.dtype, jnp.floating):
            raise ValueError("first-loss boundary must return a floating scalar")
        # Inactive terminal states may be far inside/outside; they have no
        # implicit-root derivative. Freeze every metadata/branch decision.
        t = eqx.error_if(t, active & (~jnp.isfinite(g) | (jnp.abs(g) > root_atol)),
            "terminal event is not on the requested first-loss boundary")
        safe_velocity = jnp.where(active, velocity, jnp.zeros_like(velocity))
        slope = jax.jvp(lambda tt, yy:boundary(tt, yy, args),
            (t, y), (jnp.ones_like(t), safe_velocity))[1]
        slope = lax.stop_gradient(slope)
        valid = lax.stop_gradient(jnp.isfinite(slope) & (slope < -crossing_speed_floor))
        return _implicit_time(t, g, slope, active, valid)
    flat = jax.vmap(one)(time.reshape(-1), state.reshape((-1,state.shape[-1])),
        tangent.reshape((-1,state.shape[-1])), status.reshape(-1))
    return flat.reshape(time.shape)


def loss_time_penalty(times, horizon, *, t0=0., rate=2.):
    """Per-particle normalized exponential penalty: birth loss=1, retained=0.

    Equivalent to a positive affine rescaling of exp(-rate*t_hit/T).
    Delaying a loss improves this objective even if the loss fraction is
    unchanged. It is not a physical collisional energy-loss measurement.
    """
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    span = jnp.asarray(horizon)-t0
    span = eqx.error_if(span, ~jnp.isfinite(span) | (span <= 0),
        "loss-time horizon must be finite and later than t0")
    times = jnp.asarray(times)
    times = eqx.error_if(times, jnp.any(~jnp.isfinite(times) | (times < t0) | (times > horizon)),
        "capped loss times must lie between t0 and horizon")
    return jnp.expm1(rate*(horizon-times)/span)/jnp.expm1(jnp.asarray(rate,times.dtype))


def mean_loss_time_cost(solution, boundary, horizon, args=None, *, t0=0., rate=2.,
                        root_atol=1e-7, crossing_speed_floor=0.):
    """Mean first-loss penalty, including continuous crossing-time AD."""
    times = first_loss_time(solution,boundary,args,t0=t0,root_atol=root_atol,
        crossing_speed_floor=crossing_speed_floor)
    return jnp.mean(loss_time_penalty(times,horizon,t0=t0,rate=rate))


def toroidal_loss_boundary(major_radius, minor_radius, axis_z=0.):
    """Signed squared-radius level set; Cartesian GC and Lorentz share it."""
    def boundary(t, y, args):
        del t, args
        radial = jnp.hypot(y[0],y[1])-major_radius
        return minor_radius**2-radial**2-(y[2]-axis_z)**2
    return boundary


def first_loss_event(boundary):
    """Forward event catching initial exterior states; keep raw boundary for AD."""
    return lambda t,y,args:jnp.maximum(jnp.zeros((),y.dtype),boundary(t,y,args))
