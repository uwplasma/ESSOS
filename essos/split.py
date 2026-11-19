"""Forward nonrelativistic orbit/collision splitting.

An independently controlled native RK orbit solve precedes each Milstein
collision map. Brownian rejection memory preserves every sampled interval.
"""
from enum import IntEnum
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util
from .background_species import nu_D_ab, nu_par_ab, d_nu_par_ab, nu_s_ab
from .rk import RKStatus, solve_rk, _concrete_check, _prepare_rhs, OrbitErrorNorm, _prepare_vmec_data


def _ratio(numerator, denominator):
    """numerator/denominator with 0/0 -> 0 and x/0 -> inf (vacuum has no error)."""
    safe = jnp.where(denominator > 0, denominator, 1.)
    return jnp.where(denominator > 0, numerator/safe,
                     jnp.where(numerator == 0, 0., jnp.inf))


# Collisions


class CollisionCoefficients(NamedTuple):
    parallel: jax.Array             # D_parallel [m^2/s^3]
    perpendicular: jax.Array        # D_perp [m^2/s^3]
    parallel_derivative: jax.Array   # dD_parallel/dv
    friction: jax.Array             # Q = -nu_s v [m/s^2]
    friction_derivative: jax.Array
    particle_drift_derivative: jax.Array  # dK/dv, needed only by particle error control

class CollisionStep(NamedTuple):
    y: jax.Array
    drift_error: jax.Array           # dimensionless, acceptable if <= 1
    diffusion_error: jax.Array

def _diffusion_root(value):
    """sqrt(max(value, 0)) with a zero AD tangent at zero diffusion.

    The safe inner argument prevents 0*inf from contaminating vacuum-state
    sensitivities. Positive-coefficient values/derivatives and invalid NaN
    values are preserved. Turning on noise exactly at zero is not a regular
    pathwise derivative; zero is the chosen boundary convention there.
    """
    positive = value > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, value, 1.)),
                     jnp.where(jnp.isnan(value), value, 0.))

def guiding_center_spatial_diffusion(pitch, c, gyrofrequency):
    """Paper Eq. (36), in SI velocity units: D_X [m^2/s].

    This is the leading uniform-field GC coefficient, with no relativistic
    factor c^2 because ``c.parallel/perpendicular`` already diffuse velocity.
    """
    return (c.parallel*(1-pitch*pitch) + c.perpendicular*(1+pitch*pitch))/(2*gyrofrequency**2)

def guiding_center_spatial_increment(pitch, dw, c, unit_b, gyrofrequency):
    """Paper Eq. (69): frozen coefficient and perpendicular projection.

    ``dw`` is an independent spatial Wiener vector, with covariance h I;
    speed/pitch use separate Wiener components. No spatial error estimator
    or full multidimensional Milstein correction is supplied by this map.
    """
    dx = guiding_center_spatial_diffusion(pitch, c, gyrofrequency)
    return _diffusion_root(2*dx)*(dw-unit_b*jnp.dot(unit_b, dw))

def collision_coefficients(species, mass, charge, speed, point, *, full_orbit=False):
    """SI diffusion and friction, including the speed-dependent Coulomb log.

    ``speed`` must be positive. ``point`` uses the field's profile coordinates.
    The particle Ito drift is Q + D_parallel' + 2(D_parallel-D_perp)/v;
    the speed Ito drift is Q + D_parallel' + 2 D_parallel/v.
    """
    def values(v):
        def pair(i):
            nd = nu_D_ab(mass, charge, i, v, point, species)
            np_ = nu_par_ab(mass, charge, i, v, point, species)
            dnp = d_nu_par_ab(mass, charge, i, v, point, species)
            ns = nu_s_ab(mass, charge, i, v, point, species)
            return jnp.array([.5*v*v*np_, .5*v*v*nd,
                              v*np_ + .5*v*v*dnp, -ns*v])
        return jnp.sum(jax.vmap(pair)(species.species_indeces), axis=0)

    val, slope = jax.jvp(values, (speed,), (jnp.ones_like(speed),))
    dp, dt, ddp, q = val
    if full_orbit:
        dk = slope[3] + slope[2] + 2*(slope[0]-slope[1])/speed - 2*(dp-dt)/speed**2
    else:
        dk = jnp.zeros_like(speed)
    return CollisionCoefficients(dp, dt, ddp, q, slope[3], dk)

def reflect_pitch(pitch):
    """Repeated reflection onto [-1, 1], including overshoots beyond one wall."""
    return 1 - jnp.abs(jnp.mod(pitch + 1, 4) - 2)

def _speed_diffusion_error(c, dw, tolerance):
    return _ratio(c.parallel_derivative**2 * jnp.abs(dw)**3,
                  6*tolerance*jnp.sqrt(jnp.maximum(c.parallel, 0.)))

def guiding_center_collision_step(speed, pitch, h, dw, c, *, tolerance=1e-3, speed_floor=0.):
    """Paper Eqs. (70)--(75), with speed [m/s] replacing normalized momentum.

    ``dw`` contains two independent Wiener increments of variance h, not unit
    normals. The diffusion coefficients are frozen at the initial speed for
    this local speed/pitch map, as in the paper. This is not a claim of a full
    multidimensional Milstein expansion for arbitrary noncommuting noise.
    """
    dp, dt, ddp, q, dq, _ = c
    nu = 2*dt/speed**2
    drift = q + ddp + 2*dp/speed
    # The 2 D_par/v part of the drift is the radial (Bessel) drift of a speed whose perpendicular
    # velocity components also diffuse; explicit Euler overshoots when v^2 < 2 D_par h. It is applied
    # as the perpendicular kick instead, v -> sqrt(v1^2 + 4 D_par h): same drift to O(h), v > 0.
    v = speed + (drift-2*dp/speed)*h + _diffusion_root(2*dp)*dw[0] + .5*ddp*(dw[0]**2-h)
    v = jnp.sqrt(v*v + 4*dp*h)
    v = speed_floor + jnp.abs(v-speed_floor)
    angular_noise = _diffusion_root((1-pitch**2)*nu)
    xi = pitch - pitch*nu*h + angular_noise*dw[1] - .5*pitch*nu*(dw[1]**2-h)
    eps_v = tolerance*(jnp.abs(drift)*h + jnp.sqrt(jnp.maximum(2*dp*h, 0.)))
    ed = jnp.maximum(_ratio(.5*jnp.abs(q*dq)*h*h, eps_v),
                     .5*jnp.abs(pitch)*nu**2*h*h/tolerance)
    # The paragraph after paper Eq. (55) explicitly uses |dW| + sqrt(h/3).
    # Eq. (75)'s printed |dW + sqrt(h/3)| is inconsistent with that text;
    # retain the sign-symmetric estimate rather than cancelling negative dW.
    es = jnp.maximum(_speed_diffusion_error(c, dw[0], eps_v),
                     .5*angular_noise*nu*h*(jnp.abs(dw[1])+jnp.sqrt(h/3))/tolerance)
    return CollisionStep(jnp.array([v, reflect_pitch(xi)]), ed, es)

def velocity_basis(velocity):
    """Orthonormal columns (perp1, perp2, parallel), evaluated at this step."""
    norm = jnp.linalg.norm(velocity)
    direction = velocity/jnp.where(norm > 0, norm, 1.)
    direction = jnp.where(norm > 0, direction, jnp.array([0., 0., 1.]))
    reference = jnp.eye(3, dtype=velocity.dtype)[jnp.argmin(jnp.abs(direction))]
    first = jnp.cross(direction, reference)
    first = first/jnp.linalg.norm(first)
    return jnp.stack([first, jnp.cross(direction, first), direction], axis=1)

def particle_collision_step(velocity, h, dw, c, *, tolerance=1e-3):
    """Paper Eqs. (57)--(62), in the local velocity-aligned basis.

    The parallel component is a Milstein increment; perpendicular components
    use the additive-noise updates in the paper. ``dw`` has variance h.
    """
    speed = jnp.linalg.norm(velocity)
    dp, dt, ddp, q, _, dk = c
    drift = q + ddp + 2*(dp-dt)/speed
    delta = jnp.concatenate([_diffusion_root(2*dt)*dw[:2],
        jnp.reshape(drift*h + _diffusion_root(2*dp)*dw[2]
                    + .5*ddp*(dw[2]**2-h), (1,))])
    eps_v = tolerance*(jnp.abs(drift)*h + jnp.sqrt(jnp.maximum(2*dp*h, 0.)))
    return CollisionStep(velocity + velocity_basis(velocity) @ delta,
        _ratio(.5*jnp.abs(drift*dk)*h*h, eps_v), _speed_diffusion_error(c, dw[2], eps_v))

class BrownianMemory(NamedTuple):
    """Stack of unconsumed intervals: the next interval is at size-1."""
    ends: jax.Array
    increments: jax.Array
    size: jax.Array
    key: jax.Array
    peak: jax.Array
    t: jax.Array

def brownian_memory(key, noise_dim, *, capacity=64, dtype=jnp.float64, t0=0.):
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("Brownian capacity must be a positive integer")
    if isinstance(noise_dim, bool) or not isinstance(noise_dim, int) or noise_dim < 1:
        raise ValueError("noise_dim must be a positive integer")
    return BrownianMemory(jnp.zeros(capacity, dtype), jnp.zeros((capacity, noise_dim), dtype),
                          jnp.array(0), key, jnp.array(0), jnp.asarray(t0, dtype))

def brownian_propose(memory, h, *, t_end=None):
    """Return (memory, h, dW, overflow) without consuming the proposed interval.

    A retry keeps its parent increment. A shorter request replaces it by
    conditional left/right increments (paper Eqs. 83--84). Growth stops at the
    next stored endpoint, so no sampled future value is discarded or skipped.
    Only ``brownian_accept`` consumes an interval. Capacity overflow is reported
    without replacing or discarding any stored increment. Absolute endpoints
    keep roundoff in repeated duration subtraction from stranding an increment.
    ``t_end`` optionally supplies the exact requested endpoint (e.g. a save).
    """
    requested_end = memory.t+h if t_end is None else t_end
    # Creating a parent and splitting a retained parent are exclusive. Share
    # their original-key draw so vmap need not evaluate two normal generators.
    key, draw = jax.random.split(memory.key)
    normal = jax.random.normal(draw, memory.increments.shape[1:], dtype=memory.increments.dtype)
    def new(m):
        dw = jnp.sqrt(requested_end-m.t)*normal
        return BrownianMemory(m.ends.at[0].set(requested_end), m.increments.at[0].set(dw),
                              jnp.array(1), key, jnp.maximum(m.peak, 1), m.t)
    m = lax.cond(memory.size == 0, new, lambda m: m, memory)
    top = m.size-1
    parent_end, parent = m.ends[top], m.increments[top]
    duration = parent_end-m.t
    end = jnp.minimum(requested_end, parent_end)
    h = end-m.t
    split = end < parent_end
    overflow = split & (m.size == len(m.ends))

    def bridge(m):
        alpha = h/duration
        left = alpha*parent + jnp.sqrt(h*(1-alpha))*normal
        # Parent = left + right, including across repeated nested rejections.
        return BrownianMemory(m.ends.at[top+1].set(end),
            m.increments.at[top].set(parent-left).at[top+1].set(left), m.size+1, key,
            jnp.maximum(m.peak, m.size+1), m.t)
    m = lax.cond(split & ~overflow, bridge, lambda m: m, m)
    return m, h, m.increments[m.size-1], overflow

def brownian_accept(memory):
    return memory._replace(size=memory.size-1, t=memory.ends[memory.size-1])

class SplitStatus(IntEnum):
    SUCCESS = 0
    EVENT = 1
    MAX_STEPS = 2
    MIN_STEP = 3
    NONFINITE = 4
    INVALID_INPUT = 5
    BROWNIAN_CAPACITY = 6

class SplitStats(NamedTuple):
    steps: jax.Array
    accepted: jax.Array
    rejected: jax.Array
    orbit_steps: jax.Array
    orbit_rejected: jax.Array
    peak_pending: jax.Array
    max_accepted_error: jax.Array
    accepted_orbit_steps: jax.Array
    max_orbit_steps_per_collision: jax.Array

class SplitSolution(NamedTuple):
    ts: jax.Array
    ys: jax.Array
    t: jax.Array
    y: jax.Array
    status: jax.Array
    stats: SplitStats
    step_ts: jax.Array

def solve_split(rhs, collision, y0, args=None, *, t0=0.0, t1, dt0, ts, key, noise_dim, adaptive=True, dtmin=0.0, dtmax=jnp.inf, method='dopri8', orbit_dt0=None, orbit_adaptive=True, orbit_stage_execution='loop', reuse_orbit_step=True, rtol=1e-09, atol=1e-09, error_norm=None, controller='i', pcoeff=0.4, icoeff=0.3, event=None, max_steps=100000, max_orbit_steps=10000, max_pending=64, record_steps=False, step_limit=None, align_orbit=False, event_dtmax=None, _orbit_rhs_prepared=False):
    """Orbit advancement followed by an endpoint collision kick (Lie splitting).

    ``collision(t, y, h, dW, args)`` returns a ``CollisionStep`` with normalized
    drift/diffusion error estimates, acceptable when <= 1. Rejected trials repeat
    from the last accepted state and retain every Brownian sample (bridge
    conditioning, paper Eqs. 83-84), so refining a step never redraws the path.
    The next interval is 0.9*min(E_drift**(-1/2), E_diff**(-2/3)), capped at 1.5.

    ``dt0``/``dtmax`` control collision/splitting intervals; ``orbit_dt0``,
    rtol/atol and ``method`` control the independent RK solve within each
    interval. Collision estimates do not bound splitting error: refine dtmax to
    check it. Saves are exact split boundaries, so changing ts changes the
    trajectory. ``reuse_orbit_step`` carries the RK step-size guess across kicks;
    ``align_orbit`` snaps a collision endpoint to a nearby orbit step.

    Events are scalar booleans or positive-inside level sets (or PyTrees of them).
    Orbit crossings are root-found before a kick; a kick crossing stops at its
    endpoint. Optional ``step_limit(t, y, args)`` supplies a positive interval
    ceiling from the current accepted state (it must not depend on fresh noise);
    ``event_dtmax`` retries a detected kick crossing until its interval is this
    small. Stopped or failed outputs hold the last finite state; inspect
    ``status``. Work and Brownian memory are bounded and no inaccurate step is
    forced through ``dtmin`` or a memory limit."""
    if align_orbit and (not adaptive or not orbit_adaptive or not reuse_orbit_step):
        raise ValueError('orbit endpoint alignment requires adaptive orbit and collision clocks and orbit-step reuse')
    for name, value in (('max_steps', max_steps), ('max_orbit_steps', max_orbit_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if orbit_stage_execution not in ('loop', 'unrolled'):
        raise ValueError("orbit_stage_execution must be 'loop' or 'unrolled'")
    y0 = jnp.asarray(y0, dtype=jnp.result_type(y0, float))
    ts = jnp.asarray(ts, dtype=y0.dtype)
    if y0.ndim != 1 or y0.size == 0 or (not jnp.issubdtype(y0.dtype, jnp.floating)):
        raise ValueError('y0 must be a nonempty real vector')
    if ts.ndim != 1:
        raise ValueError('ts must be one-dimensional')
    orbit_dt0 = dt0 if orbit_dt0 is None else orbit_dt0
    for name, value in (('t0', t0), ('t1', t1), ('dt0', dt0), ('dtmin', dtmin), ('dtmax', dtmax), ('orbit_dt0', orbit_dt0)):
        if jnp.ndim(value) != 0:
            raise ValueError(f'{name} must be scalar')
    t0, t1, dt0, dtmin, dtmax, orbit_dt0 = (jnp.asarray(x, dtype=y0.dtype) for x in (t0, t1, dt0, dtmin, dtmax, orbit_dt0))
    orbit_rhs = rhs if _orbit_rhs_prepared else _prepare_rhs(rhs, t0, y0, args)
    if event_dtmax is not None:
        if event is None or jnp.ndim(event_dtmax) != 0:
            raise ValueError('event_dtmax requires an event and a scalar interval')
        event_dtmax = jnp.asarray(event_dtmax, dtype=y0.dtype)
    valid = jnp.isfinite(t0) & jnp.isfinite(t1) & (t1 >= t0) & jnp.isfinite(dt0) & (dt0 > 0) & jnp.isfinite(dtmin) & (dtmin >= 0) & (dt0 >= dtmin) & (dtmax > 0) & (dtmax >= dtmin) & jnp.isfinite(orbit_dt0) & (orbit_dt0 > 0) & jnp.all(jnp.isfinite(ts) & (ts >= t0) & (ts <= t1)) & jnp.all(jnp.diff(ts) >= 0)
    if event_dtmax is not None:
        valid &= jnp.isfinite(event_dtmax) & (event_dtmax > 0) & (event_dtmax >= dtmin)
    _concrete_check(valid, np.all, 'invalid split times or step limits')
    functions = tree_util.tree_leaves(event)

    def values(t, y):
        out = tuple((jnp.asarray(f(t, y, args)) for f in functions))
        if any((x.shape != () or (x.dtype != jnp.bool_ and (not jnp.issubdtype(x.dtype, jnp.floating))) for x in out)):
            raise ValueError('events must return scalar booleans or floating level sets')
        return out

    def stopped(v):
        return jnp.any(jnp.asarray([x if x.dtype == jnp.bool_ else x <= 0 for x in v], dtype=bool))

    def finite_values(v):
        return jnp.all(jnp.asarray([jnp.isfinite(x) for x in v], dtype=bool))
    v0 = values(t0, y0)
    initial_status = jnp.where(~valid, SplitStatus.INVALID_INPUT, jnp.where(~jnp.isfinite(y0).all() | ~finite_values(v0), SplitStatus.NONFINITE, jnp.where(stopped(v0), SplitStatus.EVENT, SplitStatus.SUCCESS)))
    memory = brownian_memory(key, noise_dim, capacity=max_pending, dtype=y0.dtype, t0=t0)
    s = dict(t=t0, y=y0, h=jnp.minimum(dt0, dtmax), orbit_h=orbit_dt0, status=initial_status, memory=memory, index=jnp.array(0), ys=jnp.broadcast_to(y0, (len(ts), y0.size)), steps=jnp.array(0), accepted=jnp.array(0), rejected=jnp.array(0), orbit_steps=jnp.array(0), orbit_rejected=jnp.array(0), max_error=jnp.zeros((), y0.dtype), history=jnp.full((max_steps + 1 if record_steps else 0,), jnp.inf, dtype=y0.dtype), accepted_orbit_steps=jnp.array(0), max_orbit_steps_per_collision=jnp.array(0))
    if record_steps:
        s['history'] = s['history'].at[0].set(t0)
    if not adaptive:
        s['grid_index'] = jnp.array(0)
    if event_dtmax is not None:
        s['event_h'] = jnp.asarray(jnp.inf, dtype=y0.dtype)

    def next_save(s):
        return jnp.where(s['index'] < len(ts), ts[jnp.minimum(s['index'], len(ts) - 1)], t1) if len(ts) else t1

    def save(s):
        if not len(ts):
            return s

        def one(s):
            return dict(s, ys=s['ys'].at[s['index']].set(s['y']), index=s['index'] + 1)
        return lax.while_loop(lambda s: (s['index'] < len(ts)) & (ts[jnp.minimum(s['index'], len(ts) - 1)] <= s['t']), one, s)
    s = save(s)

    def integrate(t, y, end, guess):
        return solve_rk(orbit_rhs, y, args, t0=t, t1=end, dt0=jnp.minimum(guess, end-t), ts=jnp.empty((0,), y0.dtype), method=method, adaptive=orbit_adaptive, rtol=rtol, atol=atol, stage_execution=orbit_stage_execution, error_norm=error_norm, controller=controller, pcoeff=pcoeff, icoeff=icoeff, event=event, root_find=True, max_steps=max_orbit_steps)

    def kick(t, y, h, dw):
        result = collision(t, y, h, dw, args)
        if result.y.shape != y0.shape:
            raise ValueError('collision must preserve the state shape')
        if jnp.ndim(result.drift_error) or jnp.ndim(result.diffusion_error):
            raise ValueError('collision errors must be scalars')
        return result

    def valid_trial(trial):
        errors = jnp.asarray([trial.drift_error, trial.diffusion_error])
        return jnp.isfinite(trial.y).all() & jnp.isfinite(errors).all() & jnp.all(errors >= 0)

    def advance_with_limit(s, limit):
        physical_barrier = next_save(s)
        barrier = physical_barrier
        if step_limit is not None:
            barrier = jnp.minimum(barrier, s['t'] + limit)
        if event_dtmax is not None:
            barrier = jnp.minimum(barrier, s['t'] + s['event_h'])
        if adaptive:
            wanted_end = jnp.minimum(s['t'] + s['h'], barrier)
        else:
            grid_end = t0 + (s['grid_index'] + 1) * jnp.minimum(dt0, dtmax)
            close = jnp.abs(grid_end - barrier) <= 8 * jnp.finfo(y0.dtype).eps * jnp.maximum(jnp.abs(grid_end), jnp.abs(barrier))
            grid_end = jnp.where(close, barrier, grid_end)
            wanted_end = jnp.minimum(grid_end, barrier)
        if align_orbit:
            length = wanted_end - s['t']
            guess = s['orbit_h']
            near = (jnp.isfinite(guess) & (guess > 0) & (wanted_end < physical_barrier)
                    & (guess >= .9*length) & (guess < length*(1-1e-12)))
            wanted_end = jnp.where(near, s['t'] + guess, wanted_end)
        m, h, dw, overflow = brownian_propose(s['memory'], wanted_end - s['t'], t_end=wanted_end)
        end = m.ends[m.size - 1]
        usable = ~overflow & (h > 0) & (end > s['t'])

        def attempt(_):
            guess = s['orbit_h'] if reuse_orbit_step and orbit_adaptive else orbit_dt0
            orbit = integrate(s['t'], s['y'], end, guess)
            trial = lax.cond(orbit.status == RKStatus.SUCCESS, lambda _: kick(end, orbit.y, h, dw), lambda _: CollisionStep(orbit.y, jnp.zeros((), y0.dtype), jnp.zeros((), y0.dtype)), None)
            v = values(end, trial.y)
            finite = valid_trial(trial) & finite_values(v)
            error = jnp.maximum(trial.drift_error, trial.diffusion_error)
            accepted = (orbit.status == RKStatus.SUCCESS) & finite & ((error <= 1) | (not adaptive))
            refine_event = jnp.array(False)
            if event_dtmax is not None:
                refine_event = accepted & stopped(v) & (h > event_dtmax * (1 + 8 * jnp.finfo(y0.dtype).eps))
                accepted &= ~refine_event
            rejected = (orbit.status == RKStatus.SUCCESS) & finite & ~accepted
            orbit_event = orbit.status == RKStatus.EVENT
            status = jnp.where(orbit.status != RKStatus.SUCCESS, orbit.status, jnp.where(~finite, SplitStatus.NONFINITE, jnp.where(accepted & stopped(v), SplitStatus.EVENT, jnp.where(rejected & (h <= dtmin), SplitStatus.MIN_STEP, SplitStatus.SUCCESS))))
            if adaptive:
                factor = 0.9 * jnp.minimum(jnp.maximum(trial.drift_error, 1e-30) ** (-0.5), jnp.maximum(trial.diffusion_error, 1e-30) ** (-2 / 3))
                factor = jnp.clip(factor, 0.1, jnp.where(accepted, 1.5, 0.9))
                next_h = jnp.clip(h * factor, dtmin, dtmax)
            else:
                next_h = jnp.minimum(dt0, dtmax)
            path_steps = jnp.where(accepted, orbit.stats.accepted, 0).astype(s['accepted'].dtype)
            state = dict(s, t=jnp.where(orbit_event, orbit.t, jnp.where(accepted, end, s['t'])), y=jnp.where(orbit_event, orbit.y, jnp.where(accepted, trial.y, s['y'])), h=next_h, status=status, steps=s['steps'] + 1, accepted=s['accepted'] + accepted.astype(s['accepted'].dtype), rejected=s['rejected'] + rejected.astype(s['rejected'].dtype), orbit_steps=s['orbit_steps'] + orbit.stats.steps, orbit_rejected=s['orbit_rejected'] + orbit.stats.rejected, memory=lax.cond(accepted, brownian_accept, lambda m: m, m), max_error=jnp.maximum(s['max_error'], jnp.where(accepted, error, 0.0)), accepted_orbit_steps=s['accepted_orbit_steps'] + path_steps, max_orbit_steps_per_collision=jnp.maximum(s['max_orbit_steps_per_collision'], path_steps))
            if event_dtmax is not None:
                state['event_h'] = jnp.where(refine_event, jnp.maximum(dtmin, 0.5 * h), jnp.inf)
                state['h'] = jnp.where(refine_event, state['event_h'], state['h'])
            if reuse_orbit_step and orbit_adaptive:
                state['orbit_h'] = orbit.next_dt
            if not adaptive:
                state['grid_index'] = s['grid_index'] + (accepted & (end == grid_end)).astype(s['grid_index'].dtype)
            if record_steps:
                state['history'] = lax.cond(accepted, lambda history: history.at[state['accepted']].set(end), lambda history: history, state['history'])
            state = save(state)
            return state
        return lax.cond(usable, attempt, lambda _: dict(s, memory=m, status=jnp.where(overflow, SplitStatus.BROWNIAN_CAPACITY, SplitStatus.MIN_STEP)), None)

    def advance(s):
        if step_limit is None:
            return advance_with_limit(s, None)
        limit = jnp.asarray(step_limit(s['t'], s['y'], args), dtype=y0.dtype)
        if limit.ndim != 0:
            raise ValueError('step_limit must return a scalar interval')
        return lax.cond((limit > 0) & ~jnp.isnan(limit), lambda _: advance_with_limit(s, limit), lambda _: dict(s, status=jnp.asarray(SplitStatus.INVALID_INPUT)), None)
    s = lax.while_loop(lambda s: (s['status'] == SplitStatus.SUCCESS) & (s['t'] < t1) & (s['steps'] < max_steps), advance, s)
    status = jnp.where((s['status'] == SplitStatus.SUCCESS) & (s['t'] < t1), SplitStatus.MAX_STEPS, s['status'])
    ys = jnp.where((jnp.arange(len(ts)) >= s['index'])[:, None], s['y'], s['ys'])
    stats = SplitStats(s['steps'], s['accepted'], s['rejected'], s['orbit_steps'], s['orbit_rejected'], s['memory'].peak, s['max_error'], s['accepted_orbit_steps'], s['max_orbit_steps_per_collision'])
    solution = SplitSolution(ts, ys, s['t'], s['y'], status, stats, s['history'])
    return solution

def trace_split(field, particles, species, *, model='GuidingCenter', tmax, timestep, times=None, n_save=100, integrator='essos_dopri8', collision_tolerance=0.001, adaptive=True, collision_dtmax=None, orbit_dt0=None, orbit_adaptive=True, orbit_stage_execution=None, reuse_orbit_step=True, collision_controller='moments', moment_tolerance=.005, moment_reference_time=.005, rtol=1e-09, atol=1e-09, controller='i', pcoeff=0.4, icoeff=0.3, speed_floor=None, spatial_diffusion=None, spatial_geometry=None, orbit=True, electric_field=None, event=None, max_steps=100000, max_orbit_steps=10000, max_pending=64, record_steps=False, step_limit=None):
    """Batched split tracing using ESSOS field and particle objects.

    Returns a ``SplitSolution`` with shared ts and particle-leading arrays. GC states
    are ``(position[3], v_parallel, mu)`` with **physical** mu=m*v_perp^2/(2 B)
    [J/T], unlike the legacy normalized collisional state. FullOrbit states are
    ``(xyz, vx, vy, vz)``. VMEC/Boozer positions are (s, theta, zeta); both stop at
    the LCFS and ``event`` receives public physical states.

    GC includes the paper Eq. (69) perpendicular spatial diffusion by default
    (``spatial_diffusion=False`` isolates the velocity operator); FullOrbit uses its
    particle collision map. VMEC uses its own spatial geometry; Boozer requires
    ``spatial_geometry=BoozerGeometry.from_booz_xform(b)``. Spatial kicks are inverted
    nonlinearly in the regular chart, preserving the Cartesian update. ``orbit=False``
    runs collision-only statistics. ``speed_floor`` defaults to 5% of the smallest
    active background thermal speed. Per-particle keys come from ``particles.random_keys``.

    ``collision_controller="moments"`` (default, adaptive only) adds a coupled
    mean/covariance interval ceiling with local budget ``moment_tolerance``, on top of
    the collision checks at ``collision_tolerance`` and orbit checks at ``rtol/atol``;
    none of these is a global error bound. ``"conservative"`` keeps only the collision
    checks. An explicit ``step_limit(t, y, args)`` [s] replaces the moment predictor.
    """
    from .boozer import BoozerField, guiding_center_rhs
    from .coils import Coils
    from .fields import BiotSavart, Vmec
    from .dynamics import _guiding_center_velocity, _axis_regular, _to_axis_regular, _from_axis_regular
    from .electric_field import Electric_field_zero
    from .constants import ELEMENTARY_CHARGE

    if integrator not in ('essos_rk45', 'essos_dopri8'):
        raise ValueError("split integrator must be 'essos_rk45' or 'essos_dopri8'")
    if model not in ('GuidingCenter', 'FullOrbit'):
        raise ValueError("split model must be 'GuidingCenter' or 'FullOrbit'")
    if collision_controller not in ('moments', 'conservative'):
        raise ValueError("collision_controller must be 'moments' or 'conservative'")
    moment_control = collision_controller == 'moments' and adaptive and step_limit is None
    for name,value in (('moment_tolerance',moment_tolerance),('moment_reference_time',moment_reference_time)):
        _concrete_check(value,lambda x:np.ndim(x)==0 and np.isfinite(x) and x>0,name+' must be positive and finite')
    _concrete_check(collision_tolerance, lambda x: np.ndim(x) == 0 and np.isfinite(x) and (x > 0), 'collision_tolerance must be positive and finite')
    if species is None:
        raise ValueError('split collisions require BackgroundSpecies')
    if species.number_species < 1:
        raise ValueError('BackgroundSpecies must contain at least one species')
    field = BiotSavart(field) if isinstance(field, Coils) else field
    electric_field = Electric_field_zero() if electric_field is None else electric_field
    boozer, vmec = (isinstance(field, BoozerField), isinstance(field, Vmec))
    if step_limit is not None and (boozer or vmec):
        raise ValueError('step_limit currently requires shared Cartesian tracing')
    if orbit_stage_execution is None:
        orbit_stage_execution = 'unrolled' if integrator == 'essos_dopri8' and model == 'GuidingCenter' and isinstance(field, BiotSavart) and (not adaptive) and (not orbit_adaptive) and (jax.default_backend() == 'gpu') else 'loop'
    gc = model == 'GuidingCenter'
    spatial_diffusion=gc if spatial_diffusion is None else spatial_diffusion
    if not gc and (boozer or vmec):
        raise ValueError('FullOrbit requires a Cartesian field')
    if spatial_diffusion and (not gc or (boozer and spatial_geometry is None)):
        raise ValueError('spatial_diffusion requires GC and Cartesian spatial_geometry for Boozer')
    if boozer and (not isinstance(electric_field, Electric_field_zero)):
        raise ValueError('Boozer split tracing does not yet support electric fields')
    mass, charge = (particles.mass, particles.charge)
    _concrete_check(mass, lambda x: np.isfinite(x) and x > 0, 'mass must be positive and finite')
    _concrete_check(charge, lambda x: np.isfinite(x) and x != 0, 'charge must be finite and nonzero')
    if times is None:
        if isinstance(n_save, bool) or not isinstance(n_save, int) or n_save < 2:
            raise ValueError('n_save must be an integer >= 2')
        times = jnp.linspace(0.0, tmax, n_save)
    times = jnp.asarray(times)
    if times.ndim != 1:
        raise ValueError('times must be one-dimensional')
    moment_vmec_data=None
    if vmec and moment_control:
        with jax.ensure_compile_time_eval():
            moment_vmec_data=_prepare_vmec_data(field)

    def magnitude(point,source=field):
        return source.modB(*point) if boozer else source.AbsB(point)

    def public(y):
        if vmec:
            return _from_axis_regular(y)
        if boozer:
            return y.at[0].set(y[0] ** 2 + y[1] ** 2).at[1].set(jnp.arctan2(y[1], y[0]))
        return y

    def spatial_position(y,source=field):
        if boozer:
            geometry=getattr(source,'to_xyz',spatial_geometry)
            if hasattr(geometry,'regular_xyz'):return geometry.regular_xyz(y[:3])
            return geometry(public(y)[:3])
        return source.to_xyz(public(y)[:3]) if vmec else y[:3]

    def spatial_direction(y,B,source):
        if vmec:
            direction=_axis_regular(lambda t,q,a:jnp.concatenate((source.B_contravariant(q[:3]),jnp.zeros_like(q[3:]))))(0.,y,None)
            tangent=jax.jvp(lambda q:spatial_position(q,source),(y,),(direction,))[1]
            return tangent/jnp.linalg.norm(tangent)
        if boozer:
            (iota,G,I),_=source.profiles(public(y)[0])
            direction=jnp.zeros_like(y).at[:3].set(jnp.array([-y[1]*iota,y[0]*iota,1.]))
            tangent=jax.jvp(lambda q:spatial_position(q,source),(y,),(direction,))[1]
            return jnp.sign(G+iota*I)*tangent/jnp.linalg.norm(tangent)
        return source.B_contravariant(y[:3])/B

    def gc_drift(t, y, args,source=field):
        dx, dv = _guiding_center_velocity(source, electric_field, charge, mass, y[:3], y[3], mu=y[4])
        return jnp.concatenate([dx, jnp.array([dv, 0.0])])

    def rhs(t, y, args,source=field):
        if not orbit:
            return jnp.zeros_like(y)
        if boozer:
            return jnp.append(guiding_center_rhs(source, y[:4], y[4] / mass, mass, charge), 0.0)
        if vmec:
            return _axis_regular(lambda t,y,a:gc_drift(t,y,a,source))(t, y, args)
        if gc:
            return gc_drift(t, y, args,source)
        return jnp.concatenate([y[3:], charge / mass * (electric_field.E_covariant(y[:3]) + jnp.cross(y[3:], source.B_contravariant(y[:3])))])
    if gc:
        points = jnp.asarray(particles.initial_xyz)
        b0 = jax.vmap(magnitude)(points)
        _concrete_check(b0, lambda x: np.all(np.isfinite(x) & (x > 0)), 'birth |B| must be positive and finite')
        _concrete_check(particles.initial_vparallel_over_v, lambda x: np.all(np.isfinite(x) & (np.abs(x) <= 1)), 'birth pitch must lie in [-1, 1]')
        y0 = jnp.concatenate([points, particles.initial_vparallel[:, None], (mass * particles.initial_vperpendicular ** 2 / (2 * b0))[:, None]], axis=1)
        if vmec:
            y0 = jax.vmap(_to_axis_regular)(y0)
        elif boozer:
            _concrete_check(points[:, 0], lambda x: np.all(np.isfinite(x) & (x >= 0)), 'birth s must be nonnegative')
            r = jnp.sqrt(points[:, 0])
            y0 = y0.at[:, 0].set(r * jnp.cos(points[:, 1])).at[:, 1].set(r * jnp.sin(points[:, 1]))
    else:
        if particles.initial_xyz_fullorbit is None or particles.initial_vxvyvz is None:
            raise ValueError('FullOrbit requires initial full-orbit positions and velocities')
        y0 = jnp.concatenate([jnp.asarray(particles.initial_xyz_fullorbit), jnp.asarray(particles.initial_vxvyvz)], axis=1)
    if y0.shape[0]:
        rhs = _prepare_rhs(rhs, jnp.zeros((), y0.dtype), y0[0], None)
    def one(initial, key):
        birth = public(initial)
        birth_speed = jnp.sqrt(birth[3] ** 2 + 2 * birth[4] * magnitude(birth[:3]) / mass) if gc else jnp.linalg.norm(birth[3:])
        if speed_floor is None:
            thermal = jnp.stack([jnp.where(species.get_density(i, birth[:3]) > 0, jnp.sqrt(2 * ELEMENTARY_CHARGE * species.get_temperature(i, birth[:3]) / mass), jnp.inf) for i in range(species.number_species)])
            floor = 0.05 * jnp.min(thermal)
            floor = jnp.where(jnp.isfinite(floor), floor, birth_speed * 1e-12)
        else:
            floor = jnp.asarray(speed_floor, dtype=initial.dtype)

        def kick(t, y, h, dw, args,source=field):
            state = public(y)
            point = state[:3]
            B = magnitude(point,source) if gc else jnp.ones((), y.dtype)
            speed = jnp.sqrt(state[3] ** 2 + 2 * state[4] * B / mass) if gc else jnp.linalg.norm(state[3:])
            c = collision_coefficients(species, mass, charge, speed, point, full_orbit=not gc)
            if gc:
                step = guiding_center_collision_step(speed, state[3] / speed, h, dw[-2:], c, tolerance=collision_tolerance, speed_floor=floor)
                v, pitch = step.y
                if spatial_diffusion:
                    displacement=guiding_center_spatial_increment(state[3]/speed,dw[:3],c,
                        spatial_direction(y,B,source),charge*B/mass)
                    if vmec or boozer:
                        y=move_in_cartesian(y,displacement,lambda q:spatial_position(q,source))
                        point=public(y)[:3]
                    else:
                        point=point+displacement;y=y.at[:3].set(point)
                    B = magnitude(point,source)
                result = y.at[3].set(v * pitch).at[4].set(mass * v * v * (1 - pitch * pitch) / (2 * B))
            else:
                step = particle_collision_step(state[3:], h, dw, c, tolerance=collision_tolerance)
                result = y.at[3:].set(step.y)
            valid_c = jnp.isfinite(jnp.asarray(c)).all() & (c.parallel >= 0) & (c.perpendicular >= 0)
            result = jnp.where(jnp.isfinite(B) & (B > 0) & (speed > 0) & valid_c, result, jnp.full_like(result, jnp.nan))
            return step._replace(y=result)

        def position(y):
            point = public(y)[:3]
            return field.to_xyz(point) if vmec else point

        active_limit=step_limit
        if moment_control:
            moment_position=position
            if boozer:
                moment_position=lambda y:y[:3]
            elif vmec:
                from .rk import _vmec_xyz_values
                prepared=moment_vmec_data
                moment_position=lambda y:jnp.asarray(_vmec_xyz_values(prepared.arrays,y[jnp.array([0,1,2,3,5])],prepared.tables))
            active_limit=native_step_limit(field,species,mass,charge,initial,rhs,public,moment_position,magnitude,
                gc=gc,coordinate=vmec or boozer,boozer=boozer,
                electric_field=electric_field,tmax=tmax,dtmax=timestep if collision_dtmax is None else collision_dtmax,
                tolerance=moment_tolerance,reference_time=moment_reference_time,orbit=orbit)

        if boozer:

            def norm(a, b, error):
                return jnp.maximum(jnp.linalg.norm(error[:2]), jnp.maximum(jnp.abs(error[2]), jnp.abs(error[3]) / birth_speed)) / (rtol + atol)
        else:
            norm = OrbitErrorNorm(rtol=rtol, atol=atol, speed_scale=birth_speed, velocity_components=1 if gc else 3, position_map=position if vmec else None)
        events = []
        if vmec or boozer:
            events.append(lambda t, y, args: 1 - y[0] ** 2 - y[1] ** 2)
        if event is not None:
            events.extend([f if not (vmec or boozer) else
                lambda t, y, args, f=f: f(t, public(y), args) for f in tree_util.tree_leaves(event)])
        solution = solve_split(rhs, kick, initial, t1=tmax, dt0=timestep, ts=times, key=key, noise_dim=(5 if spatial_diffusion else 2) if gc else 3, dtmax=timestep if collision_dtmax is None else collision_dtmax, adaptive=adaptive, orbit_dt0=orbit_dt0, orbit_adaptive=orbit_adaptive, orbit_stage_execution=orbit_stage_execution, reuse_orbit_step=reuse_orbit_step, method='dopri5' if integrator == 'essos_rk45' else 'dopri8', rtol=rtol, atol=atol, controller=controller, pcoeff=pcoeff, icoeff=icoeff, error_norm=norm, step_limit=active_limit, align_orbit=moment_control and orbit_adaptive and reuse_orbit_step, event=tuple(events), max_steps=max_steps, max_orbit_steps=max_orbit_steps, max_pending=max_pending, record_steps=record_steps, _orbit_rhs_prepared=True)
        return solution._replace(ys=jax.vmap(public)(solution.ys), y=public(solution.y))
    if speed_floor is not None:
        _concrete_check(speed_floor, lambda x: np.ndim(x) == 0 and np.isfinite(x) and (x > 0), 'speed_floor must be positive and finite')
    return jax.jit(jax.vmap(one))(y0, particles.random_keys)._replace(ts=times)


# Spatial geometry

def move_in_cartesian(state, displacement, position):
    """Invert the Cartesian Euler kick in the existing regular position chart.

    Solving the nonlinear map retains its Ito coordinate drift automatically;
    multiplying the noise by an inverse Jacobian alone would omit that drift.
    Velocity and the VMEC poloidal-angle gauge are held fixed. A failed inverse
    returns NaNs so the driver reports failure instead of dropping the kick.
    """
    target=position(state)+displacement
    mapping=lambda point:position(state.at[:3].set(point))
    threshold=64*jnp.finfo(state.dtype).eps
    def advance(carry):
        point,_,iteration=carry
        residual=mapping(point)-target
        matrix=jax.jacfwd(mapping)(point)
        a,b,c=matrix[:,0],matrix[:,1],matrix[:,2]
        rows=jnp.stack((jnp.cross(b,c),jnp.cross(c,a),jnp.cross(a,b)))
        point=point-(rows@residual)/jnp.dot(a,rows[0])
        scale=jnp.maximum(1.,jnp.maximum(jnp.linalg.norm(target),
            jnp.linalg.norm(matrix)*jnp.maximum(1.,jnp.linalg.norm(point))))
        return point,jnp.linalg.norm(mapping(point)-target)/scale,iteration+1
    point,error,_=jax.lax.while_loop(lambda c:((c[2]==0)|(c[1]>threshold))&(c[2]<8),advance,
        (state[:3],jnp.linalg.norm(displacement),jnp.asarray(0)))
    return state.at[:3].set(jnp.where(error<=threshold,point,jnp.full_like(point,jnp.nan)))

class BoozerGeometry(NamedTuple):
    """Spatial geometry exported alongside Boozer magnetic tables.

    The toroidal shift is nu=zeta_B-phi, following booz_xform's numn*_b
    convention. Geometry modes are retained independently of magnetic cuts.
    """
    knots: object
    coefficients: object
    xm: object
    xn: object

    @classmethod
    def from_booz_xform(cls,booz):
        from .boozer import _spline
        s=np.asarray(booz.s_b,float);m=np.asarray(booz.xm_b,int)
        if s.ndim!=1 or len(s)<2 or np.any(~np.isfinite(s)) or np.any(s<=0) or np.any(np.diff(s)<=0):
            raise ValueError('Boozer geometry requires at least two increasing positive surfaces')
        shape=(len(m),len(s))
        def spectrum(name):
            value=getattr(booz,name,None)
            if value is None or not np.size(value):return np.zeros(shape)
            value=np.asarray(value,float)
            if value.shape!=shape:raise ValueError(name+' must have mode/surface axes')
            return value
        for name in ('rmnc_b','zmns_b','numns_b'):
            if getattr(booz,name,None) is None or not np.size(getattr(booz,name,None)):
                raise ValueError('Boozer spatial diffusion requires geometry: '+name)
        values=np.stack([spectrum(name) for name in
            ('rmnc_b','rmns_b','zmnc_b','zmns_b','numnc_b','numns_b')],axis=-1)
        # A mode m scales as r**m at the axis. Evaluate its angular factor as
        # a polynomial in u+iw rather than differentiating atan2 there.
        powers=np.minimum(m,2+m%2)
        scaled=np.moveaxis(values,1,0)/np.sqrt(s)[:,None,None]**powers[None,:,None]
        axis=scaled[0]-s[0]*(scaled[1]-scaled[0])/(s[1]-s[0])
        knots,coefficients=_spline(np.r_[0.,s],np.concatenate((axis[None],scaled)))
        return cls(knots,coefficients,jnp.asarray(m),jnp.asarray(booz.xn_b))

    def regular_xyz(self,point):
        """Cartesian position from (sqrt(s)cos(theta),sqrt(s)sin(theta),zeta)."""
        from .boozer import _evaluate
        u,w,zeta=point;s=u*u+w*w
        amplitude=_evaluate(self.knots,self.coefficients,s)[0]
        # Recur on complex harmonics using real operations supported by Pallas.
        real,imag=jnp.ones_like(self.xm,dtype=u.dtype),jnp.zeros_like(self.xm,dtype=u.dtype)
        for k in range(int(np.max(np.asarray(self.xm)))):
            active=k<self.xm
            real,imag=jnp.where(active,real*u-imag*w,real),jnp.where(active,imag*u+real*w,imag)
        powers=jnp.minimum(self.xm,2+self.xm%2)
        factor=jnp.where(self.xm>powers,jnp.maximum(s,1e-30)**((powers-self.xm)/2),1.)
        c,t=jnp.cos(self.xn*zeta),jnp.sin(self.xn*zeta)
        cosine,sine=(real*c+imag*t)*factor,(imag*c-real*t)*factor
        R=jnp.sum(amplitude[:,0]*cosine+amplitude[:,1]*sine)
        Z=jnp.sum(amplitude[:,2]*cosine+amplitude[:,3]*sine)
        shift=jnp.sum(amplitude[:,4]*cosine+amplitude[:,5]*sine)
        return jnp.array([R*jnp.cos(zeta-shift),R*jnp.sin(zeta-shift),Z])

    def __call__(self,point):
        r=jnp.sqrt(point[0])
        return self.regular_xyz(jnp.array([r*jnp.cos(point[1]),r*jnp.sin(point[1]),point[2]]))

    def _kernel_view(self):
        """Keep geometry-table reads vectorized inside the independent kernel."""
        from .rk import _select_one
        nr,nm=(1<<(n-1).bit_length() for n in (len(self.knots),len(self.xm)))
        knots=np.full(nr,np.inf);knots[:len(self.knots)]=np.asarray(self.knots)
        coefficients=np.zeros((nr,nm,8,4));c=np.asarray(self.coefficients)
        coefficients[:c.shape[0],:c.shape[1],:6]=c
        arrays=tuple(map(jnp.asarray,(knots,coefficients,np.pad(np.asarray(self.xm),(0,nm-len(self.xm))),
            np.pad(np.asarray(self.xn),(0,nm-len(self.xn))))))
        def evaluate(refs,s,theta,zeta):
            grid,table,m,n=refs
            i=jnp.clip(jnp.sum((s>=grid[...]).astype(jnp.int32))-1,0,len(self.knots)-2)
            d=s-grid[i]
            amplitude=((table[i,:,:,0]*d+table[i,:,:,1])*d+table[i,:,:,2])*d+table[i,:,:,3]
            power=jnp.minimum(m[...],2+m[...]%2);root=jnp.sqrt(s)
            factor=jnp.where(power==0,1.,jnp.where(power==1,root,jnp.where(power==2,s,s*root)))
            phase=m[...]*theta-n[...]*zeta;cosine,sine=factor*jnp.cos(phase),factor*jnp.sin(phase)
            column=lambda i:_select_one(amplitude,i,1).reshape((nm,))
            values=tuple(jnp.sum(column(i)*cosine+column(i+1)*sine) for i in (0,2,4))
            R,Z,shift=values
            return R*jnp.cos(zeta-shift),R*jnp.sin(zeta-shift),Z
        return arrays,evaluate


# Moment predictor

class CollisionJet(NamedTuple):
    drift: object
    noise: object
    drift_jacobian: object
    drift_hessian: object
    variance_gradient: object
    variance_hessian: object
    spatial_drift: object
    spatial_variance: object

def collision_jet(radial,r,xi,point,direction):
    """Analytic pitch derivatives of drift/variance; two radial derivatives.

    Differentiate Q=g*g directly: Q is smooth at pitch endpoints even though
    derivatives of sqrt(1-xi*xi) are singular. Profile convection is retained.
    """
    values,spatial=jax.jvp(lambda x:radial(r,x),(point,),(direction,))
    first=jax.jacfwd(radial)(r,point)
    second=jax.jacfwd(jax.jacfwd(radial))(r,point)
    a,b,d=values;da,db,dd=first;dda,ddb,ddd=second
    drift=jnp.array([a,-xi*d*d]);noise=jnp.array([b,jnp.sqrt(jnp.maximum(0.,1-xi*xi))*d])
    aj=jnp.array([[da,0.],[-2*xi*d*dd,-d*d]])
    ah=jnp.array([[[dda,0.],[0.,0.]],
        [[-2*xi*(dd*dd+d*ddd),-2*d*dd],[-2*d*dd,0.]]])
    qj=jnp.array([[2*b*db,0.],[2*(1-xi*xi)*d*dd,-2*xi*d*d]])
    qh=jnp.array([[[2*(db*db+b*ddb),0.],[0.,0.]],
        [[2*(1-xi*xi)*(dd*dd+d*ddd),-4*xi*d*dd],[-4*xi*d*dd,-2*d*d]]])
    return CollisionJet(drift,noise,aj,ah,qj,qh,
        jnp.array([spatial[0],-2*xi*d*spatial[2]]),
        jnp.array([2*b*spatial[1],2*(1-xi*xi)*d*spatial[2]]))

def interval_from_budget(linear,quadratic,tolerance,cap=.0025):
    """Largest h satisfying linear*h+quadratic*h*h <= .8*tolerance."""
    target=.8*tolerance
    denominator=linear+jnp.sqrt(linear*linear+4*quadratic*target)
    h=2*target/jnp.maximum(denominator,1e-300)
    return jnp.minimum(cap,h)

def _gc_velocity_jet(field,point,vel,physical):
    """Exact velocity derivatives of the existing E=0 Cartesian GC RHS.

    At a fixed position the five phase velocities have the form N(r,xi)/D,
    where N is polynomial and D=d0+d1*r*xi. Quotient derivatives supply the
    Jacobian and diagonal Hessian used above. Public field quantities and
    the GC equations are unchanged; field/profile dependence remains in AD.
    """
    from essos.dynamics import _gc_quantities
    from essos.constants import PROTON_MASS as M,ELEMENTARY_CHARGE as Q
    M,Q=getattr(physical,"mass",M),getattr(physical,"charge",Q)
    Bc,B,Bmag,grad,curl,kappa,sqrtg=_gc_quantities(field,point)
    r,xi=vel;U=B/Bmag;A=grad/(2*Bmag);K=kappa-A
    d0=Bc@B;d1=physical.sigma*M/Q*(Bc@curl)
    c0=M*physical.sigma/(Q*sqrtg)*jnp.cross(Bc,A)
    c2=M*physical.sigma/(Q*sqrtg)*jnp.cross(Bc,K)+d1*U
    v0=B@A;v2=B@K
    v1=physical.sigma*M/Q*(curl@A);v3=physical.sigma*M/Q*(curl@K)
    factor=-Bmag*physical.sigma/physical.nu
    q0=v0+v2*xi**2;q1=v1*xi+v3*xi**3
    N=jnp.r_[d0*U*r*xi+r*r*(c0+c2*xi*xi),0.,factor*(r*q0+r*r*q1)]
    Nr=jnp.r_[d0*U*xi+2*r*(c0+c2*xi*xi),0.,factor*(q0+2*r*q1)]
    Nrr=jnp.r_[2*(c0+c2*xi*xi),0.,2*factor*q1]
    Nx=jnp.r_[d0*U*r+2*r*r*c2*xi,0.,factor*(2*r*v2*xi+r*r*(v1+3*v3*xi*xi))]
    Nxx=jnp.r_[2*r*r*c2,0.,factor*(2*r*v2+6*r*r*v3*xi)]
    D=d0+d1*r*xi;Dr=d1*xi;Dx=d1*r
    value=N/D
    first=jnp.stack(((Nr-value*Dr)/D,(Nx-value*Dx)/D),axis=1)
    second=jnp.stack(((Nrr-2*first[:,0]*Dr)/D,(Nxx-2*first[:,1]*Dx)/D),axis=1)
    return value,first,second

def expanded_moment_budget(velocity,jet,vel,tau):
    """Same weak-moment budget, expanding the two diagonal noise directions.

    This removes small batched matrix products. Radial and field derivatives
    remain unchanged; no term of the existing estimator is dropped.
    """
    f,F,H=velocity;a,g,aj,ah,qj,qh,xa,xq=jet;q=g*g
    multiply=lambda A,v:A[:,0]*v[0]+A[:,1]*v[1]
    contraction=lambda A:A[:,0,0]*q[0]+A[:,1,1]*q[1]
    mean=-multiply(F,a)-.5*multiply(H,q)
    mean=mean.at[-2:].add(multiply(aj,f[-2:])+xa-multiply(aj,a)-.5*contraction(ah))
    aq=aj*q[None,:]
    correction=jnp.diag(.25*jnp.diag(qj)**2-multiply(qj,a)-.5*contraction(qh))-aq-aq.T
    cov=jnp.zeros((5,5)).at[-2:,-2:].set(jnp.diag(multiply(qj,f[-2:])+xq)+correction)
    fq=F*q[None,:];cov=cov.at[:,-2:].add(-fq).at[-2:,:].add(-fq.T)
    B=F*g[None,:];noise=B.at[-2:,:].set(0.)
    mean=mean+.5*tau*multiply(F,mean[-2:])
    def transport(A):
        left=F[:,0,None]*A[None,-2,:]+F[:,1,None]*A[None,-1,:]
        right=A[:,-2,None]*F[None,:,0]+A[:,-1,None]*F[None,:,1]
        inner=(F[:,0,None]*(A[-2,-2]*F[None,:,0]+A[-2,-1]*F[None,:,1])+
            F[:,1,None]*(A[-1,-2]*F[None,:,0]+A[-1,-1]*F[None,:,1]))
        return A+.5*tau*(left+right)+tau*tau/3*inner
    cov=transport(cov)
    xi,r=vel[::-1]
    def vector(A):return jnp.concatenate((A,(xi*A[-2]+r*A[-1])[None]),axis=0)
    def matrix(A):
        right=xi*A[:,-2]+r*A[:,-1];bottom=xi*A[-2,:]+r*A[-1,:]
        return jnp.concatenate((jnp.concatenate((A,right[:,None]),axis=1),jnp.r_[bottom,xi*right[-2]+r*right[-1]][None,:]),axis=0)
    bp=vector(mean).at[-1].add(cov[-2,-1]);Cp=matrix(cov)
    Gp=vector(jnp.zeros((5,2)).at[-2:,:].set(jnp.diag(g)));Bp=vector(B)
    spread=tau*jnp.sum((Gp+.5*tau*Bp)**2,axis=1)+tau**3/12*jnp.sum(Bp*Bp,axis=1)
    inverse=1/jnp.sqrt(jnp.maximum(spread,1e-28))
    # Covariance symmetry lets scalar lowering omit duplicated lower entries.
    pairs=jnp.stack([.5*tau*jnp.abs(Cp[i,j])*(inverse[i]*inverse[j])
        for i in range(6) for j in range(i,6)])
    linear=jnp.maximum(jnp.max(.5*tau*jnp.abs(bp)*inverse),jnp.max(pairs))
    # J*noise=0: cubic transport/projection leaves only its spatial block.
    # The normalized absolute maximum of its noise Gram block is diagonal.
    diagonal=(noise[:3,0]**2+noise[:3,1]**2)/3
    quadratic=jnp.max(tau*diagonal*(inverse[:3]*inverse[:3]))
    return linear,quadratic

def make_step_limit(field,bath,physical,tolerance=.005,*,reference_time=.005,cap=.0025,projection=None):
    """Predict intervals from a fractional moment-defect budget.

    reference_time is in t*nu0 units, NOT a timestep cap. It sets the earliest
    observation timescale; elapsed time subsequently grows the spread proxy.
    Independent physical/paired tests, rather than this proxy, determine achieved error.
    """
    to_phase,_,_,radial=_cartesian_model(field,bath,physical)
    def limit(t,y,args):
        u=to_phase(y);vel=u[-2:];point=y[:3]
        velocity=_gc_velocity_jet(field,point,vel,physical)
        jet=collision_jet(radial,vel[0],vel[1],point,velocity[0][:3]*physical.sigma/physical.nu)
        if projection is not None:
            jacobian=jax.jacfwd(projection)(point)
            velocity=tuple(jnp.concatenate((jacobian@value[:3],value[3:]),axis=0) for value in velocity)
        tau=jnp.maximum(reference_time,t*physical.nu)
        linear,quadratic=expanded_moment_budget(velocity,jet,vel,tau)
        return interval_from_budget(linear,quadratic,tolerance,cap=cap)/physical.nu
    return limit

def _cartesian_model(field,bath,physical):
    """Native GC coefficients in normalized physical coordinates."""
    from .constants import PROTON_MASS as M,ELEMENTARY_CHARGE as Q
    from .dynamics import _guiding_center_velocity
    from .electric_field import Electric_field_zero
    M,Q=getattr(physical,"mass",M),getattr(physical,"charge",Q)
    sigma,nu=physical.sigma,physical.nu
    electric=Electric_field_zero()
    def phase(y):
        speed=jnp.sqrt(y[3]**2+2*y[4]*field.AbsB(y[:3])/M)
        return jnp.r_[y[:3]*nu/sigma,speed/sigma,y[3]/speed]
    def restore(u):
        point=u[:3]*sigma/nu
        return jnp.r_[point,sigma*u[3]*u[4],M*sigma**2*u[3]**2*(1-u[4]**2)/(2*field.AbsB(point))]
    def drift(u):
        y=restore(u)
        dx,dv=_guiding_center_velocity(field,electric,Q,M,y[:3],y[3],mu=y[4])
        return jnp.r_[dx/sigma,0.,dv/(nu*sigma*u[3])]
    def radial(r,point):
        c=collision_coefficients(bath,M,Q,sigma*r,point)
        return jnp.array([(c.friction+c.parallel_derivative+2*c.parallel/(sigma*r))/(sigma*nu),
            _diffusion_root(2*c.parallel/(sigma*sigma*nu)),
            _diffusion_root(2*c.perpendicular/(sigma*sigma*r*r*nu))])
    return phase,restore,drift,radial

def _particle_ito(velocity,unit_b,rates):
    K,P,T=rates
    r=jnp.linalg.norm(velocity);u=velocity/r;xi=unit_b@u
    rows=jnp.stack((u,(unit_b-xi*u)/r,unit_b));projected=rows@u
    cross=2*T*rows+2*(P-T)*projected[:,None]*u[None,:]
    low=2*T*(rows@rows.T)+2*(P-T)*projected[:,None]*projected[None,:]
    vv=2*T*jnp.eye(3)+2*(P-T)*u[:,None]*u[None,:]
    covariance=jnp.concatenate((jnp.concatenate((vv,cross.T),axis=1),jnp.concatenate((cross,low),axis=1)),axis=0)
    return jnp.r_[K*u,K+2*T/r,-2*T*xi/r**2,K*xi],covariance

def _particle_local_moments(velocity,unit_b,coefficients):
    """Exact weak-two moments of the existing frozen Lorentz collision map.

    Coordinates are (v_x,v_y,v_z,speed,pitch,v_parallel), normalized before
    this call. Return drift, diffusion covariance, h^2 mean and h^2 covariance.
    P/T are parallel/perpendicular diffusion; K is the Cartesian Ito drift
    magnitude and Pprime is its parallel-diffusion speed derivative. No
    collision update, derivative approximation or sampled calibration occurs.
    """
    K,P,T,Pprime=coefficients
    r=jnp.linalg.norm(velocity);u=velocity/r;xi=unit_b@u
    tangent=unit_b-xi*u
    drift,diffusion=_particle_ito(velocity,unit_b,jnp.array([K,P,T]))
    second_mean=jnp.zeros(6).at[3].set(-2*K*T/r**2+4*T*(P-T)/r**3)
    second_mean=second_mean.at[4].set(xi*(4*K*T/r**3+12*T*(T-P)/r**4))
    parallel=.5*Pprime**2
    vr=u*(parallel-4*P*T/r**2)
    vx=u*xi*8*P*T/r**3+tangent*(-2*K*T/r**2+(4*P*T-8*T*T)/r**3)
    second=jnp.zeros((5,5)).at[:3,:3].set(parallel*u[:,None]*u[None,:])
    second=second.at[:3,3].set(vr).at[3,:3].set(vr).at[:3,4].set(vx).at[4,:3].set(vx)
    second=second.at[3,3].set(parallel+(4*T*T-8*P*T)/r**2)
    cross=4*xi*T*(2*P-T)/r**3
    second=second.at[3,4].set(cross).at[4,3].set(cross)
    second=second.at[4,4].set(-4*K*T*(1-xi*xi)/r**3+
        ((20*xi*xi-16)*T*T+12*P*T*(1-xi*xi))/r**4)
    parallel_projection=jnp.concatenate((jnp.eye(5),jnp.r_[unit_b,0.,0.][None,:]))
    return drift,diffusion,second_mean,parallel_projection@second@parallel_projection.T

def _particle_collision_generator(velocity,unit_b,rates,first,second):
    """Exact Coulomb generator on the six drift/covariance observables.

    In polar velocity coordinates L=A*d_r+P*d_rr+(T/r^2)*Delta_sphere.
    The observables contain spherical harmonics of degree at most two, so
    these contractions need no materialized third/fourth-order tensors.
    """
    r=jnp.linalg.norm(velocity);u=velocity/r;xi=unit_b@u
    K,P,T=rates;A=K+2*T/r
    RK,RP,RT=A*first+P*second
    def radial_power(n):
        return 2/r**n*(A*(first[2]-n*T/r)+P*(second[2]-2*n*first[2]/r+n*(n+1)*T/r**2))
    h1,h2=2*T/r,2*T/r**2;R1,R2=radial_power(1),radial_power(2)
    vector=(RK-2*T*K/r**2)*u
    mean=jnp.r_[vector,RK+R1,(-R2+2*T*h2/r**2)*xi,unit_b@vector]
    outer=u[:,None]*u[None,:]
    vv=(2*RT+4*T*(P-T)/r**2)*jnp.eye(3)+(2*(RP-RT)-12*T*(P-T)/r**2)*outer
    vr=(2*RP-4*T*P/r**2)*u
    vx=(R1-2*T*h1/r**2)*unit_b+(-R1+6*T*h1/r**2)*xi*u
    covariance=jnp.zeros((6,6)).at[:3,:3].set(vv)
    covariance=covariance.at[:3,3].set(vr).at[3,:3].set(vr).at[:3,4].set(vx).at[4,:3].set(vx)
    covariance=covariance.at[3,3].set(2*RP).at[4,4].set(R2*(1-xi*xi)+T*h2/r**2*(-2+6*xi*xi))
    projected=covariance[:5,:3]@unit_b
    covariance=covariance.at[:5,5].set(projected).at[5,:5].set(projected).at[5,5].set(unit_b@vv@unit_b)
    return mean,covariance

def _particle_flow_generator(v,b,gradient,rotation,electric,rates,orbit):
    """Same exact generator on position, Lorentz force, speed and pitch flow."""
    r=jnp.linalg.norm(v);u=v/r;xi=b@u;K,P,T=rates;A=K+2*T/r
    quad=u@gradient@u;trace=jnp.trace(gradient);e=u@electric;parallel=b@electric
    speed=-2*T/r**2*e
    pitch=(A-6*T/r)*quad+2*T/r*trace+(-A/r**2+2*P/r**3)*(parallel-xi*e)+T/r**3*(-2*parallel+6*xi*e)
    par=(2*r*A+2*P-6*T)*quad+2*T*trace
    return jnp.r_[K*u if orbit else jnp.zeros(3),rotation@(K*u),speed,pitch,par]

def _particle_velocity_jacobians(v,b,gradient,rotation,electric,rates,first,*,orbit=True):
    """Exact velocity Jacobians of the existing collision drift and orbit flow."""
    r=jnp.linalg.norm(v);u=v/r;xi=b@u;K,_,T=rates;dK,_,dT=first
    pitch_gradient=(b-xi*u)/r
    drift=K/r*jnp.eye(3)+(dK-K/r)*u[:,None]*u[None,:]
    speed=(dK+2*dT/r-2*T/r**2)*u
    pitch=-2*dT*xi/r**2*u-2*T/r**2*pitch_gradient+4*T*xi/r**3*u
    parallel=dK*xi*u+K*pitch_gradient
    A1=jnp.concatenate((drift,speed[None,:],pitch[None,:],parallel[None,:]))
    e=u@electric;de=(electric-e*u)/r
    p=(gradient@v)@v+b@electric;dp=(gradient+gradient.T)@v
    pitch_rate=(p-xi*e)/r
    df=(dp-e*pitch_gradient-xi*de-pitch_rate*u)/r
    F=jnp.concatenate((jnp.eye(3) if orbit else jnp.zeros((3,3)),rotation,de[None,:],df[None,:],dp[None,:]))
    return A1,F

def _particle_budget(velocity,unit_b,unit_gradient,rotation,electric,rates,
                     rate_first,rate_second,profile_rate,tau,*,orbit=True):
    """Same weak-moment budget for Lorentz motion, including speed and pitch.

    Field/profile derivatives are current exact derivatives. The local radial
    polynomial supplies their exact value/first/second velocity jets at this
    state; it is not a cached coefficient model or a trajectory approximation.
    Accepted tracing still calls the original orbit and collision maps.
    """
    r=jnp.linalg.norm(velocity);u=velocity/r;xi=unit_b@u
    K,P,T=rates
    diffusion=2*T*jnp.eye(3)+2*(P-T)*u[:,None]*u[None,:]
    A1,F=_particle_velocity_jacobians(velocity,unit_b,unit_gradient,rotation,electric,rates,rate_first,orbit=orbit)
    la,lq=_particle_collision_generator(velocity,unit_b,rates,rate_first,rate_second)
    force=rotation@velocity+electric
    values,convected=jax.jvp(_particle_ito,(velocity,unit_b,rates),
        (force,unit_gradient@velocity,profile_rate+rate_first*(u@electric)))
    a,q=values;oa,oq=convected
    _,_,mean2,cov2=_particle_local_moments(velocity,unit_b,jnp.r_[rates,rate_first[1]])
    extend=lambda x:jnp.concatenate((jnp.zeros((3,)+x.shape[1:]),x),axis=0)
    G=jnp.concatenate((jnp.zeros((3,3)),jnp.eye(3),u[None,:],
        (unit_b-xi*u)[None,:]/r,unit_b[None,:]))
    mean=extend(oa-la+2*mean2)-_particle_flow_generator(velocity,unit_b,unit_gradient,rotation,electric,rates,orbit)
    covariance=jnp.pad(oq-lq+2*cov2,((3,0),(3,0)))
    cross=(F+extend(A1))@diffusion@G.T
    covariance=covariance-cross-cross.T
    mean=mean+.5*tau*(F@mean[3:6])
    covariance=(covariance+.5*tau*(F@covariance[3:6,:]+covariance[:,3:6]@F.T)
        +tau*tau/3*(F@covariance[3:6,3:6]@F.T))
    noise=velocity_basis(velocity)*jnp.array([_diffusion_root(2*T),_diffusion_root(2*T),_diffusion_root(2*P)])[None,:]
    gp,bp=G@noise,F@noise
    spread=tau*jnp.sum((gp+.5*tau*bp)**2,axis=1)+tau**3/12*jnp.sum(bp*bp,axis=1)
    scale=jnp.sqrt(jnp.maximum(spread,1e-28))
    pairs=jnp.stack([.5*tau*jnp.abs(covariance[i,j])/(scale[i]*scale[j])
        for i in range(9) for j in range(i,9)])
    linear=jnp.maximum(jnp.max(.5*tau*jnp.abs(mean)/scale),jnp.max(pairs))
    spatial_response=F[:3]@noise
    quadratic=jnp.max(tau*jnp.sum(spatial_response*spatial_response,axis=1)/(3*scale[:3]*scale[:3]))
    return linear,quadratic

def native_step_limit(field,bath,mass,charge,initial,rhs,public,position,magnitude,
                      *,gc,coordinate=False,boozer=False,
                      electric_field,tmax,dtmax,tolerance=.005,reference_time=.005,orbit=True):
    """Construct the default physical moment controller without benchmark data.

    Reference units use birth density-weighted temperature and the friction
    frequency at sqrt(T/m). They only normalize the estimator. VMEC uses its
    Cartesian position map; Boozer uses its regular flux/angle chart because
    magnetic Boozer tables do not specify a Cartesian surface embedding.

    The selected GC predictor is retained when the spatial collision update
    is enabled. That update does not select a different error estimator.
    """
    from types import SimpleNamespace
    from .constants import ELEMENTARY_CHARGE
    from .electric_field import Electric_field_zero
    birth=public(initial);point=birth[:3]
    density=jnp.stack([bath.get_density(i,point) for i in range(bath.number_species)])
    temperatures=jnp.stack([bath.get_temperature(i,point) for i in range(bath.number_species)])
    initial_magnitude=magnitude
    speed=jnp.sqrt(birth[3]**2+2*birth[4]*initial_magnitude(point)/mass) if gc else jnp.linalg.norm(birth[3:])
    temperature=jnp.sum(density*temperatures)/jnp.where(jnp.sum(density)>0,jnp.sum(density),1.)
    sigma=jnp.where(jnp.sum(density)>0,jnp.sqrt(ELEMENTARY_CHARGE*temperature/mass),speed/jnp.sqrt(10.))
    friction=collision_coefficients(bath,mass,charge,sigma,point).friction
    nu=jnp.where(friction==0,1/jnp.maximum(tmax,1e-30),-friction/sigma)
    physical=SimpleNamespace(sigma=sigma,nu=nu,mass=mass,charge=charge)
    cap=dtmax*nu;zero_electric=isinstance(electric_field,Electric_field_zero)
    if gc and not boozer and zero_electric and orbit:
        limit=make_step_limit(field,bath,physical,tolerance,reference_time=reference_time,cap=cap,
            projection=field.to_xyz if coordinate else None)
        return (lambda t,y,args:limit(t,public(y),args)) if coordinate else limit
    if gc:
        def phase(y):
            state=public(y);v=jnp.sqrt(state[3]**2+2*state[4]*magnitude(state[:3])/mass)
            return jnp.r_[position(y)*(1. if boozer else nu/sigma),v/sigma,state[3]/v]
        def radial(r,y):
            point=public(y)[:3]
            c=collision_coefficients(bath,mass,charge,sigma*r,point)
            return jnp.array([(c.friction+c.parallel_derivative+2*c.parallel/(sigma*r))/(sigma*nu),
                _diffusion_root(2*c.parallel/(sigma*sigma*nu)),_diffusion_root(2*c.perpendicular/(sigma*sigma*r*r*nu))])
        def velocity(v,template):
            B=magnitude(public(template)[:3]);r,xi=v
            y=template.at[3].set(sigma*r*xi).at[4].set(mass*sigma*sigma*r*r*(1-xi*xi)/(2*B))
            dy=rhs(0.,y,None)
            dx=jax.jvp(position,(y,),(dy,))[1]/(nu if boozer else sigma)
            dr=0. if zero_electric else jax.jvp(lambda y:phase(y)[3],(y,),(dy,))[1]/nu
            return jnp.r_[dx,dr,dy[3]/(sigma*nu*r)-xi*dr/r]
        def limit(t,y,args):
            vel=phase(y)[-2:]
            f=lambda v:velocity(v,y)
            f0=f(vel);F=jax.jacfwd(f)(vel);H=jax.jacfwd(jax.jacfwd(f))(vel)
            jet=collision_jet(radial,*vel,y,rhs(t,y,args)/nu)
            linear,quadratic=expanded_moment_budget((f0,F,jnp.stack((H[:,0,0],H[:,1,1]),axis=1)),jet,vel,jnp.maximum(reference_time,t*nu))
            return interval_from_budget(linear,quadratic,tolerance,cap=cap)/nu
        return limit
    def radial(r,point):
        c=collision_coefficients(bath,mass,charge,sigma*r,point,full_orbit=True)
        return jnp.array([(c.friction+c.parallel_derivative+2*(c.parallel-c.perpendicular)/(sigma*r))/(sigma*nu),
            c.parallel/(sigma*sigma*nu),c.perpendicular/(sigma*sigma*nu)])
    def unit_b(point):
        B=field.B_contravariant(point)
        return B/jnp.linalg.norm(B)
    def limit(t,y,args):
        point=y[:3];v=y[3:]/sigma;r=jnp.linalg.norm(v);B=field.B_contravariant(point);b=unit_b(point)
        _,profile=jax.jvp(lambda x:radial(r,x),(point,),(sigma*v/nu if orbit else jnp.zeros(3),))
        values=radial(r,point);first=jax.jacfwd(radial)(r,point);second=jax.jacfwd(jax.jacfwd(radial))(r,point)
        bx,by,bz=B
        rotation=charge/(mass*nu)*jnp.array([[0.,bz,-by],[-bz,0.,bx],[by,-bx,0.]])
        electric=charge/(mass*sigma*nu)*electric_field.E_covariant(point)
        gradient=jax.jacfwd(unit_b)(point)*sigma/nu if orbit else jnp.zeros((3,3))
        linear,quadratic=_particle_budget(v,b,gradient,rotation if orbit else jnp.zeros((3,3)),
            electric if orbit else jnp.zeros(3),values,first,second,profile,jnp.maximum(reference_time,t*nu),orbit=orbit)
        return interval_from_budget(linear,quadratic,tolerance,cap=cap)/nu
    return limit
