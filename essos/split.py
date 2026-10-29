"""Forward nonrelativistic orbit/collision splitting.

An independently controlled native RK orbit solve precedes each Milstein
collision map. Brownian rejection memory preserves every sampled interval.
Private GPU adapters leave public field classes unchanged. Optional accepted-map
recording connects to the checkpointed reverse rules in ``essos.native_ad``.
"""
from enum import IntEnum
from functools import lru_cache
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util
from .background_species import nu_D_ab, nu_par_ab, d_nu_par_ab, nu_s_ab
from .rk import RKStatus, solve_rk, _concrete_check, _prepare_rhs, OrbitErrorNorm



# Collision error


class CollisionErrorModel(NamedTuple):
    abs_drift: jax.Array
    parallel: jax.Array
    drift: jax.Array                 # .5 |Q Q'| (GC), or .5 |K K'| (particle)
    diffusion: jax.Array             # (D_parallel')**2
    pitch_drift: jax.Array           # .5 |xi| nu**2; zero for particle control
    pitch_diffusion: jax.Array       # .5 sqrt((1-xi**2) nu) nu; zero for particle
    tolerance: jax.Array
    speed_noise_index: jax.Array     # -2 for GC (including spatial noise), 2 for particle

def _ratio(numerator, denominator):
    safe = jnp.where(denominator > 0, denominator, 1.)
    return jnp.where(denominator > 0, numerator/safe,
                     jnp.where(numerator == 0, 0., jnp.inf))

def collision_error_estimates(model, h, dw):
    """Evaluate both normalized estimates without another coefficient/RHS call."""
    eps_v = model.tolerance*(model.abs_drift*h+jnp.sqrt(jnp.maximum(2*model.parallel*h, 0.)))
    drift = jnp.maximum(_ratio(model.drift*h*h, eps_v),
                        model.pitch_drift*h*h/model.tolerance)
    diffusion = jnp.maximum(
        _ratio(model.diffusion*jnp.abs(dw[model.speed_noise_index])**3,
               6*eps_v*jnp.sqrt(jnp.maximum(model.parallel, 0.))),
        model.pitch_diffusion*h*(jnp.abs(dw[-1])+jnp.sqrt(h/3))/model.tolerance)
    return drift, diffusion

def collision_error_passes(model, h, dw, *, target=1.):
    """Check the same four bounds without forming normalized error ratios.

    Coefficients and tolerance are finite/nonnegative, with positive tolerance,
    as supplied by the physical collision maps. Zero numerators and zero speed
    diffusion retain the estimator's 0/0 -> 0 convention. The actual trial still
    computes its normalized errors and independently checks finiteness.
    """
    tolerance = target*model.tolerance
    speed_scale = tolerance*(model.abs_drift*h+jnp.sqrt(jnp.maximum(2*model.parallel*h, 0.)))
    return ((model.drift*h*h <= speed_scale)
            & (model.diffusion*jnp.abs(dw[model.speed_noise_index])**3
               <= 6*speed_scale*jnp.sqrt(jnp.maximum(model.parallel, 0.)))
            & (model.pitch_drift*h*h <= tolerance)
            & (model.pitch_diffusion*h*(jnp.abs(dw[-1])+jnp.sqrt(h/3)) <= tolerance))


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
    error_model: CollisionErrorModel | None = None

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

def guiding_center_collision_step(speed, pitch, h, dw, c, *, tolerance=1e-3, speed_floor=0.,
                                  return_error_model=False):
    """Paper Eqs. (70)--(75), with speed [m/s] replacing normalized momentum.

    ``dw`` contains two independent Wiener increments of variance h, not unit
    normals. The diffusion coefficients are frozen at the initial speed for
    this local speed/pitch map, as in the paper. This is not a claim of a full
    multidimensional Milstein expansion for arbitrary noncommuting noise.
    """
    dp, dt, ddp, q, dq, _ = c
    nu = 2*dt/speed**2
    drift = q + ddp + 2*dp/speed
    v = speed + drift*h + _diffusion_root(2*dp)*dw[0] + .5*ddp*(dw[0]**2-h)
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
    model = (CollisionErrorModel(jnp.abs(drift), dp, .5*jnp.abs(q*dq), ddp**2,
        .5*jnp.abs(pitch)*nu**2, .5*angular_noise*nu, tolerance, jnp.array(-2))
        if return_error_model else None)
    return CollisionStep(jnp.array([v, reflect_pitch(xi)]), ed, es, model)

def velocity_basis(velocity):
    """Orthonormal columns (perp1, perp2, parallel), evaluated at this step."""
    norm = jnp.linalg.norm(velocity)
    direction = velocity/jnp.where(norm > 0, norm, 1.)
    direction = jnp.where(norm > 0, direction, jnp.array([0., 0., 1.]))
    reference = jnp.eye(3, dtype=velocity.dtype)[jnp.argmin(jnp.abs(direction))]
    first = jnp.cross(direction, reference)
    first = first/jnp.linalg.norm(first)
    return jnp.stack([first, jnp.cross(direction, first), direction], axis=1)

def particle_collision_step(velocity, h, dw, c, *, tolerance=1e-3, return_error_model=False):
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
    model = (CollisionErrorModel(jnp.abs(drift), dp, .5*jnp.abs(drift*dk), ddp**2,
        jnp.zeros_like(speed), jnp.zeros_like(speed), tolerance, jnp.array(2))
        if return_error_model else None)
    return CollisionStep(velocity + velocity_basis(velocity) @ delta,
        _ratio(.5*jnp.abs(drift*dk)*h*h, eps_v), _speed_diffusion_error(c, dw[2], eps_v), model)

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

def brownian_sample(memory, end, *, normal=None):
    """Sample W(end)-W(memory.t), retaining every sampled future point.

    Unlike ``brownian_propose``, this can span several stored intervals. New
    points inside a stored interval use the conditional law in Eqs. 83--84;
    points beyond the last sample extend it by an independent increment.
    The descending endpoint/interval representation is shared with the stack
    driver. Overflow leaves the memory and random key unchanged.
    An explicit standard ``normal`` can be supplied by a caller that owns the
    random-key advancement; otherwise a new key is split only for insertion.
    """
    if normal is not None:
        normal = jnp.asarray(normal, dtype=memory.increments.dtype)
        if normal.shape != memory.increments.shape[1:]:
            raise ValueError("normal must match the Wiener increment shape")
    m = memory
    indices = jnp.arange(len(m.ends))
    active = indices < m.size
    matches = active & (m.ends == end)
    known = jnp.any(matches) | (end == m.t)
    overflow = ~known & (m.size == len(m.ends))

    def insert(m):
        # k=-1 extends the furthest endpoint; otherwise split interval k.
        k = jnp.sum(active & (m.ends > end))-1
        extending = k < 0
        start = jnp.where(extending,
            jnp.where(m.size > 0, m.ends[0], m.t),
            jnp.where(k+1 < m.size, m.ends[jnp.minimum(k+1, len(m.ends)-1)], m.t))
        stop = m.ends[jnp.maximum(k, 0)]
        duration = jnp.where(extending, end-start, stop-start)
        fraction = jnp.where(extending, 1., (end-start)/jnp.where(duration > 0, duration, 1.))
        if normal is None:
            key, draw = jax.random.split(m.key)
            z = jax.random.normal(draw, m.increments.shape[1:], dtype=m.increments.dtype)
        else:
            key, z = m.key, normal
        parent = m.increments[jnp.maximum(k, 0)]
        left = jnp.where(extending, jnp.sqrt(jnp.maximum(end-start, 0.))*z,
            fraction*parent+jnp.sqrt(jnp.maximum((end-start)*(1-fraction), 0.))*z)
        at = k+1
        old_indices = jnp.clip(indices-(indices > at), 0, len(m.ends)-1)
        ends, increments = m.ends[old_indices], m.increments[old_indices]
        ends = ends.at[at].set(end)
        increments = increments.at[at].set(left)
        increments = lax.cond(extending, lambda a: a,
            lambda a: a.at[k].set(parent-left), increments)
        return BrownianMemory(ends, increments, m.size+1, key,
                              jnp.maximum(m.peak, m.size+1), m.t)

    m = lax.cond(~known & ~overflow, insert, lambda m: m, m)
    value = jnp.sum(jnp.where(((indices < m.size) & (m.ends <= end))[:, None], m.increments, 0.), axis=0)
    return m, value, overflow

def brownian_consume(memory, end):
    """Consume through an endpoint previously inserted by brownian_sample."""
    keep = (jnp.arange(len(memory.ends)) < memory.size) & (memory.ends > end)
    return memory._replace(size=jnp.sum(keep), t=end)

def paper_collision_step_size(memory, h, dw, drift_error, diffusion_error,
                              accepted, *, dtmin, dtmax, barrier, wiener_norm=None):
    """Paper section 5.3, Eqs. 78--82, with retained Wiener look-ahead.

    Probe cumulative Wiener values at successive thirds, stopping at the
    first threshold crossing. ``wiener_norm`` projects onto the noise components
    present in the error estimate; its default is the Euclidean vector norm.
    For the particle estimate (Eq. 62), only the parallel increment is relevant.
    All probes, including the first failing one, remain in Brownian memory.
    The first third is retried if no candidate passes; actual error estimates
    still decide acceptance. dtmax and the next output/final time cap probes.
    """
    drift_dominated = drift_error >= diffusion_error
    base = jnp.where(drift_dominated,
        jnp.minimum(1.5, .9*jnp.maximum(drift_error, 1e-30)**(-.5))*h, h)
    limit = jnp.minimum(dtmax, barrier-memory.t)
    unit = jnp.minimum(jnp.maximum(base/3, dtmin), limit)
    norm = jnp.linalg.norm if wiener_norm is None else wiener_norm
    amplitude = norm(dw)
    optimum = .9*jnp.maximum(diffusion_error, 1e-30)**(-1/3)*amplitude
    lmax = jnp.where(drift_dominated, 3,
        jnp.where(~accepted, 2, jnp.where(amplitude < 2*jnp.sqrt(h), 4, 6)))

    def probe(carry):
        m, n, chosen, running, overflow = carry
        length = jnp.minimum(n*unit, limit)
        end = jnp.minimum(m.t+length, barrier)
        m, candidate, full = brownian_sample(m, end)
        passes = norm(candidate) < optimum
        chosen = jnp.where((n == 1) | passes, end-m.t, chosen)
        return m, n+1, chosen, passes & ~full & (length < limit), overflow | full

    def stochastic(_):
        m, _, chosen, _, overflow = lax.while_loop(
            lambda c: (c[1] <= lmax) & c[3], probe,
            (memory, jnp.array(1), unit, jnp.array(True), jnp.array(False)))
        return m, chosen, overflow

    # A zero realized diffusion estimate leaves Eq. (80)'s threshold undefined
    # when the controlled increment is also zero. Use the drift proposal as a
    # fallback; zero realized error does not imply zero diffusion coefficients.
    # The next actual trial must still satisfy both error limits.
    return lax.cond(diffusion_error > 0, stochastic,
        lambda _: (memory, jnp.minimum(jnp.maximum(base, dtmin), limit), jnp.array(False)), None)


# Collision controller


def _fit_error_step(model, bound, minimum, normal_magnitude, target):
    """Largest bracketed step for frozen coefficients and fixed |dW|/sqrt(h).

    Eight bisections avoid assuming a common error power for speed and pitch.
    This is a cheap prediction: the next actual Brownian sample is checked
    separately. A sub-grid positive fallback still goes through real acceptance.
    """
    # At fixed |dW|/sqrt(h), cancel the common sqrt(h) in the speed
    # numerators/denominator and compare cross-products. These are exactly the
    # same four error inequalities, including 0/0 -> 0 in vacuum, without
    # repeated divisions, noise indexing, or coefficient square roots.
    parallel = jnp.sqrt(jnp.maximum(model.parallel, 0.))
    thermal_noise = jnp.sqrt(jnp.maximum(2*model.parallel, 0.))
    speed_noise = model.diffusion*jnp.abs(normal_magnitude[model.speed_noise_index])**3
    pitch_noise = model.pitch_diffusion*(jnp.abs(normal_magnitude[-1])+1/jnp.sqrt(3.))
    tolerance = target*model.tolerance

    def passes(h):
        root_h = jnp.sqrt(h)
        scale = tolerance*(model.abs_drift*root_h+thermal_noise)
        return ((model.drift*h*root_h <= scale)
                & (speed_noise*h <= 6*parallel*scale)
                & (model.pitch_drift*h*h <= tolerance)
                & (pitch_noise*h*root_h <= tolerance))

    def bisect(_):
        def one(_, bracket):
            lo, hi = bracket
            middle = (lo+hi)/2
            passed = passes(middle)
            return jnp.where(passed, middle, lo), jnp.where(passed, hi, middle)
        lo, hi = lax.fori_loop(0, 8, one, (minimum, bound))
        return jnp.where(lo > 0, lo, hi/2)
    return lax.cond(passes(bound), lambda _: bound, bisect, None)

def candidate_collision_step_size(memory, h, error_model, accepted, *,
                                  dtmin, dtmax, barrier, target=.9):
    """Predict, probe and backtrack using both frozen normalized error estimates.

    Accepted trials can grow by at most two; rejected trials shrink by at least
    20%, subject to dtmin and save/final boundaries. The initial predictor uses
    unit Gaussian magnitudes. A failed probe refits using its realized Gaussian
    magnitudes and a smaller upper bound. At most four Wiener points are probed.
    If the budget is exhausted, the last sampled step is returned for the real
    acceptance test; no inaccurate step is forced through.
    """
    limit = jnp.minimum(dtmax, barrier-memory.t)
    bound = jnp.minimum(jnp.maximum(jnp.where(accepted, 2., .8)*h, dtmin), limit)
    minimum = jnp.minimum(dtmin, bound)
    unit_noise = jnp.ones(memory.increments.shape[1:], dtype=memory.increments.dtype)
    proposal = _fit_error_step(error_model, bound, minimum, unit_noise, target)

    def stochastic(_):
        def probe(carry):
            m, wanted, last, count, running, overflow = carry
            end = jnp.minimum(m.t+wanted, barrier)
            length = end-m.t
            m, dw, full = brownian_sample(m, end)
            passed = collision_error_passes(error_model, length, dw, target=target)
            retry = ~passed & ~full & (length > minimum) & (count < 3)

            def refit(_):
                smaller = jnp.maximum(.8*length, minimum)
                normal = jnp.abs(dw)/jnp.sqrt(length)
                return _fit_error_step(error_model, smaller, minimum, normal, target)
            next_h = lax.cond(retry, refit, lambda _: length, None)
            return m, next_h, length, count+1, retry, overflow | full
        m, _, last, _, _, overflow = lax.while_loop(
            lambda c: c[4], probe,
            (memory, proposal, proposal, jnp.array(0), jnp.array(True), jnp.array(False)))
        return m, last, overflow

    has_noise_error = (error_model.diffusion > 0) | (error_model.pitch_diffusion > 0)
    return lax.cond(has_noise_error, stochastic,
        lambda _: (memory, proposal, jnp.array(False)), None)

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
    combined_rejected: jax.Array = 0
    max_accepted_combined_error: jax.Array = 0.
    collision_updates: jax.Array = 0
    accepted_orbit_steps: jax.Array | None = None
    max_orbit_steps_per_collision: jax.Array | None = None

class SplitSolution(NamedTuple):
    ts: jax.Array
    ys: jax.Array
    t: jax.Array
    y: jax.Array
    status: jax.Array
    stats: SplitStats
    step_ts: jax.Array

def solve_split(rhs, collision, y0, args=None, *, t0=0.0, t1, dt0, ts, key, noise_dim, adaptive=True, dtmin=0.0, dtmax=jnp.inf, method='dopri8', orbit_dt0=None, orbit_adaptive=True, orbit_stage_execution='loop', reuse_orbit_step=True, collision_controller='conservative', collision_wiener_norm=None, rtol=1e-09, atol=1e-09, error_norm=None, controller='i', pcoeff=0.4, icoeff=0.3, event=None, max_steps=100000, max_orbit_steps=10000, max_pending=64, record_steps=False, step_limit=None, align_orbit=False, event_dtmax=None, combined_error_norm=None, shared_clock=False, reuse_combined_orbit=True, combined_controller='rms', _orbit_rhs_prepared=False,
                adjoint='none', adjoint_checkpoint_steps=16, adjoint_orbit_checkpoint_steps=16,
                adjoint_max_accepted_steps=None, adjoint_max_orbit_steps=None,
                _record_ad=False, _collision_state_indices=None, _recorder=None):
    """Orbit advancement followed by an endpoint collision kick.

``collision(t, y, h, dW, args)`` returns ``CollisionStep`` with normalized
drift/diffusion estimates. Rejected trials repeat from the last accepted
state and retain all Brownian samples. The conservative controller uses
factors 0.9*min(E_drift**(-1/2), E_diff**(-2/3)), capped at 1.5.
``collision_controller="paper"`` selects section 5.3's Wiener look-ahead
(Eqs. 78--82), retaining every sampled candidate including failed probes.
``collision_controller="candidate"`` predicts both errors using frozen
local coefficients, then probes/backtracks on retained Wiener candidates.
It requires ``CollisionStep.error_model`` and still checks actual errors
for acceptance. The native collision kernels supply this metadata without
extra runtime coefficient evaluations.
``collision_wiener_norm`` selects the Wiener components that enter the
collision error estimate; by default it is the Euclidean norm of all noise.

``dt0``/``dtmax`` control collision/splitting intervals; ``orbit_dt0`` and
rtol/atol control the independent RK solve within each interval. Collision
estimates do not bound operator-splitting error: refine dtmax to check it.
Saves are exact split boundaries, so changing ts can change the trajectory.
No ODE dense interpolation is applied to a stochastic trajectory.
``reuse_orbit_step`` carries the next adaptive RK step-size guess across
kicks and retries. Every solve recomputes its initial derivative; no FSAL
derivative or PI error history is carried across a collision.
``orbit_stage_execution`` selects the RK loop or unrolled execution layout;
it does not change the orbit formula or the collision sampling.

``combined_error_norm(coarse, fine)`` optionally compares one full split step
with two Brownian-coupled half steps and accepts the fine result only when
this scalar estimate and all three collision estimates are <=1. Midpoints
remain in Brownian memory on rejection; growth cannot skip stored endpoints.
This experimental mode requires conservative adaptive collisions and no
events/step limits. Its H**(3/2) step-size predictor is a heuristic, not an
ensemble-error bound. ``accepted`` counts outer intervals; ``collision_updates``
counts the two accepted half kicks. Recorded step_ts includes both half ends.
Work counters include the full comparison trajectory and rejected trials.
``reuse_combined_orbit`` shares the initial half-orbit between the coarse and
fine paths. The coarse orbit then continues without the midpoint kick; the
fine path continues after it. The coarse collision still spans the full
interval. RK tolerances and retained Brownian samples are unchanged. False
retains the original independent full-orbit comparison for verification.
``combined_controller="rms"`` predicts the next interval from an exponential
average of squared defects divided by (h/dt0)**3, including rejected trials.
The averaging weight is 1/4 and the normalized RMS target is .55. Save times
and retained endpoints can truncate h without shrinking the intended proposal.
Every actual trial still requires the same realized defect and collision
estimates <=1; this predictor is not a certified conditional RMS error bound.
``combined_controller="i"`` retains the original single-defect I proposal.

``shared_clock=True`` is an experimental policy comparison: one accepted RK
step per collision update, with both error estimates constraining the common
interval. Rejected orbit trials retain their Wiener endpoints. It requires
adaptive conservative collisions/orbits, reused RK proposals and no events,
step limits or combined control. This uses native formulas, not ASCOT's solver.

Events are scalar booleans or positive-inside level sets (or their PyTrees).
Orbit crossings are root-found before a kick. A kick crossing stops at its
endpoint; Brownian crossings are not root-found by ODE interpolation.
Optional ``step_limit(t, y, args)`` supplies a positive interval ceiling
from the current accepted state. Optional ``event_dtmax`` retries a detected
kick crossing until its interval is this small, retaining Brownian samples.
The two controls support boundary-resolution diagnostics; they do not detect
every crossing followed by a return, or certify an exact first-passage time.
Neither is enabled by default. Step limits also subdivide fixed grids while
retaining their original grid anchors. They must not depend on fresh noise.
Stopped/failed outputs hold the last finite state; always inspect status.
Work and Brownian memory are bounded. No inaccurate step is forced through
dtmin or a memory limit."""
    if adjoint not in ('none','accepted'):
        raise ValueError("adjoint must be 'none' or 'accepted'")
    if adjoint == 'accepted' or _record_ad:
        options=locals().copy()
        from .native_ad import solve_accepted_split, _SplitRecorder
        for name in ('rhs','collision','y0','args','adjoint','_record_ad','_recorder'):
            options.pop(name)
        if adjoint == 'accepted':
            return solve_accepted_split(rhs,collision,y0,args,**options)
        _recorder=_SplitRecorder.from_options(jnp.asarray(y0),options)
    if _recorder is not None and combined_error_norm is not None:
        raise ValueError('accepted AD records the production moment controller; experimental step doubling is unsupported')
    combined = combined_error_norm is not None
    if combined_controller not in ('i','rms'):
        raise ValueError("combined_controller must be 'i' or 'rms'")
    rms = combined and combined_controller == 'rms'
    if align_orbit and (not adaptive or not orbit_adaptive or not reuse_orbit_step or shared_clock or combined):
        raise ValueError('orbit endpoint alignment requires separate adaptive clocks and orbit-step reuse')
    if shared_clock and (combined or not adaptive or not orbit_adaptive or not reuse_orbit_step or collision_controller != 'conservative' or tree_util.tree_leaves(event) or step_limit is not None or event_dtmax is not None):
        raise ValueError('shared_clock requires adaptive conservative collisions/orbits with reused steps, without combined control, events or step limits')
    if combined and (not adaptive or collision_controller != 'conservative' or tree_util.tree_leaves(event) or step_limit is not None or event_dtmax is not None):
        raise ValueError('combined control requires adaptive conservative collisions without events or step limits')
    if combined and not callable(combined_error_norm):
        raise ValueError('combined_error_norm must be callable')
    for name, value in (('max_steps', max_steps), ('max_orbit_steps', max_orbit_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if orbit_stage_execution not in ('loop', 'unrolled'):
        raise ValueError("orbit_stage_execution must be 'loop' or 'unrolled'")
    if collision_controller not in ('conservative', 'paper', 'candidate'):
        raise ValueError("collision_controller must be 'conservative', 'paper' or 'candidate'")
    candidate = adaptive and collision_controller == 'candidate'
    lookahead = adaptive and collision_controller in ('paper', 'candidate')
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
    empty_model = None
    if candidate:
        schema = jax.eval_shape(collision, t0, y0, dt0, jnp.zeros((noise_dim,), y0.dtype), args)
        if not isinstance(schema.error_model, CollisionErrorModel):
            raise ValueError('candidate controller requires CollisionStep.error_model')
        if any((x.shape != () for x in tree_util.tree_leaves(schema.error_model))):
            raise ValueError('collision error model coefficients must be scalars')
        empty_model = tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), schema.error_model)
        empty_model = empty_model._replace(tolerance=jnp.ones_like(empty_model.tolerance))
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
    s = dict(t=t0, y=y0, h=jnp.minimum(dt0, dtmax), orbit_h=orbit_dt0, status=initial_status, memory=memory, index=jnp.array(0), ys=jnp.broadcast_to(y0, (len(ts), y0.size)), steps=jnp.array(0), accepted=jnp.array(0), rejected=jnp.array(0), orbit_steps=jnp.array(0), orbit_rejected=jnp.array(0), max_error=jnp.zeros((), y0.dtype), history=jnp.full(((2 if combined else 1)*max_steps + 1 if record_steps else 0,), jnp.inf, dtype=y0.dtype))
    if combined:
        s.update(combined_rejected=jnp.array(0), combined_error=jnp.zeros((), y0.dtype),
            accepted_orbit_steps=jnp.array(0), max_orbit_steps_per_collision=jnp.array(0))
    else:
        s.update(accepted_orbit_steps=jnp.array(0), max_orbit_steps_per_collision=jnp.array(0))
    if rms:
        s.update(combined_rate=jnp.zeros((),y0.dtype),combined_rate_ready=jnp.array(False))
    if record_steps:
        s['history'] = s['history'].at[0].set(t0)
    if not adaptive:
        s['grid_index'] = jnp.array(0)
    if event_dtmax is not None:
        s['event_h'] = jnp.asarray(jnp.inf, dtype=y0.dtype)

    def save(s):
        if not len(ts):
            return s

        def one(s):
            return dict(s, ys=s['ys'].at[s['index']].set(s['y']), index=s['index'] + 1)
        return lax.while_loop(lambda s: (s['index'] < len(ts)) & (ts[jnp.minimum(s['index'], len(ts) - 1)] <= s['t']), one, s)
    s = save(s)
    initial_saves = s['index']
    if _recorder is not None:
        s = _recorder.initialize(s)

    def integrate(t, y, end, guess):
        if _recorder is not None:
            from .native_ad import _OrbitRecorder
        result = solve_rk(orbit_rhs, y, args, t0=t, t1=end, dt0=jnp.minimum(guess, end-t), ts=jnp.empty((0,), y0.dtype), method=method, adaptive=orbit_adaptive, rtol=rtol, atol=atol, stage_execution=orbit_stage_execution, error_norm=error_norm, controller=controller, pcoeff=pcoeff, icoeff=icoeff, event=event, root_find=True, max_steps=1 if shared_clock else max_orbit_steps,
            _recorder=None if _recorder is None else _OrbitRecorder(max_orbit_steps,adjoint_orbit_checkpoint_steps,schedule=True))
        return result

    def kick(t, y, h, dw):
        result = collision(t, y, h, dw, args)
        if result.y.shape != y0.shape:
            raise ValueError('collision must preserve the state shape')
        if jnp.ndim(result.drift_error) or jnp.ndim(result.diffusion_error):
            raise ValueError('collision errors must be scalars')
        return result if candidate else result._replace(error_model=None)

    def valid_trial(trial):
        errors=jnp.asarray([trial.drift_error,trial.diffusion_error])
        return jnp.isfinite(trial.y).all() & jnp.isfinite(errors).all() & jnp.all(errors>=0)

    def advance_with_limit(s, limit):
        physical_barrier = t1 if not len(ts) else jnp.where(s['index'] < len(ts), ts[jnp.minimum(s['index'], len(ts) - 1)], t1)
        barrier = physical_barrier
        if step_limit is not None:
            barrier = jnp.minimum(barrier, s['t'] + limit)
        if event_dtmax is not None:
            barrier = jnp.minimum(barrier, s['t'] + s['event_h'])
        if adaptive:
            proposal = jnp.minimum(s['h'], jnp.maximum(s['orbit_h'], dtmin)) if shared_clock else s['h']
            wanted_end = jnp.minimum(s['t'] + proposal, barrier)
        else:
            grid_end = t0 + (s['grid_index'] + 1) * jnp.minimum(dt0, dtmax)
            close = jnp.abs(grid_end - barrier) <= 8 * jnp.finfo(y0.dtype).eps * jnp.maximum(jnp.abs(grid_end), jnp.abs(barrier))
            grid_end = jnp.where(close, barrier, grid_end)
            wanted_end = jnp.minimum(grid_end, barrier)
        if align_orbit:
            length=wanted_end-s['t'];guess=s['orbit_h']
            near=(jnp.isfinite(guess)&(guess>0)&(wanted_end<physical_barrier)
                &(guess>=.9*length)&(guess<length*(1-1e-12)))
            wanted_end=jnp.where(near,s['t']+guess,wanted_end)
        if combined or shared_clock:
            # Do not create a roundoff-sized remainder before a known endpoint.
            # Snapping occurs before sampling; no Wiener point is discarded.
            rounding = 8*jnp.finfo(y0.dtype).eps*jnp.maximum(jnp.abs(barrier),jnp.abs(s['t']))
            wanted_end = jnp.where((wanted_end>s['t']) & (barrier-wanted_end<=rounding),barrier,wanted_end)
            parent = s['memory'].ends[jnp.maximum(s['memory'].size-1,0)]
            wanted_end = jnp.where((s['memory'].size>0)&(parent>=wanted_end)&(parent<=barrier)&(parent-wanted_end<=rounding),parent,wanted_end)
        if lookahead:
            m, dw, overflow = brownian_sample(s['memory'], wanted_end)
            end, h = (wanted_end, wanted_end - s['t'])
        else:
            m, h, dw, overflow = brownian_propose(s['memory'], wanted_end - s['t'], t_end=wanted_end)
            end = m.ends[m.size - 1]
        usable = ~overflow & (h > 0) & (end > s['t'])

        def attempt(_):
            guess = h if shared_clock else s['orbit_h'] if reuse_orbit_step and orbit_adaptive else orbit_dt0
            if combined and reuse_combined_orbit:
                half_end=s['t']+h/2
                shared_first=integrate(s['t'],s['y'],half_end,guess)
                tail=lax.cond(shared_first.status==RKStatus.SUCCESS,
                    lambda _:integrate(half_end,shared_first.y,end,shared_first.next_dt if reuse_orbit_step and orbit_adaptive else orbit_dt0),
                    lambda _:shared_first._replace(stats=type(shared_first.stats)(*(jnp.zeros_like(v) for v in shared_first.stats))),None)
                orbit=tail._replace(stats=type(tail.stats)(*(a+b for a,b in zip(shared_first.stats,tail.stats))))
            else:
                result = integrate(s['t'], s['y'], end, guess)
                orbit, schedule = result if _recorder is not None else (result, None)
            trial = lax.cond(orbit.status == RKStatus.SUCCESS, lambda _: kick(end, orbit.y, h, dw), lambda _: CollisionStep(orbit.y, jnp.zeros((), y0.dtype), jnp.zeros((), y0.dtype), empty_model), None)
            memory, combined_error = m, jnp.zeros((), y0.dtype)
            if combined:
                middle = s['t']+h/2
                splittable = (middle > s['t']) & (middle < end)
                memory, _, left, full = lax.cond(splittable,
                    lambda m: brownian_propose(m, middle-s['t'], t_end=middle),
                    lambda m: (m, jnp.zeros_like(h), jnp.zeros_like(dw), jnp.array(False)), m)
                def refine(_):
                    first = shared_first if reuse_combined_orbit else integrate(s['t'], s['y'], middle, guess)
                    a = kick(middle, first.y, middle-s['t'], left)
                    second = integrate(middle, a.y, end, first.next_dt if reuse_orbit_step and orbit_adaptive else orbit_dt0)
                    b = kick(end, second.y, end-middle, dw-left)
                    estimate = jnp.asarray(combined_error_norm(trial.y, b.y), dtype=y0.dtype)
                    if estimate.ndim != 0:
                        raise ValueError('combined error norm must return a scalar')
                    status = jnp.where(first.status != RKStatus.SUCCESS, first.status,
                        jnp.where(~jnp.isfinite(a.y).all(), RKStatus.NONFINITE, second.status))
                    stats = type(orbit.stats)(*(x+z if reuse_combined_orbit else x+y+z for x,y,z in zip(orbit.stats,first.stats,second.stats)))
                    refined = b._replace(y=jnp.where(valid_trial(a)&valid_trial(b),b.y,jnp.full_like(b.y,jnp.nan)),
                        drift_error=jnp.maximum(trial.drift_error,jnp.maximum(a.drift_error,b.drift_error)),
                        diffusion_error=jnp.maximum(trial.diffusion_error,jnp.maximum(a.diffusion_error,b.diffusion_error)))
                    return (second._replace(status=status,stats=stats),refined,estimate,
                        (first.stats.accepted+second.stats.accepted).astype(s['accepted'].dtype),
                        jnp.maximum(first.stats.accepted,second.stats.accepted).astype(s['accepted'].dtype))
                orbit,trial,combined_error,path_steps,path_max = lax.cond(splittable & ~full & (orbit.status == RKStatus.SUCCESS) & valid_trial(trial),
                    refine, lambda _: (orbit,trial,combined_error,jnp.zeros_like(s['accepted']),jnp.zeros_like(s['accepted'])), None)
            v = values(end, trial.y)
            finite = valid_trial(trial) & finite_values(v)
            finite &= jnp.isfinite(combined_error) & (combined_error >= 0)
            error = jnp.maximum(trial.drift_error, trial.diffusion_error)
            accepted = (orbit.status == RKStatus.SUCCESS) & finite & ((jnp.maximum(error,combined_error) <= 1) | (not adaptive))
            if combined:
                accepted &= splittable & ~full
            refine_event = jnp.array(False)
            if event_dtmax is not None:
                refine_event = accepted & stopped(v) & (h > event_dtmax * (1 + 8 * jnp.finfo(y0.dtype).eps))
                accepted &= ~refine_event
            orbit_retry = (orbit.status == RKStatus.MAX_STEPS) & (orbit.stats.accepted == 0) & (orbit.stats.rejected == 1) if shared_clock else jnp.array(False)
            orbit_status = jnp.where(orbit_retry, RKStatus.SUCCESS, orbit.status)
            rejected = (orbit_status == RKStatus.SUCCESS) & finite & ~accepted
            orbit_event = orbit.status == RKStatus.EVENT
            status = jnp.where(orbit_status != RKStatus.SUCCESS, orbit_status, jnp.where(~finite, SplitStatus.NONFINITE, jnp.where(accepted & stopped(v), SplitStatus.EVENT, jnp.where(rejected & (h <= dtmin), SplitStatus.MIN_STEP, SplitStatus.SUCCESS))))
            if combined:
                status = jnp.where(full, SplitStatus.BROWNIAN_CAPACITY, jnp.where(~splittable, SplitStatus.MIN_STEP, status))
            if rms:
                observe=(h>dt0*1e-6)&(combined_error>1e-8)&jnp.isfinite(combined_error)
                observation=combined_error**2/jnp.maximum((h/dt0)**3,1e-30)
                rate=jnp.where(observe,jnp.where(s['combined_rate_ready'],.75*s['combined_rate']+.25*observation,observation),s['combined_rate'])
                ready=s['combined_rate_ready']|observe
                proposal=dt0*(.55**2/jnp.maximum(rate,1e-30))**(1/3)
                collision=.9*h*jnp.minimum(jnp.maximum(trial.drift_error,1e-30)**(-.5),jnp.maximum(trial.diffusion_error,1e-30)**(-2/3))
                proposal=jnp.minimum(proposal,collision)
                ceiling=jnp.where(accepted & (h<s['h']),1.5*s['h'],jnp.where(accepted,1.5,.9)*h)
                next_h=jnp.clip(proposal,jnp.maximum(dtmin,.1*h),jnp.minimum(dtmax,ceiling))
            elif adaptive and (not lookahead):
                factor = 0.9 * jnp.minimum(jnp.maximum(trial.drift_error, 1e-30) ** (-0.5), jnp.maximum(trial.diffusion_error, 1e-30) ** (-2 / 3))
                if combined:
                    factor = jnp.minimum(factor, .9*jnp.maximum(combined_error,1e-30)**(-2/3))
                factor = jnp.clip(factor, 0.1, jnp.where(accepted, 1.5, 0.9))
                next_h = jnp.clip(h * factor, dtmin, dtmax)
            else:
                next_h = jnp.minimum(dt0, dtmax)
            if shared_clock:
                next_h = jnp.clip(jnp.minimum(next_h, orbit.next_dt), dtmin, dtmax)
            state = dict(s, t=jnp.where(orbit_event, orbit.t, jnp.where(accepted, end, s['t'])), y=jnp.where(orbit_event, orbit.y, jnp.where(accepted, trial.y, s['y'])), h=next_h, status=status, steps=s['steps'] + 1, accepted=s['accepted'] + accepted.astype(s['accepted'].dtype), rejected=s['rejected'] + rejected.astype(s['rejected'].dtype), orbit_steps=s['orbit_steps'] + orbit.stats.steps, orbit_rejected=s['orbit_rejected'] + orbit.stats.rejected, memory=lax.cond(accepted, (lambda m: brownian_consume(m, end)) if lookahead or combined else brownian_accept, lambda m: m, memory), max_error=jnp.maximum(s['max_error'], jnp.where(accepted, error, 0.0)))
            if rms:
                state.update(combined_rate=rate,combined_rate_ready=ready)
            if combined:
                state.update(combined_rejected=s['combined_rejected']+(rejected & (combined_error>1)).astype(s['rejected'].dtype),
                    combined_error=jnp.maximum(s['combined_error'],jnp.where(accepted,combined_error,0.)),
                    accepted_orbit_steps=s['accepted_orbit_steps']+jnp.where(accepted,path_steps,0),
                    max_orbit_steps_per_collision=jnp.maximum(s['max_orbit_steps_per_collision'],jnp.where(accepted,path_max,0)))
            else:
                path_steps = jnp.where(accepted,orbit.stats.accepted,0).astype(s['accepted'].dtype)
                state.update(accepted_orbit_steps=s['accepted_orbit_steps']+path_steps,
                    max_orbit_steps_per_collision=jnp.maximum(s['max_orbit_steps_per_collision'],path_steps))
            if event_dtmax is not None:
                state['event_h'] = jnp.where(refine_event, jnp.maximum(dtmin, 0.5 * h), jnp.inf)
                state['h'] = jnp.where(refine_event, state['event_h'], state['h'])
            if reuse_orbit_step and orbit_adaptive:
                state['orbit_h'] = orbit.next_dt
            if not adaptive:
                state['grid_index'] = s['grid_index'] + (accepted & (end == grid_end)).astype(s['grid_index'].dtype)
            if record_steps:
                if combined:
                    state['history'] = lax.cond(accepted,lambda a:a.at[2*state['accepted']-1].set(middle).at[2*state['accepted']].set(end),lambda a:a,state['history'])
                else:
                    state['history'] = lax.cond(accepted, lambda history: history.at[state['accepted']].set(end), lambda history: history, state['history'])
            state = save(state)
            if _recorder is not None:
                state = _recorder.record(s,state,accepted,orbit_event,orbit,schedule,dw)
            if lookahead:

                def choose(state):
                    barrier = t1 if not len(ts) else jnp.where(state['index'] < len(ts), ts[jnp.minimum(state['index'], len(ts) - 1)], t1)
                    if candidate:
                        memory, next_h, overflow = candidate_collision_step_size(state['memory'], h, trial.error_model, accepted, dtmin=dtmin, dtmax=dtmax, barrier=barrier)
                    else:
                        memory, next_h, overflow = paper_collision_step_size(state['memory'], h, dw, trial.drift_error, trial.diffusion_error, accepted, dtmin=dtmin, dtmax=dtmax, barrier=barrier, wiener_norm=collision_wiener_norm)
                    return dict(state, memory=memory, h=next_h, status=jnp.where(overflow, SplitStatus.BROWNIAN_CAPACITY, state['status']))
                state = lax.cond((state['status'] == SplitStatus.SUCCESS) & (state['t'] < t1) & ~refine_event, choose, lambda state: state, state)
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
    solution = SplitSolution(ts, ys, s['t'], s['y'], status, SplitStats(s['steps'], s['accepted'], s['rejected'], s['orbit_steps'], s['orbit_rejected'], s['memory'].peak, s['max_error'],s.get('combined_rejected',0),s.get('combined_error',0.),(2 if combined else 1)*s['accepted'],s.get('accepted_orbit_steps'),s.get('max_orbit_steps_per_collision')), s['history'])
    return solution if _recorder is None else _recorder.finish(s,solution,initial_saves)

def trace_split(field, particles, species, *, model='GuidingCenter', tmax, timestep, times=None, n_save=100, integrator='essos_dopri8', collision_tolerance=0.001, adaptive=True, collision_dtmax=None, orbit_dt0=None, orbit_adaptive=True, orbit_stage_execution=None, reuse_orbit_step=True, collision_controller='moments', moment_tolerance=.005, moment_reference_time=.005, rtol=1e-09, atol=1e-09, controller='i', pcoeff=0.4, icoeff=0.3, speed_floor=None, spatial_diffusion=None, spatial_geometry=None, orbit=True, electric_field=None, event=None, max_steps=100000, max_orbit_steps=10000, max_pending=64, record_steps=False, execution='shared', combined_tolerance=None, combined_length_scale=1., shared_clock=False, reuse_combined_orbit=True, combined_controller='rms', step_limit=None,
                adjoint='none', adjoint_checkpoint_steps=16, adjoint_orbit_checkpoint_steps=16,
                adjoint_max_accepted_steps=None, adjoint_max_orbit_steps=None, _record_ad=False, _replay_tapes=None, reverse_execution='auto', perturbation=None):
    """Batched split tracing with optional accepted-map reverse differentiation.

Returns a ``SplitSolution`` with shared ts and particle-leading arrays. GC states are
``(position[3], v_parallel, mu)`` with **physical** mu=m*v_perp^2/(2 B)
[J/T], unlike the legacy normalized collisional state. FullOrbit states
are ``(xyz, vx, vy, vz)``. VMEC/Boozer positions are (s, theta, zeta).
VMEC uses its regular axis chart; Boozer uses (sqrt(s)cos(theta),
sqrt(s)sin(theta)) internally. Both stop at the LCFS. Exterior continuation
is not part of this entry point. ``event`` receives public physical states.

The default includes the existing paper Eq. (69) position diffusion for GC.
FullOrbit retains its particle collision map. Explicit ``spatial_diffusion=False``
is retained for isolated operator diagnostics. VMEC uses its existing spatial
geometry; Boozer requires ``spatial_geometry=BoozerGeometry.from_booz_xform(b)``
or an equivalent public-flux-coordinate to Cartesian map. Spatial kicks are
inverted nonlinearly in the regular chart, preserving the Cartesian update.
``orbit=False`` runs collision-only statistics.
``speed_floor`` defaults to 5% of the smallest active background thermal
speed for the test mass. The GC reflection wall affects near-zero speeds;
refine/check that parameter when studying the thermal population.

Independent per-particle keys come from ``particles.random_keys``. The
field classes and existing Tracing/trace_boozer entry points are unchanged.

``adjoint="accepted"`` records accepted outer/inner schedules and retained
Wiener increments. Reverse differentiates all accepted orbit/collision maps,
including spatial kicks, while freezing controller/rejection/event decisions.
``adjoint_orbit_checkpoint_steps`` bounds the live replay segment; the two
``adjoint_max_*`` options bound tape storage and raise on reverse overflow.
Independent forward execution keeps the same production GPU kernel. With
``reverse_execution="auto"``, uniform alpha/direct-coil cases use persistent
GPU reverse; other supported fields use checkpointed JAX reverse on the
selected device. Explicit ``"independent"`` rejects unsupported reverse cases.

    ``execution="independent"`` uses one persistent GPU program per particle.
    It uses the same physical collision maps and moment predictor, including
    background profiles, Coulomb-log selection and Cartesian GC spatial kicks.
    It requires GPU float64, Threefry keys, zero electric field, Dopri8/I,
    matching orbit/collision adaptation, and no step recording. Unsupported
    GPU options raise; ``execution="shared"`` retains ordinary JAX execution.

    ``collision_controller="moments"`` is the adaptive default. It adds a
    coupled mean/covariance interval ceiling while retaining native collision
    checks at ``collision_tolerance`` and orbit checks at ``rtol/atol``.
    ``moment_tolerance`` is a local predictor budget, not a global error bound.

    ``combined_tolerance`` enables experimental full-step/two-half-step control
    on the shared Cartesian path. Position differences are scaled by
    ``combined_length_scale`` [m]; velocity differences by birth speed, with
    an additional pitch block for GC. This is a separate local budget from
    collision_tolerance and RK rtol/atol. None retains the original method.
    ``reuse_combined_orbit`` defaults to True; False retains independent
    coarse/fine orbit integrations for comparison. It affects combined mode only.
    ``combined_controller="rms"`` smooths squared defects for the next-step
    proposal while retaining the existing per-trial checks. "i" is the old policy.
    ``step_limit(t, y, args)`` optionally supplies a positive interval ceiling
    in seconds using the current Cartesian physical state. It must not draw
    fresh noise. This exposes solve_split's existing hook on the shared path;
    an explicit callback replaces the default moment predictor.
    ``shared_clock=True`` selects a native one-orbit-step/one-collision policy
    for comparison, also limited to the shared Cartesian path without events.
    ``perturbation`` adds prescribed alpha/Phi corrections to the existing GC
    RHS at every orbit stage time. Collisions use the current kinetic energy
    and physical mu. Independent wave forwards retain private field adapters;
    accepted wave reverse uses the checkpointed shared implementation. An
    inner wave cutoff is an EVENT, distinct from an LCFS loss by final flux.
    """
    from .boozer import BoozerField, guiding_center_rhs
    from .coils import Coils
    from .fields import BiotSavart, Vmec
    from .dynamics import _guiding_center_velocity, _axis_regular, _to_axis_regular, _from_axis_regular
    from .electric_field import Electric_field_zero
    from .constants import ELEMENTARY_CHARGE

    if adjoint not in ('none','accepted'):
        raise ValueError("adjoint must be 'none' or 'accepted'")
    if reverse_execution not in ('auto','shared','independent'):
        raise ValueError("reverse_execution must be 'auto', 'shared' or 'independent'")
    if (_record_ad and _replay_tapes is not None) or (adjoint!='none' and (_record_ad or _replay_tapes is not None)):
        raise ValueError('record/replay and accepted AD are separate operations')
    if adjoint=='accepted' and reverse_execution=='independent' and execution!='independent':
        raise ValueError('independent reverse requires independent forward execution')
    if adjoint=='accepted' or _record_ad:
        from .native_ad import _validate_ad_sizes
        _validate_ad_sizes(adjoint_checkpoint_steps=adjoint_checkpoint_steps,
            adjoint_orbit_checkpoint_steps=adjoint_orbit_checkpoint_steps,
            adjoint_max_accepted_steps=adjoint_max_accepted_steps,adjoint_max_orbit_steps=adjoint_max_orbit_steps)
    if integrator not in ('essos_rk45', 'essos_dopri8'):
        raise ValueError("split integrator must be 'essos_rk45' or 'essos_dopri8'")
    if model not in ('GuidingCenter', 'FullOrbit'):
        raise ValueError("split model must be 'GuidingCenter' or 'FullOrbit'")
    if combined_controller not in ('i','rms'):
        raise ValueError("combined_controller must be 'i' or 'rms'")
    moment_control=collision_controller=='moments' and adaptive and step_limit is None and combined_tolerance is None and not shared_clock
    if collision_controller=='moments':
        collision_controller='conservative'
    for name,value in (('moment_tolerance',moment_tolerance),('moment_reference_time',moment_reference_time)):
        _concrete_check(value,lambda x:np.ndim(x)==0 and np.isfinite(x) and x>0,name+' must be positive and finite')
    _concrete_check(collision_tolerance, lambda x: np.ndim(x) == 0 and np.isfinite(x) and (x > 0), 'collision_tolerance must be positive and finite')
    if species is None:
        raise ValueError('split collisions require BackgroundSpecies')
    if species.number_species < 1:
        raise ValueError('BackgroundSpecies must contain at least one species')
    field = BiotSavart(field) if isinstance(field, Coils) else field
    electric_field = Electric_field_zero() if electric_field is None else electric_field
    if execution not in ('shared', 'independent'):
        raise ValueError("split execution must be 'shared' or 'independent'")
    boozer, vmec = (isinstance(field, BoozerField), isinstance(field, Vmec))
    if perturbation is not None:
        from .perturbations import BoozerPerturbation
        from .gc_perturbations import GuidingCenterPerturbation
        expected = BoozerPerturbation if boozer else GuidingCenterPerturbation
        if (model != 'GuidingCenter' or not isinstance(perturbation, expected)
                or (boozer and isinstance(perturbation, GuidingCenterPerturbation))):
            raise ValueError('perturbations require GuidingCenter and potentials in the field coordinates')
    axis_cutoff = None if perturbation is None else perturbation.axis_stop_threshold()
    if axis_cutoff is not None and not (vmec or boozer):
        raise ValueError('an inner flux cutoff requires VMEC or Boozer coordinates')
    if step_limit is not None and (execution != 'shared' or boozer or vmec):
        raise ValueError('step_limit currently requires shared Cartesian tracing')
    if shared_clock and (execution != 'shared' or boozer or vmec):
        raise ValueError('shared_clock currently requires shared Cartesian tracing')
    if combined_tolerance is not None:
        if execution != 'shared' or boozer or vmec:
            raise ValueError('combined control currently requires shared Cartesian tracing')
        for name,value in (('combined_tolerance',combined_tolerance),('combined_length_scale',combined_length_scale)):
            _concrete_check(value, lambda x: np.ndim(x)==0 and np.isfinite(x) and x>0, name+' must be positive and finite')
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
    if execution == 'independent':
        _validate_independent(field, particles, species, model=model, integrator=integrator, adaptive=adaptive, orbit_adaptive=orbit_adaptive, controller=controller, collision_controller=collision_controller, electric_field=electric_field, spatial_diffusion=spatial_diffusion, orbit=orbit, record_steps=record_steps, reuse_orbit_step=reuse_orbit_step, event=event, times=times, tmax=tmax)
    field_view = _split_field_data(field) if execution == 'independent' and isinstance(field,(BiotSavart,Vmec,BoozerField)) else None
    moment_vmec_data=None
    if vmec and moment_control and execution=='shared':
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
        dx, dv = _guiding_center_velocity(source, electric_field, charge, mass, y[:3], y[3], mu=y[4],
                                          t=t, perturbation=perturbation)
        return jnp.concatenate([dx, jnp.array([dv, 0.0])])

    def rhs(t, y, args,source=field):
        if not orbit:
            return jnp.zeros_like(y)
        if boozer:
            return jnp.append(guiding_center_rhs(source, y[:4], y[4] / mass, mass, charge,
                                                t=t, perturbation=perturbation), 0.0)
        if vmec:
            return _axis_regular(lambda t,y,a:gc_drift(t,y,a,source))(t, y, args)
        if gc:
            return gc_drift(t, y, args,source)
        return jnp.concatenate([y[3:], charge / mass * (electric_field.E_covariant(y[:3]) + jnp.cross(y[3:], source.B_contravariant(y[:3])))])
    physical_rhs=rhs
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
    if execution == "shared" and y0.shape[0]:
        rhs = _prepare_rhs(rhs, jnp.zeros((), y0.dtype), y0[0], None)
    def one(initial, key, tape=None):
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
                step = guiding_center_collision_step(speed, state[3] / speed, h, dw[-2:], c, tolerance=collision_tolerance, speed_floor=floor, return_error_model=adaptive and collision_controller == 'candidate')
                v, pitch = step.y
                if spatial_diffusion:
                    displacement=guiding_center_spatial_increment(state[3]/speed,dw[:3],c,
                        spatial_direction(y,B,source),charge*B/mass)
                    if vmec or boozer:
                        y=move_in_cartesian(y,displacement,lambda q:spatial_position(q,source),
                            implicit=adjoint=='accepted' and source is field)
                        point=public(y)[:3]
                    else:
                        point=point+displacement;y=y.at[:3].set(point)
                    B = magnitude(point,source)
                result = y.at[3].set(v * pitch).at[4].set(mass * v * v * (1 - pitch * pitch) / (2 * B))
            else:
                step = particle_collision_step(state[3:], h, dw, c, tolerance=collision_tolerance, return_error_model=adaptive and collision_controller == 'candidate')
                result = y.at[3:].set(step.y)
            valid_c = jnp.isfinite(jnp.asarray(c)).all() & (c.parallel >= 0) & (c.perpendicular >= 0)
            result = jnp.where(jnp.isfinite(B) & (B > 0) & (speed > 0) & valid_c, result, jnp.full_like(result, jnp.nan))
            return step._replace(y=result)

        def position(y):
            point = public(y)[:3]
            return field.to_xyz(point) if vmec else point

        active_limit=step_limit
        if moment_control and execution=='shared':
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
                tolerance=moment_tolerance,reference_time=moment_reference_time,orbit=orbit,
                time_dependent=perturbation is not None)

        def collision_wiener_norm(dw):
            return jnp.linalg.norm(dw[-2:]) if gc else jnp.abs(dw[2])
        norm = OrbitErrorNorm(rtol=rtol, atol=atol, speed_scale=birth_speed, velocity_components=1 if gc else 3, position_map=position if vmec else None)
        combined_norm = None
        if combined_tolerance is not None:
            def combined_norm(coarse,fine):
                delta_x=jnp.linalg.norm(fine[:3]-coarse[:3])/combined_length_scale
                if gc:
                    def velocity(y):
                        speed=jnp.sqrt(y[3]**2+2*y[4]*magnitude(y[:3])/mass)
                        return speed,y[3]/speed
                    va,xa=velocity(coarse);vb,xb=velocity(fine)
                    delta_v=jnp.maximum(jnp.abs(va-vb)/birth_speed,jnp.abs(xa-xb))
                else:
                    delta_v=jnp.linalg.norm(fine[3:]-coarse[3:])/birth_speed
                value=jnp.maximum(delta_x,delta_v)/combined_tolerance
                return jnp.where((combined_tolerance>0)&(combined_length_scale>0)&jnp.isfinite(combined_tolerance)&jnp.isfinite(combined_length_scale),value,jnp.nan)
        if boozer:

            def norm(a, b, error):
                return jnp.maximum(jnp.linalg.norm(error[:2]), jnp.maximum(jnp.abs(error[2]), jnp.abs(error[3]) / birth_speed)) / (rtol + atol)
        events = []
        if vmec or boozer:
            events.append(lambda t, y, args: 1 - y[0] ** 2 - y[1] ** 2)
        if axis_cutoff is not None:
            events.append(lambda t, y, args: jnp.where(axis_cutoff > 0,
                jnp.maximum(y[0]**2+y[1]**2-axis_cutoff, 0.), 1.))
        if event is not None:
            events.extend([f if not (vmec or boozer) else
                lambda t, y, args, f=f: f(t, public(y), args) for f in tree_util.tree_leaves(event)])
        if _replay_tapes is not None:
            from .native_ad import replay_accepted_split
            saved,final=replay_accepted_split(rhs,kick,initial,None,tape,times,
                method='dopri5' if integrator=='essos_rk45' else 'dopri8',stage_execution=orbit_stage_execution)
            return jax.vmap(public)(saved),public(final)
        solve = solve_split
        if execution == 'independent':
            from functools import partial
            def physics(source):
                drift=lambda t,y,a:physical_rhs(t,y,a,source)
                collision=lambda t,y,h,dw,a:kick(t,y,h,dw,a,source)
                xyz=(lambda y:source.to_xyz(public(y)[:3])) if vmec else (lambda y:y[:3])
                measure=(lambda a,b,e:jnp.maximum(jnp.linalg.norm(e[:2]),jnp.maximum(jnp.abs(e[2]),jnp.abs(e[3])/birth_speed))/(rtol+atol)) if boozer else OrbitErrorNorm(rtol,atol,birth_speed,velocity_components=1 if gc else 3,position_map=xyz if vmec else None)
                limit=native_step_limit(source,species,mass,charge,initial,drift,public,xyz,lambda p:magnitude(p,source),
                    gc=gc,coordinate=vmec or boozer,boozer=boozer,
                    electric_field=electric_field,tmax=tmax,dtmax=timestep if collision_dtmax is None else collision_dtmax,
                    tolerance=moment_tolerance,reference_time=moment_reference_time,orbit=orbit,reference_magnitude=magnitude,
                    time_dependent=perturbation is not None) if moment_control else lambda t,y,a:jnp.asarray(jnp.inf)
                level=lambda t,y,a:jnp.min(jnp.asarray([f(t,y,a) for f in events])) if events else jnp.asarray(1.,y.dtype)
                return drift,collision,measure,level,limit
            solve = partial(_solve_independent, field=field, species=species, gc=gc, birth_speed=birth_speed, speed_floor=floor, collision_tolerance=collision_tolerance, field_view=field_view,physics=physics,mass=mass,charge=charge,spatial_diffusion=spatial_diffusion,spatial_geometry=spatial_geometry,_reverse_execution=reverse_execution,time_dependent=perturbation is not None)
        solution = solve(rhs, kick, initial, t1=tmax, dt0=timestep, ts=times, key=key, noise_dim=(5 if spatial_diffusion else 2) if gc else 3, dtmax=timestep if collision_dtmax is None else collision_dtmax, adaptive=adaptive, orbit_dt0=orbit_dt0, orbit_adaptive=orbit_adaptive, orbit_stage_execution=orbit_stage_execution, reuse_orbit_step=reuse_orbit_step, collision_controller=collision_controller, collision_wiener_norm=collision_wiener_norm, method='dopri5' if integrator == 'essos_rk45' else 'dopri8', rtol=rtol, atol=atol, controller=controller, pcoeff=pcoeff, icoeff=icoeff, error_norm=norm, combined_error_norm=combined_norm, shared_clock=shared_clock, reuse_combined_orbit=reuse_combined_orbit, combined_controller=combined_controller, step_limit=active_limit, align_orbit=moment_control and orbit_adaptive and reuse_orbit_step, event=tuple(events), max_steps=max_steps, max_orbit_steps=max_orbit_steps, max_pending=max_pending, record_steps=record_steps, _orbit_rhs_prepared=execution == "shared",
            adjoint=adjoint,adjoint_checkpoint_steps=adjoint_checkpoint_steps,
            adjoint_orbit_checkpoint_steps=adjoint_orbit_checkpoint_steps,
            adjoint_max_accepted_steps=adjoint_max_accepted_steps,adjoint_max_orbit_steps=adjoint_max_orbit_steps,
            _record_ad=_record_ad,_collision_state_indices=tuple(range(3,initial.size)) if not gc else None)
        if _record_ad:
            solution,tape=solution
            return solution._replace(ys=jax.vmap(public)(solution.ys),y=public(solution.y)),tape
        return solution._replace(ys=jax.vmap(public)(solution.ys), y=public(solution.y))
    if speed_floor is not None:
        _concrete_check(speed_floor, lambda x: np.ndim(x) == 0 and np.isfinite(x) and (x > 0), 'speed_floor must be positive and finite')
    if _replay_tapes is not None:
        return jax.jit(jax.vmap(one))(y0,particles.random_keys,_replay_tapes)
    if _record_ad:
        solution,tapes=jax.jit(jax.vmap(one))(y0,particles.random_keys)
        return solution._replace(ts=times),tapes
    return jax.jit(jax.vmap(one))(y0, particles.random_keys)._replace(ts=times)



# Independent GPU execution
from jax.extend.core import Literal
from .rk import (_select_one, _kernel_is_finite, _evaluate_inline, _bind_primitive,
    _A8, _C8, _ERROR8, _DENSE8, _pack, _cross, _dot, _coil_geometry,
    _coil_field_values, _coil_rhs_values, _prepare_vmec_data, _vmec_chart,
    _vmec_series, _vmec_rhs_values, _vmec_xyz_values, _padded_spline, _KernelField)
from .constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q
from jax._src import random as jax_random
threefry = getattr(jax_random, "threefry2x32", None) or jax_random.prng

def scalar(value, i):
    return _select_one(value, i, 0).reshape(())

pack = _pack

@lru_cache(maxsize=1)
def _kernel_erf():
    from jax.extend.core import Primitive
    from jax._src.pallas.triton import lowering
    from jax._src.interpreters import mlir
    primitive = Primitive("essos_split_erf")
    primitive.def_abstract_eval(lambda value: value)
    primitive.def_impl(lax.erf)
    mlir.register_lowering(primitive, mlir.lower_fun(lax.erf, multiple_results=False))
    lowering.register_lowering(primitive)(lowering._make_dispatch_table(
        "erf", cuda=[lowering._Extern([jnp.float64], "__nv_erf", jnp.float64),
                    lowering._Extern([jnp.float32], "__nv_erff", jnp.float32)]))
    return primitive


def _split_select(value, index, axis):
    if value.dtype == jnp.bool_:
        shape = [1] * value.ndim
        shape[axis] = value.shape[axis]
        return jnp.max((value & (jnp.arange(value.shape[axis]).reshape(shape) == index)).astype(jnp.int32), axis=axis, keepdims=True) > 0
    return _select_one(value, index, axis)

def _flatten(function, *examples):
    from jax._src.core import DropVar
    from jax._src.interpreters import partial_eval
    graph = jax.make_jaxpr(function)(*examples)
    graph = jax.make_jaxpr(lambda *x: _evaluate_inline(graph.jaxpr, graph.consts, *x))(*examples)
    readonly=all(type(effect).__name__=='ReadEffect' for effect in graph.jaxpr.effects)
    aliases,expressions,kept={},{},[]
    def freeze(value):
        if isinstance(value,dict):return tuple((k,freeze(v)) for k,v in sorted(value.items()))
        if isinstance(value,(tuple,list)):return tuple(freeze(v) for v in value)
        if isinstance(value,np.ndarray):return value.dtype.str,value.shape,value.tobytes()
        hash(value)
        return value
    def operand(value):
        if not isinstance(value,Literal):return aliases.get(value,value)
        data=np.asarray(value.val)
        return ('literal',data.dtype.str,data.shape,data.tobytes(),value.aval.weak_type)
    for eq in graph.jaxpr.eqns:
        inputs=[v if isinstance(v,Literal) else aliases.get(v,v) for v in eq.invars]
        key=None
        if not eq.effects or (readonly and eq.primitive.name in ('get','masked_load')):
            try:key=eq.primitive,tuple(operand(v) for v in inputs),freeze(eq.params)
            except TypeError:pass
        if key is not None and key in expressions:
            index=expressions[key];previous=kept[index]
            # Retain newly used outputs when an earlier identical call dropped
            # them. A DropVar must never become an input alias.
            outputs=[new if isinstance(old,DropVar) else old for old,new in zip(previous.outvars,eq.outvars)]
            kept[index]=previous.replace(outvars=outputs)
            aliases.update((new,old) for new,old in zip(eq.outvars,outputs) if not isinstance(new,DropVar))
        else:
            kept.append(eq.replace(invars=inputs))
            if key is not None:expressions[key]=len(kept)-1
    graph=type(graph)(graph.jaxpr.replace(eqns=kept,outvars=[v if isinstance(v,Literal) else aliases.get(v,v) for v in graph.jaxpr.outvars]),graph.consts)
    live,_=partial_eval.dce_jaxpr(graph.jaxpr,[True]*len(graph.jaxpr.outvars),instantiate=True)
    return type(graph)(live,graph.consts)

def _kernel_integer_power(value, exponent):
    """Use the same power-by-squaring order with literal-safe GPU operations."""
    if exponent == 0:
        return jnp.ones_like(value)
    reciprocal = exponent < 0
    exponent = abs(exponent)
    result = None
    while exponent:
        exponent, odd = divmod(exponent, 2)
        if odd:
            result = value if result is None else result * value
        if exponent:
            value = value * value
    return jnp.ones_like(result) / result if reciprocal else result


def _evaluate_kernel(graph, constants, *arguments, scalarize=False):
    literals={}
    def local_constant(value):
        if not isinstance(value,np.ndarray) or not value.ndim:return value
        key=value.dtype.str,value.shape,value.tobytes()
        if key not in literals:
            first=value.flat[0]
            if np.issubdtype(value.dtype,np.number) and value.tobytes()==np.arange(value.size,dtype=value.dtype).tobytes():
                result=jnp.arange(value.size,dtype=value.dtype).reshape(value.shape)
            else:
                result=jnp.full(value.shape,first.item(),dtype=value.dtype)
                index=jnp.arange(value.size).reshape(value.shape)
                for i,item in enumerate(value.flat):
                    if item.tobytes()!=first.tobytes():
                        result=jnp.where(index==i,jnp.asarray(item.item(),dtype=value.dtype),result)
            literals[key]=result
        return literals[key]
    def shrink(value):
        return value.reshape(()) if scalarize and getattr(value, 'size', 0) == 1 else value
    constants = tuple((shrink(v) for v in constants))
    env = dict(zip(graph.constvars, constants))
    env.update(zip(graph.invars, arguments))

    def read(v):
        return np.asarray(v.val, dtype=v.aval.dtype) if isinstance(v, Literal) else env[v]
    for eq in graph.eqns:
        vals = [read(v) for v in eq.invars]
        name, p = (eq.primitive.name, eq.params)
        if scalarize and len(eq.outvars) == 1 and (eq.outvars[0].aval.size == 1):
            if name in ('broadcast_in_dim', 'reshape', 'squeeze', 'slice', 'dynamic_slice', 'gather', 'reduce_sum', 'reduce_or', 'reduce_and', 'stack'):
                if vals[0].size == 1:
                    env[eq.outvars[0]] = shrink(vals[0])
                    continue
            if name == 'dynamic_update_slice' and vals[0].size == vals[1].size == 1:
                env[eq.outvars[0]] = shrink(vals[1])
                continue
        numpy_ops = {'add': np.add, 'add_any': np.add, 'sub': np.subtract, 'mul': np.multiply, 'div': np.divide, 'neg': np.negative, 'abs': np.abs, 'sqrt': np.sqrt, 'log': np.log, 'exp': np.exp, 'pow': np.power, 'square': np.square, 'max': np.maximum, 'min': np.minimum, 'lt': np.less, 'le': np.less_equal, 'gt': np.greater, 'ge': np.greater_equal, 'eq': np.equal, 'ne': np.not_equal, 'and': np.bitwise_and, 'or': np.bitwise_or}
        if all((not isinstance(v, jax.core.Tracer) for v in vals)):
            vals=[np.asarray(value) for value in vals]
            folded = None
            if name in numpy_ops:
                folded = [numpy_ops[name](*vals)]
            elif name == 'integer_pow':
                folded = [np.power(vals[0], p['y'])]
            elif name == 'convert_element_type':
                folded = [np.asarray(vals[0], dtype=p['new_dtype'])]
            elif name == 'broadcast_in_dim':
                shape=[1]*len(p['shape'])
                for i,axis in enumerate(p['broadcast_dimensions']):shape[axis]=vals[0].shape[i]
                folded=[np.broadcast_to(np.reshape(vals[0],shape),p['shape'])]
            elif name in ('reshape','squeeze','transpose'):
                value=vals[0]
                dimensions=p.get('dimensions') if name=='reshape' else p.get('permutation') if name=='transpose' else None
                if dimensions is not None:value=np.transpose(value,dimensions)
                folded=[np.reshape(value,eq.outvars[0].aval.shape)]
            elif name=='iota':
                shape=[1]*len(p['shape']);shape[p['dimension']]=p['shape'][p['dimension']]
                folded=[np.broadcast_to(np.arange(shape[p['dimension']],dtype=p['dtype']).reshape(shape),p['shape'])]
            elif name in ('concatenate','stack'):
                folded=[(np.concatenate if name=='concatenate' else np.stack)(vals,axis=p.get('dimension',p.get('axis')))]
            if folded is not None:
                env.update(((out, np.asarray(value, dtype=out.aval.dtype)) for out, value in zip(eq.outvars, folded)))
                continue
        vals=[local_constant(value) for value in vals]
        if name == 'slice':
            value = vals[0]
            for axis, (start, end, size) in enumerate(zip(p['start_indices'], p['limit_indices'], value.shape)):
                if start != 0 or end != size:
                    if end - start != 1:
                        raise ValueError(('slice', p))
                    value = _split_select(value, start, axis)
            result = [value]
        elif name == 'dynamic_slice':
            value = vals[0]
            for axis, (index, size, wanted) in enumerate(zip(vals[1:], value.shape, p['slice_sizes'])):
                if size != wanted:
                    if wanted != 1:
                        raise ValueError(('dynamic slice', p))
                    value = _split_select(value, jnp.clip(index, 0, size - 1), axis)
            result = [value]
        elif name == 'gather' and vals[0].ndim == 1 and (p['slice_sizes'] == (1,)):
            operand, indices = vals
            if indices.size != 1:
                raise ValueError(('gather indices', indices.shape, p))
            index = jnp.clip(indices.reshape(()), 0, len(operand) - 1)
            result = [_split_select(operand, index, 0).reshape(eq.outvars[0].aval.shape)]
        elif name == 'stack' and len(vals) == 1:
            result = [jnp.expand_dims(vals[0], p['axis'])]
        elif name == 'unstack':
            result = [_split_select(vals[0], i, p['axis']).reshape(out.aval.shape) for i, out in enumerate(eq.outvars)]
        elif name == 'split' and vals[0].ndim == 1 and all((out.aval.size == 1 for out in eq.outvars)):
            result = [_split_select(vals[0], i, 0).reshape(out.aval.shape) for i, out in enumerate(eq.outvars)]
        elif name == 'reduce_or':
            result = [jnp.max(vals[0].astype(jnp.int32), axis=p['axes']) > 0]
        elif name == 'reduce_and':
            result = [jnp.min(vals[0].astype(jnp.int32), axis=p['axes']) > 0]
        elif name == 'add_any':
            result = [vals[0] + vals[1]]
        elif name == 'pad' and vals[0].size == 1 and (len(p['padding_config']) == 1):
            low, _, _ = p['padding_config'][0]
            result = [jnp.where(jnp.arange(eq.outvars[0].aval.size) == low, vals[0].reshape(()), vals[1])]
        elif name == 'erf':
            result = [_kernel_erf().bind(vals[0])]
        elif name == 'integer_pow':
            result = [_kernel_integer_power(vals[0], p['y'])]
        elif name == 'le_to':
            result = [jnp.less_equal(vals[0], vals[1]) | jnp.isnan(vals[1])]
        elif name=='stack':
            axis=p['axis'];shape=[1]*(vals[0].ndim+1);shape[axis]=len(vals)
            index=jnp.arange(len(vals)).reshape(shape)
            value=jnp.zeros(eq.outvars[0].aval.shape,vals[0].dtype)
            for i,v in enumerate(vals):value=jnp.where(index==i,jnp.expand_dims(v,axis),value)
            result=[value]
        elif name=='concatenate' and (len(vals)>2 or all(v.size==1 for v in vals)):
            outshape = eq.outvars[0].aval.shape
            if any((v.size != 1 for v in vals)):
                raise ValueError(('concatenate', p))
            result = jnp.zeros((len(vals),), vals[0].dtype)
            for i, v in enumerate(vals):
                result = jnp.where(jnp.arange(len(vals)) == i, v.reshape(()), result)
            result = [result.reshape(outshape)]
        elif name == 'is_finite':
            result = [_kernel_is_finite(vals[0]) if isinstance(vals[0], jax.core.Tracer) else np.isfinite(vals[0])]
        else:
            parameters = dict(p)
            if name == 'convert_element_type' and parameters.get('weak_type'):
                parameters['weak_type'] = False
            result = _bind_primitive(eq.primitive, vals, parameters)
            if not eq.primitive.multiple_results:
                result = [result]
        env.update(((v, shrink(value)) for v, value in zip(eq.outvars, result)))
    return tuple((read(v) for v in graph.outvars))

def _evaluate_surface(graph, constants, state):
    """Scalarize short state-vector operations in a spatial level set.

    A physical five/six-vector is represented as scalar components, avoiding
    non-power-of-two local arrays. This covers scalar indexing, vector norms
    and dot products used by the toroidal/spherical safety surfaces.
    """
    word = lambda value, i: _select_one(value, i, 0).reshape(())
    env = dict(zip(graph.constvars, constants))
    env[graph.invars[0]] = tuple((word(state, i) for i in range(graph.invars[0].aval.size)))

    def read(var):
        return np.asarray(var.val, dtype=var.aval.dtype) if isinstance(var, Literal) else env[var]
    for eq in graph.eqns:
        values = [read(var) for var in eq.invars]
        name, p = (eq.primitive.name, eq.params)
        vector = lambda x: isinstance(x, tuple)
        if name == 'slice' and vector(values[0]):
            result = [values[0][p['start_indices'][0]:p['limit_indices'][0]:None if p.get('strides') is None else p['strides'][0]]]
        elif name == 'squeeze' and vector(values[0]) and (len(values[0]) == 1):
            result = [values[0][0]]
        elif name == 'unstack':
            result = list(values[0])
        elif name in ('stack', 'concatenate'):
            result = [tuple((item for value in values for item in (value if vector(value) else (value,))))]
        elif name in ('reshape', 'broadcast_in_dim'):
            size = eq.outvars[0].aval.size
            result = [values[0] if vector(values[0]) else (values[0],) * size]
            if not eq.outvars[0].aval.shape:
                result = [result[0][0]]
        elif name in ('reduce_sum', 'reduce_max', 'reduce_min') and vector(values[0]):
            value = values[0][0]
            combine = {'reduce_sum': lambda a, b: a + b, 'reduce_max': jnp.maximum, 'reduce_min': jnp.minimum}[name]
            for item in values[0][1:]:
                value = combine(value, item)
            result = [value]
        elif name == 'dot_general' and all((vector(value) for value in values)):
            result = [sum((a * b for a, b in zip(*values)))]
        elif name == 'gather' and vector(values[0]) and (len(values[1]) == 1):
            index = values[1][0]
            value = values[0][0]
            for i, item in enumerate(values[0]):
                value = jnp.where(index == i, item, value)
            result = [(value,) if eq.outvars[0].aval.shape else value]
        else:
            parameters = dict(p)
            if name == 'convert_element_type':
                parameters['weak_type'] = False
            if any((vector(value) for value in values)):
                width = max((len(value) for value in values if vector(value)))
                elements = []
                for i in range(width):
                    elements.append(_bind_primitive(eq.primitive, [value[i] if vector(value) else value for value in values], parameters))
                result = [tuple(elements)]
            else:
                value = _bind_primitive(eq.primitive, values, parameters)
                result = value if eq.primitive.multiple_results else [value]
        env.update(zip(eq.outvars, result))
    result = read(graph.outvars[0])
    return result[0] if isinstance(result, tuple) else result

def _draw_normal(key, dim):
    if not jax.config.jax_threefry_partitionable:
        raise ValueError('The exact adapter currently requires partitionable Threefry')
    word = lambda a, i: _select_one(a, i, 0).reshape(())
    k0, k1 = (word(key, 0), word(key, 1))
    a, b = threefry._threefry2x32_lowering(k0, k1, jnp.asarray(0, jnp.uint32), jnp.arange(2, dtype=jnp.uint32), use_rolled_loops=False)
    next_key = jnp.where(jnp.arange(2) == 0, word(a, 0), word(b, 0))
    d0, d1 = (word(a, 1), word(b, 1))
    padded = 1 << (dim - 1).bit_length()
    hi, lo = threefry._threefry2x32_lowering(d0, d1, jnp.asarray(0, jnp.uint32), jnp.arange(padded, dtype=jnp.uint32), use_rolled_loops=False)
    bits = hi.astype(jnp.uint64) << jnp.uint64(32) | lo.astype(jnp.uint64)
    unit = jax.lax.bitcast_convert_type(bits >> jnp.uint64(12) | jnp.uint64(4607182418800017408), jnp.float64) - 1.0
    lower = np.nextafter(np.float64(-1.0), np.float64(0.0))
    uniform = jnp.maximum(lower, unit * (1.0 - lower) + lower)
    return (next_key, jnp.sqrt(jnp.asarray(2.0, jnp.float64)) * jax.lax.erf_inv(uniform))

def _prepare_split_kernel(arrays, factory, times, *, adaptive, dt0, dtmax, speed, tolerance=1e-09, noise_dim=2, max_steps=100000, max_orbit_steps=10000, max_pending=64, warps=1, state_size=8, bank_components=8, orbit_dt0=None, final_time=None, align_orbit=False, interpret=False,
        record=False, adjoint_checkpoint_steps=16, adjoint_orbit_checkpoint_steps=16,
        adjoint_max_accepted_steps=None, adjoint_max_orbit_steps=None, collision_indices=None, time_dependent=False):
    """Persistent particle loop; factory also supplies the optional predictor."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt
    times = jnp.asarray(times, jnp.float64)
    nsave, stages, bank_size = (len(times), len(_ERROR8), 16)
    matrix = np.zeros((bank_size, bank_size))
    for i, row in enumerate(_A8, 1):
        matrix[i, :len(row)] = row
    dense = np.zeros((len(_DENSE8[0]), bank_size))
    dense[:, :stages] = np.asarray(_DENSE8).T
    nodes = (0., *_C8) if time_dependent else ()
    inputs = tuple(arrays) + (jnp.asarray(matrix), jnp.asarray(dense), jnp.asarray(tuple(_ERROR8) + (0.0,) * (bank_size - stages)), jnp.asarray(times), jnp.asarray([0.0, dt0, dtmax, times[-1] if final_time is None else final_time, speed, 2 * tolerance, dt0 if orbit_dt0 is None else orbit_dt0, *nodes], jnp.float64))
    capacity = max_pending if adaptive else 1
    noise_size = 1 << (noise_dim - 1).bit_length()

    if record:
        from .native_ad import _SplitCheckpointTape, _TapeView
        outer_capacity = min(max_steps,adjoint_max_accepted_steps or max_steps)
        inner_capacity = adjoint_max_orbit_steps or max_steps
        inner_segment = adjoint_orbit_checkpoint_steps
        outer_segment = adjoint_checkpoint_steps
        if collision_indices is None:
            node_capacity = (inner_capacity+inner_segment-1)//inner_segment+outer_capacity-1
            derivative_capacity = node_capacity
        else:
            nonempty_maps = min(outer_capacity,inner_capacity)
            node_capacity = nonempty_maps+(inner_capacity-nonempty_maps)//inner_segment
            derivative_capacity = (inner_capacity-1)//inner_segment
        endpoint_size = 8 if collision_indices is None else len(collision_indices)
        shape = lambda dims,dtype=jnp.float64:jax.ShapeDtypeStruct(dims,dtype)
        tape_shapes = _SplitCheckpointTape(
            shape((outer_capacity,2)),shape((outer_capacity,noise_dim)),shape((outer_capacity,),jnp.bool_),
            shape((outer_capacity+1,),jnp.int32),shape((inner_capacity,2)),shape((outer_capacity,)),
            shape(((outer_capacity+outer_segment-1)//outer_segment,8)),
            shape((),jnp.int32),shape((),jnp.int32),shape((),jnp.int32),shape((),jnp.int32),shape((),jnp.int32),
            shape((outer_capacity+1,),jnp.int32),shape((node_capacity,8)),
            shape((derivative_capacity,8)),shape((outer_capacity,endpoint_size)),shape((),jnp.int32))
        padded_tape_shapes = tuple(shape(tuple(1<<max(size-1,0).bit_length() for size in value.shape),value.dtype)
                                   for value in tape_shapes)
        tape_layout=[]
        bank_sizes=[0,0]
        for spec in padded_tape_shapes:
            bank=0 if jnp.issubdtype(spec.dtype,jnp.floating) else 1
            tape_layout.append((bank,bank_sizes[bank],spec.shape))
            bank_sizes[bank]+=int(np.prod(spec.shape))
        bank_sizes=[((size+127)//128)*128 for size in bank_sizes]
        tape_bank_shapes=(shape((bank_sizes[0],)),shape((bank_sizes[1],),jnp.int32))

    def solve(initial, key):

        def kernel(yref, kref, *refs):
            if record:
                banks=refs[-2:]
                refs=refs[:-2]
                tape_refs={name:_TapeView(banks[bank],offset,shape_) for name,(bank,offset,shape_) in
                           zip(_SplitCheckpointTape._fields,tape_layout)}
                for ref,size in zip(banks,bank_sizes):
                    def clear(i,_):
                        ref[i*128+jnp.arange(128)]=jnp.zeros((128,),ref.dtype)
                    jax.lax.fori_loop(0,size//128,clear,None)
                tape_refs['fractions'][...]=jnp.ones(tape_refs['fractions'].shape,jnp.float64)
            *data, aref, dref, eref, tsref, pref, ysout, yout, tout, statsout, statusout, ends, increments = refs
            y0 = yref[...]
            zero, first_h, ceiling, final_t, birth_speed, budget, first_orbit_h = (pref[i] for i in range(7))
            operations=factory(data,y0)
            rhs,kick,norm,level=operations[:4]
            step_limit=operations[4] if len(operations)==5 else None

            def dynamic_rhs(t,y):
                value = rhs(t,y) if time_dependent else rhs(y)
                return pack(tuple((scalar(value, i) for i in range(4)))) if bank_components == 4 else value

            def expand(value):
                return pack(tuple((scalar(value, i) for i in range(4))) + (zero, zero, zero, zero)) if bank_components == 4 else value
            integer = lambda x: jnp.asarray(x, jnp.int32)
            finite = lambda x: jnp.min(_kernel_is_finite(x+zero).astype(jnp.int32)) > 0 if isinstance(x,jax.core.Tracer) else jnp.asarray(np.isfinite(x).all())
            bi = jnp.arange(bank_size)[:, None]

            def polynomial(y, endpoint, bank, fraction):
                weights = jnp.zeros((bank_size,), y.dtype)
                for power in range(dense.shape[0]):
                    weights = weights * fraction + dref[power, :]
                result = y + expand(jnp.sum((weights * fraction)[:, None] * bank, axis=0))
                return jnp.where(fraction == 0, y, jnp.where(fraction == 1, endpoint, result))

            def orbit(y, t, end, guess, outer_index, inner_index, node_index):
                old = level(y)
                start = (t, y, dynamic_rhs(t,y), jnp.minimum(guess, end - t), old, integer(0), integer(0), integer(0)) + ((zero+1,) if record else ())

                def step(c):
                    t, y, f0, guess, old, status, trials, rejected = c[:8]
                    last_fraction = c[8] if record else zero+1
                    remaining = end - t
                    h = jnp.minimum(guess, remaining)
                    close = remaining - h <= 8 * jnp.finfo(y.dtype).eps * jnp.maximum(jnp.abs(t), jnp.abs(end))
                    h = jnp.where(close, remaining, h)
                    stop = jnp.where(h == remaining, end, t + h)
                    bank = jnp.where(bi == 0, h * f0[None, :], jnp.zeros((bank_size, bank_components), y.dtype))

                    def stage(i, carry):
                        bank, _, checked = carry
                        contribution = jnp.where(bi < i, aref[i, :][:, None] * bank, 0.0)
                        yi = y + expand(jnp.sum(contribution, axis=0))
                        fi = dynamic_rhs(t+pref[7+i]*h if time_dependent else t,yi)
                        increment = h * fi
                        return (jnp.where(bi == i, increment[None, :], bank), fi, checked & finite(increment) & finite(fi))
                    bank, f1, stage_finite = jax.lax.fori_loop(1, stages, stage, (bank, f0, finite(h * f0) & finite(f0)))
                    endpoint = y + expand(jnp.sum(jnp.where(bi < stages - 1, aref[stages - 1, :][:, None] * bank, 0.0), axis=0))
                    embedded = expand(jnp.sum(jnp.where(bi < stages, eref[...][:, None] * bank, 0.0), axis=0))
                    error = norm(endpoint, embedded, birth_speed, budget) if adaptive else zero
                    new = level(endpoint)
                    valid = finite(endpoint) & stage_finite & finite(f1) & finite(embedded) & finite(error) & finite(new) & (error >= 0)
                    accept = valid & ((error <= 1) | (not adaptive)) & (stop > t)
                    crossing = accept & (((old > 0) != (new > 0)) | (new == 0))
                    factor = jnp.where(valid, jnp.clip(0.9 * jnp.maximum(error, 1e-16) ** (-1 / 8), 0.2, 5.0), 0.2)
                    proposal = h * jnp.where(accept, factor, jnp.minimum(factor, 0.9)) if adaptive else first_orbit_h

                    def bisect(c):
                        lo, hi, i, ok = c
                        mid = (lo + hi) / 2
                        value = level(polynomial(y, endpoint, bank, mid))
                        same = (value > 0) == (old > 0)
                        return (jnp.where(same, mid, lo), jnp.where(same, hi, mid), i + 1, ok & finite(value))
                    lo, hi, _, ok = jax.lax.while_loop(lambda c: crossing & c[3] & (c[2] < 52), bisect, (zero, zero + 1, integer(0), jnp.asarray(True)))
                    root = (lo + hi) / 2
                    failed_root = crossing & ~ok
                    accept &= ~failed_root
                    proposal = jnp.where(failed_root, 0.2 * h, proposal)
                    stop_y = jax.lax.cond(crossing, lambda _: polynomial(y, endpoint, bank, root), lambda _: endpoint, None)
                    stuck = (stop <= t) | ~accept & ((h <= 0) | (t + proposal <= t))
                    status = jnp.where(crossing & ok, 1, jnp.where(stuck, jnp.where(valid, 3, 4), 0)).astype(jnp.int32)
                    if not adaptive:
                        status = jnp.where(~valid, 4, status)
                    if record:
                        local_index = trials-rejected
                        slot = inner_index+local_index
                        node = node_index+local_index//inner_segment
                        def save_step(_):
                            tape_refs['orbit_times'][slot,0] = t
                            tape_refs['orbit_times'][slot,1] = h
                            def checkpoint(_):
                                tape_refs['orbit_states'][node,:] = y
                                if collision_indices is None:
                                    tape_refs['orbit_derivatives'][node,:] = expand(f0)
                                elif derivative_capacity:
                                    def derivative(_):
                                        tape_refs['orbit_derivatives'][node-outer_index-1,:] = expand(f0)
                                    jax.lax.cond(local_index>0,derivative,lambda _:None,None)
                            jax.lax.cond(local_index%inner_segment==0,checkpoint,lambda _:None,None)
                        jax.lax.cond(accept&(outer_index<outer_capacity)&(slot<inner_capacity)&(node<node_capacity),
                                     save_step,lambda _:None,None)
                    return (jnp.where(accept, jnp.where(crossing, t + root * h, stop), t), jnp.where(accept, stop_y, y), jnp.where(accept, f1, f0), proposal, jnp.where(accept, new, old), status, trials + 1, rejected + (~accept).astype(jnp.int32)) + ((jnp.where(accept,jnp.where(crossing,root,zero+1),last_fraction),) if record else ())
                out = jax.lax.while_loop(lambda c: (c[0] < end) & (c[5] == 0) & (c[6] < max_orbit_steps), step, start)
                t, y, _, guess, _, status, trials, rejected = out[:8]
                fraction = out[8] if record else zero+1
                status = jnp.where((t < end) & (status == 0), 2, status).astype(jnp.int32)
                return (t, y, guess, status, trials, rejected, fraction)

            def save(y, t, index):

                def store(i):
                    ysout[i, :] = y
                    return i + 1
                return jax.lax.while_loop(lambda i: (i < nsave) & (tsref[jnp.minimum(i, nsave - 1)] <= t), store, index)
            valid_inputs = (first_h > 0) & finite(first_h) & (ceiling > 0) & ~jnp.isnan(ceiling) & (first_orbit_h > 0) & finite(first_orbit_h) & (birth_speed > 0) & finite(birth_speed) & (budget > 0) & finite(budget) & (final_t >= 0) & finite(final_t)
            previous = zero
            for i in range(nsave):
                stamp = tsref[i]
                valid_inputs &= finite(stamp) & (stamp >= previous) & (stamp <= final_t)
                previous = stamp
            initial_status = jnp.where(valid_inputs, jnp.where(finite(y0) & finite(level(y0)), jnp.where(level(y0) <= 0, 1, 0), 4), 5).astype(jnp.int32)
            index = save(y0, zero, integer(0))
            if record:
                tape_refs['initial_saves'][...] = index
            start = (zero, y0, jnp.minimum(first_h, ceiling), first_orbit_h, kref[...], integer(0), integer(0), index, integer(0), initial_status, integer(0), integer(0), integer(0), integer(0), integer(0), zero,integer(0),integer(0)) + ((integer(0),)*3 if record else ())

            def advance(c):
                t, y, h, oh, key, size, peak, index, grid, status, trials, accepted, rejected, ot, orr, maxerr,path_steps,path_max = c[:18]
                ad_count,ad_inner,ad_nodes = c[18:] if record else (integer(0),)*3
                barrier = jnp.where(index < nsave, tsref[jnp.minimum(index, nsave - 1)], final_t)
                limit=step_limit(t,y) if adaptive and step_limit is not None else zero+jnp.inf
                valid_limit=(limit>0)&~jnp.isnan(limit)
                if adaptive:
                    wanted = jnp.minimum(t + jnp.minimum(h,limit), barrier)
                    if align_orbit:
                        length=wanted-t
                        near=finite(oh)&(oh>0)&(wanted<barrier)&(oh>=.9*length)&(oh<length*(1-1e-12))
                        wanted=jnp.where(near,t+oh,wanted)
                    grid_end = wanted
                else:
                    grid_end = (grid + 1) * jnp.minimum(first_h, ceiling)
                    close = jnp.abs(grid_end - barrier) <= 8 * jnp.finfo(y.dtype).eps * jnp.maximum(jnp.abs(grid_end), jnp.abs(barrier))
                    grid_end = jnp.where(close, barrier, grid_end)
                    wanted = jnp.minimum(grid_end, barrier)

                pending_end=jax.lax.cond(size>0,lambda _:ends[size-1],lambda _:wanted,None)
                draw=(size==0)|((wanted<pending_end)&(size<capacity))
                key,z=jax.lax.cond(draw,lambda key:_draw_normal(key,noise_dim),
                    lambda key:(key,jnp.zeros((noise_size,),y.dtype)),key)
                def fresh(c):
                    size, peak = c
                    ends[0], increments[0, :] = (wanted, jnp.sqrt(wanted - t) * z)
                    return (size + 1, jnp.maximum(peak, 1))
                size, peak = jax.lax.cond(size == 0, fresh, lambda c: c, (size, peak))
                parent_end, parent = (ends[size - 1], increments[size - 1, :])
                end = jnp.minimum(wanted, parent_end)
                duration, h = (parent_end - t, end - t)
                split = end < parent_end
                overflow = split & (size == capacity)

                def bridge(c):
                    size, peak = c
                    alpha = h / duration
                    left = alpha * parent + jnp.sqrt(h * (1 - alpha)) * z
                    increments[size - 1, :], increments[size, :], ends[size] = (parent - left, left, end)
                    return (size + 1, jnp.maximum(peak, size + 1))
                size, peak = jax.lax.cond(split & ~overflow, bridge, lambda c: c, (size, peak))
                usable = valid_limit & ~overflow & (h > 0) & (end > t)

                def attempt(_):
                    stop, endpoint, next_oh, os, orbit_trials, orbit_rejected, fraction = orbit(y, t, end, jnp.minimum(oh, h) if adaptive else jnp.minimum(first_orbit_h, h), ad_count, ad_inner, ad_nodes)

                    def collide(_):
                        return kick(endpoint, h, increments[size - 1, :])
                    trial, ed, es = jax.lax.cond(os == 0, collide, lambda _: (endpoint, zero, zero), None)
                    valid = finite(trial) & finite(level(trial)) & finite(ed) & finite(es) & (ed >= 0) & (es >= 0)
                    error = jnp.maximum(ed, es)
                    accept = (os == 0) & valid & ((error <= 1) | (not adaptive))
                    reject = (os == 0) & valid & ~accept
                    event = os == 1
                    ns = jnp.where(os != 0, os, jnp.where(~valid, 4, jnp.where(accept & (level(trial) <= 0), 1, 0))).astype(jnp.int32)
                    if adaptive:
                        factor = 0.9 * jnp.minimum(jnp.maximum(ed, 1e-30) ** (-0.5), jnp.maximum(es, 1e-30) ** (-2 / 3))
                        upper = jnp.where(accept, zero + 1.5, zero + 0.9)
                        factor = jnp.maximum(zero + 0.1, jnp.minimum(factor, upper))
                        nh = jnp.minimum(h * factor, ceiling)
                    else:
                        nh = first_h
                    nt = jnp.where(event, stop, jnp.where(accept, end, t))
                    ny = jnp.where(event, endpoint, jnp.where(accept, trial, y))
                    ni = save(ny, nt, index)
                    committed = accept|event
                    inner_count = orbit_trials-orbit_rejected
                    if record:
                        nodes = (inner_count+inner_segment-1)//inner_segment
                        def commit(_):
                            tape_refs['times'][ad_count,0],tape_refs['times'][ad_count,1] = t,nt
                            for component in range(noise_dim):
                                tape_refs['increments'][ad_count,component] = increments[size-1,component]
                            tape_refs['kicks'][ad_count] = accept
                            tape_refs['offsets'][ad_count+1] = ad_inner+inner_count
                            tape_refs['checkpoint_offsets'][ad_count+1] = ad_nodes+nodes
                            tape_refs['fractions'][ad_count] = fraction
                            endpoint_components = range(8) if collision_indices is None else collision_indices
                            for component,index_ in enumerate(endpoint_components):
                                tape_refs['orbit_endpoints'][ad_count,component] = scalar(endpoint,index_)
                            def outer_checkpoint(_):
                                tape_refs['states'][ad_count//outer_segment,:] = y
                            jax.lax.cond(ad_count%outer_segment==0,outer_checkpoint,lambda _:None,None)
                        jax.lax.cond(committed&(ad_count<outer_capacity)&(ad_inner+inner_count<=inner_capacity),
                                     commit,lambda _:None,None)
                        next_nodes = ad_nodes+jnp.where(committed,nodes,0)
                    else:
                        next_nodes = ad_nodes
                    kept=jnp.where(accept|event,orbit_trials-orbit_rejected,0)
                    return (nt, ny, nh, next_oh, key, size - accept.astype(jnp.int32), peak, ni, grid + (accept & (end == grid_end)).astype(jnp.int32), ns, trials + 1, accepted + accept.astype(jnp.int32), rejected + reject.astype(jnp.int32), ot + orbit_trials, orr + orbit_rejected, jnp.maximum(maxerr, jnp.where(accept, error, zero)),path_steps+kept,jnp.maximum(path_max,kept)) + ((ad_count+committed.astype(jnp.int32),ad_inner+jnp.where(committed,inner_count,0),next_nodes) if record else ())
                return jax.lax.cond(usable, attempt, lambda _: c[:4] + (key, size, peak, index, grid, jnp.where(~valid_limit,5,jnp.where(overflow,6,3)).astype(jnp.int32)) + c[10:], None)
            out = jax.lax.while_loop(lambda c: (c[0] < final_t) & (c[9] == 0) & (c[10] < max_steps), advance, start)
            t, y, _, _, _, _, peak, index, _, status, trials, accepted, rejected, ot, orr, maxerr,path_steps,path_max = out[:18]
            ad_count,ad_inner,ad_nodes = out[18:] if record else (integer(0),)*3
            status = jnp.where((t < final_t) & (status == 0), 2, status)

            def tail(i, _):
                ysout[i, :] = y
                return ()
            jax.lax.fori_loop(index, nsave, tail, ())
            yout[...], tout[...], statusout[...] = (y, t, status)
            statsout[...] = pack(tuple((v.astype(y.dtype) for v in (trials, accepted, rejected, ot, orr, peak))) + (maxerr,path_steps.astype(y.dtype),path_max.astype(y.dtype))+(zero,)*7)
            if record:
                tape_refs['count'][...],tape_refs['orbit_count'][...] = ad_count,ad_inner
                tape_refs['checkpoint_count'][...] = ad_nodes
                tape_refs['saved_count'][...],tape_refs['status'][...] = index,status.astype(jnp.int32)

        shapes = (jax.ShapeDtypeStruct((nsave, 8), jnp.float64), jax.ShapeDtypeStruct((8,), jnp.float64), jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((16,), jnp.float64), jax.ShapeDtypeStruct((), jnp.int32), jax.ShapeDtypeStruct((capacity,), jnp.float64), jax.ShapeDtypeStruct((capacity, noise_size), jnp.float64))
        outputs = pl.pallas_call(kernel, out_shape=shapes + (tape_bank_shapes if record else ()), grid=(), compiler_params=plt.CompilerParams(num_warps=warps), interpret=interpret)(initial, key, *inputs)
        ys, y, t, stats, status = outputs[:5]
        if record:
            banks=outputs[len(shapes):]
            values=(banks[bank][offset:offset+int(np.prod(shape_))].reshape(shape_).astype(spec.dtype)
                    for (bank,offset,shape_),spec in zip(tape_layout,tape_shapes))
            tape = _SplitCheckpointTape(*(value[tuple(slice(size) for size in spec.shape)]
                for value,spec in zip(values,tape_shapes)))
            return (ys,y,t,stats,status),tape
        return (ys, y, t, stats, status)
    return solve


def _validate_independent(field, particles, species, *, model, integrator, adaptive,
        orbit_adaptive, controller, collision_controller, electric_field,
        spatial_diffusion, orbit, record_steps, reuse_orbit_step, event, times, tmax):
    from .fields import BiotSavart, Vmec
    from .boozer import BoozerField
    from .electric_field import Electric_field_zero
    if integrator != "essos_dopri8" or controller != "i" or collision_controller != "conservative":
        raise ValueError("independent execution requires Dopri8/I and conservative collision control")
    if adaptive != orbit_adaptive or not reuse_orbit_step:
        raise ValueError("independent execution requires matching adaptation and orbit-step reuse")
    if not isinstance(electric_field, Electric_field_zero):
        raise ValueError("independent execution requires zero electric field")
    if record_steps:
        raise ValueError("independent execution does not provide step histories")
    if jax.default_backend() != "gpu" or not jax.config.jax_enable_x64:
        raise ValueError("independent execution requires GPU float64")
    if jax.random.key_data(particles.random_keys).shape[-1] != 2:
        raise ValueError("independent execution requires Threefry particle keys")
    if not len(times):
        raise ValueError("independent execution requires nonempty save times")
    for function in tree_util.tree_leaves(event):
        shape = jax.eval_shape(lambda t, y: function(t, y, None), jnp.asarray(0.),
                               jnp.zeros(5 if model == "GuidingCenter" else 6))
        if shape.shape or not jnp.issubdtype(shape.dtype, jnp.floating):
            raise ValueError("independent events must be scalar floating positive-inside levels")
        graph = _flatten(lambda t, y: function(t, y, None), jnp.asarray(0.),
                          jnp.zeros(5 if model == "GuidingCenter" else 6))
        live = {graph.jaxpr.invars[0]}
        for equation in graph.jaxpr.eqns:
            if any(v in live for v in equation.invars if not isinstance(v, Literal)):
                live.update(equation.outvars)
        if not isinstance(graph.jaxpr.outvars[0], Literal) and graph.jaxpr.outvars[0] in live:
            raise ValueError("independent events must be time independent")


def _coefficient_graph(species, full_orbit, mass=M, charge=Q):
    """Sum the physical rates and hoist species-only work out of speed updates."""
    from jax._src.interpreters.partial_eval import partial_eval_jaxpr_nounits
    from jax.extend.core import ClosedJaxpr
    def coefficients(speed):
        def values(v):
            point = jnp.zeros(4)
            total=lambda function:sum(function(mass,charge,i,v,point,species) for i in range(species.number_species))
            nd,np_,dnp,ns=(total(function) for function in (nu_D_ab,nu_par_ab,d_nu_par_ab,nu_s_ab))
            return .5*v*v*np_, .5*v*v*nd, v*np_+.5*v*v*dnp, -ns*v
        val, slope = jax.jvp(values, (speed,), (jnp.ones_like(speed),))
        dp, dt, ddp, q = val
        dk = slope[3]+slope[2]+2*(slope[0]-slope[1])/speed-2*(dp-dt)/speed**2 if full_orbit else jnp.zeros_like(speed)
        return dp, dt, ddp, q, slope[3], dk
    graph = _flatten(coefficients, jnp.asarray(4e5))
    setup, stage, _, _ = partial_eval_jaxpr_nounits(graph, [True], instantiate=True)
    fixed = tuple(jax.core.eval_jaxpr(setup.jaxpr, setup.consts))
    body = stage.jaxpr
    return ClosedJaxpr(body.replace(constvars=list(body.constvars)+list(body.invars[:len(fixed)]),
        invars=body.invars[len(fixed):]), tuple(stage.consts)+fixed)


def _split_field_data(field):
    """Prepare field views before vmap/JIT, including cached coil properties."""
    from .fields import BiotSavart, Vmec
    coil, vmec = isinstance(field, BiotSavart), isinstance(field, Vmec)
    if coil:
        arrays, nc, nq = _coil_geometry(field)
    elif vmec:
        with jax.ensure_compile_time_eval():
            data = _prepare_vmec_data(field)
        arrays = data.arrays
    else:
        if field.table is not None or field.sine_coef is not None:
            raise ValueError("independent Boozer execution requires fixed cosine Fourier fields")
        radial, b = _padded_spline(field.r_knots, field.b_coef)
        flux, profiles = _padded_spline(field.s_knots, field.profile_coef)
        arrays = (radial, b, flux, profiles, jnp.pad(field.xm, (0, b.shape[1]-len(field.xm))),
                  jnp.pad(field.xn, (0, b.shape[1]-len(field.xn))))
    return arrays, (nc, nq) if coil else data.tables if vmec else None


class ClassifierStop:
    """Fixed Cartesian SurfaceClassifier level, with direct GPU table reads.

    Positive inside the classifier's outward offset. Both execution paths use
    the same linear interpolant and fill value; event decisions remain frozen
    under accepted-map AD. The classifier and its storage are unchanged.
    """
    def __init__(self, boundary, margin=0.):
        self.boundary, self.margin = boundary, margin
        self._data, self._level = self._kernel_view()

    def __call__(self, t, y, args=None, **kwargs):
        return jnp.maximum(0.,self.evaluate_xyz(y[:3])+self.margin)

    def evaluate_xyz(self, xyz):
        """Unshifted signed distance for the differentiable particle objectives."""
        return self._level(self._data,xyz,False)

    def _kernel_view(self):
        if hasattr(self,'_data'):return self._data,self._level
        table = self.boundary.dist
        axes = tuple(table.grid)
        shape = tuple(len(axis) for axis in axes)
        if (len(shape)!=3 or min(shape)<2 or table.method!='linear'
                or table.values.size!=int(np.prod(shape)) or table.fill_value is None):
            raise ValueError('ClassifierStop requires a scalar 3D linear classifier with a fill value')
        uniform = tuple(np.allclose(axis,np.linspace(axis[0],axis[-1],len(axis)),
            rtol=0.,atol=32*np.finfo(axis.dtype).eps*max(1.,float(np.max(np.abs(axis)))))
            for axis in map(np.asarray,axes))
        arrays = (*axes, table.values.reshape(-1), jnp.asarray(table.fill_value), jnp.asarray(self.margin))
        def level(refs, y, stopping=True):
            x, yy, z = (scalar(y,i) if stopping else y[i] for i in range(3))
            coordinates = jnp.sqrt(x*x+yy*yy), jnp.mod(jnp.arctan2(yy,x),2*jnp.pi), z
            indices, fractions = [], []
            outside = jnp.asarray(False)
            for axis, n, value, regular in zip(refs[:3], shape, coordinates, uniform):
                if regular:
                    index=jnp.clip(jnp.floor((value-axis[0])*(n-1)/(axis[n-1]-axis[0])).astype(jnp.int32),0,n-2)
                    # Match searchsorted(side='left'), including rounded grid knots.
                    index=jnp.where((index>0)&(value<=axis[index]),index-1,index)
                    index=jnp.where((index<n-2)&(value>axis[index+1]),index+1,index)
                else:
                    def search(_, bounds):
                        low, high = bounds
                        middle = (low+high)//2
                        below = axis[jnp.minimum(middle,n-1)] < value
                        return jnp.where(below,middle+1,low),jnp.where(below,high,middle)
                    insertion, _ = lax.fori_loop(0,n.bit_length(),search,(jnp.int32(0),jnp.int32(n)))
                    index = jnp.clip(insertion-1,0,n-2)
                indices.append(index)
                fractions.append((value-axis[index])/(axis[index+1]-axis[index]))
                outside |= (value<axis[0]) | (value>axis[n-1])
            result = jnp.asarray(0., y.dtype)
            for corner in range(8):
                index, weight = jnp.int32(0), jnp.asarray(1., y.dtype)
                for axis, (start, fraction, n) in enumerate(zip(indices,fractions,shape)):
                    high = (corner>>(2-axis))&1
                    index = index*n+start+high
                    weight *= fraction if high else 1-fraction
                result += weight*refs[3][jnp.clip(index,0,int(np.prod(shape))-1)]
            distance=jnp.where(outside,refs[4][()],result)
            return jnp.maximum(0.,distance+refs[5][()]) if stopping else distance
        return arrays, level


def _prepare_split_field(field, species, *, gc, collision_tolerance, speed_floor,
                         event=None, state_size=5, field_view=None,mass=M,charge=Q,spatial_diffusion=False):
    """Reuse the orbit PR's private field views, carrying physical mu per kick."""
    from .fields import BiotSavart, Vmec
    from .boozer import guiding_center_rhs
    coil, vmec = isinstance(field, BiotSavart), isinstance(field, Vmec)
    if spatial_diffusion and (not gc or not coil):
        raise ValueError('the native coordinate adapter does not yet transform GC spatial diffusion')
    arrays, metadata = _split_field_data(field) if field_view is None else field_view
    if coil:
        nc, nq = metadata
    nfield = len(arrays)
    cg = _coefficient_graph(species, not gc,mass,charge)
    # Keep the per-particle floor as explicit data, not a captured map constant.
    def gc_map(v, xi, h, dw, floor, *c):
        out = guiding_center_collision_step(v, xi, h, dw, CollisionCoefficients(*c),
            tolerance=collision_tolerance, speed_floor=floor)
        return out.y, out.drift_error, out.diffusion_error
    gg = _flatten(gc_map, jnp.asarray(4e5), jnp.asarray(.5), jnp.asarray(1e-7),
                   jnp.zeros(2),jnp.asarray(0.), *(jnp.asarray(1.),)*6) if gc else None
    functions = tree_util.tree_leaves(event)
    tables = [function._kernel_view() for function in functions if isinstance(function,ClassifierStop)]
    functions = [function for function in functions if not isinstance(function,ClassifierStop)]
    eg = _flatten(lambda y: jnp.min(jnp.asarray([f(0., y, None) for f in functions])),
                  jnp.zeros(state_size)) if functions else None
    event_arrays = []
    if eg is not None:
        for value in eg.consts:
            if value.ndim > 1:
                raise ValueError("independent stopping surfaces require scalar/vector constants")
            event_arrays.append(value if value.ndim == 0 else
                jnp.pad(value, (0, (1 << (value.size-1).bit_length())-value.size)))
    all_arrays = (tuple(arrays)+tuple(cg.consts)+(tuple(gg.consts)+(jnp.asarray(speed_floor),) if gc else ())+tuple(event_arrays)
        +tuple(value for values,_ in tables for value in values))

    def factory(refs, y0):
        frefs = refs[:nfield]
        ci = nfield+len(cg.consts)
        cc = [ref[...] for ref in refs[nfield:ci]]
        mc = [ref[...] for ref in refs[ci:ci+len(gg.consts)+1]] if gc else ()
        event_start = ci+(len(gg.consts)+1 if gc else 0)
        erefs = refs[event_start:event_start+len(event_arrays)]
        ec = ([ref[...] if value.ndim == 0 else tuple(scalar(ref[...], i) for i in range(value.size))
               for ref, value in zip(erefs, eg.consts)] if eg is not None else ())
        zero = scalar(y0, 7)
        def orbit_state(y):
            return pack(tuple(scalar(y, i) for i in (0, 1, 2, 3, 5))+(zero, zero, zero))
        if coil:
            def rhs(y):
                value = _coil_rhs_values(frefs, y, nc, nq, "gc" if gc else "lorentz", 0., mass, charge,scalar(y,4) if gc else None)
                return pack(tuple(scalar(value, i) for i in range(4))+(zero, zero, zero, zero)) if gc else value
            def magnitude(y):
                B, _ = _coil_field_values(frefs, tuple(scalar(y, i) for i in range(3)), nc, nq, False)
                return jnp.sqrt(_dot(B, B))
            def norm(y, e, speed, budget):
                pos = jnp.sqrt(sum(scalar(e, i)**2 for i in range(3)))
                vel = jnp.abs(scalar(e, 3)) if gc else jnp.sqrt(sum(scalar(e, i)**2 for i in range(3, 6)))
                return jnp.maximum(pos, vel/speed)/budget
        elif vmec:
            def rhs(y):
                value = _vmec_rhs_values(frefs, orbit_state(y), metadata, mass, charge,0.,scalar(y,4))
                return pack(tuple(scalar(value, i) for i in range(4))+(zero, scalar(value, 4), zero, zero))
            magnitude = lambda y: _vmec_series(frefs, _vmec_chart(orbit_state(y)), metadata['bmnc'])[0]
            def norm(y, e, speed, budget):
                a = _vmec_xyz_values(frefs, orbit_state(y), metadata)
                b = _vmec_xyz_values(frefs, orbit_state(y-e), metadata)
                return jnp.maximum(jnp.sqrt(sum((u-v)**2 for u, v in zip(a, b))), jnp.abs(scalar(e, 3))/speed)/budget
        else:
            model = _KernelField(frefs, float(field.psi0))
            def physical(y):
                value = guiding_center_rhs(model, y, y[4]/mass, mass, charge)
                return pack(tuple(value[i] for i in range(4))+(zero, zero, zero, zero))
            graph = _flatten(physical, y0)
            rhs = lambda y: _evaluate_kernel(graph.jaxpr, graph.consts, y)[0]
            bg = _flatten(lambda y: model.modB_derivatives(jnp.sqrt(y[0]**2+y[1]**2),
                          jnp.arctan2(y[1], y[0]), y[2])[0], y0)
            magnitude = lambda y: _evaluate_kernel(bg.jaxpr, bg.consts, y)[0]
            def norm(y, e, speed, budget):
                return jnp.maximum(jnp.sqrt(scalar(e, 0)**2+scalar(e, 1)**2),
                    jnp.maximum(jnp.abs(scalar(e, 2)), jnp.abs(scalar(e, 3))/speed))/budget
        table_levels = []
        start = event_start+len(event_arrays)
        for values, evaluate in tables:
            table_levels.append(lambda y, evaluate=evaluate, data=refs[start:start+len(values)]:evaluate(data,y))
            start += len(values)
        def level(y):
            result = _evaluate_surface(eg.jaxpr,ec,y) if eg is not None else zero+jnp.inf
            for function in table_levels:
                result = jnp.minimum(result,function(y))
            return result if eg is not None or table_levels else zero+1.
        def kick(y, h, dw):
            if gc and spatial_diffusion:
                vector,_=_coil_field_values(frefs,tuple(scalar(y,i) for i in range(3)),nc,nq,False)
                B=jnp.sqrt(_dot(vector,vector))
            else:
                B = magnitude(y) if gc else zero+1.
            speed = jnp.sqrt(scalar(y, 3)**2+2*scalar(y, 4)*B/mass) if gc else jnp.sqrt(sum(scalar(y, i)**2 for i in range(3, 6)))
            c = _evaluate_kernel(cg.jaxpr, cc, speed, scalarize=True)
            if gc:
                xi=scalar(y,3)/speed
                velocity_noise=pack(tuple(scalar(dw,i) for i in (3,4))) if spatial_diffusion else dw
                changed, ed, es = _evaluate_kernel(gg.jaxpr, mc[:-1], speed, xi, h, velocity_noise,mc[-1], *c, scalarize=True)
                v, xi = scalar(changed, 0), scalar(changed, 1)
                if spatial_diffusion:
                    unit=tuple(b/B for b in vector);noise=tuple(scalar(dw,i) for i in range(3))
                    scale=_diffusion_root(2*guiding_center_spatial_diffusion(scalar(y,3)/speed,CollisionCoefficients(*c),charge*B/mass))
                    parallel=_dot(unit,noise)
                    point=tuple(scalar(y,i)+scale*(noise[i]-unit[i]*parallel) for i in range(3))
                    y=pack(point+tuple(scalar(y,i) for i in range(3,8)))
                    B=magnitude(y)
                result = jnp.where(jnp.arange(8) == 3, v*xi,
                    jnp.where(jnp.arange(8) == 4, mass*v*v*(1-xi*xi)/(2*B), y))
            else:
                # Particle map (57)-(62), scalarized in the velocity-aligned basis.
                dp, dt, ddp, q, _, dk = c
                drift = q+ddp+2*(dp-dt)/speed
                delta = (_diffusion_root(2*dt)*scalar(dw, 0), _diffusion_root(2*dt)*scalar(dw, 1),
                         drift*h+_diffusion_root(2*dp)*scalar(dw, 2)+.5*ddp*(scalar(dw, 2)**2-h))
                direction = tuple(scalar(y, i)/speed for i in range(3, 6))
                a = tuple(jnp.abs(v) for v in direction)
                i0 = (a[0] <= a[1]) & (a[0] <= a[2]); i1 = ~i0 & (a[1] <= a[2])
                ref = tuple(v.astype(y.dtype) for v in (i0, i1, ~i0 & ~i1))
                first = _cross(direction, ref); length = jnp.sqrt(_dot(first, first))
                first = tuple(v/length for v in first); second = _cross(direction, first)
                velocity = tuple(scalar(y, 3+i)+first[i]*delta[0]+second[i]*delta[1]+direction[i]*delta[2] for i in range(3))
                result = pack(tuple(scalar(y, i) for i in range(3))+velocity+(zero, zero))
                eps = collision_tolerance*(jnp.abs(drift)*h+jnp.sqrt(jnp.maximum(2*dp*h, 0.)))
                def ratio(numerator, denominator):
                    safe = jnp.where(denominator > 0, denominator, zero+1.)
                    return jnp.where(denominator > 0, numerator/safe,
                                     jnp.where(numerator == 0, zero, zero+jnp.inf))
                ed = ratio(.5*jnp.abs(drift*dk)*h*h, eps)
                es = ratio(ddp**2*jnp.abs(scalar(dw, 2))**3, 6*eps*jnp.sqrt(jnp.maximum(dp, 0.)))
            valid = _kernel_is_finite(B) & (B > 0) & (speed > 0) & (c[0] >= 0) & (c[1] >= 0)
            for value in c:
                valid &= _kernel_is_finite(value) if isinstance(value, jax.core.Tracer) else np.isfinite(value)
            return jnp.where(valid, result, jnp.nan), ed, es
        return rhs, kick, norm, level
    return all_arrays, factory


def _solve_independent(rhs, collision, initial, args=None, *, field, species, gc,
                       birth_speed, speed_floor, collision_tolerance, field_view=None, physics=None,
                       mass=M,charge=Q,spatial_diffusion=False,spatial_geometry=None,time_dependent=False,**options):
    from .fields import BiotSavart,Vmec
    for name in ('max_steps', 'max_orbit_steps', 'max_pending'):
        value = options[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(name+" must be a positive integer")
    nstate = initial.size
    screening=getattr(species,'screening_species',None)
    fast=not time_dependent and field_view is not None and species.radial_grid is None and (screening is None or screening.radial_grid is None)
    if spatial_diffusion and field_view is not None and not isinstance(field,BiotSavart):fast=False
    if physics is None or fast:
        arrays, factory = _prepare_split_field(field, species, gc=gc,
            collision_tolerance=collision_tolerance, speed_floor=speed_floor,
            event=options['event'], state_size=nstate, field_view=field_view,
            mass=mass,charge=charge,spatial_diffusion=spatial_diffusion)
        if physics is not None:
            predictor_arrays,predictor_factory=prepare_physics(field,field_view,physics,initial,options['noise_dim'],predictor_only=True)
            nfield=len(field_view[0]);nphysics=len(arrays);original_factory=factory
            arrays=tuple(arrays)+tuple(predictor_arrays[nfield:])
            def factory(refs,y0):
                physical=original_factory(refs[:nphysics],y0)
                prediction=predictor_factory(refs[:nfield]+refs[nphysics:],y0)[4]
                return (*physical,prediction)
    else:
        arrays,factory=prepare_physics(field,field_view,physics,initial,options['noise_dim'],spatial_geometry=spatial_geometry,time_dependent=time_dependent)
    def forward(initial, parameters=None, *, record=False):
        solve = _prepare_split_kernel(arrays, factory, options['ts'], adaptive=options['adaptive'],
            dt0=options['dt0'], dtmax=options['dtmax'], speed=birth_speed,
            tolerance=(options['rtol']+options['atol'])/2, noise_dim=options['noise_dim'],
            max_steps=options['max_steps'], max_orbit_steps=options['max_orbit_steps'],
            max_pending=options['max_pending'], state_size=nstate,
            bank_components=4 if gc and nstate == 5 else 8,
            orbit_dt0=options['orbit_dt0'], final_time=options['t1'],align_orbit=options.get('align_orbit',False),
            warps=4 if isinstance(field,Vmec) and not spatial_diffusion else 1,
            record=record,adjoint_checkpoint_steps=options.get('adjoint_checkpoint_steps',16),
            adjoint_orbit_checkpoint_steps=options.get('adjoint_orbit_checkpoint_steps',16),
            adjoint_max_accepted_steps=options.get('adjoint_max_accepted_steps'),
            adjoint_max_orbit_steps=options.get('adjoint_max_orbit_steps'),
            collision_indices=options.get('_collision_state_indices'),time_dependent=time_dependent)
        result = solve(jnp.pad(initial, (0, 8-nstate)), jax.random.key_data(options['key']))
        if record:
            result,tape=result
            tape=tape._replace(states=tape.states[:,:nstate],orbit_states=tape.orbit_states[:,:nstate],
                orbit_derivatives=tape.orbit_derivatives[:,:nstate],
                orbit_endpoints=tape.orbit_endpoints[:,:nstate] if options.get('_collision_state_indices') is None else tape.orbit_endpoints)
        ys,y,t,stats,status=result
        solution = SplitSolution(options['ts'], ys[:, :nstate], t, y[:nstate], status,
            SplitStats(*(stats[i].astype(jnp.int32) for i in range(6)), stats[6],collision_updates=stats[1].astype(jnp.int32),
                accepted_orbit_steps=stats[7].astype(jnp.int32),max_orbit_steps_per_collision=stats[8].astype(jnp.int32)),
            jnp.empty((0,), initial.dtype))
        return (solution,tape) if record else solution
    if options.get('adjoint','none') == 'accepted':
        from .native_ad import solve_accepted_split, solve_gpu_accepted
        direct=(isinstance(field,BiotSavart) and fast and mass==M and charge==Q)
        if options.get('_reverse_execution','auto') not in ('auto','shared','independent'):
            raise ValueError("reverse_execution must be 'auto', 'shared' or 'independent'")
        if options.get('_reverse_execution','auto')=='independent' and not direct:
            raise ValueError('independent reverse currently requires direct alpha coils and uniform background')
        if direct and options.get('_reverse_execution','auto')!='shared':
            return solve_gpu_accepted(forward,initial,field,species,gc=gc,speed_floor=speed_floor,
                options=options,arrays=arrays,spatial_diffusion=spatial_diffusion)
        ad_options={k:v for k,v in options.items() if k not in ('adjoint','_record_ad','_reverse_execution')}
        return solve_accepted_split(rhs,collision,initial,args,_forward_override=forward,**ad_options)
    return forward(initial,record=options.get('_record_ad',False))


# Spatial geometry

def move_in_cartesian(state, displacement, position, *, implicit=False):
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
    if not implicit:
        point,error,_=jax.lax.while_loop(lambda c:((c[2]==0)|(c[1]>threshold))&(c[2]<8),advance,
            (state[:3],jnp.linalg.norm(displacement),jnp.asarray(0)))
        return state.at[:3].set(jnp.where(error<=threshold,point,jnp.full_like(point,jnp.nan)))
    def solve_root(function, guess):
        del function
        return jax.lax.while_loop(lambda c:((c[2]==0)|(c[1]>threshold))&(c[2]<8),advance,
            (guess,jnp.linalg.norm(displacement),jnp.asarray(0)))[0]
    def tangent_solve(linear, value):
        a,b,c=jax.vmap(linear)(jnp.eye(3,dtype=state.dtype))
        rows=jnp.stack((jnp.cross(b,c),jnp.cross(c,a),jnp.cross(a,b)))
        return (rows@value)/jnp.dot(a,rows[0])
    point=jax.lax.custom_root(lambda q:mapping(q)-target,state[:3],solve_root,tangent_solve)
    matrix=jax.jacfwd(mapping)(point)
    scale=jnp.maximum(1.,jnp.maximum(jnp.linalg.norm(target),
        jnp.linalg.norm(matrix)*jnp.maximum(1.,jnp.linalg.norm(point))))
    error=jnp.linalg.norm(mapping(point)-target)/scale
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

class MomentDefect(NamedTuple):
    mean: object
    covariance: object
    cubic_covariance: object
    noise: object
    orbit_jacobian: object

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

def moment_defect(f,vel,jet,*,collision_remainder=True,orbit_jet=None):
    """E[split-exact] = mean*h^2/2, delta Cov = covariance*h^2/2.

    Splitting terms are [L_orbit,L_collision] on linear/quadratic tests.
    The optional collision remainder matches the implemented diagonal
    Milstein map, not an exact collision semigroup. The cubic spatial block
    is the missing integrated velocity noise, exact for free Brownian flight.
    Higher-order terms for a general nonlinear field are not included.
    """
    a,g,aj,ah,qj,qh,xa,xq=jet;q=g*g
    if orbit_jet is None:
        f0=f(vel);fj=jax.jacfwd(f)(vel)
        trace=jnp.einsum('ijj,j->i',jax.jacfwd(jax.jacfwd(f))(vel),q)
    else:
        f0,fj,diagonal=orbit_jet;trace=diagonal@q
    n=f0.size
    J=jnp.zeros((n,n)).at[:,-2:].set(fj)
    G=jnp.zeros((n,2)).at[-2:,:].set(jnp.diag(g));Q=G@G.T
    mean=-fj@a-.5*trace
    mean=mean.at[-2:].add(aj@f0[-2:]+xa)
    cov=jnp.zeros((n,n)).at[-2:,-2:].set(jnp.diag(qj@f0[-2:]+xq))
    cov=cov-J@Q-Q@J.T
    if collision_remainder:
        la=aj@a+.5*jnp.einsum('ijj,j->i',ah,q)
        lq=qj@a+.5*jnp.einsum('ijj,j->i',qh,q)
        mean=mean.at[-2:].add(-la)
        correction=jnp.diag(.25*jnp.diag(qj)**2-lq)-aj@jnp.diag(q)-jnp.diag(q)@aj.T
        cov=cov.at[-2:,-2:].add(correction)
    B=(J@G).at[-2:,:].set(0.)
    return MomentDefect(mean,cov,-B@B.T/3,G,J)

def budget_components(defect,vel,tau):
    """Mean/covariance error rates over tau using linear orbit transport.

    Averaging I+sJ over 0<=s<=tau propagates local mixed defects into
    later position variance. This captures the leading global Brownian-
    flight variance bias which a diagonal local RMS error omits.
    """
    b,C,D,G,J=defect
    b=b+.5*tau*(J@b)
    def transport(A):return A+.5*tau*(J@A+A@J.T)+(tau*tau/3)*(J@A@J.T)
    C,D=transport(C),transport(D)
    B=J@G
    # PSD Gramian diagonal: integral_0^tau |G+s JG|^2 ds.
    # Add physical parallel velocity p=r*xi as an observable, retaining its
    # Ito mean correction from the mixed covariance defect.
    n=b.size;P=jnp.concatenate((jnp.eye(n),jnp.zeros((1,n)).at[0,-2:].set(vel[::-1])),axis=0)
    bp=P@b;bp=bp.at[-1].add(C[-2,-1])
    Cp,Dp=P@C@P.T,P@D@P.T
    Gp,Bp=P@G,P@B
    spreadp=tau*jnp.sum((Gp+.5*tau*Bp)**2,axis=1)+tau**3/12*jnp.sum(Bp*Bp,axis=1)
    scale=jnp.sqrt(jnp.maximum(spreadp,1e-28))
    return (.5*tau*jnp.abs(bp)/scale,
        .5*tau*jnp.abs(Cp)/(scale[:,None]*scale[None,:]),
        tau*jnp.abs(Dp)/(scale[:,None]*scale[None,:]))

def transported_budget(defect,vel,tau):
    """Reduce the individual mean/covariance limits without extra evaluations."""
    means,covariance,quadratic=budget_components(defect,vel,tau)
    return jnp.maximum(jnp.max(means),jnp.max(covariance)),jnp.max(quadratic)

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

def make_step_limit(field,bath,physical,tolerance=.005,*,reference_time=.005,cap=.0025,expanded=True,projection=None):
    """Predict intervals from a fractional moment-defect budget.

    reference_time is in t*nu0 units, NOT a timestep cap. It sets the earliest
    observation timescale; elapsed time subsequently grows the spread proxy.
    Tolerance differs from the old local-trajectory RMS tolerance. Independent
    physical/paired tests, rather than this proxy, determine achieved error.
    expanded=False retains the dense algebra reference; both use the same
    native coefficient and field derivatives.
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
        if expanded:
            linear,quadratic=expanded_moment_budget(velocity,jet,vel,tau)
        else:
            defect=moment_defect(None,vel,jet,orbit_jet=velocity)
            linear,quadratic=transported_budget(defect,vel,tau)
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
                     rate_first,rate_second,profile_rate,tau,*,details=False,orbit=True):
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
    return (linear,quadratic,mean,covariance) if details else (linear,quadratic)

def native_step_limit(field,bath,mass,charge,initial,rhs,public,position,magnitude,
                      *,gc,coordinate=False,boozer=False,time_dependent=False,
                      electric_field,tmax,dtmax,tolerance=.005,reference_time=.005,orbit=True,reference_magnitude=None):
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
    initial_magnitude=magnitude if reference_magnitude is None else reference_magnitude
    speed=jnp.sqrt(birth[3]**2+2*birth[4]*initial_magnitude(point)/mass) if gc else jnp.linalg.norm(birth[3:])
    temperature=jnp.sum(density*temperatures)/jnp.where(jnp.sum(density)>0,jnp.sum(density),1.)
    sigma=jnp.where(jnp.sum(density)>0,jnp.sqrt(ELEMENTARY_CHARGE*temperature/mass),speed/jnp.sqrt(10.))
    friction=collision_coefficients(bath,mass,charge,sigma,point).friction
    nu=jnp.where(friction==0,1/jnp.maximum(tmax,1e-30),-friction/sigma)
    physical=SimpleNamespace(sigma=sigma,nu=nu,mass=mass,charge=charge)
    cap=dtmax*nu;zero_electric=isinstance(electric_field,Electric_field_zero) and not time_dependent
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
        def velocity(v,template,t):
            B=magnitude(public(template)[:3]);r,xi=v
            y=template.at[3].set(sigma*r*xi).at[4].set(mass*sigma*sigma*r*r*(1-xi*xi)/(2*B))
            dy=rhs(t,y,None)
            dx=jax.jvp(position,(y,),(dy,))[1]/(nu if boozer else sigma)
            dr=0. if zero_electric else jax.jvp(lambda y:phase(y)[3],(y,),(dy,))[1]/nu
            return jnp.r_[dx,dr,dy[3]/(sigma*nu*r)-xi*dr/r]
        def limit(t,y,args):
            vel=phase(y)[-2:]
            f=lambda v:velocity(v,y,t)
            if hasattr(field,'_velocity_jet'):
                velocity_jet=field._velocity_jet(f,vel)
            else:
                f0=f(vel);F=jax.jacfwd(f)(vel);H=jax.jacfwd(jax.jacfwd(f))(vel)
                velocity_jet=(f0,F,jnp.stack((H[:,0,0],H[:,1,1]),axis=1))
            jet=collision_jet(radial,*vel,y,rhs(t,y,args)/nu)
            linear,quadratic=expanded_moment_budget(velocity_jet,jet,vel,jnp.maximum(reference_time,t*nu))
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


# GPU predictor lowering

class KernelCalls:
    """Keep native field reductions intact while scalarizing moment algebra."""
    def __init__(self):self.handlers={}

    def first_jet(self,function,kernel_function,*,outputs=3,dimensions=3):
        """Share one exact value/Jacobian evaluation across JVP directions."""
        def ordinary(*point):
            jacobian=jax.jacfwd(lambda x:jnp.asarray(function(*x)))(jnp.asarray(point))
            return (*function(*point),*jacobian.reshape(-1))
        packed=self.add(ordinary,kernel_function,outputs*(dimensions+1))
        @jax.custom_jvp
        def value(*point):return packed(*point)[:outputs]
        @value.defjvp
        def directional(primals,tangents):
            data=packed(*primals)
            return data[:outputs],tuple(sum(data[outputs+dimensions*i+j]*tangents[j]
                for j in range(dimensions)) for i in range(outputs))
        return value

    def add(self, function, kernel_function, count,*,kernel_jvp=None):
        from jax.extend.core import Primitive
        from jax._src.interpreters import ad, batching
        primitive=Primitive('essos_split_field_call')
        primitive.multiple_results=True
        primitive.def_impl(function)
        primitive.def_abstract_eval(lambda *avals:tuple(jax.core.ShapedArray((),avals[0].dtype) for _ in range(count)))
        self.handlers[primitive]=kernel_function
        derivatives={}
        def jvp(primals,tangents):
            active=tuple(i for i,t in enumerate(tangents) if not isinstance(t,ad.Zero))
            result=primitive.bind(*primals)
            if not active:return result,tuple(ad.Zero.from_primal_value(x) for x in result)
            if active not in derivatives:
                size=len(primals)
                def directional(fn):
                    def evaluate(*args):
                        def restricted(*values):
                            primals=list(args[:size])
                            for i,v in zip(active,values):primals[i]=v
                            return fn(*primals)
                        return jax.jvp(restricted,tuple(args[i] for i in active),args[size:])[1]
                    return evaluate
                def kernel_direction(refs,*args):
                    if kernel_jvp is not None:
                        directions=dict(zip(active,args[size:]))
                        return kernel_jvp(refs,args[:size],tuple(directions.get(i,jnp.zeros_like(args[i])) for i in range(size)))
                    graph=_flatten(directional(lambda *x:kernel_function(refs,*x)),*args)
                    return _evaluate_kernel(graph.jaxpr,graph.consts,*args)
                derivatives[active]=self.add(directional(function),kernel_direction,count)
            return result,derivatives[active](*primals,*(tangents[i] for i in active))
        ad.primitive_jvps[primitive]=jvp
        def batch(values,axes):
            size=next(x.shape[axis] for x,axis in zip(values,axes) if axis is not None)
            rows=[primitive.bind(*(x if axis is None else jnp.take(x,i,axis=axis) for x,axis in zip(values,axes))) for i in range(size)]
            return tuple(jnp.stack(column) for column in zip(*rows)),(0,)*count
        batching.primitive_batchers[primitive]=batch
        return lambda *args:tuple(primitive.bind(*args))

def _vmec_position_tangent(refs,point,direction,tables):
    """Exact first derivative already returned by the native Fourier series."""
    from .rk import _vmec_series
    R,dR=_vmec_series(refs,point,tables['rmnc'])
    _,dZ=_vmec_series(refs,point,tables['zmns'])
    radial=sum(a*b for a,b in zip(dR,direction));vertical=sum(a*b for a,b in zip(dZ,direction))
    c,s=jnp.cos(point[2]),jnp.sin(point[2])
    return radial*c-R*s*direction[2],radial*s+R*c*direction[2],vertical

class _KernelTable:
    """Lazy constant-table views: load the requested cell, not every row."""
    def __init__(self, ref, shape, indices=None):
        self.ref=ref
        self.indices=np.arange(np.prod(shape),dtype=np.int64).reshape(shape) if indices is None else indices
        self.shape,self.ndim=self.indices.shape,self.indices.ndim
        self.offset=int(self.indices.flat[0])
        self.strides=tuple(int(np.take(self.indices,1,axis=i).flat[0])-self.offset if n>1 else 0
                           for i,n in enumerate(self.shape))
    def __array__(self,dtype=None,copy=None):
        values=np.empty(self.shape,object)
        for i in np.ndindex(self.shape):values[i]=self.ref[int(self.indices[i])]
        return values if dtype is None else values.astype(dtype)
    def _view(self,indices):
        view=type(self)(self.ref,indices.shape,indices)
        expected=np.full(indices.shape,view.offset,np.int64)
        for i,(n,stride) in enumerate(zip(indices.shape,view.strides)):
            shape=[1]*indices.ndim;shape[i]=n
            expected+=np.arange(n).reshape(shape)*stride
        return view if np.array_equal(indices,expected) else np.asarray(view)
    def __getitem__(self,index):
        index=index if isinstance(index,tuple) else (index,)
        if len(index)==self.ndim and all(isinstance(i,(int,np.integer,jax.core.Tracer,jax.Array)) for i in index):
            return self.ref[self.offset+sum(i*s for i,s in zip(index,self.strides))]
        return self._view(self.indices[index])
    def reshape(self,*args,**kwargs):return self._view(self.indices.reshape(*args,**kwargs))
    def transpose(self,*args):return self._view(self.indices.transpose(*args))
    def squeeze(self,axis=None):return self._view(self.indices.squeeze(axis))


def scalar_graph(graph, constants, *arguments, calls=None, refs=(),memo=None,defer=False):
    memo={} if memo is None else memo
    def word_key(value):
        if isinstance(value,(jax.core.Tracer,jax.Array)):
            memo[('retain',id(value))]=value
            return ('value',id(value))
        value=np.asarray(value)
        return ('literal',value.dtype.str,value.shape,value.tobytes())
    def logical(value, shape):
        if isinstance(value,_KernelTable):
            if shape:return value.reshape(shape)
            value=value.reshape(shape)[()]
        if isinstance(value, tuple):
            out=np.empty((len(value),),object)
            for i,v in enumerate(value):out[i]=v
            return out.reshape(shape)
        if isinstance(value,(np.ndarray,np.generic,int,float,bool)):
            return np.asarray(value).reshape(shape)
        out=np.empty(shape,object)
        for index in np.ndindex(shape):
            item=value
            for axis,i in enumerate(index):item=_select_one(item,i,axis)
            out[index]=item if not item.shape else item.reshape(())
        return out

    env={v:logical(x,v.aval.shape) for v,x in zip(graph.constvars,constants)}
    env.update({v:logical(x,v.aval.shape) for v,x in zip(graph.invars,arguments)})
    read=lambda v:np.asarray(v.val,dtype=v.aval.dtype) if isinstance(v,Literal) else env[v]
    def elements(function,*values):
        return np.frompyfunc(function,len(values),1)(*values)
    def scalar(value):
        while isinstance(value,np.ndarray):value=value.reshape(()).item()
        return value
    def device_scalar(value,dtype):
        while isinstance(value,np.ndarray) and not value.shape:value=value.item()
        return jnp.asarray(value,dtype=dtype)
    def reduced(value,axes,combine):
        result=value
        for axis in sorted(axes,reverse=True):
            result=np.frompyfunc(combine,2,1).reduce(result,axis=axis)
        return result

    for equation in graph.eqns:
        values=[read(v) for v in equation.invars];p=equation.params;name=equation.primitive.name
        folded={'add':np.add,'sub':np.subtract,'mul':np.multiply,'div':np.divide,
            'neg':np.negative,'abs':np.abs,'sqrt':np.sqrt,'eq':np.equal,'ne':np.not_equal,
            'lt':np.less,'le':np.less_equal,'gt':np.greater,'ge':np.greater_equal,
            'max':np.maximum,'min':np.minimum,'and':np.bitwise_and,'or':np.bitwise_or}
        static=all(not isinstance(v,_KernelTable) and np.asarray(v).dtype!=object for v in values)
        if calls is not None and equation.primitive in calls.handlers:
            words=tuple(scalar(v) for v in values)
            key=('field',equation.primitive,tuple(word_key(v) for v in words))
            if key not in memo:
                arguments=tuple(device_scalar(v,var.aval.dtype) for v,var in zip(words,equation.invars))
                if defer:
                    memo[key]=equation.primitive.bind(*arguments,**equation.params)
                else:
                    memo[key]=calls.handlers[equation.primitive](refs,*arguments)
            result=memo[key]
        elif static and name in folded:result=[folded[name](*values)]
        elif static and name=='is_finite':result=[np.isfinite(values[0])]
        elif static and name=='convert_element_type':result=[values[0].astype(p['new_dtype'])]
        elif static and name=='integer_pow':result=[np.power(values[0],p['y'])]
        elif static and name=='select_n':result=[np.choose(values[0],values[1:])]
        elif name=='platform_index':
            backend='cuda' if jax.default_backend()=='gpu' else jax.default_backend()
            default=next((i for i,group in enumerate(p['platforms']) if group is None),0)
            result=[np.asarray(next((i for i,group in enumerate(p['platforms']) if group is not None and backend in group),default),np.int32)]
        elif name=='cond':
            branch=p['branches'][int(scalar(values[0]))]
            result=list(scalar_graph(branch.jaxpr,branch.consts,*values[1:],calls=calls,refs=refs,memo=memo,defer=defer))
        elif name in ('custom_jvp_call','custom_vjp_call','closed_call','jit'):
            nested=p.get('call_jaxpr',p.get('fun_jaxpr',p.get('jaxpr')))
            body=getattr(nested,'jaxpr',nested)
            result=list(scalar_graph(body,getattr(nested,'consts',()),*values,calls=calls,refs=refs,memo=memo,defer=defer))
        elif name=='rev':result=[np.flip(values[0],axis=p['dimensions'])]
        elif name in ('reshape','squeeze'):
            value=values[0]
            if name=='reshape' and p.get('dimensions') is not None:value=np.transpose(value,p['dimensions'])
            result=[np.reshape(value,equation.outvars[0].aval.shape)]
        elif name=='transpose':result=[np.transpose(values[0],p['permutation'])]
        elif name=='broadcast_in_dim':
            shape=[1]*len(equation.outvars[0].aval.shape)
            for i,axis in enumerate(p['broadcast_dimensions']):shape[axis]=values[0].shape[i]
            result=[np.broadcast_to(np.reshape(values[0],shape),equation.outvars[0].aval.shape)]
        elif name=='iota':
            shape=[1]*len(p['shape']);shape[p['dimension']]=p['shape'][p['dimension']]
            result=[np.broadcast_to(np.arange(shape[p['dimension']],dtype=p['dtype']).reshape(shape),p['shape'])]
        elif name=='slice':
            strides=p.get('strides') or (1,)*len(p['start_indices'])
            result=[values[0][tuple(slice(a,b,c) for a,b,c in zip(p['start_indices'],p['limit_indices'],strides))]]
        elif name=='dynamic_slice':
            operand=values[0];out=np.empty(p['slice_sizes'],object)
            starts=[jnp.clip(scalar(v),0,n-k) for v,n,k in zip(values[1:],operand.shape,p['slice_sizes'])]
            for index in np.ndindex(out.shape):
                if isinstance(operand,_KernelTable):
                    out[index]=operand[tuple(start+i for start,i in zip(starts,index))]
                    continue
                value=operand.flat[0]
                for source in np.ndindex(operand.shape):
                    match=jnp.asarray(True)
                    for start,i,j in zip(starts,index,source):match &= start+i==j
                    value=jnp.where(match,operand[source],value)
                out[index]=value
            result=[out]
        elif name=='while':
            nc,nb=p['cond_nconsts'],p['body_nconsts']
            condition,body=p['cond_jaxpr'],p['body_jaxpr']
            variables=body.jaxpr.invars[nb:]
            sizes=[v.aval.size for v in variables];offsets=np.cumsum([0]+sizes)
            def unpack(words):return [logical(tuple(words[a:b]),v.aval.shape)
                for a,b,v in zip(offsets[:-1],offsets[1:],variables)]
            def pack(items):return tuple(device_scalar(word,v.aval.dtype)
                for item,v in zip(items,variables) for word in np.asarray(logical(item,v.aval.shape),object).flat)
            def test(words):
                result=scalar_graph(condition.jaxpr,condition.consts,*values[:nc],*unpack(words),calls=calls,refs=refs,defer=defer)
                return device_scalar(result[0],jnp.bool_)
            def advance(words):
                result=scalar_graph(body.jaxpr,body.consts,*values[nc:nc+nb],*unpack(words),calls=calls,refs=refs,defer=defer)
                return pack(result)
            result=unpack(jax.lax.while_loop(test,advance,pack(values[nc+nb:])))
        elif name=='scan':
            body=p['jaxpr']
            if 'num_consts' in p:count,carry_count=p['num_consts'],p['num_carry']
            else:count,carry_count=(len(group) for group in p['ft_in'].unpack()[:2])
            fixed=values[:count];carry=values[count:count+carry_count];outputs=[]
            order=range(p['length']-1,-1,-1) if p['reverse'] else range(p['length'])
            for index in order:
                row=scalar_graph(body.jaxpr,body.consts,*fixed,*carry,*(v[index] for v in values[count+carry_count:]),calls=calls,refs=refs,memo=memo,defer=defer)
                carry=list(row[:carry_count]);outputs.append(row[carry_count:])
            if p['reverse']:outputs.reverse()
            result=carry+[np.stack([logical(value,var.aval.shape) for value in column])
                for var,column in zip(body.jaxpr.outvars[carry_count:],zip(*outputs))]
        elif name in ('concatenate','stack'):
            result=[(np.concatenate if name=='concatenate' else np.stack)(values,axis=p.get('dimension',p.get('axis')))]
        elif name=='unstack':result=[np.take(values[0],i,axis=p['axis']) for i in range(len(equation.outvars))]
        elif name=='split':
            result=list(np.split(values[0],np.cumsum(p['sizes'])[:-1],axis=p['axis']))
        elif name.startswith('reduce_'):
            combine={'reduce_sum':lambda a,b:a+b,'reduce_prod':lambda a,b:a*b,
                'reduce_max':jnp.maximum,'reduce_min':jnp.minimum,
                'reduce_or':lambda a,b:a|b,'reduce_and':lambda a,b:a&b}[name]
            result=[reduced(values[0],p['axes'],combine)]
        elif name in ('argmin','argmax'):
            axis=p.get('axes',(p.get('axis',0),))[0]
            value=np.moveaxis(values[0],axis,-1);out=np.empty(value.shape[:-1],object)
            for index in np.ndindex(out.shape):
                best=value[index+(0,)];chosen=jnp.asarray(0,p['index_dtype'])
                for i in range(1,value.shape[-1]):
                    item=value[index+(i,)];better=(item<best) if name=='argmin' else (item>best)
                    better=better|(jnp.isnan(item)&~jnp.isnan(best))
                    chosen=jnp.where(better,jnp.asarray(i,p['index_dtype']),chosen);best=jnp.where(better,item,best)
                out[index]=chosen
            result=[out]
        elif name=='dot_general':
            (lc,rc),(lb,rb)=p['dimension_numbers'];lhs,rhs=values
            lf=tuple(i for i in range(lhs.ndim) if i not in lc+lb)
            rf=tuple(i for i in range(rhs.ndim) if i not in rc+rb)
            batch=tuple(lhs.shape[i] for i in lb)
            left=tuple(lhs.shape[i] for i in lf);right=tuple(rhs.shape[i] for i in rf)
            inner=int(np.prod([lhs.shape[i] for i in lc]))
            lhs=np.transpose(lhs,lb+lf+lc).reshape(batch+(int(np.prod(left)),inner))
            rhs=np.transpose(rhs,rb+rc+rf).reshape(batch+(inner,int(np.prod(right))))
            result=[np.matmul(lhs,rhs).reshape(batch+left+right)]
        elif name=='pad':
            value=values[0];config=p['padding_config']
            shape=tuple(a+b+n+(n-1)*c for n,(a,b,c) in zip(value.shape,config))
            out=np.empty(shape,object);out.fill(scalar(values[1]))
            out[tuple(slice(a,a+n*(c+1),c+1) for n,(a,b,c) in zip(value.shape,config))]=value
            result=[out]
        elif name.startswith('scatter'):
            operand,indices,updates=values;dims=p['dimension_numbers']
            if indices.dtype==object or isinstance(operand,_KernelTable):
                raise ValueError('predictor lowering requires static scalar scatter indices')
            out=np.array(operand,object,copy=True)
            window_axes=[i for i in range(out.ndim) if i not in dims.inserted_window_dims]
            batch_axes=[i for i in range(updates.ndim) if i not in dims.update_window_dims]
            for index in np.ndindex(updates.shape):
                batch=tuple(index[i] for i in batch_axes);target=[0]*out.ndim
                for axis,item in zip(dims.scatter_dims_to_operand_dims,indices[batch]):target[axis]=int(item)
                for axis,i in zip(window_axes,dims.update_window_dims):target[axis]+=index[i]
                if any(i<0 or i>=n for i,n in zip(target,out.shape)):continue
                target=tuple(target);new=updates[index]
                out[target]=out[target]+new if name=='scatter-add' else new
            result=[out]
        elif name=='gather':
            operand,indices=values;dims=p['dimension_numbers']
            if indices.dtype==object:
                from itertools import product
                out=np.empty(equation.outvars[0].aval.shape,object)
                batch_axes=[i for i in range(out.ndim) if i not in dims.offset_dims]
                window_axes=[i for i in range(operand.ndim) if i not in dims.collapsed_slice_dims]
                for index in np.ndindex(out.shape):
                    batch=tuple(index[i] for i in batch_axes);selected=[0]*operand.ndim
                    for axis,item in zip(dims.start_index_map,indices[batch]):selected[axis]=jnp.clip(item,0,operand.shape[axis]-p['slice_sizes'][axis])
                    for axis,i in zip(window_axes,dims.offset_dims):selected[axis]+=index[i]
                    if isinstance(operand,_KernelTable):
                        out[index]=operand[tuple(selected)]
                        continue
                    choices=[range(n) if isinstance(i,jax.core.Tracer) else (int(i),) for i,n in zip(selected,operand.shape)]
                    value=operand[tuple(next(iter(c)) for c in choices)]
                    for source in product(*choices):
                        match=jnp.asarray(True)
                        for i,j in zip(selected,source):match &= i==j
                        value=jnp.where(match,operand[source],value)
                    out[index]=value
                result=[out]
                env.update((v,logical(x,v.aval.shape)) for v,x in zip(equation.outvars,result))
                continue
            pieces=[]
            for index in np.ndindex(indices.shape[:-1]):
                starts=[0]*operand.ndim
                for axis,item in zip(dims.start_index_map,indices[index]):starts[axis]=int(item)
                block=operand[tuple(slice(a,a+n) for a,n in zip(starts,p['slice_sizes']))]
                pieces.append(np.squeeze(block,axis=dims.collapsed_slice_dims))
            result=[np.asarray(pieces,dtype=object).reshape(equation.outvars[0].aval.shape)]
        else:
            parameters=dict(p)
            if name=='convert_element_type':parameters['weak_type']=False
            def bind(*xs):
                if all(not isinstance(x,jax.core.Tracer) for x in xs):
                    if name in folded:return folded[name](*xs)
                    if name=='integer_pow':return np.power(xs[0],p['y'])
                    if name=='convert_element_type':return np.asarray(xs[0],dtype=p['new_dtype'])
                    if name=='select_n':return xs[1+int(xs[0])]
                if name=='is_finite':return _kernel_is_finite(xs[0]) if isinstance(xs[0],jax.core.Tracer) else np.isfinite(xs[0])
                if name=='erf':
                    return _kernel_erf().bind(jnp.asarray(xs[0]))
                if name=='integer_pow':
                    return _kernel_integer_power(device_scalar(xs[0],equation.invars[0].aval.dtype),p['y'])
                if name=='add_any':return xs[0]+xs[1]
                if name=='le_to':return jnp.less_equal(*xs)|jnp.isnan(xs[1])
                return _bind_primitive(equation.primitive,[device_scalar(x,v.aval.dtype) for x,v in zip(xs,equation.invars)],parameters)
            signature=None
            try:
                signature=('scalar',equation.primitive,tuple(parameters.items()),
                    tuple((v.aval.dtype.str,v.aval.weak_type) for v in equation.invars))
                hash(signature)
            except TypeError:signature=None
            if equation.effects:signature=None
            def cached(*xs):
                if signature is None:return bind(*xs)
                key=signature+tuple(word_key(v) for v in xs)
                if key not in memo:memo[key]=bind(*xs)
                return memo[key]
            try:result=[elements(cached,*values)]
            except (ValueError,TypeError,NotImplementedError) as error:
                raise ValueError(f"scalar predictor operation {name}, input shapes {[v.aval.shape for v in equation.invars]}: {error}") from error
        env.update((v,logical(x,v.aval.shape)) for v,x in zip(equation.outvars,result))
    return tuple(jnp.asarray(scalar(read(v)),dtype=v.aval.dtype) if not v.aval.shape else read(v) for v in graph.outvars)

def _scalar_predictor(graph,calls):
    """Simplify scalar moment algebra before expanding native field reductions."""
    # Large spline tables stay as indexed kernel inputs, not scalar arguments.
    if any(v.size>1024 for v in graph.consts):return None
    variables=(*graph.jaxpr.constvars,*graph.jaxpr.invars)
    def evaluate(*words):
        values=[];offset=0
        for variable in variables:
            size=variable.aval.size
            values.append(tuple(words[offset:offset+size]));offset+=size
        count=len(graph.jaxpr.constvars)
        return scalar_graph(graph.jaxpr,values[:count],*values[count:],calls=calls,defer=True)
    examples=tuple(jnp.zeros((),v.aval.dtype) for v in variables for _ in range(v.aval.size))
    return _flatten(evaluate,*examples)

def prepare_physics(field, field_view, build, initial, noise_dim,*,predictor_only=False,spatial_geometry=None,time_dependent=False):
    """Compile the production orbit, kick and predictor into one GPU factory.

    Only field evaluation has a private adapter. Every collision map, physical
    normalization and moment derivative is the same function as shared JAX.
    """
    from .rk import _coil_field_values, _dot, _cross, _vmec_quantities, _vmec_series, _KernelField
    from .dynamics import _gc_quantities
    from .fields import BiotSavart, Vmec
    from .boozer import BoozerField
    calls=KernelCalls();coil=isinstance(field,BiotSavart);vmec=isinstance(field,Vmec)
    boozer=isinstance(field,BoozerField)
    arrays,metadata=field_view if field_view is not None else ((),None)
    nsource=len(arrays)
    extra=[]

    def flatten_values(values):
        return tuple(v for item in values for v in (tuple(item) if isinstance(item,(tuple,list)) or getattr(item,'ndim',0) else (item,)))
    def ordinary(fn):
        graph=_flatten(fn,*(jnp.zeros((),jnp.float64) for _ in range(3)))
        start=len(arrays)+len(extra);extra.extend(graph.consts)
        def evaluate(refs,*args):
            constants=[tuple(ref[index] for index in np.ndindex(v.shape)) if v.ndim else ref[()]
                for v,ref in zip(graph.consts,refs[start:start+len(graph.consts)])]
            return scalar_graph(graph.jaxpr,constants,*args)
        return evaluate
    def geometric(*point):return flatten_values(_gc_quantities(field,jnp.asarray(point)))
    def fast_geometry(refs,*point):
        if vmec:return flatten_values(_vmec_quantities(refs,point,metadata))
        B,J=_coil_field_values(refs,point,*metadata,True)
        magnitude=jnp.sqrt(_dot(B,B))
        gradient=tuple(sum(J[i][j]*B[i]/magnitude for i in range(3)) for j in range(3))
        curl=(J[2][1]-J[1][2],J[0][2]-J[2][0],J[1][0]-J[0][1])
        correction=_cross(B,gradient)
        curl=tuple(curl[i]/magnitude+correction[i]/magnitude**2 for i in range(3))
        curvature=tuple(-v/magnitude for v in _cross(B,curl))
        return (*B,*B,magnitude,*gradient,*curl,*curvature,jnp.asarray(1.,magnitude.dtype))
    geometry=None if boozer else calls.add(geometric,fast_geometry if coil or vmec else ordinary(geometric),17)

    def absb(*point):return (field.modB(*point) if boozer else field.AbsB(jnp.asarray(point)),)
    def fast_absb(refs,*point):
        if coil:
            B,_=_coil_field_values(refs,point,*metadata,False)
            return (jnp.sqrt(_dot(B,B)),)
        if vmec:return (_vmec_series(refs,point,metadata['bmnc'])[0],)
        model=_KernelField(refs[:nsource],float(field.psi0))
        graph=_flatten(lambda r,theta,z:model.modB_derivatives(r,theta,z)[0],jnp.sqrt(point[0]),point[1],point[2])
        return _evaluate_kernel(graph.jaxpr,graph.consts,jnp.sqrt(point[0]),point[1],point[2])
    magnitude=calls.add(absb,fast_absb if coil or vmec or boozer else ordinary(absb),1)
    if not boozer:
        vector=lambda *point:tuple(field.B_contravariant(jnp.asarray(point)))
        fast_vector=(lambda refs,*point:_coil_field_values(refs,point,*metadata,False)[0]) if coil else (
            lambda refs,*point:_vmec_quantities(refs,point,metadata)[1]) if vmec else ordinary(vector)
        magnetic_vector=calls.add(vector,fast_vector,3)

    class View:
        def _velocity_jet(self,function,vel):
            # Scalarize before AD so fixed-position tangents stay symbolic zero.
            graph=_flatten(lambda r,xi:tuple(function(jnp.stack((r,xi)))),*vel)
            def evaluate(r,xi):
                return scalar_graph(graph.jaxpr,graph.consts,r,xi,calls=calls,defer=True)
            first=[];second=[]
            for i in range(2):
                def column(v):return evaluate(v,vel[1]) if i==0 else evaluate(vel[0],v)
                first.append(jnp.asarray(jax.jacfwd(column)(vel[i])))
                second.append(jnp.asarray(jax.jacfwd(jax.jacfwd(column))(vel[i])))
            return jnp.asarray(evaluate(*vel)),jnp.stack(first,axis=1),jnp.stack(second,axis=1)
        def gc_quantities(self,point):
            v=geometry(*point)
            return jnp.asarray(v[:3]),jnp.asarray(v[3:6]),v[6],jnp.asarray(v[7:10]),jnp.asarray(v[10:13]),jnp.asarray(v[13:16]),v[16]
        def AbsB(self,point):return magnitude(*point)[0]
        def B_contravariant(self,point):return jnp.asarray(magnetic_vector(*point))
        def B_covariant(self,point):return self.gc_quantities(point)[0]
    view=View()
    if vmec:
        xyz=lambda s,theta,z:tuple(field.to_xyz(jnp.array([s,theta,z])))
        def fast_xyz(refs,s,theta,z):
            R=_vmec_series(refs,(s,theta,z),metadata['rmnc'])[0]
            Z=_vmec_series(refs,(s,theta,z),metadata['zmns'])[0]
            return R*jnp.cos(z),R*jnp.sin(z),Z
        position=calls.add(xyz,fast_xyz,3,kernel_jvp=lambda refs,p,d:_vmec_position_tangent(refs,p,d,metadata))
        view.to_xyz=lambda point:jnp.asarray(position(*point))
    if boozer:
        table_calls={}
        def tabulated(mode,point,t):
            key=id(mode)
            if key not in table_calls:
                data,evaluate=mode._kernel_view();start=len(arrays)+len(extra);extra.extend(data)
                def fast(refs,*q):
                    model=_KernelField(refs[:nsource],float(field.psi0))
                    def jet(*x):
                        value=lambda point:evaluate(refs[start:start+len(data)],model,
                            (point[0],point[1],point[2]),point[3])
                        point=jnp.asarray(x)
                        directions=jnp.eye(4,dtype=point.dtype)
                        jacobian=[jax.jvp(value,(point,),(directions[i],))[1] for i in range(4)]
                        return (*value(point),*(jacobian[j][i] for i in range(2) for j in range(4)))
                    graph=_flatten(jet,*q)
                    return _evaluate_kernel(graph.jaxpr,graph.consts,*q)
                table_calls[key]=calls.first_jet(lambda *q:tuple(mode.potentials(field,q[:3],q[3])),
                                                fast,outputs=2,dimensions=4)
            return jnp.asarray(table_calls[key](*point,t))
        view._tabulated_potentials=tabulated
        def modb(r,theta,z):return tuple(field.modB_derivatives(r,theta,z))
        def profiles(s):return flatten_values(field.profiles(s))
        def boozer_fast(fn):
            def evaluate(refs,*args):
                model=_KernelField(refs[:nsource],float(field.psi0))
                graph=_flatten(lambda *x:fn(model,*x),*args)
                return _evaluate_kernel(graph.jaxpr,graph.consts,*args)
            return evaluate
        mod=calls.add(modb,boozer_fast(lambda model,*x:tuple(model.modB_derivatives(*x))),4)
        prof=calls.add(profiles,boozer_fast(lambda model,s:flatten_values(model.profiles(s))),6)
        view.modB_derivatives=lambda *point:mod(*point)
        view.modB=lambda *point:magnitude(*point)[0]
        view.profiles=lambda s:(prof(s)[:3],prof(s)[3:])
        view.psi0=field.psi0
        if spatial_geometry is not None:
            xyz=lambda *q:tuple(spatial_geometry(jnp.asarray(q)))
            if hasattr(spatial_geometry,'_kernel_view'):
                data,evaluate=spatial_geometry._kernel_view();start=nsource+len(extra);extra.extend(data)
                fast_xyz=lambda refs,*q:evaluate(refs[start:start+len(data)],*q)
            else:fast_xyz=ordinary(xyz)
            position=calls.add(xyz,fast_xyz,3)
            view.to_xyz=lambda point:jnp.asarray(position(*point))
    rhs,kick,norm,level,limit=build(view)
    nstate=initial.size
    def kicked(y,h,dw):
        result=kick(0.,y,h,dw,None)
        return (*result.y,result.drift_error,result.diffusion_error)
    graphs=([] if predictor_only else [
        (_flatten(lambda t,y:tuple(rhs(t,y,None)),jnp.asarray(0.),initial) if time_dependent else
         _flatten(lambda y:tuple(rhs(0.,y,None)),initial)),
        _flatten(kicked,initial,jnp.asarray(1e-9),jnp.zeros(noise_dim)),
        _flatten(lambda y,e:norm(y,y,e),initial,jnp.zeros_like(initial)),
        _flatten(lambda y:level(0.,y,None),initial)])
    if coil and nstate==6:
        def vector_jet(refs,*point):
            B,J=_coil_field_values(refs,point,*metadata,True)
            return (*B,*(x for row in J for x in row))
        magnetic_vector=calls.first_jet(vector,vector_jet)
    graphs.append(_flatten(lambda t,y:limit(t,y,None),jnp.asarray(0.),initial))
    prediction=_scalar_predictor(graphs[-1],calls)
    nfield=len(arrays)+len(extra)
    arrays=tuple(arrays)+tuple(extra)
    all_arrays=tuple(arrays)+tuple(v.reshape(-1) if v.ndim else v for graph in graphs for v in graph.consts)
    def factory(refs,y0):
        offset=nfield;constants=[]
        for graph in graphs:
            values=[]
            for v,ref in zip(graph.consts,refs[offset:offset+len(graph.consts)]):
                values.append(_KernelTable(ref,v.shape) if v.ndim else ref[()])
            constants.append(values);offset+=len(graph.consts)
        def evaluate(i,*args):
            if i==len(graphs)-1 and prediction is not None:
                words=tuple(word for value in (*constants[i],*args)
                    for word in (tuple(np.asarray(value).flat) if isinstance(value,_KernelTable) else value
                                 if isinstance(value,tuple) else (value,)))
                return scalar_graph(prediction.jaxpr,prediction.consts,*words,calls=calls,refs=refs[:nfield])
            return scalar_graph(graphs[i].jaxpr,constants[i],*args,calls=calls,refs=refs[:nfield])
        state=lambda y:tuple(scalar(y,i) for i in range(nstate))
        if predictor_only:
            return None,None,None,None,lambda t,y:evaluate(0,t,state(y))[0]
        def padded(values):return pack(tuple(values)+(scalar(y0,7),)*(8-len(values)))
        def rhs(*arguments):
            t,y=arguments if time_dependent else (None,arguments[0])
            return padded(evaluate(0,t,state(y)) if time_dependent else evaluate(0,state(y)))
        def kick(y,h,dw):
            result=evaluate(1,state(y),h,tuple(scalar(dw,i) for i in range(noise_dim)))
            return padded(result[:nstate]),result[-2],result[-1]
        def norm(y,e,speed,budget):return evaluate(2,state(y),state(e))[0]
        return rhs,kick,norm,lambda y:evaluate(3,state(y))[0],lambda t,y:evaluate(4,t,state(y))[0]
    return all_arrays,factory
