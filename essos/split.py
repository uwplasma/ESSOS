"""Forward nonrelativistic orbit/collision splitting.

An independently controlled native RK orbit solve precedes each Milstein
collision map. Brownian rejection memory preserves every sampled interval.
Private GPU adapters leave public field classes unchanged. No custom reverse
rule or checkpoint tape is installed in this forward-only module.
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
    def new(m):
        key, draw = jax.random.split(m.key)
        dw = jnp.sqrt(requested_end-m.t)*jax.random.normal(draw, m.increments.shape[1:], dtype=m.increments.dtype)
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
        key, draw = jax.random.split(m.key)
        alpha = h/duration
        left = alpha*parent + jnp.sqrt(h*(1-alpha))*jax.random.normal(
            draw, parent.shape, dtype=parent.dtype)
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

class SplitSolution(NamedTuple):
    ts: jax.Array
    ys: jax.Array
    t: jax.Array
    y: jax.Array
    status: jax.Array
    stats: SplitStats
    step_ts: jax.Array

def solve_split(rhs, collision, y0, args=None, *, t0=0.0, t1, dt0, ts, key, noise_dim, adaptive=True, dtmin=0.0, dtmax=jnp.inf, method='dopri8', orbit_dt0=None, orbit_adaptive=True, orbit_stage_execution='loop', reuse_orbit_step=True, collision_controller='conservative', collision_wiener_norm=None, rtol=1e-09, atol=1e-09, error_norm=None, controller='i', pcoeff=0.4, icoeff=0.3, event=None, max_steps=100000, max_orbit_steps=10000, max_pending=64, record_steps=False, step_limit=None, event_dtmax=None, _orbit_rhs_prepared=False):
    """Lie split: orbit(t, t+h), then collisions over h at the orbit endpoint.

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
    rhs_prepared = True
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
    s = dict(t=t0, y=y0, h=jnp.minimum(dt0, dtmax), orbit_h=orbit_dt0, status=initial_status, memory=memory, index=jnp.array(0), ys=jnp.broadcast_to(y0, (len(ts), y0.size)), steps=jnp.array(0), accepted=jnp.array(0), rejected=jnp.array(0), orbit_steps=jnp.array(0), orbit_rejected=jnp.array(0), max_error=jnp.zeros((), y0.dtype), history=jnp.full((max_steps + 1 if record_steps else 0,), jnp.inf, dtype=y0.dtype))
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

    def advance_with_limit(s, limit):
        barrier = t1 if not len(ts) else jnp.where(s['index'] < len(ts), ts[jnp.minimum(s['index'], len(ts) - 1)], t1)
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
        if lookahead:
            m, dw, overflow = brownian_sample(s['memory'], wanted_end)
            end, h = (wanted_end, wanted_end - s['t'])
        else:
            m, h, dw, overflow = brownian_propose(s['memory'], wanted_end - s['t'], t_end=wanted_end)
            end = m.ends[m.size - 1]
        usable = ~overflow & (h > 0) & (end > s['t'])

        def attempt(_):
            orbit_out = solve_rk(orbit_rhs, s['y'], args, t0=s['t'], t1=end, dt0=jnp.minimum(s['orbit_h'] if reuse_orbit_step and orbit_adaptive else orbit_dt0, h), ts=jnp.empty((0,), y0.dtype), method=method, adaptive=orbit_adaptive, rtol=rtol, atol=atol, stage_execution=orbit_stage_execution, error_norm=error_norm, controller=controller, pcoeff=pcoeff, icoeff=icoeff, event=event, root_find=True, max_steps=max_orbit_steps, _rhs_prepared=rhs_prepared)
            orbit = orbit_out

            def kick(_):
                result = collision(end, orbit.y, h, dw, args)
                if result.y.shape != y0.shape:
                    raise ValueError('collision must preserve the state shape')
                if jnp.ndim(result.drift_error) or jnp.ndim(result.diffusion_error):
                    raise ValueError('collision errors must be scalars')
                return result if candidate else result._replace(error_model=None)
            trial = lax.cond(orbit.status == RKStatus.SUCCESS, kick, lambda _: CollisionStep(orbit.y, jnp.zeros((), y0.dtype), jnp.zeros((), y0.dtype), empty_model), None)
            v = values(end, trial.y)
            finite = jnp.isfinite(trial.y).all() & finite_values(v) & jnp.isfinite(trial.drift_error) & (trial.drift_error >= 0) & jnp.isfinite(trial.diffusion_error) & (trial.diffusion_error >= 0)
            error = jnp.maximum(trial.drift_error, trial.diffusion_error)
            accepted = (orbit.status == RKStatus.SUCCESS) & finite & ((error <= 1) | (not adaptive))
            refine_event = jnp.array(False)
            if event_dtmax is not None:
                refine_event = accepted & stopped(v) & (h > event_dtmax * (1 + 8 * jnp.finfo(y0.dtype).eps))
                accepted &= ~refine_event
            rejected = (orbit.status == RKStatus.SUCCESS) & finite & ~accepted
            orbit_event = orbit.status == RKStatus.EVENT
            status = jnp.where(orbit.status != RKStatus.SUCCESS, orbit.status, jnp.where(~finite, SplitStatus.NONFINITE, jnp.where(accepted & stopped(v), SplitStatus.EVENT, jnp.where(rejected & (h <= dtmin), SplitStatus.MIN_STEP, SplitStatus.SUCCESS))))
            if adaptive and (not lookahead):
                factor = 0.9 * jnp.minimum(jnp.maximum(trial.drift_error, 1e-30) ** (-0.5), jnp.maximum(trial.diffusion_error, 1e-30) ** (-2 / 3))
                factor = jnp.clip(factor, 0.1, jnp.where(accepted, 1.5, 0.9))
                next_h = jnp.clip(h * factor, dtmin, dtmax)
            else:
                next_h = jnp.minimum(dt0, dtmax)
            state = dict(s, t=jnp.where(orbit_event, orbit.t, jnp.where(accepted, end, s['t'])), y=jnp.where(orbit_event, orbit.y, jnp.where(accepted, trial.y, s['y'])), h=next_h, status=status, steps=s['steps'] + 1, accepted=s['accepted'] + accepted.astype(s['accepted'].dtype), rejected=s['rejected'] + rejected.astype(s['rejected'].dtype), orbit_steps=s['orbit_steps'] + orbit.stats.steps, orbit_rejected=s['orbit_rejected'] + orbit.stats.rejected, memory=lax.cond(accepted, (lambda m: brownian_consume(m, end)) if lookahead else brownian_accept, lambda m: m, m), max_error=jnp.maximum(s['max_error'], jnp.where(accepted, error, 0.0)))
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
    solution = SplitSolution(ts, ys, s['t'], s['y'], status, SplitStats(s['steps'], s['accepted'], s['rejected'], s['orbit_steps'], s['orbit_rejected'], s['memory'].peak, s['max_error']), s['history'])
    return solution

def trace_split(field, particles, species, *, model='GuidingCenter', tmax, timestep, times=None, n_save=100, integrator='essos_dopri8', collision_tolerance=0.001, adaptive=True, collision_dtmax=None, orbit_dt0=None, orbit_adaptive=True, orbit_stage_execution=None, reuse_orbit_step=True, collision_controller='conservative', rtol=1e-09, atol=1e-09, controller='i', pcoeff=0.4, icoeff=0.3, speed_floor=None, spatial_diffusion=False, orbit=True, electric_field=None, event=None, max_steps=100000, max_orbit_steps=10000, max_pending=64, record_steps=False, execution='shared'):
    """Batched forward split tracing using ESSOS field and particle objects.

Returns a ``SplitSolution`` with shared ts and particle-leading arrays. GC states are
``(position[3], v_parallel, mu)`` with **physical** mu=m*v_perp^2/(2 B)
[J/T], unlike the legacy normalized collisional state. FullOrbit states
are ``(xyz, vx, vy, vz)``. VMEC/Boozer positions are (s, theta, zeta).
VMEC uses its regular axis chart; Boozer uses (sqrt(s)cos(theta),
sqrt(s)sin(theta)) internally. Both stop at the LCFS. Exterior continuation
is not part of this entry point. ``event`` receives public physical states.

``spatial_diffusion=True`` adds paper Eq. (69) for Cartesian GC fields;
coordinate-field diffusion needs a separate coordinate transformation and
is rejected here. By default the kick acts on velocity only, like ESSOS's
existing mu collision path. ``orbit=False`` runs collision-only statistics.
``speed_floor`` defaults to 5% of the smallest active background thermal
speed for the test mass. The GC reflection wall affects near-zero speeds;
refine/check that parameter when studying the thermal population.

Independent per-particle keys come from ``particles.random_keys``. The
field classes and existing Tracing/trace_boozer entry points are unchanged.

    ``execution="independent"`` uses one persistent GPU program per particle.
    It currently supports alpha particles, one uniform NRL background, zero
    electric field, Dopri8/I orbits, conservative collisions, matching orbit/
    collision adaptation, and no explicit GC spatial kicks. The shared path
    supports the general physical options. Unsupported GPU options raise.
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
    if orbit_stage_execution is None:
        orbit_stage_execution = 'unrolled' if integrator == 'essos_dopri8' and model == 'GuidingCenter' and isinstance(field, BiotSavart) and (not adaptive) and (not orbit_adaptive) and (jax.default_backend() == 'gpu') else 'loop'
    gc = model == 'GuidingCenter'
    if not gc and (boozer or vmec):
        raise ValueError('FullOrbit requires a Cartesian field')
    if spatial_diffusion and (not gc or boozer or vmec):
        raise ValueError('spatial_diffusion requires a Cartesian guiding-center field')
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
    field_view = _split_field_data(field) if execution == 'independent' else None

    def magnitude(point):
        return field.modB(*point) if boozer else field.AbsB(point)

    def public(y):
        if vmec:
            return _from_axis_regular(y)
        if boozer:
            return y.at[0].set(y[0] ** 2 + y[1] ** 2).at[1].set(jnp.arctan2(y[1], y[0]))
        return y

    def gc_drift(t, y, args):
        dx, dv = _guiding_center_velocity(field, electric_field, charge, mass, y[:3], y[3], mu=y[4])
        return jnp.concatenate([dx, jnp.array([dv, 0.0])])

    def rhs(t, y, args):
        if not orbit:
            return jnp.zeros_like(y)
        if boozer:
            return jnp.append(guiding_center_rhs(field, y[:4], y[4] / mass, mass, charge), 0.0)
        if vmec:
            return _axis_regular(gc_drift)(t, y, args)
        if gc:
            return gc_drift(t, y, args)
        return jnp.concatenate([y[3:], charge / mass * (electric_field.E_covariant(y[:3]) + jnp.cross(y[3:], field.B_contravariant(y[:3])))])
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
    def one(initial, key):
        birth = public(initial)
        birth_speed = jnp.sqrt(birth[3] ** 2 + 2 * birth[4] * magnitude(birth[:3]) / mass) if gc else jnp.linalg.norm(birth[3:])
        if speed_floor is None:
            thermal = jnp.stack([jnp.where(species.get_density(i, birth[:3]) > 0, jnp.sqrt(2 * ELEMENTARY_CHARGE * species.get_temperature(i, birth[:3]) / mass), jnp.inf) for i in range(species.number_species)])
            floor = 0.05 * jnp.min(thermal)
            floor = jnp.where(jnp.isfinite(floor), floor, birth_speed * 1e-12)
        else:
            floor = jnp.asarray(speed_floor, dtype=initial.dtype)

        def kick(t, y, h, dw, args):
            state = public(y)
            point = state[:3]
            B = magnitude(point) if gc else jnp.ones((), y.dtype)
            speed = jnp.sqrt(state[3] ** 2 + 2 * state[4] * B / mass) if gc else jnp.linalg.norm(state[3:])
            c = collision_coefficients(species, mass, charge, speed, point, full_orbit=not gc)
            if gc:
                step = guiding_center_collision_step(speed, state[3] / speed, h, dw[-2:], c, tolerance=collision_tolerance, speed_floor=floor, return_error_model=adaptive and collision_controller == 'candidate')
                v, pitch = step.y
                if spatial_diffusion:
                    b = field.B_contravariant(point) / B
                    point = point + guiding_center_spatial_increment(state[3] / speed, dw[:3], c, b, charge * B / mass)
                    y = y.at[:3].set(point)
                    B = magnitude(point)
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

        def collision_wiener_norm(dw):
            return jnp.linalg.norm(dw[-2:]) if gc else jnp.abs(dw[2])
        norm = OrbitErrorNorm(rtol=rtol, atol=atol, speed_scale=birth_speed, velocity_components=1 if gc else 3, position_map=position if vmec else None)
        if boozer:

            def norm(a, b, error):
                return jnp.maximum(jnp.linalg.norm(error[:2]), jnp.maximum(jnp.abs(error[2]), jnp.abs(error[3]) / birth_speed)) / (rtol + atol)
        events = []
        if vmec or boozer:
            events.append(lambda t, y, args: 1 - y[0] ** 2 - y[1] ** 2)
        if event is not None:
            events.extend([lambda t, y, args, f=f: f(t, public(y), args) for f in tree_util.tree_leaves(event)])
        solve = solve_split
        if execution == 'independent':
            from functools import partial

            solve = partial(_solve_independent, field=field, species=species, gc=gc, birth_speed=birth_speed, speed_floor=floor, collision_tolerance=collision_tolerance, field_view=field_view)
        solution = solve(rhs, kick, initial, t1=tmax, dt0=timestep, ts=times, key=key, noise_dim=(5 if spatial_diffusion else 2) if gc else 3, dtmax=timestep if collision_dtmax is None else collision_dtmax, adaptive=adaptive, orbit_dt0=orbit_dt0, orbit_adaptive=orbit_adaptive, orbit_stage_execution=orbit_stage_execution, reuse_orbit_step=reuse_orbit_step, collision_controller=collision_controller, collision_wiener_norm=collision_wiener_norm, method='dopri5' if integrator == 'essos_rk45' else 'dopri8', rtol=rtol, atol=atol, controller=controller, pcoeff=pcoeff, icoeff=icoeff, error_norm=norm, event=tuple(events), max_steps=max_steps, max_orbit_steps=max_orbit_steps, max_pending=max_pending, record_steps=record_steps, _orbit_rhs_prepared=execution == "shared")
        return solution._replace(ys=jax.vmap(public)(solution.ys), y=public(solution.y))
    if speed_floor is not None:
        _concrete_check(speed_floor, lambda x: np.ndim(x) == 0 and np.isfinite(x) and (x > 0), 'speed_floor must be positive and finite')
    return jax.jit(jax.vmap(one))(y0, particles.random_keys)._replace(ts=times)



# Independent GPU execution
from jax.extend.core import Literal
from .rk import (_select_one, _kernel_is_finite, _evaluate_inline, _bind_primitive,
    _A8, _ERROR8, _DENSE8, _pack, _cross, _dot, _coil_geometry,
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
    graph = jax.make_jaxpr(function)(*examples)
    return jax.make_jaxpr(lambda *x: _evaluate_inline(graph.jaxpr, graph.consts, *x))(*examples)

def _evaluate_kernel(graph, constants, *arguments, scalarize=False):

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
            folded = None
            if name in numpy_ops:
                folded = [numpy_ops[name](*vals)]
            elif name == 'integer_pow':
                folded = [np.power(vals[0], p['y'])]
            elif name == 'convert_element_type':
                folded = [np.asarray(vals[0], dtype=p['new_dtype'])]
            if folded is not None:
                env.update(((out, np.asarray(value, dtype=out.aval.dtype)) for out, value in zip(eq.outvars, folded)))
                continue
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
        elif name == 'le_to':
            result = [jnp.less_equal(vals[0], vals[1]) | jnp.isnan(vals[1])]
        elif name in ('concatenate', 'stack') and (len(vals) > 2 or (scalarize and all((v.size == 1 for v in vals)))):
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

def _prepare_split_kernel(arrays, factory, times, *, adaptive, dt0, dtmax, speed, tolerance=1e-09, noise_dim=2, max_steps=100000, max_orbit_steps=10000, max_pending=64, warps=1, state_size=8, bank_components=8, orbit_dt0=None, final_time=None, interpret=False):
    """factory(refs, initial) -> (rhs, kick, orbit_error, level)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt
    times = jnp.asarray(times, jnp.float64)
    nsave, stages, bank_size = (len(times), len(_ERROR8), 16)
    matrix = np.zeros((bank_size, bank_size))
    for i, row in enumerate(_A8, 1):
        matrix[i, :len(row)] = row
    dense = np.zeros((len(_DENSE8[0]), bank_size))
    dense[:, :stages] = np.asarray(_DENSE8).T
    inputs = tuple(arrays) + (jnp.asarray(matrix), jnp.asarray(dense), jnp.asarray(tuple(_ERROR8) + (0.0,) * (bank_size - stages)), jnp.asarray(times), jnp.asarray([0.0, dt0, dtmax, times[-1] if final_time is None else final_time, speed, 2 * tolerance, dt0 if orbit_dt0 is None else orbit_dt0], jnp.float64))
    capacity = max_pending if adaptive else 1
    noise_size = 1 << (noise_dim - 1).bit_length()

    def solve(initial, key):

        def kernel(yref, kref, *refs):
            *data, aref, dref, eref, tsref, pref, ysout, yout, tout, statsout, statusout, ends, increments = refs
            y0 = yref[...]
            zero, first_h, ceiling, final_t, birth_speed, budget, first_orbit_h = (pref[i] for i in range(7))
            rhs, kick, norm, level = factory(data, y0)

            def dynamic_rhs(y):
                value = rhs(y)
                return pack(tuple((scalar(value, i) for i in range(4)))) if bank_components == 4 else value

            def expand(value):
                return pack(tuple((scalar(value, i) for i in range(4))) + (zero, zero, zero, zero)) if bank_components == 4 else value
            integer = lambda x: jnp.asarray(x, jnp.int32)
            finite = lambda x: jnp.min(_kernel_is_finite(x).astype(jnp.int32)) > 0
            bi = jnp.arange(bank_size)[:, None]

            def polynomial(y, endpoint, bank, fraction):
                weights = jnp.zeros((bank_size,), y.dtype)
                for power in range(dense.shape[0]):
                    weights = weights * fraction + dref[power, :]
                result = y + expand(jnp.sum((weights * fraction)[:, None] * bank, axis=0))
                return jnp.where(fraction == 0, y, jnp.where(fraction == 1, endpoint, result))

            def orbit(y, t, end, guess):
                old = level(y)
                start = (t, y, dynamic_rhs(y), jnp.minimum(guess, end - t), old, integer(0), integer(0), integer(0))

                def step(c):
                    t, y, f0, guess, old, status, trials, rejected = c
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
                        fi = dynamic_rhs(yi)
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
                    return (jnp.where(accept, jnp.where(crossing, t + root * h, stop), t), jnp.where(accept, stop_y, y), jnp.where(accept, f1, f0), proposal, jnp.where(accept, new, old), status, trials + 1, rejected + (~accept).astype(jnp.int32))
                out = jax.lax.while_loop(lambda c: (c[0] < end) & (c[5] == 0) & (c[6] < max_orbit_steps), step, start)
                t, y, _, guess, _, status, trials, rejected = out
                status = jnp.where((t < end) & (status == 0), 2, status).astype(jnp.int32)
                return (t, y, guess, status, trials, rejected)

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
            start = (zero, y0, jnp.minimum(first_h, ceiling), first_orbit_h, kref[...], integer(0), integer(0), index, integer(0), initial_status, integer(0), integer(0), integer(0), integer(0), integer(0), zero)

            def advance(c):
                t, y, h, oh, key, size, peak, index, grid, status, trials, accepted, rejected, ot, orr, maxerr = c
                barrier = jnp.where(index < nsave, tsref[jnp.minimum(index, nsave - 1)], final_t)
                if adaptive:
                    wanted = jnp.minimum(t + h, barrier)
                    grid_end = wanted
                else:
                    grid_end = (grid + 1) * jnp.minimum(first_h, ceiling)
                    close = jnp.abs(grid_end - barrier) <= 8 * jnp.finfo(y.dtype).eps * jnp.maximum(jnp.abs(grid_end), jnp.abs(barrier))
                    grid_end = jnp.where(close, barrier, grid_end)
                    wanted = jnp.minimum(grid_end, barrier)

                def fresh(c):
                    key, size, peak = c
                    key, z = _draw_normal(key, noise_dim)
                    ends[0], increments[0, :] = (wanted, jnp.sqrt(wanted - t) * z)
                    return (key, size + 1, jnp.maximum(peak, 1))
                key, size, peak = jax.lax.cond(size == 0, fresh, lambda c: c, (key, size, peak))
                parent_end, parent = (ends[size - 1], increments[size - 1, :])
                end = jnp.minimum(wanted, parent_end)
                duration, h = (parent_end - t, end - t)
                split = end < parent_end
                overflow = split & (size == capacity)

                def bridge(c):
                    key, size, peak = c
                    key, z = _draw_normal(key, noise_dim)
                    alpha = h / duration
                    left = alpha * parent + jnp.sqrt(h * (1 - alpha)) * z
                    increments[size - 1, :], increments[size, :], ends[size] = (parent - left, left, end)
                    return (key, size + 1, jnp.maximum(peak, size + 1))
                key, size, peak = jax.lax.cond(split & ~overflow, bridge, lambda c: c, (key, size, peak))
                usable = ~overflow & (h > 0) & (end > t)

                def attempt(_):
                    stop, endpoint, next_oh, os, orbit_trials, orbit_rejected = orbit(y, t, end, jnp.minimum(oh, h) if adaptive else jnp.minimum(first_orbit_h, h))

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
                    return (nt, ny, nh, next_oh, key, size - accept.astype(jnp.int32), peak, ni, grid + (accept & (end == grid_end)).astype(jnp.int32), ns, trials + 1, accepted + accept.astype(jnp.int32), rejected + reject.astype(jnp.int32), ot + orbit_trials, orr + orbit_rejected, jnp.maximum(maxerr, jnp.where(accept, error, zero)))
                return jax.lax.cond(usable, attempt, lambda _: c[:4] + (key, size, peak, index, grid, jnp.where(overflow, 6, 3).astype(jnp.int32)) + c[10:], None)
            out = jax.lax.while_loop(lambda c: (c[0] < final_t) & (c[9] == 0) & (c[10] < max_steps), advance, start)
            t, y, _, _, _, _, peak, index, _, status, trials, accepted, rejected, ot, orr, maxerr = out
            status = jnp.where((t < final_t) & (status == 0), 2, status)

            def tail(i, _):
                ysout[i, :] = y
                return ()
            jax.lax.fori_loop(index, nsave, tail, ())
            yout[...], tout[...], statusout[...] = (y, t, status)
            statsout[...] = pack(tuple((v.astype(y.dtype) for v in (trials, accepted, rejected, ot, orr, peak))) + (maxerr, zero))
        shapes = (jax.ShapeDtypeStruct((nsave, 8), jnp.float64), jax.ShapeDtypeStruct((8,), jnp.float64), jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((8,), jnp.float64), jax.ShapeDtypeStruct((), jnp.int32), jax.ShapeDtypeStruct((capacity,), jnp.float64), jax.ShapeDtypeStruct((capacity, noise_size), jnp.float64))
        outputs = pl.pallas_call(kernel, out_shape=shapes + (), grid=(), compiler_params=plt.CompilerParams(num_warps=warps), interpret=interpret)(initial, key, *inputs)
        ys, y, t, stats, status = outputs[:5]
        return (ys, y, t, stats, status)
    return solve


def _validate_independent(field, particles, species, *, model, integrator, adaptive,
        orbit_adaptive, controller, collision_controller, electric_field,
        spatial_diffusion, orbit, record_steps, reuse_orbit_step, event, times, tmax):
    from .fields import BiotSavart, Vmec
    from .boozer import BoozerField
    from .electric_field import Electric_field_zero
    if not isinstance(field, (BiotSavart, Vmec, BoozerField)):
        raise ValueError("independent split execution requires BiotSavart, VMEC or Boozer fields")
    if integrator != "essos_dopri8" or controller != "i" or collision_controller != "conservative":
        raise ValueError("independent execution requires Dopri8/I and conservative collision control")
    if adaptive != orbit_adaptive or not reuse_orbit_step:
        raise ValueError("independent execution requires matching adaptation and orbit-step reuse")
    if not isinstance(electric_field, Electric_field_zero) or spatial_diffusion or not orbit:
        raise ValueError("independent execution requires zero electric field and velocity-only kicks")
    if record_steps:
        raise ValueError("independent execution does not provide step histories")
    if species.number_species != 1 or species.radial_grid is not None or species.coulomb_log_model != "nrl":
        raise ValueError("independent execution requires one uniform NRL background species")
    for value, expected in ((particles.mass, M), (particles.charge, Q)):
        if isinstance(value, jax.core.Tracer):
            raise ValueError("independent execution requires fixed alpha mass and charge")
        _concrete_check(value, lambda x: np.all(x == expected), "independent execution requires alpha particles")
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


def _coefficient_graph(species, full_orbit):
    """Remove the singleton species axis without changing the physical rates."""
    def coefficients(speed):
        def values(v):
            point = jnp.zeros(4)
            nd = nu_D_ab(M, Q, 0, v, point, species)
            np_ = nu_par_ab(M, Q, 0, v, point, species)
            dnp = d_nu_par_ab(M, Q, 0, v, point, species)
            ns = nu_s_ab(M, Q, 0, v, point, species)
            return .5*v*v*np_, .5*v*v*nd, v*np_+.5*v*v*dnp, -ns*v
        val, slope = jax.jvp(values, (speed,), (jnp.ones_like(speed),))
        dp, dt, ddp, q = val
        dk = slope[3]+slope[2]+2*(slope[0]-slope[1])/speed-2*(dp-dt)/speed**2 if full_orbit else jnp.zeros_like(speed)
        return dp, dt, ddp, q, slope[3], dk
    return _flatten(coefficients, jnp.asarray(4e5))


def _split_field_data(field):
    """Prepare field views before vmap/JIT, including cached coil properties."""
    from .fields import BiotSavart, Vmec
    coil, vmec = isinstance(field, BiotSavart), isinstance(field, Vmec)
    if coil:
        arrays, nc, nq = _coil_geometry(field)
    elif vmec:
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


def _prepare_split_field(field, species, *, gc, collision_tolerance, speed_floor,
                         event=None, state_size=5, field_view=None):
    """Reuse the orbit PR's private field views, carrying physical mu per kick."""
    from .fields import BiotSavart, Vmec
    from .boozer import guiding_center_rhs
    coil, vmec = isinstance(field, BiotSavart), isinstance(field, Vmec)
    arrays, metadata = _split_field_data(field) if field_view is None else field_view
    if coil:
        nc, nq = metadata
    nfield = len(arrays)
    cg = _coefficient_graph(species, not gc)
    def gc_map(v, xi, h, dw, *c):
        out = guiding_center_collision_step(v, xi, h, dw, CollisionCoefficients(*c),
            tolerance=collision_tolerance, speed_floor=speed_floor)
        return out.y, out.drift_error, out.diffusion_error
    gg = _flatten(gc_map, jnp.asarray(4e5), jnp.asarray(.5), jnp.asarray(1e-7),
                   jnp.zeros(2), *(jnp.asarray(1.),)*6) if gc else None
    functions = tree_util.tree_leaves(event)
    eg = _flatten(lambda y: jnp.min(jnp.asarray([f(0., y, None) for f in functions])),
                  jnp.zeros(state_size)) if functions else None
    event_arrays = []
    if eg is not None:
        for value in eg.consts:
            if value.ndim > 1:
                raise ValueError("independent stopping surfaces require scalar/vector constants")
            event_arrays.append(value if value.ndim == 0 else
                jnp.pad(value, (0, (1 << (value.size-1).bit_length())-value.size)))
    all_arrays = tuple(arrays)+tuple(cg.consts)+(tuple(gg.consts) if gc else ())+tuple(event_arrays)

    def factory(refs, y0):
        frefs = refs[:nfield]
        ci = nfield+len(cg.consts)
        cc = [ref[...] for ref in refs[nfield:ci]]
        mc = [ref[...] for ref in refs[ci:ci+len(gg.consts)]] if gc else ()
        erefs = refs[ci+(len(gg.consts) if gc else 0):]
        ec = ([ref[...] if value.ndim == 0 else tuple(scalar(ref[...], i) for i in range(value.size))
               for ref, value in zip(erefs, eg.consts)] if eg is not None else ())
        zero = scalar(y0, 7)
        def orbit_state(y):
            return pack(tuple(scalar(y, i) for i in (0, 1, 2, 3, 5))+(zero, zero, zero))
        if coil:
            def rhs(y):
                value = _coil_rhs_values(frefs, y, nc, nq, "gc" if gc else "lorentz", 0., M, Q,
                                         scalar(y, 4) if gc else None)
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
                value = _vmec_rhs_values(frefs, orbit_state(y), metadata, M, Q, 0., scalar(y, 4))
                return pack(tuple(scalar(value, i) for i in range(4))+(zero, scalar(value, 4), zero, zero))
            magnitude = lambda y: _vmec_series(frefs, _vmec_chart(orbit_state(y)), metadata['bmnc'])[0]
            def norm(y, e, speed, budget):
                a = _vmec_xyz_values(frefs, orbit_state(y), metadata)
                b = _vmec_xyz_values(frefs, orbit_state(y-e), metadata)
                return jnp.maximum(jnp.sqrt(sum((u-v)**2 for u, v in zip(a, b))), jnp.abs(scalar(e, 3))/speed)/budget
        else:
            model = _KernelField(frefs, float(field.psi0))
            def physical(y):
                value = guiding_center_rhs(model, y, y[4]/M, M, Q)
                return pack(tuple(value[i] for i in range(4))+(zero, zero, zero, zero))
            graph = _flatten(physical, y0)
            rhs = lambda y: _evaluate_kernel(graph.jaxpr, graph.consts, y)[0]
            bg = _flatten(lambda y: model.modB_derivatives(jnp.sqrt(y[0]**2+y[1]**2),
                          jnp.arctan2(y[1], y[0]), y[2])[0], y0)
            magnitude = lambda y: _evaluate_kernel(bg.jaxpr, bg.consts, y)[0]
            def norm(y, e, speed, budget):
                return jnp.maximum(jnp.sqrt(scalar(e, 0)**2+scalar(e, 1)**2),
                    jnp.maximum(jnp.abs(scalar(e, 2)), jnp.abs(scalar(e, 3))/speed))/budget
        level = (lambda y: zero+1.) if eg is None else lambda y: _evaluate_surface(eg.jaxpr, ec, y)
        def kick(y, h, dw):
            B = magnitude(y) if gc else zero+1.
            speed = jnp.sqrt(scalar(y, 3)**2+2*scalar(y, 4)*B/M) if gc else jnp.sqrt(sum(scalar(y, i)**2 for i in range(3, 6)))
            c = _evaluate_kernel(cg.jaxpr, cc, speed, scalarize=True)
            if gc:
                changed, ed, es = _evaluate_kernel(gg.jaxpr, mc, speed, scalar(y, 3)/speed, h, dw, *c, scalarize=True)
                v, xi = scalar(changed, 0), scalar(changed, 1)
                result = jnp.where(jnp.arange(8) == 3, v*xi,
                    jnp.where(jnp.arange(8) == 4, M*v*v*(1-xi*xi)/(2*B), y))
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
                       birth_speed, speed_floor, collision_tolerance, field_view=None, **options):
    for name in ('max_steps', 'max_orbit_steps', 'max_pending'):
        value = options[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(name+" must be a positive integer")
    nstate = initial.size
    arrays, factory = _prepare_split_field(field, species, gc=gc,
        collision_tolerance=collision_tolerance, speed_floor=speed_floor,
        event=options['event'], state_size=nstate, field_view=field_view)
    solve = _prepare_split_kernel(arrays, factory, options['ts'], adaptive=options['adaptive'],
        dt0=options['dt0'], dtmax=options['dtmax'], speed=birth_speed,
        tolerance=(options['rtol']+options['atol'])/2, noise_dim=2 if gc else 3,
        max_steps=options['max_steps'], max_orbit_steps=options['max_orbit_steps'],
        max_pending=options['max_pending'], state_size=nstate,
        bank_components=4 if gc and nstate == 5 else 8,
        orbit_dt0=options['orbit_dt0'], final_time=options['t1'])
    ys, y, t, stats, status = solve(jnp.pad(initial, (0, 8-nstate)), jax.random.key_data(options['key']))
    return SplitSolution(options['ts'], ys[:, :nstate], t, y[:nstate], status,
        SplitStats(*(stats[i].astype(jnp.int32) for i in range(6)), stats[6]),
        jnp.empty((0,), initial.dtype))
