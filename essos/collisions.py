"""Nonrelativistic collision maps and bounded Brownian rejection memory.

The velocity-aligned particle map and speed/pitch guiding-center map follow
Sarkimaki et al., arXiv:1701.05043, Eqs. (57)--(75). Coefficients come from
ESSOS's Maxwellian backgrounds, not the paper's relativistic coefficients.
These are pure numerical maps with explicit Wiener increments; no adjoint or
stop_gradient is installed here. In particular, coefficient derivatives remain
part of the differentiable collision update.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .background_species import nu_D_ab, nu_par_ab, d_nu_par_ab, nu_s_ab
from .collision_error import CollisionErrorModel, _ratio


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
