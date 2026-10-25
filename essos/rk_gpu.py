"""Forward-only independent GPU orbit loops with private field adapters.

Each GPU program keeps one orbit's Dopri5/8 stages, I controller, dense saves
and edge root search together. Field adapters supply the RHS, error blocks
and stopping level set. This module has no collision or reverse-AD rule.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from .rk import _A, _ERROR, _MID, _A8, _ERROR8, _DENSE8, RKSolution, RKStats
from .rk_gpu_utils import _kernel_is_finite, _select_one


def prepare_gpu_solver(arrays, factory, speed, times, dt0, rtol, atol, max_steps, *,
                       state_size=4, method="dopri8", adaptive=True, num_warps=1, interpret=False):
    """Prepare a CUDA solver outside jit; return solve(y0, invariant).

    Fixed and adaptive Dopri5/8 share FSAL stages and standard dense output.
    Adaptive acceptance uses the adapter's error blocks and I control. Save
    times do not constrain steps. Each particle stops independently; there is
    no particle chunking or compaction. Numerical equivalence is subject to
    ordinary floating-point changes from a different compiler execution path.
    """
    if method not in ("dopri5", "dopri8"):
        raise ValueError("The fused orbit method must be dopri5 or dopri8")
    if not interpret and (jax.default_backend() != "gpu" or "cuda" not in jax.devices()[0].client.platform_version.lower()):
        raise ValueError("The experimental fused orbit solver requires CUDA")
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or
            times[0] != 0 or times[-1] <= 0 or np.any(np.diff(times) < 0)):
        raise ValueError("Save times must be ordered, start at zero and end at a positive time")
    if not (np.isfinite(dt0) and dt0 > 0 and np.isfinite(speed) and speed > 0 and
            np.isfinite(rtol) and rtol >= 0 and np.isfinite(atol) and atol > 0):
        raise ValueError("Require positive finite dt0, speed, atol and finite rtol >= 0")
    if (isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)) or
            not 0 < max_steps <= np.iinfo(np.int32).max):
        raise ValueError("max_steps must be a positive int32 work bound")
    rows, weights, order = (_A, _ERROR, 5) if method == "dopri5" else (_A8, _ERROR8, 8)
    stages = len(weights)
    bank_size = 1 << (stages-1).bit_length()
    matrix = np.zeros((bank_size, bank_size))
    for i, row in enumerate(rows, 1):
        matrix[i, :len(row)] = row
    coefficients = jnp.asarray(matrix)
    error_weights = jnp.asarray(tuple(weights)+(0.,)*(bank_size-stages))
    if method == "dopri5":
        dense_matrix = np.zeros((1, bank_size))
        dense_matrix[0, :stages] = np.asarray(_MID)
    else:
        dense_matrix = np.zeros((len(_DENSE8[0]), bank_size))
        dense_matrix[:, :stages] = np.asarray(_DENSE8).T
    dense_coefficients = jnp.asarray(dense_matrix)
    tmax = float(times[-1])
    # Load float64 scalars explicitly. Nested Pallas loops must not infer
    # their device scalar precision from weak Python constant initializers.
    scalars = jnp.asarray([0., dt0, tmax, 8*np.finfo(np.float64).eps*tmax,
                           speed, rtol+atol], jnp.float64)
    inputs = tuple(arrays)+(coefficients, error_weights, dense_coefficients, jnp.asarray(times), scalars)
    nsave = len(times)

    def solve(y0, moment):
        if y0.shape != (state_size,) or y0.dtype != jnp.float64:
            raise ValueError("The fused orbit solver requires a float64 power-of-two state")
        # Validate the birth once with ordinary JAX, outside the persistent
        # loop. Invalid inputs must not spend max_steps rejecting NaN trials.
        initial_level = jnp.asarray(1., y0.dtype)
        initial_status = jnp.where(jnp.isfinite(y0).all() & jnp.isfinite(initial_level),
            jnp.where(initial_level == 0, 1, 0), 4).astype(jnp.int32)

        def kernel(yref, mref, initial_status_ref, *refs):
            (*field_refs, aref, eref, dref, tsref, paramsref,
             ysout, tsout, yout, tout, hout, statsout, statusout) = refs
            point = yref[...]
            zero, initial_h, final_t, roundoff, particle_speed, budget = (paramsref[i] for i in range(6))
            evaluate, estimate, level = factory(field_refs, point, mref[...])

            def finite(value):
                return jnp.min(_kernel_is_finite(value).astype(jnp.int32)) > 0

            def dense(y, endpoint, bank, fraction):
                if method == "dopri5":
                    # Same quartic extension as rk._rk45_dense; the padded
                    # bank's last row is not the final FSAL stage.
                    first = _select_one(bank, 0, 0).reshape(state_size)
                    last = _select_one(bank, stages-1, 0).reshape(state_size)
                    midpoint = jnp.sum(dref[0, :][:, None]*bank, axis=0)
                    delta = endpoint-y
                    a = 2*(last-first)-8*delta+16*midpoint
                    b = 5*first-3*last+14*delta-32*midpoint
                    c = last-4*first-5*delta+16*midpoint
                    value = y+fraction*(first+fraction*(c+fraction*(b+fraction*a)))
                    return jnp.where(fraction == 0, y, jnp.where(fraction == 1, endpoint, value))
                weights = jnp.zeros((bank_size,), point.dtype)
                for power in range(dense_matrix.shape[0]):
                    weights = weights*fraction+dref[power, :]
                value = y+jnp.sum((weights*fraction)[:, None]*bank, axis=0)
                return jnp.where(fraction == 0, y, jnp.where(fraction == 1, endpoint, value))

            def clear(i, _):
                ysout[i, :] = jnp.full((state_size,), jnp.inf, point.dtype)
                tsout[i] = tsref[0]+jnp.inf
                return ()

            jax.lax.fori_loop(0, nsave, clear, ())
            initial_level = level(point)
            status = jnp.where(_kernel_is_finite(initial_level),
                jnp.where(initial_level == 0, 1, initial_status_ref[...]), 4).astype(jnp.int32)

            def initial_save(i):
                ysout[i, :] = point
                tsout[i] = zero
                return i+1

            index = jax.lax.while_loop(lambda i: (status == 0) & (i < nsave) &
                (tsref[jnp.minimum(i, nsave-1)] == 0), initial_save, jnp.asarray(0, jnp.int32))
            indexbank = jnp.arange(bank_size)[:, None]
            # Time, state, FSAL derivative, proposal, event value, status,
            # next save index, trials, accepted trials, rejected trials.
            initial = (zero, point, evaluate(point), initial_h, initial_level, status, index,
                       jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32))

            def advance(carry):
                t, y, f0, next_h, old_value, status, index, trials, accepted, rejected = carry
                remaining = final_t-t
                h = jnp.minimum(next_h if adaptive else initial_h, remaining)
                # 0 <= t <= final_t, so max(abs(t), abs(final_t)) is final_t.
                h = jnp.where(remaining-h <= roundoff, remaining, h)
                end_t = jnp.where(h == remaining, final_t, t+h)
                if not adaptive:
                    end_t = jnp.minimum((accepted+1)*initial_h, final_t)
                    h = end_t-t
                bank = jnp.where(indexbank == 0, h*f0[None, :], jnp.zeros((bank_size, state_size), point.dtype))

                def stage(i, stage_carry):
                    bank, _ = stage_carry
                    yi = y+jnp.sum(aref[i, :][:, None]*bank, axis=0)
                    fi = evaluate(yi)
                    return jnp.where(indexbank == i, h*fi[None, :], bank), fi

                bank, f1 = jax.lax.fori_loop(1, stages, stage, (bank, f0))
                endpoint = y+jnp.sum(aref[stages-1, :][:, None]*bank, axis=0)
                embedded = jnp.sum(eref[...][:, None]*bank, axis=0)
                error = estimate(endpoint, embedded, particle_speed)/budget if adaptive else zero
                new_value = level(endpoint)
                valid = (finite(endpoint) & finite(bank) & finite(f1) & finite(embedded) &
                    _kernel_is_finite(error) & (error >= 0) & _kernel_is_finite(new_value))
                accept = valid & ((error <= 1) | (not adaptive)) & (end_t > t)
                crossing = accept & (((old_value > 0) != (new_value > 0)) | (new_value == 0))
                factor = jnp.where(valid, jnp.clip(.9*jnp.maximum(error, 1e-16)**(-1/order), .2, 5.), .2)
                proposed = h*jnp.where(accept, factor, jnp.minimum(factor, .9))
                stuck = (end_t <= t) | (~accept & ((h <= 0) | (t+proposed <= t)))
                if not adaptive:
                    stuck = stuck | ~valid
                new_status = jnp.where(stuck, jnp.where(valid, 3, 4), 0).astype(jnp.int32)

                def bisect(root_carry):
                    lo, hi, i, valid_root = root_carry
                    mid = (lo+hi)/2
                    value = level(dense(y, endpoint, bank, mid))
                    same = (value > 0) == (old_value > 0)
                    return (jnp.where(same, mid, lo), jnp.where(same, hi, mid), i+1,
                            valid_root & _kernel_is_finite(value))

                lo, hi, _, valid_root = jax.lax.while_loop(lambda c: crossing & c[3] & (c[2] < 52),
                    bisect, (zero, zero+1., jnp.asarray(0, jnp.int32), jnp.asarray(True)))
                fraction = (lo+hi)/2
                stop_t = jnp.where(crossing, t+fraction*h, end_t)
                stop_y = jax.lax.cond(crossing,
                    lambda _: dense(y, endpoint, bank, fraction), lambda _: endpoint, None)
                # A nonfinite interpolant rejects the crossing and retries from
                # the previous state with .2*h, matching the deferred root path.
                root_invalid = crossing & ~valid_root
                accept = accept & ~root_invalid
                proposed = jnp.where(root_invalid, .2*h, proposed)
                root_stuck = (h <= 0) | (t+proposed <= t)
                new_status = jnp.where(crossing & valid_root, 1,
                    jnp.where(root_invalid, jnp.where(root_stuck, 4, 0), new_status)).astype(jnp.int32)

                def save(i):
                    ysout[i, :] = dense(y, endpoint, bank, (tsref[i]-t)/h)
                    tsout[i] = tsref[i]
                    return i+1

                index = jax.lax.while_loop(lambda i: accept & (i < nsave) &
                    (tsref[jnp.minimum(i, nsave-1)] <= stop_t), save, index)
                return (jnp.where(accept, stop_t, t), jnp.where(accept, stop_y, y),
                    jnp.where(accept, f1, f0), proposed, jnp.where(accept, new_value, old_value),
                    new_status, index, trials+1, accepted+accept.astype(jnp.int32),
                    rejected+(~accept).astype(jnp.int32))

            result = jax.lax.while_loop(lambda c: (c[0] < final_t) & (c[5] == 0) &
                (c[7] < max_steps), advance, initial)
            t, y, _, h, _, status, _, trials, accepted, rejected = result
            status = jnp.where((t < final_t) & (status == 0), 2, status)
            yout[...], tout[...], hout[...] = y, t, h
            statsout[0], statsout[1], statsout[2] = trials, accepted, rejected
            statusout[...] = status

        shapes = (jax.ShapeDtypeStruct((nsave, state_size), y0.dtype), jax.ShapeDtypeStruct((nsave,), y0.dtype),
            jax.ShapeDtypeStruct((state_size,), y0.dtype), jax.ShapeDtypeStruct((), y0.dtype),
            jax.ShapeDtypeStruct((), y0.dtype), jax.ShapeDtypeStruct((3,), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32))
        ys, saved, y, t, h, counts, status = pl.pallas_call(kernel, out_shape=shapes, grid=(),
            interpret=interpret, compiler_params=plt.CompilerParams(num_warps=num_warps))(y0, moment, initial_status, *inputs)
        return RKSolution(saved, ys, t, y, status, status == 1,
            RKStats(counts[0], counts[1], counts[2]), jnp.zeros((0,), y0.dtype))

    solve.method = method
    return solve
