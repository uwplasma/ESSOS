"""Compare shared RK execution candidates against Diffrax with the same RHS.

These are experiments, not production solver options. A fixed scan gives all
implementations the same steps, FSAL reuse, error reduction and finite checks.
No saving, event handling or adaptive controller enters the timed region.
--one-step measures a single compiled RK call with an existing FSAL cache,
returning the same state, error, stage increments and next FSAL value.
Direct while-loop diagnostics check forward AD against the public reverse-AD
map; the other experimental kernels also check their own reverse AD.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from examples.benchmarks.benchmark_orbit_integrators import make_case, compile_solve, metadata
from examples.benchmarks.benchmark_native_stage_modes import measure, scales_for
from examples.benchmarks.profile_orbit_stages import make_scan, trial_step
import jax
import jax.numpy as jnp
import numpy as np

import essos.rk as rk
from essos.rk import RKStep, _C, _A, _ERROR, _C8, _A8, _ERROR8


def tableau(nodes, rows, errors):
    n = len(errors)
    coefficients = np.zeros((n + 1, n))
    for i, row in enumerate(rows, 1):
        coefficients[i, :len(row)] = row
    coefficients[-1] = errors
    return coefficients


def scan_accumulate(rhs, t, y, h, args, f0, nodes, rows, errors, *, queue=False):
    """Emit derivatives from scan; carry only future stage sums and endpoint."""
    coefficients = tableau(nodes, rows, errors)
    n = len(errors)
    initial = jnp.asarray(coefficients[1:, 0], dtype=y.dtype)[:, None] * (h * f0)
    if queue:
        updates = np.zeros((n - 1, n))
        for i in range(1, n):
            updates[i - 1, :n - i] = coefficients[i + 1:, i]
    else:
        updates = coefficients[1:, 1:].T
    updates = jnp.asarray(updates, dtype=y.dtype)

    def stage(carry, inputs):
        pending, _ = carry
        i, c, weights = inputs
        trial = y + pending[0 if queue else i]
        derivative = rhs(t + c * h, trial, args)
        if queue:
            pending = jnp.concatenate([pending[1:], jnp.zeros_like(pending[:1])])
        return (pending + weights[:, None] * (h * derivative), trial), derivative

    (pending, endpoint), tail = jax.lax.scan(stage, (initial, y),
        (jnp.arange(n - 1), jnp.asarray(nodes, dtype=y.dtype), updates))
    error = pending[0 if queue else -1]
    return RKStep(endpoint, error, jnp.concatenate([f0[None], tail]))


def switched_step(rhs, t, y, h, args, f0, nodes, rows, errors):
    """A single RHS loop with sparse, constant stage combinations."""
    n = len(errors)
    stages = jnp.zeros((n,) + y.shape, dtype=y.dtype).at[0].set(f0)
    branches = tuple((lambda bank, weights=weights:
        y + h * sum(a * bank[j] for j, a in enumerate(weights) if a != 0)) for weights in rows)
    def stage(i, carry):
        bank, _ = carry
        trial = jax.lax.switch(i, branches, bank)
        derivative = rhs(t + jnp.asarray(nodes, dtype=y.dtype)[i] * h, trial, args)
        return bank.at[i + 1].set(derivative), trial
    stages, endpoint = jax.lax.fori_loop(0, n - 1, stage, (stages, y))
    error = h * sum(a * stages[j] for j, a in enumerate(errors) if a != 0)
    return RKStep(endpoint, error, stages)


def component_first(rhs, t, y, h, args, f0, nodes, rows, errors):
    """Store stage index last, keeping state components contiguous by stage."""
    n = len(errors)
    coefficients = jnp.asarray(tableau(nodes, rows, errors)[1:-1], dtype=y.dtype)
    stages = jnp.zeros(y.shape + (n,), dtype=y.dtype).at[..., 0].set(f0)
    def stage(i, carry):
        bank, _ = carry
        trial = y + h * jnp.sum(coefficients[i] * bank, axis=-1)
        derivative = rhs(t + jnp.asarray(nodes, dtype=y.dtype)[i] * h, trial, args)
        return bank.at[..., i + 1].set(derivative), trial
    stages, endpoint = jax.lax.fori_loop(0, n - 1, stage, (stages, y))
    return RKStep(endpoint, h * jnp.sum(jnp.asarray(errors, dtype=y.dtype) * stages, axis=-1), stages.T)


def scalar_step(rhs, t, y, h, args, f0, nodes, rows, errors, *, differences=False):
    """Use scalar component sums, without packing an array at every operation."""
    stages = [tuple(f0)]
    previous = tuple(y)
    old_weights = ()
    for c, weights in zip(nodes, rows):
        if differences:
            delta = tuple(a - (old_weights[j] if j < len(old_weights) else 0.) for j, a in enumerate(weights))
            trial = tuple(previous[d] + h * sum(a * k[d] for a, k in zip(delta, stages) if a != 0)
                          for d in range(len(y)))
            previous, old_weights = trial, weights
        else:
            trial = tuple(y[d] + h * sum(a * k[d] for a, k in zip(weights, stages) if a != 0)
                          for d in range(len(y)))
        stages.append(tuple(rhs(t + c * h, jnp.stack(trial), args)))
    error = jnp.stack([h * sum(a * k[d] for a, k in zip(errors, stages) if a != 0) for d in range(len(y))])
    return RKStep(jnp.stack(trial), error, jnp.stack([jnp.stack(k) for k in stages]))


def affine_step(rhs, t, y, h, args, f0, nodes, rows, errors, *, left_sum=False):
    """Fold stage contributions without an initial scalar-zero addition."""
    stages = [f0]
    def combine(weights, base=None):
        terms = [(a * k if left_sum else (h * a) * k)
                 for a, k in zip(weights, stages) if a != 0]
        if base is None:
            return sum(terms[1:], terms[0])
        return sum(terms, base)
    for c, weights in zip(nodes, rows):
        trial = y + h * combine(weights) if left_sum else combine(weights, y)
        stages.append(rhs(t + c * h, trial, args))
    error = combine(errors)
    return RKStep(trial, h * error if left_sum else error, jnp.stack(stages))


def padded_step(rhs, t, y, h, args, f0, nodes, rows, errors, *, scaled=False):
    """Align the stage reduction length; keep a static, differentiable loop."""
    n = len(errors)
    size = ((n + 3) // 4) * 4
    weights = np.zeros((n, size))
    for i, row in enumerate(rows):
        weights[i, :len(row)] = row
    weights[-1, :n] = errors
    weights = jnp.asarray(weights, dtype=y.dtype)
    stages = jnp.zeros((size,) + y.shape, dtype=y.dtype).at[0].set(h * f0 if scaled else f0)
    def stage(i, carry):
        bank, _, _ = carry
        increment = jnp.tensordot(weights[i], bank, axes=1)
        trial = y + (increment if scaled else h * increment)
        f = rhs(t + jnp.asarray(nodes, dtype=y.dtype)[i] * h, trial, args)
        return bank.at[i + 1].set(h * f if scaled else f), trial, f
    stages, endpoint, last = jax.lax.fori_loop(0, n - 1, stage, (stages, y, f0))
    error = jnp.tensordot(weights[-1], stages, axes=1)
    if scaled:
        # Experimental only: the public contract is derivatives, not increments.
        # Keep the last derivative exact for FSAL and return the other stages.
        stages = (stages[:n] / h).at[0].set(f0).at[-1].set(last)
    else:
        stages, error = stages[:n], h * error
    return RKStep(endpoint, error, stages)


def slice_step(rhs, t, y, h, args, f0, nodes, rows, errors, *, increments=False):
    """Use contiguous slice writes instead of general indexed scatter updates."""
    n = len(errors)
    coefficients = jnp.asarray(tableau(nodes, rows, errors)[1:-1], dtype=y.dtype)
    bank0 = jnp.concatenate([(h * f0 if increments else f0)[None], jnp.zeros((n - 1,) + y.shape, y.dtype)])
    times = jnp.asarray(nodes, dtype=y.dtype)
    def stage(carry, i):
        bank, _ = carry
        row = jax.lax.dynamic_index_in_dim(coefficients, i, keepdims=False)
        delta = jnp.tensordot(row, bank, axes=1)
        trial = y + (delta if increments else h * delta)
        f = rhs(t + times[i] * h, trial, args)
        bank = jax.lax.dynamic_update_index_in_dim(bank, h * f if increments else f, i + 1, axis=0)
        return (bank, trial), f
    (bank, endpoint), tail = jax.lax.scan(stage, (bank0, y), jnp.arange(n - 1))
    error = jnp.tensordot(jnp.asarray(errors, dtype=y.dtype), bank, axes=1)
    return RKStep(endpoint, error if increments else h * error, jnp.concatenate([f0[None], tail]))


def peeled_step(rhs, t, y, h, args, f0, nodes, rows, errors, *, first=0):
    """Form the endpoint then its FSAL derivative outside the stage loop."""
    n = len(errors)
    coefficients = jnp.asarray(tableau(nodes, rows, errors)[1:-1, :-1], dtype=y.dtype)
    early = [f0]
    for c, weights in zip(nodes[:first], rows[:first]):
        trial = y + h * sum(a * k for a, k in zip(weights, early) if a != 0)
        early.append(rhs(t + c * h, trial, args))
    bank = jnp.concatenate([jnp.stack(early), jnp.zeros((n - 1 - len(early),) + y.shape, y.dtype)])
    def stage(i, bank):
        row = jax.lax.dynamic_index_in_dim(coefficients, i - 1, keepdims=False)
        trial = y + h * jnp.tensordot(row, bank, axes=1)
        f = rhs(t + jnp.asarray(nodes, dtype=y.dtype)[i - 1] * h, trial, args)
        return jax.lax.dynamic_update_index_in_dim(bank, f, i, axis=0)
    bank = jax.lax.fori_loop(first + 1, n - 1, stage, bank)
    endpoint = y + h * sum(a * bank[j] for j, a in enumerate(rows[-1]) if a != 0)
    final = rhs(t + h, endpoint, args)
    stages = jnp.concatenate([bank, final[None]])
    error = h * sum(a * stages[j] for j, a in enumerate(errors) if a != 0)
    return RKStep(endpoint, error, stages)


def while_step(rhs, t, y, h, args, f0, nodes, rows, errors, *, peeled=False, scale_coefficients=False):
    """Forward-only diagnostic of a direct stage loop, without scan outputs."""
    n = len(errors)
    coefficients = jnp.asarray(tableau(nodes, rows, errors)[:-1], dtype=y.dtype)
    if scale_coefficients:
        coefficients = coefficients * h
    times = jnp.asarray((0.,) + nodes, dtype=y.dtype)
    bank = jnp.concatenate([f0[None], jnp.zeros((n - 1,) + y.shape, y.dtype)])
    def stage(carry):
        i, bank, _ = carry
        row = jax.lax.dynamic_index_in_dim(coefficients, i, keepdims=False)
        increment = jnp.tensordot(row, bank, axes=1)
        trial = y + (increment if scale_coefficients else h * increment)
        f = rhs(t + times[i] * h, trial, args)
        bank = jax.lax.dynamic_update_index_in_dim(bank, f, i, axis=0)
        return i + 1, bank, trial
    _, bank, endpoint = jax.lax.while_loop(lambda s: s[0] < n - peeled, stage, (jnp.asarray(1, jnp.int32), bank, y))
    if peeled:
        endpoint = y + (1 if scale_coefficients else h) * jnp.tensordot(coefficients[-1], bank, axes=1)
        bank = jax.lax.dynamic_update_index_in_dim(bank, rhs(t + h, endpoint, args), n - 1, axis=0)
    weights = jnp.asarray(errors, dtype=y.dtype) * (h if scale_coefficients else 1)
    error = jnp.tensordot(weights, bank, axes=1)
    return RKStep(endpoint, error if scale_coefficients else h * error, bank)


def scalar_affine_step(rhs, t, y, h, args, f0, nodes, rows, errors):
    """Fold each scalar stage combination directly into the starting state."""
    stages = [f0]
    for c, weights in zip(nodes, rows):
        scaled = tuple((h * a, k) for a, k in zip(weights, stages) if a != 0)
        components = []
        for d in range(len(y)):
            value = y[d]
            for a, k in scaled:
                value = value + a * k[d]
            components.append(value)
        trial = jnp.stack(components)
        stages.append(rhs(t + c * h, trial, args))
    error = h * sum(a * k for a, k in zip(errors, stages) if a != 0)
    return RKStep(trial, error, jnp.stack(stages))


def bank_only_step(rhs, t, y, h, args, f0, nodes, rows, errors, *, unroll=1):
    """Carry only the derivative bank; reconstruct the endpoint after stages."""
    n = len(errors)
    coefficients = jnp.asarray(tableau(nodes, rows, errors)[1:-1], dtype=y.dtype)
    times = jnp.asarray(nodes, dtype=y.dtype)
    bank = jnp.concatenate([f0[None], jnp.zeros((n - 1,) + y.shape, dtype=y.dtype)])
    def stage(i, bank):
        row = jax.lax.dynamic_index_in_dim(coefficients, i - 1, keepdims=False)
        trial = y + h * jnp.tensordot(row, bank, axes=1)
        derivative = rhs(t + times[i - 1] * h, trial, args)
        return jax.lax.dynamic_update_index_in_dim(bank, derivative, i, axis=0)
    bank = jax.lax.fori_loop(1, n, stage, bank, unroll=unroll)
    endpoint = y + h * sum(a * bank[i] for i, a in enumerate(rows[-1]) if a != 0)
    error = h * sum(a * bank[i] for i, a in enumerate(errors) if a != 0)
    return RKStep(endpoint, error, bank)


def tuple_bank_step(rhs, t, y, h, args, f0, nodes, rows, errors, *, group=1, peeled=False):
    """Store each state component's stage history in its own small vector."""
    n = len(errors)
    coefficients = jnp.asarray(tableau(nodes, rows, errors)[1:-1], dtype=y.dtype)
    times = jnp.asarray(nodes, dtype=y.dtype)
    errors = jnp.asarray(errors, dtype=y.dtype)
    offsets = tuple(range(0, len(y), group))
    initial = tuple(jnp.concatenate([f0[d:d + group, None], jnp.zeros((len(f0[d:d + group]), n - 1), y.dtype)], axis=1)
                    for d in offsets)
    def stage(i, carry):
        bank, _ = carry
        row = jax.lax.dynamic_index_in_dim(coefficients, i - 1, keepdims=False)
        trial = y + h * jnp.concatenate([values @ row for values in bank])
        f = rhs(t + times[i - 1] * h, trial, args)
        bank = tuple(jax.lax.dynamic_update_index_in_dim(values, f[d:d + group], i, axis=1)
                     for d, values in zip(offsets, bank))
        return bank, trial
    bank, endpoint = jax.lax.fori_loop(1, n - peeled, stage, (initial, y))
    if peeled:
        endpoint = y + h * jnp.concatenate([values @ coefficients[-1] for values in bank])
        f = rhs(t + h, endpoint, args)
        bank = tuple(jax.lax.dynamic_update_index_in_dim(values, f[d:d + group], n - 1, axis=1)
                     for d, values in zip(offsets, bank))
    error = h * jnp.concatenate([values @ errors for values in bank])
    return RKStep(endpoint, error, jnp.concatenate(bank, axis=0).T)


def separated_step(rhs, t, y, h, args, f0, nodes, rows, errors, *, inputs=False):
    """Keep materialized stage states or derivatives between RHS evaluations."""
    stages = [f0]
    for c, weights in zip(nodes, rows):
        trial = y + h * sum(a * k for a, k in zip(weights, stages) if a != 0)
        if inputs:
            trial = jax.lax.optimization_barrier(trial)
        f = rhs(t + c * h, trial, args)
        stages.append(f if inputs else jax.lax.optimization_barrier(f))
    error = h * sum(a * k for a, k in zip(errors, stages) if a != 0)
    return RKStep(trial, error, jnp.stack(stages))


def scalar_history_step(rhs, t, y, h, args, f0, nodes, rows, errors, *, peeled=False, padded=False):
    """One vector per state component, with an optional separate FSAL stage."""
    n = len(errors)
    size = (n + 3) // 4 * 4 if padded else n
    weights = np.zeros((n - 1, size))
    for i, row in enumerate(rows):
        weights[i, :len(row)] = row
    weights = jnp.asarray(weights, dtype=y.dtype)
    times = jnp.asarray(nodes, dtype=y.dtype)
    errors = jnp.asarray(tuple(errors) + (0.,) * (size - n), dtype=y.dtype)
    initial = tuple(jnp.concatenate([f0[d:d + 1], jnp.zeros(size - 1, y.dtype)]) for d in range(len(y)))
    def point(row, bank):
        return jnp.stack([y[d] + h * jnp.dot(row, values) for d, values in enumerate(bank)])
    def update(bank, f, i):
        return tuple(jax.lax.dynamic_update_index_in_dim(values, f[d], i, axis=0)
                     for d, values in enumerate(bank))
    def stage(i, carry):
        bank, _ = carry
        row = jax.lax.dynamic_index_in_dim(weights, i - 1, keepdims=False)
        trial = point(row, bank)
        return update(bank, rhs(t + times[i - 1] * h, trial, args), i), trial
    bank, endpoint = jax.lax.fori_loop(1, n - peeled, stage, (initial, y))
    if peeled:
        endpoint = point(weights[-1], bank)
        bank = update(bank, rhs(t + h, endpoint, args), n - 1)
    error = h * jnp.stack([jnp.dot(errors, values) for values in bank])
    return RKStep(endpoint, error, jnp.stack(bank, axis=1)[:n])


def candidate_kernel(name):
    if name.startswith("trial_"):
        from examples.benchmarks.profile_rk_increment_storage import increment_kernel
        return increment_kernel(name)
    if name.startswith("scalar_history"):
        return lambda *args: scalar_history_step(*args, peeled="peeled" in name, padded="padded" in name)
    if name.startswith("group"):
        return lambda *args: tuple_bank_step(*args, group=int(name.removeprefix("group")))
    if name == "tuple_peeled":
        return lambda *args: tuple_bank_step(*args, peeled=True)
    if name == "tuple_bank":
        return tuple_bank_step
    if name in ("separate_input", "separate_output"):
        return lambda *args: separated_step(*args, inputs=name == "separate_input")
    if name.startswith("bank"):
        return lambda *args: bank_only_step(*args, unroll=int(name.removeprefix("bank")))
    if name == "scalar_affine":
        return scalar_affine_step
    if name in ("affine", "left_sum"):
        return lambda *args: affine_step(*args, left_sum=name == "left_sum")
    if name.startswith("rolled") or name in ("dot", "increments", "pairwise", "pairwise_increments"):
        return lambda rhs, t, y, h, args, f0, *constants: trial_step(
            name, lambda t, y, unused: rhs(t, y, args), t, y, h, f0, *constants)
    if name.startswith("while"):
        return lambda *args: while_step(*args, peeled="peeled" in name, scale_coefficients="scaled" in name)
    if name.startswith("peeled"):
        return lambda *args: peeled_step(*args, first=int(name.removeprefix("peeled")))
    if name == "queue":
        return lambda *args: scan_accumulate(*args, queue=True)
    if name == "differences":
        return lambda *args: scalar_step(*args, differences=True)
    if name == "padded_scaled":
        return lambda *args: padded_step(*args, scaled=True)
    if name == "slice_increments":
        return lambda *args: slice_step(*args, increments=True)
    return {"scan_accumulate": scan_accumulate, "switch": switched_step,
            "component_first": component_first, "scalar": scalar_step, "padded": padded_step,
            "slice": slice_step}[name]


def make_candidate_scan(case, method, name, steps, *, batch_size=None):
    kernel = candidate_kernel(name)
    constants = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    rhs, h = case["rhs"], case["maxtime"] / steps
    def single(y0):
        def advance(carry, index):
            y, f, errors, finite = carry
            out = kernel(rhs, index * h, y, h, None, f, *constants)
            bank = out.increments if name.startswith("trial_") else out.stages
            final = out.f1 if name.startswith("trial_") else out.stages[-1]
            return (out.y, final, errors + jnp.sum(out.error**2),
                    finite & jnp.isfinite(bank).all()), None
        initial = y0, rhs(0., y0, None), jnp.zeros((), y0.dtype), jnp.asarray(True)
        (y, _, errors, finite), _ = jax.lax.scan(advance, initial, jnp.arange(steps))
        return y, errors, finite
    if batch_size is None:
        return jax.jit(jax.vmap(single))
    return jax.jit(lambda ys: jax.lax.map(single, ys, batch_size=batch_size))


def check_candidate(method, name):
    constants = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    candidate = candidate_kernel(name)
    def observe(x, kernel):
        rhs = lambda t, y, args: x[4] * jnp.array([jnp.sin(y[1] + t), -y[0]**2 + jnp.cos(t)])
        out = kernel(rhs, x[2], x[:2], x[3], None, x[5:], *constants)
        if name.startswith("trial_"):
            increments = out.increments if isinstance(out, rk._RKTrial) else x[3] * out.stages
            final = out.f1 if isinstance(out, rk._RKTrial) else out.stages[-1]
            return jnp.concatenate([out.y, out.error, increments.ravel(), final])
        return jnp.concatenate([out.y, out.error, out.stages.ravel()])
    initial = jnp.array([.7, -.3, .2, .1, .8, -.08, .392])
    before = lambda x: observe(x, rk._rk_step)
    after = lambda x: observe(x, candidate)
    np.testing.assert_allclose(after(initial), before(initial), rtol=2e-12, atol=2e-12)
    reference = jax.jacfwd(before)(initial)
    derivatives = ((jax.jacfwd(after),) if name.startswith("while") else
                   (jax.jacfwd(after), jax.jacrev(after)))
    for derivative in derivatives:
        np.testing.assert_allclose(derivative(initial), reference, rtol=2e-10, atol=2e-12)


def make_one_step(case, method, name):
    """Use identical input/output contracts, including external FSAL input."""
    import diffrax
    constants = (_C, _A, _ERROR) if method == "dopri5" else (_C8, _A8, _ERROR8)
    rhs, h = case["rhs"], case["dt0"]
    solver = diffrax.Dopri5() if method == "dopri5" else diffrax.Dopri8()
    term = diffrax.ODETerm(rhs)
    def single(inputs):
        y, f0 = inputs
        if name == "diffrax":
            y1, error, dense, state, _ = solver.step(term, 0., h, y, None, (False, f0), False)
            return y1, error, dense["k"], state[1]
        kernel = rk._rk_step if name == "sparse" else candidate_kernel(name)
        out = kernel(rhs, 0., y, h, None, f0, *constants)
        if name.startswith("trial_"):
            return out.y, out.error, out.increments, out.f1
        return out.y, out.error, h * out.stages, out.stages[-1]
    return jax.jit(jax.vmap(single))


def measure_blocks(executables, initial, repeats, rng, block_calls):
    """Warm each implementation's instruction/data caches before its block."""
    samples = {name: [] for name in executables}
    for _ in range(repeats):
        for name in rng.permutation(tuple(executables)):
            executable = executables[name]
            for _ in range(2):
                jax.block_until_ready(executable(initial))
            start = perf_counter()
            for _ in range(block_calls):
                jax.block_until_ready(executable(initial))
            samples[name].append(1000 * (perf_counter() - start) / block_calls)
    return {name: dict(samples_ms=values, median_ms=float(np.median(values)),
                       q25_ms=float(np.quantile(values, .25)), q75_ms=float(np.quantile(values, .75)))
            for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", nargs="+", default=["sparse", "scan_accumulate", "queue", "switch", "component_first", "diffrax"])
    parser.add_argument("--models", nargs="+", default=["gc", "lorentz"])
    parser.add_argument("--methods", nargs="+", default=["dopri5", "dopri8"])
    parser.add_argument("--field", default="qa")
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hlo-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 vmaps all particles; a positive size tiles with lax.map equally for all kernels")
    parser.add_argument("--one-step", action="store_true")
    parser.add_argument("--calls-per-block", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "rk_candidates.json")
    options = parser.parse_args()
    if options.one_step:
        options.steps = 1
    options.tolerance, options.accuracy_limit = None, None
    report = dict(metadata=metadata(options), steps=options.steps, rows=[])
    report["metadata"]["scope"] = __doc__
    report["metadata"]["particle_batch_size"] = options.batch_size or "all"
    report["metadata"]["one_step_external_fsal"] = options.one_step
    report["metadata"]["calls_per_block"] = options.calls_per_block
    report["metadata"]["warm_calls_before_block"] = 2 if options.calls_per_block > 1 else 0
    report["metadata"]["source_sha256"]["examples/benchmarks/profile_orbit_stages.py"] = hashlib.sha256(Path(__file__).with_name("profile_orbit_stages.py").read_bytes()).hexdigest()
    report["metadata"]["source_sha256"]["examples/benchmarks/profile_rk_candidates.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rng = np.random.default_rng(125)
    for method in options.methods:
        for name in options.kernels:
            if name not in ("sparse", "diffrax"):
                print(f"{method}/{name}: derivative checks", flush=True)
                check_candidate(method, name)
    for model in options.models:
        case = make_case(model, options.particles, options.field)
        initial = ((case["y0"], jax.vmap(lambda y: case["rhs"](0., y, None))(case["y0"]))
                   if options.one_step else case["y0"])
        for method in options.methods:
            executables, outputs, rows = {}, {}, {}
            for name in options.kernels:
                print(f"{model}/{method}/{name}: compiling", flush=True)
                batch_size = options.batch_size or None
                fn = (make_one_step(case, method, name) if options.one_step else
                      make_scan(case, method, name, options.steps, batch_size=batch_size) if name in ("sparse", "diffrax") else
                      make_candidate_scan(case, method, name, options.steps, batch_size=batch_size))
                executable, output, timing = compile_solve(fn, initial)
                executables[name], outputs[name] = executable, output
                rows[name] = dict(model=model, method=method, kernel=name, **timing,
                                  temporary_bytes=executable.memory_analysis().temp_size_in_bytes,
                                  cost_analysis=executable.cost_analysis(),
                                  derivative_map="public unrolled map" if name.startswith("while") else "ordinary JAX AD")
                if options.hlo_dir is not None:
                    options.hlo_dir.mkdir(parents=True, exist_ok=True)
                    (options.hlo_dir / f"{model}_{method}_{name}.txt").write_text(executable.as_text())
            reference = np.asarray(outputs["diffrax"][0])
            for name, output in outputs.items():
                if options.one_step:
                    state, error, bank, fsal = jax.device_get(output)
                    finite = np.isfinite(bank).all() and np.isfinite(fsal).all()
                    for index, (candidate, baseline) in enumerate(zip(output, outputs["diffrax"])):
                        factor = case["dt0"] if index == 3 else 1.
                        np.testing.assert_allclose(candidate * factor / scales_for(case), baseline * factor / scales_for(case), rtol=0., atol=1e-10)
                else:
                    state, error, finite = jax.device_get(output)
                if not np.all(finite) or not np.isfinite(state).all() or not np.isfinite(error).all():
                    raise RuntimeError("nonfinite output")
                difference = np.abs(state - reference)
                rows[name]["position_difference_m"] = float(np.max(difference[:, :3]))
                rows[name]["velocity_difference_over_speed"] = float(np.max(difference[:, 3:]) / case["speed"])
                np.testing.assert_allclose(state / scales_for(case), reference / scales_for(case), rtol=0., atol=1e-10)
            timings = (measure_blocks(executables, initial, options.repeats, rng, options.calls_per_block)
                       if options.calls_per_block > 1 else measure(executables, initial, options.repeats, rng))
            for name in options.kernels:
                rows[name].update(timings[name])
                report["rows"].append(rows[name])
                print(f"  {name:18s} {rows[name]['median_ms']:.3f} ms", flush=True)
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
