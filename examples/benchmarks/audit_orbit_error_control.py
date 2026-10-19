"""Audit actual native trials, endpoint/dense errors, and tolerance convergence.

This is a diagnostic, not a timing benchmark: ordered host callbacks observe
the unmodified trial step. Fine fixed Diffrax Dopri8 solutions independently
check local defects and accumulated errors. Both reference resolutions must
agree. Field classes, controllers, tableaus and error policies are unchanged.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from examples.benchmarks.benchmark_orbit_integrators import make_case, metadata
import diffrax
import jax
import jax.numpy as jnp
import numpy as np

import essos.rk as rk
from essos.orbit_error import OrbitErrorNorm


def block_errors(error, case):
    error = np.asarray(error)
    return np.maximum(np.linalg.norm(error[..., :3], axis=-1),
                      np.linalg.norm(error[..., 3:], axis=-1) / case["speed"])


def errors(error, case):
    error = np.asarray(error)
    if not np.isfinite(error).all():
        raise AssertionError("Non-finite reference/trajectory error")
    return dict(position_component_max_m=float(np.max(np.abs(error[..., :3]))),
                position_norm_max_m=float(np.max(np.linalg.norm(error[..., :3], axis=-1))),
                velocity_component_max_m_s=float(np.max(np.abs(error[..., 3:]))),
                velocity_component_max_over_speed=float(np.max(np.abs(error[..., 3:])) / case["speed"]),
                velocity_norm_max_over_speed=float(np.max(np.linalg.norm(error[..., 3:], axis=-1)) / case["speed"]))


def global_reference(case, steps):
    solver, term = diffrax.Dopri8(), diffrax.ODETerm(case["rhs"])
    def single(y):
        return diffrax.diffeqsolve(term, solver, t0=0., t1=case["maxtime"], dt0=case["maxtime"] / steps,
                                  y0=y, stepsize_controller=diffrax.ConstantStepSize(),
                                  saveat=diffrax.SaveAt(ts=case["times"], dense=True), max_steps=steps + 2)
    result = jax.jit(jax.vmap(single))(case["y0"])
    jax.block_until_ready(result)
    return result


def local_reference(case, times, states, sizes, steps):
    solver, term = diffrax.Dopri8(), diffrax.ODETerm(case["rhs"])
    def single(inputs):
        t, y, h = inputs
        initial = (y, (False, case["rhs"](t, y, None)), y)
        def advance(i, carry):
            y, state, mid = carry
            start, end = t + h * (i / steps), t + h * ((i + 1) / steps)
            y, _, _, state, _ = solver.step(term, start, end, y, None, state, False)
            return y, state, jnp.where(i + 1 == steps // 2, y, mid)
        final, _, middle = jax.lax.fori_loop(0, steps, advance, initial)
        return middle, final
    return jax.jit(lambda t, y, h: jax.lax.map(single, (t, y, h), batch_size=32))(times, states, sizes)


def matching_diffrax_steps(case, method, times, states, sizes):
    solver = diffrax.Dopri5() if method == "dopri5" else diffrax.Dopri8()
    term = diffrax.ODETerm(case["rhs"])
    def single(inputs):
        t, y, h = inputs
        state = (False, case["rhs"](t, y, None))
        end, error, dense, _, _ = solver.step(term, t, t + h, y, None, state, False)
        middle = solver.interpolation_cls(t0=t, t1=t + h, **dense).evaluate(t + .5 * h)
        return end, error, middle
    return jax.jit(lambda t, y, h: jax.lax.map(single, (t, y, h), batch_size=32))(times, states, sizes)


@contextmanager
def record_trials(case, method, norm, observations):
    original = rk._trial_step
    dense = rk._rk45_dense if method == "dopri5" else rk._dopri8_dense
    def capture(index, t, y, h, end, estimate, middle, ratio):
        observations[int(index)].append((float(t), np.asarray(y), float(h), np.asarray(end),
                                         np.asarray(estimate), np.asarray(middle), float(ratio)))
    def audited(rhs, t, y, h, args, f0, nodes, rows, weights):
        trial = original(rhs, t, y, h, args, f0, nodes, rows, weights)
        middle = dense(y, trial.y, trial.increments, jnp.asarray(.5, y.dtype))
        jax.debug.callback(capture, args, t, y, h, trial.y, trial.error, middle,
                           norm(y, trial.y, trial.error), ordered=True)
        return trial
    rk._trial_step = audited
    try:
        yield
    finally:
        rk._trial_step = original


def audit(case, method, tolerance, reference):
    norm = OrbitErrorNorm(tolerance, tolerance, case["speed"], 1., case["velocity_components"])
    observed = {i: [] for i in range(len(case["y0"]))}
    def solve(index, y):
        return rk.solve_rk(case["rhs"], y, index, t1=case["maxtime"], dt0=case["dt0"],
                           ts=case["times"], method=method, error_norm=norm,
                           rtol=tolerance, atol=tolerance, record_steps=True, max_steps=2048)
    with record_trials(case, method, norm, observed):
        compiled = jax.jit(solve).lower(jnp.asarray(0), case["y0"][0]).compile()
    solutions = []
    for index, y in enumerate(case["y0"]):
        result = jax.block_until_ready(compiled(jnp.asarray(index), y))
        jax.effects_barrier()
        if int(result.status) != rk.RKStatus.SUCCESS:
            raise RuntimeError(f"Native solve failed: {result.status}")
        solutions.append(result)
    accepted, rejected, counts = [], [], []
    for index, solution in enumerate(solutions):
        attempts = observed[index]
        good = [x for x in attempts if np.isfinite(x[6]) and x[6] <= 1.]
        bad = [x for x in attempts if not np.isfinite(x[6]) or x[6] > 1.]
        count = tuple(int(x) for x in solution.stats)
        if count != (len(attempts), len(good), len(bad)):
            raise AssertionError(f"Trial decision/count mismatch: {index}, {count}, {len(good)}, {len(bad)}")
        accepted.extend((index, *x) for x in good)
        rejected.extend((index, *x) for x in bad)
        counts.append(count)
    ratios = np.array([x[-1] for x in accepted])
    estimates = np.stack([x[5] for x in accepted])
    independent_ratios = block_errors(estimates, case) / (2 * tolerance)
    np.testing.assert_allclose(ratios, independent_ratios, rtol=3e-14, atol=3e-14)
    t = jnp.asarray([x[1] for x in accepted])
    y = jnp.asarray(np.stack([x[2] for x in accepted]))
    h = jnp.asarray([x[3] for x in accepted])
    middle8, end8 = local_reference(case, t, y, h, 8)
    middle16, end16 = local_reference(case, t, y, h, 16)
    middle8, end8, middle16, end16 = [np.asarray(x) for x in (middle8, end8, middle16, end16)]
    local_reference_difference = max(float(block_errors(end8 - end16, case).max()),
                                     float(block_errors(middle8 - middle16, case).max()))
    end = np.stack([x[4] for x in accepted])
    middle = np.stack([x[6] for x in accepted])
    matching_end, matching_error, matching_mid = matching_diffrax_steps(case, method, t, y, h)
    matching = dict(endpoint_max_scaled=float(block_errors(end - matching_end, case).max()),
                    embedded_max_scaled=float(block_errors(estimates - matching_error, case).max()),
                    midpoint_max_scaled=float(block_errors(middle - matching_mid, case).max()))
    if max(matching.values()) > 1e-11:
        raise AssertionError(f"Native and Diffrax maps differ on identical trials: {matching}")
    endpoint_defect = block_errors(end - end16, case)
    midpoint_defect = block_errors(middle - middle16, case)
    if local_reference_difference > .1 * tolerance:
        raise AssertionError(f"Local reference did not converge: {local_reference_difference=}")
    embedded = block_errors(estimates, case)
    resolved = embedded > max(100 * local_reference_difference, .001 * (2 * tolerance))
    query = jnp.asarray([x[1] + x[3] for x in accepted])
    owners = jnp.asarray([x[0] for x in accepted])
    def evaluate_endpoints():
        return jax.lax.map(lambda inp: jax.tree.map(lambda x: x[inp[0]], reference).evaluate(inp[1]),
                           (owners, query), batch_size=32)
    reference_endpoints = np.asarray(jax.jit(evaluate_endpoints)())
    saved = np.stack([np.asarray(x.ys) for x in solutions])
    output = dict(tolerance=tolerance, counts_per_particle=counts,
                  accepted_mean=float(np.mean([x[1] for x in counts])),
                  rejected_mean=float(np.mean([x[2] for x in counts])),
                  accepted_ratio_max=float(ratios.max()), accepted_ratio_median=float(np.median(ratios)),
                  rejected_ratio_min=float(min((x[-1] for x in rejected), default=np.inf)) if rejected else None,
                  acceptance_violations=int(np.sum(ratios > 1)),
                  same_step_diffrax_agreement=matching,
                  local_reference_difference_scaled=local_reference_difference,
                  local_endpoint_defect_over_budget_max=float(endpoint_defect.max() / (2 * tolerance)),
                  local_midpoint_defect_over_budget_max=float(midpoint_defect.max() / (2 * tolerance)),
                  local_endpoint_true_over_embedded_max_resolved=float(np.max(endpoint_defect[resolved] / embedded[resolved])) if resolved.any() else None,
                  saved_grid_errors=errors(saved - np.asarray(reference.ys), case),
                  accepted_endpoint_errors=errors(end - reference_endpoints, case),
                  local_endpoint_errors=errors(end - end16, case), local_midpoint_errors=errors(middle - middle16, case))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("gc", "lorentz"), default=["gc", "lorentz"])
    parser.add_argument("--methods", nargs="+", choices=("dopri5", "dopri8"), default=["dopri5", "dopri8"])
    parser.add_argument("--tolerances", nargs="+", type=float, default=[1e-7, 1e-9, 1e-11])
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "orbit_error_audit.json")
    options = parser.parse_args()
    options.field, options.repeats, options.tolerance, options.accuracy_limit = "qa", None, None, None
    options.error_control, options.length_scale = "physics", 1.
    report = dict(metadata=metadata(options), rows=[], reference_checks=[])
    report["metadata"]["scope"] = __doc__
    report["metadata"]["source_sha256"]["examples/benchmarks/audit_orbit_error_control.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["metadata"]["global_reference"] = "fixed Diffrax Dopri8: 1024 vs 2048 steps; use 2048"
    report["metadata"]["local_reference"] = "fixed Diffrax Dopri8: 8 vs 16 subdivisions from actual trial starts; use 16"
    options.output.parent.mkdir(parents=True, exist_ok=True)
    for model in options.models:
        case = make_case(model, options.particles, "qa")
        print(f"{model}: fixed reference convergence", flush=True)
        reference_coarse, reference = global_reference(case, 1024), global_reference(case, 2048)
        difference = np.asarray(reference.ys) - np.asarray(reference_coarse.ys)
        report["reference_checks"].append(dict(model=model, **errors(difference, case)))
        if float(block_errors(difference, case).max()) > .05 * min(options.tolerances):
            raise AssertionError("Global fixed-step references did not converge sufficiently")
        for method in options.methods:
            for tolerance in options.tolerances:
                print(f"{model}/{method}/{tolerance:.0e}: auditing actual trials", flush=True)
                row = dict(model=model, method=method, **audit(case, method, tolerance, reference))
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
                options.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {options.output}", flush=True)


if __name__ == "__main__":
    main()
