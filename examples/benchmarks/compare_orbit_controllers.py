"""Compare native I, native PI and Diffrax PI on identical orbit problems.

Run: python -m examples.benchmarks.compare_orbit_controllers
The PI coefficients, safety/growth bounds, physical error norm and dt0 match.
Errors are maximum saved Cartesian components against fixed-step Dopri8, and
do not represent the local embedded-error acceptance test.
"""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .benchmark_orbit_integrators import check_output, compile_solve, make_case, make_solve, metadata

import jax
import numpy as np


def write_report(report, path):
    path.write_text(json.dumps(report, indent=2) + "\n")
    columns = ("model", "method", "implementation", "rtol", "atol", "physical_error_budget", "dense_output",
               "accepted_mean", "rejected_mean", "warm_median_ms", "position_error_m",
               "velocity_error_m_s", "velocity_error_over_speed")
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--field", choices=("qa", "small"), default="qa")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--native-i-tolerances", type=float, nargs=2, action="append", default=[],
                        metavar=("RTOL", "ATOL"),
                        help="Append a native I case with explicit rtol/atol; may be repeated.")
    parser.add_argument("--pcoeff", type=float, default=.4)
    parser.add_argument("--icoeff", type=float, default=.3)
    parser.add_argument("--compare-dense-output", action="store_true",
                        help="Append native I with coefficient-only Dopri8 save interpolation.")
    parser.add_argument("--compare-higher-order", action="store_true",
                        help="Append native I with extra output stages (DP5 order 5, PD8 order 7).")
    parser.add_argument("--models", nargs="+", choices=("gc", "lorentz"), default=["gc", "lorentz"])
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "output" / "orbit_controller_comparison.json")
    options = parser.parse_args()
    if options.particles < 1 or options.repeats < 3:
        parser.error("use at least one particle and three timing repeats")
    if not (np.isfinite(options.tolerance) and options.tolerance > 0):
        parser.error("tolerance must be finite and positive")
    for rtol, atol in options.native_i_tolerances:
        if not (np.isfinite(rtol) and rtol >= 0 and np.isfinite(atol) and atol > 0):
            parser.error("native I requires finite rtol >= 0 and atol > 0")
    if not (np.isfinite(options.pcoeff) and options.pcoeff >= 0 and
            np.isfinite(options.icoeff) and options.icoeff > 0):
        parser.error("PI requires finite pcoeff >= 0 and icoeff > 0")
    options.error_control, options.length_scale, options.accuracy_limit = "physics", 1., 1e-6
    report = dict(metadata=metadata(options), rows=[], comparisons=[])
    report["metadata"].update(
        adaptive_controllers={"native_i": "original I", "native_pi": "PI", "diffrax_pi": "PI"},
        pcoeff=options.pcoeff, icoeff=options.icoeff, safety=.9,
        pi_accepted_factor_bounds=[1., 10.], pi_rejected_factor_bounds=[.2, .9],
        reference="Diffrax Dopri8; 1024 fixed steps; identical saved times",
        reference_tolerance=None, reference_steps=1024,
        native_i_extra_tolerances=[dict(rtol=r, atol=a) for r, a in options.native_i_tolerances],
        compare_dense_output=options.compare_dense_output,
        compare_higher_order=options.compare_higher_order,
        memory_scope="XLA forward executable buffers; not process RSS or AD memory")
    report["metadata"]["source_sha256"]["examples/benchmarks/compare_orbit_controllers.py"] = (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for relative in ("essos/rk_dense.py", "examples/benchmarks/derive_rk_dense_output.py",
                     "essos/rk_dense_high.py", "examples/benchmarks/derive_rk_higher_dense_output.py"):
        report["metadata"]["source_sha256"][relative] = hashlib.sha256(
            (Path(__file__).resolve().parents[2] / relative).read_bytes()).hexdigest()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    configurations = [(label, backend, controller, options.tolerance, options.tolerance, "standard")
                      for label, backend, controller in
                      (("native_i", "native", "i"), ("native_pi", "native", "pi"),
                       ("diffrax_pi", "diffrax", "pi"))]
    configurations.extend((f"native_i_rtol{rtol:g}_atol{atol:g}", "native", "i", rtol, atol, "standard")
                          for rtol, atol in options.native_i_tolerances)
    if options.compare_dense_output:
        configurations.append(("native_i_optimized", "native", "i", options.tolerance, options.tolerance, "optimized"))
    if options.compare_higher_order:
        configurations.append(("native_i_higher_order", "native", "i", options.tolerance, options.tolerance, "higher_order"))
    if len({item[0] for item in configurations}) != len(configurations):
        parser.error("extra native I tolerance pairs must be distinct")
    for model in options.models:
        case = make_case(model, options.particles, options.field)
        reference_case = dict(case, dt0=case["maxtime"] / 1024)
        print(f"{model}: compiling fixed-step accuracy reference...", flush=True)
        reference_exe, reference_out, _ = compile_solve(
            make_solve(reference_case, "diffrax_dopri8", False, options.tolerance), case["y0"])
        reference, _ = check_output(reference_out, case["maxtime"])
        del reference_exe, reference_out
        for method in ("dopri5", "dopri8"):
            executables, rows, outputs = {}, {}, {}
            for label, backend, controller, rtol, atol, dense_output in configurations:
                print(f"{model}/{method}/{label}: compiling and checking...", flush=True)
                executable, output, timing = compile_solve(make_solve(
                    case, f"{backend}_{method}", True, options.tolerance, "physics",
                    native_controller=controller, pcoeff=options.pcoeff, icoeff=options.icoeff,
                    rtol=rtol, atol=atol, native_dense_output=dense_output), case["y0"])
                trajectory, counts = check_output(output, case["maxtime"])
                error = np.abs(trajectory - reference)
                analysis = executable.memory_analysis()
                memory = None
                if analysis is not None:
                    memory = {name: int(getattr(analysis, name)) for name in (
                        "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
                    memory["total_buffer_bytes"] = (memory["argument_size_in_bytes"] + memory["output_size_in_bytes"]
                                                    + memory["temp_size_in_bytes"] - memory["alias_size_in_bytes"])
                rows[label] = dict(model=model, method=method, implementation=label, **timing,
                                   rtol=rtol, atol=atol, physical_error_budget=rtol + atol,
                                   dense_output=dense_output,
                                   maxtime=case["maxtime"], dt0=case["dt0"], saves=len(case["times"]),
                                   speed_m_s=case["speed"],
                                   counts_per_particle=counts.tolist(),
                                   accepted_mean=float(counts[:, 1].mean()),
                                   rejected_mean=float(counts[:, 2].mean()),
                                   position_error_m=float(error[..., :3].max()),
                                   velocity_error_m_s=float(error[..., 3:].max()),
                                   velocity_error_over_speed=float(error[..., 3:].max() / case["speed"]),
                                   compiled_memory=memory,
                                   warm_samples_s=[])
                executables[label], outputs[label] = executable, (trajectory, counts, np.asarray(output[1]))
            for variant in ("native_i_optimized", "native_i_higher_order"):
                if variant not in outputs:
                    continue
                np.testing.assert_array_equal(outputs["native_i"][1], outputs[variant][1])
                scales = np.array([1.] * 3 + [case["speed"]] * case["velocity_components"])
                np.testing.assert_allclose(outputs["native_i"][2]/scales,
                                           outputs[variant][2]/scales, rtol=0., atol=3e-13)
            for executable in executables.values():
                jax.block_until_ready(executable(case["y0"]))
            for _ in range(options.repeats):
                for label in rng.permutation(list(executables)):
                    start = perf_counter()
                    jax.block_until_ready(executables[label](case["y0"]))
                    rows[label]["warm_samples_s"].append(perf_counter() - start)
            difference = np.abs(outputs["native_pi"][0] - outputs["diffrax_pi"][0])
            comparison = dict(model=model, method=method,
                              pi_counts_match_per_particle=bool(np.array_equal(
                                  outputs["native_pi"][1], outputs["diffrax_pi"][1])),
                              pi_max_position_difference_m=float(difference[..., :3].max()),
                              pi_max_velocity_difference_m_s=float(difference[..., 3:].max()))
            report["comparisons"].append(comparison)
            for label, *_ in configurations:
                row = rows[label]
                row["warm_median_ms"] = float(1000 * np.median(row["warm_samples_s"]))
                report["rows"].append(row)
                print(f"  {label}: {row['accepted_mean']:.3f} accepted, {row['rejected_mean']:.3f} rejected; "
                      f"{row['warm_median_ms']:.3f} ms; max saved errors "
                      f"{row['position_error_m']:.3e} m, {row['velocity_error_m_s']:.3e} m/s", flush=True)
            print(f"  PI counts match for every particle: {comparison['pi_counts_match_per_particle']}", flush=True)
            write_report(report, options.output)
            executables.clear()
            del executable, output
            jax.clear_caches()
            gc.collect()
    print(f"Saved {options.output}", flush=True)


if __name__ == "__main__":
    main()
