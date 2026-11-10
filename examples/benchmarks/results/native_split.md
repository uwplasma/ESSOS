# Native split orbit and collisions: validation and timing

`essos.split.trace_split` advances a native Dopri RK orbit, then applies one Milstein
collision map per interval (Lie splitting), with Brownian bridge memory so rejected steps
never redraw the path. The collision and moment tolerances are local controls, not global
error bounds. Population studies are not part of CI (CI uses `tests/test_native_split.py`).

## Collision operator

`python -m examples.benchmarks.benchmark_native --suite collisions` runs the production
collision maps and Brownian driver (guiding-center and full-Lorentz) in a fixed 1 keV proton
bath and compares them with an independent NumPy/SciPy finite-volume Fokker-Planck
generator that uses no SDE solver:

* `--study thermal`: relaxation of a monoenergetic beam (speed `sqrt(10) sigma`, pitch 0.6).
  Speed and pitch means and variances at 40 times are tested against the reference with
  Student-t standard errors; the run reports the largest |z| against a familywise
  critical value and the mesh refinement change of the reference.
* `--study passage`: first passage of the speed from 5 T to 2 T. The mean, variance and
  Kolmogorov-Smirnov distance of the passage times are compared with the absorbing-boundary
  reference.
* `--law constant` fixes the Coulomb logarithm (exact Maxwell equilibrium); `--law impact`
  uses the production impact-parameter law.

These are statistical-consistency screens for a finite ensemble, not numerical-accuracy
bounds. The 2,000-particle thermalization comparison recorded for the README:

![Thermalization](../../../docs/readme_split_thermalization.png)

CI checks the same generator analytically: Lorentz pitch-angle moments, Maxwellian
stationarity, the Einstein relation, Brownian-bridge conditioning and the exact first-order
splitting error. The earlier coupled mirror/helix/uniform-field population studies were
removed with their scripts; their results are not reproduced here.

## Timing record (local RTX 5060 laptop GPU, JAX 0.11.2)

Single cold measurements from the previous per-case scripts, fresh process, persistent
cache disabled; 2 alpha particles, 1 ns, adaptive Dopri8 with the moment predictor
(`moment_tolerance=.005`). These are small-population startup costs, not throughput.

| Field / model | Execution | Lowering s | Backend compile s | First execution s |
|---|---|---:|---:|---:|
| Coils / GC | shared JAX | 2.91 | 8.79 | 0.039 |
| Coils / GC | independent GPU | 5.14 | 65.32 | 0.093 |
| Coils / Lorentz | shared JAX | 2.82 | 8.81 | 0.028 |
| Coils / Lorentz | independent GPU | 6.97 | 346.09 | 0.108 |
| VMEC / GC | shared JAX | 9.25 | 13.63 | 0.041 |
| VMEC / GC | independent GPU | 15.89 | 260.03 | 0.041 |
| Boozer / GC | shared JAX | 3.22 | 10.26 | 0.025 |
| Boozer / GC | independent GPU | 5.07 | 66.88 | 0.039 |

A warm-throughput comparison of the independent kernel against shared execution at
production particle counts is still needed to justify its compile cost.

## Reproduction

```sh
python -m examples.benchmarks.benchmark_native --suite collisions --study thermal -N 2000 --out output/thermal
python -m examples.benchmarks.benchmark_native --suite collisions --study passage -N 2000 \
  --law impact --out output/passage
python -m examples.benchmarks.benchmark_native --suite split --field coils --model gc -N 64 \
  --execution both --out output/timing
```

Reports stay in the ignored `output/` directory.
