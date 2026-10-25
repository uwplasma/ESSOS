# Native forward orbit comparison

These records belong to the forward-only native orbit PR. They contain no
collision stepping, reverse AD, particle chunking or compaction. The private
GPU adapters leave the public field classes intact.
The implementation, including those adapters, lives in `essos/rk.py`.

`forward.csv` and the eight per-workload JSON reports contain fresh local
measurements on an RTX 5060 Laptop GPU/JAX 0.11.2. Each report compares identical
births, fields, save grids and physical/chart error blocks. Both adaptive
controllers use I coefficients; their implementation details and rounding
can produce different accepted meshes. Fixed pairs use the same timestep.
The Diffrax adaptive row uses its public solver/controller interfaces with
the same scalar physical embedded error, rather than its default component
RMS acceptance test. These are forward comparisons, not AD timings.

Timings block on every output. Field setup and a tighter Diffrax Dopri8
reference are excluded from warm samples. Two executions warm each executable;
three timed repetitions alternate backend order. Compilation is recorded
separately and can benefit from a local persistent cache: it is not a cold
compilation study. Position errors are Cartesian metres for coils/VMEC;
Boozer errors use `(sqrt(s) cos(theta), sqrt(s) sin(theta), zeta)` chart units.
Velocity errors are parallel speed for GC and vector speed for Lorentz,
reported both in m/s and divided by birth speed. Errors use common finite
pre-loss saves against the tighter reference; no global tolerance guarantee
is inferred. All completed rows have zero solver failures.

Compiler buffers mean argument + output + temporary bytes − aliases; they
exclude executable constants and resident field data. They are not measured
peak VRAM. Allocator peaks in a complete benchmark output are cumulative
across setup, compilation, all selected cases and the reference; they cannot
be attributed to one solver. Those cumulative allocator dumps are omitted
from the compact saved records.

Native CUDA timings include the private solver field adapters; Diffrax uses
the public field evaluation. The measured gains apply to the complete forward
implementation and do not establish a faster intrinsic Runge–Kutta stage.

## Reproduce

Use the normal JAX backend on the machine. CUDA execution is explicit in
these commands; pass `--execution jax` for the portable native path. No
benchmark script changes the platform or allocator environment.

```sh
python -m examples.benchmarks.benchmark_native_orbits --field coils --model gc \
  --output output/forward_coils_gc_adaptive
python -m examples.benchmarks.benchmark_native_orbits --field coils --model gc \
  --stepping fixed --steps 224 --output output/forward_coils_gc_fixed
python -m examples.benchmarks.benchmark_native_orbits --field coils --model lorentz \
  --horizon 2e-6 --output output/forward_coils_lorentz_adaptive
python -m examples.benchmarks.benchmark_native_orbits --field coils --model lorentz \
  --horizon 2e-6 --stepping fixed --steps 1792 --output output/forward_coils_lorentz_fixed
python -m examples.benchmarks.benchmark_native_orbits --field vmec \
  --output output/forward_vmec_adaptive
python -m examples.benchmarks.benchmark_native_orbits --field vmec \
  --stepping fixed --output output/forward_vmec_fixed
python -m examples.benchmarks.benchmark_native_orbits --field boozer \
  --cases native_dopri5 diffrax_dopri5 native_dopri8 diffrax_dopri8 \
  --output output/forward_boozer_adaptive
python -m examples.benchmarks.benchmark_native_orbits --field boozer \
  --stepping fixed --output output/forward_boozer_fixed
```

Defaults are 64 particles, 4 keV alphas, seed 42, eight saves, 20 us,
rtol=atol=1e-9 and 160 fixed steps. Coil GC uses 224 fixed steps and Lorentz
1792 as specified above. The bundled unscaled QA coil geometry and
reactor-scale QA equilibrium are separate workloads; no cross-field timing
equivalence is claimed. Boozer setup uses the optional `essos[booz]` dependency;
reuse a previously saved `boozer_inputs.npz` with `--field-cache` to skip its
transform. The recorded local Boozer rows reused that same transform cache.

For the full alpha workload, run the separate
[`benchmark_native_boozer_alpha`](../../benchmark_native_boozer_alpha.py)
module with `--particles 8192 --horizon .02 --rtol 1e-10 --atol 1e-10`.
It saves its transform so Dopri5, Dopri8 and RK4 can reuse the same field.
`boozer_remote_console.json` retains the prior completed full-run console
evidence. Those measurements predate extraction, and their full traces and
peak-memory reports were not supplied. They are not a rerun of this PR.
`boozer_extraction.json` is the paired, bitwise-equal 64-particle check against
the archived lean implementation, including counts, buffers and nine samples.
`consolidation.json` compares the subsequent single-module implementation
against the saved nine-module version. All four 64-particle field/model
cases produce bitwise-identical results and identical compiler buffers;
five interleaved warm samples show essentially unchanged execution time.

## CI validation

`tests/test_native_orbit.py` collects 19 tests. It checks formulas and dense
output against Diffrax, analytic orbits, I/PI control, rejections, independent
events, failure bounds, physical error budgets, field values/derivatives and
the GPU loop through Pallas CPU interpretation. No GPU is required.
The RHS preparation test also removes the legacy public JAX partial-evaluation
export to reproduce the CI compatibility failure; the solver imports the
helper from its defining module.
`validation.json` records local line/branch coverage and source hashes.
The coverage command uses `pytest-cov`, which GitHub Actions installs; the
same tests run without coverage by omitting the `--cov` options.

```sh
JAX_PLATFORMS=cpu pytest -q tests/test_native_orbit.py \
  --cov=essos --cov-branch --cov-report=term-missing
```
