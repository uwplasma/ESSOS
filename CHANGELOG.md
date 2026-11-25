# Changelog

## Unreleased

### Breaking

- Importing `essos` or any submodule no longer changes JAX settings: it does not enable float64, set the CPU
  device count or set the compilation cache directory, and it starts no JAX backend. Call
  `essos.configure_jax()` once, before any JAX computation, to get the previous behaviour (float64, one CPU
  device per usable core, the cache in `~/.cache/essos/xla`; `ESSOS_CPU_DEVICES`, `SLURM_CPUS_PER_TASK` and
  `ESSOS_XLA_CACHE` as before).
- With float64 off, `Curves`, `BiotSavart_from_gamma`, `Vmec`, `SurfaceRZFourier`, `BoozerField.from_booz`,
  `Particles`, `Tracing`, `nu_D_ab`, `nu_s_ab` and `trace_boozer` with `species` raise a `RuntimeError` that
  names `jax.config.update("jax_enable_x64", True)`. Collisionless `trace_boozer` runs in float32.
