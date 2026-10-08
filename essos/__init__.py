import os as _os

# A vmapped trace on one CPU device runs its many small kernels on about one core, so
# Tracing shards particles over one JAX device per core. Unless XLA_FLAGS or
# jax_num_cpu_devices already chose a count, ESSOS asks for os.cpu_count() CPU devices;
# ESSOS_CPU_DEVICES overrides it (1 restores a single device). This only works before
# JAX initializes its backends, i.e. when essos is imported before any JAX computation.
if "xla_force_host_platform_device_count" not in _os.environ.get("XLA_FLAGS", ""):
    import jax as _jax
    try:
        if _jax.config.jax_num_cpu_devices < 0:
            _requested = _os.environ.get("ESSOS_CPU_DEVICES", "")
            _count = int(_requested) if _requested.strip().isdigit() else (_os.cpu_count() or 1)
            _jax.config.update("jax_num_cpu_devices", max(1, _count))
    except RuntimeError:  # backends already initialized: keep their device count
        pass
