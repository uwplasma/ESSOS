import os as _os

# A vmapped trace on one CPU device runs its many small kernels on about one core, so
# Tracing shards particles over one JAX device per core. Unless XLA_FLAGS or
# jax_num_cpu_devices already chose a count, ESSOS asks for one CPU device per core this
# process may use (affinity and SLURM_CPUS_PER_TASK); ESSOS_CPU_DEVICES overrides it (1 restores a single device). This only works before
# JAX initializes its backends, i.e. when essos is imported before any JAX computation.
if "xla_force_host_platform_device_count" not in _os.environ.get("XLA_FLAGS", ""):
    import jax as _jax
    try:
        if _jax.config.jax_num_cpu_devices < 0:
            _requested = _os.environ.get("ESSOS_CPU_DEVICES", "")
            _count = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else (_os.cpu_count() or 1)
            _slurm = _os.environ.get("SLURM_CPUS_PER_TASK", "")
            _count = int(_requested) if _requested.strip().isdigit() else min(_count, int(_slurm) if _slurm.isdigit() else _count)
            _jax.config.update("jax_num_cpu_devices", max(1, _count))
    except RuntimeError:  # backends already initialized: keep their device count
        pass
