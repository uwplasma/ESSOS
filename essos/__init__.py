import os as _os

X64_SETTING = 'jax.config.update("jax_enable_x64", True)'


def require_x64(what):
    """Raise unless JAX uses 64-bit floats: ESSOS tolerances assume them and float32 would pass silently."""
    import jax
    if not jax.config.jax_enable_x64:
        raise RuntimeError(f"{what} needs 64-bit floats: call {X64_SETTING} (or essos.configure_jax()) "
                           "before building fields, coils or traces")


def configure_jax(x64=True, cpu_devices=None, cache_dir=None):
    """Set the JAX options ESSOS is tuned for; importing ESSOS sets none of them.

    ``x64`` enables 64-bit floats. A vmapped trace on one CPU device runs its many small kernels on
    about one core, so Tracing shards particles over one JAX device per core: ``cpu_devices``
    (default ``ESSOS_CPU_DEVICES``, else the cores this process may use, affinity and
    ``SLURM_CPUS_PER_TASK``) sets that count, unless ``XLA_FLAGS`` or ``jax_num_cpu_devices`` already
    chose one; it only works before JAX initializes its backends. Compiled programs are cached in
    ``cache_dir`` (default ``ESSOS_XLA_CACHE``, else ``~/.cache/essos/xla``; empty disables it).
    """
    import jax
    jax.config.update("jax_enable_x64", x64)
    if "xla_force_host_platform_device_count" not in _os.environ.get("XLA_FLAGS", ""):
        try:
            if jax.config.jax_num_cpu_devices < 0:
                requested = str(cpu_devices or _os.environ.get("ESSOS_CPU_DEVICES", ""))
                count = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else (_os.cpu_count() or 1)
                slurm = _os.environ.get("SLURM_CPUS_PER_TASK", "")
                count = int(requested) if requested.strip().isdigit() else min(count, int(slurm) if slurm.isdigit() else count)
                jax.config.update("jax_num_cpu_devices", max(1, count))
        except RuntimeError:  # backends already initialized: keep their device count
            pass
    cache = _os.environ.get("ESSOS_XLA_CACHE", _os.path.join(_os.path.expanduser("~"), ".cache", "essos", "xla"))
    cache = cache if cache_dir is None else cache_dir
    if cache and not jax.config.jax_compilation_cache_dir:
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
