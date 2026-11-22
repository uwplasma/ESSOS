import os as _os

_X64_SETTING = 'jax.config.update("jax_enable_x64", True)'


def _require_x64(what):
    """Raise unless JAX uses 64-bit floats: ESSOS tolerances assume them and float32 would pass silently."""
    import jax
    if not jax.config.jax_enable_x64:
        raise RuntimeError(f"{what} needs 64-bit floats: call {_X64_SETTING} (or essos.configure_jax()) "
                           "before building fields, coils or traces")


def _jaxlib_version():
    """Leading integers of the jaxlib version ("0.10.0rc1" -> (0, 10)); () when unknown."""
    import re
    import jaxlib.version
    match = re.match(r"(\d+)\.(\d+)", jaxlib.version.__version__)
    return tuple(map(int, match.groups())) if match else ()


def configure_jax(x64=True, cpu_devices=None, cache_dir=None):
    """Set the JAX options ESSOS is tuned for; importing ESSOS sets none of them.

    ``x64`` enables 64-bit floats. A vmapped trace on one CPU device runs its many small kernels on
    about one core, so Tracing shards particles over one JAX device per core: ``cpu_devices``
    (default ``ESSOS_CPU_DEVICES``, else the cores this process may use, affinity and
    ``SLURM_CPUS_PER_TASK``) sets that count, unless ``XLA_FLAGS`` or ``jax_num_cpu_devices`` already
    chose one. The count can only be set before JAX starts its backends (the first array or
    ``jax.devices()`` call): a count asked for afterwards raises ``RuntimeError``, the default one is
    skipped with a warning. On Python 3.10 (diffrax 0.7.0), importing
    ``diffrax`` starts the CPU backend, so call this before importing ``essos.dynamics``. Compiled programs are cached in ``cache_dir`` (default ``ESSOS_XLA_CACHE``,
    else ``~/.cache/essos/xla``; empty disables it) with jaxlib 0.10 or later, which can reload them; an
    explicit ``cache_dir`` enables the cache on older jaxlib too.
    """
    import jax
    jax.config.update("jax_enable_x64", x64)
    if "xla_force_host_platform_device_count" not in _os.environ.get("XLA_FLAGS", "") and jax.config.jax_num_cpu_devices < 0:
        requested = str(cpu_devices if cpu_devices is not None else _os.environ.get("ESSOS_CPU_DEVICES", "")).strip()
        count = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else (_os.cpu_count() or 1)
        slurm = _os.environ.get("SLURM_CPUS_PER_TASK", "")
        count = int(requested) if requested.isdigit() else min(count, int(slurm) if slurm.isdigit() else count)
        try:
            jax.config.update("jax_num_cpu_devices", max(1, count))
        except RuntimeError as error:  # backends already started
            message = (f"jax_num_cpu_devices={max(1, count)} cannot be set after JAX has started its backends: "
                       "call essos.configure_jax() before the first JAX array or jax.devices()")
            if requested:
                raise RuntimeError(message) from error
            import warnings
            warnings.warn(message, stacklevel=2)
    cache = _os.environ.get("ESSOS_XLA_CACHE", _os.path.join(_os.path.expanduser("~"), ".cache", "essos", "xla"))
    cache = cache if cache_dir is None else cache_dir
    # jaxlib < 0.10 overflows a worker-thread stack reloading a large cached XLA:CPU executable (a crash on
    # the first warm rerun, on macOS and Linux), so older or unknown jaxlib compiles every run unless the
    # caller passed cache_dir explicitly.
    if cache and (cache_dir is not None or _jaxlib_version() >= (0, 10)) and not jax.config.jax_compilation_cache_dir:
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
