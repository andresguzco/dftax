"""The persistent XLA compilation cache, on by default.

Almost all of a first dftax calculation is XLA compiling the integral kernels,
and JAX can keep compiled programs on disk, but only when a cache directory is
configured, which by default it is not. dftax therefore turns the cache on when
it is imported, unless the user has already chosen one or opted out:

* ``JAX_COMPILATION_CACHE_DIR`` set, or ``jax_compilation_cache_dir``
  configured before importing dftax: that directory is used as is;
* ``DFTAX_COMPILATION_CACHE=0``: nothing is changed;
* ``DFTAX_COMPILATION_CACHE=/some/path``: that directory;
* otherwise ``$XDG_CACHE_HOME/dftax/xla`` (``~/.cache/dftax/xla``).

Two JAX defaults are adjusted alongside, because both would defeat the point:

* ``jax_persistent_cache_min_compile_time_secs`` is lowered from 1 s to
  :data:`MIN_COMPILE_SECS`, unless it was changed from that default. The tiled
  integral kernels (:mod:`dftax.integrals.tiled`) compile in under a second
  each, and there are a few hundred of them; at 1 s none would be kept.
* When dftax picks the directory, the XLA-side per-fusion autotuning cache is
  not written into it (``jax_persistent_cache_enable_xla_caches="none"``). XLA
  writes one file per fusion there and on network filesystems those writes
  fail with "Device or resource busy", which kills the run at its first jit.

The settings are JAX-global, so they also apply to the user's other JAX
programs in the same process; set ``DFTAX_COMPILATION_CACHE=0`` to keep JAX's
defaults.
"""

from __future__ import annotations

import os

MIN_COMPILE_SECS = 0.1

_JAX_DEFAULT_MIN_COMPILE_SECS = 1.0


def default_cache_dir() -> str:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(root, "dftax", "xla")


def enable_compilation_cache(path: str | None = None, *,
                             min_compile_secs: float = MIN_COMPILE_SECS
                             ) -> str | None:
    """Turn on JAX's persistent compilation cache for dftax's kernels.

    Args:
        path: cache directory; ``None`` keeps a directory the user already
            configured, or falls back to :func:`default_cache_dir`.
        min_compile_secs: programs that compile faster than this are not
            written. Applied only if the threshold is still at JAX's default.

    Returns:
        The directory in use, or ``None`` if it could not be set up.
    """
    import jax

    chosen = path or jax.config.jax_compilation_cache_dir or os.environ.get(
        "JAX_COMPILATION_CACHE_DIR")
    picked_by_dftax = not chosen
    if picked_by_dftax:
        chosen = default_cache_dir()
    try:
        os.makedirs(chosen, exist_ok=True)
    except OSError:
        return None
    jax.config.update("jax_compilation_cache_dir", chosen)
    if (jax.config.jax_persistent_cache_min_compile_time_secs
            == _JAX_DEFAULT_MIN_COMPILE_SECS):
        jax.config.update("jax_persistent_cache_min_compile_time_secs",
                          float(min_compile_secs))
    if picked_by_dftax:
        jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
    return chosen


def _enable_on_import() -> None:
    setting = os.environ.get("DFTAX_COMPILATION_CACHE", "")
    if setting == "0":
        return
    try:
        enable_compilation_cache(setting or None)
    except Exception:  # a cache problem must never break the import
        pass
