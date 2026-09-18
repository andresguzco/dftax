# Environment every perf run sources, so a number is never a setting in disguise.
#
#   source scripts/perf/env.sh
#   uv run --frozen --no-sync python scripts/perf/profile_terms.py --ladder
#
# Three settings are load-bearing:
#
#   JAX_COMPILATION_CACHE_DIR must be NODE-LOCAL. XLA writes one autotune file
#   per fusion and they fail with "Device or resource busy" on NFS, killing the
#   job at its first jit. Set DFTAX_PERF_CACHE=cold to run against an empty
#   cache, since a warm one hides most of the compile cost.
#
#   XLA_PYTHON_CLIENT_PREALLOCATE=false, or peak_bytes_in_use reports the
#   allocator pool rather than the data.
#
#   The GPU must be an a100l. dftax is float64 throughout, and a generic
#   --gres=gpu:1 can land on a card with no fast f64 units: ~10x the wall time,
#   silently.

export PATH="$HOME/.local/bin:$PATH"

_cache_root="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
if [ "${DFTAX_PERF_CACHE:-warm}" = "cold" ]; then
  export JAX_COMPILATION_CACHE_DIR="$(mktemp -d "$_cache_root/jaxcache.XXXXXX")"
  echo "perf/env: COLD compile cache at $JAX_COMPILATION_CACHE_DIR" >&2
else
  export JAX_COMPILATION_CACHE_DIR="$_cache_root/dftax_jaxcache"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
fi
unset _cache_root

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3

if command -v nvidia-smi >/dev/null 2>&1; then
  _gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  case "$_gpu" in
    *A100*|*H100*|*H200*) ;;
    *) echo "perf/env: WARNING: $_gpu has no fast f64 units; timings are not" \
            "comparable to the A100 baseline" >&2 ;;
  esac
  unset _gpu
fi
