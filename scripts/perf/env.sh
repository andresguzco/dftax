# Environment every perf run sources, so a number is never a setting in disguise.
#
#   source scripts/perf/env.sh
#   uv run --frozen --no-sync python scripts/perf/profile_terms.py --ladder
#
# Three settings are load-bearing and one is a trap:
#
#   JAX_COMPILATION_CACHE_DIR must be NODE-LOCAL. XLA writes one autotune file
#   per fusion and they fail with "Device or resource busy" on /network/scratch
#   (NFS), which kills a job at its first jit. $SLURM_TMPDIR is node-local; the
#   fallback keeps a laptop run working. The cache is worth ~90 s of coronene's
#   286 s cold build, and since compile time is one of the metrics this plan
#   targets, whether it was warm has to be recorded rather than assumed: set
#   DFTAX_PERF_CACHE=cold to run against an empty cache.
#
#   XLA_PYTHON_CLIENT_PREALLOCATE=false, or peak_bytes_in_use reports the pool
#   rather than the data, which is the mistake the 2026-07-25 GPU4PySCF run
#   made (it reported a column 2.4x the real peak).
#
#   The GPU must be an a100l. dftax is float64 throughout, and Mila's generic
#   --gres=gpu:1 can land on rtx8000/l40s cards with no fast f64 units, which
#   is ~10x the wall time and silently poisons every row.

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
