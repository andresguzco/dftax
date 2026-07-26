#!/bin/bash
# Set up dftax on a Digital Research Alliance cluster (Tamia, Narval, Fir, ...)
# and submit the two-node distributed validation.
#
#   git clone https://github.com/andresguzco/dftax.git && cd dftax
#   git checkout perf/sharded-scale
#   bash scripts/gpu/alliance_setup.sh              # build env, then submit
#   bash scripts/gpu/alliance_setup.sh --env-only   # build env, do not submit
#
# Alliance specifics this handles: their module stack rather than a system
# python, their local wheelhouse first (compute nodes have no internet, and
# --no-index is the supported path), and the mandatory --account, taken from
# SLURM_ACCOUNT or the first allocation `sacctmgr` reports for you.
set -euo pipefail

cd "$(dirname "$0")/../.."
VENV=${VENV:-$PWD/.venv-alliance}

module --force purge 2>/dev/null || true
module load StdEnv/2023 2>/dev/null || module load StdEnv 2>/dev/null || true
module load python/3.12 2>/dev/null || module load python 2>/dev/null || true
module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
echo "python: $(command -v python) ($(python --version 2>&1))"

if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip

# The wheelhouse carries jax/jaxlib built for this cluster's CUDA; prefer it,
# and only reach for PyPI for what it does not have (basis-set-exchange,
# equinox, jaxtyping tend to need it, and login nodes can reach PyPI).
python -m pip install --no-index --upgrade "jax[cuda12]" numpy optax 2>/dev/null \
    || python -m pip install --upgrade "jax[cuda12]" numpy optax
python -m pip install --upgrade basis-set-exchange equinox jaxtyping periodictable
python -m pip install --no-deps -e .

python - <<'PY'
import jax
print("jax", jax.__version__, "| local devices:", jax.local_devices())
PY

if [ "${1:-}" = "--env-only" ]; then
    echo "environment ready at $VENV; submit with:"
    echo "  PYTHON=$VENV/bin/python sbatch scripts/gpu/distributed.sbatch"
    exit 0
fi

ACCOUNT=${SLURM_ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=account \
    2>/dev/null | grep -E '^(def|rrg|ctb)-' | head -1)}
if [ -z "${ACCOUNT:-}" ]; then
    echo "could not determine an allocation account; rerun with" >&2
    echo "  SLURM_ACCOUNT=<your-account> bash $0" >&2
    exit 1
fi
# Overridable on the command line, because GPU counts and memory flags differ
# per cluster and these take precedence over the #SBATCH lines in the file:
#   GPUS_PER_NODE=2 NODES=2 TIME=00:20:00 bash scripts/gpu/alliance_setup.sh
echo "submitting ${NODES:-2} nodes x ${GPUS_PER_NODE:-4} GPUs under $ACCOUNT"
PYTHON=$VENV/bin/python sbatch \
    --account="$ACCOUNT" \
    --nodes="${NODES:-2}" \
    --gpus-per-task="${GPUS_PER_NODE:-4}" \
    --time="${TIME:-00:30:00}" \
    scripts/gpu/distributed.sbatch
