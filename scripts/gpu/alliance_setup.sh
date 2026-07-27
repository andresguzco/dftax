#!/bin/bash
# Set up dftax on a Digital Research Alliance cluster and submit the two-node
# distributed validation.
#
#   git clone https://github.com/andresguzco/dftax.git ~/dftax-engine
#   cd ~/dftax-engine && git checkout perf/sharded-scale
#   bash scripts/gpu/alliance_setup.sh              # build env, then submit
#   bash scripts/gpu/alliance_setup.sh --env-only   # build env, do not submit
#
# What is cluster-specific here, and why it is not guessed:
#   * --account is mandatory on DRAC (no default). Tamia's is aip-necludov;
#     aip-aspuru is the old group and must not be used.
#   * A DRAC partition IS the walltime tier, not a queue: gpubase_bynode_b1
#     <=3h, _b2 <=12h, _b3 <=24h. Picked from --time below.
#   * Allocations are by whole node, so --mem=0 takes the node's memory, and
#     GPUs are requested by type (h100:4 on Tamia's 53 four-GPU nodes; the
#     twelve h200:8 nodes need GPUS=h200:8).
#
# Overrides: ACCOUNT, PARTITION, NODES, GPUS, CPUS, TIME, VENV.
set -euo pipefail
cd "$(dirname "$0")/../.."

ACCOUNT=${ACCOUNT:-aip-necludov}
NODES=${NODES:-2}
GPUS=${GPUS:-h100:4}
CPUS=${CPUS:-16}
TIME=${TIME:-00:30:00}
VENV=${VENV:-$PWD/.venv}

# Walltime tier -> partition, unless one was passed explicitly.
if [ -z "${PARTITION:-}" ]; then
    hours=${TIME%%:*}
    if   [ "$((10#$hours))" -lt 3 ];  then PARTITION=gpubase_bynode_b1
    elif [ "$((10#$hours))" -lt 12 ]; then PARTITION=gpubase_bynode_b2
    else                                   PARTITION=gpubase_bynode_b3
    fi
fi

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x "$UV" ] || { echo "no uv on PATH nor at $HOME/.local/bin/uv" >&2; exit 127; }
# Login nodes have outbound network, compute nodes do not, so the environment
# is resolved here and only used there.
"$UV" sync --extra cuda12
"$VENV/bin/python" -c "import jax, dftax; print('jax', jax.__version__, '| dftax ok')"

if [ "${1:-}" = "--env-only" ]; then
    echo "environment ready; submit with:"
    echo "  PYTHON=$VENV/bin/python sbatch --account=$ACCOUNT" \
         "--partition=$PARTITION --nodes=$NODES --gpus-per-node=$GPUS" \
         "scripts/gpu/distributed.sbatch"
    exit 0
fi

echo "submitting $NODES nodes x $GPUS, $TIME on $PARTITION ($ACCOUNT)"
PYTHON="$VENV/bin/python" sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --nodes="$NODES" \
    --gpus-per-node="$GPUS" \
    --cpus-per-task="$CPUS" \
    --mem=0 \
    --time="$TIME" \
    scripts/gpu/distributed.sbatch
