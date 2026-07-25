#!/bin/bash
# Multi-node dftax: one task per node, that task owning the node's GPUs.
#
#   sbatch scripts/gpu/distributed.sbatch
#
# JAX is multi-controller, so the launcher's only job is to start the same
# program once per node; jax.distributed.initialize() (called by dftax's
# distributed()) reads the SLURM environment to find the coordinator and the
# process ids, and the mesh then spans every GPU of every node.
#SBATCH --job-name=dftax-dist
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=dftax-dist-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Fraction of each GPU JAX may pool. Leave headroom for NCCL buffers.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-.85}
# Uncomment when debugging the interconnect:
# export NCCL_DEBUG=INFO

srun --kill-on-bad-exit=1 \
    "${PYTHON:-.venv/bin/python}" scripts/gpu/validate_distributed.py \
    --mol ethanol --basis def2-svp --xc PBE0
