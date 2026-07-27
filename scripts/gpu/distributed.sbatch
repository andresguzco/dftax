#!/bin/bash
# Multi-node dftax: one task per node, that task owning the node's GPUs.
#
#   sbatch scripts/gpu/distributed.sbatch
#   PYTHON=/path/to/venv/bin/python sbatch --account=<acct> scripts/gpu/distributed.sbatch
#
# JAX is multi-controller, so the launcher's only job is to start the same
# program once per node; jax.distributed.initialize() (called by dftax's
# distributed()) reads the SLURM environment to find the coordinator and the
# process ids, and the mesh then spans every GPU of every node.
#
# On a Digital Research Alliance cluster, --account and the partition (which
# is the walltime tier there) are mandatory and GPUs are requested by type;
# scripts/gpu/alliance_setup.sh fills those in and submits this file.
#SBATCH --job-name=dftax-dist
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
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

# Smoke first: water/sto-3g builds in seconds, so a broken interconnect or a
# process group that never forms shows up immediately rather than after the
# ethanol build. Then the real case.
srun --kill-on-bad-exit=1 \
    "${PYTHON:-.venv/bin/python}" scripts/gpu/validate_distributed.py \
    --mol water --basis sto-3g --xc PBE0

srun --kill-on-bad-exit=1 \
    "${PYTHON:-.venv/bin/python}" scripts/gpu/validate_distributed.py \
    --mol ethanol --basis def2-svp --xc PBE0
