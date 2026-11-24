#!/bin/bash
# PySCF Al32 references with canonical orthogonalization of the overlap
# (threshold 1e-8, at every k): gth-dzvp is nearly linearly dependent in fcc
# aluminium (smallest overlap eigenvalue 1.8e-10), and PySCF's default
# generalized eigensolver then lands 1 mHa off on Al4 (and its forces with
# it). dftax's scf_k orthogonalizes the same way, so these are the
# like-for-like references. 2x2x2 k-mesh, Fermi smearing 0.005 Ha, 42^3.
# CPU only, a whole node; from the repository root:
#
#   sbatch --output=$SCRATCH/dftax-pbc/logs/%x-%j.out scripts/pbc/ref_al_lindep_pyscf.sbatch
#SBATCH --job-name=pbc-al-lindep
#SBATCH --account=aip-necludov
#SBATCH --partition=cpubase_bynode_b2
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=06:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export PYSCF_MAX_MEMORY=200000
PY=${PYTHON:-.venv/bin/python}
OUT=${OUT:-$SCRATCH/dftax-pbc/reference}
mkdir -p "$OUT"
echo "job ${SLURM_JOB_ID:-local} on $(hostname), $OMP_NUM_THREADS threads, $(git rev-parse --short HEAD)"

fail=0
run() {
    local name=$1; shift
    local t0=$SECONDS
    if "$PY" scripts/pbc/ref_pyscf.py "$@" --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1; then
        echo "ok   $name ($((SECONDS - t0)) s): $(tail -n 1 "$OUT/$name.log")"
    else
        echo "FAIL $name ($((SECONDS - t0)) s):"; tail -n 15 "$OUT/$name.log"; fail=1
    fi
}

common="--kmesh 2 2 2 --smearing 0.005 --mesh 42 42 42 --lindep 1e-8 --conv-tol 1e-11"
run pyscf_al32_lda_k222_lindep          al32 $common
run pyscf_al32_lda_k222_lindep_rattled  al32 $common --rattle 0.05 --forces --conv-tol-grad 1e-8
exit $fail
