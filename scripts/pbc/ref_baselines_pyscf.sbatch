#!/bin/bash
# PySCF references for the three baselines of the periodic extension: Si64,
# alpha-quartz with 72 atoms, Al32, each ideal and rattled. The meshes come
# from the mesh-convergence runs on the small cells (the smallest that holds
# the energy: 33^3 per Si8, 61x61x67 per quartz 9, 21^3 per Al4). CPU only;
# run from the repository root:
#
#   sbatch --output=$SCRATCH/dftax-pbc/logs/%x-%j.out scripts/pbc/ref_baselines_pyscf.sbatch
#
# Cheapest first, so the early results do not wait for quartz. Runs with
# forces converge the orbital gradient to 1e-8: PySCF's default,
# sqrt(conv_tol), leaves force errors near 1e-6 Ha/Bohr (measured on Si8).
#SBATCH --job-name=pbc-base-pyscf
#SBATCH --account=aip-necludov
#SBATCH --partition=cpubase_bynode_b2
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=12:00:00
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

run pyscf_si64_lda               si64 --mesh 66 66 66
run pyscf_si64_lda_rattled       si64 --mesh 66 66 66 --rattle 0.05 --forces --conv-tol-grad 1e-8
run pyscf_al32_lda_k222          al32 --kmesh 2 2 2 --smearing 0.005 --mesh 42 42 42
run pyscf_al32_lda_k222_rattled  al32 --kmesh 2 2 2 --smearing 0.005 --mesh 42 42 42 --rattle 0.05 --forces --conv-tol-grad 1e-8
run pyscf_quartz72_lda           quartz72 --mesh 122 122 134
run pyscf_quartz72_lda_rattled   quartz72 --mesh 122 122 134 --rattle 0.05
exit $fail
