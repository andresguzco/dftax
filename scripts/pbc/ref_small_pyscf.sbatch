#!/bin/bash
# PySCF references for the small periodic cells (Phase 0 of the periodic
# extension). CPU only; run from the repository root:
#
#   mkdir -p $SCRATCH/dftax-pbc/logs
#   sbatch --output=$SCRATCH/dftax-pbc/logs/%x-%j.out scripts/pbc/ref_small_pyscf.sbatch
#
# Each run writes <name>.json (the reference) and <name>.log (PySCF's own
# output) under $OUT; the job log keeps one summary line per run.
#SBATCH --job-name=pbc-ref-pyscf
#SBATCH --account=aip-necludov
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
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

run pyscf_si8_lda          si8     --forces --stress
run pyscf_si8_lda_rattled  si8     --rattle 0.05 --forces
run pyscf_quartz9_lda      quartz9 --forces --stress
run pyscf_si8_pbe          si8     --xc pbe --pseudo gth-pbe --forces
run pyscf_si2_lda_k444     si2     --kmesh 4 4 4
run pyscf_al4_lda_k444     al4     --kmesh 4 4 4 --smearing 0.005 --forces
exit $fail
