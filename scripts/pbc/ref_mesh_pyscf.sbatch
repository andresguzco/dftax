#!/bin/bash
# Mesh convergence of the PySCF energy on the small periodic cells (Phase 0
# of the periodic extension): which FFT mesh each material needs. PySCF's own
# mesh at its default precision (51^3 for Si8, 123x123x135 for quartz 9) is
# the reference; the baselines then run at the smallest mesh that holds the
# energy. CPU only; run from the repository root:
#
#   sbatch --output=$SCRATCH/dftax-pbc/logs/%x-%j.out scripts/pbc/ref_mesh_pyscf.sbatch
#SBATCH --job-name=pbc-mesh-pyscf
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

for n in 27 33 39 45; do
    run pyscf_si8_lda_m$n si8 --mesh $n $n $n
done
for m in "61 61 67" "75 75 83" "91 91 99" "105 105 115"; do
    run pyscf_quartz9_lda_m${m%% *} quartz9 --mesh $m
done
for n in 21 25 29 33; do
    run pyscf_al4_lda_k444_m$n al4 --kmesh 4 4 4 --smearing 0.005 --mesh $n $n $n
done
exit $fail
