#!/bin/bash
# GPAW plane waves with the HGH potential on quartz 9 at higher cutoffs: the
# HGH oxygen is hard, and 80 Ha was far from converged (-108.602 Ha, still
# falling by 0.2 Ha from 60). CPU only; run from the repository root:
#
#   sbatch --output=$SCRATCH/dftax-pbc/logs/%x-%j.out scripts/pbc/ref_quartz_gpaw.sbatch
#SBATCH --job-name=pbc-quartz-gpaw
#SBATCH --account=aip-necludov
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT=${OUT:-$SCRATCH/dftax-pbc/reference}
mkdir -p "$OUT"
echo "job ${SLURM_JOB_ID:-local} on $(hostname), ${SLURM_NTASKS:-16} MPI ranks, $(git rev-parse --short HEAD)"
source scripts/pbc/gpaw_env.sh

fail=0
run() {
    local name=$1; shift
    local t0=$SECONDS
    if $launch python scripts/pbc/ref_gpaw.py "$@" --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1; then
        echo "ok   $name ($((SECONDS - t0)) s): $(tail -n 1 "$OUT/$name.log")"
    else
        echo "FAIL $name ($((SECONDS - t0)) s):"; tail -n 15 "$OUT/$name.log"; fail=1
    fi
}

run gpaw_quartz9_hgh_e120  quartz9 --ecut 120
run gpaw_quartz9_hgh_e160  quartz9 --ecut 160
run gpaw_quartz9_hgh_e200  quartz9 --ecut 200 --forces --stress
exit $fail
