#!/bin/bash
# PySCF equations of state (energies and band edges) of Si8 and quartz 9, the
# same model as dftax's (gth-dzvp, gth-pade, LDA, same meshes): the seven
# volumes of systems.EOS_VOLUME_FACTORS centred on each model's own minimum
# (dftax: 1.09 V_exp for Si8, 1.03 for quartz 9), plus the experimental
# volume for the band-gap comparison. CPU only; from the repository root:
#
#   sbatch --output=$SCRATCH/dftax-pbc/logs/%x-%j.out scripts/pbc/ref_eos_pyscf.sbatch
#SBATCH --job-name=pbc-eos-pyscf
#SBATCH --account=aip-necludov
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export PYSCF_MAX_MEMORY=50000
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

eos() {      # system center mesh...
    local system=$1 center=$2; shift 2
    run "pyscf_${system}_lda_eos_v1.0000" "$system" --mesh "$@"
    for f in 0.94 0.96 0.98 1.00 1.02 1.04 1.06; do
        v=$(awk -v c="$center" -v f="$f" 'BEGIN { printf "%.10f", c * f }')
        run "pyscf_${system}_lda_eos_v$(printf %.4f "$v")" "$system" --mesh "$@" --volume-factor "$v"
    done
}

eos si8 1.09 33 33 33
eos quartz9 1.03 91 91 99
exit $fail
