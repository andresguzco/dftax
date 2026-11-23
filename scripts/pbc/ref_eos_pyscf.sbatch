#!/bin/bash
# PySCF equations of state (energies and band edges) at the volumes of
# dftax's baseline equations of state, point for point. A small cell with a
# Gamma-centred 2x2x2 k-mesh is exactly its 2x2x2 supercell at the Gamma
# point, so Si8 k222 (mesh 33^3) is Si64 at Gamma (66^3) and quartz 9 k222
# (61x61x67) is quartz 72 (122x122x134): eight times the energy, the same
# band edges. Volumes: systems.EOS_VOLUME_FACTORS around 1.00 for Si and
# around 1.03 for quartz (dftax's minima: 1.008 and 1.041), plus the
# experimental volume. Same model as dftax (gth-dzvp, gth-pade, LDA with VWN).
# CPU only; from the repository root:
#
#   sbatch --output=$SCRATCH/dftax-pbc/logs/%x-%j.out scripts/pbc/ref_eos_pyscf.sbatch
#SBATCH --job-name=pbc-eos-pyscf
#SBATCH --account=aip-necludov
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export PYSCF_MAX_MEMORY=100000
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

eos() {      # system center mesh(3)
    local system=$1 center=$2; shift 2
    local factors="1.0000000000"
    for f in 0.94 0.96 0.98 1.00 1.02 1.04 1.06; do
        factors="$factors $(awk -v c="$center" -v f="$f" 'BEGIN { printf "%.10f", c * f }')"
    done
    for v in $(echo "$factors" | tr ' ' '\n' | sort -u); do
        run "pyscf_${system}_lda_k222_eos_v$(printf %.4f "$v")" "$system" \
            --kmesh 2 2 2 --mesh "$@" --volume-factor "$v"
    done
}

eos si8 1.00 33 33 33
eos quartz9 1.03 61 61 67
exit $fail
