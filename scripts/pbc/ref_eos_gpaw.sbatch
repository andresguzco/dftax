#!/bin/bash
# GPAW plane waves, HGH potential (the same as gth-pade), LDA with VWN
# correlation: the Si8 equation of state at 40 Ha (energies, band edges,
# stress), at the Gamma point like the Gaussian runs. GPAW puts Si8 under
# +5.9 GPa at the experimental volume, so the seven volumes are centred on
# 1.06 V_exp, and the experimental volume is added for the band-gap
# comparison. Quartz waits for its cutoff series (ref_quartz_gpaw.sbatch).
# CPU only; from the repository root:
#
#   sbatch --output=$SCRATCH/dftax-pbc/logs/%x-%j.out scripts/pbc/ref_eos_gpaw.sbatch
#SBATCH --job-name=pbc-eos-gpaw
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

run gpaw_si8_hgh_e40_eos_v1.0000 si8 --ecut 40 --stress
for f in 0.94 0.96 0.98 1.00 1.02 1.04 1.06; do
    v=$(awk -v c=1.06 -v f="$f" 'BEGIN { printf "%.10f", c * f }')
    run "gpaw_si8_hgh_e40_eos_v$(printf %.4f "$v")" si8 --ecut 40 --volume-factor "$v" --stress
done
exit $fail
