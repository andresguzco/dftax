#!/bin/bash
# GPAW plane waves, HGH potential (the same as gth-pade), LDA with VWN
# correlation: the Si8 equation of state at 40 Ha with a Gamma-centred 2x2x2
# k-mesh (exactly Si64 at the Gamma point), at the volumes of dftax's Si64
# equation of state (systems.EOS_VOLUME_FACTORS around 1.00): energies,
# band edges, stress. Quartz waits for its cutoff series
# (ref_quartz_gpaw.sbatch). CPU only; from the repository root:
#
#   sbatch --output=$SCRATCH/dftax-pbc/logs/%x-%j.out scripts/pbc/ref_eos_gpaw.sbatch
#SBATCH --job-name=pbc-eos-gpaw
#SBATCH --account=aip-necludov
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

for v in 0.94 0.96 0.98 1.00 1.02 1.04 1.06; do
    run "gpaw_si8_hgh_e40_k222_eos_v$(printf %.4f "$v")" si8 --ecut 40 \
        --kmesh 2 2 2 --volume-factor "$v" --stress
done
exit $fail
