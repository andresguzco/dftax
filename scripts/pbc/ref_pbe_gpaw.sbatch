#!/bin/bash
# GPAW plane waves with the HGH potentials and PBE: the GGA check of the
# plane-wave engine (the density gradient by FFT), on Si8 at 40 Ha and
# quartz 9 at 200 Ha, Gamma point, with forces and stress. HGH potentials
# were fitted for LDA; with PBE they are simply the same Hamiltonian in
# both codes. CPU only; from the repository root:
#
#   sbatch --output=$SCRATCH/dftax-pbc/logs/%x-%j.out scripts/pbc/ref_pbe_gpaw.sbatch
#SBATCH --job-name=pbc-pbe-gpaw
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

run gpaw_si8_hgh_pbe_e40       si8 --xc PBE --ecut 40 --stress
run gpaw_quartz9_hgh_pbe_e200  quartz9 --xc PBE --ecut 200 --forces --stress
exit $fail
