#!/bin/bash
# GPAW plane-wave references with the HGH potential for the small periodic
# cells (Phase 0 of the periodic extension). CPU only; run from the
# repository root:
#
#   mkdir -p $SCRATCH/dftax-pbc/logs
#   sbatch --output=$SCRATCH/dftax-pbc/logs/%x-%j.out scripts/pbc/ref_small_gpaw.sbatch
#
# GPAW comes from the cluster's own wheels, installed into a throwaway
# environment on the node's local disk: compute nodes have no network, and
# the HGH potentials need no data files.
#SBATCH --job-name=pbc-ref-gpaw
#SBATCH --account=aip-necludov
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=1
NTASKS=${SLURM_NTASKS:-8}
OUT=${OUT:-$SCRATCH/dftax-pbc/reference}
mkdir -p "$OUT"
echo "job ${SLURM_JOB_ID:-local} on $(hostname), $NTASKS MPI ranks, $(git rev-parse --short HEAD)"

module load python 2>&1
virtualenv --no-download "$SLURM_TMPDIR/gpaw" > /dev/null
source "$SLURM_TMPDIR/gpaw/bin/activate"
pip install --no-index --quiet gpaw ase || { echo "pip install failed"; exit 1; }

# The wheel links the cluster's libxc; which version is not recorded
# anywhere we can read, so take the first whose extension loads.
loaded=""
for v in "" 7.1.2 7.0.0 6.2.2; do
    [ -n "$v" ] && module load "libxc/$v" 2>/dev/null
    if python -c "import _gpaw" 2>/dev/null; then loaded="libxc/${v:-none}"; break; fi
done
if [ -z "$loaded" ]; then
    echo "the GPAW extension does not load; unresolved libraries:"
    python -c "import _gpaw" 2>&1 | tail -n 3
    ldd "$(python -c 'import importlib.util as u; print(u.find_spec("_gpaw").origin)')" | grep "not found"
    exit 1
fi
python -c "import gpaw, ase; print('gpaw', gpaw.__version__, '| ase', ase.__version__, '| $loaded')"

# The wheel is MPI-linked (built against the default openmpi/4.1.5, as its
# run path says) and initialises MPI when gpaw.mpi is imported, so it needs
# a launch under which MPI comes up. Slurm here has no PMIx plugin (srun
# --mpi=list: none, pmi2, cray_shasta), a plain `srun -n` is refused by Open
# MPI, and a bare `python` crashes in MPI_Init. Take the first launcher that
# works, and say why each of the others did not.
ulimit -c 0
ompi_info 2>/dev/null | grep -i -E "slurm|pmi" | tr -s ' ' | cut -c1-110 | head -n 8
launch=""
probe='from gpaw.mpi import world; print("ranks", world.size)'
for cand in "mpirun -n $NTASKS" "mpiexec --oversubscribe -n $NTASKS" \
            "srun --mpi=pmi2 -n $NTASKS" \
            "env OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader" "env"; do
    if msg=$($cand python -c "$probe" 2>&1); then
        launch=$cand; echo "launcher: $cand ($(echo "$msg" | tail -n 1))"; break
    fi
    echo "-- $cand: failed"; echo "$msg" | grep -v -E '^\[|^-+$|^\s*$' | tail -n 6 | cut -c1-200
done
[ -n "$launch" ] || { echo "no launcher brings MPI up"; exit 1; }

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

run gpaw_si8_hgh_e20      si8     --ecut 20
run gpaw_si8_hgh_e30      si8     --ecut 30
run gpaw_si8_hgh_e40      si8     --ecut 40 --forces --stress
run gpaw_quartz9_hgh_e40  quartz9 --ecut 40
run gpaw_quartz9_hgh_e60  quartz9 --ecut 60
run gpaw_quartz9_hgh_e80  quartz9 --ecut 80 --forces --stress
run gpaw_al4_hgh_e15      al4     --kmesh 4 4 4 --smearing 0.005 --ecut 15
run gpaw_al4_hgh_e20      al4     --kmesh 4 4 4 --smearing 0.005 --ecut 20
run gpaw_al4_hgh_e30      al4     --kmesh 4 4 4 --smearing 0.005 --ecut 30 --forces --stress
exit $fail
