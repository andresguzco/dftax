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

# The wheel is MPI-linked and initialises MPI when gpaw.mpi is imported. A
# bare `python` inside the job failed there (Open MPI: "Failed to create
# keyval for UCX datatypes"), so take the first launcher under which the
# world communicator comes up.
launch=""
for cand in "srun -n $NTASKS" "mpirun -n $NTASKS" "env OMPI_MCA_pml=ob1"; do
    if $cand python -c "from gpaw.mpi import world; assert world.size >= 1" > /dev/null 2>&1; then
        launch=$cand; break
    fi
done
if [ -z "$launch" ]; then
    echo "no launcher brings MPI up; the last attempt said:"
    env OMPI_MCA_pml=ob1 python -c "from gpaw.mpi import world" 2>&1 | tail -n 8
    pip list 2>/dev/null | grep -i -E "mpi|gpaw|ase|numpy|scipy"
    exit 1
fi
echo "launcher: $launch"

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
