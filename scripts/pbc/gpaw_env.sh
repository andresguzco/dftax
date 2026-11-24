# Sourced by the GPAW job scripts on Tamia: a throwaway environment from the
# cluster's own wheels, and the launcher under which the MPI-linked wheel
# starts. The wheel is an x86-64-v3 build on an x86-64-v4 module stack;
# preloading the node's MPI library leaves one copy in the process (found by
# scripts/pbc/ref_small_gpaw.sbatch, which tries the alternatives).
module load python 2>&1
virtualenv --no-download "$SLURM_TMPDIR/gpaw" > /dev/null
source "$SLURM_TMPDIR/gpaw/bin/activate"
pip install --no-index --quiet gpaw ase || { echo "pip install failed"; exit 1; }
ulimit -c 0
export OMP_NUM_THREADS=1
launch="mpirun -x LD_PRELOAD=${EBROOTOPENMPI:-}/lib/libmpi.so -n ${SLURM_NTASKS:-8}"
$launch python -c 'from gpaw.mpi import world; print("gpaw ranks", world.size)' \
    | tail -n 1 || { echo "MPI does not come up"; exit 1; }
