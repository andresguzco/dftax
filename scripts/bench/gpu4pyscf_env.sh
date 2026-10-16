# The GPU4PySCF side of the head-to-head (scripts/bench/gpu4pyscf_bench.py).
#
#   bash scripts/bench/gpu4pyscf_env.sh install   # once, inside an allocation
#   source scripts/bench/gpu4pyscf_env.sh         # every run, sets G4P_PYTHON
#
# GPU4PySCF lives in its own environment (it brings CuPy and its own PySCF).
# CuPy compiles kernels at run time, so it needs the CUDA toolkit pieces the
# pip wheels ship (the [ctk] extra) on LD_LIBRARY_PATH and CUDA_PATH pointing
# at them; without that it fails at its first JIT. geomeTRIC provides the
# molecule files both engines read (DFTAX_GEOM_DIR).
#
# The environment lives on $SCRATCH, which is purged: if G4P_PYTHON does not
# exist, run the install again.

G4P_ENV="${G4P_ENV:-$SCRATCH/g4pvenv}"

if [ "${1:-}" = "install" ]; then
  set -eu
  "$HOME/.local/bin/uv" venv --python 3.11 "$G4P_ENV"
  "$HOME/.local/bin/uv" pip install --python "$G4P_ENV/bin/python" \
    gpu4pyscf-cuda12x "cupy-cuda12x[ctk]" cutensor-cu12 geometric
  echo "installed GPU4PySCF into $G4P_ENV"
  exit 0
fi

export G4P_PYTHON="$G4P_ENV/bin/python"
_site="$("$G4P_PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
_libs="$(find "$_site/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${_libs}${LD_LIBRARY_PATH:-}"
export CUDA_PATH="$_site/nvidia/cuda_runtime"
export DFTAX_GEOM_DIR="$("$G4P_PYTHON" -c 'import geometric, os; print(os.path.join(os.path.dirname(geometric.__file__), "data"))')"
unset _site _libs
