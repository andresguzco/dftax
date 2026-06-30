"""dftax: a self-contained, pure-JAX/Equinox Kohn-Sham DFT engine.

The package exposes a differentiable KS-DFT toolkit with no PySCF runtime
dependency for the core compute path:

- ``dftax.integrals``: analytical one- and two-electron integral matrices
  (overlap S, kinetic T, nuclear attraction V, nuclear repulsion, ERIs 2c/3c/4c)
  via the Obara-Saika recurrence, all jit/vmap/grad-friendly.
- ``dftax.energy``: GTO basis evaluation, the Boys function, density fitting,
  Hartree and hybrid exchange, exchange-correlation functionals, real-space
  grids, and pointwise potentials.
- ``dftax.utils``: chunked ``vmap`` helpers and shared types.
- ``dftax.ks``: Kohn-Sham drivers (energy functional + DIIS SCF) for both
  closed-shell (RKS) and open-shell (UKS) systems, plus the unified ``run_ks``.

A few of the most common entry points are re-exported here for convenience;
import the submodules directly for the full surface.
"""

from dftax.energy.gto import BasisData, extract_basis_data, eval_gto
from dftax.integrals import (
    overlap_matrix,
    kinetic_matrix,
    nuclear_attraction_matrix,
    nuclear_repulsion,
    eri2c_matrix,
    eri3c_matrix,
)
from dftax.ks import (
    RKS, run_rks, rks_scf, rks_minimize, rks_forces, SCFResult,
    UKS, run_uks, uks_scf, uks_minimize, uks_forces, UKSResult,
    run_ks,
    run_ks_batched, run_rks_batched, run_uks_batched, BatchedResult,
    dipole, polarizability, hessian, vibrations, ir_spectrum, raman_spectrum,
    alchemical_deriv, implicit_density,
)

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("dftax")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+local"
del _pkg_version, PackageNotFoundError

__all__ = [
    "__version__",
    "BasisData",
    "extract_basis_data",
    "eval_gto",
    "overlap_matrix",
    "kinetic_matrix",
    "nuclear_attraction_matrix",
    "nuclear_repulsion",
    "eri2c_matrix",
    "eri3c_matrix",
    # restricted Kohn-Sham driver
    "RKS",
    "run_rks",
    "rks_scf",
    "rks_minimize",
    "rks_forces",
    "SCFResult",
    # unrestricted (open-shell) Kohn-Sham driver
    "UKS",
    "run_uks",
    "uks_scf",
    "uks_minimize",
    "uks_forces",
    "UKSResult",
    # unified dispatcher
    "run_ks",
    # batched (vmap over geometries)
    "run_ks_batched",
    "run_rks_batched",
    "run_uks_batched",
    "BatchedResult",
    # response properties
    "dipole",
    "polarizability",
    "hessian",
    "vibrations",
    "ir_spectrum",
    "raman_spectrum",
    "alchemical_deriv",
    # implicit-differentiation SCF response
    "implicit_density",
]
