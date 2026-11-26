"""Differentiable Kohn-Sham DFT in JAX.

Example::

    import jax
    jax.config.update("jax_enable_x64", True)
    from dftax import KS, Molecule, becke, exact, scf
    from dftax.energy.xc import PBE

    mol = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "def2-svp")
    res = scf(KS(mol, PBE()))
    res = scf(KS(mol, PBE(), grid=becke(75, 302), coulomb=exact()))
    print(res.e_tot)

The common entry points are importable from ``dftax``; the rest of the API
lives in ``dftax.integrals``, ``dftax.energy``, ``dftax.grid``, ``dftax.ks``
and ``dftax.pbc``.
"""

from dftax.energy.d3 import d3bj
from dftax.energy.d4 import d4
from dftax.energy.gto import BasisData, extract_basis_data, eval_gto
from dftax.energy.xc import functional
from dftax.grid import becke, points
from dftax.integrals import (
    overlap_matrix,
    kinetic_matrix,
    nuclear_attraction_matrix,
    nuclear_repulsion,
    eri2c_matrix,
    eri3c_matrix,
)
from dftax.ks import (
    KS, System, exact, df, mesh, distributed, is_coordinator, barrier,
    core, sad, minao, sap,
    scf, adiis, fermi, newton, roks, minimize, gdm, forces, scf_batched,
    KSResult,
    BatchedResult,
    dipole, polarizability, hessian, vibrations, ir_spectrum, raman_spectrum,
    alchemical_deriv, implicit_density,
)
from dftax.system.molecule import Molecule

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("dftax")
except PackageNotFoundError:
    __version__ = "0.0.0+local"
del _pkg_version, PackageNotFoundError

__all__ = [
    "__version__",
    "KS", "System", "Molecule", "exact", "df", "becke", "points", "mesh",
    "distributed", "is_coordinator", "barrier",
    "core", "sad", "minao", "sap",
    "functional",
    "d3bj",
    "d4",
    "scf", "adiis", "fermi", "newton", "roks", "minimize", "gdm", "forces",
    "scf_batched",
    "KSResult",
    "BatchedResult",
    "dipole", "polarizability", "hessian", "vibrations", "ir_spectrum",
    "raman_spectrum", "alchemical_deriv",
    "implicit_density",
    "BasisData", "extract_basis_data", "eval_gto",
    "overlap_matrix", "kinetic_matrix", "nuclear_attraction_matrix",
    "nuclear_repulsion", "eri2c_matrix", "eri3c_matrix",
]
