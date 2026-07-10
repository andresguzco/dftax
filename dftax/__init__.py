"""dftax: a self-contained, pure-JAX/Equinox Kohn-Sham DFT engine.

The package exposes a differentiable KS-DFT toolkit with no PySCF runtime
dependency for the core compute path:

- ``dftax.integrals``: analytical one- and two-electron integral matrices
  (overlap S, kinetic T, nuclear attraction V, nuclear repulsion, ERIs 2c/3c/4c)
  via the Obara-Saika recurrence, all jit/vmap/grad-friendly.
- ``dftax.energy``: GTO basis evaluation, the Boys function, density fitting,
  Hartree and hybrid exchange, exchange-correlation functionals, real-space
  grids, and pointwise potentials.
- ``dftax.grid``: native Becke quadrature + grid specs (``becke``, ``points``).
- ``dftax.utils``: chunked ``vmap`` helpers and shared types.
- ``dftax.ks``: the KS energy functional and solvers.

The canonical flow::

    from dftax import KS, Molecule, becke, df, scf
    from dftax.energy.xc import PBE

    mol = Molecule.from_xyz("O 0 0 0; H ...", "sto-3g")
    ks  = KS(mol, PBE())                        # exact ERI, default Becke grid
    ks  = KS(mol, PBE(), grid=becke(75, 302),   # or: choices as values
             coulomb=df("def2-universal-jkfit"))
    res = scf(ks)                               # DIIS; res.e_tot, res.P, ...

The most common entry points are re-exported here; import the submodules
directly for the full surface.
"""

from dftax.energy.gto import BasisData, extract_basis_data, eval_gto
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
    KS, System, exact, df,
    scf, minimize, forces, scf_batched, KSResult,
    dipole, polarizability, hessian, vibrations, ir_spectrum, raman_spectrum,
    alchemical_deriv, implicit_density,
)
from dftax.system.molecule import Molecule

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("dftax")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+local"
del _pkg_version, PackageNotFoundError

__all__ = [
    "__version__",
    # build: system + choices-as-values
    "KS", "System", "Molecule", "exact", "df", "becke", "points",
    # run: solver verbs + result
    "scf", "minimize", "forces", "scf_batched", "KSResult",
    # response properties
    "dipole", "polarizability", "hessian", "vibrations", "ir_spectrum",
    "raman_spectrum", "alchemical_deriv",
    # implicit-differentiation SCF response
    "implicit_density",
    # low-level building blocks
    "BasisData", "extract_basis_data", "eval_gto",
    "overlap_matrix", "kinetic_matrix", "nuclear_attraction_matrix",
    "nuclear_repulsion", "eri2c_matrix", "eri3c_matrix",
]
