"""Kohn-Sham DFT drivers built on the dftax integral/XC engine.

``dftax.ks`` assembles the pieces in ``dftax.integrals`` and ``dftax.energy``
into runnable calculations: differentiable total-energy functionals of the
density matrix (closed-shell :class:`~dftax.ks.energy.RKS`, open-shell
:class:`~dftax.ks.energy.UKS`), autodiff-Fock SCF loops with Pulay DIIS
(:func:`~dftax.ks.scf.rks_scf`, :func:`~dftax.ks.scf.uks_scf`), direct
minimizers, analytic forces, and high-level entry points, including the unified
:func:`~dftax.ks.driver.run_ks` that dispatches restricted/unrestricted by spin.
"""

from dftax.ks.energy import KS, RKS, UKS
from dftax.ks.terms import (
    exact, df,
    CoulombTerm, ExactCoulomb, StreamedExactCoulomb, DFCoulomb, StreamedDFCoulomb,
    XCTerm, GridXC, StreamedGridXC,
)
from dftax.ks.scf import rks_scf, uks_scf, SCFResult, UKSResult
from dftax.ks.minimize import rks_minimize, uks_minimize
from dftax.ks.forces import rks_forces, uks_forces
from dftax.ks.driver import run_rks, run_uks, run_ks
from dftax.ks.batched import (
    run_ks_batched, run_rks_batched, run_uks_batched, BatchedResult,
)
from dftax.ks.properties import (
    dipole, polarizability, hessian, vibrations, ir_spectrum, raman_spectrum,
    alchemical_deriv,
)
from dftax.ks.implicit import implicit_density

__all__ = [
    "KS", "exact", "df",
    "CoulombTerm", "ExactCoulomb", "StreamedExactCoulomb", "DFCoulomb",
    "StreamedDFCoulomb", "XCTerm", "GridXC", "StreamedGridXC",
    "RKS", "rks_scf", "rks_minimize", "rks_forces", "SCFResult", "run_rks",
    "UKS", "uks_scf", "uks_minimize", "uks_forces", "UKSResult", "run_uks",
    "run_ks",
    "run_ks_batched", "run_rks_batched", "run_uks_batched", "BatchedResult",
    "dipole", "polarizability", "hessian", "vibrations", "ir_spectrum",
    "raman_spectrum", "alchemical_deriv",
    "implicit_density",
]
