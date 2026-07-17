"""Kohn-Sham DFT drivers built on the dftax integral/XC engine.

``dftax.ks`` assembles the pieces in ``dftax.integrals`` and ``dftax.energy``
into runnable calculations: the differentiable total-energy functional
:class:`~dftax.ks.energy.KS` (built with choices-as-values: grid specs from
:mod:`dftax.grid`, Coulomb backends from :func:`~dftax.ks.terms.exact` /
:func:`~dftax.ks.terms.df`), the solver verbs :func:`~dftax.ks.scf.scf`
(autodiff-Fock DIIS) and :func:`~dftax.ks.minimize.minimize` (direct
variational descent with any optax optimizer), analytic
:func:`~dftax.ks.forces.forces`, the batched :func:`~dftax.ks.batched.scf_batched`,
and the response-property layer. Restricted and spin-polarized systems run
through the same spin-stacked code path.
"""

from dftax.ks.energy import KS, System
from dftax.ks.guess import core, sad, minao, sap
from dftax.ks.shard import mesh, MeshSpec
from dftax.ks.terms import (
    exact, df,
    CoulombTerm, ExactCoulomb, StreamedExactCoulomb, DFCoulomb, StreamedDFCoulomb,
    XCTerm, GridXC, StreamedGridXC,
)
from dftax.ks.scf import scf, adiis, KSResult
from dftax.ks.minimize import minimize
from dftax.ks.forces import forces
from dftax.ks.batched import scf_batched, BatchedResult
from dftax.ks.properties import (
    dipole, polarizability, hessian, vibrations, ir_spectrum, raman_spectrum,
    alchemical_deriv, Vibrations, IRSpectrum, RamanSpectrum,
)
from dftax.ks.implicit import implicit_density

__all__ = [
    "KS", "System", "exact", "df", "mesh",
    "core", "sad", "minao", "sap",
    "scf", "minimize", "forces", "scf_batched", "KSResult", "BatchedResult",
    "CoulombTerm", "ExactCoulomb", "StreamedExactCoulomb", "DFCoulomb",
    "StreamedDFCoulomb", "XCTerm", "GridXC", "StreamedGridXC",
    "dipole", "polarizability", "hessian", "vibrations", "ir_spectrum",
    "raman_spectrum", "alchemical_deriv",
    "Vibrations", "IRSpectrum", "RamanSpectrum",
    "implicit_density",
]
