"""Second-order SCF (newton): parity, quadratic cleanup, hard-case roles."""

import warnings

import pytest

from dftax import KS, Molecule, becke, newton, scf
from dftax.energy.xc import PBE

WATER = "O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50"


@pytest.mark.float64
def test_newton_matches_scf_on_easy_case():
    """Same fixed point as DIIS, in fewer iterations (quadratic tail)."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r0 = scf(ks)
    r1 = newton(ks)
    assert r0.converged and r1.converged
    assert abs(r1.e_tot - r0.e_tot) < 1e-7
    assert r1.n_iter <= r0.n_iter


@pytest.mark.float64
def test_newton_warm_start_is_quadratic_cleanup():
    """From a converged density the Newton step count is O(1) (Fe/sto-3g,
    the open-shell case where ADIIS is unfavorable)."""
    mol = Molecule.from_xyz("Fe 0 0 0", "sto-3g", spin=4)
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=4)
    r0 = scf(ks)
    r1 = newton(ks, guess=r0.P)
    assert r0.converged and r1.converged
    assert r1.n_iter <= 3
    assert abs(r1.e_tot - r0.e_tot) < 1e-7


@pytest.mark.float64
def test_newton_cold_start_stretched_bond():
    """Stretched N2 converges cold in a handful of Newton steps."""
    mol = Molecule.from_xyz("N 0 0 0; N 0 0 2.0", "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r0 = scf(ks)
    r1 = newton(ks)
    assert r0.converged and r1.converged
    assert r1.n_iter <= 8
    assert abs(r1.e_tot - r0.e_tot) < 1e-7


@pytest.mark.float64
def test_newton_reaches_tight_tolerances_directly():
    """The coarse-grid tight-tolerance case from the DF conditioning study:
    DIIS grinds against its noise floor (borderline non-convergence, >100
    iterations when it does close); Newton drives the orbital gradient to
    g_tol=1e-9 in a handful of steps."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = scf(ks, e_tol=1e-11, d_tol=1e-9, max_iter=128)
    r1 = newton(ks, g_tol=1e-9, e_tol=1e-12)
    assert r1.converged
    assert r1.n_iter <= 10
    assert (not r0.converged) or r0.n_iter >= 50
