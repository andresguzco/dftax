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
def test_newton_escapes_indefinite_hessian():
    """A strongly stretched N2 (2.5 A) sits at an ill-conditioned/indefinite
    point where plain-CG Newton stalls: the decrease-only trust region cannot
    make progress from the direction a positive-definite CG solve returns, so
    it never converges. The Steihaug-Toint truncated-CG step follows negative
    curvature to the trust boundary and converges to a critical point (which
    restricted-symmetry critical point of a stretched bond is basin-dependent;
    the claim here is anti-stall, not global optimality -- easy cases with a
    clear minimum match DIIS, see test_newton_matches_scf_on_easy_case).
    """
    mol = Molecule.from_xyz("N 0 0 0; N 0 0 2.5", "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r1 = newton(ks, max_iter=60)
    assert r1.converged                                # no longer stalls
    assert r1.n_iter <= 40
    assert -108.0 < float(r1.e_tot) < -106.0           # a physical N2 solution


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
