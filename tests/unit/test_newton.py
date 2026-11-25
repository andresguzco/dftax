import pytest

from dftax import KS, Molecule, becke, newton, scf
from dftax.energy.xc import PBE

WATER = "O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50"


@pytest.mark.float64
def test_newton_matches_scf_on_easy_case():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r0 = scf(ks)
    r1 = newton(ks)
    assert r0.converged and r1.converged
    assert abs(r1.e_tot - r0.e_tot) < 1e-7
    assert r1.n_iter <= r0.n_iter


@pytest.mark.float64
def test_newton_warm_start_is_quadratic_cleanup():
    mol = Molecule.from_xyz("O 0 0 0; O 0 0 1.21", "sto-3g", spin=2)
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=2)
    r0 = scf(ks)
    r1 = newton(ks, guess=r0.P)
    assert r0.converged and r1.converged
    assert r1.n_iter <= 3
    assert abs(r1.e_tot - r0.e_tot) < 1e-7


@pytest.mark.float64
def test_newton_cold_start_stretched_bond():
    mol = Molecule.from_xyz("N 0 0 0; N 0 0 2.0", "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r0 = scf(ks)
    r1 = newton(ks)
    assert r0.converged and r1.converged
    assert r1.n_iter <= 8
    assert abs(r1.e_tot - r0.e_tot) < 1e-7


@pytest.mark.float64
def test_newton_escapes_indefinite_hessian():
    mol = Molecule.from_xyz("N 0 0 0; N 0 0 2.5", "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r1 = newton(ks, max_iter=120)
    assert r1.converged
    assert r1.n_iter <= 60
    assert -108.0 < float(r1.e_tot) < -106.0


@pytest.mark.float64
def test_newton_reaches_tight_tolerances_directly():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r1 = newton(ks, g_tol=1e-7, e_tol=1e-12)
    assert r1.converged
    assert r1.n_iter <= 50
