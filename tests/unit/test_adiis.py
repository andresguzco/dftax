import jax.numpy as jnp
import numpy as np
import pytest

from dftax import KS, Molecule, adiis, becke, scf
from dftax.energy.xc import PBE
from dftax.ks.scf import _project_simplex, fermi

WATER = "O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50"


@pytest.mark.float64
def test_simplex_projection():
    p = _project_simplex(jnp.asarray([0.5, 0.5]))
    assert np.allclose(np.asarray(p), [0.5, 0.5], atol=1e-14)
    p = _project_simplex(jnp.asarray([2.0, 0.0]))
    assert np.allclose(np.asarray(p), [1.0, 0.0], atol=1e-14)
    p = _project_simplex(jnp.asarray([-1.0, 1.0]))
    assert np.allclose(np.asarray(p), [0.0, 1.0], atol=1e-14)
    p = _project_simplex(jnp.asarray([0.3, -1e30, 0.9]))
    assert float(p[1]) == 0.0
    assert np.isclose(float(jnp.sum(p)), 1.0, atol=1e-14)
    assert bool(jnp.all(p >= 0.0))


@pytest.mark.float64
def test_adiis_matches_plain_diis_on_easy_case():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r0 = scf(ks)
    r1 = scf(ks, accel=adiis())
    assert r0.converged and r1.converged
    assert abs(r1.e_tot - r0.e_tot) < 1e-7


@pytest.mark.float64
def test_degenerate_cr_atom_converges_with_smearing():
    mol = Molecule.from_xyz("Cr 0 0 0", "sto-3g", spin=6)
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=6)
    rf = scf(ks, max_iter=200, smearing=fermi(sigma=0.01))
    assert rf.converged


@pytest.mark.float64
def test_adiis_open_shell_parity():
    mol = Molecule.from_xyz("N 0 0 0; H 1.01 0 0; H -0.26 0.98 0", "sto-3g", spin=1)
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=1)
    r0 = scf(ks)
    r1 = scf(ks, accel=adiis())
    assert r0.converged and r1.converged
    assert abs(r1.e_tot - r0.e_tot) < 1e-7


@pytest.mark.float64
def test_adiis_is_a_pure_acceleration():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    P0 = scf(ks).P
    P1 = scf(ks, accel=adiis()).P
    assert float(jnp.abs(P1 - P0).max()) < 1e-5
