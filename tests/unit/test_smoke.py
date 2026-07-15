"""Fast end-to-end smoke tests: the gate that runs on every PR.

These cover the core restricted and spin-polarized KS energy path, an analytic
force, and the KS spin inference on tiny systems, so a broken end-to-end path
fails CI quickly. The exhaustive validation (functional sweeps,
finite-difference properties, CPHF, density fitting and streaming, forces)
lives in the heavier modules that are auto-marked slow in conftest and run
nightly.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import dft, gto

from dftax.energy.xc import LDA
from dftax.system.molecule import Molecule
from dftax import KS, becke, scf, forces, exact
from dftax.grid import becke_grid


@pytest.mark.pyscf
@pytest.mark.float64
def test_smoke_rks_energy(water_mol):
    """RKS water LDA energy matches PySCF on the same grid."""
    mf = dft.RKS(water_mol)
    mf.xc = "slater,vwn5"
    mf.grids.level = 1
    mf.verbose = 0
    e_ref = mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    res = scf(KS(water_mol, LDA(), grid=(grid[0], grid[1]), coulomb=exact()))
    assert res.converged
    assert abs(res.e_tot - e_ref) < 5e-5


@pytest.mark.pyscf
@pytest.mark.float64
def test_smoke_uks_energy():
    """UKS Li-atom doublet LDA energy matches PySCF on the same grid."""
    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1).build()
    mf = dft.UKS(mol)
    mf.xc = "slater,vwn5"
    mf.grids.level = 1
    mf.verbose = 0
    e_ref = mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    res = scf(KS(mol, LDA(), grid=(grid[0], grid[1]), spin=mol.spin, coulomb=exact()))
    assert res.converged
    assert abs(res.e_tot - e_ref) < 5e-5


@pytest.mark.float64
def test_smoke_force_matches_fd():
    """Analytic RKS force on H2 matches central finite difference (tiny grid)."""
    xc = LDA()
    NR, LEB = 30, 50
    mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.85", "sto-3g")
    c0 = mol.atom_coords()

    def energy(coords):
        m = Molecule(mol.symbols, coords, mol.basis)
        gc, gw = becke_grid(m.symbols, m.atom_coords(), NR, LEB)
        return scf(KS(m, xc, grid=(gc, gw)), e_tol=1e-10, d_tol=1e-8).e_tot

    gc, gw = becke_grid(mol.symbols, c0, NR, LEB)
    res = scf(KS(mol, xc, grid=(gc, gw)), e_tol=1e-10, d_tol=1e-8)
    F = forces(mol, xc, (res.mo_coeff[0][:, :1],), grid=becke(NR, LEB))
    assert float(np.abs(np.asarray(F.sum(axis=0))).max()) < 1e-8
    eps = 1e-3
    cp, cm = c0.copy(), c0.copy()
    cp[1, 2] += eps
    cm[1, 2] -= eps
    fd = -(energy(cp) - energy(cm)) / (2 * eps)
    assert abs(float(F[1, 2]) - fd) < 1e-4


@pytest.mark.float64
def test_smoke_spin_dispatch():
    """KS infers spin: open shell -> two channels, closed shell -> one."""
    rad = Molecule.from_xyz("Li 0 0 0", "sto-3g", spin=1)
    r_open = scf(KS(rad, LDA(), grid=becke(30, 50)))
    assert len(r_open.nocc) == 2

    h2 = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
    r_closed = scf(KS(h2, LDA(), grid=becke(30, 50)))
    assert len(r_closed.nocc) == 1
    assert r_closed.converged
