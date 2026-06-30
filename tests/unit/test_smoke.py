"""Fast end-to-end smoke tests: the gate that runs on every PR.

These cover the core RKS and UKS energy path, an analytic force, and the run_ks
dispatch on tiny systems, so a broken end-to-end path fails CI quickly. The
exhaustive validation (functional sweeps, finite-difference properties, CPHF,
density fitting and streaming, forces) lives in the heavier modules that are
auto-marked slow in conftest and run nightly.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import dft, gto

from dftax.energy.xc import LDA
from dftax.system.molecule import Molecule
from dftax.ks.energy import RKS
from dftax.ks.energy_uks import UKS
from dftax.ks.scf import rks_scf
from dftax.ks.scf_uks import uks_scf, UKSResult
from dftax.ks.driver import run_ks
from dftax.ks.forces import rks_forces
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
    res = rks_scf(RKS.from_pyscf(water_mol, LDA(), grid[0], grid[1]))
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
    res = uks_scf(UKS.from_pyscf(mol, LDA(), grid[0], grid[1]))
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
        return rks_scf(RKS.from_molecule(m, xc, gc, gw), e_tol=1e-10, d_tol=1e-8).e_tot

    gc, gw = becke_grid(mol.symbols, c0, NR, LEB)
    res = rks_scf(RKS.from_molecule(mol, xc, gc, gw), e_tol=1e-10, d_tol=1e-8)
    F = rks_forces(mol, xc, res.mo_coeff[:, :1], n_radial=NR, lebedev=LEB)
    assert float(np.abs(np.asarray(F.sum(axis=0))).max()) < 1e-8
    eps = 1e-3
    cp, cm = c0.copy(), c0.copy()
    cp[1, 2] += eps
    cm[1, 2] -= eps
    fd = -(energy(cp) - energy(cm)) / (2 * eps)
    assert abs(float(F[1, 2]) - fd) < 1e-4


@pytest.mark.float64
def test_smoke_run_ks_dispatch():
    """run_ks routes by spin: open shell -> UKS, closed shell -> RKS."""
    rad = Molecule.from_xyz("Li 0 0 0", "sto-3g", spin=1)
    r_open = run_ks(rad, LDA(), n_radial=30, lebedev=50)
    assert isinstance(r_open, UKSResult)

    h2 = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
    r_closed = run_ks(h2, LDA(), n_radial=30, lebedev=50)
    assert not isinstance(r_closed, UKSResult)
    assert r_closed.converged
