"""Cauchy-Schwarz ERI screening (Phase 4b A5).

|(μν|λσ)| ≤ √(μν|μν)·√(λσ|λσ): quartets below a threshold are skipped entirely.
Screening must (a) actually drop negligible quartets, (b) leave the retained
integrals exact, and (c) be exact-path only (rejected under density fitting).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import gto, dft

from dftax.energy.gto import extract_basis_data
from dftax.energy.xc import LDA
from dftax.integrals.eri4c import eri4c_matrix, screened_quartets, _unique_quartets
from dftax import KS, df, exact, scf
from dftax.ks.terms import DFCoulomb

AUX = "def2-universal-jkfit"


@pytest.mark.pyscf
@pytest.mark.float64
def test_screening_drops_negligible_quartets_exactly():
    # Two H2 molecules 20 bohr apart: the cross-molecule quartets are negligible.
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 20.0; H 0 0 20.74",
        basis="sto-3g", spin=0,
    ).build()
    b = extract_basis_data(mol)
    n = b.centers.shape[0]
    nuniq = len(_unique_quartets(n)[0])

    full = np.asarray(eri4c_matrix(b))                       # unscreened (A4)
    q, qof = screened_quartets(b, 1e-10)
    scr = np.asarray(eri4c_matrix(b, quartets=jnp.asarray(q), qof=jnp.asarray(qof)))

    assert q.shape[0] < nuniq, "screening removed nothing"   # (a)
    assert np.max(np.abs(scr - full)) < 1e-12                # (b) retained exact
    assert np.max(np.abs(full - mol.intor("int2e"))) < 1e-10


@pytest.mark.pyscf
@pytest.mark.float64
def test_screened_scf_matches_unscreened(water_mol):
    mf = dft.RKS(water_mol)
    mf.xc = "slater,vwn5"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))

    # coulomb=exact(): the screened side is exact, and the default is now DF;
    # an unpinned build would compare DF vs exact (RI error, not screening).
    e_unscr = float(
        scf(KS(water_mol, LDA(), grid=(grid[0], grid[1]), coulomb=exact())).e_tot
    )
    res = scf(
        KS(water_mol, LDA(), grid=(grid[0], grid[1]), coulomb=exact(screen=1e-10))
    )
    assert res.converged
    assert abs(res.e_tot - e_unscr) < 1e-8, f"screened {res.e_tot} vs {e_unscr}"


@pytest.mark.pyscf
def test_eri_screen_is_exact_path_only(water_mol):
    # Materialized density fitting has no 4-center tensor to screen: df()
    # rejects a bare Schwarz threshold (it applies only to the streamed RI-J
    # path), so an ERI screen can never leak into the DF backend.
    with pytest.raises(ValueError):
        df(AUX, screen=1e-10)
    mf = dft.RKS(water_mol)
    mf.xc = "slater,vwn5"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks = KS(water_mol, LDA(), grid=(grid[0], grid[1]), coulomb=df(AUX))
    assert isinstance(ks.coulomb, DFCoulomb)
