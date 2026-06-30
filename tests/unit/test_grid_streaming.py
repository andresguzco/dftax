"""Chunked/streamed XC grid matches the materialized path.

With ``grid_chunk`` set, the AO grid is recomputed per chunk (O(chunk·nao)
memory) instead of materialized; the resulting energy must be identical.
"""

import jax.numpy as jnp
import pytest

from pyscf import dft

from dftax.energy.xc import LDA, PBE
from dftax.ks.energy import RKS
from dftax.ks.scf import rks_scf


@pytest.mark.pyscf
@pytest.mark.float64
@pytest.mark.parametrize("xc_cls,pyscf_xc", [(LDA, "slater,vwn5"), (PBE, "pbe")])
def test_streamed_matches_materialized(xc_cls, pyscf_xc, water_mol):
    mf = dft.RKS(water_mol)
    mf.xc = pyscf_xc
    mf.grids.level = 2
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))

    e_mat = rks_scf(RKS.from_pyscf(water_mol, xc_cls(), grid[0], grid[1])).e_tot
    res = rks_scf(
        RKS.from_pyscf(water_mol, xc_cls(), grid[0], grid[1], grid_chunk=2000)
    )
    assert res.converged
    assert abs(res.e_tot - e_mat) < 1e-9, f"streamed {res.e_tot} vs mat {e_mat}"
