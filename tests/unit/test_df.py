"""Density fitting (RI-J / RI-K) reproduces the exact-ERI energy.

With a standard JK-fitting auxiliary basis, the RI Coulomb (and RI exact
exchange, for hybrids) should match the full 4-center result to sub-mHa.
"""

import jax.numpy as jnp
import pytest

from pyscf import gto, dft

from dftax.energy.xc import LDA, PBE0
from dftax.ks.energy import RKS
from dftax.ks.scf import rks_scf

H2O = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"
AUX = "def2-universal-jkfit"


@pytest.mark.pyscf
@pytest.mark.float64
class TestDensityFitting:
    # LDA exercises RI-J; PBE0 additionally exercises RI-K.
    @pytest.mark.parametrize("xc_cls,pyscf_xc", [(LDA, "slater,vwn5"), (PBE0, "pbe0")])
    def test_ri_matches_full_eri(self, xc_cls, pyscf_xc):
        mol = gto.M(atom=H2O, basis="sto-3g").build()
        mf = dft.RKS(mol)
        mf.xc = pyscf_xc
        mf.grids.level = 3
        mf.verbose = 0
        mf.kernel()
        grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))

        e_full = rks_scf(RKS.from_pyscf(mol, xc_cls(), grid[0], grid[1])).e_tot
        res_df = rks_scf(
            RKS.from_pyscf(mol, xc_cls(), grid[0], grid[1], auxbasis=AUX)
        )
        assert res_df.converged
        assert res_df.P is not None
        # RI error vs the exact 4-center result: sub-mHa.
        assert abs(res_df.e_tot - e_full) < 1e-3, f"RI err {res_df.e_tot - e_full}"
