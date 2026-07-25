"""Density fitting (RI-J / RI-K) reproduces the exact-ERI energy.

With a standard JK-fitting auxiliary basis, the RI Coulomb (and RI exact
exchange, for hybrids) should match the full 4-center result to sub-mHa.
"""

import jax.numpy as jnp
import pytest

from pyscf import gto, dft

from dftax.energy.xc import LDA, PBE0
from dftax import KS, df, scf

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

        e_full = scf(KS(mol, xc_cls(), grid=(grid[0], grid[1]))).e_tot
        res_df = scf(
            KS(mol, xc_cls(), grid=(grid[0], grid[1]), coulomb=df(AUX))
        )
        assert res_df.converged
        assert res_df.P is not None
        # RI error vs the exact 4-center result: sub-mHa.
        assert abs(res_df.e_tot - e_full) < 1e-3, f"RI err {res_df.e_tot - e_full}"


@pytest.mark.pyscf
@pytest.mark.float64
def test_spherical_aux_knob():
    """df(spherical=...) controls the auxiliary span on the materialized path.

    The default upgrades to spherical harmonics (fewer auxiliary functions,
    positive definite metric); spherical=False keeps the cartesian span the
    streamed and sharded backends use; spherical=True with a streamed chunk
    is rejected at the factory.
    """
    mol = gto.M(atom=H2O, basis="sto-3g").build()
    grid = jnp.zeros((8, 3)), jnp.zeros((8,))
    ks_sph = KS(mol, LDA(), grid=grid, coulomb=df(AUX))
    ks_cart = KS(mol, LDA(), grid=grid, coulomb=df(AUX, spherical=False))
    naux_sph = ks_sph.coulomb.int3c.shape[-1]
    naux_cart = ks_cart.coulomb.int3c.shape[-1]
    assert naux_sph < naux_cart  # h/i cartesian contaminants dropped
    with pytest.raises(ValueError):
        df(AUX, chunk=50, spherical=True)
