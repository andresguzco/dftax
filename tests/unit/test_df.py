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
    @pytest.mark.parametrize("xc_cls,pyscf_xc", [(PBE0, "pbe0")])
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
        assert abs(res_df.e_tot - e_full) < 1e-3, f"RI err {res_df.e_tot - e_full}"


@pytest.mark.pyscf
@pytest.mark.float64
def test_spherical_aux_knob():
    mol = gto.M(atom=H2O, basis="sto-3g").build()
    grid = jnp.zeros((8, 3)), jnp.zeros((8,))
    ks_sph = KS(mol, LDA(), grid=grid, coulomb=df(AUX, whiten=False))
    ks_cart = KS(mol, LDA(), grid=grid,
                 coulomb=df(AUX, spherical=False, whiten=False))
    naux_sph = ks_sph.coulomb.int3c.shape[-1]
    naux_cart = ks_cart.coulomb.int3c.shape[-1]
    assert naux_sph < naux_cart
    with pytest.raises(ValueError):
        df(AUX, chunk=50, spherical=True)
