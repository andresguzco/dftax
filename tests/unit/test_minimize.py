"""Direct variational minimization reaches the SCF ground state.

The Adam-based direct minimizer optimizes E(P) over Löwdin-orthonormalized
coefficients; at convergence it must reach the same energy as the DIIS SCF
solver (and hence PySCF).
"""

import jax.numpy as jnp
import pytest

from pyscf import gto, dft

from dftax.energy.xc import LDA, PBE
from dftax import KS, scf, minimize

H2O = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"


@pytest.mark.pyscf
@pytest.mark.float64
class TestDirectMinimization:
    @pytest.mark.parametrize("xc_cls,pyscf_xc", [(LDA, "slater,vwn5"), (PBE, "pbe")])
    def test_matches_scf(self, xc_cls, pyscf_xc):
        mol = gto.M(atom=H2O, basis="sto-3g").build()
        mf = dft.RKS(mol)
        mf.xc = pyscf_xc
        mf.grids.level = 3
        mf.verbose = 0
        mf.kernel()
        grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))

        ks = KS(mol, xc_cls(), grid=(grid[0], grid[1]))
        e_scf = scf(ks).e_tot
        res = minimize(ks, max_steps=4000)

        assert res.converged, "direct minimization did not converge"
        assert abs(res.e_tot - e_scf) < 1e-6, f"min {res.e_tot} vs scf {e_scf}"


@pytest.mark.float64
def test_params_dependent_optimizer():
    """opt.update receives the params, so transforms that need them work."""
    import optax
    from dftax import Molecule, becke

    mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
    ks = KS(mol, LDA(), grid=becke(35, 50))
    opt = optax.chain(optax.add_decayed_weights(0.0), optax.adam(0.3))
    res = minimize(ks, opt, max_steps=1500)
    assert res.converged
    assert abs(res.e_tot - scf(ks).e_tot) < 1e-6
