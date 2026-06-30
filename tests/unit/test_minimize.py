"""Direct variational minimization reaches the SCF ground state.

The Adam-based direct minimizer optimizes E(P) over Löwdin-orthonormalized
coefficients; at convergence it must reach the same energy as the DIIS SCF
solver (and hence PySCF).
"""

import jax.numpy as jnp
import pytest

from pyscf import gto, dft

from dftax.energy.xc import LDA, PBE
from dftax.ks.energy import RKS
from dftax.ks.scf import rks_scf
from dftax.ks.minimize import rks_minimize

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

        ks = RKS.from_pyscf(mol, xc_cls(), grid[0], grid[1])
        e_scf = rks_scf(ks).e_tot
        res = rks_minimize(ks, learning_rate=0.3, max_steps=4000)

        assert res.converged, "direct minimization did not converge"
        assert abs(res.e_tot - e_scf) < 1e-6, f"min {res.e_tot} vs scf {e_scf}"
