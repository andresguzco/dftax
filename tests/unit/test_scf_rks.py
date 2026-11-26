import jax
import jax.numpy as jnp
import pytest

from pyscf import dft

from dftax.energy.xc import PBE, PBE0
from dftax import KS, scf, exact
from dftax.ks.scf import canonical_orthonormalizer

FUNCTIONALS = {
    "pbe": (PBE(), "pbe"),
    "pbe0": (PBE0(), "pbe0"),
}


def _pyscf_ref(mol, pyscf_xc, level=3):
    mf = dft.RKS(mol)
    mf.xc = pyscf_xc
    mf.grids.level = level
    mf.verbose = 0
    e_ref = mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    return float(e_ref), grid


@pytest.mark.pyscf
@pytest.mark.float64
class TestRKSEnergyMatchesPyscf:
    @pytest.mark.parametrize("key", list(FUNCTIONALS))
    def test_water(self, water_mol, key):
        xc_obj, pyscf_xc = FUNCTIONALS[key]
        e_ref, grid = _pyscf_ref(water_mol, pyscf_xc)
        res = scf(KS(water_mol, xc_obj, grid=grid, coulomb=exact()))
        assert res.converged
        assert abs(res.e_tot - e_ref) < 5e-5, f"{key}: {res.e_tot} vs {e_ref}"

    def test_h2_pbe(self, h2_mol):
        xc_obj, pyscf_xc = FUNCTIONALS["pbe"]
        e_ref, grid = _pyscf_ref(h2_mol, pyscf_xc)
        res = scf(KS(h2_mol, xc_obj, grid=grid, coulomb=exact()))
        assert res.converged
        assert abs(res.e_tot - e_ref) < 5e-5


@pytest.mark.pyscf
@pytest.mark.float64
class TestAutodiffFock:
    def test_fock_matches_fd(self, water_mol, key):
        xc_obj, _ = FUNCTIONALS["pbe"]
        _, grid = _pyscf_ref(water_mol, "pbe")
        ks = KS(
            water_mol, xc_obj, grid=(jnp.asarray(grid[0]), jnp.asarray(grid[1])),
            coulomb=exact(),
        )

        X = canonical_orthonormalizer(ks.S)
        _, Cp = jnp.linalg.eigh(X.T @ ks.hcore @ X)
        C = X @ Cp
        nocc = ks.nelec // 2
        P = (2.0 * C[:, :nocc] @ C[:, :nocc].T)[None]

        g = jax.grad(ks.electronic)(P)
        F = 0.5 * (g + g.transpose(0, 2, 1))

        nao = P.shape[-1]
        dP = jax.random.normal(key, (nao, nao))
        dP = 0.5 * (dP + dP.T)
        dP = dP / jnp.linalg.norm(dP)
        eps = 1e-5
        fd = (
            float(ks.electronic(P + eps * dP[None]))
            - float(ks.electronic(P - eps * dP[None]))
        ) / (2 * eps)
        ad = float(jnp.sum(F[0] * dP))
        assert abs(ad - fd) < 5e-6, f"Fock AD={ad} vs FD={fd}"
