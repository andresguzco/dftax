import jax
import jax.numpy as jnp
import pytest

from pyscf import dft, gto

from dftax.basis.loader import build_basis_data
from dftax.energy.xc import LDA, PBE, PBE0
from dftax import KS, df
from dftax.ks.terms import _streamed_df_rij

AUX = "def2-universal-jkfit"


def _e2_rks(ks, P):
    return ks.coulomb.energy(P[None], ks.S, (ks.nelec // 2,))


def _e2_uks(ks, Pa, Pb):
    return ks.coulomb.energy(jnp.stack([Pa, Pb]), ks.S, (ks.nocc[0], ks.nocc[1]))


def _ch3():
    return gto.M(
        atom="C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0",
        basis="sto-3g", spin=1,
    ).build()


@pytest.mark.pyscf
@pytest.mark.float64
def test_streamed_rij_matches_materialized(water_mol):
    mf = dft.RKS(water_mol)
    mf.xc = "slater,vwn5"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))

    ks = KS(water_mol, LDA(), grid=(grid[0], grid[1]),
            coulomb=df(AUX, spherical=False, whiten=False))
    P = jnp.asarray(mf.make_rdm1())

    aux = build_basis_data(
        [water_mol.atom_symbol(i) for i in range(water_mol.natm)],
        water_mol.atom_coords(), AUX,
    )
    g = jnp.einsum("mnP,mn->P", ks.coulomb.int3c, P)
    ej_mat = 0.5 * float(jnp.dot(g, ks.coulomb.int2c_inv @ g))
    ej_str = float(_streamed_df_rij(ks.basis, aux, ks.coulomb.int2c_inv, P, 50))
    assert abs(ej_str - ej_mat) < 1e-8, f"streamed {ej_str} vs mat {ej_mat}"


@pytest.mark.slow
@pytest.mark.pyscf
@pytest.mark.float64
def test_df_chunk_hybrid_matches_materialized(water_mol):
    mf = dft.RKS(water_mol)
    mf.xc = "pbe0"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_mat = KS(water_mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, spherical=False))
    P = jnp.asarray(mf.make_rdm1())
    ks_str = KS(water_mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, chunk=50))
    e_mat = float(_e2_rks(ks_mat, P))
    e_str = float(_e2_rks(ks_str, P))
    assert abs(e_str - e_mat) < 1e-8, f"streamed RI-J+RI-K {e_str} vs mat {e_mat}"


@pytest.mark.slow
@pytest.mark.pyscf
@pytest.mark.float64
def test_screened_df_chunk_matches_dense(water_mol):
    mf = dft.RKS(water_mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_dense = KS(water_mol, PBE(), grid=(grid[0], grid[1]), coulomb=df(AUX, chunk=50))
    P = jnp.asarray(mf.make_rdm1())
    ks_scr = KS(water_mol, PBE(), grid=(grid[0], grid[1]),
                coulomb=df(AUX, chunk=50, screen=1e-10))
    e_dense = float(_e2_rks(ks_dense, P))
    e_scr = float(_e2_rks(ks_scr, P))
    assert abs(e_scr - e_dense) < 1e-8, f"screened {e_scr} vs dense {e_dense}"


@pytest.mark.slow
@pytest.mark.pyscf
@pytest.mark.float64
def test_streamed_rik_fock_matches_materialized(water_mol):
    mf = dft.RKS(water_mol)
    mf.xc = "pbe0"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_mat = KS(water_mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, spherical=False))
    P = jnp.asarray(mf.make_rdm1())
    ks_str = KS(water_mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, chunk=50))

    sym = lambda M: 0.5 * (M + M.T)
    F_mat = sym(jax.grad(lambda Q: _e2_rks(ks_mat, Q))(P))
    F_str = sym(jax.grad(lambda Q: _e2_rks(ks_str, Q))(P))
    assert float(jnp.max(jnp.abs(F_str - F_mat))) < 1e-6


@pytest.mark.slow
@pytest.mark.pyscf
@pytest.mark.float64
def test_streamed_uks_hybrid_empty_beta():
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1).build()
    mf = dft.UKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_mat = KS(mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, spherical=False), spin=mol.spin)
    Pa, Pb = jnp.asarray(mf.make_rdm1())
    ks_str = KS(
        mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, chunk=50), spin=mol.spin
    )
    e_mat = float(_e2_uks(ks_mat, Pa, Pb))
    e_str = float(_e2_uks(ks_str, Pa, Pb))
    assert abs(e_str - e_mat) < 1e-8, f"empty-β streamed {e_str} vs mat {e_mat}"


@pytest.mark.slow
@pytest.mark.pyscf
@pytest.mark.float64
def test_streamed_uks_rik_matches_materialized():
    mol = _ch3()
    mf = dft.UKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_mat = KS(mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, spherical=False), spin=mol.spin)
    Pa, Pb = jnp.asarray(mf.make_rdm1())
    ks_str = KS(
        mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, chunk=50), spin=mol.spin
    )
    e_mat = float(_e2_uks(ks_mat, Pa, Pb))
    e_str = float(_e2_uks(ks_str, Pa, Pb))
    assert abs(e_str - e_mat) < 1e-8, f"UKS streamed RI-K {e_str} vs mat {e_mat}"
