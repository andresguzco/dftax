"""Streamed RI-J / RI-K (df(..., chunk=)) matches materialized density fitting.

The streamed path forms γ_P (and, for hybrids, the orbital-chunk RI-K exchange)
without materializing the nao²×naux 3-center tensor; the DF energy must match the
materialized DF path. The materialized references pin ``spherical=False``: the
streamed backend contracts cartesian auxiliary elements, and these tests verify
the streaming mechanics, which needs both sides in the same fit space (the
default materialized path upgrades to a spherical auxiliary basis).
"""

import jax
import jax.numpy as jnp
import pytest

from pyscf import dft, gto

from dftax.basis.loader import build_basis_data
from dftax.energy.xc import LDA, PBE, PBE0
from dftax import KS, df, scf
from dftax.ks.terms import _streamed_df_rij

AUX = "def2-universal-jkfit"


def _e2_rks(ks, P):
    """Coulomb + exact-exchange energy of a closed-shell ks at density P."""
    return ks.coulomb.energy(P[None], ks.S, (ks.nelec // 2,))


def _e2_uks(ks, Pa, Pb):
    """Coulomb + exact-exchange energy of an open-shell ks at (Pα, Pβ)."""
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

    ks = KS(water_mol, LDA(), grid=(grid[0], grid[1]), coulomb=df(AUX, spherical=False))
    P = scf(ks).P[0]

    aux = build_basis_data(
        [water_mol.atom_symbol(i) for i in range(water_mol.natm)],
        water_mol.atom_coords(), AUX,
    )
    g = jnp.einsum("mnP,mn->P", ks.coulomb.int3c, P)
    ej_mat = 0.5 * float(jnp.dot(g, ks.coulomb.int2c_inv @ g))
    ej_str = float(_streamed_df_rij(ks.basis, aux, ks.coulomb.int2c_inv, P, 50))
    assert abs(ej_str - ej_mat) < 1e-8, f"streamed {ej_str} vs mat {ej_mat}"


@pytest.mark.slow  # streamed RI-K custom_vjp compile is heavy on CPU
@pytest.mark.pyscf
@pytest.mark.float64
def test_df_chunk_hybrid_matches_materialized(water_mol):
    """Hybrid + df(chunk=) streams RI-K (orbital-chunk custom_vjp); the DF
    energy must match the materialized DF hybrid. Fixed-P (no streamed SCF loop)
    keeps the unit test cheap."""
    mf = dft.RKS(water_mol)
    mf.xc = "pbe0"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_mat = KS(water_mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, spherical=False))
    P = scf(ks_mat).P[0]
    ks_str = KS(water_mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, chunk=50))
    e_mat = float(_e2_rks(ks_mat, P))
    e_str = float(_e2_rks(ks_str, P))
    assert abs(e_str - e_mat) < 1e-8, f"streamed RI-J+RI-K {e_str} vs mat {e_mat}"


@pytest.mark.slow  # two streamed-RI-J compiles (dense + screened shapes)
@pytest.mark.pyscf
@pytest.mark.float64
def test_screened_df_chunk_matches_dense(water_mol):
    """Schwarz-screened streamed RI-J (df(screen=)) matches the dense streamed RI-J
    at a fixed density (water has no negligible pairs, so screening is exact)."""
    mf = dft.RKS(water_mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_dense = KS(water_mol, PBE(), grid=(grid[0], grid[1]), coulomb=df(AUX, chunk=50))
    P = scf(KS(water_mol, PBE(), grid=(grid[0], grid[1]), coulomb=df(AUX, spherical=False))).P[0]
    ks_scr = KS(water_mol, PBE(), grid=(grid[0], grid[1]),
                coulomb=df(AUX, chunk=50, screen=1e-10))
    e_dense = float(_e2_rks(ks_dense, P))
    e_scr = float(_e2_rks(ks_scr, P))
    assert abs(e_scr - e_dense) < 1e-8, f"screened {e_scr} vs dense {e_dense}"


@pytest.mark.slow
@pytest.mark.pyscf
@pytest.mark.float64
def test_streamed_rik_fock_matches_materialized(water_mol):
    """The streamed RI-K ``custom_vjp`` backward must reproduce the materialized DF
    exchange Fock at the (idempotent) converged density, i.e. it yields the right
    KS Fock for the SCF, which is its only sanctioned gradient use."""
    mf = dft.RKS(water_mol)
    mf.xc = "pbe0"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_mat = KS(water_mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, spherical=False))
    P = scf(ks_mat).P[0]
    ks_str = KS(water_mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, chunk=50))

    sym = lambda M: 0.5 * (M + M.T)
    F_mat = sym(jax.grad(lambda Q: _e2_rks(ks_mat, Q))(P))  # materialized: full autodiff
    F_str = sym(jax.grad(lambda Q: _e2_rks(ks_str, Q))(P))  # streamed: custom_vjp backward
    assert float(jnp.max(jnp.abs(F_str - F_mat))) < 1e-6


@pytest.mark.slow
@pytest.mark.pyscf
@pytest.mark.float64
def test_streamed_uks_hybrid_empty_beta():
    """Regression (nocc=0 bug): streamed UKS RI-K with an EMPTY β channel (nbeta=0,
    here the H atom) must match the materialized DF hybrid. Before the fix the
    occupied slice ``evec[:, -nocc:]`` returned ALL orbitals for nocc=0, giving a
    catastrophically wrong β exchange."""
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1).build()
    mf = dft.UKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_mat = KS(mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, spherical=False), spin=mol.spin)
    Pa, Pb = scf(ks_mat).P
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
    """Streamed per-spin UKS RI-K (df(chunk=)) == materialized DF hybrid, on a doublet
    with a non-empty β channel (CH₃)."""
    mol = _ch3()
    mf = dft.UKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_mat = KS(mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, spherical=False), spin=mol.spin)
    Pa, Pb = scf(ks_mat).P
    ks_str = KS(
        mol, PBE0(), grid=(grid[0], grid[1]), coulomb=df(AUX, chunk=50), spin=mol.spin
    )
    e_mat = float(_e2_uks(ks_mat, Pa, Pb))
    e_str = float(_e2_uks(ks_str, Pa, Pb))
    assert abs(e_str - e_mat) < 1e-8, f"UKS streamed RI-K {e_str} vs mat {e_mat}"
