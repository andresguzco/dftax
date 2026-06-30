"""Streamed RI-J / RI-K (df_chunk) matches materialized density fitting.

The streamed path forms γ_P (and, for hybrids, the orbital-chunk RI-K exchange)
without materializing the nao²×naux 3-center tensor; the DF energy must match the
materialized DF path.
"""

import jax
import jax.numpy as jnp
import pytest

from pyscf import dft, gto

from dftax.energy.xc import LDA, PBE, PBE0
from dftax.ks.energy import RKS, _streamed_df_rij
from dftax.ks.energy_uks import UKS
from dftax.ks.scf import rks_scf
from dftax.ks.scf_uks import uks_scf

AUX = "def2-universal-jkfit"


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

    ks = RKS.from_pyscf(water_mol, LDA(), grid[0], grid[1], auxbasis=AUX)
    P = rks_scf(ks).P

    g = jnp.einsum("mnP,mn->P", ks.int3c, P)
    ej_mat = 0.5 * float(jnp.dot(g, ks.int2c_inv @ g))
    ej_str = float(_streamed_df_rij(ks.basis, ks.aux_basis, ks.int2c_inv, P, 50))
    assert abs(ej_str - ej_mat) < 1e-8, f"streamed {ej_str} vs mat {ej_mat}"


@pytest.mark.slow  # streamed RI-K custom_vjp compile is heavy on CPU
@pytest.mark.pyscf
@pytest.mark.float64
def test_df_chunk_hybrid_matches_materialized(water_mol):
    """Hybrid + df_chunk streams RI-K (orbital-chunk custom_vjp); the DF
    energy must match the materialized DF hybrid. Fixed-P (no streamed SCF loop)
    keeps the unit test cheap."""
    mf = dft.RKS(water_mol)
    mf.xc = "pbe0"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_mat = RKS.from_pyscf(water_mol, PBE0(), grid[0], grid[1], auxbasis=AUX)
    P = rks_scf(ks_mat).P
    ks_str = RKS.from_pyscf(water_mol, PBE0(), grid[0], grid[1], auxbasis=AUX, df_chunk=50)
    e_mat = float(ks_mat._coulomb_exchange(P))
    e_str = float(ks_str._coulomb_exchange(P))
    assert abs(e_str - e_mat) < 1e-8, f"streamed RI-J+RI-K {e_str} vs mat {e_mat}"


@pytest.mark.slow  # two streamed-RI-J compiles (dense + screened shapes)
@pytest.mark.pyscf
@pytest.mark.float64
def test_screened_df_chunk_matches_dense(water_mol):
    """Schwarz-screened streamed RI-J (df_screen) matches the dense streamed RI-J
    at a fixed density (water has no negligible pairs, so screening is exact)."""
    mf = dft.RKS(water_mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_dense = RKS.from_pyscf(water_mol, PBE(), grid[0], grid[1], auxbasis=AUX, df_chunk=50)
    P = rks_scf(RKS.from_pyscf(water_mol, PBE(), grid[0], grid[1], auxbasis=AUX)).P
    ks_scr = RKS.from_pyscf(water_mol, PBE(), grid[0], grid[1], auxbasis=AUX,
                            df_chunk=50, df_screen=1e-10)
    e_dense = float(ks_dense._coulomb_exchange(P))
    e_scr = float(ks_scr._coulomb_exchange(P))
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
    ks_mat = RKS.from_pyscf(water_mol, PBE0(), grid[0], grid[1], auxbasis=AUX)
    P = rks_scf(ks_mat).P
    ks_str = RKS.from_pyscf(water_mol, PBE0(), grid[0], grid[1], auxbasis=AUX, df_chunk=50)

    sym = lambda M: 0.5 * (M + M.T)
    F_mat = sym(jax.grad(ks_mat._coulomb_exchange)(P))   # materialized: full autodiff
    F_str = sym(jax.grad(ks_str._coulomb_exchange)(P))   # streamed: custom_vjp backward
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
    ks_mat = UKS.from_pyscf(mol, PBE0(), grid[0], grid[1], auxbasis=AUX)
    Pa, Pb = uks_scf(ks_mat).P
    ks_str = UKS.from_pyscf(mol, PBE0(), grid[0], grid[1], auxbasis=AUX, df_chunk=50)
    e_mat = float(ks_mat._coulomb_exchange(Pa, Pb))
    e_str = float(ks_str._coulomb_exchange(Pa, Pb))
    assert abs(e_str - e_mat) < 1e-8, f"empty-β streamed {e_str} vs mat {e_mat}"


@pytest.mark.slow
@pytest.mark.pyscf
@pytest.mark.float64
def test_streamed_uks_rik_matches_materialized():
    """Streamed per-spin UKS RI-K (df_chunk) == materialized DF hybrid, on a doublet
    with a non-empty β channel (CH₃)."""
    mol = _ch3()
    mf = dft.UKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks_mat = UKS.from_pyscf(mol, PBE0(), grid[0], grid[1], auxbasis=AUX)
    Pa, Pb = uks_scf(ks_mat).P
    ks_str = UKS.from_pyscf(mol, PBE0(), grid[0], grid[1], auxbasis=AUX, df_chunk=50)
    e_mat = float(ks_mat._coulomb_exchange(Pa, Pb))
    e_str = float(ks_str._coulomb_exchange(Pa, Pb))
    assert abs(e_str - e_mat) < 1e-8, f"UKS streamed RI-K {e_str} vs mat {e_mat}"
