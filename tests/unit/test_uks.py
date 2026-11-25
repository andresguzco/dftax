import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pyscf import dft, gto

from dftax.energy.xc import LDA, PBE, PBE0
from dftax.system.molecule import Molecule
from dftax import KS, becke, df, scf, minimize, forces, exact
from dftax.grid import becke_grid

AUX = "def2-universal-jkfit"


def _ref_uks(mol, pyscf_xc, level=3):
    mf = dft.UKS(mol)
    mf.xc = pyscf_xc
    mf.grids.level = level
    mf.verbose = 0
    e = mf.kernel()
    return float(e), (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))


def _ch3():
    return gto.M(
        atom="C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0",
        basis="sto-3g", spin=1,
    ).build()


@pytest.mark.pyscf
@pytest.mark.float64
@pytest.mark.parametrize("xc_cls,pyscf_xc", [(PBE, "pbe")])
def test_uks_reduces_to_rks(xc_cls, pyscf_xc, water_mol):
    mf = dft.RKS(water_mol)
    mf.xc = pyscf_xc
    mf.grids.level = 2
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    e_r = scf(KS(water_mol, xc_cls(), grid=(grid[0], grid[1]))).e_tot
    res = scf(KS(water_mol, xc_cls(), grid=(grid[0], grid[1]), spin=0))
    assert res.converged
    assert abs(res.e_tot - e_r) < 1e-7, f"UKS {res.e_tot} vs RKS {e_r}"


@pytest.mark.pyscf
@pytest.mark.float64
@pytest.mark.parametrize(
    "xc_obj,pyscf_xc",
    [(PBE(), "pbe"), (PBE0(), "pbe0")],
)
def test_uks_doublet_matches_pyscf(xc_obj, pyscf_xc):
    mol = _ch3()
    e_ref, grid = _ref_uks(mol, pyscf_xc)
    res = scf(KS(mol, xc_obj, grid=(grid[0], grid[1]), spin=mol.spin, coulomb=exact()))
    assert res.converged
    assert abs(res.e_tot - e_ref) < 5e-5, f"{pyscf_xc}: {res.e_tot} vs {e_ref}"


@pytest.mark.pyscf
@pytest.mark.float64
def test_uks_triplet_o2():
    mol = gto.M(atom="O 0 0 0; O 0 0 1.21", basis="sto-3g", spin=2).build()
    e_ref, grid = _ref_uks(mol, "slater,vwn5")
    res = scf(KS(mol, LDA(), grid=(grid[0], grid[1]), spin=mol.spin, coulomb=exact()))
    assert res.converged
    assert abs(res.e_tot - e_ref) < 5e-5


@pytest.mark.pyscf
@pytest.mark.float64
def test_uks_fock_matches_fd(key):
    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1).build()
    _, grid = _ref_uks(mol, "pbe")
    ks = KS(
        mol, PBE(), grid=(jnp.asarray(grid[0]), jnp.asarray(grid[1])), spin=mol.spin
    )
    Pa, Pb = scf(ks).P

    g = jax.grad(ks.electronic)(jnp.stack([Pa, Pb]))
    Fa, Fb = 0.5 * (g[0] + g[0].T), 0.5 * (g[1] + g[1].T)

    nao = Pa.shape[0]
    k1, k2 = jax.random.split(key)

    def tangent(P, k):
        W = jax.random.normal(k, (nao, nao))
        W = W - W.T
        return W @ P - P @ W

    dPa, dPb = tangent(Pa, k1), tangent(Pb, k2)
    eps = 1e-4
    fd = (
        float(ks.electronic(jnp.stack([Pa + eps * dPa, Pb + eps * dPb])))
        - float(ks.electronic(jnp.stack([Pa - eps * dPa, Pb - eps * dPb])))
    ) / (2 * eps)
    ad = float(jnp.sum(Fa * dPa) + jnp.sum(Fb * dPb))
    assert abs(ad - fd) < 1e-5, f"AD={ad} vs FD={fd}"


@pytest.mark.pyscf
@pytest.mark.float64
def test_uks_minimize_matches_scf():
    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1).build()
    _, grid = _ref_uks(mol, "slater,vwn5")
    ks = KS(mol, LDA(), grid=(grid[0], grid[1]), spin=mol.spin)
    e_scf = scf(ks).e_tot
    e_min = minimize(ks, optax.adam(0.5), max_steps=4000, g_tol=1e-6).e_tot
    assert abs(e_min - e_scf) < 1e-5, f"min {e_min} vs scf {e_scf}"


@pytest.mark.pyscf
@pytest.mark.float64
def test_uks_df_matches_exact():
    mol = _ch3()
    _, grid = _ref_uks(mol, "slater,vwn5")
    e_exact = scf(KS(mol, LDA(), grid=(grid[0], grid[1]), spin=mol.spin)).e_tot
    res = scf(
        KS(mol, LDA(), grid=(grid[0], grid[1]), spin=mol.spin, coulomb=df(AUX))
    )
    assert res.converged
    assert abs(res.e_tot - e_exact) < 1e-3, f"DF {res.e_tot} vs exact {e_exact}"


@pytest.mark.float64
def test_uks_forces_matches_fd():
    xc = LDA()
    NR, LEB = 35, 50
    mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.9", "sto-3g", charge=1, spin=1)
    c0 = mol.atom_coords()
    na, nb = (mol.nelectron + mol.spin) // 2, (mol.nelectron - mol.spin) // 2

    def energy(coords):
        m = Molecule(mol.symbols, coords, mol.basis, charge=mol.charge, spin=mol.spin)
        gc, gw = becke_grid(m.symbols, m.atom_coords(), NR, LEB)
        return scf(
            KS(m, xc, grid=(gc, gw), spin=m.spin), e_tol=1e-10, d_tol=1e-8
        ).e_tot

    gc, gw = becke_grid(mol.symbols, c0, NR, LEB)
    res = scf(KS(mol, xc, grid=(gc, gw), spin=mol.spin), e_tol=1e-10, d_tol=1e-8)
    Ca, Cb = res.mo_coeff
    F = forces(mol, xc, (Ca[:, :na], Cb[:, :nb]), grid=becke(NR, LEB))

    assert float(np.abs(np.asarray(F.sum(axis=0))).max()) < 1e-8
    eps = 1e-3
    cp, cm = c0.copy(), c0.copy()
    cp[1, 2] += eps
    cm[1, 2] -= eps
    fd = -(energy(cp) - energy(cm)) / (2 * eps)
    assert abs(float(F[1, 2]) - fd) < 1e-4, f"F={float(F[1,2])} fd={fd}"


@pytest.mark.float64
def test_ks_spin_dispatch():
    NR, LEB = 30, 50
    rad = Molecule.from_xyz("Li 0 0 0", "sto-3g", spin=1)
    r_auto = scf(KS(rad, LDA(), grid=becke(NR, LEB)))
    r_uks = scf(KS(rad, LDA(), grid=becke(NR, LEB), spin=rad.spin))
    assert len(r_auto.nocc) == 2
    assert abs(r_auto.e_tot - r_uks.e_tot) < 1e-9

    h2 = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
    r_closed = scf(KS(h2, LDA(), grid=becke(NR, LEB)))
    assert len(r_closed.nocc) == 1
    assert r_closed.converged
