"""End-to-end UKS (open-shell) validation against PySCF + RKS consistency.

The unrestricted pipeline is parallel to the restricted one: separate spin
density matrices, per-spin autodiff Fock, spin aufbau, and spin-polarized XC.
These tests pin (a) the closed-shell limit (UKS ≡ RKS), (b) open-shell energies
against ``pyscf.dft.UKS``, (c) the per-spin autodiff Fock, (d) direct-min/SCF
agreement, (e) density fitting, (f) analytic forces, and (g) the run_ks dispatch.

OH is deliberately avoided as a test molecule: its unpaired electron sits in a
near-degenerate π* pair, a notoriously hard SCF case (needs level shifting). The
doublets/triplets used here (CH3, O2, Li, H2+) converge cleanly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import dft, gto

from dftax.energy.xc import LDA, PBE, PBE0, B3LYP
from dftax.system.molecule import Molecule
from dftax.ks.energy import RKS
from dftax.ks.energy_uks import UKS
from dftax.ks.scf import rks_scf
from dftax.ks.scf_uks import uks_scf, UKSResult
from dftax.ks.minimize_uks import uks_minimize
from dftax.ks.forces_uks import uks_forces
from dftax.ks.driver import run_ks, run_uks
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
@pytest.mark.parametrize("xc_cls,pyscf_xc", [(LDA, "slater,vwn5"), (PBE, "pbe")])
def test_uks_reduces_to_rks(xc_cls, pyscf_xc, water_mol):
    """Closed shell: UKS (Pα=Pβ) must reproduce the RKS energy exactly."""
    mf = dft.RKS(water_mol)
    mf.xc = pyscf_xc
    mf.grids.level = 2
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    e_r = rks_scf(RKS.from_pyscf(water_mol, xc_cls(), grid[0], grid[1])).e_tot
    res = uks_scf(UKS.from_pyscf(water_mol, xc_cls(), grid[0], grid[1]))
    assert res.converged
    assert abs(res.e_tot - e_r) < 1e-7, f"UKS {res.e_tot} vs RKS {e_r}"


@pytest.mark.pyscf
@pytest.mark.float64
@pytest.mark.parametrize(
    "xc_obj,pyscf_xc",
    [(LDA(), "slater,vwn5"), (PBE(), "pbe"), (PBE0(), "pbe0"), (B3LYP(), "b3lyp")],
)
def test_uks_doublet_matches_pyscf(xc_obj, pyscf_xc):
    """CH3 doublet vs pyscf.dft.UKS across LDA / PBE / PBE0 / B3LYP (hybrids)."""
    mol = _ch3()
    e_ref, grid = _ref_uks(mol, pyscf_xc)
    res = uks_scf(UKS.from_pyscf(mol, xc_obj, grid[0], grid[1]))
    assert res.converged
    assert abs(res.e_tot - e_ref) < 5e-5, f"{pyscf_xc}: {res.e_tot} vs {e_ref}"


@pytest.mark.pyscf
@pytest.mark.float64
def test_uks_triplet_o2():
    """O2 ground-state triplet (spin=2) vs pyscf.dft.UKS (LDA)."""
    mol = gto.M(atom="O 0 0 0; O 0 0 1.21", basis="sto-3g", spin=2).build()
    e_ref, grid = _ref_uks(mol, "slater,vwn5")
    res = uks_scf(UKS.from_pyscf(mol, LDA(), grid[0], grid[1]))
    assert res.converged
    assert abs(res.e_tot - e_ref) < 5e-5


@pytest.mark.pyscf
@pytest.mark.float64
def test_uks_fock_matches_fd(key):
    """Per-spin Fock F_σ = sym(∂E/∂P_σ): verify against finite differences.

    The perturbation is a manifold-tangent direction dP_σ = W_σ P_σ − P_σ W_σ
    (W antisymmetric ⇒ dP symmetric), evaluated at the converged density, so the
    spin densities stay physical (PSD); a random non-PSD perturbation would push
    a spin channel negative and probe the (legitimately non-smooth) unphysical
    region of the spin functional.
    """
    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1).build()
    _, grid = _ref_uks(mol, "pbe")
    ks = UKS.from_pyscf(mol, PBE(), jnp.asarray(grid[0]), jnp.asarray(grid[1]))
    Pa, Pb = uks_scf(ks).P

    ga, gb = jax.grad(lambda A, B: ks.electronic(A, B), argnums=(0, 1))(Pa, Pb)
    Fa, Fb = 0.5 * (ga + ga.T), 0.5 * (gb + gb.T)

    nao = Pa.shape[0]
    k1, k2 = jax.random.split(key)

    def tangent(P, k):
        W = jax.random.normal(k, (nao, nao))
        W = W - W.T                                   # antisymmetric
        return W @ P - P @ W                          # symmetric, PSD-preserving

    dPa, dPb = tangent(Pa, k1), tangent(Pb, k2)
    eps = 1e-4
    fd = (
        float(ks.electronic(Pa + eps * dPa, Pb + eps * dPb))
        - float(ks.electronic(Pa - eps * dPa, Pb - eps * dPb))
    ) / (2 * eps)
    ad = float(jnp.sum(Fa * dPa) + jnp.sum(Fb * dPb))
    assert abs(ad - fd) < 1e-5, f"AD={ad} vs FD={fd}"


@pytest.mark.pyscf
@pytest.mark.float64
def test_uks_minimize_matches_scf():
    """Direct (Adam) UKS minimization reaches the same energy as the SCF."""
    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1).build()
    _, grid = _ref_uks(mol, "slater,vwn5")
    ks = UKS.from_pyscf(mol, LDA(), grid[0], grid[1])
    e_scf = uks_scf(ks).e_tot
    e_min = uks_minimize(ks, learning_rate=0.5, max_steps=4000, g_tol=1e-6).e_tot
    assert abs(e_min - e_scf) < 1e-5, f"min {e_min} vs scf {e_scf}"


@pytest.mark.pyscf
@pytest.mark.float64
def test_uks_df_matches_exact():
    """RI density fitting reproduces the exact UKS energy to sub-mHa."""
    mol = _ch3()
    _, grid = _ref_uks(mol, "slater,vwn5")
    e_exact = uks_scf(UKS.from_pyscf(mol, LDA(), grid[0], grid[1])).e_tot
    res = uks_scf(UKS.from_pyscf(mol, LDA(), grid[0], grid[1], auxbasis=AUX))
    assert res.converged
    assert abs(res.e_tot - e_exact) < 1e-3, f"DF {res.e_tot} vs exact {e_exact}"


@pytest.mark.float64
def test_uks_forces_matches_fd():
    """Analytic UKS forces vs finite differences on a doublet (H2+, LDA)."""
    xc = LDA()
    NR, LEB = 35, 50
    mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.9", "sto-3g", charge=1, spin=1)
    c0 = mol.atom_coords()
    na, nb = (mol.nelectron + mol.spin) // 2, (mol.nelectron - mol.spin) // 2

    def energy(coords):
        m = Molecule(mol.symbols, coords, mol.basis, charge=mol.charge, spin=mol.spin)
        gc, gw = becke_grid(m.symbols, m.atom_coords(), NR, LEB)
        return uks_scf(
            UKS.from_molecule(m, xc, gc, gw), e_tol=1e-10, d_tol=1e-8
        ).e_tot

    gc, gw = becke_grid(mol.symbols, c0, NR, LEB)
    res = uks_scf(UKS.from_molecule(mol, xc, gc, gw), e_tol=1e-10, d_tol=1e-8)
    Ca, Cb = res.mo_coeff
    F = uks_forces(mol, xc, Ca[:, :na], Cb[:, :nb], n_radial=NR, lebedev=LEB)

    assert float(np.abs(np.asarray(F.sum(axis=0))).max()) < 1e-8  # net force ~ 0
    eps = 1e-3
    cp, cm = c0.copy(), c0.copy()
    cp[1, 2] += eps
    cm[1, 2] -= eps
    fd = -(energy(cp) - energy(cm)) / (2 * eps)
    assert abs(float(F[1, 2]) - fd) < 1e-4, f"F={float(F[1,2])} fd={fd}"


@pytest.mark.float64
def test_run_ks_dispatch():
    """run_ks routes by spin: closed→RKS, open→UKS (== run_uks)."""
    NR, LEB = 30, 50
    # Open shell: dispatches to UKS and matches an explicit run_uks.
    rad = Molecule.from_xyz("Li 0 0 0", "sto-3g", spin=1)
    r_auto = run_ks(rad, LDA(), n_radial=NR, lebedev=LEB)
    r_uks = run_uks(rad, LDA(), n_radial=NR, lebedev=LEB)
    assert isinstance(r_auto, UKSResult)
    assert abs(r_auto.e_tot - r_uks.e_tot) < 1e-9

    # Closed shell: dispatches to RKS (SCFResult, not UKSResult).
    h2 = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
    r_closed = run_ks(h2, LDA(), n_radial=NR, lebedev=LEB)
    assert not isinstance(r_closed, UKSResult)
    assert r_closed.converged
