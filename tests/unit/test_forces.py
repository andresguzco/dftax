"""Analytic nuclear forces: finite-difference consistency and stationarity.

Uses small grids; the analytic-vs-FD check is grid-independent (both use the
same grid), and the stationarity guard needs no accuracy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.energy.xc import LDA
from dftax.system.molecule import Molecule
from dftax.ks.energy import RKS
from dftax.ks.scf import rks_scf
from dftax.ks.forces import rks_forces, _density_from_Z
from dftax.grid import becke_grid

NR, LEB = 35, 50


@pytest.mark.float64
class TestForces:
    def test_h2_force_matches_finite_difference(self):
        xc = LDA()
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.85", "sto-3g")
        c0 = mol.atom_coords()

        def energy(coords):
            m = Molecule(mol.symbols, coords, mol.basis)
            gc, gw = becke_grid(m.symbols, m.atom_coords(), NR, LEB)
            return rks_scf(
                RKS.from_molecule(m, xc, gc, gw), e_tol=1e-10, d_tol=1e-8
            ).e_tot

        gc, gw = becke_grid(mol.symbols, c0, NR, LEB)
        res = rks_scf(RKS.from_molecule(mol, xc, gc, gw), e_tol=1e-10, d_tol=1e-8)
        F = rks_forces(mol, xc, res.mo_coeff[:, :1], n_radial=NR, lebedev=LEB)

        # Translational invariance: net force vanishes.
        assert float(np.abs(np.asarray(F.sum(axis=0))).max()) < 1e-8

        eps = 1e-3
        cp, cm = c0.copy(), c0.copy()
        cp[1, 2] += eps
        cm[1, 2] -= eps
        fd = -(energy(cp) - energy(cm)) / (2 * eps)
        assert abs(float(F[1, 2]) - fd) < 1e-4, f"F={float(F[1,2])} fd={fd}"

    def test_stationarity_multiple_occupied(self):
        # Guards the degenerate-eigh fix: at the converged density of a system
        # with nocc > 1, dE/dZ through the (solve-based) density must vanish.
        mol = Molecule.from_xyz("O 0 0 0; H 0.96 0 0; H -0.24 0.93 0", "sto-3g")
        gc, gw = becke_grid(mol.symbols, mol.atom_coords(), 30, 50)
        ks = RKS.from_molecule(mol, LDA(), gc, gw)
        res = rks_scf(ks, e_tol=1e-10, d_tol=1e-8)
        Z = res.mo_coeff[:, : mol.nelectron // 2]
        gZ = jax.grad(lambda Y: ks.total(_density_from_Z(Y, ks.S)))(Z)
        assert float(jnp.linalg.norm(gZ)) < 1e-6


@pytest.mark.float64
class TestDensityFittingForces:
    AUX = "def2-universal-jkfit"

    def test_metric_pinv_degenerate_derivative(self):
        # The RI Coulomb metric has *exactly* degenerate eigenvalues on symmetric
        # molecules (Td/Oh). There the plain eigh pseudo-inverse derivative is wrong
        # (finite-but-incorrect on CPU, NaN on GPU/cuSolver, so DF forces broke). The
        # custom_jvp must return the exact d(V⁺) = -V⁺ dV V⁺, finite and correct.
        from dftax.ks.energy import _metric_pinv

        Q, _ = jnp.linalg.qr(jnp.asarray(np.arange(16.0).reshape(4, 4) + np.eye(4)))
        V = Q @ jnp.diag(jnp.array([2.0, 2.0, 5.0, 7.0])) @ Q.T   # eigenvalue 2 is doubled
        V = 0.5 * (V + V.T)
        dV = jnp.asarray(np.random.default_rng(0).standard_normal((4, 4)))
        dV = 0.5 * (dV + dV.T)

        Vp = _metric_pinv(V)
        _, jvp = jax.jvp(_metric_pinv, (V,), (dV,))
        assert bool(np.all(np.isfinite(np.asarray(jvp))))
        assert float(np.abs(np.asarray(jvp) - np.asarray(-Vp @ dV @ Vp)).max()) < 1e-10

    def test_df_force_matches_finite_difference(self):
        xc = LDA()
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.85", "sto-3g")
        c0 = mol.atom_coords()

        def energy(coords):
            m = Molecule(mol.symbols, coords, mol.basis)
            gc, gw = becke_grid(m.symbols, m.atom_coords(), NR, LEB)
            return rks_scf(
                RKS.from_molecule(m, xc, gc, gw, auxbasis=self.AUX),
                e_tol=1e-10, d_tol=1e-8,
            ).e_tot

        gc, gw = becke_grid(mol.symbols, c0, NR, LEB)
        res = rks_scf(
            RKS.from_molecule(mol, xc, gc, gw, auxbasis=self.AUX), e_tol=1e-10, d_tol=1e-8
        )
        F = rks_forces(mol, xc, res.mo_coeff[:, :1], auxbasis=self.AUX, n_radial=NR, lebedev=LEB)

        assert bool(np.all(np.isfinite(np.asarray(F))))
        assert float(np.abs(np.asarray(F.sum(axis=0))).max()) < 1e-8

        eps = 1e-3
        cp, cm = c0.copy(), c0.copy()
        cp[1, 2] += eps
        cm[1, 2] -= eps
        fd = -(energy(cp) - energy(cm)) / (2 * eps)
        assert abs(float(F[1, 2]) - fd) < 1e-4, f"F={float(F[1,2])} fd={fd}"
