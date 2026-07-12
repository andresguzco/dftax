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
from dftax import KS, becke, df, scf, forces
from dftax.ks.forces import _density_from_Z
from dftax.grid import becke_grid

NR, LEB = 35, 50


@pytest.mark.float64
class TestForces:
    def test_forces_use_density_not_coefficient_packing(self):
        """Contract (audit finding 4): from a KSResult, forces must freeze the
        density the solver actually returned (result.P) — not whatever the
        mo_coeff packing implies. minimize's aufbau-ordered canonical orbitals
        need not span P when unconverged or at a degenerate frontier, so
        scrambling mo_coeff must not change the forces."""
        import dataclasses

        xc = LDA()
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.85", "sto-3g")
        grid = becke(NR, LEB)
        res = scf(KS(mol, xc, grid=grid), e_tol=1e-10, d_tol=1e-8)
        scrambled = dataclasses.replace(
            res, mo_coeff=jnp.zeros_like(res.mo_coeff)
        )
        F = forces(mol, xc, res, grid=grid)
        F2 = forces(mol, xc, scrambled, grid=grid)
        assert float(jnp.max(jnp.abs(F - F2))) < 1e-12

    def test_forces_honor_grid_chunk(self, monkeypatch):
        """Regression (audit finding 6): becke(chunk=...) must stream the XC
        grid inside the force rebuild — not be silently dropped — and the
        streamed geometry gradient must match the materialized one."""
        import dftax.ks.terms as terms

        calls = []
        orig = terms._streamed_e_xc
        monkeypatch.setattr(
            terms, "_streamed_e_xc",
            lambda *a, **k: (calls.append(1), orig(*a, **k))[1],
        )
        xc = LDA()
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.85", "sto-3g")
        res = scf(KS(mol, xc, grid=becke(NR, LEB)), e_tol=1e-10, d_tol=1e-8)
        F_mat = forces(mol, xc, res, grid=becke(NR, LEB))
        assert not calls                                  # materialized path
        F_str = forces(mol, xc, res, grid=becke(NR, LEB, chunk=500))
        assert calls                                      # streamed path taken
        assert float(jnp.max(jnp.abs(F_str - F_mat))) < 1e-9

    def test_h2_force_matches_finite_difference(self):
        xc = LDA()
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.85", "sto-3g")
        c0 = mol.atom_coords()

        def energy(coords):
            m = Molecule(mol.symbols, coords, mol.basis)
            gc, gw = becke_grid(m.symbols, m.atom_coords(), NR, LEB)
            return scf(
                KS(m, xc, grid=(gc, gw)), e_tol=1e-10, d_tol=1e-8
            ).e_tot

        gc, gw = becke_grid(mol.symbols, c0, NR, LEB)
        res = scf(KS(mol, xc, grid=(gc, gw)), e_tol=1e-10, d_tol=1e-8)
        F = forces(mol, xc, (res.mo_coeff[0][:, :1],), grid=becke(NR, LEB))

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
        ks = KS(mol, LDA(), grid=(gc, gw))
        res = scf(ks, e_tol=1e-10, d_tol=1e-8)
        Z = res.mo_coeff[0][:, : mol.nelectron // 2]
        gZ = jax.grad(lambda Y: ks.total(_density_from_Z(Y, ks.S)[None]))(Z)
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
            return scf(
                KS(m, xc, grid=(gc, gw), coulomb=df(self.AUX)),
                e_tol=1e-10, d_tol=1e-8,
            ).e_tot

        gc, gw = becke_grid(mol.symbols, c0, NR, LEB)
        res = scf(
            KS(mol, xc, grid=(gc, gw), coulomb=df(self.AUX)), e_tol=1e-10, d_tol=1e-8
        )
        F = forces(
            mol, xc, (res.mo_coeff[0][:, :1],),
            grid=becke(NR, LEB), coulomb=df(self.AUX),
        )

        assert bool(np.all(np.isfinite(np.asarray(F))))
        assert float(np.abs(np.asarray(F.sum(axis=0))).max()) < 1e-8

        eps = 1e-3
        cp, cm = c0.copy(), c0.copy()
        cp[1, 2] += eps
        cm[1, 2] -= eps
        fd = -(energy(cp) - energy(cm)) / (2 * eps)
        assert abs(float(F[1, 2]) - fd) < 1e-4, f"F={float(F[1,2])} fd={fd}"
