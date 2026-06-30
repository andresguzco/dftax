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
