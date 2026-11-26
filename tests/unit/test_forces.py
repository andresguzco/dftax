import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.energy.xc import LDA
from dftax.system.molecule import Molecule
from dftax import KS, becke, df, exact, fermi, scf, forces
from dftax.ks.forces import _density_from_Z
from dftax.grid import becke_grid

NR, LEB = 35, 50


@pytest.mark.float64
class TestForces:
    def test_forces_use_density_not_coefficient_packing(self):
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
        assert not calls
        F_str = forces(mol, xc, res, grid=becke(NR, LEB, chunk=500))
        assert calls
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

        assert float(np.abs(np.asarray(F.sum(axis=0))).max()) < 1e-8

        eps = 1e-3
        cp, cm = c0.copy(), c0.copy()
        cp[1, 2] += eps
        cm[1, 2] -= eps
        fd = -(energy(cp) - energy(cm)) / (2 * eps)
        assert abs(float(F[1, 2]) - fd) < 1e-4, f"F={float(F[1,2])} fd={fd}"

    def test_smeared_force_matches_finite_difference(self):
        xc = LDA()
        c0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]])
        sig = 0.02

        def free_energy(coords):
            m = Molecule(["C", "C"], coords, "sto-3g")
            gc, gw = becke_grid(m.symbols, m.atom_coords(), NR, LEB)
            return scf(KS(m, xc, grid=(gc, gw), coulomb=exact()),
                       smearing=fermi(sigma=sig),
                       e_tol=1e-11, d_tol=1e-9, max_iter=300).e_tot

        mol = Molecule(["C", "C"], c0, "sto-3g")
        gc, gw = becke_grid(mol.symbols, c0, NR, LEB)
        res = scf(KS(mol, xc, grid=(gc, gw), coulomb=exact()),
                  smearing=fermi(sigma=sig), e_tol=1e-11, d_tol=1e-9,
                  max_iter=300)
        assert res.converged and float(res.ts) > 1e-6

        F = forces(mol, xc, res, grid=becke(NR, LEB), coulomb=exact())
        eps = 1e-3
        cp, cm = c0.copy(), c0.copy()
        cp[1, 2] += eps
        cm[1, 2] -= eps
        fd = -(float(free_energy(cp)) - float(free_energy(cm))) / (2 * eps)
        assert abs(float(F[1, 2]) - fd) < 1e-4, f"F={float(F[1, 2])} fd={fd}"

    def test_stationarity_multiple_occupied(self):
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
        from dftax.ks.energy import _metric_pinv

        Q, _ = jnp.linalg.qr(jnp.asarray(np.arange(16.0).reshape(4, 4) + np.eye(4)))
        V = Q @ jnp.diag(jnp.array([2.0, 2.0, 5.0, 7.0])) @ Q.T
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

    def test_materialized_df_forces_take_the_packed_path(self, monkeypatch):
        import importlib

        from dftax.energy.xc import PBE0

        forces_mod = importlib.import_module("dftax.ks.forces")

        calls = []
        orig = forces_mod._packed_forces_tensors
        monkeypatch.setattr(forces_mod, "_packed_forces_tensors",
                            lambda *a: (calls.append(1), orig(*a))[1])
        xc = PBE0()
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.9; H 0 0.2 1.8", "sto-3g",
                                spin=1)
        res = scf(KS(mol, xc, grid=becke(NR, LEB), coulomb=df("sto-3g")),
                  e_tol=1e-10, d_tol=1e-8)
        Zs = (res.mo_coeff[0][:, :2], res.mo_coeff[1][:, :1])
        Fp = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df("sto-3g", chunk=None, spherical=False))
        assert calls
        Fs = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df("sto-3g", chunk=4))
        assert float(jnp.max(jnp.abs(Fp - Fs))) < 1e-12

    def test_streamed_df_forces_match_materialized(self):
        xc = LDA()
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.85", "sto-3g")
        res = scf(
            KS(mol, xc, grid=becke(NR, LEB), coulomb=df(self.AUX)),
            e_tol=1e-10, d_tol=1e-8,
        )
        Zs = (res.mo_coeff[0][:, :1],)
        Fm = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df(self.AUX, chunk=None, spherical=False))
        Fs = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df(self.AUX, chunk=4))
        assert float(jnp.max(jnp.abs(Fm - Fs))) < 1e-8

    def test_streamed_df_forces_hybrid(self):
        from dftax.energy.xc import PBE0

        xc = PBE0()
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.85", "sto-3g")
        res = scf(
            KS(mol, xc, grid=becke(NR, LEB), coulomb=df(self.AUX)),
            e_tol=1e-10, d_tol=1e-8,
        )
        Zs = (res.mo_coeff[0][:, :1],)
        Fm = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df(self.AUX, chunk=None, spherical=False))
        Fs = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df(self.AUX, chunk=4))
        assert float(jnp.max(jnp.abs(Fm - Fs))) < 1e-8

    def test_streamed_df_forces_uks_hybrid(self):
        from dftax.energy.xc import PBE0

        xc = PBE0()
        mol = Molecule.from_xyz(
            "H 0 0 0; H 0 0 0.9; H 0 0.2 1.8", "sto-3g", spin=1
        )
        res = scf(
            KS(mol, xc, grid=becke(NR, LEB), coulomb=df("sto-3g")),
            e_tol=1e-10, d_tol=1e-8,
        )
        Zs = (res.mo_coeff[0][:, :2], res.mo_coeff[1][:, :1])
        Fm = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df("sto-3g", chunk=None, spherical=False))
        Fs = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df("sto-3g", chunk=4))
        assert float(jnp.max(jnp.abs(Fm - Fs))) < 1e-12

    def test_streamed_df_forces_rsh(self):
        from dftax.energy.xc import CAMB3LYP

        xc = CAMB3LYP()
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.85", "sto-3g")
        res = scf(
            KS(mol, xc, grid=becke(NR, LEB), coulomb=df("sto-3g")),
            e_tol=1e-10, d_tol=1e-8,
        )
        Zs = (res.mo_coeff[0][:, :1],)
        Fm = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df("sto-3g", chunk=None, spherical=False))
        Fs = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df("sto-3g", chunk=4))
        assert float(jnp.max(jnp.abs(Fm - Fs))) < 1e-12

    def test_forces_auto_chunk_streams_past_budget(self, monkeypatch):
        import dftax.ks.energy as energy_mod
        import dftax.ks.terms as terms

        xc = LDA()
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.85", "sto-3g")
        res = scf(
            KS(mol, xc, grid=becke(NR, LEB), coulomb=df(self.AUX)),
            e_tol=1e-10, d_tol=1e-8,
        )
        Zs = (res.mo_coeff[0][:, :1],)
        F_mat = forces(mol, xc, Zs, grid=becke(NR, LEB),
                       coulomb=df(self.AUX, chunk=None, spherical=False))

        calls = []
        orig = terms._streamed_df_rij_slabs
        monkeypatch.setattr(
            terms, "_streamed_df_rij_slabs",
            lambda *a, **k: (calls.append(1), orig(*a, **k))[1],
        )
        monkeypatch.setattr(energy_mod, "_df_materialize_budget", lambda: 16)
        F_auto = forces(mol, xc, Zs, grid=becke(NR, LEB), coulomb=df(self.AUX))
        assert calls
        assert float(jnp.max(jnp.abs(F_auto - F_mat))) < 1e-8

    def test_streamed_screened_forces_match_dense(self):
        xc = LDA()
        atoms = "; ".join(f"H 0 0 {i * 1.4:.3f}" for i in range(6))
        mol = Molecule.from_xyz(atoms, "sto-3g")
        res = scf(
            KS(mol, xc, grid=becke(NR, LEB), coulomb=df("sto-3g")),
            e_tol=1e-10, d_tol=1e-8,
        )
        Zs = (res.mo_coeff[0][:, :3],)
        Fd = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df("sto-3g", chunk=6))
        Fs = forces(mol, xc, Zs, grid=becke(NR, LEB),
                    coulomb=df("sto-3g", chunk=6, screen=1e-10))
        assert float(jnp.max(jnp.abs(Fd - Fs))) < 1e-8

    def test_streamed_smeared_forces(self):
        from dftax.energy.xc import PBE0

        xc = LDA()
        c0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]])
        mol = Molecule(["C", "C"], c0, "sto-3g")
        gc, gw = becke_grid(mol.symbols, c0, NR, LEB)
        res = scf(KS(mol, xc, grid=(gc, gw), coulomb=df("sto-3g")),
                  smearing=fermi(sigma=0.02), e_tol=1e-11, d_tol=1e-9,
                  max_iter=300)
        assert res.converged and float(res.ts) > 1e-6

        Fm = forces(mol, xc, res, grid=becke(NR, LEB),
                    coulomb=df("sto-3g", chunk=None, spherical=False))
        Fs = forces(mol, xc, res, grid=becke(NR, LEB),
                    coulomb=df("sto-3g", chunk=6))
        assert float(jnp.max(jnp.abs(Fm - Fs))) < 1e-8

        with pytest.raises(NotImplementedError, match="smeared"):
            forces(mol, PBE0(), res, grid=becke(NR, LEB),
                   coulomb=df("sto-3g", chunk=6))
