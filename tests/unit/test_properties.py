import jax.numpy as jnp
import numpy as np
import pytest

from dftax.energy.xc import PBE
from dftax.system.molecule import Molecule
from dftax import (
    KS, System, becke, df, exact, scf,
    dipole, polarizability, hessian, ir_spectrum, raman_spectrum,
    alchemical_deriv,
)
from dftax.ks.properties import _solve_field, _grid
from dftax.integrals.multipole import dipole_matrices

WATER = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"
TOL = dict(e_tol=1e-11, d_tol=1e-9)


@pytest.mark.float64
@pytest.mark.default_grid
class TestDipole:
    def test_dipole_vs_finite_field(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        mu = np.asarray(dipole(mol, PBE(), coulomb=exact(), **TOL))
        (gc, gw), _ = _grid(mol, becke(75, 302))
        h = 1e-4
        fd = []
        for i in range(3):
            ep = _solve_field(mol, PBE(), gc, gw, field=jnp.zeros(3).at[i].set(h),
                              coulomb=exact(), **TOL)[1]
            em = _solve_field(mol, PBE(), gc, gw, field=jnp.zeros(3).at[i].set(-h),
                              coulomb=exact(), **TOL)[1]
            fd.append(-(ep - em) / (2 * h))
        assert np.max(np.abs(mu - np.array(fd))) < 1e-6

    def test_dipole_vs_finite_field_df(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        mu = np.asarray(dipole(mol, PBE(), coulomb=df(), **TOL))
        (gc, gw), _ = _grid(mol, becke(75, 302))
        h = 1e-4
        fd = []
        for i in range(3):
            ep = _solve_field(mol, PBE(), gc, gw, field=jnp.zeros(3).at[i].set(h),
                              coulomb=df(), **TOL)[1]
            em = _solve_field(mol, PBE(), gc, gw, field=jnp.zeros(3).at[i].set(-h),
                              coulomb=df(), **TOL)[1]
            fd.append(-(ep - em) / (2 * h))
        assert np.max(np.abs(mu - np.array(fd))) < 2e-5

    def test_dipole_vs_pyscf(self):
        pyscf = pytest.importorskip("pyscf")
        from pyscf import dft
        mol = Molecule.from_xyz(WATER, "sto-3g")
        coords = mol.atom_coords()
        mu = np.asarray(dipole(mol, PBE(), coulomb=exact(), **TOL))
        m = pyscf.gto.M(atom=[[s, tuple(coords[i])] for i, s in enumerate(mol.symbols)],
                        basis="sto-3g", unit="Bohr", verbose=0)
        mf = dft.RKS(m); mf.xc = "pbe"; mf.grids.level = 5; mf.conv_tol = 1e-12; mf.kernel()
        assert np.max(np.abs(mu - mf.dip_moment(unit="au", verbose=0))) < 1e-5

    def test_dipole_open_shell_vs_pyscf(self):
        pyscf = pytest.importorskip("pyscf")
        from pyscf import dft
        mol = Molecule.from_xyz("O 0 0 0; H 0.9697 0 0", "sto-3g", spin=1)
        coords = mol.atom_coords()
        mu = np.asarray(dipole(mol, PBE(), **TOL))
        m = pyscf.gto.M(atom=[[s, tuple(coords[i])] for i, s in enumerate(mol.symbols)],
                        basis="sto-3g", unit="Bohr", spin=1, verbose=0)
        mf = dft.UKS(m); mf.xc = "pbe"; mf.grids.level = 5; mf.conv_tol = 1e-12; mf.kernel()
        assert np.max(np.abs(mu - mf.dip_moment(unit="au", verbose=0))) < 1e-4

    def test_dipole_matrix_symmetric(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        basis = KS(mol, PBE(), grid=_grid(mol, becke(35, 50))[0]).basis
        D = np.asarray(dipole_matrices(basis))
        assert np.max(np.abs(D - np.transpose(D, (0, 2, 1)))) < 1e-12


@pytest.mark.float64
class TestResponse:
    def test_hessian_preserves_charge_and_spin(self):
        mol = Molecule.from_xyz("He 0 0 0; H 0 0 0.774", "sto-3g", charge=1)
        H = np.asarray(hessian(mol, PBE(), grid=becke(35, 50), **TOL))
        assert H.shape == (6, 6)
        assert np.max(np.abs(H - H.T)) < 1e-6

    def test_polarizability_symmetric_positive(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        a = np.asarray(polarizability(mol, PBE(), field=2e-3, grid=becke(60, 194), **TOL))
        assert np.max(np.abs(a - a.T)) < 1e-4
        assert np.all(np.linalg.eigvalsh(a) > -1e-6)

    def test_frequencies_vs_pyscf(self):
        pyscf = pytest.importorskip("pyscf")
        from pyscf import dft
        from pyscf.hessian import thermo
        mol = Molecule.from_xyz(WATER, "sto-3g")
        coords = mol.atom_coords()
        freq = np.sort(np.asarray(ir_spectrum(
            mol, PBE(), step=2e-3, grid=becke(60, 194), coulomb=exact(),
            **TOL).frequencies))
        m = pyscf.gto.M(atom=[[s, tuple(coords[i])] for i, s in enumerate(mol.symbols)],
                        basis="sto-3g", unit="Bohr", verbose=0)
        mf = dft.RKS(m); mf.xc = "pbe"; mf.grids.level = 5; mf.conv_tol = 1e-12; mf.kernel()
        fp = np.sort(np.asarray(thermo.harmonic_analysis(m, mf.Hessian().kernel())["freq_wavenumber"]).real)
        assert np.max(np.abs(freq[-3:] - fp[-3:])) < 5.0
        assert np.max(np.abs(freq[:6])) < 30.0

    def test_alchemical_vs_finite_difference(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        coords = mol.atom_coords()
        dEdZ = np.asarray(alchemical_deriv(mol, PBE(), grid=becke(60, 194), **TOL))
        (gc, gw), _ = _grid(mol, becke(60, 194))
        basis = KS(mol, PBE(), grid=(gc, gw)).basis

        def E_at(charges):
            k = KS(
                System(basis=basis, coords=jnp.asarray(coords),
                       charges=jnp.asarray(charges), nelec=mol.nelectron),
                PBE(), grid=(gc, gw),
            )
            return float(scf(k, **TOL).e_tot)

        ch0 = np.asarray(mol.atom_charges(), float); h = 1e-3
        fd = np.array([(E_at(ch0 + h * (np.arange(len(ch0)) == a))
                        - E_at(ch0 - h * (np.arange(len(ch0)) == a))) / (2 * h)
                       for a in range(len(ch0))])
        assert np.max(np.abs(dEdZ - fd)) < 1e-5

    def test_alchemical_df_vs_finite_difference(self):
        from dftax.basis.loader import build_basis_data

        mol = Molecule.from_xyz(WATER, "sto-3g")
        coords = mol.atom_coords()
        aux = "def2-universal-jkfit"
        dEdZ = np.asarray(alchemical_deriv(
            mol, PBE(), grid=becke(60, 194), coulomb=df(aux), **TOL
        ))
        (gc, gw), _ = _grid(mol, becke(60, 194))
        basis = KS(mol, PBE(), grid=(gc, gw)).basis
        aux_b = build_basis_data(mol.symbols, coords, aux, spherical=True)

        def E_at(charges):
            k = KS(
                System(basis=basis, coords=jnp.asarray(coords),
                       charges=jnp.asarray(charges), nelec=mol.nelectron),
                PBE(), grid=(gc, gw), coulomb=df(aux_b),
            )
            return float(scf(k, **TOL).e_tot)

        ch0 = np.asarray(mol.atom_charges(), float); h = 1e-3
        fd = np.array([(E_at(ch0 + h * (np.arange(len(ch0)) == a))
                        - E_at(ch0 - h * (np.arange(len(ch0)) == a))) / (2 * h)
                       for a in range(len(ch0))])
        assert np.max(np.abs(dEdZ - fd)) < 1e-5

    def test_alchemical_open_shell_vs_finite_difference(self):
        mol = Molecule.from_xyz("Li 0 0 0", "sto-3g", spin=1)
        coords = mol.atom_coords()
        dEdZ = np.asarray(alchemical_deriv(mol, PBE(), grid=becke(35, 50), **TOL))
        (gc, gw), _ = _grid(mol, becke(35, 50))
        basis = KS(mol, PBE(), grid=(gc, gw)).basis

        def E_at(charges):
            k = KS(
                System(basis=basis, coords=jnp.asarray(coords),
                       charges=jnp.asarray(charges), nelec=mol.nelectron, spin=1),
                PBE(), grid=(gc, gw),
            )
            return float(scf(k, **TOL).e_tot)

        ch0 = np.asarray(mol.atom_charges(), float); h = 1e-3
        fd = np.array([(E_at(ch0 + h * (np.arange(len(ch0)) == a))
                        - E_at(ch0 - h * (np.arange(len(ch0)) == a))) / (2 * h)
                       for a in range(len(ch0))])
        assert np.max(np.abs(dEdZ - fd)) < 1e-5


@pytest.mark.float64
class TestAnalyticHessian:
    def test_h2_matches_fd(self):
        from dftax.energy.xc import LDA

        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
        Ha = np.asarray(hessian(mol, LDA(), method="analytic",
                                grid=becke(35, 50), coulomb=exact(), **TOL))
        Hf = np.asarray(hessian(mol, LDA(), step=1e-3, grid=becke(35, 50),
                                coulomb=exact(), **TOL))
        assert np.abs(Ha - Hf).max() < 1e-5
        assert np.abs(Ha - Ha.T).max() == 0.0

    def test_water_matches_fd(self):
        from dftax.energy.xc import LDA

        mol = Molecule.from_xyz(WATER, "sto-3g")
        Ha = np.asarray(hessian(mol, LDA(), method="analytic",
                                grid=becke(35, 50), coulomb=exact(),
                                e_tol=1e-10, d_tol=1e-8))
        Hf = np.asarray(hessian(mol, LDA(), step=1e-3, grid=becke(35, 50),
                                coulomb=exact(), e_tol=1e-10, d_tol=1e-8))
        assert np.abs(Ha - Hf).max() < 2e-5
        n = Ha.shape[0] // 3
        trans = np.abs(Ha.reshape(n, 3, n, 3).sum(axis=0)).max()
        assert trans < 1e-4

    def test_open_shell_matches_fd(self):
        from dftax.energy.xc import LDA

        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.9; H 0 0.2 1.8",
                                "sto-3g", spin=1)
        Ha = np.asarray(hessian(mol, LDA(), method="analytic",
                                grid=becke(35, 50), coulomb=exact(),
                                e_tol=1e-10, d_tol=1e-8))
        Hf = np.asarray(hessian(mol, LDA(), step=1e-3, grid=becke(35, 50),
                                coulomb=exact(), e_tol=1e-10, d_tol=1e-8))
        assert np.abs(Ha - Hf).max() < 2e-5
        trans = np.abs(Ha.reshape(3, 3, 3, 3).sum(axis=0)).max()
        assert trans < 1e-4

    def test_uks_path_matches_rks_on_closed_shell(self):
        from dftax.energy.xc import LDA
        from dftax.grid import becke_grid, points
        from dftax.ks.hessian import _analytic_hessian
        from dftax.ks.newton import newton

        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
        g = becke(35, 50)
        Hr = np.asarray(hessian(mol, LDA(), method="analytic", grid=g,
                                coulomb=exact(), **TOL))
        gc, gw = becke_grid(mol.symbols, mol.atom_coords(), g.n_radial,
                            g.lebedev, g.prune, g.r_max)
        ks_u = KS(mol, LDA(), grid=points(gc, gw), coulomb=exact(), spin=0)
        res_u = newton(ks_u, g_tol=1e-9, e_tol=1e-13, max_iter=64)
        Hu = np.asarray(_analytic_hessian(mol, LDA(), res_u, g, exact(),
                                          None))
        assert np.abs(Hr - Hu).max() < 1e-7

    def test_streamed_matches_materialized(self):
        from dftax.energy.xc import LDA, PBE0

        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
        g = becke(35, 50)
        for xc, aux, tol in ((LDA(), "sto-3g", 1e-12),
                             (PBE0(), "sto-3g", 1e-12),
                             (PBE0(), "def2-universal-jkfit", 5e-7)):
            Hm = np.asarray(hessian(
                mol, xc, method="analytic", grid=g,
                coulomb=df(aux, chunk=None, spherical=False), **TOL))
            Hs = np.asarray(hessian(mol, xc, method="analytic", grid=g,
                                    coulomb=df(aux, chunk=4), **TOL))
            assert np.abs(Hm - Hs).max() < tol

    def test_streamed_open_shell_matches_materialized(self):
        from dftax.energy.xc import PBE0

        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.9; H 0 0.2 1.8",
                                "sto-3g", spin=1)
        g = becke(35, 50)
        kw = dict(e_tol=1e-10, d_tol=1e-8)
        Hm = np.asarray(hessian(
            mol, PBE0(), method="analytic", grid=g,
            coulomb=df("sto-3g", chunk=None, spherical=False), **kw))
        Hs = np.asarray(hessian(mol, PBE0(), method="analytic", grid=g,
                                coulomb=df("sto-3g", chunk=4), **kw))
        assert np.abs(Hm - Hs).max() < 1e-9

    def test_empty_beta_channel(self):
        from dftax.energy.xc import LDA

        mol = Molecule.from_xyz("H 0 0 0", "sto-3g", spin=1)
        Ha = np.asarray(hessian(mol, LDA(), method="analytic",
                                grid=becke(35, 50), coulomb=exact(), **TOL))
        assert Ha.shape == (3, 3)
        assert np.abs(Ha).max() < 1e-6


@pytest.mark.float64
class TestVibrationalSpectra:
    @pytest.mark.slow
    def test_ir_intensities_sane(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        ir = ir_spectrum(mol, PBE(), grid=becke(35, 50), **TOL)
        freq = np.asarray(ir.frequencies); inten = np.asarray(ir.intensities)
        assert inten.shape == freq.shape == (9,)
        assert np.all(np.isfinite(inten))
        assert np.all(inten > -1e-10)
        assert inten.max() > 1e-2

    @pytest.mark.slow
    def test_raman_activities_sane(self):
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
        r = raman_spectrum(mol, PBE(), grid=becke(20, 50), **TOL)
        freq = np.asarray(r.frequencies); act = np.asarray(r.activities)
        assert act.shape == freq.shape == (6,)
        assert np.all(np.isfinite(act))
        assert np.all(act > -1e-10)
        assert act.max() > 0
