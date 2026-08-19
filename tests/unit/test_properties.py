"""Response properties: dipole, polarizability, Hessian/frequencies, IR, alchemy.

Validated against PySCF and/or finite difference. Small grids keep the FD-derivative
checks affordable; they are grid-consistent (the same grid is used throughout), and
the integrals/forces underneath are already machine-precision against PySCF.
"""

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
class TestDipole:
    def test_dipole_vs_finite_field(self):
        """μ_i == -dE/dE_i (consistency of dipole integrals + field coupling)."""
        mol = Molecule.from_xyz(WATER, "sto-3g")
        # coulomb=exact() on both legs: the 1e-6 FD agreement was calibrated
        # on the exact path; under DF the FD legs lose ~2e-6 (follow-up).
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
        """μ_i == -dE/dE_i on the (default) DF backend. Calibrated at 2e-5:
        measured 9.2e-6 (spherical jkfit fit), the exact-path 1e-6 plus what
        the FD legs lose to RI-metric rounding of the two field-displaced
        solves (~1e-9 Ha per leg over h=1e-4). The analytic dipole is the
        clean side here; the FD reference carries the DF noise."""
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
        # coulomb=exact(): the PySCF oracle uses exact ERIs.
        mu = np.asarray(dipole(mol, PBE(), coulomb=exact(), **TOL))
        m = pyscf.gto.M(atom=[[s, tuple(coords[i])] for i, s in enumerate(mol.symbols)],
                        basis="sto-3g", unit="Bohr", verbose=0)
        mf = dft.RKS(m); mf.xc = "pbe"; mf.grids.level = 5; mf.conv_tol = 1e-12; mf.kernel()
        assert np.max(np.abs(mu - mf.dip_moment(unit="au", verbose=0))) < 1e-5

    def test_dipole_open_shell_vs_pyscf(self):
        """Regression: a spin-polarized solve must contribute BOTH spin channels
        to Tr(D·P) — using P[0] alone drops every β electron (silently wrong
        dipole for any spin != 0 molecule)."""
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
        """Regression: the displaced-geometry rebuild inside the FD Hessian must
        keep the molecule's charge/spin — dropping charge made every displaced
        SCF run the neutral (odd-electron) system and raise mid-Hessian."""
        mol = Molecule.from_xyz("He 0 0 0; H 0 0 0.774", "sto-3g", charge=1)
        H = np.asarray(hessian(mol, PBE(), grid=becke(35, 50), **TOL))
        assert H.shape == (6, 6)
        assert np.max(np.abs(H - H.T)) < 1e-6

    def test_polarizability_symmetric_positive(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        a = np.asarray(polarizability(mol, PBE(), field=2e-3, grid=becke(60, 194), **TOL))
        assert np.max(np.abs(a - a.T)) < 1e-4
        assert np.all(np.linalg.eigvalsh(a) > -1e-6)      # positive semidefinite

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
        assert np.max(np.abs(freq[-3:] - fp[-3:])) < 5.0   # cm^-1 (grid-limited)
        assert np.max(np.abs(freq[:6])) < 30.0             # external modes projected ~0

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
        """DF-backed alchemical gradient (the aux basis resolved eagerly, so
        the raw-System charge closure shares the fitted backend) against a
        finite difference on the same DF surface. The FD reference builds the
        aux spherical, matching the resolver's materialized default."""
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
        """Regression: a spin-polarized solve must hold BOTH converged channel
        densities fixed — the old code sliced the α coefficients to nelec//2
        columns and doubled them, a density that is neither the converged
        polarized one nor any valid closed-shell one. Li doublet: Hellmann-
        Feynman needs a stationary SCF, and OH's coarse-grid limit cycle would
        contaminate the FD reference."""
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
    """The orbital-rotation Schur-complement Hessian (dftax.ks.hessian):
    H = E_RR − E_Rκ (E_κκ)⁻¹ E_κR at one converged reference. The FD-of-
    analytic-forces reference is exact-backend only here: on DF paths the FD
    legs' cross-solve force noise (metric-amplified, ~2e-6 Ha/Bohr; see
    _metric_pinv) divided by the step swamps the comparison at ~1e-3. A
    direct energy-FD curvature step-scan pins the DF analytic Hessian at
    1.3e-7 (h=1e-2; smaller steps diverge as ~2e-10 Ha of per-leg DF
    energy noise amplifies by 1/h²), so the analytic path is more accurate
    than either FD reference on DF."""

    def test_h2_matches_fd(self):
        from dftax.energy.xc import LDA

        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
        Ha = np.asarray(hessian(mol, LDA(), method="analytic",
                                grid=becke(35, 50), coulomb=exact(), **TOL))
        Hf = np.asarray(hessian(mol, LDA(), step=1e-3, grid=becke(35, 50),
                                coulomb=exact(), **TOL))
        assert np.abs(Ha - Hf).max() < 1e-5          # measured 8.5e-7
        assert np.abs(Ha - Ha.T).max() == 0.0        # symmetrized exactly

    def test_water_matches_fd(self):
        """Multi-orbital response (nocc=5): the full matrix against the FD
        reference, plus the translational sum rule."""
        from dftax.energy.xc import LDA

        mol = Molecule.from_xyz(WATER, "sto-3g")
        Ha = np.asarray(hessian(mol, LDA(), method="analytic",
                                grid=becke(35, 50), coulomb=exact(),
                                e_tol=1e-10, d_tol=1e-8))
        Hf = np.asarray(hessian(mol, LDA(), step=1e-3, grid=becke(35, 50),
                                coulomb=exact(), e_tol=1e-10, d_tol=1e-8))
        assert np.abs(Ha - Hf).max() < 2e-5          # measured 3.0e-6
        n = Ha.shape[0] // 3
        trans = np.abs(Ha.reshape(n, 3, n, 3).sum(axis=0)).max()
        assert trans < 1e-4                          # sum rule (response-limited)

    def test_open_shell_matches_fd(self):
        """UKS Schur complement (per-channel kappa pytree, unequal nocc):
        the H3 doublet against FD-of-forces on the exact backend, plus the
        translational sum rule."""
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
        """Invariant: a closed-shell molecule pushed through the spin-
        polarized path (spin=0 forces two channels) must reproduce the RKS
        analytic Hessian; the two parametrizations describe the same
        surface."""
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

    def test_empty_beta_channel(self):
        """Edge channels: the H atom's beta channel is empty (nocc=0) and its
        alpha channel has no virtuals in sto-3g (kappa is (1, 0)); the
        Hessian of a free atom is ~0."""
        from dftax.energy.xc import LDA

        mol = Molecule.from_xyz("H 0 0 0", "sto-3g", spin=1)
        Ha = np.asarray(hessian(mol, LDA(), method="analytic",
                                grid=becke(35, 50), coulomb=exact(), **TOL))
        assert Ha.shape == (3, 3)
        assert np.abs(Ha).max() < 1e-6


@pytest.mark.float64
class TestVibrationalSpectra:
    """Sanity for IR / Raman (no pyscf-properties reference available, so we check
    shape, finiteness, the non-negativity guaranteed by the formulas, and that the
    spectrum has a clearly active mode). The underlying Hessian/forces and the
    analytic polarizability are validated to FD/PySCF elsewhere."""

    @pytest.mark.slow
    def test_ir_intensities_sane(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        ir = ir_spectrum(mol, PBE(), grid=becke(35, 50), **TOL)
        freq = np.asarray(ir.frequencies); inten = np.asarray(ir.intensities)
        assert inten.shape == freq.shape == (9,)           # 3N for N=3
        assert np.all(np.isfinite(inten))
        assert np.all(inten > -1e-10)                      # A_k ∝ |dμ/dQ_k|² ≥ 0
        assert inten.max() > 1e-2                           # water has IR-active modes

    @pytest.mark.slow
    def test_raman_activities_sane(self):
        # H2 (N=2) keeps the per-displacement polarizability FD affordable.
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
        r = raman_spectrum(mol, PBE(), grid=becke(20, 50), **TOL)
        freq = np.asarray(r.frequencies); act = np.asarray(r.activities)
        assert act.shape == freq.shape == (6,)             # 3N for N=2
        assert np.all(np.isfinite(act))
        assert np.all(act > -1e-10)                        # 45·ᾱ′² + 7·γ′² ≥ 0
        assert act.max() > 0                               # the H2 stretch is Raman-active
