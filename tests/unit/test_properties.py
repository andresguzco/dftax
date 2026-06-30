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
from dftax import dipole, polarizability, ir_spectrum, raman_spectrum, alchemical_deriv
from dftax.ks.properties import _solve_field, _grid
from dftax.integrals.multipole import dipole_matrices
from dftax.ks.energy import RKS
from dftax.ks.scf import rks_scf

WATER = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"
TOL = dict(e_tol=1e-11, d_tol=1e-9)


@pytest.mark.float64
class TestDipole:
    def test_dipole_vs_finite_field(self):
        """μ_i == -dE/dE_i (consistency of dipole integrals + field coupling)."""
        mol = Molecule.from_xyz(WATER, "sto-3g")
        mu = np.asarray(dipole(mol, PBE(), **TOL))
        gc, gw = _grid(mol, 75, 302)
        h = 1e-4
        fd = []
        for i in range(3):
            ep = _solve_field(mol, PBE(), gc, gw, field=jnp.zeros(3).at[i].set(h), **TOL)[1]
            em = _solve_field(mol, PBE(), gc, gw, field=jnp.zeros(3).at[i].set(-h), **TOL)[1]
            fd.append(-(ep - em) / (2 * h))
        assert np.max(np.abs(mu - np.array(fd))) < 1e-6

    def test_dipole_vs_pyscf(self):
        pyscf = pytest.importorskip("pyscf")
        from pyscf import dft
        mol = Molecule.from_xyz(WATER, "sto-3g")
        coords = mol.atom_coords()
        mu = np.asarray(dipole(mol, PBE(), **TOL))
        m = pyscf.gto.M(atom=[[s, tuple(coords[i])] for i, s in enumerate(mol.symbols)],
                        basis="sto-3g", unit="Bohr", verbose=0)
        mf = dft.RKS(m); mf.xc = "pbe"; mf.grids.level = 5; mf.conv_tol = 1e-12; mf.kernel()
        assert np.max(np.abs(mu - mf.dip_moment(unit="au", verbose=0))) < 1e-5

    def test_dipole_matrix_symmetric(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        basis = RKS.from_molecule(mol, PBE(), *_grid(mol, 35, 50)).basis
        D = np.asarray(dipole_matrices(basis))
        assert np.max(np.abs(D - np.transpose(D, (0, 2, 1)))) < 1e-12


@pytest.mark.float64
class TestResponse:
    def test_polarizability_symmetric_positive(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        a = np.asarray(polarizability(mol, PBE(), field=2e-3, n_radial=60, lebedev=194, **TOL))
        assert np.max(np.abs(a - a.T)) < 1e-4
        assert np.all(np.linalg.eigvalsh(a) > -1e-6)      # positive semidefinite

    def test_frequencies_vs_pyscf(self):
        pyscf = pytest.importorskip("pyscf")
        from pyscf import dft
        from pyscf.hessian import thermo
        mol = Molecule.from_xyz(WATER, "sto-3g")
        coords = mol.atom_coords()
        freq = np.sort(np.asarray(ir_spectrum(
            mol, PBE(), step=2e-3, n_radial=60, lebedev=194, **TOL)["frequencies"]))
        m = pyscf.gto.M(atom=[[s, tuple(coords[i])] for i, s in enumerate(mol.symbols)],
                        basis="sto-3g", unit="Bohr", verbose=0)
        mf = dft.RKS(m); mf.xc = "pbe"; mf.grids.level = 5; mf.conv_tol = 1e-12; mf.kernel()
        fp = np.sort(np.asarray(thermo.harmonic_analysis(m, mf.Hessian().kernel())["freq_wavenumber"]).real)
        assert np.max(np.abs(freq[-3:] - fp[-3:])) < 5.0   # cm^-1 (grid-limited)
        assert np.max(np.abs(freq[:6])) < 30.0             # external modes projected ~0

    def test_alchemical_vs_finite_difference(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        coords = mol.atom_coords()
        dEdZ = np.asarray(alchemical_deriv(mol, PBE(), n_radial=60, lebedev=194, **TOL))
        gc, gw = _grid(mol, 60, 194)
        basis = RKS.from_molecule(mol, PBE(), gc, gw).basis

        def E_at(charges):
            k = RKS._assemble(basis, jnp.asarray(coords), jnp.asarray(charges),
                              mol.nelectron, PBE(), gc, gw)
            return float(rks_scf(k, **TOL).e_tot)

        ch0 = np.asarray(mol.atom_charges(), float); h = 1e-3
        fd = np.array([(E_at(ch0 + h * (np.arange(len(ch0)) == a))
                        - E_at(ch0 - h * (np.arange(len(ch0)) == a))) / (2 * h)
                       for a in range(len(ch0))])
        assert np.max(np.abs(dEdZ - fd)) < 1e-5


@pytest.mark.float64
class TestVibrationalSpectra:
    """Sanity for IR / Raman (no pyscf-properties reference available, so we check
    shape, finiteness, the non-negativity guaranteed by the formulas, and that the
    spectrum has a clearly active mode). The underlying Hessian/forces and the
    analytic polarizability are validated to FD/PySCF elsewhere."""

    @pytest.mark.slow
    def test_ir_intensities_sane(self):
        mol = Molecule.from_xyz(WATER, "sto-3g")
        ir = ir_spectrum(mol, PBE(), n_radial=35, lebedev=50, **TOL)
        freq = np.asarray(ir["frequencies"]); inten = np.asarray(ir["intensities"])
        assert inten.shape == freq.shape == (9,)           # 3N for N=3
        assert np.all(np.isfinite(inten))
        assert np.all(inten > -1e-10)                      # A_k ∝ |dμ/dQ_k|² ≥ 0
        assert inten.max() > 1e-2                           # water has IR-active modes

    @pytest.mark.slow
    def test_raman_activities_sane(self):
        # H2 (N=2) keeps the per-displacement polarizability FD affordable.
        mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
        r = raman_spectrum(mol, PBE(), n_radial=20, lebedev=50, **TOL)
        freq = np.asarray(r["frequencies"]); act = np.asarray(r["activities"])
        assert act.shape == freq.shape == (6,)             # 3N for N=2
        assert np.all(np.isfinite(act))
        assert np.all(act > -1e-10)                        # 45·ᾱ′² + 7·γ′² ≥ 0
        assert act.max() > 0                               # the H2 stretch is Raman-active
