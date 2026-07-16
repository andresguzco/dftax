"""PySCF-free pipeline: native Molecule + BSE basis + native Becke grid.

Validates that a calculation built with zero PySCF objects (geometry, basis,
and integration grid all native) reproduces a PySCF reference. PySCF appears
only as the oracle.
"""

import jax.numpy as jnp
import pytest

from pyscf import gto, dft

from dftax.energy.xc import LDA, PBE, B3LYP
from dftax.system.molecule import Molecule
from dftax.basis.loader import build_basis_data
from dftax.integrals import overlap_matrix
from dftax import KS, scf, exact
from dftax.grid import becke_grid, lebedev_grid, available_lebedev

H2O = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"


@pytest.mark.pyscf
class TestNativeBasis:
    def test_nao_and_normalization(self):
        pmol = gto.M(atom=H2O, basis="sto-3g").build()
        nmol = Molecule(["O", "H", "H"], pmol.atom_coords(), "sto-3g")
        basis = build_basis_data(nmol.symbols, nmol.atom_coords(), "sto-3g")
        S = overlap_matrix(basis)
        assert S.shape[0] == pmol.nao  # l<=1: Cartesian == spherical count
        # Each contracted AO is normalized: diag(S) == 1.
        assert float(jnp.max(jnp.abs(jnp.diag(S) - 1.0))) < 1e-10

    @pytest.mark.float64
    def test_energy_matches_pyscf_on_pyscf_grid(self):
        # Native basis, PySCF grid: isolates basis-loading correctness.
        pmol = gto.M(atom=H2O, basis="sto-3g").build()
        nmol = Molecule(["O", "H", "H"], pmol.atom_coords(), "sto-3g")
        for pyscf_xc, xc_obj in [("slater,vwn5", LDA()), ("b3lyp", B3LYP())]:
            mf = dft.RKS(pmol)
            mf.xc = pyscf_xc
            mf.grids.level = 3
            mf.verbose = 0
            e_ref = mf.kernel()
            grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
            res = scf(KS(nmol, xc_obj, grid=(grid[0], grid[1]), coulomb=exact()))
            assert res.converged
            assert abs(res.e_tot - e_ref) < 1e-6


class TestNativeGrid:
    def test_lebedev_available(self):
        assert 302 in available_lebedev()
        pts, w = lebedev_grid(302)
        assert pts.shape == (302, 3)
        assert abs(float(jnp.sum(w)) - 1.0) < 1e-10

    @pytest.mark.float64
    def test_integrates_density_to_nelec(self):
        nmol = Molecule.from_xyz(H2O, "sto-3g")
        coords, weights = becke_grid(nmol.symbols, nmol.atom_coords(), 75, 302)
        ks = KS(nmol, LDA(), grid=(coords, weights))
        res = scf(ks)
        rho, _ = ks.density(res.P)
        assert abs(float(jnp.sum(weights * rho)) - nmol.nelectron) < 1e-4


@pytest.mark.pyscf
@pytest.mark.float64
class TestFullyPyscfFree:
    """Native geometry + native basis + native grid vs a PySCF reference."""

    @pytest.mark.parametrize("pyscf_xc,xc_cls", [("slater,vwn5", LDA), ("pbe", PBE)])
    def test_water(self, pyscf_xc, xc_cls):
        nmol = Molecule.from_xyz(H2O, "sto-3g")
        pmol = gto.M(atom=H2O, basis="sto-3g").build()
        mf = dft.RKS(pmol)
        mf.xc = pyscf_xc
        mf.grids.level = 3
        mf.verbose = 0
        e_ref = mf.kernel()

        coords, weights = becke_grid(nmol.symbols, nmol.atom_coords(), 75, 302)
        res = scf(KS(nmol, xc_cls(), grid=(coords, weights), coulomb=exact()))
        assert res.converged
        assert abs(res.e_tot - e_ref) < 5e-5


def test_bragg_radius_beyond_xe_raises():
    from dftax.grid.becke import bragg_radius

    with pytest.raises(ValueError, match="Xe"):
        bragg_radius(55)
    with pytest.raises(ValueError, match="Xe"):
        bragg_radius(0)                                   # ghost placeholder
