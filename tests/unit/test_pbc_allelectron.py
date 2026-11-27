import numpy as np
import pytest

from dftax.basis.loader import build_basis_data
from dftax.pbc.allelectron import nuclear_attraction_cell, plan_nuclear
from dftax.system.molecule import ANGSTROM_TO_BOHR

A = 0.5 * 3.567 * ANGSTROM_TO_BOHR * np.array(
    [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
COORDS = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]) @ A
CHARGES = np.array([6.0, 6.0])


def _basis(name):
    return build_basis_data(["C", "C"], COORDS, name, spherical=True)


def _pyscf_cell(name):
    import basis_set_exchange as bse
    from pyscf.pbc import gto

    cell = gto.Cell()
    cell.atom = [["C", tuple(r)] for r in COORDS]
    cell.a = A
    cell.unit = "B"
    cell.basis = {"C": gto.basis.parse(
        bse.get_basis(name, elements=["C"], fmt="nwchem", header=False))}
    cell.precision = 1e-12
    cell.verbose = 0
    cell.build()
    return cell


@pytest.mark.pyscf
@pytest.mark.parametrize("name", ["sto-3g", pytest.param("pob-tzvp", marks=pytest.mark.slow)])
def test_nuclear_attraction_matches_pyscf(name):
    from pyscf.pbc import df

    cell = _pyscf_cell(name)
    ref = np.asarray(df.GDF(cell).get_nuc()).reshape(cell.nao, cell.nao)
    b = _basis(name)
    v = np.asarray(nuclear_attraction_cell(b, A, COORDS, CHARGES,
                                           plan_nuclear(b, A, COORDS)))
    print(f"{name}: nao {cell.nao}, max |V| {np.abs(ref).max():.3f}, "
          f"max |dV| {np.abs(v - ref).max():.2e}")
    assert np.abs(v - ref).max() < 1e-9


def test_nuclear_attraction_is_independent_of_omega():
    b = _basis("sto-3g")
    v1, v2 = (np.asarray(nuclear_attraction_cell(
        b, A, COORDS, CHARGES, plan_nuclear(b, A, COORDS, omega=w))) for w in (0.4, 0.8))
    print(f"max |V(0.4) - V(0.8)| {np.abs(v1 - v2).max():.2e}")
    assert np.abs(v1 - v2).max() < 1e-10
