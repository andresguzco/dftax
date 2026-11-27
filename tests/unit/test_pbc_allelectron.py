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


AUX = "def2-universal-jkfit"


def _density(n):
    d = np.random.default_rng(0).standard_normal((n, n))
    return d + d.T


@pytest.mark.pyscf
@pytest.mark.parametrize("name,auxname", [
    ("sto-3g", "sto-3g"),
    pytest.param("sto-3g", AUX, marks=pytest.mark.slow),
    pytest.param("pob-tzvp", AUX, marks=pytest.mark.slow),
])
def test_gdf_coulomb_matches_pyscf(name, auxname):
    import basis_set_exchange as bse
    from pyscf.pbc import df, gto

    from dftax.pbc.gdf import gdf_coulomb, gdf_tensors, plan_gdf

    cell = _pyscf_cell(name)
    mydf = df.GDF(cell)
    mydf.auxbasis = {"C": gto.basis.parse(
        bse.get_basis(auxname, elements=["C"], fmt="nwchem", header=False))}
    D = _density(cell.nao)
    ref = np.asarray(mydf.get_jk(D, with_k=False)[0]).reshape(cell.nao, cell.nao)
    b = _basis(name)
    x = build_basis_data(["C", "C"], COORDS, auxname, spherical=True)
    j3c, j2c = gdf_tensors(b, x, A, plan_gdf(b, x, A, COORDS))
    J = np.asarray(gdf_coulomb(j3c, j2c, D))
    w = np.linalg.eigvalsh(np.asarray(j2c))
    print(f"{name}/{auxname}: metric eig {w[0]:.2e}..{w[-1]:.2e}, "
          f"max |J| {np.abs(ref).max():.3f}, max |dJ| {np.abs(J - ref).max():.2e}")
    assert np.abs(J - ref).max() < 1e-9 * max(1.0, np.abs(ref).max())


def test_gdf_coulomb_is_independent_of_omega():
    from dftax.pbc.gdf import gdf_coulomb, gdf_tensors, plan_gdf

    b = _basis("sto-3g")
    x = build_basis_data(["C", "C"], COORDS, "sto-3g", spherical=True)
    D = _density(b.cart2sph.shape[1] if b.cart2sph is not None else b.centers.shape[0])
    J1, J2 = (np.asarray(gdf_coulomb(*gdf_tensors(b, x, A, plan_gdf(b, x, A, COORDS, omega=w)),
                                     D)) for w in (0.8, 1.2))
    print(f"max |J(0.8) - J(1.2)| {np.abs(J1 - J2).max():.2e}")
    assert np.abs(J1 - J2).max() < 1e-10 * max(1.0, np.abs(J1).max())
