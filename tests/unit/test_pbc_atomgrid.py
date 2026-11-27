import numpy as np
import pytest

from dftax.pbc.atomgrid import becke_cell, cell_grid, plan_cell_grid
from dftax.pbc.lattice import lattice_images
from dftax.system.molecule import ANGSTROM_TO_BOHR

A = 0.5 * 3.567 * ANGSTROM_TO_BOHR * np.array(
    [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
COORDS = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]) @ A


def _gaussians(points, alpha):
    shifts = lattice_images(A, np.sqrt(40.0 / alpha) + 10.0) @ A
    centres = (COORDS[:, None] + shifts[None]).reshape(-1, 3)
    d2 = np.sum((points[:, None] - centres[None]) ** 2, axis=-1)
    return (alpha / np.pi) ** 1.5 * np.exp(-alpha * d2).sum(axis=1)


@pytest.mark.pyscf
def test_becke_cell_matches_pyscf():
    import basis_set_exchange as bse
    from pyscf.pbc import gto
    from pyscf.pbc.dft import gen_grid

    cell = gto.Cell()
    cell.atom = [["C", tuple(r)] for r in COORDS]
    cell.a = A
    cell.unit = "B"
    cell.basis = {"C": gto.basis.parse(
        bse.get_basis("pob-tzvp", elements=["C"], fmt="nwchem", header=False))}
    cell.precision = 1e-10
    cell.verbose = 0
    cell.build()
    ref = gen_grid.BeckeGrids(cell)
    ref.atom_grid = (75, 302)
    ref.build()

    pts, w = cell_grid(plan_cell_grid(becke_cell(75, 302), ["C", "C"], COORDS, A),
                       COORDS, A)
    pts, w = np.asarray(pts), np.asarray(w)
    print(f"points dftax {w.size} pyscf {ref.weights.size}; "
          f"sum w {w.sum():.12f} {ref.weights.sum():.12f} vol {cell.vol:.12f}")
    assert w.size == ref.weights.size
    assert abs(w.sum() - ref.weights.sum()) < 1e-10
    for alpha in (0.5, 20.0):
        n, n_ref = np.sum(w * _gaussians(pts, alpha)), np.sum(
            ref.weights * _gaussians(ref.coords, alpha))
        print(f"alpha {alpha}: {n:.12f} pyscf {n_ref:.12f}")
        assert abs(n - n_ref) < 1e-10
