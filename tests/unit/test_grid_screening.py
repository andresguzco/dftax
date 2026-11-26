import numpy as np
import pytest

from dftax import KS, Molecule, becke, scf
from dftax.energy.xc import LDA, PBE
from dftax.grid.screen import plan_grid_screen
from dftax.ks.terms import ScreenedGridXC

WATER = "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0"


def _screened(n_bucket=3, block=256, screen=1e-10):
    return becke(35, 50, screen=screen, screen_block=block,
                 screen_buckets=n_bucket)


@pytest.mark.float64
@pytest.mark.parametrize("xc", [PBE()], ids=["pbe"])
def test_screened_xc_matches_dense(xc):
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks0 = KS(mol, xc, grid=becke(35, 50))
    kss = KS(mol, xc, grid=_screened())
    assert isinstance(kss.xc_term, ScreenedGridXC)

    P = scf(ks0).P
    assert float(kss.e_xc(P)) == pytest.approx(float(ks0.e_xc(P)), abs=1e-12)

    r0, r1 = scf(ks0), scf(kss)
    assert r0.converged and r1.converged
    assert r1.e_tot == pytest.approx(r0.e_tot, abs=5e-9)


@pytest.mark.float64
def test_screening_plan_covers_every_point_and_pads_with_zero_weight():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, LDA(), grid=becke(35, 50))
    coords = np.asarray(ks.xc_term.coords)
    plan = plan_grid_screen(ks.basis, coords, np.asarray(mol.atom_coords()),
                            block=256, cutoff=1e-10, n_bucket=3)
    ng = coords.shape[0]
    assert plan.n_block * plan.block == ng + plan.n_pad
    real = plan.order[:ng] if plan.n_pad == 0 else plan.order[:ng]
    assert sorted(real.tolist()) == list(range(ng))
    ids = np.concatenate([np.asarray(b.block_ids) for b in plan.buckets])
    assert sorted(ids.tolist()) == list(range(plan.n_block))


@pytest.mark.float64
def test_tighter_cutoff_keeps_more_functions():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks0 = KS(mol, PBE(), grid=becke(35, 50))
    P = scf(ks0).P
    ref = float(ks0.e_xc(P))
    widths = []
    for cut in (1e-14, 1e-6):
        ks = KS(mol, PBE(), grid=_screened(screen=cut))
        widths.append(max(b.sph.shape[1] for b in ks.xc_term.buckets))
        assert float(ks.e_xc(P)) == pytest.approx(ref, abs=1e-9)
    assert widths[0] >= widths[1]


@pytest.mark.float64
@pytest.mark.parametrize("xc", [PBE()], ids=["pbe"])
def test_screened_xc_matches_dense_open_shell(xc):
    ch3 = Molecule.from_xyz(
        "C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0",
        "sto-3g", spin=1,
    )
    ks0 = KS(ch3, xc, grid=becke(35, 50))
    kss = KS(ch3, xc, grid=_screened())
    assert isinstance(kss.xc_term, ScreenedGridXC)

    P = scf(ks0, max_iter=40).P
    assert float(kss.e_xc(P)) == pytest.approx(float(ks0.e_xc(P)), abs=1e-12)

    r0 = scf(ks0, max_iter=60)
    r1 = scf(kss, max_iter=60)
    assert r0.converged and r1.converged
    assert r1.e_tot == pytest.approx(r0.e_tot, abs=5e-9)
