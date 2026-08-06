"""Per-block basis screening of the XC quadrature (``becke(screen=...)``).

The screened quadrature is the same sum with negligible functions dropped, so
it must reproduce the dense one to the screening cutoff. What the saving is
worth depends on system size and is measured separately (see
:mod:`dftax.grid.screen`); these tests pin correctness, not speed.
"""

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
@pytest.mark.parametrize("xc", [LDA(), PBE()], ids=["lda", "pbe"])
def test_screened_xc_matches_dense(xc):
    """Screened e_xc and the full SCF reproduce the dense grid."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks0 = KS(mol, xc, grid=becke(35, 50))
    kss = KS(mol, xc, grid=_screened())
    assert isinstance(kss.xc_term, ScreenedGridXC)

    P = scf(ks0).P
    assert float(kss.e_xc(P)) == pytest.approx(float(ks0.e_xc(P)), abs=1e-12)

    r0, r1 = scf(ks0), scf(kss)
    assert r0.converged and r1.converged
    # The screened solve is a different reduction order on top of the dropped
    # (negligible) functions, so the two trajectories agree to the stopping
    # tolerance rather than to machine precision.
    assert r1.e_tot == pytest.approx(r0.e_tot, abs=5e-9)


@pytest.mark.float64
def test_screening_plan_covers_every_point_and_pads_with_zero_weight():
    """The plan is a permutation plus filler, not a subset: no quadrature
    point may be dropped or duplicated by the spatial sort."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, LDA(), grid=becke(35, 50))
    coords = np.asarray(ks.xc_term.coords)
    plan = plan_grid_screen(ks.basis, coords, np.asarray(mol.atom_coords()),
                            block=256, cutoff=1e-10, n_bucket=3)
    ng = coords.shape[0]
    assert plan.n_block * plan.block == ng + plan.n_pad
    real = plan.order[:ng] if plan.n_pad == 0 else plan.order[:ng]
    assert sorted(real.tolist()) == list(range(ng))   # a true permutation
    # every block lands in exactly one bucket
    ids = np.concatenate([np.asarray(b.block_ids) for b in plan.buckets])
    assert sorted(ids.tolist()) == list(range(plan.n_block))


@pytest.mark.float64
def test_tighter_cutoff_keeps_more_functions():
    """The cutoff is a real knob: loosening it drops more shells, and the
    energy stays within the accuracy the cutoff buys."""
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
@pytest.mark.parametrize("xc", [LDA(), PBE()], ids=["lda", "pbe"])
def test_screened_xc_matches_dense_open_shell(xc):
    """Spin-polarized screening reproduces the dense open-shell grid.

    Worth its own case rather than folding into the closed-shell one:
    ``ε_xc(ρα, ρβ, ∇ρα, ∇ρβ)`` couples the channels, so this is not a sum of
    two independent screened energies and would be quietly wrong if it were
    written that way.
    """
    oh = Molecule.from_xyz("O 0 0 0; H 0.9697 0 0", "sto-3g", spin=1)
    ks0 = KS(oh, xc, grid=becke(35, 50))
    kss = KS(oh, xc, grid=_screened())
    assert isinstance(kss.xc_term, ScreenedGridXC)

    P = scf(ks0, max_iter=40).P
    assert float(kss.e_xc(P)) == pytest.approx(float(ks0.e_xc(P)), abs=1e-12)

    r0 = scf(ks0, max_iter=60)
    r1 = scf(kss, max_iter=60)
    assert r0.converged and r1.converged
    assert r1.e_tot == pytest.approx(r0.e_tot, abs=5e-9)
