"""gdm over a batch of different molecules, padded to common sizes.

The padding (:mod:`dftax.ks.batched_gdm`) must be inert: a padded molecule is
the same energy function on its real block, so the batched minimum is each
molecule's own minimum, whatever it was batched with.
"""

import numpy as np
import pytest

from dftax import KS, Molecule, becke, gdm, scf
from dftax.energy.xc import B3LYP, PBE
from dftax.ks.batched_gdm import Bucket, bucket_for, gdm_batched

MOLS = {
    "water": "O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50",
    "ammonia": "N 0 0 0; H 0.94 0 -0.38; H -0.47 0.81 -0.38; "
               "H -0.47 -0.81 -0.38",
    "hf": "F 0 0 0; H 0 0 0.92",
    "methane": "C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; "
               "H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63",
}


def _ks(name, basis="6-31g", xc=PBE):
    return KS(Molecule.from_xyz(MOLS[name], basis), xc(), grid=becke(35, 50))


@pytest.mark.float64
@pytest.mark.slow
def test_padding_is_inert():
    """One molecule padded far past its sizes, against its own gdm."""
    ks = _ks("water")
    ref = gdm(ks)
    b0 = bucket_for([ks])
    big = Bucket(nao=b0.nao + 9, naux=b0.naux + 17, ng=b0.ng + 333,
                 nocc=(b0.nocc[0] + 3,))
    res = gdm_batched([ks], bucket=big)
    assert ref.converged and bool(res.converged[0])
    assert abs(float(res.e_tot[0]) - ref.e_tot) < 1e-7
    np.testing.assert_allclose(res.P[0], np.asarray(ref.P), atol=1e-4)


@pytest.mark.float64
@pytest.mark.slow
def test_batch_of_different_molecules_matches_scf():
    kss = [_ks(n) for n in MOLS]
    res = gdm_batched(kss)
    for i, (name, ks) in enumerate(zip(MOLS, kss)):
        r = scf(ks)
        assert bool(res.converged[i]), name
        assert abs(float(res.e_tot[i]) - r.e_tot) < 1e-7, name


@pytest.mark.float64
@pytest.mark.slow
def test_batched_hybrid_matches_scf():
    """Hybrid exchange sees the occupations, so padded slots stay empty."""
    kss = [_ks(n, xc=B3LYP) for n in ("water", "hf", "methane")]
    res = gdm_batched(kss)
    for i, ks in enumerate(kss):
        r = scf(ks)
        assert bool(res.converged[i])
        assert abs(float(res.e_tot[i]) - r.e_tot) < 1e-7


RADICALS = {
    "oh": ("O 0 0 0; H 0 0 0.97", 1),
    "ch3": ("C 0 0 0; H 1.08 0 0; H -0.54 0.935 0; H -0.54 -0.935 0", 1),
    "nh2": ("N 0 0 0; H 1.01 0 0; H -0.26 0.98 0", 1),
    "o2": ("O 0 0 0; O 0 0 1.21", 2),
}


def _radical(name, xc=PBE):
    atoms, spin = RADICALS[name]
    return KS(Molecule.from_xyz(atoms, "6-31g", spin=spin), xc(),
              grid=becke(35, 50), spin=spin)


@pytest.mark.float64
@pytest.mark.slow
def test_open_shells_and_mixed_batches_match_scf():
    """Radicals and a triplet (different alpha/beta padding per channel),
    mixed with closed shells: one batch per spin type, results in order."""
    kss = [_radical("oh"), _ks("water"), _radical("ch3"), _radical("o2"),
           _ks("hf"), _radical("nh2"), _radical("oh", xc=B3LYP)]
    res = gdm_batched(kss)
    for i, ks in enumerate(kss):
        r = scf(ks)
        assert bool(res.converged[i]), i
        assert abs(float(res.e_tot[i]) - r.e_tot) < 1e-7, i
        assert res.P[i].shape == (len(ks.nocc), ks.S.shape[0], ks.S.shape[0])


def test_rejects_what_it_does_not_pad_yet():
    with pytest.raises(NotImplementedError, match="materialized"):
        gdm_batched([KS(Molecule.from_xyz(MOLS["water"], "sto-3g"), PBE(),
                        grid=becke(35, 50, chunk=64))])
