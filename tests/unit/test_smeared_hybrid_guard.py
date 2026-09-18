"""Fractional occupations must not silently meet a frozen-orbital exchange.

The streamed RI-K recovers occupied orbitals from ``P`` and treats the top
``nocc`` as fully occupied, which is exact at an idempotent density and wrong
at a smeared one. ``scf`` refuses the combination, as ``forces`` already did.
"""

import pytest

from dftax import KS, Molecule, df, fermi, minao, scf
from dftax.energy.xc import PBE, PBE0
from dftax.grid import becke


WATER = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"


def _ks(xc, chunk):
    mol = Molecule.from_xyz(WATER, "sto-3g", spherical=True)
    return KS(mol, xc, grid=becke(50, 194, prune=None),
              coulomb=df("def2-universal-jkfit", chunk=chunk))


def test_smeared_streamed_hybrid_is_refused():
    """Not marked slow: it raises before any solve, so it costs milliseconds
    and belongs in the default gate."""
    with pytest.raises(NotImplementedError, match="streamed RI-K"):
        scf(_ks(PBE0(), 32), guess=minao(), smearing=fermi(sigma=0.05),
            max_iter=1)


@pytest.mark.slow
def test_smeared_streamed_pure_dft_is_allowed():
    """No exact exchange means no frozen orbitals, so nothing to refuse."""
    res = scf(_ks(PBE(), 32), guess=minao(), smearing=fermi(sigma=0.05),
              max_iter=40)
    assert res.e_tot < 0.0


@pytest.mark.slow
def test_smeared_materialized_hybrid_is_allowed():
    """The materialized backend contracts the density directly, which is the
    route the error message points at."""
    res = scf(_ks(PBE0(), None), guess=minao(), smearing=fermi(sigma=0.05),
              max_iter=40)
    assert res.e_tot < 0.0


@pytest.mark.slow
def test_unsmeared_streamed_hybrid_still_runs():
    res = scf(_ks(PBE0(), 32), guess=minao(), max_iter=40)
    assert res.e_tot < 0.0
