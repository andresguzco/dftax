"""Fractional occupations must not silently meet a frozen-orbital exchange.

The streamed RI-K recovers occupied orbitals from ``P`` and treats the top
``nocc`` as fully occupied. That is exact at an idempotent density and wrong
at a smeared one, and it was wrong *quietly*: measured on water/sto-3g/PBE0
against the materialized backend, 7.1e-6 Ha apart without smearing (the
expected cartesian-vs-spherical auxiliary span difference) and 1.4e-3 Ha apart
with ``fermi(sigma=0.05)``, with ``converged=True`` reported both times.

``forces`` has always rejected the combination; the SCF did not.
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
    with pytest.raises(NotImplementedError, match="streamed RI-K"):
        scf(_ks(PBE0(), 32), guess=minao(), smearing=fermi(sigma=0.05),
            max_iter=1)


def test_smeared_streamed_pure_dft_is_allowed():
    """No exact exchange means no frozen orbitals, so nothing to refuse."""
    res = scf(_ks(PBE(), 32), guess=minao(), smearing=fermi(sigma=0.05),
              max_iter=40)
    assert res.e_tot < 0.0


def test_smeared_materialized_hybrid_is_allowed():
    """The materialized backend contracts the density directly and makes no
    idempotency assumption, so it is the route the error message points at."""
    res = scf(_ks(PBE0(), None), guess=minao(), smearing=fermi(sigma=0.05),
              max_iter=40)
    assert res.e_tot < 0.0


def test_unsmeared_streamed_hybrid_still_runs():
    res = scf(_ks(PBE0(), 32), guess=minao(), max_iter=40)
    assert res.e_tot < 0.0
