"""The native (BSE) basis loader against PySCF's, for a generally contracted
basis.

For cc-pVDZ the two ship different contractions of the same space: BSE gives
carbon's s block as the canonical general contraction, PySCF's file splits the
diffuse primitive out. The invariant is therefore span equivalence, not matrix
equality; the overlaps differ, the variational energy does not.

``test_native_pipeline`` covers the segmented case (sto-3g) elementwise.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import gto

from dftax import KS, exact, minao, scf
from dftax.basis.loader import build_basis_data
from dftax.energy.gto import extract_basis_data
from dftax.energy.xc import LDA
from dftax.integrals import overlap_matrix
from dftax.system.molecule import Molecule


H2O = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"


def _pair(basis_name):
    """The same molecule through both loaders."""
    pmol = gto.M(atom=H2O, basis=basis_name).build()
    native = build_basis_data(["O", "H", "H"], pmol.atom_coords(), basis_name,
                             spherical=True)
    return pmol, native, extract_basis_data(pmol)


@pytest.mark.pyscf
def test_segmented_basis_matches_to_the_data_precision():
    """sto-3g is segmented in both, so this one is elementwise -- but only to
    the precision at which each side publishes the basis. PySCF ships sto-3g
    to 8 significant figures and BSE to 10, which puts ~1e-8 in the overlap.

    That bound caps what any cross-loader comparison can show: an energy
    difference below ~1e-8 Ha between the two is data precision, not physics.
    """
    _pmol, native, pyscf_side = _pair("sto-3g")
    dS = float(jnp.max(jnp.abs(overlap_matrix(native)
                               - overlap_matrix(pyscf_side))))
    assert dS < 1e-7, dS
    assert dS > 1e-12, f"{dS} -- if this ever gets tighter, one side changed "\
                       "its data file and the note above is stale"


@pytest.mark.pyscf
def test_general_contraction_differs_elementwise():
    """cc-pVDZ does not, and pinning that keeps the next reader from 'fixing'
    the loader to match PySCF row for row."""
    _pmol, native, pyscf_side = _pair("cc-pvdz")
    Sn = np.asarray(overlap_matrix(native))
    Sp = np.asarray(overlap_matrix(pyscf_side))
    assert Sn.shape == Sp.shape                      # same dimension
    assert np.max(np.abs(Sn - Sp)) > 1e-3            # different functions
    # not a permutation either: the spectra differ
    assert np.max(np.abs(np.sort(np.linalg.eigvalsh(Sn))
                         - np.sort(np.linalg.eigvalsh(Sp)))) > 1e-3


@pytest.mark.pyscf
@pytest.mark.float64
@pytest.mark.slow
def test_general_contraction_spans_the_same_space():
    """Same span, same variational energy. Both sides run through dftax on one
    shared grid and the exact 4-center backend, so the only difference is which
    contraction of cc-pVDZ was used."""
    from pyscf import dft

    pmol = gto.M(atom=H2O, basis="cc-pvdz").build()
    mf = dft.RKS(pmol)
    mf.xc, mf.grids.level, mf.verbose = "slater,vwn5", 3, 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))

    nmol = Molecule(["O", "H", "H"], pmol.atom_coords(), "cc-pvdz",
                    spherical=True)
    e_native = scf(KS(nmol, LDA(), grid=grid, coulomb=exact()),
                   guess=minao()).e_tot
    e_pyscf = scf(KS(pmol, LDA(), grid=grid, coulomb=exact()),
                  guess=minao()).e_tot

    # Loose against the 2.6e-11 measured, because both sides stop at their
    # own SCF tolerance.
    assert abs(float(e_native) - float(e_pyscf)) < 1e-8
