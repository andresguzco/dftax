"""The native (BSE) basis loader against PySCF's, for a generally contracted
basis.

For cc-pVDZ the two ship different contractions of the same space: BSE gives
carbon's s block as the canonical general contraction, PySCF's file splits the
diffuse primitive out. The invariant is therefore span equivalence, not matrix
equality; the overlaps differ, the Rayleigh-Ritz values do not.

``test_native_pipeline`` covers the segmented case (sto-3g) elementwise.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import gto

from dftax.basis.loader import build_basis_data
from dftax.energy.gto import extract_basis_data
from dftax.integrals import (
    kinetic_matrix, nuclear_attraction_matrix, overlap_matrix,
)


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
def test_general_contraction_spans_the_same_space():
    """Same span, same Rayleigh-Ritz values: the core Hamiltonian's
    generalized eigenvalues ``h c = e S c`` are a property of the space, not
    of the contraction, so they agree although S and h differ (above). A
    different span moves them at the 1e-3 level the overlaps differ by."""
    pmol, native, pyscf_side = _pair("cc-pvdz")
    X = jnp.asarray(pmol.atom_coords())
    Z = jnp.asarray(pmol.atom_charges(), dtype=jnp.float64)

    def ritz(b):
        S = np.asarray(overlap_matrix(b))
        h = np.asarray(kinetic_matrix(b) + nuclear_attraction_matrix(b, X, Z))
        s, U = np.linalg.eigh(S)
        Xs = U / np.sqrt(s)
        return np.linalg.eigvalsh(Xs.T @ h @ Xs)

    # relative 1e-7: the two sides publish the basis to 8 and 10 digits
    np.testing.assert_allclose(ritz(native), ritz(pyscf_side), rtol=1e-7,
                               atol=1e-7)
