"""The native (BSE) basis loader against PySCF's, for a *generally contracted*
basis.

``test_native_pipeline`` validates the loader on sto-3g only, where BSE and
PySCF ship the same segmented contraction and every matrix matches elementwise.
That leaves the interesting case untested: for cc-pVDZ the two ship genuinely
different contractions of the same space. BSE gives carbon's s block as the
canonical general contraction (nine primitives, three coefficient columns);
PySCF's shipped file splits the diffuse primitive out (eight primitives, two
columns, plus a one-primitive shell).

The invariant is therefore **span equivalence, not matrix equality**. The
overlap matrices differ, and their eigenvalues differ, because the individual
basis functions differ; the space they span is the same, so a variational
energy in it is the same. Asserting elementwise agreement here would be
asserting something false, and asserting nothing would leave the flagship
"PySCF-free" path unchecked on every cc-pVXZ-family basis.

Measured when this was first characterized: water/cc-pVDZ/LDA on a matched
grid, |dE| = 2.6e-11 Ha between the two loaders, while max|dS| = 0.30.
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
    """sto-3g is segmented in both, so this one *is* elementwise -- but only
    to the precision at which each side publishes the basis, not to machine
    precision.

    PySCF ships sto-3g to 8 significant figures and BSE to 10:

        exponent    BSE 130.7093214    PySCF 130.70932
        coefficient BSE 0.1543289673   PySCF 0.15432897

    A ~1e-8 relative difference in the parameters gives ~1e-8 in the overlap,
    and 1.7e-8 is what this measures. Asserting 1e-12 here (as this test first
    did) asserts agreement tighter than the data itself, which no loader could
    deliver. The bound is worth pinning because it caps what *any*
    cross-loader comparison can show: an energy difference below ~1e-8 Ha
    between the two is data precision, not physics.
    """
    _pmol, native, pyscf_side = _pair("sto-3g")
    dS = float(jnp.max(jnp.abs(overlap_matrix(native)
                               - overlap_matrix(pyscf_side))))
    assert dS < 1e-7, dS
    assert dS > 1e-12, f"{dS} -- if this ever gets tighter, one side changed "\
                       "its data file and the note above is stale"


@pytest.mark.pyscf
def test_general_contraction_differs_elementwise():
    """cc-pVDZ does not, and pinning that keeps the next reader from
    'fixing' the loader to match PySCF row for row."""
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
    """The invariant that actually matters: same span, same variational energy.

    Both sides run through dftax, on one shared grid and the exact 4-center
    backend, so the only difference is which contraction of cc-pVDZ was used.
    """
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

    # Two different bases spanning one space: the variational minimum in that
    # space is the same number. 1e-8 is loose against the 2.6e-11 measured,
    # because both sides stop at their own SCF tolerance.
    assert abs(float(e_native) - float(e_pyscf)) < 1e-8
