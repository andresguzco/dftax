"""High-angular-momentum auxiliary bases (h/i, l=5/6) for density fitting.

def2-universal-jkfit carries h functions from Rb and i functions for the 3d
row, so DF for transition metals needs the 2- and 3-center engines beyond the
g cap. PySCF (cart=True) is the integral oracle; the synthetic-shell tests
also pin our l=5/6 Cartesian component ordering to libcint's.
"""

import numpy as np
import pytest

from dftax import KS, Molecule
from dftax.basis.loader import build_basis_data
from dftax.energy.gto import _CART_COMPONENTS, extract_basis_data
from dftax.energy.xc import PBE
from dftax.integrals import eri2c_matrix, eri3c_matrix
from dftax.ks import exact, df
from dftax.ks.guess import density_from_guess, minao
from dftax.ks.scf import canonical_orthonormalizer


def test_cart_component_counts():
    assert len(_CART_COMPONENTS[5]) == 21
    assert len(_CART_COMPONENTS[6]) == 28
    for l in (5, 6):
        comps = _CART_COMPONENTS[l]
        assert all(sum(c) == l for c in comps)
        assert len(set(comps)) == len(comps)
        # lexicographic: lx descending, then ly descending
        assert comps == sorted(comps, key=lambda c: (-c[0], -c[1]))


@pytest.mark.pyscf
@pytest.mark.parametrize("l", [5, 6])
def test_eri2c_high_l_vs_pyscf(l):
    """(P|Q) for a synthetic h/i shell pair matches PySCF int2c2e (cart)."""
    from pyscf import gto

    basis = {"He": [[l, [1.9, 0.7], [0.45, 0.6]], [0, [0.8, 1.0]]]}
    mol = gto.M(atom="He 0 0 0; He 0 0.5 1.4", basis=basis, cart=True)
    aux = extract_basis_data(mol)
    ours = np.asarray(eri2c_matrix(aux))
    ref = mol.intor("int2c2e")
    assert np.abs(ours - ref).max() < 1e-10


@pytest.mark.pyscf
@pytest.mark.parametrize("l", [5, 6])
def test_eri3c_high_l_aux_vs_pyscf(l):
    """(μν|P) with an h/i auxiliary matches PySCF int3c2e (cart)."""
    from pyscf import df as pyscf_df
    from pyscf import gto

    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", cart=True)
    auxbasis = {"H": [[l, [1.3, 1.0]], [1, [0.9, 1.0]]]}
    auxmol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis=auxbasis, cart=True)
    basis = extract_basis_data(mol)
    aux = extract_basis_data(auxmol)
    ours = np.asarray(eri3c_matrix(basis, aux))
    ref = pyscf_df.incore.aux_e2(mol, auxmol, intor="int3c2e")
    assert np.abs(ours - ref).max() < 1e-10


def test_jkfit_loads_for_iron():
    """The full def2-universal-jkfit (with i functions) now builds."""
    aux = build_basis_data(["Fe"], np.zeros((1, 3)), "def2-universal-jkfit")
    assert int(aux.max_l) == 6


def test_fe_df_energy_matches_exact_within_ri_error():
    """DF J energy for an Fe atom (jkfit aux, i functions) vs the exact path.

    Fixed-density comparison (the minao guess density): PBE has no exact
    exchange, so the Coulomb term is RI-J only and the difference is pure RI
    fit error. Measured at 1.7 mHa (~1e-6 relative): the jkfit set is
    optimized for def2 orbital bases, not sto-3g, and this is a guess density,
    so the tolerance is looser than the sub-mHa matched-basis figure. The
    integral correctness itself is pinned by the 1e-10 PySCF oracles above.
    """
    mol = Molecule(["Fe"], np.zeros((1, 3)), "sto-3g", spin=4)
    ks_ex = KS(mol, PBE(), coulomb=exact())
    ks_df = KS(mol, PBE(), coulomb=df("def2-universal-jkfit"))
    X = canonical_orthonormalizer(ks_ex.S)
    P0 = density_from_guess(ks_ex, minao(), X)
    e_ex = float(ks_ex.total(P0))
    e_df = float(ks_df.total(P0))
    assert abs(e_df - e_ex) < 5e-3
    assert abs(e_df - e_ex) / abs(e_ex) < 5e-6
