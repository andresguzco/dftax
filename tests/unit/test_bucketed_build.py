"""Shared class programs for the 3-center tensor (dftax.integrals.bucketed).

The bucketed build must write exactly the tensor the per-molecule build
writes, and its class programs must depend on the class and the batch sizes
only, so a second molecule with the same shell classes reuses them.
"""

import numpy as np
import pytest

from dftax import KS, Molecule
from dftax.basis.loader import build_basis_data
from dftax.energy.xc import PBE
from dftax.integrals.bucketed import class_program, eri3c_square
from dftax.integrals.eri3c_bucketed import eri3c_matrix_bucketed, plan_eri3c

WATER = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
METHANOL = ("C -0.047 0.665 0; O -0.047 -0.758 0; H -1.086 0.976 0; "
            "H 0.437 1.084 0.887; H 0.437 1.084 -0.887; H 0.861 -1.086 0")


def _bases(atoms, spherical=True):
    mol = Molecule.from_xyz(atoms, "6-31g*")
    return (build_basis_data(mol.symbols, mol.atom_coords(), "6-31g*",
                             spherical=spherical),
            build_basis_data(mol.symbols, mol.atom_coords(), "def2-svp",
                             spherical=True))


@pytest.mark.float64
@pytest.mark.parametrize("spherical", [True, False])
def test_bucketed_tensor_is_the_bucketed_build(spherical):
    basis, aux = _bases(WATER, spherical)
    plan0 = plan_eri3c(basis, aux, tol=0.0)
    # rounding only: the nested vmap orders the contractions differently
    # (measured 8.9e-16 on values up to ~8)
    np.testing.assert_allclose(eri3c_square(basis, aux, plan0),
                               eri3c_matrix_bucketed(basis, aux), rtol=0,
                               atol=1e-13)


@pytest.mark.float64
def test_class_programs_are_shared_across_molecules():
    """Water then methanol: every class methanol shares with water reuses
    water's program (fixed tiles: the program depends on the class alone)."""
    class_program.cache_clear()
    basis, aux = _bases(WATER)
    eri3c_square(basis, aux, plan_eri3c(basis, aux, tol=0.0))
    n_water = class_program.cache_info().currsize
    basis, aux = _bases(METHANOL)
    plan = plan_eri3c(basis, aux, tol=0.0)
    eri3c_square(basis, aux, plan)
    info = class_program.cache_info()
    # every water class is an O/H class methanol also has
    assert info.hits >= n_water


@pytest.mark.float64
@pytest.mark.parametrize("spherical", [True, False])
def test_pair_integrals_are_the_bucketed_build(spherical):
    from dftax.integrals.bucketed import pairs_host
    from dftax.integrals.eri3c_bucketed import (
        eri2c_matrix_bucketed, nuclear_attraction_bucketed,
        overlap_kinetic_bucketed, plan_pairs,
    )

    basis, aux = _bases(WATER, spherical)
    mol = Molecule.from_xyz(WATER, "6-31g*")
    coords, charges = mol.atom_coords(), mol.atom_charges()
    pp = plan_pairs(basis, tol=0.0)
    S, T = pairs_host("st", basis, pp)
    S0, T0 = overlap_kinetic_bucketed(basis)
    (V,) = pairs_host("nuc", basis, pp, atom_coords=coords,
                      atom_charges=charges)
    V0 = nuclear_attraction_bucketed(basis, coords, charges)
    (M,) = pairs_host("eri2c", aux, plan_pairs(aux, tol=0.0))
    M0 = eri2c_matrix_bucketed(aux)
    for a, b in ((S, S0), (T, T0), (V, V0), (M, M0)):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.float64
def test_ks_with_shared_programs_matches(monkeypatch):
    """The whole eager build from shared class programs: the same KS, to
    rounding, as the per-molecule build (energy and Fock at a fixed
    density)."""
    import jax.numpy as jnp

    mol = Molecule.from_xyz(WATER, "6-31g*")
    ref = KS(mol, PBE())
    monkeypatch.setenv("DFTAX_BUCKETED", "1")
    ks = KS(mol, PBE())
    for a, b in ((ks.S, ref.S), (ks.hcore, ref.hcore)):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-11)
    n = ref.nocc[0]
    s, U = jnp.linalg.eigh(ref.S)
    C = (U / jnp.sqrt(s)) @ U.T
    P = (2.0 * C[:, :n] @ C[:, :n].T)[None]
    assert abs(float(ks.total(P) - ref.total(P))) < 1e-9
