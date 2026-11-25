import numpy as np
import pytest

from dftax import KS, Molecule, df
from dftax.basis.loader import build_basis_data
from dftax.energy.xc import PBE, PBE0
from dftax.integrals.bucketed import class_jobs, eri3c_square
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
@pytest.mark.slow
def test_bucketed_tensor_is_the_bucketed_build():
    basis, aux = _bases(WATER)
    plan0 = plan_eri3c(basis, aux, tol=0.0)
    np.testing.assert_allclose(eri3c_square(basis, aux, plan0),
                               eri3c_matrix_bucketed(basis, aux), rtol=0,
                               atol=1e-13)


def test_class_programs_are_shared_across_molecules():
    def keys(atoms):
        basis, aux = _bases(atoms)
        return {k for _, k in class_jobs(basis, aux,
                                         plan_eri3c(basis, aux, tol=0.0))}

    assert keys(WATER) <= keys(METHANOL)


@pytest.mark.float64
def test_pair_integrals_are_the_bucketed_build():
    from dftax.integrals.bucketed import pairs_host
    from dftax.integrals.eri3c_bucketed import (
        eri2c_matrix_bucketed, nuclear_attraction_bucketed,
        overlap_kinetic_bucketed, plan_pairs,
    )

    basis, aux = _bases(WATER)
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
@pytest.mark.slow
def test_ks_with_shared_programs_matches(monkeypatch):
    import jax.numpy as jnp

    mol = Molecule.from_xyz(WATER, "6-31g*")
    kw = dict(coulomb=df(chunk=None, packed=True))
    monkeypatch.setenv("DFTAX_BUCKETED", "0")
    ref = KS(mol, PBE0(), **kw)
    monkeypatch.setenv("DFTAX_BUCKETED", "1")
    ks = KS(mol, PBE0(), **kw)
    for a, b in ((ks.S, ref.S), (ks.hcore, ref.hcore)):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-11)
    n = ref.nocc[0]
    s, U = jnp.linalg.eigh(ref.S)
    C = (U / jnp.sqrt(s)) @ U.T
    P = (2.0 * C[:, :n] @ C[:, :n].T)[None]
    assert abs(float(ks.total(P) - ref.total(P))) < 1e-9


@pytest.mark.float64
def test_hybrid_switches_to_the_fused_build(monkeypatch):
    import dftax.ks.energy as energy_mod

    monkeypatch.setenv("DFTAX_BUCKETED", "1")
    monkeypatch.setattr(energy_mod, "_FUSED_BUILDS", {})
    monkeypatch.setattr(energy_mod, "_SEEN", set())
    shared = []
    orig = energy_mod._build_integrals_shared
    monkeypatch.setattr(energy_mod, "_build_integrals_shared",
                        lambda *a, **k: (shared.append(1), orig(*a, **k))[1])
    mol = Molecule.from_xyz(WATER, "sto-3g")
    first = KS(mol, PBE())
    assert len(shared) == 1 and not energy_mod._FUSED_BUILDS
    KS(mol, PBE())
    assert len(shared) == 2 and len(energy_mod._FUSED_BUILDS) == 1
    (fut,) = energy_mod._FUSED_BUILDS.values()
    fut.result()
    second = KS(mol, PBE())
    assert len(shared) == 2
    np.testing.assert_allclose(second.S, first.S, rtol=0, atol=1e-11)
    np.testing.assert_allclose(second.hcore, first.hcore, rtol=0, atol=1e-11)
    def gram(ks):
        B = np.asarray(ks.coulomb.B)
        B = B.reshape(-1, B.shape[-1])
        return B @ B.T

    np.testing.assert_allclose(gram(second), gram(first), rtol=0, atol=1e-10)


@pytest.mark.float64
def test_packed_tensor_is_the_packed_build():
    from dftax.integrals.bucketed import eri3c_host_packed
    from dftax.integrals.eri3c_bucketed import _shell_pair_keep, eri3c_packed

    basis, aux = _bases(WATER, spherical=False)
    keep = _shell_pair_keep(basis, 1e-3)
    plan0 = plan_eri3c(basis, aux, keep_pairs=keep, tol=0.0)
    np.testing.assert_allclose(
        eri3c_host_packed(basis, aux, plan0),
        eri3c_packed(basis, aux, plan=plan_eri3c(basis, aux, keep_pairs=keep)),
        rtol=0, atol=1e-13)
