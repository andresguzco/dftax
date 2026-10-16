"""The whitened density-fitting term, ``B = Lᵀ T`` with ``L Lᵀ = V⁺``.

:class:`~dftax.ks.terms.WhitenedDFCoulomb` is :class:`DFCoulomb` with the
metric folded into the tensor, so the two are the same functional: machine
precision apart with a well-conditioned auxiliary metric, and within the
documented density-fitting floor with the overcomplete jkfit set (whose kept
band amplifies the reordered rounding). Whitening is the default only where
the build is concrete; traced builds keep the differentiable metric inverse.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax import KS, Molecule, becke, forces, gdm, scf
from dftax.energy.xc import PBE, PBE0, CAMB3LYP
from dftax.ks import DFCoulomb, PackedDFCoulomb, WhitenedDFCoulomb, df
from dftax.ks.batched_gdm import gdm_batched
from dftax.ks.terms import _metric_factor, _metric_pinv, _whiten

WATER = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
OH = "O 0 0 0; H 0 0 0.97"


def _pair(geom, basis, xc, aux="def2-universal-jkfit", spin=None):
    mol = Molecule.from_xyz(geom, basis, spin=spin or 0)
    kw = dict(grid=becke(35, 50), spin=spin)
    plain = KS(mol, xc, coulomb=df(aux, chunk=None, whiten=False), **kw)
    white = KS(mol, xc, coulomb=df(aux, chunk=None), **kw)
    return plain, white


def _orbitals(ks, seed=0):
    """S-orthonormal occupied orbitals, one block per spin channel."""
    rng = np.random.default_rng(seed)
    nao = ks.S.shape[0]
    s, U = jnp.linalg.eigh(ks.S)
    C = ((U / jnp.sqrt(s)) @ U.T) @ jnp.linalg.qr(
        jnp.asarray(rng.normal(size=(nao, nao))))[0]
    w = 2.0 if len(ks.nocc) == 1 else 1.0
    Cs = tuple(C[:, :n] for n in ks.nocc)
    return Cs, jnp.stack([w * c @ c.T for c in Cs])


@pytest.mark.float64
def test_factor_and_in_place_whitening():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(9, 6))
    V = jnp.asarray(A @ A.T)                       # rank 6: three dropped
    L = _metric_factor(V)
    np.testing.assert_allclose(L @ L.T, _metric_pinv(V), atol=1e-12)
    T = jnp.asarray(rng.normal(size=(6, 6, 9)))
    np.testing.assert_array_equal(jax.jit(_whiten)(T, L),
                                  jnp.einsum("mnP,Px->mnx", T, L))


@pytest.mark.float64
@pytest.mark.parametrize("aux", ["def2-svp"])
def test_matches_plain_df_to_machine_precision(aux):
    """A well-conditioned metric: no floor between the contraction orders."""
    plain, white = _pair(WATER, "sto-3g", PBE0(), aux=aux)
    assert isinstance(white.coulomb, WhitenedDFCoulomb)
    assert isinstance(plain.coulomb, DFCoulomb)
    _, P = _orbitals(plain)
    assert abs(float(plain.total(P) - white.total(P))) < 1e-11  # 2e-13 relative


@pytest.mark.float64
@pytest.mark.parametrize("xc, geom, spin", [
    (CAMB3LYP, WATER, None), (PBE0, OH, 1),
])
def test_same_functional_through_every_entry_point(xc, geom, spin):
    """Energy, Fock and orbital gradient against DFCoulomb at the jkfit floor,
    and the closed-form orbital gradient against autodiff of the energy."""
    plain, white = _pair(geom, "sto-3g", xc(), spin=spin)
    Cs, P = _orbitals(plain)
    assert abs(float(plain.total(P) - white.total(P))) < 1e-8
    (e0, F0), (e1, F1) = plain.energy_and_fock(P), white.energy_and_fock(P)
    assert abs(float(e0 - e1)) < 1e-8
    np.testing.assert_allclose(F1, F0, atol=1e-8)
    (e0, g0), (e1, g1) = (plain.energy_and_grad_orbitals(Cs),
                          white.energy_and_grad_orbitals(Cs))
    assert abs(float(e0 - e1)) < 1e-8
    for a, b in zip(g1, g0):
        np.testing.assert_allclose(a, b, atol=1e-8)
    w = 2.0 if len(Cs) == 1 else 1.0
    gad = jax.grad(lambda cs: white.total(
        jnp.stack([w * c @ c.T for c in cs])))(Cs)
    for a, b in zip(g1, gad):
        np.testing.assert_allclose(a, b, atol=1e-11)


@pytest.mark.float64
def test_policy():
    """Whitened by default when concrete; DFCoulomb under a trace; explicit
    requests that cannot apply raise."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    assert isinstance(KS(mol, PBE()).coulomb, WhitenedDFCoulomb)
    assert isinstance(KS(mol, PBE(), coulomb=df(whiten=False)).coulomb,
                      DFCoulomb)
    with pytest.raises(ValueError, match="whiten=True"):
        df(whiten=True, chunk=64)
    import equinox as eqx

    from dftax.basis.loader import build_basis_data
    from dftax.grid import becke_grid, points
    from dftax.ks.energy import System

    sym, x0 = mol.symbols, mol.atom_coords()
    basis_t, idx = build_basis_data(sym, x0, mol.basis,
                                    return_atom_index=True)
    aux_t, aidx = build_basis_data(sym, x0, "def2-universal-jkfit",
                                   return_atom_index=True)

    def term_type(coords):              # the forces rebuild, in miniature
        basis = eqx.tree_at(lambda b: b.centers, basis_t, coords[idx])
        aux = eqx.tree_at(lambda b: b.centers, aux_t, coords[aidx])
        gc, gw = becke_grid(sym, coords, 20, 26)
        ks = KS(System(basis=basis, coords=coords,
                       charges=jnp.asarray(mol.atom_charges()),
                       nelec=mol.nelectron), PBE(),
                grid=points(gc, gw), coulomb=df(aux, chunk=None))
        return isinstance(ks.coulomb, WhitenedDFCoulomb)

    seen = []
    jax.jit(lambda c: seen.append(term_type(c)) or c)(jnp.asarray(x0))
    assert seen == [False]


@pytest.mark.float64
@pytest.mark.slow
def test_solvers_and_forces_on_the_whitened_term():
    """scf, gdm and batched gdm on the whitened term converge to the plain
    term's energy; forces (a traced, un-whitened rebuild) agree too."""
    plain, white = _pair(WATER, "6-31g", PBE0())
    r0, r1 = scf(plain), scf(white)
    assert r0.converged and r1.converged
    assert abs(r0.e_tot - r1.e_tot) < 1e-8
    g = gdm(white)
    assert g.converged and abs(g.e_tot - r0.e_tot) < 1e-8
    res = gdm_batched([white, plain])     # grouped by term type
    assert bool(np.all(res.converged))
    np.testing.assert_allclose(res.e_tot, r0.e_tot, atol=1e-8)
    mol = Molecule.from_xyz(WATER, "6-31g")
    f0 = forces(mol, PBE0(), r0, grid=becke(35, 50),
                coulomb=df(chunk=None, whiten=False))
    f1 = forces(mol, PBE0(), r1, grid=becke(35, 50), coulomb=df(chunk=None))
    np.testing.assert_allclose(np.asarray(f1), np.asarray(f0), atol=1e-6)


@pytest.mark.float64
@pytest.mark.parametrize("xc, geom, spin", [
    (CAMB3LYP, WATER, None), (PBE0, OH, 1),
])
def test_packed_is_the_square_whitened_term(xc, geom, spin):
    """Packing changes the storage, not the contractions: energy, Fock and
    orbital gradient agree with the square whitened term to rounding, the
    Fock is exactly symmetric, and the closed-form gradient is autodiff's."""
    mol = Molecule.from_xyz(geom, "sto-3g", spin=spin or 0)
    kw = dict(grid=becke(35, 50), spin=spin)
    sq = KS(mol, xc(), coulomb=df(chunk=None), **kw)
    pk = KS(mol, xc(), coulomb=df(chunk=None, packed=True), **kw)
    assert isinstance(pk.coulomb, PackedDFCoulomb)
    assert pk.coulomb.B.size < 0.7 * sq.coulomb.B.size
    Cs, P = _orbitals(sq)
    (e0, F0), (e1, F1) = sq.energy_and_fock(P), pk.energy_and_fock(P)
    assert abs(float(e0 - e1)) < 1e-12
    np.testing.assert_allclose(F1, F0, atol=1e-12)
    np.testing.assert_array_equal(F1, F1.transpose(0, 2, 1))
    (e0, g0), (e1, g1) = (sq.energy_and_grad_orbitals(Cs),
                          pk.energy_and_grad_orbitals(Cs))
    assert abs(float(e0 - e1)) < 1e-12
    w = 2.0 if len(Cs) == 1 else 1.0
    gad = jax.grad(lambda cs: pk.total(
        jnp.stack([w * c @ c.T for c in cs])))(Cs)
    for a, b, c in zip(g1, g0, gad):
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(a, c, atol=1e-11)


def test_packed_needs_the_whitened_materialized_build():
    with pytest.raises(ValueError, match="packed=True"):
        df(packed=True, whiten=False)
    with pytest.raises(ValueError, match="packed=True"):
        df(packed=True, chunk=64)


@pytest.mark.float64
@pytest.mark.slow
def test_solvers_on_the_packed_term():
    sq = KS(Molecule.from_xyz(WATER, "6-31g"), PBE0(), grid=becke(35, 50))
    pk = KS(Molecule.from_xyz(WATER, "6-31g"), PBE0(), grid=becke(35, 50),
            coulomb=df(chunk=None, packed=True))
    r0, r1, g = scf(sq), scf(pk), gdm(pk)
    assert r0.converged and r1.converged and g.converged
    assert abs(r0.e_tot - r1.e_tot) < 1e-9
    assert abs(g.e_tot - r0.e_tot) < 1e-8


def test_df_screen_auto_is_size_gated():
    """df()'s default pair screen is off below the grid's 50-atom gate and
    1e-10 relative from it up; explicit values pass through."""
    from dftax.ks.terms import (
        DF_SCREEN_AUTO_CUTOFF, DF_SCREEN_AUTO_MIN_ATOMS, resolve_df_screen,
    )

    assert df().screen == "auto"
    assert resolve_df_screen("auto", DF_SCREEN_AUTO_MIN_ATOMS - 1) is None
    assert (resolve_df_screen("auto", DF_SCREEN_AUTO_MIN_ATOMS)
            == DF_SCREEN_AUTO_CUTOFF)
    assert resolve_df_screen(None, 500) is None
    assert resolve_df_screen(1e-8, 3) == 1e-8


@pytest.mark.float64
def test_packing_is_size_and_budget_gated(monkeypatch):
    """packed="auto": square below the screening gate; packed when only the
    packed tensor fits the budget (or screening is on)."""
    import dftax.ks.energy as energy_mod

    mol = Molecule.from_xyz(WATER, "sto-3g")
    sq = KS(mol, PBE()).coulomb
    assert isinstance(sq, WhitenedDFCoulomb)
    assert isinstance(KS(mol, PBE(), coulomb=df(screen=1e-12)).coulomb,
                      PackedDFCoulomb)
    nao, naux = sq.B.shape[0], sq.B.shape[2]
    budget = (nao * (nao + 1) // 2) * naux + 1      # packed fits, square not
    monkeypatch.setattr(energy_mod, "_df_materialize_budget", lambda: budget)
    assert isinstance(KS(mol, PBE()).coulomb, PackedDFCoulomb)
