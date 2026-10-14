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
from dftax.ks import DFCoulomb, WhitenedDFCoulomb, df
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
@pytest.mark.parametrize("aux", ["sto-3g", "def2-svp"])
def test_matches_plain_df_to_machine_precision(aux):
    """A well-conditioned metric: no floor between the contraction orders."""
    plain, white = _pair(WATER, "sto-3g", PBE0(), aux=aux)
    assert isinstance(white.coulomb, WhitenedDFCoulomb)
    assert isinstance(plain.coulomb, DFCoulomb)
    _, P = _orbitals(plain)
    assert abs(float(plain.total(P) - white.total(P))) < 1e-11  # 2e-13 relative


@pytest.mark.float64
@pytest.mark.parametrize("xc, geom, spin", [
    (PBE, WATER, None), (PBE0, WATER, None), (CAMB3LYP, WATER, None),
    (PBE0, OH, 1),
])
def test_same_functional_through_every_entry_point(xc, geom, spin):
    """Energy, Fock and orbital gradient against DFCoulomb at the jkfit floor,
    and the closed-form orbital gradient against autodiff of the energy."""
    plain, white = _pair(geom, "6-31g", xc(), spin=spin)
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
