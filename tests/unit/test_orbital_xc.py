"""The energy and its orbital derivative through the occupied orbitals.

:meth:`KS.energy_and_grad_orbitals` is what the direct minimizer evaluates:
``E`` and ``dE/dC_sigma`` for ``P_sigma = w C_sigma C_sigma^T``, the XC part
taken through the orbitals on the materialized grid rather than through the
density matrix. It has to be the same function as ``energy_and_fock`` read
through the chain rule, for every functional rung and spin case, and the
commutator ``gdm`` reads off it has to be the one ``scf`` measures.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from dftax import KS, Molecule, becke, df, gdm, scf
from dftax.energy.xc import PBE, R2SCAN, functional
from dftax.ks.scf import canonical_orthonormalizer

WATER = "O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50"
OH = "O 0 0 0; H 0 0 0.97"
NH2 = "N 0 0 0; H 1.01 0 0; H -0.26 0.98 0"


def _orbitals(ks, seed=0):
    """S-orthonormal occupied orbitals per channel (a fixed random draw)."""
    X = canonical_orthonormalizer(ks.S)
    rng = np.random.default_rng(seed)
    Cs = []
    for n in ks.nocc:
        Q = np.linalg.qr(rng.standard_normal((X.shape[1], n)))[0]
        Cs.append(X @ jnp.asarray(Q))
    return tuple(Cs), X


@pytest.mark.float64
@pytest.mark.parametrize("xc, atoms, spin, chunk", [
    (PBE, WATER, 0, None), (R2SCAN, OH, 1, 64),
])
def test_orbital_gradient_is_the_chain_rule(xc, atoms, spin, chunk):
    """Materialized grid (chunk=None, the orbital route) and streamed grid
    (the density-matrix default) both match 2 w F C from energy_and_fock."""
    mol = Molecule.from_xyz(atoms, "sto-3g", spin=spin)
    ks = KS(mol, xc(), grid=becke(35, 50, chunk=chunk),
            spin=spin if spin else None)
    Cs, _ = _orbitals(ks)
    w = 2.0 if len(Cs) == 1 else 1.0
    P = jnp.stack([w * C @ C.T for C in Cs])
    e_ref, F = ks.energy_and_fock(P)
    e, grads = ks.energy_and_grad_orbitals(Cs)
    assert abs(float(e - e_ref)) < 1e-10
    for s, (C, G) in enumerate(zip(Cs, grads)):
        np.testing.assert_allclose(G, 2.0 * w * F[s] @ C, rtol=0, atol=1e-9)


@pytest.mark.float64
def test_commutator_from_the_orbital_gradient():
    """||X^T (F P S - S P F) X|| = (1/sqrt 2) ||X^T (G - S C (C^T G))|| with
    G = 2 w F C: gdm's convergence test needs no Fock."""
    mol = Molecule.from_xyz(OH, "6-31g*", spin=1)
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=1)
    Cs, X = _orbitals(ks, seed=3)
    P = jnp.stack([C @ C.T for C in Cs])
    _, F = ks.energy_and_fock(P)
    S = ks.S
    ref = jnp.linalg.norm(X.T @ (F @ P @ S - S @ P @ F) @ X)
    sq = 0.0
    for s, C in enumerate(Cs):
        G = 2.0 * F[s] @ C
        r = X.T @ (G - S @ (C @ (C.T @ G)))
        sq += 0.5 * float(jnp.sum(r * r))
    assert abs(np.sqrt(sq) - float(ref)) < 1e-12 * max(1.0, float(ref))


@pytest.mark.float64
@pytest.mark.slow
# NH2, not OH: OH's degenerate pi shell is a knife edge for the DIIS reference
@pytest.mark.parametrize("atoms,spin", [(WATER, 0), (NH2, 1)])
def test_gdm_orbital_route_matches_fock_route(atoms, spin):
    mol = Molecule.from_xyz(atoms, "6-31g*", spin=spin)
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=spin if spin else None)
    r0 = scf(ks)
    r1 = gdm(ks, orbital_xc=True)
    r2 = gdm(ks, orbital_xc=False)
    assert r0.converged and r1.converged and r2.converged
    assert abs(r1.e_tot - r0.e_tot) < 1e-7
    assert abs(r2.e_tot - r0.e_tot) < 1e-7


@pytest.mark.float64
@pytest.mark.parametrize("name, atoms, spin", [
    ("cam-b3lyp", WATER, 0), ("pbe0", OH, 1),
])
def test_hybrid_orbital_route_is_the_chain_rule(name, atoms, spin):
    """The DF exchange from the orbitals (no orbital recovery from P) against
    energy_and_fock, global and range-separated."""
    mol = Molecule.from_xyz(atoms, "sto-3g", spin=spin)
    ks = KS(mol, functional(name), grid=becke(35, 50),
            coulomb=df(chunk=None), spin=spin if spin else None)
    Cs, _ = _orbitals(ks, seed=5)
    w = 2.0 if len(Cs) == 1 else 1.0
    P = jnp.stack([w * C @ C.T for C in Cs])
    e_ref, F = ks.energy_and_fock(P)
    e, grads = ks.energy_and_grad_orbitals(Cs)
    assert abs(float(e - e_ref)) < 1e-10
    for s, (C, G) in enumerate(zip(Cs, grads)):
        np.testing.assert_allclose(G, 2.0 * w * F[s] @ C, rtol=0, atol=1e-9)
