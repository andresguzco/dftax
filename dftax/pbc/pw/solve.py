"""The self-consistent field of the plane-wave engine.

Each iteration takes the effective potential of the input density, improves
the lowest ``nbands`` eigenvectors of every ``k`` by a few steps of a block
eigensolver, occupies them, and mixes the output density with the earlier
ones (Pulay, with Kerker's preconditioner on the residual). The energy is
that of the orbitals themselves (:func:`~dftax.pbc.pw.energy.energy_terms`),
not a Harris-Foulkes estimate, so at convergence it is variational.

The eigensolver is LOBPCG with the Teter-Payne-Allan preconditioner: the
trial space of each step is the current vectors, their preconditioned
residuals and the previous step's direction, made orthonormal explicitly
(two passes of projection and QR), so every step has one shape and a plain
Hermitian Rayleigh-Ritz problem.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from dftax.pbc.lattice import gvectors, volume
from dftax.pbc.pw.basis import PlaneWaves, monkhorst_pack
from dftax.pbc.pw.energy import (
    PWModel, apply_h, build_model, density, energy_terms, initial_density,
    potential, projector_matrices, wavevectors,
)


def _teter(kin, X):
    """Teter-Payne-Allan preconditioner for the bands ``X`` ``(npw, nb)``."""
    ekin = jnp.sum(kin[:, None] * (X * X.conj()).real, axis=0)
    x = kin[:, None] / (1.5 * jnp.maximum(ekin[None, :], 1e-3))
    num = 27.0 + 18.0 * x + 12.0 * x ** 2 + 8.0 * x ** 3
    return num / (num + 16.0 * x ** 4)


def _orthonormal_against(V, Q, mask):
    """``V`` made orthonormal and orthogonal to the orthonormal ``Q``: two
    passes of projection and QR, so even columns that were nearly in the
    span of ``Q`` (converged bands) come out as clean directions."""
    for _ in range(2):
        V = (V - Q @ (Q.conj().T @ V)) * mask[:, None]
        V, _ = jnp.linalg.qr(V)
    return V


def lobpcg(X, kin, vr, fft_idx, mask, mesh, P, D, n_iter: int):
    """``n_iter`` LOBPCG steps on one sphere. Returns the Ritz vectors
    ``(npw, nb)`` (orthonormal), their values and residual norms.

    The trial space of a step, ``[X, W, Pd]``, is made orthonormal
    explicitly and the Hamiltonian applied to its new directions, so the
    Rayleigh-Ritz problem is a plain Hermitian one at any conditioning.
    """
    nb = X.shape[1]
    h = lambda V: apply_h(V, kin, vr, fft_idx, mask, mesh, P, D)
    HX = h(X)
    Pd = jnp.zeros_like(X)

    def step(carry, _):
        X, HX, Pd = carry
        lam = jnp.sum((X.conj() * HX).real, axis=0)
        R = HX - X * lam[None, :]
        W = _orthonormal_against(_teter(kin, X) * R, X, mask)
        Pd = _orthonormal_against(Pd, jnp.concatenate([X, W], axis=1), mask)
        S = jnp.concatenate([X, W, Pd], axis=1)
        HS = jnp.concatenate([HX, h(W), h(Pd)], axis=1)
        A = S.conj().T @ HS
        _, U = jnp.linalg.eigh(0.5 * (A + A.conj().T))
        U = U[:, :nb]
        # the step's direction: the new vectors' parts outside the old ones
        return (S @ U, HS @ U, S[:, nb:] @ U[nb:]), None

    (X, HX, _), _ = jax.lax.scan(step, (X, HX, Pd), None, length=n_iter)
    A = X.conj().T @ HX
    e, U = jnp.linalg.eigh(0.5 * (A + A.conj().T))
    X, HX = X @ U, HX @ U
    res = jnp.linalg.norm(HX - X * e[None, :], axis=0)
    return X, e, res


@eqx.filter_jit
def _eigensolve(model: PWModel, C, vr, coords, a, n_iter: int):
    """LOBPCG on every k: ``(nk, npw, nb)`` vectors, values, residuals."""
    q = wavevectors(model, a)
    kin = 0.5 * jnp.sum(q * q, axis=-1) * model.mask
    P = projector_matrices(model, coords, a)

    def one(args):
        Ck, kk, idx, mk, Pk = args
        return lobpcg(Ck, kk, vr, idx, mk, model.mesh, Pk, model.D, n_iter)

    Pstack = P if P is not None else jnp.zeros((model.kfrac.shape[0], 0, 1))
    if P is None:
        def one(args):                                           # noqa: F811
            Ck, kk, idx, mk, _ = args
            return lobpcg(Ck, kk, vr, idx, mk, model.mesh, None, None, n_iter)
    return jax.lax.map(one, (C, kin, model.fft_idx, model.mask, Pstack))


@eqx.filter_jit
def _potential(model, rho, coords, a):
    return potential(model, rho, coords, a)


@eqx.filter_jit
def _density(model, C, occ, a):
    return density(model, C, occ, a)


@eqx.filter_jit
def _energy(model, C, occ, coords, a):
    return energy_terms(model, C, occ, coords, a)


class _Pulay:
    """Pulay (DIIS) mixing of densities with Kerker's preconditioner,
    ``G^2 / (G^2 + q0^2)``, on the residual (None: no preconditioner)."""

    def __init__(self, a, mesh, beta=0.5, history=8, q0=1.5):
        self.beta, self.history, self.mesh = beta, history, tuple(mesh)
        self.rho, self.res = [], []
        if q0 is None:
            self.kerker = None
        else:
            G2 = jnp.sum(gvectors(jnp.asarray(a), mesh) ** 2, axis=-1)
            self.kerker = (G2 / (G2 + q0 * q0)).reshape(self.mesh)

    def __call__(self, rho_in, rho_out):
        self.rho.append(rho_in)
        self.res.append(rho_out - rho_in)
        self.rho, self.res = self.rho[-self.history:], self.res[-self.history:]
        n = len(self.res)
        R = jnp.stack(self.res)
        A = np.asarray(R @ R.T)
        M = np.zeros((n + 1, n + 1))
        M[:n, :n], M[n, :n], M[:n, n] = A, 1.0, 1.0
        rhs = np.zeros(n + 1)
        rhs[n] = 1.0
        c = np.linalg.lstsq(M, rhs, rcond=None)[0][:n]
        rho = sum(ci * r for ci, r in zip(c, self.rho))
        res = sum(ci * r for ci, r in zip(c, self.res))
        if self.kerker is not None:
            res = jnp.fft.ifftn(self.kerker * jnp.fft.fftn(
                res.reshape(self.mesh))).real.reshape(-1)
        return rho + self.beta * res


class PWResult(NamedTuple):
    """A converged (or not) plane-wave calculation."""

    e_tot: float
    terms: dict
    C: jax.Array               # (nk, npw, nbands) orbital coefficients
    occ: jax.Array             # (nk, nbands) occupations
    eps: jax.Array             # (nk, nbands) band energies
    rho: jax.Array             # (N,) density on the mesh
    converged: bool
    n_iter: int
    model: PWModel
    ts: float = 0.0            # entropy term σS; e_tot is then E - TS
    mu: float | None = None    # Fermi level under smearing


def scf_pw(cell, xc, pw: PlaneWaves, *, kpts=None,
           nbands: int | None = None, e_tol: float = 1e-9,
           smearing=None,
           d_tol: float = 1e-6, max_iter: int = 100, mixing: float = 0.5,
           kerker: float | None = 1.5, n_iter: int = 4, seed: int = 0,
           model: PWModel | None = None,
           verbose: bool = False) -> PWResult:
    """Self-consistent field of a cell in plane waves (closed shell, an
    insulator: the lowest ``nocc`` bands of every ``k`` doubly occupied; with
    ``smearing``, Fermi-Dirac occupations with one Fermi level, and
    ``e_tot`` the free energy ``E - TS``).

    Args:
        cell: the :class:`~dftax.pbc.Cell` (its Gaussian basis is unused).
        xc: the functional (LDA or GGA).
        pw: the basis, :func:`~dftax.pbc.pw.planewaves`.
        kpts: ``(kfrac, weights)``, as :func:`~dftax.pbc.kmesh` returns them
            (default: the Gamma point alone).
        smearing: a :func:`~dftax.ks.scf.fermi` spec, or its width
            (Hartree) as a float; ``None`` is the insulator fill.
        nbands: bands per ``k`` (default: occupied plus a tenth, at least
            4; with smearing a fifth, at least 8).
        e_tol, d_tol: convergence of the energy (Ha) and of the density
            residual (electrons, ``∫|ρ_out - ρ_in|``).
        mixing, kerker: Pulay step and Kerker wave vector (None: plain).
        n_iter: eigensolver steps per SCF iteration.
        model: the plane-wave set of an earlier result (`result.model`) to
            keep, at this cell's atoms and lattice: finite differences at a
            fixed basis, which the stress is the derivative of.
    """
    smearing = getattr(smearing, "sigma", smearing) or None
    if model is None:
        kfrac, weights = kpts if kpts is not None else monkhorst_pack((1, 1, 1))
        if nbands is None and smearing:
            nocc = int(cell.nelectron) // 2
            nbands = nocc + max(8, nocc // 5)
        model = build_model(cell, xc, pw, kfrac, weights, nbands)
    coords = jnp.asarray(cell.atom_coords())
    a = jnp.asarray(cell.a)
    nk, npw, nb = model.miller.shape[0], model.miller.shape[1], model.nbands
    N = int(np.prod(model.mesh))
    dv = float(volume(a)) / N

    # random start, damped at high kinetic energy, orthonormal per k
    q = np.asarray(wavevectors(model, a))
    kin = 0.5 * np.sum(q * q, axis=-1)
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(nk, npw, nb)) + 1j * rng.normal(size=(nk, npw, nb)))
    X *= (np.asarray(model.mask) / (1.0 + kin) ** 2)[:, :, None]
    C = jnp.asarray(np.stack([np.linalg.qr(x)[0] for x in X]))
    occ = jnp.zeros((nk, nb)).at[:, :model.nocc].set(2.0)
    ts, mu = 0.0, None

    rho_in = initial_density(model, coords, a)
    mixer = _Pulay(cell.a, model.mesh, beta=mixing, q0=kerker)
    e_old, converged, it = np.inf, False, 0
    for it in range(1, max_iter + 1):
        t0 = time.perf_counter()
        vr = _potential(model, rho_in, coords, a)
        C, eps, res = _eigensolve(model, C, vr, coords, a,
                                  4 * n_iter if it == 1 else n_iter)
        if smearing:
            from dftax.pbc.kpoints import _fermi

            f, ts, mu = _fermi(eps, np.asarray(model.weights),
                               2 * model.nocc, smearing)
            occ = jnp.asarray(f)
        rho_out = _density(model, C, occ, a)
        terms = _energy(model, C, occ, coords, a)
        e = float(sum(terms.values())) - ts
        drho = float(jnp.sum(jnp.abs(rho_out - rho_in))) * dv
        if verbose:
            print(f"  pw scf {it:3d}: E {e:.10f}  dE {e - e_old:+.2e}  "
                  f"|drho| {drho:.2e}  max residual "
                  f"{float(jnp.max(res[:, :model.nocc])):.1e}  "
                  f"({time.perf_counter() - t0:.2f} s)", flush=True)
        if abs(e - e_old) < e_tol and drho < d_tol:
            converged = True
            break
        e_old = e
        rho_in = mixer(rho_in, rho_out)
    return PWResult(
        e_tot=e, terms={k: float(v) for k, v in terms.items()}, C=C, occ=occ,
        ts=float(ts), mu=mu,
        eps=eps, rho=rho_out, converged=converged, n_iter=it, model=model)
