"""Restricted Kohn-Sham SCF: autodiff Fock + canonical orthonormalization + DIIS.

The Fock matrix is obtained by automatic differentiation of the electronic
energy, ``F = sym(∂E/∂P)``, exact for any XC functional and free of a
hand-coded XC potential matrix. Convergence is accelerated with Pulay DIIS on
the orthonormal-basis commutator ``Xᵀ(FPS − SPF)X``.

The whole self-consistency loop runs on device in a single ``lax.while_loop``
(DIIS history kept in a fixed-size circular buffer), so there are no per-iter
host round-trips and the entire solve compiles once.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from dftax.ks.energy import RKS


def _sym(m: Float[Array, "n n"]) -> Float[Array, "n n"]:
    return 0.5 * (m + m.T)


@eqx.filter_jit
def _total_energy(ks: RKS, P: Float[Array, "nao nao"]) -> Array:
    """Total KS energy E(P) (filter-jit so ks arrays are traced, not baked)."""
    return ks.total(P)


@eqx.filter_jit
def _fock(ks: RKS, P: Float[Array, "nao nao"]) -> Array:
    """KS Fock matrix F = sym(dE/dP) by automatic differentiation."""
    return _sym(jax.grad(lambda Q: ks.electronic(Q))(P))


@dataclass
class SCFResult:
    """Outcome of an RKS SCF run."""

    e_tot: float
    e_elec: float
    converged: bool
    n_iter: int
    mo_energy: Float[Array, "nmo"]
    mo_coeff: Float[Array, "nao nmo"]
    P: Float[Array, "nao nao"]


def canonical_orthonormalizer(
    S: Float[Array, "nao nao"], thresh: float = 1e-7
) -> Float[Array, "nao nmo"]:
    """Canonical orthonormalizer ``X = U s^{-1/2}`` with ``Xᵀ S X = I``.

    Eigenvectors whose overlap eigenvalue falls below ``thresh`` are dropped to
    keep the transformed problem well conditioned for (near-)linearly-dependent
    bases. Run eagerly (outside the jitted loop) so the kept-column count, and
    hence X's shape, is concrete.
    """
    s, U = jnp.linalg.eigh(S)
    keep = s > thresh
    s_keep = s[keep]
    U_keep = U[:, keep]
    return U_keep * (1.0 / jnp.sqrt(s_keep))[None, :]


def _diis_extrapolate(dF, dErr, count, m):
    """Pulay DIIS over a fixed-size circular buffer.

    ``dF``/``dErr`` are (m, nao, nao); only the first ``count`` slots are filled
    (``count`` traced). Unfilled slots are masked to the identity so they drop
    out of the augmented system; a tiny ridge on the filled block keeps it
    solvable as the errors vanish near convergence.
    """
    errs = dErr.reshape(m, -1)
    B = errs @ errs.T                                  # (m, m)
    valid = jnp.arange(m) < count
    vv = valid[:, None] & valid[None, :]
    B = jnp.where(vv, B, jnp.eye(m))
    B = B + 1e-10 * jnp.diag(valid.astype(B.dtype))    # ridge filled slots

    A = jnp.zeros((m + 1, m + 1), dtype=B.dtype)
    A = A.at[:m, :m].set(B)
    border = jnp.where(valid, -1.0, 0.0)
    A = A.at[:m, m].set(border)
    A = A.at[m, :m].set(border)
    rhs = jnp.zeros(m + 1, dtype=B.dtype).at[m].set(-1.0)

    c = jnp.linalg.solve(A, rhs)[:m]                   # (m,)
    return jnp.einsum("i,iab->ab", c, dF)


@eqx.filter_jit
def _scf_solve(ks, X, nocc, max_iter, e_tol, d_tol, m, verbose, level_shift):
    """On-device SCF: autodiff Fock + DIIS in a single while_loop.

    ``level_shift`` (Saunders-Hillier) adds ``b·(S − ½SPS)`` to the Fock before
    diagonalization, raising the virtual orbital energies by ``b`` while leaving the
    occupied block untouched, widening the HOMO-LUMO gap to damp oscillation for
    near-degenerate cases. The occupied subspace is unchanged at the fixed point, so
    the converged density (and energy) is identical to ``level_shift=0``.
    """
    S = ks.S
    nao = S.shape[0]
    nmo = X.shape[1]

    def make_density(F):
        eps, Cp = jnp.linalg.eigh(X.T @ F @ X)
        C = X @ Cp
        Cocc = C[:, :nocc]
        return 2.0 * Cocc @ Cocc.T, C, eps

    P0, C0, eps0 = make_density(ks.hcore)              # core-Hamiltonian guess
    e0 = ks.total(P0)
    # Separate DIIS buffers: the Fock is nao×nao, but the commutator error
    # err = Xᵀ(FPS−SPF)X is nmo×nmo, and nmo = X.shape[1] < nao whenever the
    # canonical orthonormalizer drops linearly-dependent columns. A single shared
    # buffer shape-mismatches on such bases (mirrors the α⊕β split in
    # scf_uks._scf_solve_u).
    dF0 = jnp.zeros((m, nao, nao))
    dErr0 = jnp.zeros((m, nmo, nmo))
    # state: (it, P, C, eps, e_prev, derr, converged, dF, dErr)
    state0 = (0, P0, C0, eps0, e0, jnp.inf, jnp.array(False), dF0, dErr0)

    def cond(st):
        return (st[0] < max_iter) & jnp.logical_not(st[6])

    def body(st):
        it, P, C, eps, e_prev, _, _, dF, dErr = st
        F = _sym(jax.grad(lambda Q: ks.electronic(Q))(P))
        err = X.T @ (F @ P @ S - S @ P @ F) @ X
        derr = jnp.linalg.norm(err)

        slot = it % m
        dF = dF.at[slot].set(F)
        dErr = dErr.at[slot].set(err)
        count = jnp.minimum(it + 1, m)
        F_ext = _diis_extrapolate(dF, dErr, count, m)

        F_ls = F_ext + level_shift * (S - 0.5 * (S @ P @ S))   # raise virtuals (P = current)
        P, C, eps = make_density(F_ls)
        e = ks.total(P)
        de = e - e_prev
        converged = (jnp.abs(de) < e_tol) & (derr < d_tol)
        if verbose:
            jax.debug.print(
                "  scf {it}: E={e:.10f} dE={de:+.2e} |[F,P]|={derr:.2e}",
                it=it, e=e, de=de, derr=derr,
            )
        return (it + 1, P, C, eps, e, derr, converged, dF, dErr)

    it, P, C, eps, e_prev, _, converged, _, _ = lax.while_loop(cond, body, state0)
    return e_prev, P, C, eps, converged, it


def rks_scf(
    ks: RKS,
    *,
    max_iter: int = 128,
    e_tol: float = 1e-8,
    d_tol: float = 1e-6,
    diis_space: int = 8,
    lindep_thresh: float = 1e-7,
    level_shift: float = 0.0,
    verbose: bool = False,
) -> SCFResult:
    """Run closed-shell RKS SCF to self-consistency.

    Args:
        ks: the precomputed :class:`RKS` energy functional.
        max_iter: maximum SCF iterations.
        e_tol: convergence threshold on the total-energy change (Ha).
        d_tol: convergence threshold on the DIIS commutator norm.
        diis_space: DIIS history depth (fixed buffer size).
        lindep_thresh: overlap-eigenvalue cutoff for canonical orthonormalization.
        verbose: print per-iteration energy / error (via jax.debug.print).
    """
    if ks.nelec % 2 != 0:
        raise ValueError(f"RKS requires an even electron count, got {ks.nelec}.")
    nocc = ks.nelec // 2
    X = canonical_orthonormalizer(ks.S, lindep_thresh)

    e_tot, P, C, eps, converged, n_iter = _scf_solve(
        ks, X, nocc, max_iter, e_tol, d_tol, diis_space, verbose, level_shift
    )
    result = SCFResult(
        e_tot=float(e_tot),
        e_elec=float(e_tot) - float(ks.e_nn),
        converged=bool(converged),
        n_iter=int(n_iter),
        mo_energy=eps,
        mo_coeff=C,
        P=P,
    )
    if not result.converged:
        warnings.warn(
            f"RKS SCF did NOT converge in {result.n_iter} iterations "
            f"(e_tol={e_tol}, d_tol={d_tol}); the returned energy is unreliable. "
            f"Increase max_iter or try level_shift>0.",
            stacklevel=2,
        )
    return result
