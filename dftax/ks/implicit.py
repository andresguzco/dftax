from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from dftax.ks.energy import KS


def _sym(A):
    return 0.5 * (A + A.T)


def _warn_cphf_residual(resid, tol: float = 1e-6):
    if float(resid) > tol:
        warnings.warn(
            f"dftax: the CPHF / implicit-diff response solve did not converge "
            f"(relative residual {float(resid):.2e} > {tol:g}); the resulting "
            f"gradient (e.g. analytic polarizability) may be inaccurate, usually from a "
            f"small HOMO-LUMO gap or an ill-conditioned basis. Consider the "
            f"finite-difference path (method='fd') or raising maxiter.",
            stacklevel=2,
        )


def _lowdin(S, eps: float = 1e-9):
    s, U = jnp.linalg.eigh(S)
    return (U * (1.0 / jnp.sqrt(jnp.clip(s, eps)))) @ U.T


@jax.custom_jvp
def _proj_from_fock(Ft, nocc):
    _, U = jnp.linalg.eigh(_sym(Ft))
    Uo = U[:, :nocc]
    return 2.0 * Uo @ Uo.T


@_proj_from_fock.defjvp
def _proj_jvp(primals, tangents):
    Ft, nocc = primals
    dFt, _ = tangents
    eps, U = jnp.linalg.eigh(_sym(Ft))
    P = 2.0 * U[:, :nocc] @ U[:, :nocc].T
    g = U.T @ _sym(dFt) @ U
    n = Ft.shape[0]
    idx = jnp.arange(n)
    ov = (idx[:, None] >= nocc) & (idx[None, :] < nocc)
    denom = eps[None, :] - eps[:, None]
    T = jnp.where(ov, g / jnp.where(ov, denom, 1.0), 0.0)
    dP = 2.0 * (U @ (T + T.T) @ U.T)
    return P, dP


def _scf_step(P, ks, X, nocc):
    F = _sym(jax.grad(lambda Q: ks.electronic(Q[None]))(P))
    Pt = _proj_from_fock(X.T @ F @ X, nocc)
    return X @ Pt @ X.T


@jax.custom_vjp
def implicit_density(ks: KS):
    """Converged closed-shell density ``P*`` for the assembled functional ``ks``,
    differentiable w.r.t. ``ks`` (and hence anything it was assembled from) by implicit
    differentiation of the SCF fixed point. Restricted (closed-shell) only."""
    from dftax.ks.guess import density_from_guess
    from dftax.ks.scf import _scf_solve, canonical_orthonormalizer
    if len(ks.nocc) != 1:
        raise NotImplementedError("implicit_density supports closed shells only.")
    X = canonical_orthonormalizer(ks.S)
    P0 = density_from_guess(ks, None, X)
    _, P, *_ = _scf_solve(ks, X, P0, 128, 1e-10, 1e-8, 8, False, 0.0)
    return P[0]


def _impl_fwd(ks):
    P = implicit_density(ks)
    return P, (ks, P)


def _impl_bwd(res, Pbar):
    ks, P = res
    nocc = ks.nocc[0]
    X = _lowdin(ks.S)
    P = jax.lax.stop_gradient(P)

    _, vjp_P = jax.vjp(lambda Q: _scf_step(Q, ks, X, nocc), P)
    def lhs(w):
        return w - vjp_P(w)[0]
    w, _ = gmres(lhs, Pbar, tol=1e-10, atol=0.0, maxiter=400, restart=40)

    resid = jnp.linalg.norm(lhs(w) - Pbar) / (jnp.linalg.norm(Pbar) + 1e-30)
    jax.debug.callback(_warn_cphf_residual, resid)

    _, vjp_ks = jax.vjp(lambda kk: _scf_step(P, kk, _lowdin(kk.S), nocc), ks)
    ksbar = vjp_ks(w)[0]
    return (ksbar,)


implicit_density.defvjp(_impl_fwd, _impl_bwd)
