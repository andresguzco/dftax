"""Implicit differentiation of the SCF fixed point (CPHF / coupled-perturbed KS).

Energy gradients (forces, ``∂E/∂Z``) need no implicit diff: at the variational
minimum ``∂E/∂C = 0``, so the envelope theorem gives them from the explicit geometry
derivative (see :mod:`dftax.ks.forces`). What *does* need the orbital response is the
derivative of the *converged density* w.r.t. a parameter, hence first-order response
properties such as the polarizability ``α = dμ/dE``.

:func:`implicit_density` is the converged density ``P*`` as a ``custom_vjp`` function of
the assembled :class:`~dftax.ks.energy.KS`. The forward runs the ordinary SCF under
``stop_gradient``; the backward solves the response (CPHF) equation
``(I − ∂g/∂P)ᵀ w = P̄`` matrix-free; the single-SCF-step Jacobian ``∂g/∂P`` comes from
``jax.vjp`` of one step, reusing the autodiff Fock, with the eigendecomposition's
unstable derivative replaced by the stable occ-virt projector response below. Because
the backward differentiates the *assembled* functional, all Pulay / basis-derivative
terms are supplied automatically by the engine's own autodiff, with no hand-coded integral
derivatives. Backward memory is independent of the SCF iteration count.

Used now for the analytic polarizability (``polarizability(..., method="analytic")``),
where the perturbation is a one-electron field operator that never touches the grid.
The analytic *geometry Hessian* is parked (DEV NOTE, 2026-06-29): the implicit density
response ``dP*/dR`` is well-behaved (a forward CPHF solve), and the partial-assembly route
(first-order response + a fixed-density second derivative) sidesteps the second-order
``eigh`` backend bugs (CPU XNNPACK / GPU cuSolver). What remains is a NaN in the *second
geometric derivative of the energy at fixed density*. Ruled out so far: the Becke
grid-weights (freezing the grid still NaNs) and the XC kernel (both the dense and streamed
XC paths are already double-``where``-guarded at ``rho > 1e-10``, so ``eps_xc''`` is bounded
~1e16, not singular). So the culprit is most likely a **non-XC** second derivative, one of
the integral primitives (nuclear attraction / Coulomb / kinetic via the Boys / McMurchie-
Davidson recurrences), surfacing on water/O but not H2, i.e. a non-twice-differentiable op
(``sqrt``/``where`` kink/``clip``) somewhere in those kernels. NOT YET ISOLATED per term.
Once that primitive is made twice-differentiable, the partial assembly completes the
analytic Hessian. Until then the finite-difference Hessian (:mod:`dftax.ks.properties`,
first derivatives of the analytic forces; water/sto-3g freqs match PySCF to <5 cm⁻¹)
is the validated default. See ``project_engine_publish_roadmap`` memory for the full trail.

Assumes a well-conditioned basis (square Löwdin orthonormalizer) and a finite
HOMO-LUMO gap, the standard ground-state response setting. The CPHF solve is
matrix-free GMRES; its relative residual is checked after the solve and a warning
is emitted if it has not converged (it would otherwise return a wrong gradient
silently). Conditioning here is dominated by the Fock-response coupling rather
than the bare orbital gaps (uncoupled, the operator is ≈ I), so a simple gap
preconditioner does not help; a proper preconditioner is future work. Raise
``maxiter`` or fall back to ``method='fd'`` if the residual warning fires.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from dftax.ks.energy import KS


def _sym(A):
    return 0.5 * (A + A.T)


def _warn_cphf_residual(resid, tol: float = 1e-6):
    """Host-side warning when the CPHF/GMRES solve has not converged.

    Called via ``jax.debug.callback`` with the concrete relative residual: GMRES
    returns its best iterate regardless of convergence, so without this check a
    failed response solve silently yields a wrong implicit-diff gradient.
    """
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
    """Closed-shell density ``P̃ = 2 Σ_i u_i u_iᵀ`` in the orthonormal basis from the
    orthonormal Fock ``Ft`` (``u`` = eigenvectors). ``nocc`` is static."""
    _, U = jnp.linalg.eigh(_sym(Ft))
    Uo = U[:, :nocc]
    return 2.0 * Uo @ Uo.T


@_proj_from_fock.defjvp
def _proj_jvp(primals, tangents):
    Ft, nocc = primals
    dFt, _ = tangents
    eps, U = jnp.linalg.eigh(_sym(Ft))
    P = 2.0 * U[:, :nocc] @ U[:, :nocc].T
    # First-order projector response: δP̃ = 2 U (T + Tᵀ) Uᵀ with T on the virt-occ
    # block only, T_ai = (Uᵀ δFt U)_ai / (ε_i − ε_a); denominators are occ-virt gaps,
    # so degeneracies *within* occ or virt never appear (the eigh-derivative blow-up).
    g = U.T @ _sym(dFt) @ U
    n = Ft.shape[0]
    idx = jnp.arange(n)
    ov = (idx[:, None] >= nocc) & (idx[None, :] < nocc)   # rows virt, cols occ
    denom = eps[None, :] - eps[:, None]                    # [p,q] = ε_q − ε_p
    T = jnp.where(ov, g / jnp.where(ov, denom, 1.0), 0.0)
    dP = 2.0 * (U @ (T + T.T) @ U.T)
    return P, dP


def _scf_step(P, ks, X, nocc):
    """One SCF step ``P → 2·(occupied projector of F(P))`` in the AO basis, with the
    stable projector derivative. ``X`` is the (fixed) Löwdin orthonormalizer.
    ``P`` is the single closed-shell density (implicit diff is restricted-only)."""
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

    # (I − ∂g/∂P)ᵀ w = P̄, solved matrix-free; ∂g/∂P from vjp of one SCF step.
    _, vjp_P = jax.vjp(lambda Q: _scf_step(Q, ks, X, nocc), P)
    def lhs(w):
        return w - vjp_P(w)[0]
    w, _ = gmres(lhs, Pbar, tol=1e-10, atol=0.0, maxiter=400, restart=40)

    # GMRES returns its best iterate even if it never converged, which would make
    # this a silently-wrong gradient. Check the true relative residual and warn.
    resid = jnp.linalg.norm(lhs(w) - Pbar) / (jnp.linalg.norm(Pbar) + 1e-30)
    jax.debug.callback(_warn_cphf_residual, resid)

    # cotangent on ks: (∂g/∂ks)ᵀ w  (explicit-ks part of the step at fixed P).
    _, vjp_ks = jax.vjp(lambda kk: _scf_step(P, kk, _lowdin(kk.S), nocc), ks)
    ksbar = vjp_ks(w)[0]
    return (ksbar,)


implicit_density.defvjp(_impl_fwd, _impl_bwd)
