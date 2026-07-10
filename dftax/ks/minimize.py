"""Direct variational minimization of the KS energy.

An alternative to the SCF fixed point: minimize ``E(P)`` directly over the
occupied MO coefficients. The free parameters are one coefficient matrix
``Z_σ`` (nao × nocc_σ) per spin channel; each defines the gauge-independent
projector density ``P_σ = w Z_σ (Z_σᵀ S Z_σ)⁻¹ Z_σᵀ`` (``w = 2`` for the
doubly-occupied closed-shell channel, ``1`` per spin channel), making the
energy a smooth, unconstrained function of the ``Z``s. The whole solve is
differentiable end-to-end, the property that makes it useful for learning over
the KS energy. The closed-shell and spin-polarized cases run through one core;
:func:`rks_minimize` / :func:`uks_minimize` adapt the argument and result
shapes.

The projector is computed with ``solve`` rather than an eigendecomposition:
at the orthonormal stationary point ``ZᵀSZ ≈ I`` is fully degenerate, where
``eigh``'s derivative is ill-defined — the eigh path yields a gauge-dependent
gradient that on GPU (cuSolver) non-deterministically NaNs the minimization.
Mirrors :func:`dftax.ks.forces._density_from_Z`.

Optimization uses gradient descent (``optax`` Adam). First-order methods are the
robust choice here: the KS energy is invariant under occupied-orbital rotations,
so its Hessian is singular along that gauge, and line-search quasi-Newton solvers
(BFGS / nonlinear-CG) reliably stall at saddle points / excited determinants when
started from a core-Hamiltonian guess. Adam descends past them to the ground
state. (This matches differentiable-DFT practice, e.g. D4FT.) For a robust,
fast *energy* solver use :func:`~dftax.ks.scf.rks_scf` (DIIS); use direct
minimization when end-to-end differentiability of the solve is what you need.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dftax.ks.energy import KS, RKS, UKS
from dftax.ks.scf import (
    SCFResult,
    UKSResult,
    _fock,
    _fock_stacked,
    _total_energy,
    canonical_orthonormalizer,
)


def _orthonormalize(
    Z: Float[Array, "nao nocc"], S: Float[Array, "nao nao"], eps: float = 1e-10
) -> Float[Array, "nao nocc"]:
    """Löwdin-orthonormalize ``Z`` against ``S``: ``C = Z (Zᵀ S Z)^{-1/2}``."""
    M = Z.T @ S @ Z
    w, U = jnp.linalg.eigh(M)
    w = jnp.clip(w, eps)
    M_inv_sqrt = (U * (w ** -0.5)) @ U.T
    return Z @ M_inv_sqrt


def _density_stack(Zs, S, w: float) -> Float[Array, "nspin nao nao"]:
    """Spin-stacked projector densities ``P_σ = w Z_σ (Z_σᵀSZ_σ)⁻¹ Z_σᵀ``.

    Same value as the Löwdin form ``w C_σ C_σᵀ`` with ``C_σ = Z_σ(Z_σᵀSZ_σ)^{-1/2}``,
    but with the eigh-free, GPU-safe gradient (see the module docstring).
    """
    return jnp.stack(
        [w * (Z @ jnp.linalg.solve(Z.T @ S @ Z, Z.T)) for Z in Zs]
    )


def _core_guess(ks: KS) -> tuple[Float[Array, "nao nocc"], ...]:
    """Per-channel occupied core-Hamiltonian eigenvectors (S-orthonormal)."""
    X = canonical_orthonormalizer(ks.S)
    _, Cp = jnp.linalg.eigh(X.T @ ks.hcore @ X)
    C = X @ Cp
    return tuple(C[:, :n] for n in ks.nocc)


@eqx.filter_jit
def _value_and_grad(ks: KS, Zs):
    """Energy and dE/dZ per channel (ks arrays traced, not baked as constants)."""
    w = 2.0 if len(ks.nocc) == 1 else 1.0
    def e(Ys):
        return KS.total(ks, _density_stack(Ys, ks.S, w))
    return jax.value_and_grad(e)(Zs)


def _minimize(ks: KS, Zs, learning_rate, max_steps, g_tol, verbose):
    """Adam descent over the per-channel coefficient pytree."""
    opt = optax.adam(learning_rate)
    state = opt.init(Zs)
    converged = False
    n_iter = max_steps
    for step in range(max_steps):
        e, g = _value_and_grad(ks, Zs)
        updates, state = opt.update(g, state)
        Zs = optax.apply_updates(Zs, updates)
        gnorm = float(jnp.sqrt(sum(jnp.sum(gi ** 2) for gi in g)))
        if verbose:
            print(f"  min {step:4d}: E={float(e):.10f}  |g|={gnorm:.2e}")
        if gnorm < g_tol:
            converged = True
            n_iter = step + 1
            break
    return Zs, converged, n_iter


def rks_minimize(
    ks: RKS,
    *,
    learning_rate: float = 0.3,
    max_steps: int = 2000,
    g_tol: float = 1e-6,
    Z0: Float[Array, "nao nocc"] | None = None,
    verbose: bool = False,
) -> SCFResult:
    """Minimize the RKS energy directly over orthonormalized coefficients (Adam).

    Args:
        ks: the :class:`RKS` energy functional.
        learning_rate: Adam step size.
        max_steps: optimizer step budget.
        g_tol: stop when the gradient norm falls below this.
        Z0: optional initial (nao, nocc) coefficient guess; default is the
            core-Hamiltonian guess.
        verbose: print per-step energy and gradient norm.
    """
    if ks.nelec % 2 != 0:
        raise ValueError(f"RKS requires an even electron count, got {ks.nelec}.")
    Zs = _core_guess(ks) if Z0 is None else (Z0,)

    Zs, converged, n_iter = _minimize(
        ks, Zs, learning_rate, max_steps, g_tol, verbose
    )
    (Z,) = Zs
    P = _density_stack(Zs, ks.S, 2.0)[0]
    C = _orthonormalize(Z, ks.S)          # coefficients for the result (not differentiated)
    e_tot = float(_total_energy(ks, P))
    # Canonical orbital energies from the converged Fock, for parity with SCF.
    X = canonical_orthonormalizer(ks.S)
    eps = jnp.linalg.eigvalsh(X.T @ _fock(ks, P) @ X)
    return SCFResult(
        e_tot=e_tot,
        e_elec=e_tot - float(ks.e_nn),
        converged=converged,
        n_iter=n_iter,
        mo_energy=eps,
        mo_coeff=C,
        P=P,
    )


def uks_minimize(
    ks: UKS,
    *,
    learning_rate: float = 0.3,
    max_steps: int = 2000,
    g_tol: float = 1e-6,
    Z0: tuple[Float[Array, "nao na"], Float[Array, "nao nb"]] | None = None,
    verbose: bool = False,
) -> UKSResult:
    """Minimize the UKS energy directly over per-spin coefficients (Adam).

    Args mirror :func:`rks_minimize`; ``Z0`` is an optional ``(Zα, Zβ)`` initial
    guess (default: per-spin core-Hamiltonian guess).
    """
    Zs = _core_guess(ks) if Z0 is None else tuple(Z0)

    Zs, converged, n_iter = _minimize(
        ks, Zs, learning_rate, max_steps, g_tol, verbose
    )
    P = _density_stack(Zs, ks.S, 1.0)
    Ca = _orthonormalize(Zs[0], ks.S)     # coefficients for the result (not differentiated)
    Cb = _orthonormalize(Zs[1], ks.S)
    e_tot = float(KS.total(ks, P))
    X = canonical_orthonormalizer(ks.S)
    F = _fock_stacked(ks, P)
    ea = jnp.linalg.eigvalsh(X.T @ F[0] @ X)
    eb = jnp.linalg.eigvalsh(X.T @ F[1] @ X)
    return UKSResult(
        e_tot=e_tot,
        e_elec=e_tot - float(ks.e_nn),
        converged=converged,
        n_iter=n_iter,
        mo_energy=(ea, eb),
        mo_coeff=(Ca, Cb),
        P=(P[0], P[1]),
    )
