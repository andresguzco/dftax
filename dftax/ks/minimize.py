"""Direct variational minimization of the RKS energy.

An alternative to the SCF fixed point: minimize ``E(P)`` directly over the
occupied MO coefficients. The free parameter is a matrix ``Z`` (nao x nocc) that
is Löwdin-orthonormalized against the AO overlap at every evaluation
(``C = Z (Zᵀ S Z)^{-1/2}`` so ``Cᵀ S C = I``), making the energy a smooth,
unconstrained function of ``Z``. The whole solve is differentiable end-to-end,
the property that makes it useful for learning over the KS energy.

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

from dftax.ks.energy import RKS
from dftax.ks.scf import SCFResult, canonical_orthonormalizer, _fock, _total_energy


def _orthonormalize(
    Z: Float[Array, "nao nocc"], S: Float[Array, "nao nao"], eps: float = 1e-10
) -> Float[Array, "nao nocc"]:
    """Löwdin-orthonormalize ``Z`` against ``S``: ``C = Z (Zᵀ S Z)^{-1/2}``."""
    M = Z.T @ S @ Z
    w, U = jnp.linalg.eigh(M)
    w = jnp.clip(w, eps)
    M_inv_sqrt = (U * (w ** -0.5)) @ U.T
    return Z @ M_inv_sqrt


def _density(Z: Float[Array, "nao nocc"], S: Float[Array, "nao nao"]):
    C = _orthonormalize(Z, S)
    return 2.0 * C @ C.T, C


def _core_guess_Z(ks: RKS) -> Float[Array, "nao nocc"]:
    """Occupied core-Hamiltonian eigenvectors (already S-orthonormal)."""
    X = canonical_orthonormalizer(ks.S)
    _, Cp = jnp.linalg.eigh(X.T @ ks.hcore @ X)
    nocc = ks.nelec // 2
    return X @ Cp[:, :nocc]


@eqx.filter_jit
def _value_and_grad(ks: RKS, Z: Float[Array, "nao nocc"]):
    """Energy and dE/dZ (ks arrays traced, not baked as constants)."""
    return jax.value_and_grad(lambda Y: ks.total(_density(Y, ks.S)[0]))(Z)


def _result(ks: RKS, Z: Array, converged: bool, n_iter: int) -> SCFResult:
    P, C = _density(Z, ks.S)
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
    if Z0 is None:
        Z0 = _core_guess_Z(ks)

    opt = optax.adam(learning_rate)
    state = opt.init(Z0)
    Z = Z0
    converged = False
    n_iter = max_steps
    for step in range(max_steps):
        e, g = _value_and_grad(ks, Z)
        updates, state = opt.update(g, state)
        Z = optax.apply_updates(Z, updates)
        gnorm = float(jnp.linalg.norm(g))
        if verbose:
            print(f"  min {step:4d}: E={float(e):.10f}  |g|={gnorm:.2e}")
        if gnorm < g_tol:
            converged = True
            n_iter = step + 1
            break

    return _result(ks, Z, converged, n_iter)
