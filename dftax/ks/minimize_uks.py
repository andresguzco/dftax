"""Direct variational minimization of the UKS energy.

Open-shell counterpart of :mod:`dftax.ks.minimize`: minimize ``E(Pα, Pβ)``
directly over two coefficient matrices ``(Zα, Zβ)``, each Löwdin-orthonormalized
against the AO overlap at every evaluation. The optimizer (Adam) descends over
the ``(Zα, Zβ)`` pytree jointly. As in the restricted case, Adam is the robust
choice: the energy is invariant under per-spin occupied-orbital rotations, so
quasi-Newton solvers stall at saddles. Use :func:`dftax.ks.scf_uks.uks_scf` for
a fast energy; use this when end-to-end differentiability of the solve is needed.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dftax.ks.energy_uks import UKS
from dftax.ks.minimize import _orthonormalize
from dftax.ks.scf import canonical_orthonormalizer
from dftax.ks.scf_uks import UKSResult, _fock_uks


def _spin_densities(Za, Zb, S):
    """Per-spin densities P_σ = C_σ C_σᵀ (unit occupation) from (Zα, Zβ)."""
    Ca = _orthonormalize(Za, S)
    Cb = _orthonormalize(Zb, S)
    return Ca @ Ca.T, Cb @ Cb.T, Ca, Cb


def _core_guess_Z(ks: UKS):
    """Per-spin occupied core-Hamiltonian eigenvectors (S-orthonormal)."""
    X = canonical_orthonormalizer(ks.S)
    _, Cp = jnp.linalg.eigh(X.T @ ks.hcore @ X)
    C = X @ Cp
    return C[:, : ks.nalpha], C[:, : ks.nbeta]


@eqx.filter_jit
def _value_and_grad(ks: UKS, Za, Zb):
    """Energy and (dE/dZα, dE/dZβ)."""
    def e(A, B):
        Pa, Pb, _, _ = _spin_densities(A, B, ks.S)
        return ks.total(Pa, Pb)
    return jax.value_and_grad(e, argnums=(0, 1))(Za, Zb)


def _result(ks: UKS, Za, Zb, converged: bool, n_iter: int) -> UKSResult:
    Pa, Pb, Ca, Cb = _spin_densities(Za, Zb, ks.S)
    e_tot = float(ks.total(Pa, Pb))
    X = canonical_orthonormalizer(ks.S)
    Fa, Fb = _fock_uks(ks, Pa, Pb)
    ea = jnp.linalg.eigvalsh(X.T @ Fa @ X)
    eb = jnp.linalg.eigvalsh(X.T @ Fb @ X)
    return UKSResult(
        e_tot=e_tot,
        e_elec=e_tot - float(ks.e_nn),
        converged=converged,
        n_iter=n_iter,
        mo_energy=(ea, eb),
        mo_coeff=(Ca, Cb),
        P=(Pa, Pb),
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

    Args mirror :func:`dftax.ks.minimize.rks_minimize`; ``Z0`` is an optional
    ``(Zα, Zβ)`` initial guess (default: per-spin core-Hamiltonian guess).
    """
    Za, Zb = _core_guess_Z(ks) if Z0 is None else Z0

    opt = optax.adam(learning_rate)
    state = opt.init((Za, Zb))
    converged = False
    n_iter = max_steps
    for step in range(max_steps):
        e, (ga, gb) = _value_and_grad(ks, Za, Zb)
        updates, state = opt.update((ga, gb), state)
        Za, Zb = optax.apply_updates((Za, Zb), updates)
        gnorm = float(jnp.sqrt(jnp.sum(ga ** 2) + jnp.sum(gb ** 2)))
        if verbose:
            print(f"  umin {step:4d}: E={float(e):.10f}  |g|={gnorm:.2e}")
        if gnorm < g_tol:
            converged = True
            n_iter = step + 1
            break

    return _result(ks, Za, Zb, converged, n_iter)
