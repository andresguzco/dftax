"""Direct variational minimization of the KS energy.

An alternative to the SCF fixed point: minimize ``E(P)`` directly over the
occupied MO coefficients. The free parameters are one coefficient matrix
``Z_σ`` (nao × nocc_σ) per spin channel; each defines the gauge-independent
projector density ``P_σ = w Z_σ (Z_σᵀ S Z_σ)⁻¹ Z_σᵀ`` (``w = 2`` for the
doubly-occupied closed-shell channel, ``1`` per spin channel), making the
energy a smooth, unconstrained function of the ``Z``s. The whole solve is
differentiable end-to-end, the property that makes it useful for learning over
the KS energy.

The projector is computed with ``solve`` rather than an eigendecomposition:
at the orthonormal stationary point ``ZᵀSZ ≈ I`` is fully degenerate, where
``eigh``'s derivative is ill-defined; the eigh path yields a gauge-dependent
gradient that on GPU (cuSolver) non-deterministically NaNs the minimization.
Mirrors :func:`dftax.ks.forces._density_from_Z`.

The optimizer is any ``optax.GradientTransformation`` (default
``optax.adam(0.3)``). First-order methods are the robust choice here: the KS
energy is invariant under occupied-orbital rotations, so its Hessian is
singular along that gauge, and line-search quasi-Newton solvers (BFGS /
nonlinear-CG) reliably stall at saddle points / excited determinants when
started from a core-Hamiltonian guess. Adam descends past them to the ground
state. (This matches differentiable-DFT practice, e.g. D4FT.) For a robust,
fast *energy* solver use :func:`~dftax.ks.scf.scf` (DIIS); use direct
minimization when end-to-end differentiability of the solve is what you need.
"""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dftax.ks.energy import KS
from dftax.ks.guess import GuessSpec, density_from_guess
from dftax.ks.scf import (
    KSResult,
    _fock_stacked,
    _total_energy,
    canonical_orthonormalizer,
)


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


def _guess_orbitals(ks: KS, guess) -> tuple[Float[Array, "nao nocc"], ...]:
    """Occupied orbitals from a guess density: one Fock build at ``P0``,
    then the per-channel aufbau eigenvectors (the standard way to turn a
    density-based guess, e.g. non-idempotent SAD, into orbitals)."""
    X = canonical_orthonormalizer(ks.S)
    P0 = density_from_guess(ks, guess, X)
    F = _fock_stacked(ks, P0)                       # (nspin, nao, nao)
    _, Cp = jnp.linalg.eigh(X.T @ F @ X)
    C = X @ Cp                                       # (nspin, nao, nmo)
    return tuple(C[s][:, :n] for s, n in enumerate(ks.nocc))


@eqx.filter_jit
def _value_and_grad(ks: KS, Zs):
    """Energy and dE/dZ per channel (ks arrays traced, not baked as constants)."""
    w = 2.0 if len(ks.nocc) == 1 else 1.0
    def e(Ys):
        return ks.total(_density_stack(Ys, ks.S, w))
    return jax.value_and_grad(e)(Zs)


def minimize(
    ks: KS,
    optimizer: optax.GradientTransformation | None = None,
    *,
    max_steps: int = 2000,
    g_tol: float = 1e-6,
    check_every: int = 10,
    Z0: Array | tuple | None = None,
    guess: GuessSpec | Array | None = None,
    verbose: bool = False,
) -> KSResult:
    """Minimize the KS energy directly over orthonormalized coefficients.

    Args:
        ks: the built :class:`~dftax.ks.energy.KS` energy functional.
        optimizer: any ``optax.GradientTransformation``; default
            ``optax.adam(0.3)``.
        max_steps: optimizer step budget.
        g_tol: stop when the global gradient norm falls below this.
        check_every: host-side convergence checks happen every this many steps
            (each check blocks on the device), so the reported ``n_iter`` can
            overshoot the true crossing by up to ``check_every - 1`` steps.
            ``verbose`` checks every step.
        Z0: optional initial coefficient guess, one ``(nao, nocc_σ)`` array
            per channel (a bare array is accepted for a closed shell); default
            is the core-Hamiltonian guess.
        guess: alternative to ``Z0``: a spec from
            :func:`~dftax.ks.guess.core` / :func:`~dftax.ks.guess.sad` /
            :func:`~dftax.ks.guess.minao` / :func:`~dftax.ks.guess.sap` or an
            explicit ``(nspin, nao, nao)`` density; the starting orbitals are
            the aufbau eigenvectors of the Fock built at that density.
        verbose: print per-step energy and gradient norm.

    Example:
        ```python
        res = minimize(ks)                             # optax.adam(0.3)
        res = minimize(ks, optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(0.1)), g_tol=1e-7)
        ```
    """
    if Z0 is not None and guess is not None:
        raise ValueError("pass either Z0 (explicit orbitals) or guess, not both.")
    if Z0 is None:
        Zs = _core_guess(ks) if guess is None else _guess_orbitals(ks, guess)
    elif isinstance(Z0, (tuple, list)):
        Zs = tuple(jnp.asarray(Z) for Z in Z0)
    else:
        Zs = (jnp.asarray(Z0),)
    if len(Zs) != len(ks.nocc):
        raise ValueError(
            f"Z0 has {len(Zs)} channel(s) but the system has {len(ks.nocc)}."
        )

    opt = optax.adam(0.3) if optimizer is None else optimizer
    state = opt.init(Zs)
    converged = False
    n_iter = max_steps
    for step in range(max_steps):
        e, g = _value_and_grad(ks, Zs)
        updates, state = opt.update(g, state, Zs)
        Zs = optax.apply_updates(Zs, updates)
        # Pulling the gradient norm to the host blocks on the device, so only
        # check at the cadence asked for; between checks the steps queue up.
        if verbose or step % check_every == check_every - 1 or step == max_steps - 1:
            gnorm = float(jnp.sqrt(sum(jnp.sum(gi ** 2) for gi in g)))
            if verbose:
                print(f"  min {step:4d}: E={float(e):.10f}  |g|={gnorm:.2e}")
            if gnorm < g_tol:
                converged = True
                n_iter = step + 1
                break

    # Pack the result with the full canonical orbital set from the converged
    # Fock (forward-only eigh, not differentiated), for parity with SCF. The
    # occupied columns span the converged density only at an exactly-converged
    # aufbau minimum; P is the authoritative density (consumers that freeze
    # the density, e.g. forces, extract the occupied space from P).
    if not converged:
        warnings.warn(
            f"Direct minimization did NOT converge in {max_steps} steps "
            f"(g_tol={g_tol}); the returned energy is unreliable. Increase "
            f"max_steps or use a smaller learning rate.",
            stacklevel=2,
        )
    w = 2.0 if len(ks.nocc) == 1 else 1.0
    P = _density_stack(Zs, ks.S, w)
    e_tot = float(_total_energy(ks, P))
    X = canonical_orthonormalizer(ks.S)
    F = _fock_stacked(ks, P)
    eps, Cp = jnp.linalg.eigh(X.T @ F @ X)             # batched over channels
    C = X @ Cp
    return KSResult(
        e_tot=e_tot,
        e_elec=e_tot - float(ks.e_nn),
        converged=converged,
        n_iter=n_iter,
        nocc=ks.nocc,
        mo_energy=eps,
        mo_coeff=C,
        P=P,
    )
