"""Kohn-Sham SCF: autodiff Fock + canonical orthonormalization + DIIS.

The Fock matrices are obtained by automatic differentiation of the electronic
energy, ``F_σ = sym(∂E/∂P_σ)``, exact for any XC functional and free of a
hand-coded XC potential matrix. Convergence is accelerated with Pulay DIIS on
the orthonormal-basis commutator ``Xᵀ(FPS − SPF)X``.

One solver serves both shell structures: the density is spin-stacked
``(nspin, nao, nao)`` with per-channel occupations from ``ks.nocc`` (a doubly
occupied single channel for a closed shell, unit-occupation α/β channels for a
spin-polarized system), and DIIS runs on the channel-stacked Fock/error
super-matrices, so the restricted solve is literally the ``nspin=1`` case of
the unrestricted one. The whole self-consistency loop runs on device in a
single ``lax.while_loop`` (DIIS history kept in a fixed-size circular buffer),
so there are no per-iter host round-trips and the entire solve compiles once.

:func:`scf` wraps the solver and packs the stacked outputs into a
:class:`KSResult`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from dftax.ks.energy import KS
from dftax.ks.guess import GuessSpec, density_from_guess


@eqx.filter_jit
def _total_energy(ks: KS, P: Float[Array, "nspin nao nao"]) -> Array:
    """Total KS energy E(P) (filter-jit so ks arrays are traced, not baked)."""
    return ks.total(P)


@eqx.filter_jit
def _fock_stacked(ks: KS, P: Float[Array, "nspin nao nao"]) -> Array:
    """Per-channel KS Fock matrices ``F_σ = sym(∂E/∂P_σ)`` by autodiff."""
    g = jax.grad(lambda Q: ks.electronic(Q))(P)
    return 0.5 * (g + g.transpose(0, 2, 1))


@dataclass
class KSResult:
    """Outcome of a KS solve (SCF or direct minimization), spin-stacked.

    All orbital/density fields carry a leading ``nspin`` axis (``nspin =
    len(nocc)``): a closed shell is ``nspin = 1`` (``P[0]`` doubly occupied),
    a spin-polarized system is ``nspin = 2`` (α, β). Batched solves return the
    distinct :class:`~dftax.ks.batched.BatchedResult` instead.
    """

    e_tot: float
    e_elec: float
    converged: bool
    n_iter: int
    nocc: tuple[int, ...]
    mo_energy: Float[Array, "nspin nmo"]
    mo_coeff: Float[Array, "nspin nao nmo"]
    P: Float[Array, "nspin nao nao"]


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


def _occupations(nocc: tuple[int, ...], nmo: int) -> Float[Array, "nspin nmo"]:
    """Static per-channel occupation numbers: 2 for the closed-shell channel,
    1 per spin channel (aufbau fill of the lowest ``nocc[σ]`` orbitals)."""
    w = 2.0 if len(nocc) == 1 else 1.0
    return jnp.stack([w * (jnp.arange(nmo) < n) for n in nocc])


def _diis_extrapolate(dF, dErr, count, m):
    """Pulay DIIS over a fixed-size circular buffer.

    ``dF``/``dErr`` are (m, nspin·nao, nao) / (m, nspin·nmo, nmo) channel-stacked
    super-matrices; only the first ``count`` slots are filled (``count`` traced).
    Unfilled slots are masked to the identity so they drop out of the augmented
    system; a tiny ridge on the filled block keeps it solvable as the errors
    vanish near convergence.
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
def _scf_solve(ks: KS, X, P0, max_iter, e_tol, d_tol, m, verbose, level_shift):
    """On-device SCF: autodiff Fock + DIIS in a single while_loop (spin-stacked).

    ``P0`` is the initial spin-stacked density (see :mod:`dftax.ks.guess`);
    the loop body starts by building the Fock from it.

    ``level_shift`` (Saunders-Hillier) adds ``b·(S − S C_occ C_occᵀ S)`` per channel
    to the Fock before diagonalization, raising the virtual orbital energies by
    ``b`` while leaving the occupied block untouched, widening the HOMO-LUMO gap to
    damp oscillation for near-degenerate cases. The occupied subspace is unchanged
    at the fixed point, so the converged density (and energy) is identical to
    ``level_shift=0``.

    Returns channel-stacked ``(e, P, C, eps, converged, n_iter)``.
    """
    S = ks.S
    nao = S.shape[0]
    nmo = X.shape[1]
    nspin = len(ks.nocc)
    # Statically slice the occupied block: nocc is static, and contracting the
    # full nmo set against a mostly-zero occupation vector costs nmo/nocc times
    # the flops for the same result (XLA does not eliminate constant-zero
    # columns from a dense dot).
    nmax = max(ks.nocc)
    f = _occupations(ks.nocc, nmax)         # (nspin, nmax) occupation numbers
    # Occupied projector scale for the level shift: P_σ = (1/inv_w) C_occ C_occᵀ.
    inv_w = 0.5 if nspin == 1 else 1.0

    def make_density(F):                     # F: (nspin, nao, nao)
        eps, Cp = jnp.linalg.eigh(X.T @ F @ X)          # batched over channels
        C = X @ Cp                                       # (nspin, nao, nmo)
        Co = C[:, :, :nmax]                              # static occupied slice
        P = jnp.einsum("smi,si,sni->smn", Co, f, Co)     # aufbau fill
        return P, C, eps

    e0 = ks.total(P0)
    # C/eps placeholders: the body always runs at least one iteration
    # (``converged`` starts False), which overwrites them.
    C0 = jnp.zeros((nspin, nao, nmo))
    eps0 = jnp.zeros((nspin, nmo))
    # Channel-stacked DIIS buffers. The Fock is nao×nao per channel, but the
    # commutator error err = Xᵀ(FPS−SPF)X is nmo×nmo, and nmo = X.shape[1] < nao
    # whenever the canonical orthonormalizer drops linearly-dependent columns, so
    # the two buffers have distinct shapes.
    dF0 = jnp.zeros((m, nspin * nao, nao))
    dErr0 = jnp.zeros((m, nspin * nmo, nmo))
    # state: (it, P, C, eps, e_prev, derr, converged, dF, dErr)
    state0 = (0, P0, C0, eps0, e0, jnp.inf, jnp.array(False), dF0, dErr0)

    def cond(st):
        return (st[0] < max_iter) & jnp.logical_not(st[6])

    def body(st):
        it, P, C, eps, e_prev, _, _, dF, dErr = st
        g = jax.grad(lambda Q: ks.electronic(Q))(P)
        F = 0.5 * (g + g.transpose(0, 2, 1))
        err = X.T @ (F @ P @ S - S @ P @ F) @ X          # (nspin, nmo, nmo)
        derr = jnp.linalg.norm(err)

        slot = it % m
        dF = dF.at[slot].set(F.reshape(nspin * nao, nao))
        dErr = dErr.at[slot].set(err.reshape(nspin * nmo, nmo))
        count = jnp.minimum(it + 1, m)
        F_ext = _diis_extrapolate(dF, dErr, count, m).reshape(nspin, nao, nao)

        F_ls = F_ext + level_shift * (S - inv_w * (S @ P @ S))   # raise virtuals
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


def scf(
    ks: KS,
    *,
    max_iter: int = 128,
    e_tol: float = 1e-8,
    d_tol: float = 1e-6,
    diis_space: int = 8,
    lindep_thresh: float = 1e-7,
    level_shift: float = 0.0,
    guess: GuessSpec | Array | None = None,
    verbose: bool = False,
) -> KSResult:
    """Run KS SCF to self-consistency (restricted and spin-polarized alike).

    Args:
        ks: the built :class:`~dftax.ks.energy.KS` energy functional.
        max_iter: maximum SCF iterations.
        e_tol: convergence threshold on the total-energy change (Ha).
        d_tol: convergence threshold on the DIIS commutator norm.
        diis_space: DIIS history depth (fixed buffer size).
        lindep_thresh: overlap-eigenvalue cutoff for canonical orthonormalization.
        level_shift: Saunders-Hillier virtual level shift (Ha).
        guess: initial density, a spec from :func:`~dftax.ks.guess.core` /
            :func:`~dftax.ks.guess.sad` / :func:`~dftax.ks.guess.minao` /
            :func:`~dftax.ks.guess.sap`, or an explicit ``(nspin, nao, nao)``
            density array (warm restart). ``None`` is the core-Hamiltonian
            guess.
        verbose: print per-iteration energy / error (via jax.debug.print).

    Example:
        ```python
        ks = KS(mol, PBE())
        res = scf(ks, e_tol=1e-9)
        res = scf(ks, guess=sad())               # fewer iterations
        res.e_tot, res.converged, res.P[0]       # P is spin-stacked
        ```
    """
    X = canonical_orthonormalizer(ks.S, lindep_thresh)
    P0 = density_from_guess(ks, guess, X)
    # Tolerances ride along as traced arrays: under filter_jit a Python scalar
    # is a static argument, so retrying with level_shift or a tighter e_tol
    # would otherwise recompile the whole solve. diis_space (buffer shape) and
    # verbose (Python branch) must stay static.
    e_tot, P, C, eps, converged, n_iter = _scf_solve(
        ks, X, P0, jnp.asarray(max_iter), jnp.asarray(e_tol), jnp.asarray(d_tol),
        diis_space, verbose, jnp.asarray(level_shift)
    )
    result = KSResult(
        e_tot=float(e_tot),
        e_elec=float(e_tot) - float(ks.e_nn) - float(ks.e_disp),
        converged=bool(converged),
        n_iter=int(n_iter),
        nocc=ks.nocc,
        mo_energy=eps,
        mo_coeff=C,
        P=P,
    )
    if not result.converged:
        warnings.warn(
            f"SCF did NOT converge in {result.n_iter} iterations "
            f"(e_tol={e_tol}, d_tol={d_tol}); the returned energy is unreliable. "
            f"Increase max_iter or try level_shift>0.",
            stacklevel=2,
        )
    return result
