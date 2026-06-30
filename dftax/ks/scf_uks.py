"""Unrestricted Kohn-Sham SCF: per-spin autodiff Fock + DIIS.

Open-shell counterpart of :mod:`dftax.ks.scf`. Each spin gets its own Fock
matrix ``F_σ = sym(∂E/∂P_σ)`` (automatic differentiation of the open-shell
electronic energy), its own aufbau fill, and its own canonical-orthonormal
diagonalization. DIIS runs on the combined α⊕β commutator error: the two spin
matrices are stacked into a single ``(2·nao, nao)`` super-matrix so the existing
fixed-buffer extrapolator (:func:`dftax.ks.scf._diis_extrapolate`) is reused
verbatim. The whole solve runs on device in one ``lax.while_loop``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from dftax.ks.energy_uks import UKS, _spin_counts
from dftax.ks.scf import _sym, _diis_extrapolate, canonical_orthonormalizer


@dataclass
class UKSResult:
    """Outcome of a UKS SCF run (per-spin MOs and densities)."""

    e_tot: float
    e_elec: float
    converged: bool
    n_iter: int
    mo_energy: tuple[Float[Array, "nmo"], Float[Array, "nmo"]]
    mo_coeff: tuple[Float[Array, "nao nmo"], Float[Array, "nao nmo"]]
    P: tuple[Float[Array, "nao nao"], Float[Array, "nao nao"]]


@eqx.filter_jit
def _fock_uks(ks: UKS, Pa: Float[Array, "nao nao"], Pb: Float[Array, "nao nao"]):
    """Per-spin KS Fock matrices F_σ = sym(∂E/∂P_σ) by autodiff."""
    ga, gb = jax.grad(lambda A, B: ks.electronic(A, B), argnums=(0, 1))(Pa, Pb)
    return _sym(ga), _sym(gb)


@eqx.filter_jit
def _scf_solve_u(ks, X, max_iter, e_tol, d_tol, m, verbose, level_shift):
    """On-device UKS SCF: per-spin autodiff Fock + combined DIIS in a while_loop.

    ``level_shift`` adds ``b·(S − S P_σ S)`` per spin before diagonalization (raises
    the virtual energies by ``b``, leaving the occupied block fixed) to damp
    *oscillatory* SCF; the fixed point is unchanged, so the converged density/energy
    match ``level_shift=0`` (validated in ``test_level_shift``). Note it only widens
    the gap; it does not lower a convergence floor set by grid/functional noise
    (e.g. the spin-GGA commutator floor on OH/PBE), so such cases are unaffected.
    """
    S = ks.S
    nao = S.shape[0]
    na, nb = ks.nalpha, ks.nbeta

    def make_density(F, nocc):
        eps, Cp = jnp.linalg.eigh(X.T @ F @ X)
        C = X @ Cp
        Cocc = C[:, :nocc]
        return Cocc @ Cocc.T, C, eps                 # unit occupation (per spin)

    Pa0, Ca0, ea0 = make_density(ks.hcore, na)        # core-Hamiltonian guess
    Pb0, Cb0, eb0 = make_density(ks.hcore, nb)
    e0 = ks.total(Pa0, Pb0)
    z_f = jnp.zeros((m, 2 * nao, nao))                # stacked α⊕β Fock buffer
    z_e = jnp.zeros((m, 2 * X.shape[1], X.shape[1]))  # stacked α⊕β error buffer
    # state: (it, Pa, Pb, Ca, Cb, ea, eb, e_prev, converged, dF, dErr)
    state0 = (0, Pa0, Pb0, Ca0, Cb0, ea0, eb0, e0, jnp.array(False), z_f, z_e)

    def cond(st):
        return (st[0] < max_iter) & jnp.logical_not(st[8])

    def body(st):
        it, Pa, Pb, Ca, Cb, ea, eb, e_prev, _, dF, dErr = st
        Fa, Fb = _fock_uks(ks, Pa, Pb)
        err_a = X.T @ (Fa @ Pa @ S - S @ Pa @ Fa) @ X
        err_b = X.T @ (Fb @ Pb @ S - S @ Pb @ Fb) @ X
        F_stack = jnp.concatenate([Fa, Fb], axis=0)            # (2nao, nao)
        err_stack = jnp.concatenate([err_a, err_b], axis=0)    # (2nmo, nmo)
        derr = jnp.linalg.norm(err_stack)

        slot = it % m
        dF = dF.at[slot].set(F_stack)
        dErr = dErr.at[slot].set(err_stack)
        count = jnp.minimum(it + 1, m)
        F_ext = _diis_extrapolate(dF, dErr, count, m)          # (2nao, nao)

        Pa, Ca, ea = make_density(F_ext[:nao] + level_shift * (S - S @ Pa @ S), na)
        Pb, Cb, eb = make_density(F_ext[nao:] + level_shift * (S - S @ Pb @ S), nb)
        e = ks.total(Pa, Pb)
        de = e - e_prev
        converged = (jnp.abs(de) < e_tol) & (derr < d_tol)
        if verbose:
            jax.debug.print(
                "  uscf {it}: E={e:.10f} dE={de:+.2e} |[F,P]|={derr:.2e}",
                it=it, e=e, de=de, derr=derr,
            )
        return (it + 1, Pa, Pb, Ca, Cb, ea, eb, e, converged, dF, dErr)

    out = lax.while_loop(cond, body, state0)
    it, Pa, Pb, Ca, Cb, ea, eb, e_prev, converged, _, _ = out
    return e_prev, Pa, Pb, Ca, Cb, ea, eb, converged, it


def uks_scf(
    ks: UKS,
    *,
    max_iter: int = 128,
    e_tol: float = 1e-8,
    d_tol: float = 1e-6,
    diis_space: int = 8,
    lindep_thresh: float = 1e-7,
    level_shift: float = 0.0,
    verbose: bool = False,
) -> UKSResult:
    """Run open-shell UKS SCF to self-consistency.

    Args mirror :func:`dftax.ks.scf.rks_scf`; the result carries per-spin MO
    energies, MO coefficients, and density matrices as ``(α, β)`` tuples.
    """
    # Validate the spin configuration (raises on inconsistent nelec/spin).
    _spin_counts(ks.nelec, ks.nalpha - ks.nbeta)
    X = canonical_orthonormalizer(ks.S, lindep_thresh)

    e_tot, Pa, Pb, Ca, Cb, ea, eb, converged, n_iter = _scf_solve_u(
        ks, X, max_iter, e_tol, d_tol, diis_space, verbose, level_shift
    )
    result = UKSResult(
        e_tot=float(e_tot),
        e_elec=float(e_tot) - float(ks.e_nn),
        converged=bool(converged),
        n_iter=int(n_iter),
        mo_energy=(ea, eb),
        mo_coeff=(Ca, Cb),
        P=(Pa, Pb),
    )
    if not result.converged:
        warnings.warn(
            f"UKS SCF did NOT converge in {result.n_iter} iterations "
            f"(e_tol={e_tol}, d_tol={d_tol}); the returned energy is unreliable. "
            f"Increase max_iter or try level_shift>0.",
            stacklevel=2,
        )
    return result
