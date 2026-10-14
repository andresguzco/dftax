"""Direct minimization over a batch of different molecules in one program.

:func:`~dftax.ks.batched.scf_batched` batches geometries of one molecule, so
every array already has one shape. Different molecules do not: they differ in
basis size, auxiliary size, grid size and electron count. This pads each to
common sizes so that one compiled minimization serves the whole batch, which
is how a GPU stays busy on small molecules that one at a time leave it idle.

The padding is inert by construction:

* padded basis functions get an identity overlap block and nothing else (no
  grid values, no 3-center integrals), so no energy term sees them; their
  one-electron diagonal is ``-PAD`` for the first few and ``+PAD`` for the
  rest, so the aufbau guess places the padded occupied slots (occupation 0)
  in the negative ones and leaves the positive ones empty;
* padded occupied slots have occupation zero
  (:meth:`~dftax.ks.energy.KS.occupations`), so they carry no density, no
  gradient and get no step in :func:`~dftax.ks.gdm.gdm`'s preconditioner;
* padded auxiliary functions have zero integrals and a zero metric block;
  padded grid points have zero weight.

The real occupied orbitals therefore keep every rotation into the real
virtual space, and the energy, its gradient and the converged density on the
real block are those of the unpadded molecule.

``vmap`` runs every branch of a ``lax.cond`` for every molecule, so the
batched loop keeps heavy work out of branches: the guess Fock is built once
before the loop, the loop evaluates only through the orbitals
(:meth:`~dftax.ks.energy.KS.energy_and_grad_orbitals`), and a reference reset
re-anchors with the closed-form rotation (:func:`~dftax.ks.gdm.rotate_full`),
which needs neither a Fock nor an eigendecomposition.

Scope of this first version: closed-shell pure functionals on the
materialized density-fitting and AO-grid backends (what ``KS`` picks for
small and mid-size molecules). Hybrids need the exchange to see occupations,
and open shells a padded diagonal per spin channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float, Scalar

from dftax.ks.energy import KS
from dftax.ks.gdm import (
    KAPPA_RESET, _lbfgs_direction, _occupied, rotate_full,
)
from dftax.ks.guess import density_from_guess
from dftax.ks.scf import canonical_orthonormalizer
from dftax.ks.terms import DFCoulomb, GridXC

PAD = 1.0e3          # Ha; the padded one-electron diagonal, far from any level


class PaddedKS(eqx.Module):
    """A closed-shell ``KS`` padded to common sizes (see the module doc).

    Exposes what the minimizer reads, ``S``, ``nocc``, ``occupations``,
    ``energy_and_fock`` and ``energy_and_grad_orbitals``, through ``KS``'s own
    implementations, so a padded molecule is the same energy function.
    """

    S: Float[Array, "nao nao"]
    hcore: Float[Array, "nao nao"]
    e_nn: Scalar
    e_disp: Scalar
    coulomb: DFCoulomb
    xc_term: GridXC
    occ: Float[Array, "nocc"]
    nocc: tuple[int, ...] = eqx.field(static=True)

    def occupations(self):
        return (self.occ,)

    energy_and_fock = KS.energy_and_fock
    energy_and_grad_orbitals = KS.energy_and_grad_orbitals


@dataclass(frozen=True)
class Bucket:
    """Common sizes a batch is padded to."""

    nao: int
    naux: int
    ng: int
    nocc: int


def _check_supported(ks: KS) -> None:
    if len(ks.nocc) != 1:
        raise NotImplementedError("gdm_batched: closed shells only, so far.")
    if not isinstance(ks.coulomb, DFCoulomb) or not isinstance(ks.xc_term,
                                                                GridXC):
        raise NotImplementedError(
            "gdm_batched needs the materialized DF and AO-grid backends "
            "(df(chunk=None), becke(chunk=None)); got "
            f"{type(ks.coulomb).__name__} / {type(ks.xc_term).__name__}.")
    if ks.coulomb.hf_coeff != 0.0 or ks.coulomb.hf_coeff_lr != 0.0:
        raise NotImplementedError("gdm_batched: pure functionals only, so far.")
    if float(getattr(ks.xc_term.xc, "nlc_b", 0.0)) != 0.0:
        raise NotImplementedError("gdm_batched: no VV10, so far.")


def bucket_for(kss) -> Bucket:
    """The smallest common sizes, with room in the padded basis for the
    padded occupied slots of every member."""
    for ks in kss:
        _check_supported(ks)
    nocc = max(ks.nocc[0] for ks in kss)
    nao = max(int(ks.S.shape[0]) + (nocc - ks.nocc[0]) for ks in kss)
    naux = max(int(ks.coulomb.int2c_inv.shape[0]) for ks in kss)
    ng = max(int(ks.xc_term.weights.shape[0]) for ks in kss)
    return Bucket(nao=nao, naux=naux, ng=ng, nocc=nocc)


def _block(a, shape):
    out = jnp.zeros(shape, a.dtype)
    return out.at[tuple(slice(0, n) for n in a.shape)].set(a)


@partial(jax.jit, static_argnames=("npad_occ", "nao", "naux", "ng"))
def _pad_arrays(S, hcore, int3c, int2c_inv, ao, dao, weights, *, npad_occ,
                nao, naux, ng):
    """Every padded array of one molecule in one compiled program: the eager
    version dispatched a dozen ops per molecule and cost 11 ms each (262 ms of
    a 24-molecule batch whose solve takes 85 ms)."""
    n = S.shape[0]
    S_p = _block(S, (nao, nao)).at[n:, n:].set(jnp.eye(nao - n))
    diag = jnp.where(jnp.arange(nao - n) < npad_occ, -PAD, PAD)
    hcore_p = _block(hcore, (nao, nao)).at[n:, n:].set(jnp.diag(diag))
    return (S_p, hcore_p, _block(int3c, (nao, nao, naux)),
            _block(int2c_inv, (naux, naux)), _block(ao, (ng, nao)),
            _block(dao, (ng, nao, 3)), _block(weights, (ng,)))


def pad_ks(ks: KS, b: Bucket) -> PaddedKS:
    """``ks`` padded to the bucket (see the module doc for why it is inert)."""
    _check_supported(ks)
    npad_occ = b.nocc - ks.nocc[0]
    c, x = ks.coulomb, ks.xc_term
    S, hcore, int3c, int2c_inv, ao, dao, weights = _pad_arrays(
        ks.S, ks.hcore, c.int3c, c.int2c_inv, x.ao, x.dao, x.weights,
        npad_occ=npad_occ, nao=b.nao, naux=b.naux, ng=b.ng)
    occ = jnp.concatenate([jnp.zeros(npad_occ), jnp.full(ks.nocc[0], 2.0)])
    return PaddedKS(
        S=S, hcore=hcore, e_nn=ks.e_nn, e_disp=ks.e_disp,
        coulomb=DFCoulomb(int3c=int3c, int2c_inv=int2c_inv, hf_coeff=0.0),
        xc_term=GridXC(ao=ao, dao=dao, weights=weights, xc=x.xc, coords=None),
        occ=occ, nocc=(b.nocc,))


def _lowdin(S):
    s, U = jnp.linalg.eigh(S)
    return (U * (1.0 / jnp.sqrt(jnp.clip(s, 1e-9)))) @ U.T


def _solve_one(ks: PaddedKS, P0, max_iter, e_tol, d_tol, max_step, gap_floor,
               m):
    """One padded molecule; vmapped by :func:`gdm_batched`."""
    S = ks.S
    no = ks.nocc[0]
    occ = ks.occ
    X = _lowdin(S)
    dt = S.dtype

    # the guess reference, outside the loop: one Fock at the guess density
    _, F0 = ks.energy_and_fock(P0[None])
    eps, Cp = jnp.linalg.eigh(X.T @ F0[0] @ X)
    C = X @ Cp
    eo, ev = eps[:no], eps[no:]
    H0 = jnp.where(occ[:, None] > 0.0, 1.0 / (2.0 * jnp.maximum(occ, 1e-30)[
        :, None] * jnp.maximum(ev[None, :] - eo[:, None], gap_floor)), 0.0)
    zk = jnp.zeros((no, C.shape[1] - no), dt)

    def evaluate(Co, Cv, k):
        Cos, vjp = jax.vjp(lambda kk: (_occupied(Co, Cv, kk),), k)
        e, dC = ks.energy_and_grad_orbitals(Cos)
        (g,) = vjp(dC)                  # the cotangent of k itself
        return e, g, dC[0], Cos[0]

    def commutator(G, Co):
        r = X.T @ (G - S @ (Co @ (Co.T @ G)))
        return jnp.sqrt(0.5 * jnp.sum(r * r))

    st0 = dict(it=jnp.asarray(0), fresh=jnp.asarray(True), Co=C[:, :no],
               Cv=C[:, no:], k=zk, d=zk, alpha=jnp.asarray(0.0, dt),
               E=jnp.asarray(jnp.inf, dt), g=zk,
               Sh=jnp.zeros((m,) + zk.shape, dt),
               Yh=jnp.zeros((m,) + zk.shape, dt), rho=jnp.zeros(m, dt),
               count=jnp.asarray(0), head=jnp.asarray(0),
               de=jnp.asarray(jnp.inf, dt), derr=jnp.asarray(jnp.inf, dt),
               converged=jnp.asarray(False), Cocc=C[:, :no])

    def cond(st):
        return (st["it"] < max_iter) & jnp.logical_not(st["converged"])

    def body(st):
        k_t = st["k"] + st["alpha"] * st["d"]
        e_t, g_t, dC_t, Cos_t = evaluate(st["Co"], st["Cv"], k_t)
        slope = jnp.vdot(st["g"], st["d"])
        noise = 1e-10 * jnp.abs(st["E"]) + 1e-12
        fresh = st["fresh"]
        accept = fresh | (e_t <= st["E"] + 1e-4 * st["alpha"] * slope + noise)
        stalled = ~accept & (st["alpha"] < 1e-4)
        accept = accept | stalled

        # accepted: L-BFGS update and the next direction
        s_k, y_k = k_t - st["k"], g_t - st["g"]
        sy = jnp.vdot(s_k, y_k)
        good = accept & ~fresh & ~stalled & (sy > 1e-12 * jnp.sqrt(
            jnp.vdot(s_k, s_k) * jnp.vdot(y_k, y_k)))
        head = st["head"]
        Sh = jnp.where(good, st["Sh"].at[head].set(s_k), st["Sh"])
        Yh = jnp.where(good, st["Yh"].at[head].set(y_k), st["Yh"])
        rho = jnp.where(good, st["rho"].at[head].set(1.0 / sy), st["rho"])
        count = jnp.where(good, jnp.minimum(st["count"] + 1, m), st["count"])
        head = jnp.where(good, (head + 1) % m, head)
        (d,) = _lbfgs_direction((g_t,), (H0,), (Sh,), (Yh,), rho, count, head,
                                m)
        d = d * jnp.minimum(1.0, max_step / jnp.maximum(jnp.max(jnp.abs(d)),
                                                         1e-300))
        bad = jnp.vdot(d, g_t) >= 0.0
        d = jnp.where(bad, -H0 * g_t, d)
        count = jnp.where(bad | stalled, 0, count)
        de = jnp.where(fresh, st["de"], e_t - st["E"])
        derr = commutator(dC_t, Cos_t)
        conv = (~fresh & (jnp.abs(de) < jnp.maximum(e_tol, 1e-10 * jnp.abs(e_t)
                                                     + 1e-12))
                & (derr < d_tol))
        taken = dict(st, k=k_t, E=e_t, g=g_t, d=d, alpha=jnp.asarray(1.0, dt),
                     Sh=Sh, Yh=Yh, rho=rho, count=count, head=head, de=de,
                     derr=derr, converged=conv, fresh=jnp.asarray(False),
                     Cocc=Cos_t)

        # rejected: shrink the step along the same direction
        a = st["alpha"]
        curv = e_t - st["E"] - slope * a
        a_q = jnp.where(curv > 0, -slope * a * a / (2.0 * curv), 0.5 * a)
        shrunk = dict(st, alpha=jnp.clip(a_q, 0.1 * a, 0.5 * a))

        new = jax.tree.map(lambda x, y: jnp.where(accept, x, y), taken, shrunk)
        # re-anchor once kappa is large: the rotated orbitals become the
        # reference at kappa = 0 (no Fock, no eigh), and the next evaluation
        # is a fresh point
        reset = accept & ~new["converged"] & (
            jnp.sqrt(jnp.vdot(new["k"], new["k"])) > KAPPA_RESET)
        Co_r, Cv_r = rotate_full(new["Co"], new["Cv"], new["k"])
        new = dict(
            new,
            Co=jnp.where(reset, Co_r, new["Co"]),
            Cv=jnp.where(reset, Cv_r, new["Cv"]),
            k=jnp.where(reset, zk, new["k"]),
            d=jnp.where(reset, zk, new["d"]),
            alpha=jnp.where(reset, 0.0, new["alpha"]),
            count=jnp.where(reset, 0, new["count"]),
            fresh=new["fresh"] | reset,
        )
        return dict(new, it=st["it"] + 1)

    final = lax.while_loop(cond, body, st0)
    Cocc = final["Cocc"]
    P = (Cocc * occ) @ Cocc.T
    return final["E"], P, final["converged"], final["it"]


@eqx.filter_jit
def _solve_batch(stacked: PaddedKS, P0s, max_iter, e_tol, d_tol, max_step,
                 gap_floor, m):
    return jax.vmap(
        lambda ks, p0: _solve_one(ks, p0, max_iter, e_tol, d_tol, max_step,
                                  gap_floor, m))(stacked, P0s)


@dataclass
class BatchedGDMResult:
    """Per-molecule results of :func:`gdm_batched` (leading axis = batch)."""

    e_tot: np.ndarray
    converged: np.ndarray
    n_iter: np.ndarray
    P: list            # each molecule's density, unpadded (nao_i, nao_i)


def gdm_batched(kss, *, max_iter: int = 200, e_tol: float = 1e-8,
                d_tol: float = 1e-6, history: int = 20, max_step: float = 0.5,
                gap_floor: float = 0.05, guess=None,
                bucket: Bucket | None = None) -> BatchedGDMResult:
    """Minimize the KS energy of every system in ``kss`` at once.

    Args:
        kss: built :class:`~dftax.ks.energy.KS` objects (closed shell, pure
            functional, materialized DF and grid; see the module doc).
        bucket: the padded sizes; default the smallest that fits the batch.
            Passing a fixed bucket lets batches of similar molecules share
            one compiled program.
        guess: a guess spec as for :func:`~dftax.ks.scf.scf`, applied per
            molecule before padding.
        The tolerances and step controls are those of :func:`~dftax.ks.gdm.gdm`.
    """
    kss = list(kss)
    b = bucket if bucket is not None else bucket_for(kss)
    padded, P0s = [], []
    for ks in kss:
        padded.append(pad_ks(ks, b))
        P0 = density_from_guess(ks, guess, canonical_orthonormalizer(ks.S))[0]
        P0s.append(_block(P0, (b.nao, b.nao)))
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *padded)
    e, P, conv, it = _solve_batch(
        stacked, jnp.stack(P0s), jnp.asarray(max_iter), jnp.asarray(e_tol),
        jnp.asarray(d_tol), jnp.asarray(max_step), jnp.asarray(gap_floor),
        history)
    P = np.asarray(P)
    return BatchedGDMResult(
        e_tot=np.asarray(e), converged=np.asarray(conv),
        n_iter=np.asarray(it),
        P=[P[i, :int(ks.S.shape[0]), :int(ks.S.shape[0])]
           for i, ks in enumerate(kss)])


__all__ = ["gdm_batched", "pad_ks", "bucket_for", "Bucket", "PaddedKS",
           "BatchedGDMResult"]
