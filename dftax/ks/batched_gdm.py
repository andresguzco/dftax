from __future__ import annotations

from collections.abc import Sequence
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
    KAPPA_RESET, _lbfgs_direction, _occupied, _tdot, gdm, rotate_full,
)
from dftax.ks.guess import GuessSpec, density_from_guess
from dftax.ks.scf import canonical_orthonormalizer
from dftax.ks.terms import (
    DFCoulomb, GridXC, PackedDFCoulomb, WhitenedDFCoulomb,
)

PAD = 1.0e3


class PaddedKS(eqx.Module):
    S: Float[Array, "nao nao"]
    hcore: Float[Array, "nao nao"]
    e_nn: Scalar
    e_disp: Scalar
    coulomb: DFCoulomb | WhitenedDFCoulomb | PackedDFCoulomb
    xc_term: GridXC
    occ: tuple
    guess_shift: Float[Array, "nspin nao"]
    nocc: tuple[int, ...] = eqx.field(static=True)

    def occupations(self):
        return self.occ

    energy_and_fock = KS.energy_and_fock
    energy_and_grad_orbitals = KS.energy_and_grad_orbitals


@dataclass(frozen=True)
class Bucket:
    nao: int
    naux: int
    ng: int
    nocc: tuple[int, ...]
    npair: int = 0


def _check_supported(ks: KS) -> None:
    if not isinstance(ks.coulomb, (DFCoulomb, WhitenedDFCoulomb,
                                   PackedDFCoulomb)) or \
            not isinstance(ks.xc_term, GridXC):
        raise NotImplementedError(
            "gdm_batched needs the materialized DF and AO-grid backends "
            "(df(chunk=None), becke(chunk=None)); got "
            f"{type(ks.coulomb).__name__} / {type(ks.xc_term).__name__}.")
    if float(getattr(ks.xc_term.xc, "nlc_b", 0.0)) != 0.0:
        raise NotImplementedError("gdm_batched: no VV10, so far.")


def _tensors(c):
    if isinstance(c, PackedDFCoulomb):
        return None, None, None, None
    if isinstance(c, WhitenedDFCoulomb):
        return c.B, None, c.B_lr, None
    return c.int3c, c.int2c_inv, c.int3c_lr, c.int2c_inv_lr


def _naux(c):
    if isinstance(c, PackedDFCoulomb):
        return int(c.B.shape[0])
    return int(_tensors(c)[0].shape[2])


@partial(jax.jit, static_argnames=("nao", "naux", "npair"))
def _pad_packed(B, pos, rows, cols, weight, B_lr, *, nao, naux, npair):
    n = pos.shape[0]
    pos_p = jnp.full((nao, nao), npair, pos.dtype).at[:n, :n].set(pos)
    lr = None if B_lr is None else _block(B_lr, (naux, npair))
    return (_block(B, (naux, npair)), pos_p, _block(rows, (npair,)),
            _block(cols, (npair,)), _block(weight, (npair,)), lr)


def bucket_for(kss) -> Bucket:
    for ks in kss:
        _check_supported(ks)
    nspins = {len(ks.nocc) for ks in kss}
    if len(nspins) != 1:
        raise ValueError("one bucket holds one spin type; gdm_batched groups "
                         "closed and open shells itself.")
    nspin = nspins.pop()
    nocc = tuple(max(ks.nocc[s] for ks in kss) for s in range(nspin))
    nao = max(int(ks.S.shape[0]) + max(nocc[s] - ks.nocc[s]
                                       for s in range(nspin)) for ks in kss)
    naux = max(_naux(ks.coulomb) for ks in kss)
    ng = max(int(ks.xc_term.weights.shape[0]) for ks in kss)
    npair = max((int(ks.coulomb.B.shape[1]) for ks in kss
                 if isinstance(ks.coulomb, PackedDFCoulomb)), default=0)
    return Bucket(nao=nao, naux=naux, ng=ng, nocc=nocc, npair=npair)


def _block(a, shape):
    if a is None:
        return None
    out = jnp.zeros(shape, a.dtype)
    return out.at[tuple(slice(0, n) for n in a.shape)].set(a)


@partial(jax.jit, static_argnames=("nao", "naux", "ng"))
def _pad_arrays(S, hcore, int3c, int2c_inv, ao, dao, weights, int3c_lr,
                int2c_inv_lr, *, nao, naux, ng):
    n = S.shape[0]
    S_p = _block(S, (nao, nao)).at[n:, n:].set(jnp.eye(nao - n))
    hcore_p = _block(hcore, (nao, nao)).at[n:, n:].set(
        PAD * jnp.eye(nao - n))
    lr = (None if int3c_lr is None else
          (_block(int3c_lr, (nao, nao, naux)),
           _block(int2c_inv_lr, (naux, naux))))
    return (S_p, hcore_p, _block(int3c, (nao, nao, naux)),
            _block(int2c_inv, (naux, naux)), _block(ao, (ng, nao)),
            _block(dao, (ng, nao, 3)), _block(weights, (ng,)), lr)


def pad_ks(ks: KS, b: Bucket) -> PaddedKS:
    _check_supported(ks)
    nspin = len(ks.nocc)
    if len(b.nocc) != nspin:
        raise ValueError(f"bucket is for {len(b.nocc)} channel(s), the system "
                         f"has {nspin}.")
    n = int(ks.S.shape[0])
    c, x = ks.coulomb, ks.xc_term
    t3, vinv, t3_lr, vinv_lr = _tensors(c)
    S, hcore, int3c, int2c_inv, ao, dao, weights, lr = _pad_arrays(
        ks.S, ks.hcore, t3, vinv, x.ao, x.dao, x.weights, t3_lr, vinv_lr,
        nao=b.nao, naux=b.naux, ng=b.ng)
    w = 2.0 if nspin == 1 else 1.0
    empty = [b.nocc[s] - ks.nocc[s] for s in range(nspin)]
    occ = tuple(jnp.concatenate([jnp.zeros(empty[s]),
                                 jnp.full(ks.nocc[s], w)])
                for s in range(nspin))
    shift = np.zeros((nspin, b.nao))
    for s in range(nspin):
        shift[s, n:n + empty[s]] = -2.0 * PAD
    if isinstance(c, PackedDFCoulomb):
        if b.npair < c.B.shape[1]:
            raise ValueError(f"bucket npair {b.npair} < {c.B.shape[1]}.")
        Bp, pos, rows, cols, wt, Blr = _pad_packed(
            c.B, c.pos, c.rows, c.cols, c.weight, c.B_lr, nao=b.nao,
            naux=b.naux, npair=b.npair)
        coulomb = PackedDFCoulomb(
            B=Bp, pos=pos, rows=rows, cols=cols, weight=wt,
            hf_coeff=c.hf_coeff, B_lr=Blr, hf_coeff_lr=c.hf_coeff_lr)
    elif isinstance(c, WhitenedDFCoulomb):
        coulomb = WhitenedDFCoulomb(
            B=int3c, hf_coeff=c.hf_coeff, B_lr=None if lr is None else lr[0],
            hf_coeff_lr=c.hf_coeff_lr)
    else:
        coulomb = DFCoulomb(
            int3c=int3c, int2c_inv=int2c_inv, hf_coeff=c.hf_coeff,
            int3c_lr=None if lr is None else lr[0],
            int2c_inv_lr=None if lr is None else lr[1],
            hf_coeff_lr=c.hf_coeff_lr)
    return PaddedKS(
        S=S, hcore=hcore, e_nn=ks.e_nn, e_disp=ks.e_disp,
        coulomb=coulomb,
        xc_term=GridXC(ao=ao, dao=dao, weights=weights, xc=x.xc, coords=None),
        occ=occ, guess_shift=jnp.asarray(shift), nocc=tuple(b.nocc))


def _lowdin(S):
    s, U = jnp.linalg.eigh(S)
    return (U * (1.0 / jnp.sqrt(jnp.clip(s, 1e-9)))) @ U.T


def _where(c, a, b):
    return jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)


def _solve_one(ks: PaddedKS, P0, max_iter, e_tol, d_tol, max_step, gap_floor,
               m):
    S = ks.S
    nocc = ks.nocc
    nspin = len(nocc)
    occ = ks.occupations()
    X = _lowdin(S)
    dt = S.dtype

    _, F0 = ks.energy_and_fock(P0)
    Co0, Cv0, H0 = [], [], []
    for s in range(nspin):
        no = nocc[s]
        eps, Cp = jnp.linalg.eigh(
            X.T @ (F0[s] + jnp.diag(ks.guess_shift[s])) @ X)
        C = X @ Cp
        Co0.append(C[:, :no])
        Cv0.append(C[:, no:])
        f = occ[s]
        gap = jnp.maximum(eps[no:][None, :] - eps[:no][:, None], gap_floor)
        H0.append(jnp.where(f[:, None] > 0.0, 1.0 / (
            2.0 * jnp.maximum(f, 1e-30)[:, None] * gap), 0.0))
    Co0, Cv0, H0 = tuple(Co0), tuple(Cv0), tuple(H0)
    zk = tuple(jnp.zeros((n, X.shape[1] - n), dt) for n in nocc)

    def evaluate(Co, Cv, k):
        Cos, vjp = jax.vjp(
            lambda kk: tuple(_occupied(a, b, c) for a, b, c in zip(Co, Cv, kk)),
            k)
        e, dC = ks.energy_and_grad_orbitals(Cos)
        (g,) = vjp(dC)
        return e, g, dC, Cos

    def commutator(dC, Cos):
        sq = 0.0
        for G, C in zip(dC, Cos):
            r = X.T @ (G - S @ (C @ (C.T @ G)))
            sq = sq + 0.5 * jnp.sum(r * r)
        return jnp.sqrt(sq)

    st0 = dict(it=jnp.asarray(0), fresh=jnp.asarray(True), Co=Co0, Cv=Cv0,
               k=zk, d=zk, alpha=jnp.asarray(0.0, dt),
               E=jnp.asarray(jnp.inf, dt), g=zk,
               Sh=tuple(jnp.zeros((m,) + z.shape, dt) for z in zk),
               Yh=tuple(jnp.zeros((m,) + z.shape, dt) for z in zk),
               rho=jnp.zeros(m, dt), count=jnp.asarray(0),
               head=jnp.asarray(0), de=jnp.asarray(jnp.inf, dt),
               derr=jnp.asarray(jnp.inf, dt), converged=jnp.asarray(False),
               Cocc=Co0)

    def cond(st):
        return (st["it"] < max_iter) & jnp.logical_not(st["converged"])

    def body(st):
        k_t = tuple(k + st["alpha"] * d for k, d in zip(st["k"], st["d"]))
        e_t, g_t, dC_t, Cos_t = evaluate(st["Co"], st["Cv"], k_t)
        slope = _tdot(st["g"], st["d"])
        noise = 1e-10 * jnp.abs(st["E"]) + 1e-12
        fresh = st["fresh"]
        accept = fresh | (e_t <= st["E"] + 1e-4 * st["alpha"] * slope + noise)
        stalled = ~accept & (st["alpha"] < 1e-4)
        accept = accept | stalled

        s_k = tuple(a - b for a, b in zip(k_t, st["k"]))
        y_k = tuple(a - b for a, b in zip(g_t, st["g"]))
        sy = _tdot(s_k, y_k)
        good = accept & ~fresh & ~stalled & (sy > 1e-12 * jnp.sqrt(
            _tdot(s_k, s_k) * _tdot(y_k, y_k)))
        head = st["head"]
        Sh = tuple(jnp.where(good, h.at[head].set(v), h)
                   for h, v in zip(st["Sh"], s_k))
        Yh = tuple(jnp.where(good, h.at[head].set(v), h)
                   for h, v in zip(st["Yh"], y_k))
        rho = jnp.where(good, st["rho"].at[head].set(1.0 / sy), st["rho"])
        count = jnp.where(good, jnp.minimum(st["count"] + 1, m), st["count"])
        head = jnp.where(good, (head + 1) % m, head)
        d = _lbfgs_direction(g_t, H0, Sh, Yh, rho, count, head, m)
        big = jnp.max(jnp.asarray([jnp.max(jnp.abs(x)) for x in d]))
        d = tuple(x * jnp.minimum(1.0, max_step / jnp.maximum(big, 1e-300))
                  for x in d)
        bad = _tdot(d, g_t) >= 0.0
        d = tuple(jnp.where(bad, -h * gi, di) for h, gi, di in zip(H0, g_t, d))
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

        a = st["alpha"]
        curv = e_t - st["E"] - slope * a
        a_q = jnp.where(curv > 0, -slope * a * a / (2.0 * curv), 0.5 * a)
        shrunk = dict(st, alpha=jnp.clip(a_q, 0.1 * a, 0.5 * a))

        new = _where(accept, taken, shrunk)
        reset = accept & ~new["converged"] & (
            jnp.sqrt(_tdot(new["k"], new["k"])) > KAPPA_RESET)
        rot = [rotate_full(a, b, c)
               for a, b, c in zip(new["Co"], new["Cv"], new["k"])]
        new = dict(
            new,
            Co=_where(reset, tuple(r[0] for r in rot), new["Co"]),
            Cv=_where(reset, tuple(r[1] for r in rot), new["Cv"]),
            k=_where(reset, zk, new["k"]),
            d=_where(reset, zk, new["d"]),
            alpha=jnp.where(reset, 0.0, new["alpha"]),
            count=jnp.where(reset, 0, new["count"]),
            fresh=new["fresh"] | reset,
        )
        return dict(new, it=st["it"] + 1)

    final = lax.while_loop(cond, body, st0)
    P = jnp.stack([(C * f) @ C.T for C, f in zip(final["Cocc"], occ)])
    return final["E"], P, final["converged"], final["it"]


@eqx.filter_jit
def _solve_batch(stacked: PaddedKS, P0s, max_iter, e_tol, d_tol, max_step,
                 gap_floor, m):
    return jax.vmap(
        lambda ks, p0: _solve_one(ks, p0, max_iter, e_tol, d_tol, max_step,
                                  gap_floor, m))(stacked, P0s)


@dataclass
class BatchedGDMResult:
    """Per-system results of :func:`gdm_batched`, in input order.

    ``fallback`` marks the systems the batch did not settle (not converged,
    or converged to a state whose occupied orbitals are not the lowest ones)
    and that were re-solved one at a time by :func:`~dftax.ks.gdm.gdm`; their
    ``n_iter`` is that solve's.
    """

    e_tot: np.ndarray
    converged: np.ndarray
    n_iter: np.ndarray
    P: list
    fallback: np.ndarray = None


AUFBAU_TOL = 1e-3


@eqx.filter_jit
def _aufbau_violation(stacked: PaddedKS, P):
    def one(ks, P):
        _, F = ks.energy_and_fock(P)
        X = _lowdin(ks.S)
        bad = 0.0
        for s, f in enumerate(ks.occupations()):
            eps, U = jnp.linalg.eigh(X.T @ F[s] @ X)
            w = jnp.maximum(jnp.max(f), 1e-30)
            Xi = X.T @ ks.S
            q = jnp.einsum("ak,ab,bk->k", U, Xi @ P[s] @ Xi.T / w, U)
            n = jnp.sum(f > 0.0)
            top = eps[jnp.maximum(n - 1, 0)]
            bad = bad + jnp.sum(jnp.where(eps > top + AUFBAU_TOL, q, 0.0))
        return bad

    return jax.vmap(one)(stacked, P)


def _solve_group(kss, b, guess, max_iter, e_tol, d_tol, max_step, gap_floor,
                 history):
    padded, P0s = [], []
    for ks in kss:
        padded.append(pad_ks(ks, b))
        P0 = density_from_guess(ks, guess, canonical_orthonormalizer(ks.S))
        P0s.append(_block(P0, (len(b.nocc), b.nao, b.nao)))
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *padded)
    e, P, conv, it = _solve_batch(
        stacked, jnp.stack(P0s), jnp.asarray(max_iter), jnp.asarray(e_tol),
        jnp.asarray(d_tol), jnp.asarray(max_step), jnp.asarray(gap_floor),
        history)
    unsettled = ~np.asarray(conv) | (np.asarray(_aufbau_violation(stacked, P))
                                     > 0.5)
    P = np.asarray(P)
    return (np.asarray(e), np.asarray(conv), np.asarray(it),
            [P[i, :, :int(ks.S.shape[0]), :int(ks.S.shape[0])]
             for i, ks in enumerate(kss)], unsettled)


def gdm_batched(kss: Sequence[KS], *, max_iter: int = 200,
                e_tol: float = 1e-8, d_tol: float = 1e-6, history: int = 20,
                max_step: float = 0.5, gap_floor: float = 0.05,
                guess: GuessSpec | jax.Array | None = None,
                bucket: Bucket | None = None,
                fallback: bool = True) -> BatchedGDMResult:
    """Minimize the KS energy of every system in ``kss`` at once. The
    tolerances and step controls are those of :func:`~dftax.ks.gdm.gdm`.

    Args:
        kss: built :class:`~dftax.ks.energy.KS` objects (materialized DF and
            grid; see the module doc). Closed and open shells, and different
            functionals, may be mixed: they are solved as one batch per spin
            type and functional, results in input order.
        bucket: the padded sizes, for a batch of one spin type; default the
            smallest that fits each group. Passing a fixed bucket lets batches
            of similar molecules share one compiled program.
        guess: a guess spec as for :func:`~dftax.ks.scf.scf`, applied per
            molecule before padding.
        fallback: re-solve one at a time, with :func:`~dftax.ks.gdm.gdm`,
            every system the batch leaves unconverged or converges to a
            non-aufbau state (the batched loop keeps its preconditioner in
            the guess frame and can settle in an excited stationary state on
            hard cases such as Cr2; :func:`~dftax.ks.gdm.gdm` re-diagonalizes
            at its resets). Marked in ``BatchedGDMResult.fallback``.
    """
    kss = list(kss)
    groups: dict[tuple, list[int]] = {}
    for i, ks in enumerate(kss):
        key = (len(ks.nocc), ks.xc_term.xc, float(ks.coulomb.hf_coeff),
               float(ks.coulomb.hf_coeff_lr), type(ks.coulomb))
        groups.setdefault(key, []).append(i)
    if bucket is not None and len(groups) > 1:
        raise ValueError("a fixed bucket needs a batch of one spin type, "
                         "one functional and one DF term type.")
    n = len(kss)
    e, conv, it = np.zeros(n), np.zeros(n, bool), np.zeros(n, int)
    redo = np.zeros(n, bool)
    P = [None] * n
    for idx in groups.values():
        sub = [kss[i] for i in idx]
        b = bucket if bucket is not None else bucket_for(sub)
        ge, gc, gi, gP, gu = _solve_group(sub, b, guess, max_iter, e_tol,
                                          d_tol, max_step, gap_floor, history)
        for j, i in enumerate(idx):
            e[i], conv[i], it[i], P[i] = ge[j], gc[j], gi[j], gP[j]
            redo[i] = gu[j]
    if fallback:
        for i in np.nonzero(redo)[0]:
            r = gdm(kss[i], max_iter=max_iter, e_tol=e_tol, d_tol=d_tol,
                    history=history, max_step=max_step, gap_floor=gap_floor,
                    guess=guess)
            e[i], conv[i], it[i] = r.e_tot, r.converged, r.n_iter
            P[i] = np.asarray(r.P)
    return BatchedGDMResult(e_tot=e, converged=conv, n_iter=it, P=P,
                            fallback=redo if fallback else np.zeros(n, bool))


__all__ = ["gdm_batched", "pad_ks", "bucket_for", "Bucket", "PaddedKS",
           "BatchedGDMResult"]
