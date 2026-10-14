"""Geometric direct minimization: the KS energy minimized over orbital rotations.

The SCF loop looks for a fixed point of the Fock map; this minimizes the energy
itself. The occupied orbitals of each spin channel are parametrized by one
occupied-virtual rotation ``kappa_sigma`` away from a fixed reference set,

    C_o(kappa) = C_o cos(sqrt(kappa kappa^T)) - C_v kappa^T sinc(sqrt(kappa kappa^T)),

the occupied block of ``[C_o C_v] exp([[0, kappa], [-kappa^T, 0]])`` in closed
form. The two matrix functions are power series in ``kappa kappa^T``
(``nocc x nocc``): entire functions, so the derivative is well defined at
``kappa = 0``, where an eigendecomposition of ``kappa kappa^T`` would be fully
degenerate. Every iterate is an orthonormal determinant, so every reported
energy is a variational upper bound, and nothing about it depends on the
solver having converged.

The gradient needs no hand derivation. ``KS.energy_and_fock`` returns the
energy and ``F = sym(dE/dP)`` from one traversal of the grid; with
``P = w C_o C_o^T`` the derivative with respect to the orbitals is
``2 w F C_o``, and one VJP through the rotation (small matrices only) turns it
into the exact derivative with respect to ``kappa``.

The step is L-BFGS in ``kappa`` (the reference stays fixed between resets, so
this is ordinary L-BFGS on a smooth function, as in the exponential
transformation direct minimization of Levi, Ivanov and Jonsson), seeded with
the diagonal orbital-Hessian model ``2 w (e_a - e_i)``, the orbital energies
read off the current Fock in the reference orbitals. A backtracking line
search with an energy-noise allowance keeps the descent monotone: the energy
has a floating-point floor of ~1e-10 |E| (see :func:`~dftax.ks.scf.scf`), and
a strict decrease test under that floor stalls, which is what collapses the
Newton trust region on ROKS CH3. When ``kappa`` grows large the reference
moves to the current orbitals, made pseudo-canonical in the current Fock.

One ``energy_and_fock`` per loop iteration and no other: the guess
evaluation, the first point and every line-search trial go through the same
call, so the expensive part of the graph is traced and compiled once.
"""

from __future__ import annotations

import math
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from dftax.ks.energy import KS
from dftax.ks.guess import GuessSpec, density_from_guess
from dftax.ks.scf import KSResult, canonical_orthonormalizer

# Terms of the rotation series (_series). ||kappa kappa^T|| stays below
# KAPPA_RESET**2 = 4 between resets, where 16 terms leave 4**16/32! ~ 2e-26.
N_SERIES = 16
KAPPA_RESET = 2.0

_GUESS, _FRESH, _TRIAL, _RESET = 0, 1, 2, 3


def _series(X, n=N_SERIES):
    """``cos(sqrt(X))``, ``sinc(sqrt(X))`` and ``(1 - cos(sqrt(X))) / X`` for
    symmetric PSD ``X``, by Horner on their power series in ``-X``. All three
    are entire, so all three differentiate at ``X = 0``."""
    # 1/(2k)! as python floats: the factorials overflow int64 past k = 10
    eye = jnp.eye(X.shape[0], dtype=X.dtype)
    inv = [1.0 / math.factorial(j) for j in range(2 * n + 3)]
    c = eye * inv[2 * n]
    s = eye * inv[2 * n + 1]
    g = eye * inv[2 * n + 2]
    for k in range(n - 1, -1, -1):
        c = eye * inv[2 * k] - X @ c
        s = eye * inv[2 * k + 1] - X @ s
        g = eye * inv[2 * k + 2] - X @ g
    return c, s, g


def _occupied(Co, Cv, kappa):
    """Occupied orbitals rotated by ``kappa`` (``nocc x nvir``)."""
    c, s, _ = _series(kappa @ kappa.T)
    return Co @ c - Cv @ (kappa.T @ s)


def rotate_full(Co, Cv, kappa):
    """Both blocks of ``[Co Cv] exp([[0, kappa], [-kappa^T, 0]])``.

    The virtual block is ``Co sinc(X) kappa + Cv (I - kappa^T g(X) kappa)``
    with ``X = kappa kappa^T`` and ``g(X) = (1 - cos sqrt X) / X``: the
    push-through identities turn every function of the ``nvir x nvir``
    ``kappa^T kappa`` into one of the ``nocc x nocc`` ``X``, so the whole
    rotation costs ``O(nao nocc nvir)`` instead of the ``O(nmo^3)`` of
    ``expm``."""
    c, s, g = _series(kappa @ kappa.T)
    Co_new = Co @ c - Cv @ (kappa.T @ s)
    Cv_new = Cv + Co @ (s @ kappa) - (Cv @ kappa.T) @ (g @ kappa)
    return Co_new, Cv_new


def _tdot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(a, b))


def _reference(F, Co, S, X):
    """Pseudo-canonical reference ``(Co, Cv, e_o, e_v)`` for occupied
    orbitals ``Co`` in Fock ``F``: the occupied block diagonalized as is, the
    virtual space the orthonormal complement of ``Co`` in the span of the
    orthonormalizer ``X``, diagonalized in turn. Forward only."""
    no = Co.shape[1]
    c = X.T @ (S @ Co)                                  # (nmo, no), orthonormal
    e_o, Uo = jnp.linalg.eigh(Co.T @ F @ Co)
    _, V = jnp.linalg.eigh(jnp.eye(c.shape[0], dtype=c.dtype) - c @ c.T)
    Cv0 = X @ V[:, no:]                                 # eigenvalue-1 vectors
    e_v, Uv = jnp.linalg.eigh(Cv0.T @ F @ Cv0)
    return Co @ Uo, Cv0 @ Uv, e_o, e_v


def _canonical(F, X, no):
    """Aufbau reference from a Fock: its own eigenvectors."""
    eps, Cp = jnp.linalg.eigh(X.T @ F @ X)
    C = X @ Cp
    return C[:, :no], C[:, no:], eps[:no], eps[no:]


def _lbfgs_direction(g, H0, Sh, Yh, rho, count, head, m):
    """Two-loop recursion; ``g``, ``H0`` and the history entries are tuples
    over spin channels, the history a circular buffer of depth ``m``."""

    def newest_first(k, carry):
        q, alphas = carry
        j = (head - 1 - k) % m
        live = k < count
        a = jnp.where(live, rho[j] * _tdot([s[j] for s in Sh], q), 0.0)
        q = tuple(qi - a * y[j] for qi, y in zip(q, Yh))
        return q, alphas.at[j].set(a)

    q, alphas = lax.fori_loop(0, m, newest_first,
                              (g, jnp.zeros(m, dtype=g[0].dtype)))
    r = tuple(h * qi for h, qi in zip(H0, q))

    def oldest_first(k, r):
        j = (head - count + k) % m
        live = k < count
        b = rho[j] * _tdot([y[j] for y in Yh], r)
        coef = jnp.where(live, alphas[j] - b, 0.0)
        return tuple(ri + coef * s[j] for ri, s in zip(r, Sh))

    r = lax.fori_loop(0, m, oldest_first, r)
    return tuple(-ri for ri in r)


@eqx.filter_jit
def _gdm_solve(ks: KS, X, P0, max_iter, e_tol, d_tol, max_step, gap_floor,
               m, verbose, orbital_xc=True):
    S = ks.S
    nocc = ks.nocc
    nspin = len(nocc)
    occ = ks.occupations()          # per-orbital weights, one vector a channel
    nmo = X.shape[1]
    dt = P0.dtype

    def density(Cos):
        return jnp.stack([(C * f) @ C.T for C, f in zip(Cos, occ)])

    def rotate(Co, Cv, kappa):
        return tuple(_occupied(a, b, k) for a, b, k in zip(Co, Cv, kappa))

    def full_eval(args):
        """Energy, d E / d kappa, d E / d C_o and the Fock, through the
        density matrix. The guess and the reference resets need the Fock."""
        P, Co, Cv, kappa = args
        e, F = ks.energy_and_fock(P)
        Cos, vjp = jax.vjp(lambda k: rotate(Co, Cv, k), kappa)
        dC = tuple(2.0 * (F[s] @ Cos[s]) * occ[s] for s in range(nspin))
        (g,) = vjp(dC)
        return e, g, dC, F

    def orbital_eval(args):
        """The same through the orbitals, which never forms the XC potential
        (KS.energy_and_grad_orbitals); the Fock is not built."""
        P, Co, Cv, kappa = args
        Cos, vjp = jax.vjp(lambda k: rotate(Co, Cv, k), kappa)
        e, dC = ks.energy_and_grad_orbitals(Cos)
        (g,) = vjp(dC)
        return e, g, dC, jnp.zeros_like(P)

    def precond(eo, ev):
        # 1 / (2 w_i (e_a - e_i)); an empty slot (w_i = 0) has no curvature
        # and no gradient, so it gets no step
        return tuple(jnp.where(
            f[:, None] > 0.0,
            1.0 / (2.0 * jnp.maximum(f[:, None], 1e-30)
                   * jnp.maximum(v[None, :] - o[:, None], gap_floor)),
            0.0) for o, v, f in zip(eo, ev, occ))

    def commutator(dC, Cos):
        # ||X^T (F P S - S P F) X|| from dE/dC_o = 2 w F C_o alone:
        # (1/sqrt 2) ||X^T (G - S C_o (C_o^T G))|| per channel, the Fock
        # never formed (see the identity in tests/unit/test_gdm.py)
        sq = 0.0
        for G, C in zip(dC, Cos):
            r = X.T @ (G - S @ (C @ (C.T @ G)))
            sq = sq + 0.5 * jnp.sum(r * r)
        return jnp.sqrt(sq)

    def new_reference(F, Cos, canonical):
        """(Co, Cv, e_o, e_v) per channel from a Fock: its aufbau eigenvectors
        at the guess, the pseudo-canonical frame of Cos at a reset."""
        out = [(_canonical(F[s], X, nocc[s]) if canonical
                else _reference(F[s], Cos[s], S, X)) for s in range(nspin)]
        return tuple(tuple(o[i] for o in out) for i in range(4))

    zeros_k = tuple(jnp.zeros((n, nmo - n), dt) for n in nocc)
    zeros_o = tuple(jnp.zeros((S.shape[0], n), dt) for n in nocc)
    zeros_v = tuple(jnp.zeros((S.shape[0], nmo - n), dt) for n in nocc)
    state0 = dict(
        it=jnp.asarray(0), mode=jnp.asarray(_GUESS),
        Co=zeros_o, Cv=zeros_v, kappa=zeros_k, d=zeros_k,
        eo=tuple(jnp.zeros(n, dt) for n in nocc),
        ev=tuple(jnp.zeros(nmo - n, dt) for n in nocc),
        alpha=jnp.asarray(0.0, dt), E=jnp.asarray(jnp.inf, dt), g=zeros_k,
        P=P0,
        Sh=tuple(jnp.zeros((m,) + k.shape, dt) for k in zeros_k),
        Yh=tuple(jnp.zeros((m,) + k.shape, dt) for k in zeros_k),
        rho=jnp.zeros(m, dt), count=jnp.asarray(0), head=jnp.asarray(0),
        derr=jnp.asarray(jnp.inf, dt), de=jnp.asarray(jnp.inf, dt),
        converged=jnp.asarray(False), resets=jnp.asarray(0),
        rejects=jnp.asarray(0),
    )

    def cond(st):
        return (st["it"] < max_iter) & jnp.logical_not(st["converged"])

    def body(st):
        mode = st["mode"]
        kappa_t = tuple(k + st["alpha"] * d
                        for k, d in zip(st["kappa"], st["d"]))
        Cos_t = rotate(st["Co"], st["Cv"], kappa_t)
        P_t = jnp.where(mode == _GUESS, st["P"], density(Cos_t))
        args = (P_t, st["Co"], st["Cv"], kappa_t)
        if orbital_xc:          # one call site each; the Fock only when needed
            e_t, g_t, dC_t, F_t = lax.cond(
                (mode == _GUESS) | (mode == _RESET), full_eval, orbital_eval,
                args)
        else:
            e_t, g_t, dC_t, F_t = full_eval(args)

        # --- the guess evaluation only builds the first reference ----------
        def from_guess(st):
            Co, Cv, eo, ev = new_reference(F_t, Cos_t, canonical=True)
            return dict(st, Co=Co, Cv=Cv, eo=eo, ev=ev, kappa=zeros_k,
                        d=zeros_k, alpha=jnp.asarray(0.0, dt),
                        mode=jnp.asarray(_FRESH))

        def accept_point(st, fresh, stalled, kappa_t, g_t, dC_t, Cos_t):
            """Take the evaluated point as the new base and step from it."""
            s_k = tuple(a - b for a, b in zip(kappa_t, st["kappa"]))
            y_k = tuple(a - b for a, b in zip(g_t, st["g"]))
            sy = _tdot(s_k, y_k)
            good = (~fresh) & ~stalled & (sy > 1e-12 * jnp.sqrt(
                _tdot(s_k, s_k) * _tdot(y_k, y_k)))
            head = st["head"]
            Sh = tuple(jnp.where(good, h.at[head].set(v), h)
                       for h, v in zip(st["Sh"], s_k))
            Yh = tuple(jnp.where(good, h.at[head].set(v), h)
                       for h, v in zip(st["Yh"], y_k))
            rho = jnp.where(good, st["rho"].at[head].set(1.0 / sy),
                            st["rho"])
            count = jnp.where(good, jnp.minimum(st["count"] + 1, m),
                              st["count"])
            head = jnp.where(good, (head + 1) % m, head)
            H0 = precond(st["eo"], st["ev"])
            d = _lbfgs_direction(g_t, H0, Sh, Yh, rho, count, head, m)
            # keep the step a rotation of bounded size
            big = jnp.max(jnp.asarray([jnp.max(jnp.abs(x)) for x in d]))
            d = tuple(x * jnp.minimum(1.0, max_step / big) for x in d)
            # a non-descent direction (stale curvature): fall back to the
            # preconditioned gradient and drop the history
            bad = _tdot(d, g_t) >= 0.0
            d = tuple(jnp.where(bad, -h * gi, di)
                      for h, gi, di in zip(H0, g_t, d))
            count = jnp.where(bad | stalled, 0, count)
            de = jnp.where(fresh, st["de"], e_t - st["E"])
            derr = commutator(dC_t, Cos_t)
            noise = 1e-10 * jnp.abs(e_t) + 1e-12
            # the energy test cannot be tighter than the energy's own floor
            # (it grows with the molecule: 3.5e-8 Ha on coronene, above the
            # default e_tol); the commutator carries the rest
            conv = ((~fresh) & (jnp.abs(de) < jnp.maximum(e_tol, noise))
                    & (derr < d_tol))
            return dict(st, kappa=kappa_t, E=e_t, g=g_t, P=density(Cos_t),
                        d=d, alpha=jnp.asarray(1.0, dt), Sh=Sh, Yh=Yh,
                        rho=rho, count=count, head=head, de=de, derr=derr,
                        converged=conv, mode=jnp.asarray(_TRIAL))

        # --- a point is evaluated: accept it, or shrink the step ----------
        def line_search(st):
            slope = _tdot(st["g"], st["d"])
            noise = 1e-10 * jnp.abs(st["E"]) + 1e-12
            fresh = mode == _FRESH
            accept = fresh | (e_t <= st["E"] + 1e-4 * st["alpha"] * slope
                              + noise)
            stalled = ~accept & (st["alpha"] < 1e-4)     # give up shrinking
            accept = accept | stalled

            def shrink(st):
                # minimizer of the quadratic through E(0), E'(0), E(alpha),
                # kept inside [0.1, 0.5] alpha
                a = st["alpha"]
                curv = e_t - st["E"] - slope * a
                a_q = jnp.where(curv > 0, -slope * a * a / (2.0 * curv),
                                0.5 * a)
                a_new = jnp.clip(a_q, 0.1 * a, 0.5 * a)
                return dict(st, alpha=a_new, rejects=st["rejects"] + 1)

            st = lax.cond(
                accept,
                lambda st: accept_point(st, fresh, stalled, kappa_t, g_t,
                                        dC_t, Cos_t),
                shrink, st)
            # move the reference once kappa is large: the next evaluation, at
            # the same point, builds the Fock for the new frame
            knorm = jnp.sqrt(_tdot(st["kappa"], st["kappa"]))
            do_reset = accept & (knorm > KAPPA_RESET) & ~st["converged"]
            return lax.cond(
                do_reset,
                lambda st: dict(st, d=zeros_k, alpha=jnp.asarray(0.0, dt),
                                mode=jnp.asarray(_RESET)),
                lambda st: st, st)

        # --- a reset: the current orbitals, made pseudo-canonical, become
        # the reference at kappa = 0; the same evaluation gives the gradient
        # there (first order: d C_o = -C_v dk^T), so it is taken as a fresh
        # point without a second evaluation
        def reset(st):
            Co, Cv, eo, ev = new_reference(F_t, Cos_t, canonical=False)
            dC0 = tuple(2.0 * (F_t[s] @ Co[s]) * occ[s] for s in range(nspin))
            g0 = tuple(-(G.T @ V) for G, V in zip(dC0, Cv))
            st = dict(st, Co=Co, Cv=Cv, eo=eo, ev=ev, kappa=zeros_k,
                      g=zeros_k, count=jnp.asarray(0),
                      resets=st["resets"] + 1)
            return accept_point(st, jnp.asarray(True), jnp.asarray(False),
                                zeros_k, g0, dC0, Co)

        branch = jnp.where(mode == _GUESS, 0, jnp.where(mode == _RESET, 2, 1))
        st = lax.switch(branch, [from_guess, line_search, reset], st)
        if verbose:
            jax.debug.print(
                "  gdm {it}: E={e:.10f} dE={de:+.2e} |[F,P]|={derr:.2e} "
                "alpha={a:.2e} resets={r} rejects={j}",
                it=st["it"], e=e_t, de=st["de"], derr=st["derr"],
                a=st["alpha"], r=st["resets"], j=st["rejects"])
        return dict(st, it=st["it"] + 1)

    final = lax.while_loop(cond, body, state0)
    return (final["E"], final["P"], final["converged"], final["it"],
            final["resets"], final["rejects"])


_energy_and_fock = eqx.filter_jit(lambda ks, P: ks.energy_and_fock(P))


def gdm(
    ks: KS,
    *,
    max_iter: int = 200,
    e_tol: float = 1e-8,
    d_tol: float = 1e-6,
    history: int = 20,
    max_step: float = 0.5,
    gap_floor: float = 0.05,
    lindep_thresh: float = 1e-7,
    guess: GuessSpec | jax.Array | None = None,
    orbital_xc: bool = True,
    verbose: bool = False,
) -> KSResult:
    """Minimize the KS energy over orbital rotations (restricted and
    spin-polarized alike).

    Args:
        ks: the built :class:`~dftax.ks.energy.KS` energy functional.
        max_iter: budget of energy evaluations (each is one Fock build).
        e_tol: convergence threshold on the energy change of an accepted
            step (Ha); same meaning and floor as in :func:`~dftax.ks.scf.scf`.
        d_tol: convergence threshold on the commutator norm, measured exactly
            as :func:`~dftax.ks.scf.scf` measures it, so the two solvers stop
            at the same point.
        history: L-BFGS memory depth.
        max_step: largest rotation angle (rad) any one step may take.
        gap_floor: smallest orbital-energy gap (Ha) the preconditioner uses,
            which keeps it positive where the current orbitals are not in
            aufbau order.
        lindep_thresh: overlap-eigenvalue cutoff for the orthonormalizer.
        guess: as for :func:`~dftax.ks.scf.scf`.
        orbital_xc: evaluate through the occupied orbitals
            (:meth:`~dftax.ks.energy.KS.energy_and_grad_orbitals`), building
            the Fock only at the guess and at reference resets; ``False``
            builds it at every evaluation.
        verbose: print every energy evaluation.

    ``n_iter`` of the result counts energy evaluations, the unit that costs.
    """
    X = canonical_orthonormalizer(ks.S, lindep_thresh)
    P0 = density_from_guess(ks, guess, X)
    e, P, converged, n_iter, _resets, _rejects = _gdm_solve(
        ks, X, P0, jnp.asarray(max_iter), jnp.asarray(e_tol),
        jnp.asarray(d_tol), jnp.asarray(max_step), jnp.asarray(gap_floor),
        history, verbose, orbital_xc,
    )
    # canonical orbitals of the converged density, for parity with scf
    # (forward only; P is the authoritative result). energy_and_fock, not
    # _fock_stacked: the latter is reverse mode through KS.electronic, whose
    # hybrid exchange is the dense density-route contraction
    _, F = _energy_and_fock(ks, P)
    eps, Cp = jnp.linalg.eigh(X.T @ F @ X)
    C = X @ Cp
    result = KSResult(
        e_tot=float(e),
        e_elec=float(e) - float(ks.e_nn) - float(ks.e_disp),
        converged=bool(converged),
        n_iter=int(n_iter),
        nocc=ks.nocc,
        mo_energy=eps,
        mo_coeff=C,
        P=P,
    )
    if not result.converged:
        warnings.warn(
            f"GDM did NOT converge in {result.n_iter} energy evaluations "
            f"(e_tol={e_tol}, d_tol={d_tol}); the returned energy is a "
            f"variational upper bound but not the minimum.",
            stacklevel=2,
        )
    return result
