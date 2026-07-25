"""Second-order SCF: trust-region Newton on occupied-virtual rotations.

The KS energy as a function of an orbital rotation ``C(kappa) = C exp(K)``,
with ``K`` the antisymmetric matrix holding one occupied-virtual block
``kappa_sigma`` per spin channel, has no gauge freedom left (occupied-occupied
and virtual-virtual rotations leave the density invariant and are excluded),
so its Hessian at a minimum is well conditioned and plain Newton converges
quadratically where damped DIIS limit-cycles.

The differentiable-programming payoff: the orbital gradient is ``jax.grad``
of that energy and the Hessian-vector products that drive the CG solve for
the Newton step are ``jax.jvp`` of the gradient. In conventional codes these
are hundreds of lines of hand-derived coupled-perturbed equations; here the
electronic-structure content is ~none. Rotations are pure matrix products of
an S-orthonormal reference (no eigendecomposition anywhere differentiated,
per the eigh-degeneracy rule; see :mod:`dftax.ks.minimize`).

Each accepted step re-anchors the reference orbitals and restarts from
``kappa = 0``, so ``exp(K)`` stays near the identity; a simple decrease-based
trust region (halve on uphill, grow on success) globalizes the iteration.
"""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from dftax.ks.energy import KS
from dftax.ks.guess import GuessSpec, density_from_guess
from dftax.ks.scf import KSResult, _fock_stacked, canonical_orthonormalizer


def _tdot(a, b):
    """Inner product of two matching pytrees (sum over leaves)."""
    return sum(jnp.sum(x * y)
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def _steihaug_cg(hvp, g, radius, maxiter):
    """Steihaug-Toint truncated CG for the trust-region Newton subproblem.

    Approximately minimizes ``m(p) = <g,p> + 1/2 <p, H p>`` over ``||p|| <=
    radius``. Plain CG assumes a positive-definite Hessian; at a saddle (or an
    ill-conditioned point) ``H`` is indefinite and plain CG produces a poor or
    uphill direction, which the decrease-only trust region then stalls on.
    Truncated CG instead follows the first direction of negative curvature
    (``<d, H d> <= 0``) to the trust boundary, and stops at the boundary when a
    CG step would cross it -- so the step is always a genuine model decrease,
    indefinite Hessian or not. Fixed ``maxiter`` inner iterations (jit-friendly:
    the loop always runs to length, freezing once the boundary is reached).

    ``hvp`` is the Hessian-vector product, ``g`` the gradient (a pytree). All
    vectors are pytrees matching ``g``; returns the step ``p``.
    """
    def to_boundary(z, d):
        # largest tau >= 0 with ||z + tau d|| = radius (positive quadratic root)
        dd = _tdot(d, d)
        zd = _tdot(z, d)
        zz = _tdot(z, z)
        disc = jnp.sqrt(jnp.maximum(zd * zd + dd * (radius * radius - zz), 0.0))
        tau = (-zd + disc) / jnp.maximum(dd, 1e-30)
        return jax.tree.map(lambda z_, d_: z_ + tau * d_, z, d)

    def sel(cond, a, b):
        return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)

    z0 = jax.tree.map(jnp.zeros_like, g)
    d0 = jax.tree.map(jnp.negative, g)                 # r0 = g (at z=0), d0 = -r0
    rr0 = _tdot(g, g)
    state = (z0, g, d0, rr0, jnp.array(False), z0)     # (z, r, d, rr, done, p)

    def step(_, st):
        z, r, d, rr, done, p = st
        Hd = hvp(d)
        dHd = _tdot(d, Hd)
        alpha = rr / jnp.where(dHd == 0.0, 1.0, dHd)
        z_next = jax.tree.map(lambda a, b: a + alpha * b, z, d)
        hit = (dHd <= 0.0) | (_tdot(z_next, z_next) >= radius * radius)
        p_step = sel(hit, to_boundary(z, d), z_next)   # boundary if truncating
        r_next = jax.tree.map(lambda r_, h_: r_ + alpha * h_, r, Hd)
        rr_next = _tdot(r_next, r_next)
        beta = rr_next / jnp.where(rr == 0.0, 1.0, rr)
        d_next = jax.tree.map(lambda rn_, d_: -rn_ + beta * d_, r_next, d)
        advanced = (z_next, r_next, d_next, rr_next, done | hit, p_step)
        return sel(done, st, advanced)                 # freeze once done

    z, r, d, rr, done, p = lax.fori_loop(0, maxiter, step, state)
    return p


def _rotate(C, kappa, nocc):
    """Apply ``C exp(K)`` per channel; K holds the occ-virt block of kappa."""
    out = []
    for s, (Cs, k) in enumerate(zip(C, kappa)):
        nmo = Cs.shape[1]
        no = nocc[s]
        K = jnp.zeros((nmo, nmo), dtype=Cs.dtype)
        K = K.at[:no, no:].set(k)
        K = K.at[no:, :no].set(-k.T)
        out.append(Cs @ jax.scipy.linalg.expm(K))
    return tuple(out)


def _density(C, nocc, w):
    """Spin-stacked density from S-orthonormal rotated orbitals."""
    return jnp.stack([w * (Cs[:, :n] @ Cs[:, :n].T)
                      for Cs, n in zip(C, nocc)])


@eqx.filter_jit
def _newton_solve(ks: KS, C0, max_iter, g_tol, e_tol, cg_iters, trust0,
                  verbose):
    """Trust-region Newton loop; C0 is a tuple of per-channel S-orthonormal
    (nao, nmo) references. Returns (e, P, C_stacked, converged, n_iter)."""
    nocc = ks.nocc
    nspin = len(nocc)
    w = 2.0 if nspin == 1 else 1.0
    kappa0 = tuple(jnp.zeros((nocc[s], C0[s].shape[1] - nocc[s]))
                   for s in range(nspin))

    def energy_at(C, kappa):
        return ks.total(_density(_rotate(C, kappa, nocc), nocc, w))

    def step(C, radius):
        e0 = energy_at(C, kappa0)
        g = jax.grad(lambda k: energy_at(C, k))(kappa0)
        gnorm = jnp.sqrt(sum(jnp.sum(x * x) for x in g))

        def hvp(v):
            return jax.jvp(jax.grad(lambda k: energy_at(C, k)),
                           (kappa0,), (v,))[1]

        # Steihaug-Toint truncated CG: trust-region-bounded and robust to the
        # indefinite Hessian at a saddle (no separate radius scaling needed).
        delta = _steihaug_cg(hvp, g, radius, cg_iters)
        C_new = _rotate(C, delta, nocc)
        e_new = energy_at(C_new, kappa0)
        return e0, gnorm, C_new, e_new

    def cond(st):
        return (st[0] < max_iter) & jnp.logical_not(st[4])

    def body(st):
        it, C, e_prev, radius, _ = st
        e0, gnorm, C_new, e_new = step(C, radius)
        accept = e_new < e0
        C = jax.tree.map(lambda a, b: jnp.where(accept, b, a), C, C_new)
        radius = jnp.where(accept, jnp.minimum(2.0 * radius, 1.0),
                           0.5 * radius)
        e = jnp.where(accept, e_new, e0)
        converged = (gnorm < g_tol) & (jnp.abs(e - e_prev) < e_tol)
        if verbose:
            jax.debug.print(
                "  newton {it}: E={e:.10f} |g|={g:.2e} r={r:.2e} ok={a}",
                it=it, e=e, g=gnorm, r=radius, a=accept,
            )
        return (it + 1, C, e, radius, converged)

    state0 = (0, C0, jnp.inf, jnp.asarray(trust0), jnp.array(False))
    it, C, e, _, converged = lax.while_loop(cond, body, state0)
    P = _density(C, nocc, w)
    return e, P, jnp.stack(C), converged, it


@eqx.filter_jit
def _newton_solve_shared(ks: KS, C0, mask, max_iter, g_tol, e_tol, cg_iters,
                         trust0, verbose):
    """Trust-region Newton over ONE shared orbital set (ROKS): both spin
    channels' densities come from the same rotated C, and kappa is masked to
    rotations that change the density (docc-socc, docc-virt, socc-virt)."""
    na, nb = ks.nocc

    def density(C):
        return jnp.stack([C[:, :na] @ C[:, :na].T, C[:, :nb] @ C[:, :nb].T])

    def rotate(C, kappa):
        k = kappa * mask
        K = k - k.T
        return C @ jax.scipy.linalg.expm(K)

    nmo = C0.shape[1]
    kappa0 = jnp.zeros((nmo, nmo))

    def energy_at(C, kappa):
        return ks.total(density(rotate(C, kappa)))

    def cond(st):
        return (st[0] < max_iter) & jnp.logical_not(st[4])

    def body(st):
        it, C, e_prev, radius, _ = st
        e0 = energy_at(C, kappa0)
        g = jax.grad(lambda k: energy_at(C, k))(kappa0) * mask
        gnorm = jnp.linalg.norm(g)

        def hvp(v):
            return jax.jvp(jax.grad(lambda k: energy_at(C, k)),
                           (kappa0,), (v * mask,))[1] * mask

        # Steihaug-Toint truncated CG (see _newton_solve): trust-bounded and
        # saddle-robust; mask keeps the step in the density-changing subspace.
        delta = _steihaug_cg(hvp, g, radius, cg_iters) * mask
        C_new = rotate(C, delta)
        e_new = energy_at(C_new, kappa0)
        accept = e_new < e0
        C = jnp.where(accept, C_new, C)
        radius = jnp.where(accept, jnp.minimum(2.0 * radius, 1.0),
                           0.5 * radius)
        e = jnp.where(accept, e_new, e0)
        converged = (gnorm < g_tol) & (jnp.abs(e - e_prev) < e_tol)
        if verbose:
            jax.debug.print(
                "  roks {it}: E={e:.10f} |g|={g:.2e} r={r:.2e} ok={a}",
                it=it, e=e, g=gnorm, r=radius, a=accept,
            )
        return (it + 1, C, e, radius, converged)

    state0 = (0, C0, jnp.inf, jnp.asarray(trust0), jnp.array(False))
    it, C, e, _, converged = lax.while_loop(cond, body, state0)
    return e, density(C), C, converged, it


def roks(
    ks: KS,
    *,
    max_iter: int = 64,
    g_tol: float = 1e-6,
    e_tol: float = 1e-10,
    cg_iters: int = 32,
    trust: float = 0.4,
    guess: GuessSpec | Array | None = None,
    verbose: bool = False,
) -> KSResult:
    """Restricted open-shell KS via shared-orbital Newton.

    One set of spatial orbitals for both spin channels (alpha fills the
    first ``nocc[0]``, beta the first ``nocc[1]``), so the wavefunction is a
    spin eigenfunction with none of UKS's spin contamination. Implemented as
    :func:`newton` over rotations of the single shared orbital set: kappa is
    masked to the doubly-occupied/singly-occupied/virtual inter-block
    rotations that change the density, and the constraint that the beta
    space sits inside the alpha space holds by construction. Requires a
    spin-polarized ``ks`` (``spin != 0``).

    Args:
        ks: the built spin-polarized :class:`~dftax.ks.energy.KS`.
        max_iter: maximum Newton iterations.
        g_tol: convergence threshold on the orbital-gradient norm.
        e_tol: convergence threshold on the energy change (Ha).
        cg_iters: CG iterations for the Newton step.
        trust: initial trust radius on the rotation norm.
        guess: initial density, same forms as :func:`~dftax.ks.scf.scf`.
        verbose: print per-iteration energy / gradient norm.

    Example:
        ```python
        ks = KS(mol, PBE(), spin=2)              # triplet
        res = roks(ks)                           # E >= UKS, no contamination
        ```
    """
    if len(ks.nocc) != 2:
        raise ValueError("roks needs a spin-polarized KS (build with spin!=0).")
    na, nb = ks.nocc
    X = canonical_orthonormalizer(ks.S)
    P0 = density_from_guess(ks, guess, X)
    F = _fock_stacked(ks, P0)
    # Shared reference orbitals from the spin-averaged Fock (values only).
    _, Cp = jnp.linalg.eigh(X.T @ (0.5 * (F[0] + F[1])) @ X)
    C0 = X @ Cp
    nmo = C0.shape[1]
    # occupation classes: 2 = doubly occupied, 1 = singly, 0 = virtual;
    # rotations within a class leave both densities invariant (gauge)
    cls = (jnp.arange(nmo) < nb).astype(int) + (jnp.arange(nmo) < na)
    mask = (cls[:, None] != cls[None, :]) & (jnp.arange(nmo)[:, None]
                                             < jnp.arange(nmo)[None, :])
    mask = mask.astype(C0.dtype)

    e_tot, P, C, converged, n_iter = _newton_solve_shared(
        ks, C0, mask, jnp.asarray(max_iter), jnp.asarray(g_tol),
        jnp.asarray(e_tol), cg_iters, trust, verbose,
    )
    F = _fock_stacked(ks, P)
    eps, Cp = jnp.linalg.eigh(X.T @ (0.5 * (F[0] + F[1])) @ X)
    result = KSResult(
        e_tot=float(e_tot),
        e_elec=float(e_tot) - float(ks.e_nn) - float(ks.e_disp),
        converged=bool(converged),
        n_iter=int(n_iter),
        nocc=ks.nocc,
        mo_energy=jnp.stack([eps, eps]),
        mo_coeff=jnp.stack([X @ Cp, X @ Cp]),
        P=P,
    )
    if not result.converged:
        warnings.warn(
            f"ROKS did NOT converge in {result.n_iter} iterations "
            f"(g_tol={g_tol}, e_tol={e_tol}); the returned energy is "
            f"unreliable. Increase max_iter or loosen the trust radius.",
            stacklevel=2,
        )
    return result


def newton(
    ks: KS,
    *,
    max_iter: int = 64,
    g_tol: float = 1e-6,
    e_tol: float = 1e-10,
    cg_iters: int = 32,
    trust: float = 0.4,
    guess: GuessSpec | Array | None = None,
    verbose: bool = False,
) -> KSResult:
    """Second-order SCF (trust-region Newton on orbital rotations).

    Converges quadratically near a minimum and handles densities that DIIS
    and ADIIS limit-cycle on (small-gap transition metals, stretched bonds).
    One Newton iteration costs a CG solve of Hessian-vector products, each
    the price of a couple of Fock builds, so for benign systems
    :func:`~dftax.ks.scf.scf` is faster; reach for this when robustness is
    the constraint, typically warm-started from a stalled SCF density
    (``newton(ks, guess=res.P)``).

    Args:
        ks: the built :class:`~dftax.ks.energy.KS` energy functional.
        max_iter: maximum Newton iterations.
        g_tol: convergence threshold on the orbital-gradient norm.
        e_tol: convergence threshold on the energy change (Ha).
        cg_iters: CG iterations for the Newton step (Hessian-vector solves).
        trust: initial trust radius on the rotation norm (radians); halved
            on an uphill proposal, doubled (capped at 1) on success.
        guess: initial density, same forms as :func:`~dftax.ks.scf.scf`.
        verbose: print per-iteration energy / gradient norm.

    Example:
        ```python
        res = scf(ks, max_iter=200)              # limit-cycles on a hard case
        res = newton(ks, guess=res.P)            # quadratic cleanup
        ```
    """
    X = canonical_orthonormalizer(ks.S)
    P0 = density_from_guess(ks, guess, X)
    # Reference orbitals: aufbau eigenvectors of the Fock at the guess
    # density (values only; nothing here is differentiated through eigh).
    F = _fock_stacked(ks, P0)
    _, Cp = jnp.linalg.eigh(X.T @ F @ X)
    C0 = tuple((X @ Cp)[s] for s in range(len(ks.nocc)))

    e_tot, P, C, converged, n_iter = _newton_solve(
        ks, C0, jnp.asarray(max_iter), jnp.asarray(g_tol),
        jnp.asarray(e_tol), cg_iters, trust, verbose,
    )
    # Canonical orbital energies at the converged density (report only).
    F = _fock_stacked(ks, P)
    eps, Cp = jnp.linalg.eigh(X.T @ F @ X)
    result = KSResult(
        e_tot=float(e_tot),
        e_elec=float(e_tot) - float(ks.e_nn) - float(ks.e_disp),
        converged=bool(converged),
        n_iter=int(n_iter),
        nocc=ks.nocc,
        mo_energy=eps,
        mo_coeff=X @ Cp,
        P=P,
    )
    if not result.converged:
        warnings.warn(
            f"Newton SCF did NOT converge in {result.n_iter} iterations "
            f"(g_tol={g_tol}, e_tol={e_tol}); the returned energy is "
            f"unreliable. Increase max_iter or loosen the trust radius.",
            stacklevel=2,
        )
    return result
