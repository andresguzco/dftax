"""Shell-class-bucketed 3-center ERI build.

The flat per-element engine in :mod:`eri3c` sizes every McMurchie-Davidson
recursion to the molecule and rebuilds the Hermite tables for every cartesian
component and every zero-padded primitive, so one heavy atom (or one g
auxiliary) taxes every light shell pair; ethanol/def2-svp built in 318 s on
an A100 with tens-of-GiB intermediates. This module groups (bra-pair, aux)
shell triples by their (l_a, l_b, l_aux) class and compiles one right-sized
kernel per class: E and Hermite-R tables are built once per primitive shell
triple and indexed by its components, primitives are trimmed to the class,
and bra symmetry halves the triple list. Same tensor to machine precision,
50x faster, sub-GiB peak.

Two-phase structure, matching the traced-vs-eager design rule (the Schwarz
screen sets the precedent): :func:`plan_eri3c` reads the *static* basis
metadata (angular momenta, contraction lengths) eagerly and returns a
hashable skeleton of python ints; :func:`eri3c_matrix_bucketed` consumes the
plan and touches the ``BasisData`` arrays only through jnp gathers with the
plan's static indices, so it traces cleanly inside the jitted
``_build_integrals`` and under ``forces``/``scf_batched`` (gradients flow
through ``centers``). Each shell reads its center from its first component
row; after the caller's ``centers = coords[atom_index]`` chain rule the
atom-coordinate gradients match the flat engine to machine precision, which
is the observable contract.
"""

from collections import defaultdict
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from dftax.energy.boys import boys
from dftax.utils.vmap import vmap as chunked_vmap


# Orbital angular-momentum ceiling of the integral engine: i (l=6), matching
# the auxiliary ceiling and the cart2sph/component tables, so cc-pV5Z (h) and
# cc-pV6Z (i) orbital bases build. Guarded eagerly in the plan builders
# (below) rather than left to fail deep in a traced build, where an over-cap
# basis silently balloons the recursion instead.
_ORBITAL_L_MAX = 6


def _check_orbital_l(basis):
    """Reject orbital bases above the engine's angular-momentum ceiling."""
    L = int(basis.max_l)
    if L > _ORBITAL_L_MAX:
        raise ValueError(
            f"the integral engine supports orbital angular momentum up to i "
            f"(l={_ORBITAL_L_MAX}); got l={L}."
        )


# ---------------------------------------------------------------------------
# Phase 1: the eager, static plan (python ints only; hashable)
# ---------------------------------------------------------------------------

def _shells(angular, exponents):
    """Shell records from static metadata: (l, row0, ncomp, nprim).

    A shell starts wherever the canonical component sequence restarts at
    ``(l, 0, 0)`` (every l=0 row is its own shell); no center reads, so two
    same-l shells on one atom split correctly by row order alone.
    """
    ang = np.asarray(angular)
    ex = np.asarray(exponents)
    ltot = ang.sum(1)
    n = ang.shape[0]
    starts = [i for i in range(n)
              if ang[i, 0] == ltot[i] and ang[i, 1] == 0 and ang[i, 2] == 0]
    if not starts or starts[0] != 0:
        raise ValueError("basis rows do not start shells at (l,0,0); the "
                         "bucketed eri3c build assumes gto.py row order")
    bounds = starts + [n]
    return [(int(ltot[s]), s, e - s, max(1, int((ex[s] != 0).sum())))
            for s, e in zip(bounds[:-1], bounds[1:])], ang


def _shell_pair_keep(basis, thresh):
    """Kept bra shell-pairs ``(ia, ib)`` with ``ia <= ib`` under a relative
    Schwarz cutoff (the shell-level analogue of
    :func:`~dftax.integrals.eri4c.significant_pairs`).

    A bra shell-pair enters ``(μν|P)`` only through its own charge density, so
    ``|(μν|P)| <= Q_μν · √(P|P)`` with ``Q_μν = √(μν|μν)``; a shell-pair whose
    largest component ``Q`` falls below ``thresh`` (relative to the global max)
    is negligible for every auxiliary ``P``. Screening whole shell-pairs (keep
    if any component survives) matches the bucketed engine's shell batching and
    is conservative w.r.t. the per-AO test. Geometry-only, host-side; for
    extended systems the survivors are O(N).
    """
    from dftax.integrals.eri4c import schwarz_diagonal

    shells, _ = _shells(basis.angular, basis.exponents)
    nao = int(np.asarray(basis.angular).shape[0])
    row_shell = np.empty(nao, dtype=np.int64)
    for si, (_l, row0, ncomp, _np) in enumerate(shells):
        row_shell[row0:row0 + ncomp] = si
    Q = np.asarray(schwarz_diagonal(basis))              # √(μν|μν), tril μ>=ν
    pa, pb = np.tril_indices(nao)
    lo = np.minimum(row_shell[pa], row_shell[pb])
    hi = np.maximum(row_shell[pa], row_shell[pb])
    ns = len(shells)
    Qsh = np.zeros((ns, ns))
    np.maximum.at(Qsh, (hi, lo), Q)                      # max over components
    qmax = float(Qsh.max()) if Qsh.size else 0.0
    hi_idx, lo_idx = np.nonzero(Qsh >= thresh * qmax)
    return {(int(a), int(b)) for b, a in zip(hi_idx, lo_idx)}  # (ia<=ib)


def plan_eri3c(basis, aux_basis, keep_pairs=None):
    """Static bucket plan for :func:`eri3c_matrix_bucketed`.

    Must be called where the basis metadata is concrete (KS.__init__, or a
    closure over the basis template): everything it returns is python ints
    in nested tuples, safe to pass through ``eqx.filter_jit`` as a static
    argument.

    ``keep_pairs`` (from :func:`_shell_pair_keep`) optionally restricts the
    build to the Schwarz-significant bra shell-pairs; screened pairs are absent
    from the plan and stay exactly zero in the scattered tensor. ``None`` keeps
    every pair (the default, bit-identical build).
    """
    _check_orbital_l(basis)
    bra, ang_b = _shells(basis.angular, basis.exponents)
    aux, ang_a = _shells(aux_basis.angular, aux_basis.exponents)
    buckets = defaultdict(lambda: ([], [], []))
    nprims = {}
    for ia, (la, ra, nca, npa) in enumerate(bra):
        for jb_off, (lb, rb, ncb, npb) in enumerate(bra[ia:]):
            if keep_pairs is not None and (ia, ia + jb_off) not in keep_pairs:
                continue
            for lc, rc, ncc, npc in aux:
                key = (la, lb, lc)
                b = buckets[key]
                b[0].append(ra); b[1].append(rb); b[2].append(rc)
                cur = nprims.get(key, (0, 0, 0))
                nprims[key] = (max(cur[0], npa), max(cur[1], npb),
                               max(cur[2], npc))

    def ang_tup(ang, row0, l, ncomp):
        return tuple(tuple(int(x) for x in ang[row0 + i])
                     for i in range(ncomp))

    classes = []
    for (la, lb, lc), (rows_a, rows_b, rows_c) in sorted(buckets.items()):
        nca = (la + 1) * (la + 2) // 2
        ncb = (lb + 1) * (lb + 2) // 2
        ncc = (lc + 1) * (lc + 2) // 2
        classes.append((
            la, lb, lc,
            ang_tup(ang_b, rows_a[0], la, nca),
            ang_tup(ang_b, rows_b[0], lb, ncb),
            ang_tup(ang_a, rows_c[0], lc, ncc),
            tuple(rows_a), tuple(rows_b), tuple(rows_c),
            nprims[(la, lb, lc)],
        ))
    nao = int(np.asarray(basis.angular).shape[0])
    naux = int(np.asarray(aux_basis.angular).shape[0])
    return (nao, naux, tuple(classes))


# ---------------------------------------------------------------------------
# Right-sized table builders (trace-time unrolled, per-level vectorized)
# ---------------------------------------------------------------------------

def _bump(row, X, i2g, mt):
    """One MD E-recursion step on a length-mt row."""
    left = jnp.concatenate([jnp.zeros_like(row[:1]), row[:-1]])
    right = jnp.concatenate([row[1:], jnp.zeros_like(row[:1])])
    t_idx = jnp.arange(mt)
    return (X * row + i2g * jnp.where(t_idx > 0, left, 0.0)
            + jnp.where(t_idx + 1 < mt, (t_idx + 1) * right, 0.0))


def _E_table(la, lb, alpha, beta, XAB, mt):
    """(la+1, lb+1, mt) two-center E table, unrolled at trace time."""
    gamma = alpha + beta
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    XPA = -beta * XAB / safe_gamma
    XPB = alpha * XAB / safe_gamma
    col = [jnp.zeros(mt).at[0].set(1.0)]
    for _ in range(la):
        col.append(_bump(col[-1], XPA, i2g, mt))
    rows = [col]
    for _ in range(lb):
        rows.append([_bump(r, XPB, i2g, mt) for r in rows[-1]])
    return jnp.stack([jnp.stack([rows[j][i] for j in range(lb + 1)])
                      for i in range(la + 1)])


def _Ec_table(lc, gamma, mt):
    """(lc+1, mt) single-center E table, unrolled at trace time."""
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    rows = [jnp.zeros(mt).at[0].set(1.0)]
    for _ in range(lc):
        rows.append(_bump(rows[-1], 0.0, i2g, mt))
    return jnp.stack(rows)


def _hermite_table(rho, RPC, mt, omega=None):
    """(mt, mt, mt) Hermite Coulomb integrals R^0_{t,u,v}.

    Same m-ladder as :func:`eri3c._hermite_coulomb`; every entry of level m
    reads only level m+1, so the descending sweep is a ``lax.fori_loop`` over
    the mt levels carrying the running (mt, mt, mt) table. The loop body
    traces once instead of mt times: the unrolled form built an O(mt^4)-node
    graph per class, and since the Hermite table is shared by the eri3c,
    eri2c and nuclear builds it dominated the engine's first-call trace and
    compile (rolling it is ~2.3x on the eri3c build, bit-identical, with no
    change to the execute time). The E tables stay unrolled: they are tiny
    (O(l*mt)) and a loop there only adds overhead. The per-level t/u/v sweeps
    remain vectorized slice updates.
    """
    T = rho * jnp.sum(RPC ** 2)
    neg2rho = -2.0 * rho
    if omega is None:
        base = jnp.stack([boys(m, T) for m in range(mt)])
    else:
        s = (omega * omega) / (omega * omega + rho)
        base = jnp.stack([s ** (m + 0.5) * boys(m, s * T) for m in range(mt)])
    # Integer powers (Python range): neg2rho is negative, so a float exponent
    # would NaN; range(mt) keeps the base real.
    powers = jnp.stack([neg2rho ** m for m in range(mt)])
    idx = jnp.arange(mt - 1)
    zrow = jnp.zeros((1,))

    Rp0 = jnp.zeros((mt, mt, mt)).at[0, 0, 0].set(powers[mt - 1] * base[mt - 1])

    def _level(k, Rp):
        m = mt - 2 - k                       # descending m = mt-2 .. 0
        R = jnp.zeros((mt, mt, mt)).at[0, 0, 0].set(powers[m] * base[m])
        row = Rp[:, 0, 0]
        shifted = jnp.concatenate([zrow, row[:-2]])
        R = R.at[1:, 0, 0].set(RPC[0] * row[:-1] + idx * shifted)
        plane = Rp[:, :-1, 0]
        pshift = jnp.concatenate([jnp.zeros((mt, 1)), Rp[:, :-2, 0]], axis=1)
        R = R.at[:, 1:, 0].set(RPC[1] * plane + idx[None, :] * pshift)
        cube = Rp[:, :, :-1]
        cshift = jnp.concatenate([jnp.zeros((mt, mt, 1)), Rp[:, :, :-2]], axis=2)
        R = R.at[:, :, 1:].set(RPC[2] * cube + idx[None, None, :] * cshift)
        return R

    return lax.fori_loop(0, mt - 1, _level, Rp0)


# ---------------------------------------------------------------------------
# Per-class kernels (compiled once per class shape, cached)
# ---------------------------------------------------------------------------

def _make_class_kernel(la, lb, lc, anga, angb, angc, omega=None):
    """Kernel for one (la, lb, lc) class; one triple -> (nca, ncb, ncc)."""
    # floor 2: a pure-s class still needs one off-origin Hermite index
    mt = max(la + lb + lc + 1, 2)
    # closure constants stay numpy: the factory can first run inside an
    # active jit trace (the cached kernel outlives it), and a jnp constant
    # created there is trace-scoped and leaks into later traces
    conv = np.zeros((mt, mt, mt))
    for t in range(mt):
        for tau in range(mt - t):
            conv[t, tau, t + tau] = 1.0
    sign = np.asarray([(-1.0) ** t for t in range(mt)])
    anga, angb, angc = np.asarray(anga), np.asarray(angb), np.asarray(angc)
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])
    kc = (angc[:, 0], angc[:, 1], angc[:, 2])

    def one_triple(A, B, C, ea, eb, ec, ca, cb, cc):
        AB = A - B

        def per_ab(al, be):
            gab = al + be
            safe = jnp.where(gab == 0.0, 1.0, gab)
            P = (al * A + be * B) / safe
            K = jnp.exp(-al * be / safe * jnp.sum(AB ** 2))
            Et = [_E_table(la, lb, al, be, AB[x], mt) for x in range(3)]

            def per_c(ga):
                safec = jnp.where(ga == 0.0, 1.0, ga)
                rho = safe * safec / (safe + safec)
                pref = (K * 2.0 * jnp.pi ** 2.5
                        / (safe * safec * jnp.sqrt(safe + safec)))
                R = _hermite_table(rho, P - C, mt, omega)
                Ec = [_Ec_table(lc, ga, mt) * sign for _ in range(3)]
                G = [jnp.einsum("ijt,ku,tus->ijks", Et[x], Ec[x], conv)
                     for x in range(3)]
                GX = G[0][ia[0][:, None, None], jb[0][None, :, None],
                          kc[0][None, None, :], :]
                GY = G[1][ia[1][:, None, None], jb[1][None, :, None],
                          kc[1][None, None, :], :]
                GZ = G[2][ia[2][:, None, None], jb[2][None, :, None],
                          kc[2][None, None, :], :]
                return pref * jnp.einsum("abcs,abcr,abcq,srq->abc",
                                         GX, GY, GZ, R)

            return jax.vmap(per_c)(ec)

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        return jnp.einsum("ijkabc,ai,bj,ck->abc", vals, ca, cb, cc)

    return one_triple


@lru_cache(maxsize=4096)
def _compiled_class_kernel(la, lb, lc, anga, angb, angc, omega, chunk_size):
    """Jitted, cached batch kernel for one class.

    Without the cache every build re-traces the unrolled graphs through
    Python, which costs more than the GPU work itself. The ang tuples are
    hashable static metadata; jit keys the array shapes.
    """
    kern = _make_class_kernel(la, lb, lc, anga, angb, angc, omega)
    # explicit in_axes: the chunked vmap only batches the axes it is told
    # about, and a bare 0 covers just the first argument
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 9,
                                chunk_size=chunk_size))


# ---------------------------------------------------------------------------
# Phase 2: the traced build (jnp gathers only; jit/grad-safe)
# ---------------------------------------------------------------------------

def plan_pairs(basis):
    """Static bra-pair plan for :func:`nuclear_attraction_bucketed`.

    Same contract as :func:`plan_eri3c`: python ints only, computed where the
    basis metadata is concrete, safe as a static jit argument.

    No orbital-l guard here: this plan also serves the 2-center aux build
    (``eri2c_matrix_bucketed``), whose aux basis legitimately reaches l=6.
    The orbital ceiling is enforced by the callers that know their basis is
    an orbital one (``overlap_kinetic_bucketed``, ``nuclear_attraction_bucketed``).
    """
    bra, ang = _shells(basis.angular, basis.exponents)
    buckets = defaultdict(lambda: ([], []))
    nprims = {}
    for ia, (la, ra, nca, npa) in enumerate(bra):
        for lb, rb, ncb, npb in bra[ia:]:
            key = (la, lb)
            b = buckets[key]
            b[0].append(ra); b[1].append(rb)
            cur = nprims.get(key, (0, 0))
            nprims[key] = (max(cur[0], npa), max(cur[1], npb))

    def ang_tup(row0, l, ncomp):
        return tuple(tuple(int(x) for x in ang[row0 + i])
                     for i in range(ncomp))

    classes = []
    for (la, lb), (rows_a, rows_b) in sorted(buckets.items()):
        nca = (la + 1) * (la + 2) // 2
        ncb = (lb + 1) * (lb + 2) // 2
        classes.append((
            la, lb,
            ang_tup(rows_a[0], la, nca), ang_tup(rows_b[0], lb, ncb),
            tuple(rows_a), tuple(rows_b), nprims[(la, lb)],
        ))
    return (int(np.asarray(basis.angular).shape[0]), tuple(classes))


def _make_pair_kernel(la, lb, anga, angb):
    """Nuclear-attraction kernel for one (la, lb) bra class.

    One pair -> (nca, ncb); the nucleus sum is vectorized inside:
    V[a, b] = sum_A -Z_A K_AB (2 pi / gamma) sum_tuv E^x_t E^y_u E^z_v
    R_tuv(gamma, P - R_A), matching _nuclear_attraction_primitive.
    """
    mt = max(la + lb + 1, 2)
    anga, angb = np.asarray(anga), np.asarray(angb)
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])

    def one_pair(A, B, ea, eb, ca, cb, atom_coords, atom_charges):
        AB = A - B

        def per_ab(al, be):
            gab = al + be
            safe = jnp.where(gab == 0.0, 1.0, gab)
            P = (al * A + be * B) / safe
            K = jnp.exp(-al * be / safe * jnp.sum(AB ** 2))
            Et = [_E_table(la, lb, al, be, AB[x], mt) for x in range(3)]

            def per_atom(C, Z):
                return -Z * _hermite_table(safe, P - C, mt)

            Rsum = jnp.sum(jax.vmap(per_atom)(atom_coords, atom_charges),
                           axis=0)
            pref = K * 2.0 * jnp.pi / safe
            GX = Et[0][ia[0][:, None], jb[0][None, :], :]
            GY = Et[1][ia[1][:, None], jb[1][None, :], :]
            GZ = Et[2][ia[2][:, None], jb[2][None, :], :]
            return pref * jnp.einsum("abs,abr,abq,srq->ab", GX, GY, GZ, Rsum)

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        return jnp.einsum("ijab,ai,bj->ab", vals, ca, cb)

    return one_pair


@lru_cache(maxsize=4096)
def _compiled_pair_kernel(la, lb, anga, angb, chunk_size):
    """Jitted, cached batch kernel for one bra class (see the eri3c cache
    note: numpy closure constants, hashable static metadata)."""
    kern = _make_pair_kernel(la, lb, anga, angb)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 6 + (None, None),
                                chunk_size=chunk_size))


def nuclear_attraction_bucketed(basis, atom_coords, atom_charges, plan=None,
                                chunk=4096):
    """(nao, nao) nuclear attraction matrix via shell-class buckets.

    Drop-in for the flat builder; the per-element build held every
    (pair, nucleus) Hermite table at molecule-padded sizes at once (27.5 GiB
    for ethanol/def2-svp, the KS build's memory peak). Differentiable
    w.r.t. ``basis.centers`` and ``atom_coords``.
    """
    _check_orbital_l(basis)
    if plan is None:
        plan = plan_pairs(basis)
    nao, classes = plan
    cen = basis.centers
    out = jnp.zeros((nao, nao), dtype=cen.dtype)
    for (la, lb, anga, angb, ra, rb, nprims) in classes:
        npa, npb = nprims
        ra = np.asarray(ra); rb = np.asarray(rb)
        nca = (la + 1) * (la + 2) // 2
        ncb = (lb + 1) * (lb + 2) // 2
        ca = basis.coefficients[ra[:, None] + np.arange(nca)][:, :, :npa]
        cb = basis.coefficients[rb[:, None] + np.arange(ncb)][:, :, :npb]
        fn = _compiled_pair_kernel(la, lb, anga, angb,
                                   min(chunk, ra.shape[0]))
        vals = fn(cen[ra], cen[rb],
                  basis.exponents[ra][:, :npa], basis.exponents[rb][:, :npb],
                  ca, cb, atom_coords, atom_charges)   # (npair, nca, ncb)
        i_idx = ra[:, None, None] + np.arange(nca)[:, None]
        j_idx = rb[:, None, None] + np.arange(ncb)[None, :]
        i_idx, j_idx = (np.broadcast_to(x, vals.shape)
                        for x in (i_idx, j_idx))
        out = out.at[i_idx.ravel(), j_idx.ravel()].set(vals.reshape(-1))
        # bra symmetry via index swap (same rule as the 3-center scatter)
        out = out.at[j_idx.ravel(), i_idx.ravel()].set(vals.reshape(-1))
    if basis.cart2sph is not None:
        out = basis.cart2sph.T @ out @ basis.cart2sph
    return out


def _os_table(la, lb_ext, PA, PB, i2g):
    """(la+1, lb_ext+1) 1D Obara-Saika overlap table, unrolled at trace time.

    S(a+1, b) = PA S(a, b) + a i2g S(a-1, b) + b i2g S(a, b-1); the kinetic
    caller extends the b axis two beyond the class l to read S(a, b+2).
    """
    rows = [[None] * (lb_ext + 1) for _ in range(la + 1)]
    rows[0][0] = jnp.ones(())
    for a in range(la):
        val = PA * rows[a][0]
        if a > 0:
            val = val + a * i2g * rows[a - 1][0]
        rows[a + 1][0] = val
    for b in range(lb_ext):
        for a in range(la + 1):
            val = PB * rows[a][b]
            if b > 0:
                val = val + b * i2g * rows[a][b - 1]
            if a > 0:
                val = val + a * i2g * rows[a - 1][b]
            rows[a][b + 1] = val
    return jnp.stack([jnp.stack(r) for r in rows])


def _make_st_kernel(la, lb, anga, angb):
    """Overlap + kinetic kernel for one (la, lb) bra class.

    One OS table per axis (b axis extended by 2) serves the overlap and all
    seven kinetic terms as index reads; the flat engine rebuilt the tables
    seven times per element (base plus the six b-shifted overlaps).
    """
    anga, angb = np.asarray(anga), np.asarray(angb)
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])
    Lb = angb.sum(1)                                   # (ncb,) static

    def one_pair(A, B, ea, eb, ca, cb):
        AB = A - B

        def per_ab(al, be):
            gab = al + be
            safe = jnp.where(gab == 0.0, 1.0, gab)
            i2g = 0.5 / safe
            P = (al * A + be * B) / safe
            PA = P - A
            PB = P - B
            K = jnp.exp(-al * be / safe * jnp.sum(AB ** 2))
            pref = (jnp.pi / safe) ** 1.5 * K
            T3 = [_os_table(la, lb + 2, PA[x], PB[x], i2g) for x in range(3)]

            def gathers(shift_axis, delta):
                out = []
                for x in range(3):
                    bidx = jb[x] + (delta if x == shift_axis else 0)
                    out.append(T3[x][ia[x][:, None],
                                     np.maximum(bidx, 0)[None, :]])
                return out[0] * out[1] * out[2]         # (nca, ncb)

            S = gathers(-1, 0)
            up = sum(gathers(x, 2) for x in range(3))
            down = sum((jb[x] * (jb[x] - 1))[None, :] * gathers(x, -2)
                       for x in range(3))
            T = (be * (2 * Lb + 3)[None, :] * S
                 - 2.0 * be ** 2 * up - 0.5 * down)
            return pref * S, pref * T

        Sv, Tv = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        S = jnp.einsum("ijab,ai,bj->ab", Sv, ca, cb)
        T = jnp.einsum("ijab,ai,bj->ab", Tv, ca, cb)
        return S, T

    return one_pair


@lru_cache(maxsize=4096)
def _compiled_st_kernel(la, lb, anga, angb, chunk_size):
    """Jitted, cached overlap+kinetic batch kernel (numpy closure constants;
    see the eri3c cache note)."""
    kern = _make_st_kernel(la, lb, anga, angb)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 6,
                                chunk_size=chunk_size))


def overlap_kinetic_bucketed(basis, plan=None, chunk=4096):
    """(S, T) overlap and kinetic matrices in one shell-class-bucketed pass.

    The kinetic matrix is written via the same index-swap symmetry as the
    other builds; its per-element formula is b-sided, so the swapped write
    holds to the same rounding the flat engine's own T = T.T asymmetry has.
    """
    _check_orbital_l(basis)
    if plan is None:
        plan = plan_pairs(basis)
    nao, classes = plan
    cen = basis.centers
    S = jnp.zeros((nao, nao), dtype=cen.dtype)
    T = jnp.zeros((nao, nao), dtype=cen.dtype)
    for (la, lb, anga, angb, ra, rb, nprims) in classes:
        npa, npb = nprims
        ra = np.asarray(ra); rb = np.asarray(rb)
        nca = (la + 1) * (la + 2) // 2
        ncb = (lb + 1) * (lb + 2) // 2
        ca = basis.coefficients[ra[:, None] + np.arange(nca)][:, :, :npa]
        cb = basis.coefficients[rb[:, None] + np.arange(ncb)][:, :, :npb]
        fn = _compiled_st_kernel(la, lb, anga, angb, min(chunk, ra.shape[0]))
        Sv, Tv = fn(cen[ra], cen[rb],
                    basis.exponents[ra][:, :npa],
                    basis.exponents[rb][:, :npb], ca, cb)
        i_idx = ra[:, None, None] + np.arange(nca)[:, None]
        j_idx = rb[:, None, None] + np.arange(ncb)[None, :]
        i_idx, j_idx = (np.broadcast_to(x, Sv.shape)
                        for x in (i_idx, j_idx))
        for out_i, vals in ((0, Sv), (1, Tv)):
            tgt = S if out_i == 0 else T
            tgt = tgt.at[i_idx.ravel(), j_idx.ravel()].set(vals.reshape(-1))
            tgt = tgt.at[j_idx.ravel(), i_idx.ravel()].set(vals.reshape(-1))
            if out_i == 0:
                S = tgt
            else:
                T = tgt
    if basis.cart2sph is not None:
        C = basis.cart2sph
        S = C.T @ S @ C
        T = C.T @ T @ C
    return S, T


def _make_eri2c_kernel(la, lb, anga, angb, omega=None):
    """2-center Coulomb kernel for one (la, lb) aux-shell class.

    (P|Q) = 2 pi^{5/2} / (a b sqrt(a+b)) sum_stq Fx_s Fy_r Fz_q R_srq with
    single-center E tables on both sides, matching _eri2c_primitive.
    """
    mt = max(la + lb + 1, 2)
    conv = np.zeros((mt, mt, mt))
    for t in range(mt):
        for tau in range(mt - t):
            conv[t, tau, t + tau] = 1.0
    sign = np.asarray([(-1.0) ** t for t in range(mt)])
    anga, angb = np.asarray(anga), np.asarray(angb)
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])

    def one_pair(A, B, ea, eb, ca, cb):
        AB = A - B

        def per_ab(al, be):
            sa = jnp.where(al == 0.0, 1.0, al)
            sb = jnp.where(be == 0.0, 1.0, be)
            rho = sa * sb / (sa + sb)
            pref = 2.0 * jnp.pi ** 2.5 / (sa * sb * jnp.sqrt(sa + sb))
            Ea = [_Ec_table(la, al, mt) for _ in range(3)]
            Eb = [_Ec_table(lb, be, mt) * sign for _ in range(3)]
            R = _hermite_table(rho, AB, mt, omega)
            G = [jnp.einsum("it,ju,tus->ijs", Ea[x], Eb[x], conv)
                 for x in range(3)]
            GX = G[0][ia[0][:, None], jb[0][None, :], :]
            GY = G[1][ia[1][:, None], jb[1][None, :], :]
            GZ = G[2][ia[2][:, None], jb[2][None, :], :]
            return pref * jnp.einsum("abs,abr,abq,srq->ab", GX, GY, GZ, R)

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        return jnp.einsum("ijab,ai,bj->ab", vals, ca, cb)

    return one_pair


@lru_cache(maxsize=4096)
def _compiled_eri2c_kernel(la, lb, anga, angb, omega, chunk_size):
    """Jitted, cached batch kernel (numpy closure constants; see the eri3c
    cache note)."""
    kern = _make_eri2c_kernel(la, lb, anga, angb, omega)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 6,
                                chunk_size=chunk_size))


def eri2c_matrix_bucketed(aux_basis, omega=None, plan=None, chunk=4096):
    """(naux, naux) 2-center Coulomb metric via shell-class buckets.

    Drop-in for the flat builder (both kernels, same cart2sph convention,
    differentiable w.r.t. ``centers``); ``plan`` is :func:`plan_pairs` of the
    auxiliary basis, derived here when the metadata is concrete.
    """
    if plan is None:
        plan = plan_pairs(aux_basis)
    naux, classes = plan
    cen = aux_basis.centers
    out = jnp.zeros((naux, naux), dtype=cen.dtype)
    for (la, lb, anga, angb, ra, rb, nprims) in classes:
        npa, npb = nprims
        ra = np.asarray(ra); rb = np.asarray(rb)
        nca = (la + 1) * (la + 2) // 2
        ncb = (lb + 1) * (lb + 2) // 2
        ca = aux_basis.coefficients[ra[:, None] + np.arange(nca)][:, :, :npa]
        cb = aux_basis.coefficients[rb[:, None] + np.arange(ncb)][:, :, :npb]
        fn = _compiled_eri2c_kernel(la, lb, anga, angb, omega,
                                    min(chunk, ra.shape[0]))
        vals = fn(cen[ra], cen[rb],
                  aux_basis.exponents[ra][:, :npa],
                  aux_basis.exponents[rb][:, :npb], ca, cb)
        i_idx = ra[:, None, None] + np.arange(nca)[:, None]
        j_idx = rb[:, None, None] + np.arange(ncb)[None, :]
        i_idx, j_idx = (np.broadcast_to(x, vals.shape)
                        for x in (i_idx, j_idx))
        out = out.at[i_idx.ravel(), j_idx.ravel()].set(vals.reshape(-1))
        # symmetry via index swap (same rule as the 3-center scatter)
        out = out.at[j_idx.ravel(), i_idx.ravel()].set(vals.reshape(-1))
    if aux_basis.cart2sph is not None:
        C = aux_basis.cart2sph
        out = C.T @ out @ C
    return out


def eri3c_matrix_bucketed(basis, aux_basis, omega=None, plan=None,
                          chunk=4096):
    """(nao, nao, naux) 3-center ERI tensor via shell-class buckets.

    Drop-in for the flat builder: same value to machine precision (both
    Coulomb and erf-attenuated kernels), same cart2sph convention,
    differentiable w.r.t. both ``centers`` arrays. ``plan`` is the static
    skeleton from :func:`plan_eri3c`; when None it is derived here, which
    requires concrete (non-traced) basis metadata.
    """
    if plan is None:
        plan = plan_eri3c(basis, aux_basis)
    nao, naux, classes = plan
    cen, cen_aux = basis.centers, aux_basis.centers
    out = jnp.zeros((nao, nao, naux), dtype=cen.dtype)
    for (la, lb, lc, anga, angb, angc, ra, rb, rc, nprims) in classes:
        npa, npb, npc = nprims
        ra = np.asarray(ra); rb = np.asarray(rb); rc = np.asarray(rc)
        nca = (la + 1) * (la + 2) // 2
        ncb = (lb + 1) * (lb + 2) // 2
        ncc = (lc + 1) * (lc + 2) // 2
        # every array is gathered from the (possibly traced) BasisData
        # leaves with static indices; rows are zero-padded past each
        # shell's true contraction length, so the class-level primitive
        # trim is a static slice
        ca = basis.coefficients[ra[:, None] + np.arange(nca)][:, :, :npa]
        cb = basis.coefficients[rb[:, None] + np.arange(ncb)][:, :, :npb]
        cc = aux_basis.coefficients[rc[:, None] + np.arange(ncc)][:, :, :npc]
        fn = _compiled_class_kernel(la, lb, lc, anga, angb, angc, omega,
                                    min(chunk, ra.shape[0]))
        vals = fn(
            cen[ra], cen[rb], cen_aux[rc],
            basis.exponents[ra][:, :npa], basis.exponents[rb][:, :npb],
            aux_basis.exponents[rc][:, :npc],
            ca, cb, cc,
        )   # (ntrip, nca, ncb, ncc)
        i_idx = ra[:, None, None, None] + np.arange(nca)[:, None, None]
        j_idx = rb[:, None, None, None] + np.arange(ncb)[None, :, None]
        k_idx = rc[:, None, None, None] + np.arange(ncc)[None, None, :]
        i_idx, j_idx, k_idx = (np.broadcast_to(x, vals.shape)
                               for x in (i_idx, j_idx, k_idx))
        out = out.at[i_idx.ravel(), j_idx.ravel(), k_idx.ravel()].set(
            vals.reshape(-1))
        # bra symmetry: writing vals[t,a,b,c] at (j0+b, i0+a, k0+c) IS the
        # transpose; swap the index arrays and keep the value order (a value
        # transpose would double-transpose against the (a, b) ravel).
        out = out.at[j_idx.ravel(), i_idx.ravel(), k_idx.ravel()].set(
            vals.reshape(-1))
    if basis.cart2sph is not None:
        Cs = basis.cart2sph
        out = jnp.einsum("ip,pqk->iqk", Cs.T, out)
        out = jnp.einsum("jq,iqk->ijk", Cs.T, out)
    if aux_basis.cart2sph is not None:
        Ca = aux_basis.cart2sph
        out = jnp.einsum("kr,ijr->ijk", Ca.T, out)
    return out
