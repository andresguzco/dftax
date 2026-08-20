"""Pure-JAX 4-center electron repulsion integrals (μν|λσ) via McMurchie-Davidson.

Four-center ERI (the "full" two-electron integral, no density fitting):

    (μν|λσ) = ∫∫ χ_μ(r₁) χ_ν(r₁) (1/|r₁-r₂|) χ_λ(r₂) χ_σ(r₂) dr₁ dr₂

This is the exact Coulomb kernel. The rest of the engine (eri2c/eri3c) only
provides the *density-fitted* approximation; this module is the reference
"full ERI" path so the two can be compared head-to-head.

It reuses the exact same primitives (already PySCF-validated) as the
3-center path (``eri3c.py``):
- ``_md_E_coefficients_1d``  (two-center Hermite expansion, bra AND ket)
- ``_hermite_coulomb``       (Hermite Coulomb integrals R_{tuv})
- ``_SIGN``, ``_MAX_T``

The only change vs. ``eri3c`` is that the ket is now a genuine two-center GTO
*pair* (λσ) rather than a single auxiliary function, so the ket Hermite
expansion uses ``_md_E_coefficients_1d`` (with K_CD and a second product
center Q) instead of ``_single_center_E_1d``.

Formula (per primitive quadruple):

    (ab|cd) = K_AB K_CD · (2π^{5/2})/(γ_ab·γ_cd·√(γ_ab+γ_cd))
              × Σ_{tuv,τυφ} E^{ab}_t E^{ab}_u E^{ab}_v · E^{cd}_τ E^{cd}_υ E^{cd}_φ
              × (-1)^{τ+υ+φ} · R_{t+τ,u+υ,v+φ}(ρ, P_ab - Q_cd)

Both builders exploit the 8-fold permutational symmetry of (μν|λσ): only the
~N⁴/8 canonical unique quartets are evaluated; the rest are filled by indexing.

WARNING: the explicit (nao,nao,nao,nao) tensor is still O(N⁴) in memory, so only
build it for small systems / validation. For energies on larger systems use
``coulomb_j_4c`` / ``coulomb_energy_4c`` which contract the density matrix on
the fly and never materialise the full tensor (O(N²) memory).
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Float, Array, Scalar

from dftax.energy.gto import BasisData
from dftax.utils.vmap import vmap as chunked_vmap
from dftax.integrals.eri3c import (
    _md_E_coefficients_1d,
    _hermite_coulomb,
    _MAX_L,
    _MAX_T,
    _MAX_M,
)


# ---------------------------------------------------------------------------
# Primitive 4-center ERI
# ---------------------------------------------------------------------------

def _eri4c_primitive(alpha, A, ang_a, beta, B, ang_b,
                     gamma, C, ang_c, delta, D, ang_d,
                     max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M, omega=None):
    """4-center ERI (ab|cd) for a single set of primitive GTOs.

    Args:
        alpha, A, ang_a: exponent, center (3,), angular (3,) for bra function a
        beta,  B, ang_b: exponent, center (3,), angular (3,) for bra function b
        gamma, C, ang_c: exponent, center (3,), angular (3,) for ket function c
        delta, D, ang_d: exponent, center (3,), angular (3,) for ket function d
        max_l, max_t, max_m: recursion sizes (per-molecule; default = g cap).
    """
    gamma_ab = alpha + beta
    gamma_cd = gamma + delta
    safe_gab = jnp.where(gamma_ab == 0.0, 1.0, gamma_ab)
    safe_gcd = jnp.where(gamma_cd == 0.0, 1.0, gamma_cd)

    P = (alpha * A + beta * B) / safe_gab          # bra product center
    Q = (gamma * C + delta * D) / safe_gcd         # ket product center
    AB = A - B
    CD = C - D
    PQ = P - Q

    K_AB = jnp.exp(-alpha * beta / safe_gab * jnp.sum(AB ** 2))
    K_CD = jnp.exp(-gamma * delta / safe_gcd * jnp.sum(CD ** 2))
    rho = safe_gab * safe_gcd / (safe_gab + safe_gcd)
    prefactor = (K_AB * K_CD * 2.0 * jnp.pi ** 2.5
                 / (safe_gab * safe_gcd * jnp.sqrt(safe_gab + safe_gcd)))

    # E-coefficients for bra pair (μν): two-center
    Ex_ab = _md_E_coefficients_1d(ang_a[0], ang_b[0], alpha, beta, AB[0], max_l, max_t)
    Ey_ab = _md_E_coefficients_1d(ang_a[1], ang_b[1], alpha, beta, AB[1], max_l, max_t)
    Ez_ab = _md_E_coefficients_1d(ang_a[2], ang_b[2], alpha, beta, AB[2], max_l, max_t)

    # E-coefficients for ket pair (λσ): two-center (this is the eri3c change)
    Ex_cd = _md_E_coefficients_1d(ang_c[0], ang_d[0], gamma, delta, CD[0], max_l, max_t)
    Ey_cd = _md_E_coefficients_1d(ang_c[1], ang_d[1], gamma, delta, CD[1], max_l, max_t)
    Ez_cd = _md_E_coefficients_1d(ang_c[2], ang_d[2], gamma, delta, CD[2], max_l, max_t)

    # Hermite Coulomb integrals with reduced exponent ρ
    R = _hermite_coulomb(rho, PQ, max_t, max_m, omega)

    # Combined E-coefficients via convolution with sign factor (-1)^τ
    sign = (-1.0) ** jnp.arange(max_t)
    F_x = jnp.convolve(Ex_ab, Ex_cd * sign, mode='full')[:max_t]
    F_y = jnp.convolve(Ey_ab, Ey_cd * sign, mode='full')[:max_t]
    F_z = jnp.convolve(Ez_ab, Ez_cd * sign, mode='full')[:max_t]

    result = jnp.einsum("s,r,q,srq->", F_x, F_y, F_z, R)
    return prefactor * result


# ---------------------------------------------------------------------------
# Contracted 4-center ERI (sum over all primitive quadruples)
# ---------------------------------------------------------------------------

def _contracted_eri4c(ea, ca, ra, la, eb, cb, rb, lb,
                      ec, cc, rc, lc, ed, cd, rd, ld,
                      max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M, omega=None):
    """Contracted 4-center ERI summed over all primitive quadruples.

    The bra primitives (a, b) are reduced with ``lax.fori_loop`` (sequential) while
    the ket primitives (c, d) stay vectorized with ``vmap``. The fully-nested-``vmap``
    form fuses into a ``(chunk · nprim⁴ · max_t³)`` intermediate that XLA materializes
    (~max_t⁸ per quartet, roughly 88 GB and a multi-minute compile at cc-pVDZ); scanning the bra
    pair drops the fused intermediate by ``nprim²`` (to ``nprim² · max_t³``), so the
    exact path compiles in seconds and fits in memory, with identical results."""
    npr = ea.shape[0]

    def _ket_sum(a_exp, a_coeff, b_exp, b_coeff):
        # Σ_{c,d} c·d·(ab|cd) for one fixed bra primitive pair; ket stays vectorized.
        def _prim_c(c_exp, c_coeff):
            def _prim_d(d_exp, d_coeff):
                return (d_coeff * _eri4c_primitive(a_exp, ra, la, b_exp, rb, lb,
                                                   c_exp, rc, lc, d_exp, rd, ld,
                                                   max_l, max_t, max_m, omega))
            return c_coeff * jnp.sum(jax.vmap(_prim_d)(ed, cd))
        return a_coeff * b_coeff * jnp.sum(jax.vmap(_prim_c)(ec, cc))

    def _loop_a(ia, acc):
        def _loop_b(ib, acc):
            return acc + _ket_sum(ea[ia], ca[ia], eb[ib], cb[ib])
        return lax.fori_loop(0, npr, _loop_b, acc)

    return lax.fori_loop(0, npr, _loop_a, jnp.zeros((), dtype=ca.dtype))


def _element(basis: BasisData, i, j, k, l, max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M,
             omega=None):
    return _contracted_eri4c(
        basis.exponents[i], basis.coefficients[i], basis.centers[i], basis.angular[i],
        basis.exponents[j], basis.coefficients[j], basis.centers[j], basis.angular[j],
        basis.exponents[k], basis.coefficients[k], basis.centers[k], basis.angular[k],
        basis.exponents[l], basis.coefficients[l], basis.centers[l], basis.angular[l],
        max_l, max_t, max_m, omega,
    )


# ---------------------------------------------------------------------------
# 8-fold permutational symmetry (A4)
# ---------------------------------------------------------------------------
#
# (μν|λσ) is invariant under bra swap (μ↔ν), ket swap (λ↔σ), and bra↔ket swap,
# generating an orbit of up to 8 equal integrals. We therefore evaluate only the
# canonical representative of each orbit: i≥j, k≥l, and braidx≥ketidx where the
# pair index of (i,j) is row-major lower-triangular, i.e. ~N⁴/8 of the tuples,
# an 8× cut in the (expensive) `_element` calls. Reconstruction is pure indexing
# (gather/scatter), which stays cheap and is differentiable wrt the centers
# (forces grad through this).

def _unique_quartets(n: int):
    """Canonical 8-fold-unique AO quartets and the orbit map.

    Returns integer index lists ``(i, j, k, l)`` of the canonical quartets
    (``i≥j``, ``k≥l``, bra-pair index ≥ ket-pair index) plus ``qof`` of shape
    ``(n, n, n, n)`` giving, for every tuple ``(a, b, c, d)``, the index of its
    canonical quartet. Host-side (``n`` is static), so it bakes in as a constant.
    """
    pa, pb = np.tril_indices(n)                  # all pairs a≥b, (npair,)
    npair = pa.shape[0]
    qa, qb = np.tril_indices(npair)              # pair-of-pairs, braidx≥ketidx
    i, j = pa[qa], pb[qa]
    k, l = pa[qb], pb[qb]
    q = np.arange(i.shape[0])
    qof = np.empty((n, n, n, n), dtype=np.int32)
    perms = (
        (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
        (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i),
    )
    for a, b, c, d in perms:                     # every tuple is in exactly one
        qof[a, b, c, d] = q                      # orbit; degenerate writes agree
    return i, j, k, l, qof


# ---------------------------------------------------------------------------
# Cauchy-Schwarz screening (A5)
# ---------------------------------------------------------------------------
#
# |(μν|λσ)| ≤ √(μν|μν) · √(λσ|λσ) =: Q_μν · Q_λσ. Quartets whose bound is below a
# threshold contribute negligibly and can be skipped entirely, which turns the
# nominal O(N⁴) work into ~O(N²) for extended systems (the number of pairs with
# non-negligible Q grows only linearly once atoms stop overlapping). The number
# of surviving quartets is data-dependent, so screening must be resolved on
# concrete values *outside* jit (the basis must be materialised): we build the
# significant-quartet list + orbit map host-side, then feed them to the jitted
# tensor build as a fixed (non-differentiable) sparsity pattern. The retained
# integrals are still differentiated exactly, so this composes with forces.

def schwarz_diagonal(basis: BasisData, chunk: int = 256) -> Float[Array, "npair"]:
    """Schwarz factors ``Q_p = √(ab|ab)`` for each unique pair ``p=(a,b)``, a≥b.

    Cartesian (pre-cart2sph) pairs, ordered like ``np.tril_indices(n)``.
    """
    L = basis.max_l
    ml, mt, mm = L + 1, 4 * L + 1, 4 * L + 1
    n = basis.centers.shape[0]
    pa, pb = np.tril_indices(n)
    pairs = jnp.stack([jnp.asarray(pa), jnp.asarray(pb)], axis=1)
    diag = chunked_vmap(
        lambda p: _element(basis, p[0], p[1], p[0], p[1], ml, mt, mm),
        chunk_size=chunk,
    )(pairs)                                          # (npair,)
    return jnp.sqrt(jnp.maximum(diag, 0.0))


def screened_quartets(basis: BasisData, thresh: float, chunk: int = 256):
    """Significant canonical quartets under the Cauchy-Schwarz bound.

    Returns ``(quartets, qof)`` where ``quartets`` is an ``(nq, 4)`` int array of
    the canonical 8-fold-unique quartets with ``Q_bra·Q_ket > thresh``, and
    ``qof`` of shape ``(n, n, n, n)`` maps every tuple ``(a,b,c,d)`` to its
    quartet's index, or to ``nq`` (a zero sentinel) when screened out. Host-side
    (basis must be concrete); pairs are prescreened first so the enumeration is
    O(n_significant²), not O(n_pair²).
    """
    n = basis.centers.shape[0]
    Q = np.asarray(schwarz_diagonal(basis, chunk))    # (npair,) concrete
    pa, pb = np.tril_indices(n)
    qmax = float(Q.max()) if Q.size else 0.0
    # Pair prescreen: a pair contributes to *some* quartet only if even its
    # strongest partner clears the bound. Drop the rest before enumerating.
    sig = np.nonzero(Q * qmax > thresh)[0]            # significant pair indices
    ii, jj = np.tril_indices(sig.shape[0])            # pair-of-pairs, braidx≥ketidx
    pbra, pket = sig[ii], sig[jj]
    keep = Q[pbra] * Q[pket] > thresh
    pbra, pket = pbra[keep], pket[keep]
    qi, qj = pa[pbra], pb[pbra]
    qk, ql = pa[pket], pb[pket]
    nq = qi.shape[0]
    qof = np.full((n, n, n, n), nq, dtype=np.int32)   # sentinel = nq → zero value
    idx = np.arange(nq)
    perms = (
        (qi, qj, qk, ql), (qj, qi, qk, ql), (qi, qj, ql, qk), (qj, qi, ql, qk),
        (qk, ql, qi, qj), (ql, qk, qi, qj), (qk, ql, qj, qi), (ql, qk, qj, qi),
    )
    for a, b, c, d in perms:
        qof[a, b, c, d] = idx
    quartets = np.stack([qi, qj, qk, ql], axis=1).astype(np.int32)
    return quartets, qof


# ---------------------------------------------------------------------------
# Full (nao, nao, nao, nao) tensor builder: SMALL systems / validation only
# ---------------------------------------------------------------------------

def eri4c_matrix(
    basis: BasisData,
    chunk: int = 256,
    quartets=None,
    qof=None,
    omega: float | None = None,
    plan: tuple | None = None,
) -> Float[Array, "nao nao nao nao"]:
    """Full 4-center ERI tensor (μν|λσ) in the AO basis (spherical).

    O(N⁴) memory, intended for the exact Coulomb/exchange backend on small
    systems and for validation against PySCF int2e. The unscreened build runs
    through the shell-quartet class kernels (:func:`eri4c_matrix_bucketed`;
    ``plan`` is its static skeleton from :func:`plan_eri4c`, derived here when
    the basis metadata is concrete). ``quartets``/``qof`` supply a pre-screened
    canonical quartet list and its orbit map (see :func:`screened_quartets`),
    which selects the flat per-element path: only those quartets are evaluated
    and screened-out positions are exactly zero.
    """
    if quartets is None:
        return eri4c_matrix_bucketed(basis, omega=omega, plan=plan)
    return _eri4c_matrix_flat(basis, chunk, quartets, qof, omega)


def _eri4c_matrix_flat(
    basis: BasisData,
    chunk: int = 256,
    quartets=None,
    qof=None,
    omega: float | None = None,
) -> Float[Array, "nao nao nao nao"]:
    """Flat per-element reference build (one ``_element`` per canonical
    quartet, orbit-map gather). Kept for the Schwarz-screened quartet-list
    path and as the bucketed engine's A/B oracle."""
    L = int(basis.max_l)                         # size the recursion to the molecule
    if L > 6:
        raise ValueError(
            f"the integral engine supports orbital angular momentum up to i "
            f"(l=6); got l={L}."
        )
    ml, mt, mm = L + 1, 4 * L + 1, 4 * L + 1
    n = basis.centers.shape[0]
    if quartets is None:
        qi, qj, qk, ql, qof = _unique_quartets(n)
        quartets = jnp.stack(
            [jnp.asarray(qi), jnp.asarray(qj), jnp.asarray(qk), jnp.asarray(ql)],
            axis=1,
        )                                        # (nuniq, 4)
    vals = chunked_vmap(
        lambda q: _element(basis, q[0], q[1], q[2], q[3], ml, mt, mm, omega),
        chunk_size=chunk,
    )(jnp.asarray(quartets))                     # (nq,)
    # Append a zero sentinel so screened-out positions (qof == nq) read as 0.
    vals = jnp.concatenate([vals, jnp.zeros((1,), dtype=vals.dtype)])
    result = vals[jnp.asarray(qof).reshape(-1)].reshape(n, n, n, n)   # cartesian

    if basis.cart2sph is not None:
        C = basis.cart2sph  # (n_cart, n_sph)
        result = jnp.einsum("ai,abcd->ibcd", C, result)
        result = jnp.einsum("bj,ibcd->ijcd", C, result)
        result = jnp.einsum("ck,ijcd->ijkd", C, result)
        result = jnp.einsum("dl,ijkd->ijkl", C, result)
    return result


# ---------------------------------------------------------------------------
# Shell-quartet-class bucketed build (the unscreened engine)
# ---------------------------------------------------------------------------

def plan_eri4c(basis):
    """Static shell-quartet bucket plan for :func:`eri4c_matrix_bucketed`.

    Canonical 8-fold-unique shell quartets (bra pair ``ia<=jb``, ket pair
    ``kc<=ld``, bra-pair index <= ket-pair index) grouped by their
    ``(l_a, l_b, l_c, l_d)`` class with per-class primitive trims
    (contraction lengths join the key and are merged back where the padding
    is cheap; see
    :func:`~dftax.integrals.eri3c_bucketed._merge_padded_buckets`). Same
    contract as :func:`~dftax.integrals.eri3c_bucketed.plan_eri3c`: python
    ints in nested tuples, computed where the basis metadata is concrete,
    safe as a static jit argument.
    """
    from collections import defaultdict

    from dftax.integrals.eri3c_bucketed import (
        _check_orbital_l, _merge_classes, _shells,
    )

    _check_orbital_l(basis)
    shells, ang = _shells(basis.angular, basis.exponents)
    ns = len(shells)
    spairs = [(ia, jb) for ia in range(ns) for jb in range(ia, ns)]
    buckets = defaultdict(lambda: ([], [], [], []))
    nprims = {}
    for Pp, (ia, jb) in enumerate(spairs):
        la, ra, nca, npa = shells[ia]
        lb, rb, ncb, npb = shells[jb]
        for kc, ld_ in spairs[Pp:]:
            lc, rc, ncc, npc = shells[kc]
            ldl, rd, ncd, npd = shells[ld_]
            key = (la, lb, lc, ldl, npa, npb, npc, npd)
            b = buckets[key]
            b[0].append(ra); b[1].append(rb); b[2].append(rc); b[3].append(rd)
            nprims[key] = (npa, npb, npc, npd)

    def ang_tup(row0, ncomp):
        return tuple(tuple(int(x) for x in ang[row0 + i]) for i in range(ncomp))

    classes = []
    for (la, lb, lc, ldl), npr, rows in _merge_classes(buckets, nprims, 4, 4):
        rows_a, rows_b, rows_c, rows_d = rows
        ncs = tuple((l + 1) * (l + 2) // 2 for l in (la, lb, lc, ldl))
        classes.append((
            la, lb, lc, ldl,
            ang_tup(rows_a[0], ncs[0]), ang_tup(rows_b[0], ncs[1]),
            ang_tup(rows_c[0], ncs[2]), ang_tup(rows_d[0], ncs[3]),
            tuple(rows_a), tuple(rows_b), tuple(rows_c), tuple(rows_d),
            npr,
        ))
    return (int(np.asarray(basis.angular).shape[0]), tuple(classes))


def _make_eri4c_kernel(la, lb, lc, ld, anga, angb, angc, angd, omega=None):
    """Kernel for one (la, lb, lc, ld) shell-quartet class; one quartet ->
    (nca, ncb, ncc, ncd). Bra and ket are both genuine two-center pairs, so
    both sides use the two-center E tables; otherwise the same structure as
    the 3-center class kernel."""
    from dftax.integrals.eri3c_bucketed import _E_table, _hermite_table

    mt = max(la + lb + lc + ld + 1, 2)
    conv = np.zeros((mt, mt, mt))
    for t in range(mt):
        for tau in range(mt - t):
            conv[t, tau, t + tau] = 1.0
    sign = np.asarray([(-1.0) ** t for t in range(mt)])
    anga, angb = np.asarray(anga), np.asarray(angb)
    angc, angd = np.asarray(angc), np.asarray(angd)
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])
    kc = (angc[:, 0], angc[:, 1], angc[:, 2])
    ld_ = (angd[:, 0], angd[:, 1], angd[:, 2])

    def one_quartet(A, B, C, D, ea, eb, ec, ed, ca, cb, cc, cd):
        AB = A - B
        CD = C - D

        def per_ab(al, be):
            gab = al + be
            safeb = jnp.where(gab == 0.0, 1.0, gab)
            P = (al * A + be * B) / safeb
            K_ab = jnp.exp(-al * be / safeb * jnp.sum(AB ** 2))
            Eab = [_E_table(la, lb, al, be, AB[x], mt) for x in range(3)]

            def per_cd(ga, de):
                gcd = ga + de
                safek = jnp.where(gcd == 0.0, 1.0, gcd)
                Q = (ga * C + de * D) / safek
                K_cd = jnp.exp(-ga * de / safek * jnp.sum(CD ** 2))
                Ecd = [_E_table(lc, ld, ga, de, CD[x], mt) for x in range(3)]
                rho = safeb * safek / (safeb + safek)
                pref = (K_ab * K_cd * 2.0 * jnp.pi ** 2.5
                        / (safeb * safek * jnp.sqrt(safeb + safek)))
                R = _hermite_table(rho, P - Q, mt, omega)
                G = [jnp.einsum("ijt,klu,tus->ijkls", Eab[x], Ecd[x] * sign,
                                conv) for x in range(3)]
                GX = G[0][ia[0][:, None, None, None], jb[0][None, :, None, None],
                          kc[0][None, None, :, None], ld_[0][None, None, None, :], :]
                GY = G[1][ia[1][:, None, None, None], jb[1][None, :, None, None],
                          kc[1][None, None, :, None], ld_[1][None, None, None, :], :]
                GZ = G[2][ia[2][:, None, None, None], jb[2][None, :, None, None],
                          kc[2][None, None, :, None], ld_[2][None, None, None, :], :]
                return pref * jnp.einsum("abcds,abcdr,abcdq,srq->abcd",
                                         GX, GY, GZ, R)

            return jax.vmap(jax.vmap(per_cd, (None, 0)), (0, None))(ec, ed)

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        return jnp.einsum("ijklabcd,ai,bj,ck,dl->abcd", vals, ca, cb, cc, cd)

    return one_quartet


def _eri4c_class_chunk(la, lb, lc, ld, nprims):
    """Quartet batch size keeping one class chunk's fused intermediates near
    the shared bra budget (Hermite cube + the three component-gathered
    tables, times the vmapped primitive quartet)."""
    mt = max(la + lb + lc + ld + 1, 2)
    ncomp = 1
    for l in (la, lb, lc, ld):
        ncomp *= (l + 1) * (l + 2) // 2
    npa, npb, npc, npd = nprims
    per = npa * npb * npc * npd * (mt ** 3 + 3 * ncomp * mt)
    return max(1, int(2.5e8 // per))


@lru_cache(maxsize=4096)
def _compiled_eri4c_kernel(la, lb, lc, ld, anga, angb, angc, angd, omega,
                           chunk_size):
    """Jitted, cached batch kernel for one quartet class (numpy closure
    constants, hashable static metadata; see the eri3c cache note)."""
    kern = _make_eri4c_kernel(la, lb, lc, ld, anga, angb, angc, angd, omega)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 12,
                                chunk_size=chunk_size))


def eri4c_matrix_bucketed(
    basis: BasisData,
    omega: float | None = None,
    plan: tuple | None = None,
) -> Float[Array, "nao nao nao nao"]:
    """(nao, nao, nao, nao) ERI tensor via shell-quartet classes.

    Drop-in for the flat unscreened build (same value to machine precision,
    Coulomb and erf kernels), differentiable w.r.t. ``basis.centers``. The
    8-fold symmetry is realized by index-permuted block scatters of each
    class's quartet blocks (block values transposed to match each permuted
    corner; overlapping writes on degenerate orbits agree by permutational
    symmetry, and blocks within one scatter are disjoint because canonical
    quartets are distinct).
    """
    from dftax.integrals.eri3c_bucketed import _IDX, _gather_rows, _scatter_blocks

    if plan is None:
        plan = plan_eri4c(basis)
    nao, classes = plan
    cen = basis.centers
    out = jnp.zeros((nao, nao, nao, nao), dtype=cen.dtype)
    for (la, lb, lc, ldl, anga, angb, angc, angd,
         ra, rb, rc, rd, nprims) in classes:
        npa, npb, npc, npd = nprims
        ra = np.asarray(ra).astype(_IDX); rb = np.asarray(rb).astype(_IDX)
        rc = np.asarray(rc).astype(_IDX); rd = np.asarray(rd).astype(_IDX)
        nca, ncb, ncc, ncd = (
            (l + 1) * (l + 2) // 2 for l in (la, lb, lc, ldl)
        )
        ca = _gather_rows(basis.coefficients, ra, nca)[:, :, :npa]
        cb = _gather_rows(basis.coefficients, rb, ncb)[:, :, :npb]
        cc = _gather_rows(basis.coefficients, rc, ncc)[:, :, :npc]
        cd = _gather_rows(basis.coefficients, rd, ncd)[:, :, :npd]
        fn = _compiled_eri4c_kernel(
            la, lb, lc, ldl, anga, angb, angc, angd, omega,
            min(_eri4c_class_chunk(la, lb, lc, ldl, nprims), ra.shape[0]),
        )
        vals = fn(
            cen[ra], cen[rb], cen[rc], cen[rd],
            basis.exponents[ra][:, :npa], basis.exponents[rb][:, :npb],
            basis.exponents[rc][:, :npc], basis.exponents[rd][:, :npd],
            ca, cb, cc, cd,
        )                                # (nquart, nca, ncb, ncc, ncd)
        rows = (ra, rb, rc, rd)
        for perm in (
            (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
            (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
        ):
            starts = np.stack([rows[p] for p in perm], axis=1).astype(_IDX)
            out = _scatter_blocks(
                out, starts, vals.transpose((0,) + tuple(p + 1 for p in perm))
            )
    if basis.cart2sph is not None:
        Cs = basis.cart2sph
        out = jnp.einsum("ai,abcd->ibcd", Cs, out)
        out = jnp.einsum("bj,ibcd->ijcd", Cs, out)
        out = jnp.einsum("ck,ijcd->ijkd", Cs, out)
        out = jnp.einsum("dl,ijkd->ijkl", Cs, out)
    return out


# ---------------------------------------------------------------------------
# Memory-light Coulomb J matrix / energy via on-the-fly density contraction
# ---------------------------------------------------------------------------

def _safe_eri_chunk(max_l, requested: int = 256, budget_gb: float = 6.0) -> int:
    """Quartet chunk that keeps the per-chunk ERI-build fusion under ``budget_gb``.

    With the bra primitive pair scanned (see :func:`_contracted_eri4c`), the per-quartet
    fused intermediate is the ket vmap ``nprim² · max_t³`` rather than the old
    ``nprim⁴ · max_t³``, an ``nprim²`` (≈81× at ``max_t=9``) reduction, dropping the
    cost exponent from ``max_t⁸`` to ``~max_t⁶``. Modelling the per-quartet cost as
    ``0.0042 GB·(max_t/9)⁶`` keeps high angular momenta within memory; at cc-pVDZ
    (``max_t=9``) this restores the full ``chunk=256``, and the exact path now compiles
    in seconds instead of OOMing / multi-minute compiles. L≤1 keeps the full 256.
    """
    mt = 4 * int(max_l) + 1
    per_unit_gb = 0.0042 * (mt / 9.0) ** 6
    cap = max(1, int(budget_gb / per_unit_gb))
    return int(min(requested, cap))


def significant_pairs(basis: BasisData, thresh: float, chunk: int = 256):
    """Significant cartesian bra pairs ``i<=j`` under a relative Schwarz cutoff.

    A density-fitting bra pair ``(μν)`` enters the 3-center ``(μν|P)`` only through
    its own charge density, so ``|(μν|P)| <= √(μν|μν)·√(P|P) = Q_μν·Q_P``. Pairs whose
    self-Schwarz ``Q_μν`` falls below ``thresh`` (relative to the largest) are
    negligible for **every** auxiliary ``P`` and can be dropped; for extended systems
    the survivors are O(N), not O(N²). Geometry-only (density-independent), so the
    list is fixed across the SCF and is built once, host-side (basis must be concrete).

    Returns ``(pi, pj, w)``: the kept pair indices (``i<=j``) and the i<->j symmetry
    weight ``w = 2`` (``i<j``) or ``1`` (``i==j``), so ``Σ_{all i,j} = Σ_{i<=j} w·(·)``.
    """
    n = basis.centers.shape[0]
    Q = np.asarray(schwarz_diagonal(basis, chunk))       # √(ij|ij), tril (i>=j) order
    pa, pb = np.tril_indices(n)
    qmax = float(Q.max()) if Q.size else 0.0
    keep = Q > thresh * qmax
    pi, pj = pa[keep], pb[keep]
    w = np.where(pi == pj, 1.0, 2.0)
    return (jnp.asarray(pi), jnp.asarray(pj),
            jnp.asarray(w, dtype=basis.centers.dtype))   # match the working precision


def coulomb_j_4c(
    P: Float[Array, "nao nao"],
    basis: BasisData,
    chunk: int | None = None,
) -> Float[Array, "nao nao"]:
    """Coulomb (Hartree) matrix J_μν = Σ_λσ (μν|λσ) P_λσ via full 4-center ERIs.

    Contracts the density on the fly, so the (nao⁴) tensor is never stored: only
    O(nao²) memory. Evaluates only the ~N⁴/8 canonical 8-fold-unique quartets:
    each contributes to its bra pair (and, off the pair-of-pairs diagonal, its
    ket pair), with the ket-pair density pre-scaled by ``2−δ`` to fold in the
    λ↔σ symmetry. P is in the spherical AO basis. ``chunk=None`` picks an
    L-aware safe chunk (see :func:`_safe_eri_chunk`).

    NOTE (scale): the unique pair-of-pairs list is enumerated host-side and is
    O(N⁴), the same order as the exact-J device work it drives (exact exchange is
    intrinsically O(N⁴)), so it does not change the asymptotics, but it does cap the
    practical exact-path size on host RAM / trace time. This is the *exact* path;
    the O(N²)-host large-system path is density fitting (``significant_pairs`` +
    streamed RI-J/RI-K), not this builder.
    """
    if chunk is None:
        chunk = _safe_eri_chunk(basis.max_l)
    # Move P into Cartesian space: Ptil_cd = Σ_λσ C_cλ P_λσ C_dσ
    if basis.cart2sph is not None:
        C = basis.cart2sph                       # (n_cart, n_sph)
        Ptil = C @ P @ C.T                       # (n_cart, n_cart)
    else:
        Ptil = P

    L = basis.max_l
    ml, mt, mm = L + 1, 4 * L + 1, 4 * L + 1
    n = basis.centers.shape[0]

    pa, pb = np.tril_indices(n)                  # pairs a≥b, (npair,)
    npair = pa.shape[0]
    qa, qb = np.tril_indices(npair)              # pair-of-pairs, braidx≥ketidx
    quartets = jnp.stack(
        [jnp.asarray(pa[qa]), jnp.asarray(pb[qa]),
         jnp.asarray(pa[qb]), jnp.asarray(pb[qb])],
        axis=1,
    )                                            # (nuniq, 4)
    vals = chunked_vmap(
        lambda q: _element(basis, q[0], q[1], q[2], q[3], ml, mt, mm),
        chunk_size=chunk,
    )(quartets)                                  # (nuniq,)

    # Pair density with the λ↔σ factor folded in (2 off-diagonal, 1 on-diagonal).
    scale = jnp.where(jnp.asarray(pa == pb), 1.0, 2.0)        # (npair,)
    Dp = Ptil[jnp.asarray(pa), jnp.asarray(pb)] * scale       # (npair,)

    qa_j, qb_j = jnp.asarray(qa), jnp.asarray(qb)
    offdiag = jnp.asarray(qa != qb)
    J_pair = jnp.zeros(npair, dtype=Ptil.dtype)
    J_pair = J_pair.at[qa_j].add(vals * Dp[qb_j])            # bra-pair J += v·D_ket
    J_pair = J_pair.at[qb_j].add(                            # ket-pair J += v·D_bra
        jnp.where(offdiag, vals * Dp[qa_j], 0.0)
    )

    pa_j, pb_j = jnp.asarray(pa), jnp.asarray(pb)
    J_cart = jnp.zeros((n, n), dtype=Ptil.dtype)
    J_cart = J_cart.at[pa_j, pb_j].set(J_pair)
    J_cart = J_cart.at[pb_j, pa_j].set(J_pair)               # symmetric

    if basis.cart2sph is not None:
        J = basis.cart2sph.T @ J_cart @ basis.cart2sph
    else:
        J = J_cart
    return J


def coulomb_energy_4c(
    P: Float[Array, "nao nao"],
    basis: BasisData,
    **kwargs,
) -> Scalar:
    """Hartree energy E_J = ½ Σ_μνλσ P_μν (μν|λσ) P_λσ via full 4-center ERIs."""
    J = coulomb_j_4c(P, basis, **kwargs)
    return 0.5 * jnp.sum(P * J)


def exchange_k_4c(
    P: Float[Array, "nao nao"],
    basis: BasisData,
    i_chunk: int = 8,
    k_chunk: int = 16,
) -> Float[Array, "nao nao"]:
    """Exact exchange matrix K_μν = Σ_λσ (μλ|νσ) P_λσ via full 4-center ERIs.

    Streams the contraction so the (nao⁴) tensor is never stored: the first index
    ``μ`` is mapped in chunks of ``i_chunk`` and, within each, the summed index
    ``λ`` is streamed in chunks of ``k_chunk``, with peak memory O(i_chunk·k_chunk·nao²),
    rematerialized in the backward pass. The per-quartet kernel ``_element`` is the
    same PySCF-validated primitive as the J path; P is in the spherical AO basis.

    NB: unlike ``coulomb_j_4c`` this baseline does not yet fold the 8-fold
    permutational symmetry (the exchange index pattern does not align with the
    bra/ket pair grouping): correct, but ~Nᴬ compute. Screening / symmetry folding
    is the next optimization.
    """
    if basis.cart2sph is not None:
        C = basis.cart2sph                                   # (n_cart, n_sph)
        Pc = C @ P @ C.T                                     # (n_cart, n_cart)
    else:
        Pc = P

    L = basis.max_l
    ml, mt, mm = L + 1, 4 * L + 1, 4 * L + 1
    n = basis.centers.shape[0]
    idx = jnp.arange(n)
    i_chunk = _safe_eri_chunk(L, requested=i_chunk)          # shrink for high angular momentum

    def k_row(i):                                            # K_cart[i, :]
        def contrib(k):                                      # contribution of index λ=k
            Mjl = jax.vmap(                                  # (n_ν, n_σ) = (i k | j l)
                lambda j: jax.vmap(
                    lambda l: _element(basis, i, k, j, l, ml, mt, mm)
                )(idx)
            )(idx)
            return Mjl @ Pc[k]                               # Σ_σ (ik|jσ) P[k,σ] -> (n_ν,)

        # Σ_λ streamed in k-chunks (peak (k_chunk, n²)).
        return jnp.sum(chunked_vmap(contrib, chunk_size=k_chunk)(idx), axis=0)

    Kc = chunked_vmap(k_row, chunk_size=i_chunk)(idx)        # (n_cart, n_cart)
    if basis.cart2sph is not None:
        return basis.cart2sph.T @ Kc @ basis.cart2sph
    return Kc
