"""Prototype: shell-class-bucketed eri3c build (perf follow-up 1, phase 2).

One jitted kernel per (la, lb, lc) class, right-sized recursions
(mt = la+lb+lc+1), Hermite R and E tables built once per primitive shell
triple and indexed by every cartesian component, primitives trimmed to the
class, bra symmetry exploited (i <= j shell pairs, index-swapped scatter).

Validated against dftax.integrals.eri3c.eri3c_matrix at machine precision
(water/sto-3g and water/def2-svp, plain and omega kernels); see
validate_bucketed.py and bench_bucketed.py.
"""

from collections import defaultdict
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from dftax.energy.boys import boys
from dftax.utils.vmap import vmap as chunked_vmap


# ---------------------------------------------------------------------------
# Eager shell tables (numpy, KS.__init__-time per the traced-vs-eager rule)
# ---------------------------------------------------------------------------

def shell_table(basis):
    """Group flat cartesian rows into shells.

    Returns a list of dicts: l, row0 (first cartesian row), ncomp, center
    (3,), exps (nprim, trimmed), coeffs (ncomp, nprim), ang (ncomp, 3).
    """
    cen = np.asarray(basis.centers)
    ex = np.asarray(basis.exponents)
    co = np.asarray(basis.coefficients)
    ang = np.asarray(basis.angular)
    ltot = ang.sum(1)
    shells = []
    i = 0
    while i < cen.shape[0]:
        j = i + 1
        while (j < cen.shape[0] and ltot[j] == ltot[i]
               and np.array_equal(cen[j], cen[i])
               and np.array_equal(ex[j], ex[i])):
            j += 1
        nprim = max(1, int((ex[i] != 0).sum()))
        shells.append(dict(
            l=int(ltot[i]), row0=i, ncomp=j - i, center=cen[i],
            exps=ex[i, :nprim], coeffs=co[i:j, :nprim], ang=ang[i:j],
        ))
        i = j
    return shells


def bucket_triples(bra_shells, aux_shells):
    """Group (bra_pair, aux_shell) triples by (la, lb, lc); i <= j bra pairs.

    Returns {class_key: dict of stacked numpy arrays + static metadata}.
    """
    buckets = defaultdict(lambda: defaultdict(list))
    for ia, sa in enumerate(bra_shells):
        for sb in bra_shells[ia:]:
            for sc in aux_shells:
                key = (sa["l"], sb["l"], sc["l"])
                b = buckets[key]
                b["A"].append(sa["center"]); b["B"].append(sb["center"])
                b["C"].append(sc["center"])
                b["ea"].append(sa["exps"]); b["eb"].append(sb["exps"])
                b["ec"].append(sc["exps"])
                b["ca"].append(sa["coeffs"]); b["cb"].append(sb["coeffs"])
                b["cc"].append(sc["coeffs"])
                b["rows"].append((sa["row0"], sb["row0"], sc["row0"]))
                b["meta"] = (sa["ang"], sb["ang"], sc["ang"])
    out = {}
    for key, b in buckets.items():
        npa = max(e.shape[0] for e in b["ea"])
        npb = max(e.shape[0] for e in b["eb"])
        npc = max(e.shape[0] for e in b["ec"])

        def pad_e(es, n):
            return np.stack([np.pad(e, (0, n - e.shape[0])) for e in es])

        def pad_c(cs, n):
            return np.stack([np.pad(c, ((0, 0), (0, n - c.shape[1])))
                             for c in cs])

        out[key] = dict(
            A=np.stack(b["A"]), B=np.stack(b["B"]), C=np.stack(b["C"]),
            ea=pad_e(b["ea"], npa), eb=pad_e(b["eb"], npb),
            ec=pad_e(b["ec"], npc),
            ca=pad_c(b["ca"], npa), cb=pad_c(b["cb"], npb),
            cc=pad_c(b["cc"], npc),
            rows=np.asarray(b["rows"]), ang=b["meta"],
        )
    return out


# ---------------------------------------------------------------------------
# Right-sized table builders (mirror eri3c.py's recursions, return tables)
# ---------------------------------------------------------------------------

def _bump(row, X, i2g, mt):
    """One MD E-recursion step on a length-mt row (trace-time unrolled)."""
    left = jnp.concatenate([jnp.zeros(1), row[:-1]])
    right = jnp.concatenate([row[1:], jnp.zeros(1)])
    t_idx = jnp.arange(mt)
    return (X * row + i2g * jnp.where(t_idx > 0, left, 0.0)
            + jnp.where(t_idx + 1 < mt, (t_idx + 1) * right, 0.0))


def _E_table(la, lb, alpha, beta, XAB, mt):
    """Full (la+1, lb+1, mt) two-center E table, recursion unrolled at trace
    time: the trip counts are tiny static ints, and unrolling lets XLA fuse
    the whole table into one kernel instead of ~la*lb sequential launches
    (the launch latency dominated the fori_loop version by ~100x)."""
    gamma = alpha + beta
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    XPA = -beta * XAB / safe_gamma
    XPB = alpha * XAB / safe_gamma
    col = [jnp.zeros(mt).at[0].set(1.0)]           # E[i, 0] rows
    for _ in range(la):
        col.append(_bump(col[-1], XPA, i2g, mt))
    rows = [col]                                    # rows[j][i]
    for _ in range(lb):
        rows.append([_bump(r, XPB, i2g, mt) for r in rows[-1]])
    # stack to (la+1, lb+1, mt)
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
    """(mt, mt, mt) Hermite Coulomb integrals R^0_{t,u,v}, unrolled at trace
    time (mirrors dftax.integrals.eri3c._hermite_coulomb, whose fori_loops
    issue ~3*mt^2 sequential launches)."""
    T = rho * jnp.sum(RPC ** 2)
    neg2rho = -2.0 * rho
    if omega is None:
        base = [boys(m, T) for m in range(mt)]
    else:
        s = (omega * omega) / (omega * omega + rho)
        base = [s ** (m + 0.5) * boys(m, s * T) for m in range(mt)]
    # R[m][t][u][v] built as nested python lists of scalars, then stacked;
    # only t+u+v+m < mt entries are ever read.
    R = {(m, 0, 0, 0): neg2rho ** m * base[m] for m in range(mt)}

    def get(m, t, u, v):
        if t < 0 or u < 0 or v < 0:
            return 0.0
        return R[(m, t, u, v)]

    for m in range(mt - 2, -1, -1):
        top = mt - 1 - m
        for t in range(top):
            R[(m, t + 1, 0, 0)] = (RPC[0] * get(m + 1, t, 0, 0)
                                   + t * get(m + 1, t - 1, 0, 0))
        for u in range(top):
            for t in range(top - u):
                R[(m, t, u + 1, 0)] = (RPC[1] * get(m + 1, t, u, 0)
                                       + u * get(m + 1, t, u - 1, 0))
        for v in range(top):
            for u in range(top - v):
                for t in range(top - v - u):
                    R[(m, t, u, v + 1)] = (RPC[2] * get(m + 1, t, u, v)
                                           + v * get(m + 1, t, u, v - 1))
    zero = jnp.zeros(())
    return jnp.stack([
        jnp.stack([
            jnp.stack([R.get((0, t, u, v), zero) * jnp.ones(())
                       for v in range(mt)])
            for u in range(mt)])
        for t in range(mt)])


# ---------------------------------------------------------------------------
# The per-class kernel
# ---------------------------------------------------------------------------

def make_class_kernel(la, lb, lc, anga, angb, angc, omega=None):
    """Jitted kernel for one (la, lb, lc) class.

    anga/angb/angc: static (ncomp, 3) integer component tables.
    Returns fn(triple_data) -> (nca, ncb, ncc) for one triple.
    """
    # class-sized recursions; floor 2 because _hermite_coulomb's vectorized
    # t/u/v build assumes at least one off-origin Hermite index
    mt = max(la + lb + lc + 1, 2)
    # static one-hot s = t + tau convolution tensor and component indices
    conv = np.zeros((mt, mt, mt))
    for t in range(mt):
        for tau in range(mt - t):
            conv[t, tau, t + tau] = 1.0
    conv = jnp.asarray(conv)
    sign = jnp.asarray([(-1.0) ** t for t in range(mt)])
    ia = tuple(np.asarray(a) for a in (anga[:, 0], anga[:, 1], anga[:, 2]))
    jb = tuple(np.asarray(a) for a in (angb[:, 0], angb[:, 1], angb[:, 2]))
    kc = tuple(np.asarray(a) for a in (angc[:, 0], angc[:, 1], angc[:, 2]))

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
                # G[x][i', j', k', s] = sum_tau Et[i',j',t] Ec[k',tau] conv
                G = [jnp.einsum("ijt,ku,tus->ijks", Et[x], Ec[x], conv)
                     for x in range(3)]
                # gather components statically -> (nca, ncb, ncc, mt)
                GX = G[0][ia[0][:, None, None], jb[0][None, :, None],
                          kc[0][None, None, :], :]
                GY = G[1][ia[1][:, None, None], jb[1][None, :, None],
                          kc[1][None, None, :], :]
                GZ = G[2][ia[2][:, None, None], jb[2][None, :, None],
                          kc[2][None, None, :], :]
                return pref * jnp.einsum("abcs,abcr,abcq,srq->abc",
                                         GX, GY, GZ, R)

            return jax.vmap(per_c)(ec)          # (npc, nca, ncb, ncc)

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        # vals: (npa, npb, npc, nca, ncb, ncc); contract coefficients
        return jnp.einsum("ijkabc,ai,bj,ck->abc", vals, ca, cb, cc)

    return one_triple


@lru_cache(maxsize=4096)
def _compiled_class_kernel(la, lb, lc, anga, angb, angc, omega, chunk_size):
    """Jitted, cached batch kernel for one class.

    Without this cache every bucketed_eri3c call re-traces the unrolled
    graphs through Python; the tracing, not the GPU, was ~100% of the
    measured wall (69 s ethanol). ang tuples are hashable static metadata;
    jit keys the shapes, so repeat builds hit the compiled executable.
    """
    kern = make_class_kernel(la, lb, lc,
                             np.asarray(anga), np.asarray(angb),
                             np.asarray(angc), omega)
    # explicit in_axes: the utility's chunked path only batches the
    # arguments named by in_axes, and a bare 0 covers just the first
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 9,
                                chunk_size=chunk_size))


def _hashable_ang(a):
    return tuple(tuple(int(x) for x in row) for row in a)


def bucketed_eri3c(basis, aux_basis, omega=None, chunk=4096):
    """Full (nao, nao, naux) tensor via class buckets, then the same
    cart2sph transforms as eri3c_matrix."""
    bra = shell_table(basis)
    aux = shell_table(aux_basis)
    buckets = bucket_triples(bra, aux)
    nao = np.asarray(basis.centers).shape[0]
    naux = np.asarray(aux_basis.centers).shape[0]
    out = jnp.zeros((nao, nao, naux))
    for (la, lb, lc), b in sorted(buckets.items()):
        anga, angb, angc = b["ang"]
        fn = _compiled_class_kernel(
            la, lb, lc, _hashable_ang(anga), _hashable_ang(angb),
            _hashable_ang(angc), omega, min(chunk, b["A"].shape[0]))
        vals = fn(
            jnp.asarray(b["A"]), jnp.asarray(b["B"]), jnp.asarray(b["C"]),
            jnp.asarray(b["ea"]), jnp.asarray(b["eb"]), jnp.asarray(b["ec"]),
            jnp.asarray(b["ca"]), jnp.asarray(b["cb"]), jnp.asarray(b["cc"]),
        )   # (ntrip, nca, ncb, ncc)
        nca, ncb, ncc = vals.shape[1:]
        r = b["rows"]
        i_idx = (r[:, 0, None, None, None] + np.arange(nca)[:, None, None])
        j_idx = (r[:, 1, None, None, None] + np.arange(ncb)[None, :, None])
        k_idx = (r[:, 2, None, None, None] + np.arange(ncc)[None, None, :])
        i_idx, j_idx, k_idx = (np.broadcast_to(x, vals.shape)
                               for x in (i_idx, j_idx, k_idx))
        out = out.at[i_idx.ravel(), j_idx.ravel(), k_idx.ravel()].set(
            vals.reshape(-1))
        # bra symmetry: writing vals[t,a,b,c] at (j0+b, i0+a, k0+c) IS the
        # transpose; swap the index arrays and keep the value order (a value
        # transpose here would double-transpose against the (a,b) ravel).
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
