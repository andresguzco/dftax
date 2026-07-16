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

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from dftax.integrals.eri3c import _hermite_coulomb
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

def _E_table(la, lb, alpha, beta, XAB, mt):
    """Full (la+1, lb+1, mt) two-center E table (same recursion as
    _md_E_coefficients_1d, without the final row slice)."""
    gamma = alpha + beta
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    XPA = -beta * XAB / safe_gamma
    XPB = alpha * XAB / safe_gamma
    t_idx = jnp.arange(mt)
    E = jnp.zeros((la + 1, lb + 1, mt)).at[0, 0, 0].set(1.0)

    def bump(row, X):
        left = jnp.concatenate([jnp.zeros(1), row[:-1]])
        right = jnp.concatenate([row[1:], jnp.zeros(1)])
        return (X * row + jnp.where(t_idx > 0, i2g * left, 0.0)
                + jnp.where(t_idx + 1 < mt, (t_idx + 1) * right, 0.0))

    def _step_i(i, E):
        return E.at[i + 1, 0, :].set(bump(E[i, 0, :], XPA))

    E = lax.fori_loop(0, la, _step_i, E)

    def _step_j(j, E):
        def _step_ji(i, E):
            return E.at[i, j + 1, :].set(bump(E[i, j, :], XPB))
        return lax.fori_loop(0, la + 1, _step_ji, E)

    return lax.fori_loop(0, lb, _step_j, E)


def _Ec_table(lc, gamma, mt):
    """(lc+1, mt) single-center E table (same recursion as
    _single_center_E_1d, without the final row slice)."""
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    t_idx = jnp.arange(mt)
    E = jnp.zeros((lc + 1, mt)).at[0, 0].set(1.0)

    def _step_i(i, E):
        row = E[i, :]
        left = jnp.concatenate([jnp.zeros(1), row[:-1]])
        right = jnp.concatenate([row[1:], jnp.zeros(1)])
        new = (jnp.where(t_idx > 0, i2g * left, 0.0)
               + jnp.where(t_idx + 1 < mt, (t_idx + 1) * right, 0.0))
        return E.at[i + 1, :].set(new)

    return lax.fori_loop(0, lc, _step_i, E)


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
                R = _hermite_coulomb(rho, P - C, mt, mt, omega)
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
        kern = make_class_kernel(la, lb, lc, anga, angb, angc, omega)
        # explicit in_axes: the utility's chunked path only batches the
        # arguments named by in_axes, and a bare 0 covers just the first
        vals = chunked_vmap(kern, in_axes=(0,) * 9,
                            chunk_size=min(chunk, b["A"].shape[0]))(
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
