from __future__ import annotations

import abc
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar

from dftax.energy.grid import xc_energy
from dftax.energy.gto import BasisData, eval_gto, eval_gto_and_grad
from dftax.energy.potentials import xc_potential
from dftax.energy.xc import XCFunctional
from dftax.integrals.eri3c import _DF_BRA_BUDGET, _contracted_eri3c, _eri3c_sizes
from dftax.integrals.eri4c import coulomb_j_4c, exchange_k_4c
from dftax.utils.vmap import vmap as _chunked_vmap


@dataclass(frozen=True)
class ExactSpec:
    screen: float | None = None
    stream: bool = False


@dataclass(frozen=True)
class ExternalCoulombSpec(ExactSpec):
    stream: bool = True


@dataclass(frozen=True)
class DFSpec:
    auxbasis: str | BasisData = "def2-universal-jkfit"
    chunk: int | str | None = "auto"
    screen: float | str | None = "auto"
    spherical: bool | None = None
    whiten: bool | None = None
    packed: bool | str = "auto"


def exact(*, screen: float | None = None, stream: bool = False) -> ExactSpec:
    """Exact 4-center ERI Coulomb/exchange (small systems; O(N⁴) memory).

    Args:
        screen: optional Cauchy-Schwarz threshold; negligible ERI quartets are
            skipped when materializing the tensor.
        stream: contract J/K on the fly instead of materializing the ERI
            tensor (O(N²) memory, slower; incompatible with ``screen``).
    """
    if screen is not None and stream:
        raise ValueError(
            "exact(screen=..., stream=True) is not supported: quartet screening "
            "applies only to the materialized ERI tensor."
        )
    return ExactSpec(screen=screen, stream=stream)


def df(
    auxbasis: str | BasisData = "def2-universal-jkfit",
    *,
    chunk: int | str | None = "auto",
    screen: float | str | None = "auto",
    spherical: bool | None = None,
    whiten: bool | None = None,
    packed: bool | str = "auto",
) -> DFSpec:
    """Density-fitted (RI) Coulomb/exchange with the given auxiliary basis.

    This is the KS default backend (O(N³) memory; RI error sub-mHa with a
    JK-fitting set); :func:`exact` remains available for small systems and
    reference comparisons.

    Args:
        auxbasis: JK-fitting auxiliary basis name (default
            ``"def2-universal-jkfit"``, the universal set covering any orbital
            basis), or an already built ``BasisData``.
        chunk: RI memory strategy. ``"auto"`` (default) materializes the
            nao²×naux 3-center tensor when it fits a memory budget and
            otherwise streams RI-J/RI-K over budget-derived auxiliary chunks;
            ``None`` forces the materialized tensor; an int streams with
            exactly that chunk.
        screen: relative Cauchy-Schwarz threshold dropping negligible bra
            pairs (survivors are O(N) for extended systems). On the
            materialized path it is a shell-pair compact gather that omits
            the screened shell-pairs from the 3-center build (they stay
            exactly zero); with an int ``chunk`` it restricts the streamed
            RI-J bra sum to the significant AO pairs. ``None`` keeps every
            pair, a bit-identical build. ``"auto"`` (default) screens at
            ``DF_SCREEN_AUTO_CUTOFF`` from ``DF_SCREEN_AUTO_MIN_ATOMS`` atoms
            up (the grid's gate) and keeps every pair below it; batched
            geometries (``scf_batched``) never screen.
        spherical: auxiliary basis span. ``None`` (default) uses spherical
            harmonics on the materialized paths, single-device and
            mesh-sharded alike (non-redundant fit, positive definite metric),
            and cartesian components on the streamed path (which contracts
            cartesian auxiliary elements on the fly). ``False`` forces
            cartesian everywhere, e.g. to
            compare a materialized result against a streamed one in the same
            fit space; ``True`` asserts the spherical span and raises where
            it is unsupported.
        whiten: fold the metric into the materialized tensor once, ``B = Lᵀ T``
            with ``L Lᵀ = V⁺`` (:class:`WhitenedDFCoulomb`): the same energy to
            the density-fitting floor, no metric multiply per build, and an
            orbital-route exchange that never forms ``nao x nao`` kernels.
            ``None`` (default) whitens whenever the build is concrete and
            single-device materialized, and keeps :class:`DFCoulomb` where the
            geometry is traced (forces, Hessians, ``scf_batched``), whose
            metric inverse carries the geometry derivative; ``False`` never
            whitens; ``True`` asserts it and raises where it cannot apply.
        packed: store the whitened tensor packed over the kept bra shell
            pairs, ``(naux, npair)`` (:class:`PackedDFCoulomb`): about half
            the memory before screening and only the kept pairs with it, at
            the price of unpacking one aux block per exchange evaluation.
            ``"auto"`` (default) packs where that pays: when pair screening
            is on (``screen``) or when the square tensor would miss the
            device budget and the packed one fits; ``True`` / ``False``
            force it. Traced builds (no whitening) stay square.

    Example:
        ```python
        KS(mol, xc)                                    # same as coulomb=df()
        KS(mol, xc, coulomb=df(chunk=None))            # force materialized
        KS(mol, xc, coulomb=df("def2-universal-jkfit", chunk=64))  # streamed
        ```
    """
    if spherical is True and isinstance(chunk, int):
        raise ValueError(
            "df(spherical=True, chunk=<int>) is not supported: the streamed "
            "backend contracts cartesian auxiliary elements on the fly."
        )
    if whiten is True and isinstance(chunk, int):
        raise ValueError(
            "df(whiten=True, chunk=<int>) is not supported: the streamed "
            "backend never materializes the tensor it would whiten."
        )
    if packed is True and (whiten is False or isinstance(chunk, int)):
        raise ValueError(
            "df(packed=True) stores the whitened materialized tensor; it "
            "needs whiten=None or True and a materialized chunk."
        )
    return DFSpec(auxbasis=auxbasis, chunk=chunk, screen=screen,
                  spherical=spherical, whiten=whiten, packed=packed)


DF_SCREEN_AUTO_MIN_ATOMS = 50
DF_SCREEN_AUTO_CUTOFF = 1e-10


def resolve_df_screen(screen, n_atom: int) -> float | None:
    if screen == "auto":
        return (DF_SCREEN_AUTO_CUTOFF if n_atom >= DF_SCREEN_AUTO_MIN_ATOMS
                else None)
    return screen


@jax.custom_jvp
def _metric_pinv(V: Float[Array, "naux naux"]) -> Float[Array, "naux naux"]:
    w, U = jnp.linalg.eigh(V)
    inv_w = jnp.where(w > 1e-7 * w[-1], 1.0 / w, 0.0)
    return (U * inv_w) @ U.T


@_metric_pinv.defjvp
def _metric_pinv_jvp(primals, tangents):
    (V,), (dV,) = primals, tangents
    Vp = _metric_pinv(V)
    dVs = 0.5 * (dV + dV.T)
    return Vp, -Vp @ dVs @ Vp


def _metric_factor(V: Float[Array, "naux naux"]) -> Float[Array, "naux naux"]:
    w, U = jnp.linalg.eigh(V)
    keep = w > 1e-7 * w[-1]
    return U * jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, w, 1.0)), 0.0)


def _whiten(T: Float[Array, "nao nao naux"],
            L: Float[Array, "naux naux"]) -> Float[Array, "nao nao naux"]:
    nao, _, naux = T.shape
    rows = max(1, (1 << 24) // max(1, nao * naux))
    k = max(d for d in range(1, min(nao, rows) + 1) if nao % d == 0)

    def body(i, T):
        blk = jax.lax.dynamic_slice_in_dim(T, i * k, k, axis=0)
        return jax.lax.dynamic_update_slice_in_dim(
            T, jnp.einsum("mnP,Px->mnx", blk, L), i * k, axis=0)

    return jax.lax.fori_loop(0, nao // k, body, T)


def _whiten_packed(T: Float[Array, "naux npair"],
                   L: Float[Array, "naux naux"]) -> Float[Array, "naux npair"]:
    naux, npair = T.shape
    k = max(1, min(npair, (1 << 24) // max(1, naux)))
    nfull, tail = divmod(npair, k)

    def body(i, T):
        blk = jax.lax.dynamic_slice_in_dim(T, i * k, k, axis=1)
        return jax.lax.dynamic_update_slice_in_dim(T, L.T @ blk, i * k, axis=1)

    T = jax.lax.fori_loop(0, nfull, body, T)
    if tail:
        T = T.at[:, nfull * k:].set(L.T @ T[:, nfull * k:])
    return T


def _streamed_e_xc(xc, basis, coords, weights, P, chunk,
                   checkpoint=True):
    gga = xc.xc_type == "GGA"
    mgga = xc.xc_type == "MGGA"

    def point(r, w):
        if gga or mgga:
            ao_g, dao_g = eval_gto_and_grad(basis, r)
        else:
            ao_g = eval_gto(basis, r)
        rho = ao_g @ P @ ao_g
        mask = rho > 1e-10
        safe_rho = jnp.where(mask, rho, 1.0)
        if gga or mgga:
            grad = 2.0 * (ao_g @ P) @ dao_g
            if mgga:
                tau = 0.5 * jnp.einsum("mx,mn,nx->", dao_g, P, dao_g)
                eps = xc(safe_rho, jnp.where(mask, grad, 0.0),
                         jnp.where(mask, tau, 1.0))
            else:
                eps = xc(safe_rho, jnp.where(mask, grad, 0.0))
        else:
            eps = xc(safe_rho)
        return jnp.where(mask, w * eps * rho, 0.0)

    contribs = _chunked_vmap(
        point, in_axes=(0, 0), chunk_size=chunk, checkpoint=checkpoint
    )(coords, weights)
    return jnp.sum(contribs)


def _streamed_e_xc_spin(xc, basis, coords, weights, Pa, Pb, chunk,
                        checkpoint=True):
    gga = xc.xc_type == "GGA"
    mgga = xc.xc_type == "MGGA"

    def point(r, w):
        if gga or mgga:
            ao, dao = eval_gto_and_grad(basis, r)
        else:
            ao = eval_gto(basis, r)
        rho_a = ao @ Pa @ ao
        rho_b = ao @ Pb @ ao
        rho_tot = rho_a + rho_b
        mask = rho_tot > 1e-10
        ta = rho_a > 1e-10
        tb = rho_b > 1e-10
        rho2 = jnp.stack([jnp.where(ta, rho_a, 1e-10), jnp.where(tb, rho_b, 1e-10)])
        if gga or mgga:
            ga = jnp.where(ta, 2.0 * (ao @ Pa) @ dao, 0.0)
            gb = jnp.where(tb, 2.0 * (ao @ Pb) @ dao, 0.0)
            if mgga:
                tau2 = jnp.stack([
                    jnp.where(ta, 0.5 * jnp.einsum("mx,mn,nx->", dao, Pa, dao), 1e-10),
                    jnp.where(tb, 0.5 * jnp.einsum("mx,mn,nx->", dao, Pb, dao), 1e-10),
                ])
                eps = xc(rho2, jnp.stack([ga, gb], axis=-1), tau2)
            else:
                eps = xc(rho2, jnp.stack([ga, gb], axis=-1))
        else:
            eps = xc(rho2)
        return jnp.where(mask, w * eps * rho_tot, 0.0)

    contribs = _chunked_vmap(
        point, in_axes=(0, 0), chunk_size=chunk, checkpoint=checkpoint
    )(coords, weights)
    return jnp.sum(contribs)


def _eri3c_elem(basis, aux_basis, i, j, k, omega=None):
    ml, mt, mm = _eri3c_sizes(basis, aux_basis)
    return _contracted_eri3c(
        basis.exponents[i], basis.coefficients[i], basis.centers[i], basis.angular[i],
        basis.exponents[j], basis.coefficients[j], basis.centers[j], basis.angular[j],
        aux_basis.exponents[k], aux_basis.coefficients[k],
        aux_basis.centers[k], aux_basis.angular[k],
        ml, mt, mm, omega,
    )


def _eri3c_bra_chunk(basis, aux_basis, inflight):
    ml, mt, _ = _eri3c_sizes(basis, aux_basis)
    nprim = int(basis.exponents.shape[1])
    nprim_aux = int(aux_basis.exponents.shape[1])
    per = mt * mt * mt * nprim * nprim * nprim_aux * ml * max(int(inflight), 1)
    return max(1, int(_DF_BRA_BUDGET // per))


def _streamed_gamma(basis, aux_basis, P, chunk, pairs, k_idx):
    Ptil = basis.cart2sph @ P @ basis.cart2sph.T if basis.cart2sph is not None else P
    bra_chunk = _eri3c_bra_chunk(basis, aux_basis, inflight=chunk)

    if pairs is None:
        n = basis.centers.shape[0]
        ii, jj = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
        ii, jj, Pflat = ii.reshape(-1), jj.reshape(-1), Ptil.reshape(-1)
        pidx = jnp.arange(n * n)

        def gamma_k(k):
            def pair(p):
                return _eri3c_elem(basis, aux_basis, ii[p], jj[p], k) * Pflat[p]
            return jnp.sum(_chunked_vmap(pair, chunk_size=bra_chunk)(pidx))
    else:
        pi, pj, w = pairs
        Pw = Ptil[pi, pj] * w
        pidx = jnp.arange(pi.shape[0])

        def gamma_k(k):
            def pair(p):
                return _eri3c_elem(basis, aux_basis, pi[p], pj[p], k) * Pw[p]
            return jnp.sum(_chunked_vmap(pair, chunk_size=bra_chunk)(pidx))

    return _chunked_vmap(gamma_k, chunk_size=chunk, checkpoint=True)(k_idx)


def _streamed_df_rij(basis, aux_basis, int2c_inv, P, chunk, pairs=None):
    naux = aux_basis.centers.shape[0]
    gamma = _streamed_gamma(basis, aux_basis, P, chunk, pairs, jnp.arange(naux))
    return 0.5 * jnp.dot(gamma, int2c_inv @ gamma)


def _aux_slab_tensor(basis, aux_basis, slab, omega=None):
    from dftax.integrals.eri3c_bucketed import (
        eri3c_matrix_bucketed, slice_aux,
    )

    lo, hi, slo, shi, plan = slab
    return eri3c_matrix_bucketed(
        basis, slice_aux(aux_basis, lo, hi, slo, shi), omega=omega, plan=plan
    )


def _streamed_df_rij_slabs(basis, aux_basis, int2c_inv, P, slabs):
    def gamma_slab(Pf, slab):
        T = _aux_slab_tensor(basis, aux_basis, slab)
        return jnp.einsum("mnk,mn->k", T, Pf)

    gamma = jnp.concatenate([
        jax.checkpoint(lambda Pf, s=slab: gamma_slab(Pf, s))(P)
        for slab in slabs
    ])
    return 0.5 * jnp.dot(gamma, int2c_inv @ gamma)


def _rik_occ_orbitals(P, S, nocc, dscale=0.5):
    sval, svec = jnp.linalg.eigh(S)
    sval = jnp.clip(sval, 1e-12, None)
    s_ih = (svec / jnp.sqrt(sval)) @ svec.T
    s_h = (svec * jnp.sqrt(sval)) @ svec.T
    _, evec = jnp.linalg.eigh(s_h @ (dscale * P) @ s_h)
    ncol = evec.shape[1]
    return s_ih @ evec[:, ncol - nocc:]


def _rik_cholesky(int2c_inv):
    w, U = jnp.linalg.eigh(int2c_inv)
    return U * jnp.sqrt(jnp.clip(w, 0.0, None))


def _rik_bmj(basis, aux_basis, Lf, cj, n, naux, omega=None):
    idx = jnp.arange(n)
    aux_idx = jnp.arange(naux)
    m_chunk = _eri3c_bra_chunk(basis, aux_basis, inflight=naux * n)

    def entry(m, k):
        col = jax.vmap(
            lambda l: _eri3c_elem(basis, aux_basis, m, l, k, omega))(idx)
        return col @ cj
    def row_m(m):
        return jax.vmap(lambda k: entry(m, k))(aux_idx)
    M = _chunked_vmap(row_m, chunk_size=m_chunk)(idx)
    return M @ Lf


def _rik_energy(basis, aux_basis, int2c_inv, Cocc, omega=None):
    Lf = _rik_cholesky(int2c_inv)
    c2s = basis.cart2sph
    Cc = c2s @ Cocc if c2s is not None else Cocc
    n, naux = Cc.shape[0], Lf.shape[0]

    def body(acc, cj):
        Bij = Cc.T @ _rik_bmj(basis, aux_basis, Lf, cj, n, naux, omega)
        return acc + jnp.sum(Bij * Bij), None
    ek, _ = jax.lax.scan(jax.checkpoint(body), jnp.array(0.0), Cc.T)
    return ek


def _rik_kmatrix(basis, aux_basis, int2c_inv, Cocc, omega=None):
    Lf = _rik_cholesky(int2c_inv)
    c2s = basis.cart2sph
    Cc = c2s @ Cocc if c2s is not None else Cocc
    n, naux = Cc.shape[0], Lf.shape[0]

    def body(Ka, cj):
        B = _rik_bmj(basis, aux_basis, Lf, cj, n, naux, omega)
        return Ka + (B @ B.T), None
    Kc, _ = jax.lax.scan(jax.checkpoint(body), jnp.zeros((n, n)), Cc.T)
    return c2s.T @ Kc @ c2s if c2s is not None else Kc


def _rik_shard_mesh(devices):
    import numpy as np

    return jax.sharding.Mesh(np.asarray(devices), ("aux",))


def _rik_pad_occ(Cc, ndev):
    nocc = Cc.shape[1]
    slab = -(-nocc // ndev)
    pad = ndev * slab - nocc
    return (jnp.pad(Cc, ((0, 0), (0, pad))) if pad else Cc), slab


def _rik_energy_sharded(basis, aux_basis, int2c_inv, Cocc, devices,
                        omega=None):
    from jax import shard_map

    Lf = _rik_cholesky(int2c_inv)
    c2s = basis.cart2sph
    Cc = c2s @ Cocc if c2s is not None else Cocc
    n, naux = Cc.shape[0], Lf.shape[0]
    Ccp, _slab = _rik_pad_occ(Cc, len(devices))
    spec = jax.sharding.PartitionSpec

    def part(Cfull, Lfull, cols):
        def body(acc, cj):
            Bij = Cfull.T @ _rik_bmj(basis, aux_basis, Lfull, cj, n, naux,
                                     omega)
            return acc + jnp.sum(Bij * Bij), None
        ek, _ = jax.lax.scan(jax.checkpoint(body), jnp.array(0.0), cols)
        return jax.lax.psum(ek, "aux")

    return shard_map(
        part, mesh=_rik_shard_mesh(devices),
        in_specs=(spec(), spec(), spec("aux")), out_specs=spec(),
        check_vma=False,
    )(Cc, Lf, Ccp.T)


def _rik_kmatrix_sharded(basis, aux_basis, int2c_inv, Cocc, devices,
                         omega=None):
    from jax import shard_map

    Lf = _rik_cholesky(int2c_inv)
    c2s = basis.cart2sph
    Cc = c2s @ Cocc if c2s is not None else Cocc
    n, naux = Cc.shape[0], Lf.shape[0]
    Ccp, _slab = _rik_pad_occ(Cc, len(devices))
    spec = jax.sharding.PartitionSpec

    def part(Lfull, cols):
        def body(Ka, cj):
            B = _rik_bmj(basis, aux_basis, Lfull, cj, n, naux, omega)
            return Ka + (B @ B.T), None
        Kc, _ = jax.lax.scan(jax.checkpoint(body), jnp.zeros((n, n)), cols)
        return jax.lax.psum(Kc, "aux")

    Kc = shard_map(
        part, mesh=_rik_shard_mesh(devices),
        in_specs=(spec(), spec("aux")), out_specs=spec(), check_vma=False,
    )(Lf, Ccp.T)
    return c2s.T @ Kc @ c2s if c2s is not None else Kc


def _rik_gmat_slabs(basis, aux_basis, slabs, C, omega=None):
    return jnp.concatenate([
        jnp.einsum("mnk,ni->kmi", _aux_slab_tensor(basis, aux_basis, sl,
                                                   omega), C)
        for sl in slabs
    ])


def _rik_slabs(basis, aux_basis, int2c_inv, Cocc, slabs, omega=None):
    G = _rik_gmat_slabs(basis, aux_basis, slabs, Cocc, omega)
    naux, n, nocc = G.shape
    H = (int2c_inv @ G.reshape(naux, -1)).reshape(naux, n, nocc)
    KK = jnp.einsum("xmi,xni->mn", G, H)
    return jnp.vdot(Cocc, KK @ Cocc), KK


def _streamed_df_rik(basis, aux_basis, int2c_inv, S, nocc, P,
                     dscale, energy_pref, grad_pref, omega=None,
                     devices=None, slabs=None):
    def _energy(Cocc):
        if devices is not None:
            return _rik_energy_sharded(basis, aux_basis, int2c_inv, Cocc,
                                       devices, omega)
        if slabs is not None:
            return _rik_slabs(basis, aux_basis, int2c_inv, Cocc, slabs,
                              omega)[0]
        return _rik_energy(basis, aux_basis, int2c_inv, Cocc, omega)

    def _kmat(Cocc):
        if devices is not None:
            return _rik_kmatrix_sharded(basis, aux_basis, int2c_inv, Cocc,
                                        devices, omega)
        if slabs is not None:
            return _rik_slabs(basis, aux_basis, int2c_inv, Cocc, slabs,
                              omega)[1]
        return _rik_kmatrix(basis, aux_basis, int2c_inv, Cocc, omega)

    @jax.custom_vjp
    def rik(P):
        Cocc = _rik_occ_orbitals(P, S, nocc, dscale)
        return energy_pref * _energy(Cocc)

    def fwd(P):
        Cocc = _rik_occ_orbitals(P, S, nocc, dscale)
        if slabs is not None and devices is None:
            e_raw, KK = _rik_slabs(basis, aux_basis, int2c_inv, Cocc, slabs,
                                   omega)
            return energy_pref * e_raw, (None, KK)
        return energy_pref * _energy(Cocc), (Cocc, None)

    def bwd(res, g):
        Cocc, KK = res
        if KK is None:
            KK = _kmat(Cocc)
        return (g * grad_pref * KK,)

    rik.defvjp(fwd, bwd)
    return rik(P)


def _rik_dmat(basis, aux_basis, slabs, C, omega=None):
    def d_slab(Cf, slab):
        T = _aux_slab_tensor(basis, aux_basis, slab, omega)
        return jnp.einsum("mnk,mi,nj->kij", T, Cf, Cf)

    return jnp.concatenate([
        jax.checkpoint(lambda Cf, s=slab: d_slab(Cf, s))(C)
        for slab in slabs
    ])


def _streamed_df_rik_frozen(basis, aux_basis, int2c_inv, S, Zs, prefs, slabs,
                            omega=None):
    e = jnp.asarray(0.0)
    for Z, pref in zip(Zs, prefs):
        if Z.shape[1] == 0:
            continue
        M = Z.T @ S @ Z
        D = _rik_dmat(basis, aux_basis, slabs, Z, omega)
        naux = D.shape[0]
        Q = jnp.linalg.solve(M, D)
        QF = Q.reshape(naux, -1)
        QT = Q.transpose(0, 2, 1).reshape(naux, -1)
        e = e + pref * jnp.vdot(QF, int2c_inv @ QT)
    return e


def _exchange_quadratic(kfun, P, ax):
    if P.shape[0] == 1:
        return -0.25 * ax * jnp.sum(P[0] * kfun(P[0]))
    return -0.5 * ax * sum(jnp.sum(Ps * kfun(Ps)) for Ps in P)


def _rik_materialized(int3c, int2c_inv, Cocc):
    G = jnp.einsum("mlX,li->Xmi", int3c, Cocc)
    naux, n, no = G.shape
    H = (int2c_inv @ G.reshape(naux, -1)).reshape(naux, n, no)
    KK = jnp.einsum("xmi,xni->mn", G, H)
    return jnp.vdot(Cocc, KK @ Cocc), KK


class CoulombTerm(eqx.Module):
    @abc.abstractmethod
    def energy(
        self,
        P: Float[Array, "nspin nao nao"],
        S: Float[Array, "nao nao"],
        nocc: tuple[int, ...],
    ) -> Scalar:
        raise NotImplementedError

    def energy_and_potential(self, P, S, nocc, idempotent: bool = True):
        return jax.value_and_grad(lambda Q: self.energy(Q, S, nocc))(P)


class ExactCoulomb(CoulombTerm):
    eri: Float[Array, "nao nao nao nao"]
    hf_coeff: float = eqx.field(static=True)
    eri_lr: Float[Array, "nao nao nao nao"] | None = None
    hf_coeff_lr: float = eqx.field(static=True, default=0.0)

    def energy(self, P, S, nocc):
        Ptot = jnp.sum(P, axis=0)
        J = jnp.einsum("ijkl,kl->ij", self.eri, Ptot)
        e = 0.5 * jnp.sum(Ptot * J)
        if self.hf_coeff != 0.0:
            e = e + _exchange_quadratic(
                lambda Q: jnp.einsum("ikjl,kl->ij", self.eri, Q), P, self.hf_coeff
            )
        if self.hf_coeff_lr != 0.0:
            e = e + _exchange_quadratic(
                lambda Q: jnp.einsum("ikjl,kl->ij", self.eri_lr, Q),
                P, self.hf_coeff_lr,
            )
        return e


class StreamedExactCoulomb(CoulombTerm):
    basis: BasisData
    hf_coeff: float = eqx.field(static=True)

    def energy(self, P, S, nocc):
        Ptot = jnp.sum(P, axis=0)
        J = coulomb_j_4c(Ptot, self.basis)
        e = 0.5 * jnp.sum(Ptot * J)
        if self.hf_coeff != 0.0:
            e = e + _exchange_quadratic(
                lambda Q: exchange_k_4c(Q, self.basis), P, self.hf_coeff
            )
        return e


class DFCoulomb(CoulombTerm):
    int3c: Float[Array, "nao nao naux"]
    int2c_inv: Float[Array, "naux naux"]
    hf_coeff: float = eqx.field(static=True)
    int3c_lr: Float[Array, "nao nao naux"] | None = None
    int2c_inv_lr: Float[Array, "naux naux"] | None = None
    hf_coeff_lr: float = eqx.field(static=True, default=0.0)

    def energy(self, P, S, nocc):
        Ptot = jnp.sum(P, axis=0)
        gamma = jnp.einsum("mnP,mn->P", self.int3c, Ptot)
        e = 0.5 * jnp.dot(gamma, self.int2c_inv @ gamma)
        if self.hf_coeff != 0.0:
            e = e + _exchange_quadratic(
                lambda Q: jnp.einsum(
                    "mlP,PQ,nsQ,ls->mn", self.int3c, self.int2c_inv, self.int3c, Q
                ),
                P,
                self.hf_coeff,
            )
        if self.hf_coeff_lr != 0.0:
            e = e + _exchange_quadratic(
                lambda Q: jnp.einsum(
                    "mlP,PQ,nsQ,ls->mn",
                    self.int3c_lr, self.int2c_inv_lr, self.int3c_lr, Q,
                ),
                P,
                self.hf_coeff_lr,
            )
        return e

    def energy_and_potential(self, P, S, nocc, idempotent=True):
        if not idempotent:
            return super().energy_and_potential(P, S, nocc, idempotent)

        Ptot = jnp.sum(P, axis=0)
        gamma = jnp.einsum("mnP,mn->P", self.int3c, Ptot)
        Vg = 0.5 * (self.int2c_inv @ gamma + gamma @ self.int2c_inv)
        e = 0.5 * jnp.dot(gamma, Vg)
        J = jnp.einsum("mnP,P->mn", self.int3c, Vg)
        V = jnp.broadcast_to(J, P.shape)

        nspin = P.shape[0]
        dscale = 0.5 if nspin == 1 else 1.0
        for ax, t3, vinv in ((self.hf_coeff, self.int3c, self.int2c_inv),
                             (self.hf_coeff_lr, self.int3c_lr,
                              self.int2c_inv_lr)):
            if ax == 0.0:
                continue
            e_pref = -ax if nspin == 1 else -0.5 * ax
            for sigma, n in enumerate(nocc):
                if n == 0:
                    continue
                C = jax.lax.stop_gradient(
                    _rik_occ_orbitals(P[sigma], S, n, dscale))
                raw, KK = _rik_materialized(t3, vinv, C)
                e = e + e_pref * raw
                V = V.at[sigma].add(-ax * KK)
        return e, V


    def energy_and_grad_orbitals(self, Cs, ws, S, nocc):
        nspin = len(Cs)
        w_full = 2.0 if nspin == 1 else 1.0
        P = jnp.stack([(C * w) @ C.T for C, w in zip(Cs, ws)])
        Ptot = jnp.sum(P, axis=0)
        gamma = jnp.einsum("mnP,mn->P", self.int3c, Ptot)
        Vg = 0.5 * (self.int2c_inv @ gamma + gamma @ self.int2c_inv)
        e = 0.5 * jnp.dot(gamma, Vg)
        J = jnp.einsum("mnP,P->mn", self.int3c, Vg)
        grads = [2.0 * (J @ C) * w for C, w in zip(Cs, ws)]

        for ax, t3, vinv in ((self.hf_coeff, self.int3c, self.int2c_inv),
                             (self.hf_coeff_lr, self.int3c_lr,
                              self.int2c_inv_lr)):
            if ax == 0.0:
                continue
            e_pref = -ax if nspin == 1 else -0.5 * ax
            vsym = 0.5 * (vinv + vinv.T)
            for s, (C, w) in enumerate(zip(Cs, ws)):
                if C.shape[1] == 0:
                    continue
                scale = jnp.sqrt(w / w_full)
                Ct = C * scale
                raw, KK = _rik_materialized(t3, vsym, Ct)
                e = e + e_pref * raw
                grads[s] = grads[s] + 4.0 * e_pref * (KK @ Ct) * scale
        return e, tuple(grads)


_K_BLOCK_ELEMS = 1 << 25


def _over_aux_blocks(Bt, ncol, fn):
    nao, _, naux = Bt.shape
    b = max(1, min(naux, _K_BLOCK_ELEMS // max(1, nao * ncol)))
    nfull, tail = divmod(naux, b)
    acc = fn(Bt[:, :, :b])
    if nfull > 1:
        def body(i, acc):
            blk = jax.lax.dynamic_slice_in_dim(Bt, i * b, b, axis=2)
            return jax.tree.map(jnp.add, acc, fn(blk))
        acc = jax.lax.fori_loop(1, nfull, body, acc)
    if tail:
        acc = jax.tree.map(jnp.add, acc, fn(Bt[:, :, nfull * b:]))
    return acc


def _rik_whitened(B, Cocc):
    def block(Bb):
        G = jnp.einsum("mlx,li->xmi", Bb, Cocc)
        return jnp.einsum("xmi,xni->mn", G, G)

    KK = _over_aux_blocks(B, Cocc.shape[1], block)
    return jnp.vdot(Cocc, KK @ Cocc), KK


def _rik_whitened_orbitals(B, Ct):
    def block(Bb):
        G = jnp.einsum("mlx,li->xmi", Bb, Ct)
        D = jnp.einsum("mi,xmj->xij", Ct, G)
        return jnp.sum(D * D), jnp.einsum("xmi,xij->mj", G, D)

    return _over_aux_blocks(B, Ct.shape[1], block)


class WhitenedDFCoulomb(CoulombTerm):
    B: Float[Array, "nao nao naux"]
    hf_coeff: float = eqx.field(static=True)
    B_lr: Float[Array, "nao nao naux"] | None = None
    hf_coeff_lr: float = eqx.field(static=True, default=0.0)

    def _exchanges(self):
        return tuple((ax, Bt) for ax, Bt in ((self.hf_coeff, self.B),
                                             (self.hf_coeff_lr, self.B_lr))
                     if ax != 0.0)

    def energy(self, P, S, nocc):
        gamma = jnp.einsum("mnx,mn->x", self.B, jnp.sum(P, axis=0))
        e = 0.5 * jnp.dot(gamma, gamma)
        for ax, Bt in self._exchanges():
            e = e + _exchange_quadratic(
                lambda Q, Bt=Bt: jnp.einsum("mlx,nsx,ls->mn", Bt, Bt, Q),
                P, ax)
        return e

    def energy_and_potential(self, P, S, nocc, idempotent=True):
        if not idempotent:
            return super().energy_and_potential(P, S, nocc, idempotent)
        gamma = jnp.einsum("mnx,mn->x", self.B, jnp.sum(P, axis=0))
        e = 0.5 * jnp.dot(gamma, gamma)
        V = jnp.broadcast_to(jnp.einsum("mnx,x->mn", self.B, gamma), P.shape)
        nspin = P.shape[0]
        dscale = 0.5 if nspin == 1 else 1.0
        for ax, Bt in self._exchanges():
            e_pref = -ax if nspin == 1 else -0.5 * ax
            for sigma, n in enumerate(nocc):
                if n == 0:
                    continue
                C = jax.lax.stop_gradient(
                    _rik_occ_orbitals(P[sigma], S, n, dscale))
                raw, KK = _rik_whitened(Bt, C)
                e = e + e_pref * raw
                V = V.at[sigma].add(-ax * KK)
        return e, V

    def energy_and_grad_orbitals(self, Cs, ws, S, nocc):
        nspin = len(Cs)
        w_full = 2.0 if nspin == 1 else 1.0
        P = jnp.stack([(C * w) @ C.T for C, w in zip(Cs, ws)])
        gamma = jnp.einsum("mnx,mn->x", self.B, jnp.sum(P, axis=0))
        e = 0.5 * jnp.dot(gamma, gamma)
        J = jnp.einsum("mnx,x->mn", self.B, gamma)
        grads = [2.0 * (J @ C) * w for C, w in zip(Cs, ws)]
        for ax, Bt in self._exchanges():
            e_pref = -ax if nspin == 1 else -0.5 * ax
            for s, (C, w) in enumerate(zip(Cs, ws)):
                if C.shape[1] == 0:
                    continue
                scale = jnp.sqrt(w / w_full)
                Ct = C * scale
                raw, GD = _rik_whitened_orbitals(Bt, Ct)
                e = e + e_pref * raw
                grads[s] = grads[s] + 4.0 * e_pref * GD * scale
        return e, tuple(grads)


def _over_row_blocks(X, per_row, fn):
    n = X.shape[0]
    b = max(1, min(n, _K_BLOCK_ELEMS // max(1, per_row)))
    nfull, tail = divmod(n, b)
    acc = fn(X[:b])
    if nfull > 1:
        def body(i, acc):
            blk = jax.lax.dynamic_slice_in_dim(X, i * b, b, axis=0)
            return jax.tree.map(jnp.add, acc, fn(blk))
        acc = jax.lax.fori_loop(1, nfull, body, acc)
    if tail:
        acc = jax.tree.map(jnp.add, acc, fn(X[nfull * b:]))
    return acc


class PackedDFCoulomb(CoulombTerm):
    B: Float[Array, "naux npair"]
    pos: Int[Array, "nao nao"]
    rows: Int[Array, "npair"]
    cols: Int[Array, "npair"]
    weight: Float[Array, "npair"]
    hf_coeff: float = eqx.field(static=True)
    B_lr: Float[Array, "naux npair"] | None = None
    hf_coeff_lr: float = eqx.field(static=True, default=0.0)

    def _exchanges(self):
        return tuple((ax, Bt) for ax, Bt in ((self.hf_coeff, self.B),
                                             (self.hf_coeff_lr, self.B_lr))
                     if ax != 0.0)

    def _unpack(self, Bb):
        z = jnp.zeros((Bb.shape[0], 1), Bb.dtype)
        return jnp.take(jnp.concatenate([Bb, z], axis=1), self.pos, axis=1)

    def _coulomb(self, Ptot):
        gamma = self.B @ (self.weight * Ptot[self.rows, self.cols])
        return gamma, 0.5 * jnp.dot(gamma, gamma)

    def _j(self, gamma):
        return self._unpack((self.B.T @ gamma)[None])[0]

    def _per_row(self, ncol):
        nao = self.pos.shape[0]
        return nao * nao + nao * ncol

    def energy(self, P, S, nocc):
        gamma, e = self._coulomb(jnp.sum(P, axis=0))
        for ax, Bt in self._exchanges():
            def kfun(Q, Bt=Bt):
                def block(Bb):
                    U = self._unpack(Bb)
                    return jnp.einsum("xml,ls,xsn->mn", U, Q, U)
                return _over_row_blocks(Bt, self._per_row(Q.shape[0]), block)
            e = e + _exchange_quadratic(kfun, P, ax)
        return e

    def energy_and_potential(self, P, S, nocc, idempotent=True):
        if not idempotent:
            return super().energy_and_potential(P, S, nocc, idempotent)
        gamma, e = self._coulomb(jnp.sum(P, axis=0))
        V = jnp.broadcast_to(self._j(gamma), P.shape)
        nspin = P.shape[0]
        dscale = 0.5 if nspin == 1 else 1.0
        for ax, Bt in self._exchanges():
            e_pref = -ax if nspin == 1 else -0.5 * ax
            for sigma, n in enumerate(nocc):
                if n == 0:
                    continue
                C = jax.lax.stop_gradient(
                    _rik_occ_orbitals(P[sigma], S, n, dscale))

                def block(Bb, C=C):
                    G = jnp.einsum("xml,li->xmi", self._unpack(Bb), C)
                    return jnp.einsum("xmi,xni->mn", G, G)

                KK = _over_row_blocks(Bt, self._per_row(n), block)
                e = e + e_pref * jnp.vdot(C, KK @ C)
                V = V.at[sigma].add(-ax * KK)
        return e, V

    def energy_and_grad_orbitals(self, Cs, ws, S, nocc):
        nspin = len(Cs)
        w_full = 2.0 if nspin == 1 else 1.0
        Ptot = sum((C * w) @ C.T for C, w in zip(Cs, ws))
        gamma, e = self._coulomb(Ptot)
        J = self._j(gamma)
        grads = [2.0 * (J @ C) * w for C, w in zip(Cs, ws)]
        for ax, Bt in self._exchanges():
            e_pref = -ax if nspin == 1 else -0.5 * ax
            for s, (C, w) in enumerate(zip(Cs, ws)):
                if C.shape[1] == 0:
                    continue
                scale = jnp.sqrt(w / w_full)
                Ct = C * scale

                def block(Bb, Ct=Ct):
                    G = jnp.einsum("xml,li->xmi", self._unpack(Bb), Ct)
                    D = jnp.einsum("mi,xmj->xij", Ct, G)
                    return jnp.sum(D * D), jnp.einsum("xmi,xij->mj", G, D)

                raw, GD = _over_row_blocks(Bt, self._per_row(C.shape[1]),
                                           block)
                e = e + e_pref * raw
                grads[s] = grads[s] + 4.0 * e_pref * GD * scale
        return e, tuple(grads)


class ShardedDFCoulomb(CoulombTerm):
    """Materialized RI-J with the 3-center tensor sharded over the aux axis.

    Each device holds a ``(nao², naux/ndev)`` slab of ``int3c`` (built
    directly in shards; see :func:`dftax.ks.shard._build_int3c_sharded`) and
    contracts its own slice of ``γ_P = Σ_μν (μν|P) P_μν``; the slices are
    ``all_gather``-ed (γ is a tiny naux-vector) and the metric quadratic form
    ``½ γᵀ V⁻¹ γ`` is evaluated replicated. Padded aux columns carry exact
    zeros in both γ and the (zero-padded) metric inverse, so the energy is the
    single-device value bit-for-bit up to summation order.

    Hybrid exact exchange uses ``V⁻¹ = LLᵀ`` and
    ``Tr(P K(P)) = Σ_X ⟨P W_X P, W_X⟩`` with ``W = int3c·L``: each device
    builds only its own aux-slab of ``W`` (an all-to-all done as ``ndev``
    rounds of ``psum``, one per destination slab), contracts its slab's
    exchange partial against the replicated density, and the scalar partials
    are ``psum``-reduced; per-device memory stays O(nao²·naux/ndev). Padded
    aux rows are zero in ``int3c`` and null in the padded metric, so they
    contribute exactly nothing to J or K.
    """

    int3c: Float[Array, "nao2 nauxp"]
    int2c_inv: Float[Array, "nauxp nauxp"]
    devices: tuple = eqx.field(static=True)
    nao: int = eqx.field(static=True, default=0)
    hf_coeff: float = eqx.field(static=True, default=0.0)
    int3c_lr: Float[Array, "nao2 nauxp"] | None = None
    int2c_inv_lr: Float[Array, "nauxp nauxp"] | None = None
    hf_coeff_lr: float = eqx.field(static=True, default=0.0)

    def energy(self, P, S, nocc):
        import numpy as np
        from jax import shard_map

        jmesh = jax.sharding.Mesh(np.asarray(self.devices), ("aux",))
        spec = jax.sharding.PartitionSpec
        ndev = len(self.devices)
        nao = self.nao
        slab = self.int3c.shape[1] // ndev
        ax = self.hf_coeff
        ax_lr = self.hf_coeff_lr
        nspin = P.shape[0]
        Lf = _rik_cholesky(self.int2c_inv) if ax != 0.0 else self.int2c_inv
        Lf_lr = (_rik_cholesky(self.int2c_inv_lr)
                 if ax_lr != 0.0 else self.int2c_inv)

        def exchange(flat, Lfull, Pst):
            my = jax.lax.axis_index("aux")
            rows = jax.lax.dynamic_slice_in_dim(Lfull, my * slab, slab, axis=0)
            W = jnp.zeros_like(flat)
            for d in range(ndev):
                part_d = flat @ rows[:, d * slab:(d + 1) * slab]
                W = jnp.where(my == d, jax.lax.psum(part_d, "aux"), W)
            W = W.reshape(nao, nao, slab)

            def tr_pkp(Q):
                QW = jnp.einsum("ls,mlX->msX", Q, W)
                return jax.lax.psum(
                    jnp.einsum("mn,nsX,msX->", Q, W, QW), "aux"
                )

            if nspin == 1:
                return 0.5 * tr_pkp(Pst[0])
            return tr_pkp(Pst[0]) + tr_pkp(Pst[1])

        def part(t3, vinv, Lfull, t3lr, Llr, Pst):
            Ptot = jnp.sum(Pst, axis=0)
            g_local = t3.T @ Ptot.reshape(-1)
            g = jax.lax.all_gather(g_local, "aux", tiled=True)
            e = 0.5 * jnp.dot(g, vinv @ g)
            if ax != 0.0:
                e = e - 0.5 * ax * exchange(t3, Lfull, Pst)
            if ax_lr != 0.0:
                e = e - 0.5 * ax_lr * exchange(t3lr, Llr, Pst)
            return e

        t3lr = self.int3c_lr if ax_lr != 0.0 else self.int3c
        return shard_map(
            part, mesh=jmesh,
            in_specs=(spec(None, "aux"), spec(), spec(),
                      spec(None, "aux"), spec(), spec()),
            out_specs=spec(), check_vma=False,
        )(self.int3c, self.int2c_inv, Lf, t3lr, Lf_lr, P)


class StreamedDFCoulomb(CoulombTerm):
    basis: BasisData
    aux_basis: BasisData
    int2c_inv: Float[Array, "naux naux"]
    pairs: tuple[Array, Array, Float[Array, "npair"]] | None
    chunk: int = eqx.field(static=True)
    hf_coeff: float = eqx.field(static=True)
    int2c_inv_lr: Float[Array, "naux naux"] | None = None
    hf_coeff_lr: float = eqx.field(static=True, default=0.0)
    omega: float = eqx.field(static=True, default=0.0)
    slab_plans: tuple | None = eqx.field(static=True, default=None)

    def _rik_sum(self, e, P, S, nocc, metric_inv, ax, omega):
        if P.shape[0] == 1:
            return e + _streamed_df_rik(
                self.basis, self.aux_basis, metric_inv,
                S, nocc[0], P[0], 0.5, -ax, -ax, omega,
                slabs=self.slab_plans,
            )
        for Ps, n in zip(P, nocc):
            e = e + _streamed_df_rik(
                self.basis, self.aux_basis, metric_inv,
                S, n, Ps, 1.0, -0.5 * ax, -ax, omega,
                slabs=self.slab_plans,
            )
        return e

    def energy(self, P, S, nocc):
        Ptot = jnp.sum(P, axis=0)
        if self.slab_plans is not None:
            e = _streamed_df_rij_slabs(
                self.basis, self.aux_basis, self.int2c_inv, Ptot,
                self.slab_plans,
            )
        else:
            e = _streamed_df_rij(
                self.basis, self.aux_basis, self.int2c_inv, Ptot,
                self.chunk, self.pairs,
            )
        if self.hf_coeff != 0.0:
            e = self._rik_sum(e, P, S, nocc, self.int2c_inv, self.hf_coeff,
                              None)
        if self.hf_coeff_lr != 0.0:
            e = self._rik_sum(e, P, S, nocc, self.int2c_inv_lr,
                              self.hf_coeff_lr, self.omega)
        return e


class StreamedDFForcesCoulomb(CoulombTerm):
    basis: BasisData
    aux_basis: BasisData
    int2c_inv: Float[Array, "naux naux"]
    Zs: tuple[Array, ...]
    hf_coeff: float = eqx.field(static=True)
    slab_plans: tuple = eqx.field(static=True)
    int2c_inv_lr: Float[Array, "naux naux"] | None = None
    hf_coeff_lr: float = eqx.field(static=True, default=0.0)
    omega: float = eqx.field(static=True, default=0.0)

    def energy(self, P, S, nocc):
        Ptot = jnp.sum(P, axis=0)
        e = _streamed_df_rij_slabs(
            self.basis, self.aux_basis, self.int2c_inv, Ptot, self.slab_plans
        )
        for ax, vinv, om in (
            (self.hf_coeff, self.int2c_inv, None),
            (self.hf_coeff_lr, self.int2c_inv_lr, self.omega),
        ):
            if ax != 0.0:
                prefs = (-ax,) if len(self.Zs) == 1 else (-0.5 * ax, -0.5 * ax)
                e = e + _streamed_df_rik_frozen(
                    self.basis, self.aux_basis, vinv,
                    S, self.Zs, prefs, self.slab_plans, om,
                )
        return e


class _NoCoulomb(CoulombTerm):
    def energy(self, P, S, nocc):
        return jnp.asarray(0.0)


def _packed_dmat(T, pos, Z):
    naux = T.shape[0]
    nao, no = Z.shape
    b = max(1, min(naux, _K_BLOCK_ELEMS // max(1, nao * nao + nao * no)))

    @jax.checkpoint
    def block(Tb):
        z = jnp.zeros((Tb.shape[0], 1), Tb.dtype)
        U = jnp.take(jnp.concatenate([Tb, z], axis=1), pos, axis=1)
        G = jnp.einsum("xmn,nj->xmj", U, Z)
        return jnp.einsum("mi,xmj->xij", Z, G)

    nfull, tail = divmod(naux, b)
    parts = []
    if nfull:
        def body(carry, i):
            return carry, block(jax.lax.dynamic_slice_in_dim(T, i * b, b, 0))
        _, D = jax.lax.scan(body, None, jnp.arange(nfull))
        parts.append(D.reshape(nfull * b, no, no))
    if tail:
        parts.append(block(T[nfull * b:]))
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts)


class PackedDFForcesCoulomb(CoulombTerm):
    T: Float[Array, "naux npair"]
    int2c_inv: Float[Array, "naux naux"]
    pos: Int[Array, "nao nao"]
    rows: Int[Array, "npair"]
    cols: Int[Array, "npair"]
    weight: Float[Array, "npair"]
    Zs: tuple
    hf_coeff: float = eqx.field(static=True)
    T_lr: Float[Array, "naux npair"] | None = None
    int2c_inv_lr: Float[Array, "naux naux"] | None = None
    hf_coeff_lr: float = eqx.field(static=True, default=0.0)

    def energy(self, P, S, nocc):
        p = self.weight * jnp.sum(P, axis=0)[self.rows, self.cols]
        gamma = self.T @ p
        e = 0.5 * jnp.dot(gamma, self.int2c_inv @ gamma)
        for ax, Tt, vinv in ((self.hf_coeff, self.T, self.int2c_inv),
                             (self.hf_coeff_lr, self.T_lr, self.int2c_inv_lr)):
            if ax == 0.0:
                continue
            prefs = (-ax,) if len(self.Zs) == 1 else (-0.5 * ax, -0.5 * ax)
            for Z, pref in zip(self.Zs, prefs):
                if Z.shape[1] == 0:
                    continue
                M = Z.T @ S @ Z
                D = _packed_dmat(Tt, self.pos, Z)
                naux = D.shape[0]
                Q = jnp.linalg.solve(M, D)
                QF = Q.reshape(naux, -1)
                QT = Q.transpose(0, 2, 1).reshape(naux, -1)
                e = e + pref * jnp.vdot(QF, vinv @ QT)
        return e


class ShardedStreamedDFCoulomb(CoulombTerm):
    """Streamed RI-J with the auxiliary axis divided across a device mesh.

    The streamed backend never builds the ``(nao², naux)`` tensor, which is the
    only thing that fits at protein scale; this divides its auxiliary axis over
    the mesh as well, so the per-device work is ``naux/ndev`` auxiliary
    functions streamed ``chunk`` at a time. Combining the two is the point: the
    materialized aux-sharded backend still holds ``nao²·naux/ndev`` per device,
    which for insulin at triple zeta is 15 TiB.

    Each device streams its own contiguous slice of the auxiliary index range
    into its piece of ``γ``, the pieces are ``all_gather``-ed (γ is a
    ``naux``-vector, so this is negligible traffic), and the metric quadratic
    form ``½ γᵀ V⁻¹ γ`` is evaluated replicated -- the same decomposition
    :class:`ShardedDFCoulomb` uses, and for the same reason: γ is the only
    quantity that has to cross devices.

    The auxiliary basis is *replicated*, not sliced. The streamed kernel looks
    an auxiliary function up by index (``_eri3c_elem(..., k)``) instead of
    slicing a stored slab, so a device needs the whole (small) auxiliary basis
    and only its own range of ``k``. Indices are assigned contiguously and the
    range is padded to a multiple of the device count, with the padding masked
    to zero, so the gathered γ is already in auxiliary order and truncating it
    to ``naux`` is exact rather than needing a position map.

    Hybrids shard too, on a different axis: streamed RI-K sums over occupied
    orbitals, so each device scans its own slice of them and the partials are
    ``psum``-reduced (:func:`_rik_energy_sharded`). Both halves of its
    ``custom_vjp`` shard the same way, so the analytic exchange Fock it returns
    is unchanged. Range-separated hybrids ride along on the attenuated metric.
    """

    basis: BasisData
    aux_basis: BasisData
    int2c_inv: Float[Array, "naux naux"]
    pairs: tuple[Array, Array, Float[Array, "npair"]] | None
    devices: tuple = eqx.field(static=True)
    chunk: int = eqx.field(static=True)
    hf_coeff: float = eqx.field(static=True, default=0.0)
    int2c_inv_lr: Float[Array, "naux naux"] | None = None
    hf_coeff_lr: float = eqx.field(static=True, default=0.0)
    omega: float = eqx.field(static=True, default=0.0)

    def _rik_sum(self, e, P, S, nocc, metric_inv, ax, omega):
        if P.shape[0] == 1:
            return e + _streamed_df_rik(
                self.basis, self.aux_basis, metric_inv, S, nocc[0], P[0],
                0.5, -ax, -ax, omega, devices=self.devices,
            )
        for Ps, n in zip(P, nocc):
            e = e + _streamed_df_rik(
                self.basis, self.aux_basis, metric_inv, S, n, Ps,
                1.0, -0.5 * ax, -ax, omega, devices=self.devices,
            )
        return e

    def energy(self, P, S, nocc):
        import numpy as np
        from jax import shard_map

        jmesh = jax.sharding.Mesh(np.asarray(self.devices), ("aux",))
        spec = jax.sharding.PartitionSpec
        rep, sh = spec(), spec("aux")
        ndev = len(self.devices)
        naux = self.aux_basis.centers.shape[0]
        slab = -(-naux // ndev)

        k_all = jnp.arange(ndev * slab)
        k_safe = jnp.minimum(k_all, naux - 1)
        keep = (k_all < naux).astype(self.int2c_inv.dtype)
        chunk, pairs = self.chunk, self.pairs

        def part(basis, aux, vinv, kk, mask, Pf):
            g_local = mask * _streamed_gamma(basis, aux, Pf, chunk, pairs, kk)
            g = jax.lax.all_gather(g_local, "aux", tiled=True)[:naux]
            return 0.5 * jnp.dot(g, vinv @ g)

        tree_rep = jax.tree.map(lambda _: rep, self.basis)
        aux_rep = jax.tree.map(lambda _: rep, self.aux_basis)
        e = shard_map(
            part, mesh=jmesh,
            in_specs=(tree_rep, aux_rep, rep, sh, sh, rep),
            out_specs=rep, check_vma=False,
        )(self.basis, self.aux_basis, self.int2c_inv, k_safe, keep,
          jnp.sum(P, axis=0))
        if self.hf_coeff != 0.0:
            e = self._rik_sum(e, P, S, nocc, self.int2c_inv, self.hf_coeff,
                              None)
        if self.hf_coeff_lr != 0.0:
            e = self._rik_sum(e, P, S, nocc, self.int2c_inv_lr,
                              self.hf_coeff_lr, self.omega)
        return e


class XCTerm(eqx.Module):
    @abc.abstractmethod
    def energy(self, P: Float[Array, "nspin nao nao"]) -> Scalar:
        raise NotImplementedError

    def energy_and_potential(
        self, P: Float[Array, "nspin nao nao"]
    ) -> tuple[Scalar, Float[Array, "nspin nao nao"]]:
        return jax.value_and_grad(self.energy)(P)

    def energy_and_grad_orbitals(self, Cs, ws):
        P = jnp.stack([(C * w) @ C.T for C, w in zip(Cs, ws)])
        e, V = self.energy_and_potential(P)
        return e, tuple(((V[s] + V[s].T) @ C) * w
                        for s, (C, w) in enumerate(zip(Cs, ws)))


class GridXC(XCTerm):
    ao: Float[Array, "ng nao"]
    dao: Float[Array, "ng nao 3"]
    weights: Float[Array, "ng"]
    xc: XCFunctional = eqx.field(static=True)
    coords: Float[Array, "ng 3"] | None = None

    def density(
        self, P: Float[Array, "nspin nao nao"]
    ) -> tuple[Float[Array, "ng"], Float[Array, "ng 3"]]:
        Ptot = jnp.sum(P, axis=0)
        rho = jnp.einsum("gm,mn,gn->g", self.ao, Ptot, self.ao)
        grad_rho = 2.0 * jnp.einsum("gm,mn,gnx->gx", self.ao, Ptot, self.dao)
        return rho, grad_rho

    def _density_spin(self, P: Float[Array, "nao nao"]):
        rho = jnp.einsum("gm,mn,gn->g", self.ao, P, self.ao)
        grad_rho = 2.0 * jnp.einsum("gm,mn,gnx->gx", self.ao, P, self.dao)
        return rho, grad_rho

    def _tau(self, P: Float[Array, "nao nao"]) -> Float[Array, "ng"]:
        return 0.5 * jnp.einsum("gmx,mn,gnx->g", self.dao, P, self.dao)

    def energy(self, P):
        mgga = self.xc.xc_type == "MGGA"
        if P.shape[0] == 1:
            rho, grad_rho = self.density(P)
            return self._energy_closed(
                rho, grad_rho, self._tau(P[0]) if mgga else None)
        rho_a, grad_a = self._density_spin(P[0])
        rho_b, grad_b = self._density_spin(P[1])
        return self._energy_spin(
            (rho_a, grad_a, self._tau(P[0]) if mgga else None),
            (rho_b, grad_b, self._tau(P[1]) if mgga else None))

    def _orbital_channel(self, C):
        psi = self.ao @ C
        dpsi = jnp.einsum("gmx,mi->gix", self.dao, C)
        rho = jnp.sum(psi * psi, axis=1)
        grad = 2.0 * jnp.einsum("gi,gix->gx", psi, dpsi)
        tau = (0.5 * jnp.sum(dpsi * dpsi, axis=(1, 2))
               if self.xc.xc_type == "MGGA" else None)
        return rho, grad, tau

    def energy_from_orbitals(self, Cs):
        chans = [self._orbital_channel(C) for C in Cs]
        if len(chans) == 1:
            return self._energy_closed(*chans[0])
        return self._energy_spin(*chans)

    def energy_and_grad_orbitals(self, Cs, ws):
        scales = tuple(jnp.sqrt(w) for w in ws)
        return jax.value_and_grad(
            lambda cs: self.energy_from_orbitals(
                tuple(c * sc for c, sc in zip(cs, scales))))(tuple(Cs))

    def _energy_closed(self, rho, grad_rho, tau):
        if self.xc.xc_type == "MGGA":
            e = xc_energy(self.xc, rho, self.weights, grad_rho=grad_rho,
                          tau=tau)
        else:
            gr = grad_rho if self.xc.xc_type == "GGA" else None
            e = xc_energy(self.xc, rho, self.weights, grad_rho=gr)
        if self.xc.nlc_b != 0.0:
            from dftax.energy.vv10 import vv10_energy

            e = e + vv10_energy(
                rho, jnp.sum(grad_rho * grad_rho, axis=-1), self.coords,
                self.weights, self.xc.nlc_b, self.xc.nlc_c,
            )
        return e

    def _energy_spin(self, chan_a, chan_b):
        rho_a, grad_a, tau_a = chan_a
        rho_b, grad_b, tau_b = chan_b
        rho_tot = rho_a + rho_b
        mask = rho_tot > 1e-10
        ta = rho_a > 1e-10
        tb = rho_b > 1e-10
        rho_stack = jnp.stack(
            [jnp.where(ta, rho_a, 1e-10), jnp.where(tb, rho_b, 1e-10)], axis=-1
        )
        if self.xc.xc_type == "MGGA":
            grad_a = jnp.where(ta[:, None], grad_a, 0.0)
            grad_b = jnp.where(tb[:, None], grad_b, 0.0)
            grad_stack = jnp.stack([grad_a, grad_b], axis=-1)
            tau_stack = jnp.stack(
                [jnp.where(ta, tau_a, 1e-10),
                 jnp.where(tb, tau_b, 1e-10)], axis=-1,
            )
            eps = xc_potential(
                self.xc, rho_stack, grad_rho=grad_stack, tau=tau_stack
            )
        elif self.xc.xc_type == "GGA":
            grad_a = jnp.where(ta[:, None], grad_a, 0.0)
            grad_b = jnp.where(tb[:, None], grad_b, 0.0)
            grad_stack = jnp.stack([grad_a, grad_b], axis=-1)
            eps = xc_potential(self.xc, rho_stack, grad_rho=grad_stack)
        else:
            eps = xc_potential(self.xc, rho_stack)
        e = jnp.sum(jnp.where(mask, self.weights * eps * rho_tot, 0.0))
        if self.xc.nlc_b != 0.0:
            from dftax.energy.vv10 import vv10_energy

            grad_tot = grad_a + grad_b
            e = e + vv10_energy(
                rho_tot, jnp.sum(grad_tot * grad_tot, axis=-1), self.coords,
                self.weights, self.xc.nlc_b, self.xc.nlc_c,
            )
        return e


class StreamedGridXC(XCTerm):
    basis: BasisData
    grid_coords: Float[Array, "ng 3"]
    weights: Float[Array, "ng"]
    chunk: int = eqx.field(static=True)
    xc: XCFunctional = eqx.field(static=True)

    def energy(self, P):
        if P.shape[0] == 1:
            return _streamed_e_xc(
                self.xc, self.basis, self.grid_coords, self.weights, P[0], self.chunk
            )
        return _streamed_e_xc_spin(
            self.xc, self.basis, self.grid_coords, self.weights,
            P[0], P[1], self.chunk,
        )

    def energy_and_potential(self, P):
        return _streamed_e_and_v(
            self.xc, self.basis, self.grid_coords, self.weights, P, self.chunk
        )


def _screened_sub_basis(basis, cart, cmask, sph):
    sub = eqx.tree_at(
        lambda t: (t.centers, t.exponents, t.coefficients, t.angular),
        basis,
        (basis.centers[cart], basis.exponents[cart],
         basis.coefficients[cart] * cmask[:, None], basis.angular[cart]),
    )
    if basis.cart2sph is not None:
        sub = eqx.tree_at(lambda t: t.cart2sph, sub,
                          basis.cart2sph[cart][:, sph])
    return sub


def _screened_rho_block(basis, P, cart, sph, cmask, smask, pts, need_grad):
    sub = _screened_sub_basis(basis, cart, cmask, sph)
    Psub = P[sph][:, sph]

    def one(r):
        if not need_grad:
            ao = eval_gto(sub, r) * smask
            return ao @ Psub @ ao, jnp.zeros(3), jnp.zeros(())
        ao, dao = eval_gto_and_grad(sub, r)
        ao = ao * smask
        dao = dao * smask[:, None]
        rho = ao @ Psub @ ao
        grad = 2.0 * (ao @ Psub) @ dao
        tau = 0.5 * jnp.einsum("mx,mn,nx->", dao, Psub, dao)
        return rho, grad, tau

    return jax.vmap(one)(pts)


def _screened_e_xc(xc, basis, coords, weights, P, buckets, block, n_block):
    gga = xc.xc_type == "GGA"
    mgga = xc.xc_type == "MGGA"
    need = gga or mgga
    cg = coords.reshape(n_block, block, 3)
    wg = weights.reshape(n_block, block)
    total = jnp.zeros(())

    for bucket in buckets:
        ids = bucket.block_ids

        def body(args, _ids=ids):
            cart, sph, cm, sm, i = args
            return _screened_block_e(xc, basis, P, cart, sph, cm, sm, cg[i],
                                     wg[i], gga, mgga, need)

        total = total + jnp.sum(jax.lax.map(
            jax.checkpoint(body),
            (bucket.cart, bucket.sph, bucket.cart_mask, bucket.sph_mask, ids),
        ))
    return total


def _screened_block_e(xc, basis, P, cart, sph, cm, sm, pts, w, gga, mgga,
                      need):
    rho, grad, tau = _screened_rho_block(basis, P, cart, sph, cm, sm, pts,
                                         need)
    mask = rho > 1e-10
    safe = jnp.where(mask, rho, 1.0)
    if mgga:
        eps = jax.vmap(xc)(safe, jnp.where(mask[:, None], grad, 0.0),
                           jnp.where(mask, tau, 1.0))
    elif gga:
        eps = jax.vmap(xc)(safe, jnp.where(mask[:, None], grad, 0.0))
    else:
        eps = jax.vmap(xc)(safe)
    return jnp.sum(jnp.where(mask, w * eps * rho, 0.0))


def _screened_block_e_spin(xc, basis, Pa, Pb, cart, sph, cm, sm, pts, w, gga,
                           mgga, need):
    sub = _screened_sub_basis(basis, cart, cm, sph)
    Pas, Pbs = Pa[sph][:, sph], Pb[sph][:, sph]

    def point(r, wt):
        if gga or mgga:
            ao, dao = eval_gto_and_grad(sub, r)
            ao, dao = ao * sm, dao * sm[:, None]
        else:
            ao = eval_gto(sub, r) * sm
        rho_a = ao @ Pas @ ao
        rho_b = ao @ Pbs @ ao
        rho_tot = rho_a + rho_b
        mask = rho_tot > 1e-10
        ta, tb = rho_a > 1e-10, rho_b > 1e-10
        rho2 = jnp.stack([jnp.where(ta, rho_a, 1e-10),
                          jnp.where(tb, rho_b, 1e-10)])
        if gga or mgga:
            ga = jnp.where(ta, 2.0 * (ao @ Pas) @ dao, 0.0)
            gb = jnp.where(tb, 2.0 * (ao @ Pbs) @ dao, 0.0)
            if mgga:
                tau2 = jnp.stack([
                    jnp.where(ta, 0.5 * jnp.einsum(
                        "mx,mn,nx->", dao, Pas, dao), 1e-10),
                    jnp.where(tb, 0.5 * jnp.einsum(
                        "mx,mn,nx->", dao, Pbs, dao), 1e-10),
                ])
                eps = xc(rho2, jnp.stack([ga, gb], axis=-1), tau2)
            else:
                eps = xc(rho2, jnp.stack([ga, gb], axis=-1))
        else:
            eps = xc(rho2)
        return jnp.where(mask, wt * eps * rho_tot, 0.0)

    return jnp.sum(jax.vmap(point)(pts, w))


def _sum_e_and_v(piece, P, xs, init_v=None):
    v0 = jnp.zeros_like(P) if init_v is None else init_v

    def step(carry, x):
        e_acc, v_acc = carry
        e, pull = jax.vjp(lambda Q: piece(Q, x), P)
        (v,) = pull(jnp.ones((), dtype=e.dtype))
        return (e_acc + e, v_acc + v), None

    (E, V), _ = jax.lax.scan(step, (jnp.zeros(()), v0), xs)
    return E, V


def _streamed_e_and_v(xc, basis, coords, weights, P, chunk):
    ng = coords.shape[0]
    nchunk = max(1, min(int(chunk), ng))
    n_full = ng // nchunk
    tail = ng - n_full * nchunk

    def piece(Q, args):
        c, w = args
        if Q.shape[0] == 1:
            return _streamed_e_xc(xc, basis, c, w, Q[0], nchunk,
                                  checkpoint=False)
        return _streamed_e_xc_spin(xc, basis, c, w, Q[0], Q[1], nchunk,
                                   checkpoint=False)

    E = jnp.zeros(())
    V = jnp.zeros_like(P)
    if n_full:
        E, V = _sum_e_and_v(
            piece, P,
            (coords[:n_full * nchunk].reshape(n_full, nchunk, 3),
             weights[:n_full * nchunk].reshape(n_full, nchunk)),
        )
    if tail:
        ct, wt = coords[n_full * nchunk:], weights[n_full * nchunk:]
        e_t, pull = jax.vjp(
            lambda Q: (_streamed_e_xc(xc, basis, ct, wt, Q[0], tail,
                                      checkpoint=False)
                       if Q.shape[0] == 1 else
                       _streamed_e_xc_spin(xc, basis, ct, wt, Q[0], Q[1],
                                           tail, checkpoint=False)),
            P)
        (v_t,) = pull(jnp.ones((), dtype=e_t.dtype))
        E, V = E + e_t, V + v_t
    return E, V


def _screened_e_and_v(xc, basis, coords, weights, P, buckets, block, n_block):
    gga = xc.xc_type == "GGA"
    mgga = xc.xc_type == "MGGA"
    need = gga or mgga
    cg = coords.reshape(n_block, block, 3)
    wg = weights.reshape(n_block, block)
    E = jnp.zeros(())
    V = jnp.zeros_like(P)

    for bucket in buckets:
        def body(Q, args):
            cart, sph, cm, sm, i = args
            if Q.shape[0] == 1:
                return _screened_block_e(xc, basis, Q[0], cart, sph, cm, sm,
                                         cg[i], wg[i], gga, mgga, need)
            return _screened_block_e_spin(xc, basis, Q[0], Q[1], cart, sph,
                                          cm, sm, cg[i], wg[i], gga, mgga,
                                          need)

        e_b, v_b = _sum_e_and_v(
            body, P,
            (bucket.cart, bucket.sph, bucket.cart_mask, bucket.sph_mask,
             bucket.block_ids),
        )
        E, V = E + e_b, V + v_b
    return E, V


def _screened_e_xc_spin(xc, basis, coords, weights, Pa, Pb, buckets, block,
                        n_block):
    gga = xc.xc_type == "GGA"
    mgga = xc.xc_type == "MGGA"
    cg = coords.reshape(n_block, block, 3)
    wg = weights.reshape(n_block, block)
    total = jnp.zeros(())

    for bucket in buckets:
        def body(args):
            cart, sph, cm, sm, i = args
            return _screened_block_e_spin(xc, basis, Pa, Pb, cart, sph, cm,
                                          sm, cg[i], wg[i], gga, mgga,
                                          gga or mgga)

        total = total + jnp.sum(jax.lax.map(
            jax.checkpoint(body),
            (bucket.cart, bucket.sph, bucket.cart_mask, bucket.sph_mask,
             bucket.block_ids),
        ))
    return total


class ScreenedGridXC(XCTerm):
    basis: BasisData
    grid_coords: Float[Array, "ng 3"]
    weights: Float[Array, "ng"]
    buckets: tuple
    block: int = eqx.field(static=True)
    n_block: int = eqx.field(static=True)
    xc: XCFunctional = eqx.field(static=True)

    def energy(self, P):
        if P.shape[0] == 1:
            return _screened_e_xc(self.xc, self.basis, self.grid_coords,
                                  self.weights, P[0], self.buckets, self.block,
                                  self.n_block)
        return _screened_e_xc_spin(self.xc, self.basis, self.grid_coords,
                                   self.weights, P[0], P[1], self.buckets,
                                   self.block, self.n_block)

    def energy_and_potential(self, P):
        return _screened_e_and_v(
            self.xc, self.basis, self.grid_coords, self.weights, P,
            self.buckets, self.block, self.n_block,
        )


class ShardedGridXC(XCTerm):
    """XC integral sharded over grid points across a 1-D device mesh.

    ``inner`` is an ordinary :class:`GridXC` or :class:`StreamedGridXC` whose
    grid-axis arrays are padded to a multiple of the device count and laid out
    sharded over the mesh (see :func:`dftax.ks.shard._pad_shard_grid`). The
    energy runs each device's slice through the *unmodified* inner math under
    ``shard_map`` (the density is replicated, the partial energies are
    ``psum``-reduced), so the quadrature is exactly the single-device sum and
    the collective differentiates natively (autodiff Fock, forces).
    """

    inner: XCTerm
    devices: tuple = eqx.field(static=True)

    def energy(self, P):
        import numpy as np
        from jax import shard_map

        jmesh = jax.sharding.Mesh(np.asarray(self.devices), ("grid",))
        spec = jax.sharding.PartitionSpec
        rep = spec()
        g = spec("grid")

        if isinstance(self.inner, GridXC):
            xc = self.inner.xc

            def part(ao, dao, w, Pf):
                local = GridXC(ao=ao, dao=dao, weights=w, xc=xc)
                return jax.lax.psum(local.energy(Pf), "grid")

            args = (self.inner.ao, self.inner.dao, self.inner.weights)
            in_specs = (g, g, g, rep)
        else:
            chunk, xc = self.inner.chunk, self.inner.xc

            def part(basis, gc, w, Pf):
                local = StreamedGridXC(
                    basis=basis, grid_coords=gc, weights=w, chunk=chunk, xc=xc
                )
                return jax.lax.psum(local.energy(Pf), "grid")

            args = (self.inner.basis, self.inner.grid_coords, self.inner.weights)
            in_specs = (jax.tree.map(lambda _: rep, self.inner.basis), g, g, rep)

        return shard_map(
            part, mesh=jmesh, in_specs=in_specs, out_specs=rep
        )(*args, P)


def _make_coulomb(spec, basis, eri, int3c, int2c_inv, pairs, hf_coeff,
                  eri_lr=None, int3c_lr=None, int2c_inv_lr=None,
                  hf_coeff_lr=0.0, omega=0.0, devices=None, slab_plans=None,
                  packed_plan=None):
    if isinstance(spec, DFSpec):
        if not isinstance(spec.auxbasis, BasisData):
            raise TypeError(
                "spec.auxbasis must be resolved to BasisData before assembly; "
                "the public constructors resolve basis-set names."
            )
        if spec.chunk is not None:
            if devices is not None:
                return ShardedStreamedDFCoulomb(
                    basis=basis, aux_basis=spec.auxbasis,
                    int2c_inv=int2c_inv, pairs=pairs,
                    devices=tuple(devices), chunk=spec.chunk,
                    hf_coeff=hf_coeff, int2c_inv_lr=int2c_inv_lr,
                    hf_coeff_lr=hf_coeff_lr, omega=omega,
                )
            return StreamedDFCoulomb(
                basis=basis, aux_basis=spec.auxbasis, int2c_inv=int2c_inv,
                pairs=pairs, chunk=spec.chunk, hf_coeff=hf_coeff,
                int2c_inv_lr=int2c_inv_lr, hf_coeff_lr=hf_coeff_lr,
                omega=omega, slab_plans=slab_plans,
            )
        if int2c_inv is None and int3c.ndim == 2:
            from dftax.integrals.eri3c_bucketed import packed_layout

            lay = packed_layout(packed_plan, basis.cart2sph is not None)
            return PackedDFCoulomb(
                B=int3c, pos=jnp.asarray(lay.pos), rows=jnp.asarray(lay.rows),
                cols=jnp.asarray(lay.cols), weight=jnp.asarray(lay.weight),
                hf_coeff=hf_coeff, B_lr=int3c_lr, hf_coeff_lr=hf_coeff_lr)
        if int2c_inv is None:
            return WhitenedDFCoulomb(B=int3c, hf_coeff=hf_coeff, B_lr=int3c_lr,
                                     hf_coeff_lr=hf_coeff_lr)
        return DFCoulomb(
            int3c=int3c, int2c_inv=int2c_inv, hf_coeff=hf_coeff,
            int3c_lr=int3c_lr, int2c_inv_lr=int2c_inv_lr,
            hf_coeff_lr=hf_coeff_lr,
        )
    if isinstance(spec, ExternalCoulombSpec):
        return _NoCoulomb()
    if spec.stream:
        if hf_coeff_lr != 0.0:
            raise NotImplementedError(
                "range-separated hybrids need a materialized backend: use "
                "exact() or df(chunk=None), not exact(stream=True)."
            )
        return StreamedExactCoulomb(basis=basis, hf_coeff=hf_coeff)
    return ExactCoulomb(
        eri=eri, hf_coeff=hf_coeff, eri_lr=eri_lr, hf_coeff_lr=hf_coeff_lr
    )
