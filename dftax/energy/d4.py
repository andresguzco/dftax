from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import jax.numpy as jnp
import numpy as np

from dftax.energy.d3 import _tables as _d3_tables

_KCN = 7.5
_CN_MAX_EEQ = 8.0
_K4 = 4.10451
_K5 = 19.08857
_K6 = 254.55531485519995
_WF = 6.0
_GA = 3.0
_GC = 2.0


@dataclass(frozen=True)
class D4Spec:
    method: str | None = None
    atm: bool = False


def d4(method: str | None = None, *, atm: bool = False) -> D4Spec:
    """Grimme's charge-dependent D4 dispersion correction (E. Caldeweyher et
    al., J. Chem. Phys. 150, 154122 (2019)).

    Pass the result as ``KS(..., dispersion=d4())``.

    Args:
        method: name of the damping-parameter set, e.g. ``"pbe0"``; ``None``
            takes the set of the functional the KS is built with.
        atm: add the Axilrod-Teller-Muto three-body term (``s9 = 1``). The
            parameter sets were fitted with it, and the reference D4 (dftd4)
            includes it by default; ``atm=False`` is the two-body energy.
    """
    return D4Spec(method=method, atm=atm)


@lru_cache(maxsize=1)
def _tables():
    path = files("dftax.energy").joinpath("data", "d4.npz")
    with path.open("rb") as fh:
        npz = np.load(fh)
        out = {k: np.asarray(npz[k]) for k in npz.files}
    return out


def _d4_params(method: str) -> tuple[float, float, float, float]:
    t = _tables()
    methods = [str(m) for m in t["methods"]]
    key = method.lower()
    if key not in methods:
        raise ValueError(
            f"no D4 parameters for {method!r}; available: {methods}. "
            f"Pass d4('<method>') explicitly."
        )
    s6, a1, s8, a2 = (float(x) for x in t["params"][methods.index(key)])
    return s6, a1, s8, a2


def _distances(coords):
    nat = coords.shape[0]
    dvec = coords[:, None, :] - coords[None, :, :]
    r2 = jnp.sum(dvec * dvec, axis=-1) + jnp.eye(nat)
    return jnp.sqrt(r2), r2, 1.0 - jnp.eye(nat)


def _cn(r, off, rcov_z, pair_weight=None, cn_max=None):
    r0 = rcov_z[:, None] + rcov_z[None, :]
    count = 0.5 * (1.0 + jax_erf(-_KCN * (r / r0 - 1.0)))
    if pair_weight is not None:
        count = count * pair_weight
    cn = jnp.sum(off * count, axis=1)
    if cn_max is not None:
        cn = jnp.log1p(math.exp(cn_max)) - jnp.log1p(jnp.exp(cn_max - cn))
    return cn


def jax_erf(x):
    import jax.scipy.special as jsp

    return jsp.erf(x)


def eeq_charges(coords, numbers, total_charge: float = 0.0):
    t = _tables()
    z = jnp.asarray(np.asarray(numbers, dtype=np.int64))
    coords = jnp.asarray(coords).reshape(-1, 3)
    nat = coords.shape[0]
    r, _, off = _distances(coords)

    cn = _cn(r, off, jnp.asarray(t["rcov"])[z], cn_max=_CN_MAX_EEQ)
    rhs_q = -jnp.asarray(t["eeq_chi"])[z] + jnp.sqrt(cn) * jnp.asarray(
        t["eeq_kcn"])[z]

    rad = jnp.asarray(t["eeq_rad"])[z]
    gamma = 1.0 / jnp.sqrt(rad[:, None] ** 2 + rad[None, :] ** 2)
    eta = jnp.asarray(t["eeq_eta"])[z] + math.sqrt(2.0 / math.pi) / rad
    A = jnp.where(off > 0.0, jax_erf(r * gamma) / r, 0.0) + jnp.diag(eta)

    M = jnp.zeros((nat + 1, nat + 1)).at[:nat, :nat].set(A)
    M = M.at[:nat, nat].set(1.0).at[nat, :nat].set(1.0)
    rhs = jnp.zeros(nat + 1).at[:nat].set(rhs_q).at[nat].set(total_charge)
    return jnp.linalg.solve(M, rhs)[:nat]


def _weights(cn, q, z, t):
    refcovcn = jnp.asarray(t["refcovcn"])[z]
    refc = jnp.asarray(t["refc"])[z]
    mask = refc > 0

    dcn = cn[:, None] - refcovcn
    tmp = jnp.exp(-dcn * dcn)
    pow1 = tmp ** _WF
    pow3 = pow1 + tmp ** (2 * _WF) + tmp ** (3 * _WF)
    expw = jnp.where(mask, jnp.where(refc == 3, pow3, pow1), 0.0)
    norm = jnp.clip(jnp.sum(expw, axis=-1, keepdims=True), 1e-300)
    gw = expw / norm

    zeff = jnp.asarray(t["zeff"])[z][:, None]
    gam = jnp.asarray(t["gam"])[z][:, None] * _GC
    qref = jnp.asarray(t["refq"])[z] + zeff
    qmod = q[:, None] + zeff
    safe = jnp.where(qmod > 0.0, qmod, 1.0)
    scale = jnp.exp(gam * (1.0 - qref / safe))
    zeta = jnp.where(qmod > 0.0, jnp.exp(_GA * (1.0 - scale)), math.exp(_GA))
    return jnp.where(mask, zeta * gw, 0.0)


def _c6_matrix(coords, numbers, q):
    t = _tables()
    z = jnp.asarray(np.asarray(numbers, dtype=np.int64))
    coords = jnp.asarray(coords).reshape(-1, 3)
    r, r2, off = _distances(coords)

    en = jnp.asarray(t["en"])[z]
    endiff = jnp.abs(en[:, None] - en[None, :])
    weight = _K4 * jnp.exp(-((endiff + _K5) ** 2) / _K6)
    cn = _cn(r, off, jnp.asarray(t["rcov"])[z], pair_weight=weight)

    w = _weights(cn, q, z, t)
    rc6 = jnp.asarray(t["rc6"])[z[:, None], z[None, :]]
    c6 = jnp.einsum("ijab,ia,jb->ij", rc6, w, w)
    return c6, r2, off


def d4_energy(
    coords,
    numbers,
    params: tuple[float, float, float, float],
    total_charge: float = 0.0,
    atm: bool = False,
):
    t3 = _d3_tables()
    z = jnp.asarray(np.asarray(numbers, dtype=np.int64))
    coords = jnp.asarray(coords).reshape(-1, 3)
    s6, a1, s8, a2 = params

    q = eeq_charges(coords, numbers, total_charge)
    c6, r2, off = _c6_matrix(coords, numbers, q)

    rr = jnp.asarray(t3["r4r2"])[z]
    c8 = 3.0 * c6 * rr[:, None] * rr[None, :]
    r0 = a1 * jnp.sqrt(jnp.clip(c8 / jnp.clip(c6, 1e-30), 0.0)) + a2

    e6 = c6 / (r2**3 + r0**6)
    e8 = c8 / (r2**4 + r0**8)
    e = -0.5 * jnp.sum(off * (s6 * e6 + s8 * e8))
    if atm:
        e = e + _atm_energy(coords, numbers, q, params)
    return e


_ALP = 16.0


def _atm_energy(coords, numbers, q, params, s9: float = 1.0):
    t3 = _d3_tables()
    _, a1, _, a2 = params
    z = jnp.asarray(np.asarray(numbers, dtype=np.int64))
    c6, r2, off = _c6_matrix(coords, numbers, jnp.zeros_like(
        jnp.asarray(q)))
    rr = jnp.asarray(t3["r4r2"])[z]
    r0bj = a1 * jnp.sqrt(3.0 * rr[:, None] * rr[None, :]) + a2

    c9 = s9 * jnp.sqrt(jnp.abs(
        c6[:, :, None] * c6[:, None, :] * c6[None, :, :]))
    r0 = r0bj[:, :, None] * r0bj[:, None, :] * r0bj[None, :, :]
    rr2 = r2[:, :, None] * r2[:, None, :] * r2[None, :, :]
    r1 = jnp.sqrt(rr2)
    r3 = r1 * rr2
    r5 = rr2 * r3

    triple = off[:, :, None] * off[:, None, :] * off[None, :, :]
    fdamp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** (_ALP / 3.0))
    r2ij, r2ik, r2jk = r2[:, :, None], r2[:, None, :], r2[None, :, :]
    s = ((r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik)
         * (-r2ij + r2jk + r2ik))
    ang = 0.375 * s / r5 + 1.0 / r3
    return jnp.sum(triple * ang * fdamp * c9) / 6.0
