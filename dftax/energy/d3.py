from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import jax.numpy as jnp
import numpy as np

_K1 = 16.0
_K3 = 4.0


@dataclass(frozen=True)
class D3BJSpec:
    method: str | None = None
    atm: bool = False


def d3bj(method: str | None = None, *, atm: bool = False) -> D3BJSpec:
    """D3(BJ) dispersion correction (two-body, optionally with ATM).

    Args:
        method: damping-parameter set name (``"pbe"``, ``"pbe0"``,
            ``"b3lyp"``, ``"cam-b3lyp"``, ``"r2scan"``); ``None`` resolves it
            from the functional the KS is built with.
        atm: add the Axilrod-Teller-Muto three-body term (``s9 = 1``); off by
            default, matching the common two-body D3(BJ) convention. The term
            is repulsive, O(nat³), and independent of the BJ damping
            parameters (it uses the pairwise van-der-Waals radii).
    """
    return D3BJSpec(method=method, atm=atm)


@lru_cache(maxsize=1)
def _tables():
    path = files("dftax.energy").joinpath("data", "d3bj.npz")
    with path.open("rb") as fh:
        npz = np.load(fh)
        out = {k: np.asarray(npz[k]) for k in npz.files}
    return out


def available_d3_methods() -> list[str]:
    return [str(m) for m in _tables()["methods"]]


def _d3_params(method: str) -> tuple[float, float, float, float]:
    t = _tables()
    methods = [str(m) for m in t["methods"]]
    key = method.lower()
    if key not in methods:
        raise ValueError(
            f"no D3(BJ) parameters for {method!r}; available: {methods}. "
            f"Pass d3bj('<method>') explicitly."
        )
    s6, a1, s8, a2 = (float(x) for x in t["params"][methods.index(key)])
    return s6, a1, s8, a2


def _c6_matrix(coords, z, t):
    rcov = jnp.asarray(t["rcov"])[z]
    cn_ref = jnp.asarray(t["cn_ref"])[z]
    c6_ref = jnp.asarray(t["c6"])[z[:, None], z[None, :]]

    nat = coords.shape[0]
    dvec = coords[:, None, :] - coords[None, :, :]
    r2 = jnp.sum(dvec * dvec, axis=-1) + jnp.eye(nat)
    r = jnp.sqrt(r2)
    off = 1.0 - jnp.eye(nat)

    rco = rcov[:, None] + rcov[None, :]
    cn = jnp.sum(off / (1.0 + jnp.exp(-_K1 * (rco / r - 1.0))), axis=1)

    dcn = cn[:, None] - cn_ref
    d2 = jnp.where(cn_ref >= 0.0, -_K3 * dcn * dcn, -jnp.inf)
    d2max = jnp.max(d2, axis=1, keepdims=True)
    w = jnp.exp(d2 - d2max)
    wij = w[:, None, :, None] * w[None, :, None, :]
    norm = jnp.clip(jnp.sum(wij, axis=(2, 3)), 1e-30)
    mask = (cn_ref >= 0.0)[:, None, :, None] & (cn_ref >= 0.0)[None, :, None, :]
    c6 = jnp.sum(jnp.where(mask, c6_ref, 0.0) * wij, axis=(2, 3)) / norm
    return c6, r2, off


def d3bj_energy(
    coords,
    numbers,
    params: tuple[float, float, float, float],
):
    t = _tables()
    z = jnp.asarray(np.asarray(numbers, dtype=np.int64))
    rr = jnp.asarray(t["r4r2"])[z]
    s6, a1, s8, a2 = params

    coords = jnp.asarray(coords).reshape(-1, 3)
    c6, r2, off = _c6_matrix(coords, z, t)

    c8 = 3.0 * c6 * rr[:, None] * rr[None, :]
    r0 = a1 * jnp.sqrt(jnp.clip(c8 / jnp.clip(c6, 1e-30), 0.0)) + a2

    e6 = c6 / (r2**3 + r0**6)
    e8 = c8 / (r2**4 + r0**8)
    return -0.5 * jnp.sum(off * (s6 * e6 + s8 * e8))


_RS9 = 4.0 / 3.0
_ALP = 14.0


def d3_atm_energy(coords, numbers, s9: float = 1.0):
    t = _tables()
    z = jnp.asarray(np.asarray(numbers, dtype=np.int64))
    coords = jnp.asarray(coords).reshape(-1, 3)
    c6, r2, off = _c6_matrix(coords, z, t)
    srvdw = _RS9 * jnp.asarray(t["vdw"])[z[:, None], z[None, :]]

    c9 = s9 * jnp.sqrt(jnp.abs(
        c6[:, :, None] * c6[:, None, :] * c6[None, :, :]))
    r0 = srvdw[:, :, None] * srvdw[:, None, :] * srvdw[None, :, :]
    rr2 = r2[:, :, None] * r2[:, None, :] * r2[None, :, :]
    r1 = jnp.sqrt(rr2)
    r3 = r1 * rr2
    r5 = rr2 * r3

    triple = (off[:, :, None] * off[:, None, :] * off[None, :, :])
    fdamp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** ((_ALP + 2.0) / 3.0))

    r2ij, r2ik, r2jk = r2[:, :, None], r2[:, None, :], r2[None, :, :]
    s = ((r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik)
         * (-r2ij + r2jk + r2ik))
    ang = 0.375 * s / r5 + 1.0 / r3

    return jnp.sum(triple * ang * fdamp * c9) / 6.0


def _resolve_dispersion(spec, xc, charges, nelec=None):
    from dftax.energy.d4 import D4Spec, _d4_params, d4_energy

    if spec is None:
        return None
    numbers = np.rint(np.asarray(charges)).astype(np.int64)
    if isinstance(spec, D4Spec):
        method = spec.method if spec.method is not None else xc.name.lower()
        params = _d4_params(method)
        qtot = (float(np.sum(numbers)) - float(nelec)
                if nelec is not None else 0.0)
        atm = spec.atm
        return lambda coords: d4_energy(coords, numbers, params, qtot, atm)
    if not isinstance(spec, D3BJSpec):
        raise TypeError(
            f"dispersion must be a d3bj() or d4() spec, got {spec!r}")
    method = spec.method if spec.method is not None else xc.name.lower()
    params = _d3_params(method)
    if spec.atm:
        return lambda coords: (d3bj_energy(coords, numbers, params)
                               + d3_atm_energy(coords, numbers))
    return lambda coords: d3bj_energy(coords, numbers, params)
