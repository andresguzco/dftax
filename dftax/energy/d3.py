"""JAX-native D3(BJ) dispersion correction (Grimme et al., two-body).

The classic DFT-D3 model with Becke-Johnson damping:

    E_disp = -½ Σ_{A≠B} [ s6·C6_AB/(R⁶ + R0⁶) + s8·C8_AB/(R⁸ + R0⁸) ],
    R0 = a1·sqrt(C8/C6) + a2,

with geometry-dependent C6 coefficients interpolated over Grimme's reference
systems by Gaussian weights in the coordination numbers (k3 = 4), CN from the
smooth counting function ``1/(1 + exp(-k1(rcov/R - 1)))`` (k1 = 16). All
smooth, pure JAX: forces and Hessians come from autodiff, and the energy is
evaluated with traced coordinates inside :class:`~dftax.ks.energy.KS`, so the
rebuilt energies in ``forces``/``scf_batched`` carry the dispersion gradient
automatically.

Tables are vendored in ``data/d3bj.npz`` (extracted from tad-dftd3,
Apache-2.0; see ``scripts/gen_d3_data.py``). The Axilrod-Teller-Muto
three-body term is available opt-in (``d3bj(atm=True)``; off by default,
matching the common two-body D3(BJ) convention).

Usage (choices-as-values)::

    KS(mol, PBE(), dispersion=d3bj())          # parameters from the functional
    KS(mol, PBE(), dispersion=d3bj("b3lyp"))   # or explicit
    KS(mol, PBE(), dispersion=d3bj(atm=True))  # with the three-body term
"""

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
    """D3(BJ) dispersion spec (see :func:`d3bj`)."""

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
    """Damping-parameter sets available in the vendored tables."""
    return [str(m) for m in _tables()["methods"]]


def _d3_params(method: str) -> tuple[float, float, float, float]:
    """``(s6, a1, s8, a2)`` for a method name (case-insensitive)."""
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
    """CN-interpolated pairwise C6 and the squared-distance matrix.

    Shared by the two-body and ATM terms. Returns ``(c6, r2, off)`` with the
    diagonal of ``r2`` guarded to 1 and ``off`` the off-diagonal mask.
    """
    rcov = jnp.asarray(t["rcov"])[z]                      # (nat,)
    cn_ref = jnp.asarray(t["cn_ref"])[z]                  # (nat, 7)
    c6_ref = jnp.asarray(t["c6"])[z[:, None], z[None, :]]  # (nat, nat, 7, 7)

    nat = coords.shape[0]
    dvec = coords[:, None, :] - coords[None, :, :]
    r2 = jnp.sum(dvec * dvec, axis=-1) + jnp.eye(nat)     # guard the diagonal
    r = jnp.sqrt(r2)
    off = 1.0 - jnp.eye(nat)

    # Coordination numbers (k1-exponential counting function).
    rco = rcov[:, None] + rcov[None, :]
    cn = jnp.sum(off / (1.0 + jnp.exp(-_K1 * (rco / r - 1.0))), axis=1)

    # Gaussian-weight C6 interpolation over the reference systems. Unused
    # reference slots carry cn_ref < 0; the max-shift keeps the exponentials
    # from underflowing all at once for CN beyond the reference range.
    dcn = cn[:, None] - cn_ref                            # (nat, 7)
    d2 = jnp.where(cn_ref >= 0.0, -_K3 * dcn * dcn, -jnp.inf)
    d2max = jnp.max(d2, axis=1, keepdims=True)            # (nat, 1)
    w = jnp.exp(d2 - d2max)                               # (nat, 7), max = 1
    wij = w[:, None, :, None] * w[None, :, None, :]       # (nat, nat, 7, 7)
    norm = jnp.clip(jnp.sum(wij, axis=(2, 3)), 1e-30)
    mask = (cn_ref >= 0.0)[:, None, :, None] & (cn_ref >= 0.0)[None, :, None, :]
    c6 = jnp.sum(jnp.where(mask, c6_ref, 0.0) * wij, axis=(2, 3)) / norm
    return c6, r2, off


def d3bj_energy(
    coords,
    numbers,
    params: tuple[float, float, float, float],
):
    """Two-body D3(BJ) dispersion energy (Ha) for traced ``coords`` (Bohr).

    ``numbers`` are the atomic numbers (any integer array; used as static
    gather indices into the vendored tables), ``params = (s6, a1, s8, a2)``.
    """
    t = _tables()
    z = jnp.asarray(np.asarray(numbers, dtype=np.int64))
    rr = jnp.asarray(t["r4r2"])[z]                        # (nat,) sqrt-scaled
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
    """Axilrod-Teller-Muto three-body dispersion energy (Ha, repulsive).

    The D3 triple-dipole term with ``C9 = s9·sqrt(C6_AB·C6_AC·C6_BC)`` (the
    same CN-interpolated C6 as the two-body term), the Heron-form angular
    factor, and zero-damping on the geometric means of the pairwise
    van-der-Waals radii (``rs9 = 4/3``, ``alpha = 14``), matching
    tad-dftd3's ``dispersion_atm``. Independent of the BJ parameters.
    """
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
    """Resolve a dispersion spec against the functional and atomic numbers.

    Returns a zero-argument-of-P energy closure of the (traced) coordinates,
    or ``None``. Atomic numbers come from the nuclear charges, so the raw
    :class:`~dftax.ks.energy.System` path works too. ``nelec`` supplies the
    electron count for D4's total molecular charge (EEQ); ``None`` means
    neutral.
    """
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
