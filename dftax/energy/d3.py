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
Apache-2.0; see ``scripts/gen_d3_data.py``); the three-body ATM term is
intentionally not implemented (matching common D3(BJ) defaults).

Usage (choices-as-values)::

    KS(mol, PBE(), dispersion=d3bj())          # parameters from the functional
    KS(mol, PBE(), dispersion=d3bj("b3lyp"))   # or explicit
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


def d3bj(method: str | None = None) -> D3BJSpec:
    """Two-body D3(BJ) dispersion correction.

    Args:
        method: damping-parameter set name (``"pbe"``, ``"pbe0"``,
            ``"b3lyp"``, ``"cam-b3lyp"``, ``"r2scan"``); ``None`` resolves it
            from the functional the KS is built with.
    """
    return D3BJSpec(method=method)


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
    rcov = jnp.asarray(t["rcov"])[z]                      # (nat,)
    rr = jnp.asarray(t["r4r2"])[z]                        # (nat,) sqrt-scaled
    cn_ref = jnp.asarray(t["cn_ref"])[z]                  # (nat, 7)
    c6_ref = jnp.asarray(t["c6"])[z[:, None], z[None, :]]  # (nat, nat, 7, 7)
    s6, a1, s8, a2 = params

    coords = jnp.asarray(coords).reshape(-1, 3)
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

    c8 = 3.0 * c6 * rr[:, None] * rr[None, :]
    r0 = a1 * jnp.sqrt(jnp.clip(c8 / jnp.clip(c6, 1e-30), 0.0)) + a2

    e6 = c6 / (r2**3 + r0**6)
    e8 = c8 / (r2**4 + r0**8)
    return -0.5 * jnp.sum(off * (s6 * e6 + s8 * e8))


def _resolve_dispersion(spec, xc, charges):
    """Resolve a dispersion spec against the functional and atomic numbers.

    Returns a zero-argument-of-P energy closure of the (traced) coordinates,
    or ``None``. Atomic numbers come from the nuclear charges, so the raw
    :class:`~dftax.ks.energy.System` path works too.
    """
    if spec is None:
        return None
    if not isinstance(spec, D3BJSpec):
        raise TypeError(f"dispersion must be a d3bj() spec, got {spec!r}")
    method = spec.method if spec.method is not None else xc.name.lower()
    params = _d3_params(method)
    numbers = np.rint(np.asarray(charges)).astype(np.int64)
    return lambda coords: d3bj_energy(coords, numbers, params)
