"""VV10 nonlocal correlation (Vydrov-Van Voorhis 2010) in pure JAX.

The nonlocal correlation energy is a double grid integral

    E_nlc = ∫ dr ρ(r) [ β + ½ ∫ dr' ρ(r') Φ(r, r') ],

with the VV10 kernel ``Φ = -3/2 · 1/(g g' (g + g'))``, ``g(r) = ω₀(r)·R² +
κ(r)``, ``ω₀ = sqrt(C·(|∇ρ|²/ρ²)² + (4π/3)ρ)``, ``κ = 1.5π·b·(ρ/9π)^{1/6}``
and ``β = (3/b²)^{3/4}/32`` chosen so a uniform density gives zero. The pair
quadrature is evaluated on the XC grid itself, streamed over outer-point
chunks so the O(ng²) pair tensor is never materialized; grid points below a
density threshold are excluded on both sides, matching PySCF's ``_vv10nlc``
(the validation oracle, matched to machine precision on identical inputs).

Fully differentiable in the density AND the grid coordinates, so SCF
potentials (via ∂E/∂P) and nuclear forces (via the moving Becke grid) come
from autodiff with no additional code.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

_THRESH = 1e-8    # density threshold on both grids (PySCF convention)


def vv10_energy(rho, gnorm2, coords, weights, b: float, c: float,
                chunk: int = 2048):
    """VV10 energy from grid densities.

    Args:
        rho: total density on the grid, shape (ng,).
        gnorm2: squared density-gradient norm ``|∇ρ|²``, shape (ng,).
        coords: grid coordinates (ng, 3).
        weights: quadrature weights (ng,).
        b, c: the VV10 damping and kernel parameters (e.g. 6.0, 0.01 for
            wB97X-V; 5.9, 0.0093 for the original VV10).
        chunk: outer-grid chunk size for the streamed pair sum.
    """
    pi = math.pi
    kvv = 1.5 * pi * b * (9.0 * pi) ** (-1.0 / 6.0)
    beta = (3.0 / (b * b)) ** 0.75 / 32.0

    mask = rho >= _THRESH
    rho_s = jnp.where(mask, rho, 1.0)                 # safe values off-mask
    w0 = jnp.sqrt(c * (gnorm2 / (rho_s * rho_s)) ** 2
                  + (4.0 * pi / 3.0) * rho_s)
    kap = kvv * rho_s ** (1.0 / 6.0)
    rw = jnp.where(mask, rho * weights, 0.0)          # inner-sum weights

    ng = rho.shape[0]
    pad = (-ng) % chunk
    coords_p = jnp.pad(coords, ((0, pad), (0, 0)))
    w0_p = jnp.pad(w0, (0, pad), constant_values=1.0)
    kap_p = jnp.pad(kap, (0, pad), constant_values=1.0)
    mask_p = jnp.pad(mask, (0, pad))

    def outer_chunk(args):
        co, w0o, ko, mo = args                        # (chunk, ...)
        r2 = jnp.sum((co[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
        g = r2 * w0o[:, None] + ko[:, None]           # (chunk, ng)
        gp = r2 * w0[None, :] + kap[None, :]
        f = -1.5 * jnp.sum(rw[None, :] / (g * gp * (g + gp)), axis=1)
        return jnp.where(mo, f, 0.0)

    F = jax.lax.map(
        outer_chunk,
        (coords_p.reshape(-1, chunk, 3), w0_p.reshape(-1, chunk),
         kap_p.reshape(-1, chunk), mask_p.reshape(-1, chunk)),
    ).reshape(-1)[:ng]

    eps = jnp.where(mask, beta + 0.5 * F, 0.0)
    return jnp.sum(jnp.where(mask, weights * rho, 0.0) * eps)
