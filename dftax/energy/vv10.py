from __future__ import annotations

import math

import jax
import jax.numpy as jnp

_THRESH = 1e-8


def vv10_energy(rho, gnorm2, coords, weights, b: float, c: float,
                chunk: int = 2048):
    pi = math.pi
    kvv = 1.5 * pi * b * (9.0 * pi) ** (-1.0 / 6.0)
    beta = (3.0 / (b * b)) ** 0.75 / 32.0

    mask = rho >= _THRESH
    rho_s = jnp.where(mask, rho, 1.0)
    w0 = jnp.sqrt(c * (gnorm2 / (rho_s * rho_s)) ** 2
                  + (4.0 * pi / 3.0) * rho_s)
    kap = kvv * rho_s ** (1.0 / 6.0)
    rw = jnp.where(mask, rho * weights, 0.0)

    ng = rho.shape[0]
    pad = (-ng) % chunk
    coords_p = jnp.pad(coords, ((0, pad), (0, 0)))
    w0_p = jnp.pad(w0, (0, pad), constant_values=1.0)
    kap_p = jnp.pad(kap, (0, pad), constant_values=1.0)
    mask_p = jnp.pad(mask, (0, pad))

    def outer_chunk(args):
        co, w0o, ko, mo = args
        r2 = jnp.sum((co[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
        g = r2 * w0o[:, None] + ko[:, None]
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
