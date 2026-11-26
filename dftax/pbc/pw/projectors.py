from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

_RADIAL = {
    (0, 0): (4.0 * math.sqrt(2.0), (1.0,)),
    (0, 1): (8.0 * math.sqrt(2.0 / 15.0), (3.0, -1.0)),
    (0, 2): (16.0 / 3.0 * math.sqrt(2.0 / 105.0), (15.0, -10.0, 1.0)),
    (1, 0): (8.0 * math.sqrt(1.0 / 3.0), (1.0,)),
    (1, 1): (16.0 * math.sqrt(1.0 / 105.0), (5.0, -1.0)),
    (1, 2): (32.0 / 3.0 * math.sqrt(1.0 / 1155.0), (35.0, -14.0, 1.0)),
    (2, 0): (8.0 * math.sqrt(2.0 / 15.0), (1.0,)),
    (2, 1): (16.0 / 3.0 * math.sqrt(2.0 / 105.0), (7.0, -1.0)),
    (2, 2): (32.0 / 3.0 * math.sqrt(2.0 / 15015.0), (63.0, -18.0, 1.0)),
    (3, 0): (16.0 * math.sqrt(1.0 / 105.0), (1.0,)),
    (3, 1): (32.0 / 3.0 * math.sqrt(1.0 / 1155.0), (9.0, -1.0)),
    (3, 2): (64.0 / 45.0 * math.sqrt(1.0 / 1001.0), (99.0, -22.0, 1.0)),
}


def radial(q2, l: int, i: int, r_l: float):
    pref, poly = _RADIAL[(l, i)]
    x2 = q2 * r_l * r_l
    p = sum(c * x2 ** n for n, c in enumerate(poly))
    return (pref * math.pi ** 1.25 * r_l ** (l + 1.5)) * p * jnp.exp(-0.5 * x2)


def solid_harmonics(q, l: int):
    x, y, z = q[..., 0], q[..., 1], q[..., 2]
    pi = math.pi
    if l == 0:
        return jnp.stack([jnp.full_like(x, 0.5 / math.sqrt(pi))])
    if l == 1:
        c = math.sqrt(3.0 / (4.0 * pi))
        return jnp.stack([c * x, c * y, c * z])
    if l == 2:
        c1 = math.sqrt(15.0 / (4.0 * pi))
        return jnp.stack([
            c1 * x * y, c1 * y * z, c1 * x * z,
            math.sqrt(5.0 / (16.0 * pi)) * (2 * z * z - x * x - y * y),
            math.sqrt(15.0 / (16.0 * pi)) * (x * x - y * y)])
    if l == 3:
        r2 = x * x + y * y + z * z
        return jnp.stack([
            math.sqrt(35.0 / (32.0 * pi)) * (3 * x * x - y * y) * y,
            math.sqrt(105.0 / (4.0 * pi)) * x * y * z,
            math.sqrt(21.0 / (32.0 * pi)) * y * (5 * z * z - r2),
            math.sqrt(7.0 / (16.0 * pi)) * z * (5 * z * z - 3 * r2),
            math.sqrt(21.0 / (32.0 * pi)) * x * (5 * z * z - r2),
            math.sqrt(105.0 / (16.0 * pi)) * (x * x - y * y) * z,
            math.sqrt(35.0 / (32.0 * pi)) * (x * x - 3 * y * y) * x])
    raise NotImplementedError(f"GTH projectors with l = {l}")


def channels(potentials):
    chans, blocks = [], []
    for atom, pot in enumerate(potentials):
        for l, (r_l, h) in enumerate(pot.projectors):
            h = np.asarray(h, dtype=np.float64)
            if h.shape[0] == 0:
                continue
            if h.shape[0] > 3:
                raise NotImplementedError("more than three projectors per l")
            chans.append((atom, l, float(r_l), int(h.shape[0])))
            blocks.append(np.kron(h, np.eye(2 * l + 1)))
    if not chans:
        return (), None
    n = sum(b.shape[0] for b in blocks)
    D = np.zeros((n, n))
    i = 0
    for blk in blocks:
        D[i:i + blk.shape[0], i:i + blk.shape[0]] = blk
        i += blk.shape[0]
    return tuple(chans), jnp.asarray(D)


def projectors(q, coords, chans, vol):
    q2 = jnp.sum(q * q, axis=-1)
    rows = []
    for atom, l, r_l, n in chans:
        phase = jnp.exp(1j * (q @ coords[atom]))
        S = solid_harmonics(q, l)
        for i in range(n):
            rows.append(radial(q2, l, i, r_l)[None, :] * S * phase[None, :])
    return jnp.concatenate(rows, axis=0) / jnp.sqrt(vol)
