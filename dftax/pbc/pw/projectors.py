"""GTH nonlocal projectors in reciprocal space (the plane-wave engine).

The projector ``p_i^l(r) Y_lm`` of an atom at ``R`` overlaps a normalized
plane wave of wave vector ``q = k + G`` as

    <q | p_ilm> = Ω^(-1/2) (-i)^l exp(-i q·R) p_i^l(q) Y_lm(q̂),

with the radial transform ``p_i^l(q) = 4π ∫ r^2 j_l(qr) p_i^l(r) dr`` in
closed form (Goedecker, Teter and Hutter 1996; Hartwigsen, Goedecker and
Hutter 1998; the table of PySCF's ``pseudo.pp.projG_li``). Within one ``l``
the factor ``(-i)^l`` cancels between bra and ket, so it is dropped, and
``q^l Y_lm(q̂)`` is written as a real solid harmonic, a polynomial in ``q``
that stays smooth at ``q = 0``. Any orthonormal real ``Y_lm`` serves: the
energy only involves ``sum_m``, which is invariant.

    E_nl = sum_k w_k sum_n f_nk  β_nk^H D β_nk,    β_nk = P_k c_nk,

with ``P_k`` the projectors on the sphere of ``k`` and ``D = h^l ⊗ 1`` over
the ``m`` of each channel (rows ordered atom, l, i, m).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

# (l, i) -> (prefactor, coefficients of the polynomial in x^2, x = q r_l)
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
    """``p_i^l(q) / q^l`` of a GTH projector (``i`` from 0), from ``|q|^2``."""
    pref, poly = _RADIAL[(l, i)]
    x2 = q2 * r_l * r_l
    p = sum(c * x2 ** n for n, c in enumerate(poly))
    return (pref * math.pi ** 1.25 * r_l ** (l + 1.5)) * p * jnp.exp(-0.5 * x2)


def solid_harmonics(q, l: int):
    """Real solid harmonics ``|q|^l Y_lm(q̂)``, orthonormal on the sphere,
    ``(2l + 1, ...)`` for ``q`` of shape ``(..., 3)``."""
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
    """The projector channels of a cell: a static tuple of
    ``(atom, l, r_l, n_projectors)`` and the coupling matrix ``D``
    (rows ordered atom, l, i, m), or ``((), None)`` without any."""
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
    """``P[p, G] = Ω^(-1/2) exp(i q·R) p_i^l(q) Y_lm(q̂)`` (up to ``i^l``),
    the projectors of all channels on the waves ``q`` ``(npw, 3)``, so that
    ``β = P c``. Rows ordered atom, l, i, m."""
    q2 = jnp.sum(q * q, axis=-1)
    rows = []
    for atom, l, r_l, n in chans:
        phase = jnp.exp(1j * (q @ coords[atom]))
        S = solid_harmonics(q, l)                              # (2l+1, npw)
        for i in range(n):
            rows.append(radial(q2, l, i, r_l)[None, :] * S * phase[None, :])
    return jnp.concatenate(rows, axis=0) / jnp.sqrt(vol)
