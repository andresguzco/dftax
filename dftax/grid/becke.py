"""Radial quadratures, Bragg-Slater radii, and the fuzzy-Voronoi partition.

Two radial schemes (Becke's and Treutler-Ahlrichs' M4) and the atomic radii
are coordinate-independent constants; the fuzzy-Voronoi partition is written
in JAX so it is differentiable w.r.t. nuclear coordinates (needed for analytic
forces). Two cell functions: Becke's three-fold smoothing, and
Stratmann-Scuseria-Frisch's polynomial, which is exactly 0 or 1 past
|ν| = 0.64, so a point's weight needs only the cells its owner's does not
decide."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from dftax.utils.vmap import vmap as _chunked_vmap

# Element budget for the per-point Becke partition transient (chunk, A, B):
# grid points are processed in chunks of ``budget // natom²`` so the peak is
# ~128 MiB in float64 regardless of molecule size (same budget pattern as
# ``eri3c._DF_BRA_BUDGET``). For small molecules the chunk covers the whole
# block, so nothing changes there.
_PARTITION_BUDGET = 2**24

PARTITIONS = ("becke", "stratmann")
# Stratmann, Scuseria and Frisch, CPL 257, 213 (1996), eq. 14: the cell
# boundary's half-width in ν.
SSF_A = 0.64
# Cells a chunk of points takes in the SSF pass besides its owners': the
# smallest size that holds the union of its points' candidates (0: every
# weight is 0 or 1), every cell past the last. Exact either way.
SSF_CELLS = (0, 32, 256)

# Bragg-Slater atomic radii (Slater, JCP 41, 3199 (1964)), the cell sizes of
# the Becke partition. Tabulated in Angstrom, indexed by atomic number Z (index
# 0 is a placeholder "ghost"), and converted with PySCF's Bohr radius so the
# radii, and with them the grids, agree with PySCF's to the last bit.
_BOHR_ANGSTROM = 0.52917721092
BRAGG_BOHR = (1.0 / _BOHR_ANGSTROM) * np.array(
    [
        1.999999,
        0.35, 1.40,
        1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,
        1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,
        2.20, 1.80,
        1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35,
        1.30, 1.25, 1.15, 1.15, 1.15, 1.90,
        2.35, 2.00,
        1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55,
        1.55, 1.45, 1.45, 1.40, 1.40, 2.10,
    ],
    dtype=np.float64,
)


def bragg_radius(Z: int) -> float:
    """Bragg-Slater radius (Bohr) for atomic number ``Z``."""
    if not 0 < Z < len(BRAGG_BOHR):
        raise ValueError(
            f"Becke grid radii are tabulated up to Xe (Z={len(BRAGG_BOHR) - 1}); "
            f"got Z={Z}."
        )
    return float(BRAGG_BOHR[Z])


# Treutler-Ahlrichs radial size parameter xi (Bohr), indexed by Z (index 0 is
# a placeholder "ghost"). H-Kr from Treutler & Ahlrichs, JCP 102, 346 (1995);
# Rb-Xe from Psi4. The same table as PySCF's ``radi._treutler_ahlrichs_xi``.
TREUTLER_XI = np.array(
    [
        1.0,
        0.8, 0.9,
        1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9,
        1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0,
        1.5, 1.4,
        1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1,
        1.1, 1.0, 0.9, 0.9, 0.9, 0.9,
        2.0, 1.7,
        1.5, 1.5, 1.35, 1.35, 1.25, 1.2, 1.25, 1.3, 1.5, 1.5,
        1.3, 1.2, 1.2, 1.15, 1.15, 1.15,
    ],
    dtype=np.float64,
)


def treutler_xi(Z: int) -> float:
    """Treutler-Ahlrichs radial size parameter (Bohr) for atomic number ``Z``."""
    if not 0 < Z < len(TREUTLER_XI):
        raise ValueError(
            f"Treutler xi is tabulated up to Xe (Z={len(TREUTLER_XI) - 1}); "
            f"got Z={Z}."
        )
    return float(TREUTLER_XI[Z])


def becke_radial(n: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Becke radial quadrature for ``∫₀^∞ f(r) r² dr ≈ Σ w_i f(r_i)``.

    Second-kind Gauss-Chebyshev nodes mapped by ``r = R (1+x)/(1-x)``.

    Args:
        n: number of radial points.
        scale: radial size parameter ``R`` (Bohr).
    """
    i = np.arange(1, n + 1)
    theta = i * np.pi / (n + 1)
    x = np.cos(theta)
    r = scale * (1.0 + x) / (1.0 - x)
    drdx = 2.0 * scale / (1.0 - x) ** 2
    w = (np.pi / (n + 1)) * np.sin(theta) * r**2 * drdx
    return r, w


def treutler_radial(n: int, xi: float) -> tuple[np.ndarray, np.ndarray]:
    """Treutler-Ahlrichs M4 radial quadrature for ``∫₀^∞ f(r) r² dr``.

    The same second-kind Gauss-Chebyshev nodes as :func:`becke_radial`, mapped
    by ``r = (xi/ln 2) (1+x)^0.6 ln(2/(1-x))`` (mapping M4 of JCP 102, 346
    (1995), with alpha = 0.6). It puts fewer points in the far tail than
    Becke's mapping, so a coarse grid spends more of them where the density is.

    Args:
        n: number of radial points.
        xi: radial size parameter (Bohr), :func:`treutler_xi` per element.
    """
    i = np.arange(1, n + 1)
    theta = i * np.pi / (n + 1)
    x = np.cos(theta)
    c = xi / np.log(2.0)
    log_term = np.log((1.0 - x) / 2.0)
    r = -c * (1.0 + x) ** 0.6 * log_term
    drdx = c * (1.0 + x) ** 0.6 * (-0.6 / (1.0 + x) * log_term + 1.0 / (1.0 - x))
    w = (np.pi / (n + 1)) * np.sin(theta) * r**2 * drdx
    return r, w


def size_adjustment(bragg, radii_adjust: str | None):
    """Heteronuclear cell-size parameters ``a_AB`` (antisymmetric, natom²) from
    the atoms' Bragg radii ``bragg`` (natom,).

    Becke 1988, App. A: ``a = u / (u² - 1)`` with ``u = (chi-1)/(chi+1)``.
    ``"becke"`` takes ``chi`` as the Bragg-radius ratio and bounds ``|a|`` by
    0.45; ``"treutler"`` takes the square root of that ratio (JCP 102, 346)
    and bounds ``|a|`` by 1/2, as PySCF does; ``None`` leaves the cells
    unadjusted. In jnp, so under ``jit`` the natom² matrix is computed from
    the natom radii rather than embedded as a constant.
    """
    bragg = jnp.asarray(bragg)
    n_atom = bragg.shape[0]
    if radii_adjust is None:
        return jnp.zeros((n_atom, n_atom), bragg.dtype)
    chi = bragg[:, None] / bragg[None, :]
    if radii_adjust == "becke":
        bound = 0.45
    elif radii_adjust == "treutler":
        chi = jnp.sqrt(chi)
        bound = 0.5
    else:
        raise ValueError(
            f"unknown radii_adjust {radii_adjust!r}; use 'becke', 'treutler' or None."
        )
    u = (chi - 1.0) / (chi + 1.0)
    return jnp.clip(u / (u * u - 1.0), -bound, bound)



def _check_partition(partition):
    if partition not in PARTITIONS:
        raise ValueError(
            f"unknown partition {partition!r}; use 'becke' or 'stratmann'."
        )


def stratmann(nu):
    """SSF's ``g(ν)`` (CPL 257, 213 (1996), eq. 14): with ``x = ν/a``,
    ``(35x - 35x³ + 21x⁵ - 5x⁷)/16`` inside ``|ν| < a = 0.64`` and the sign
    of ``ν`` outside, exactly, so a cell's factor ``(1 - g)/2`` is exactly 0
    or 1 there with a zero derivative. Evaluated as PySCF's
    ``gen_grid.stratmann``."""
    x = nu / SSF_A
    x2 = x * x
    g = 0.0625 * (x * (35.0 + x2 * (-35.0 + x2 * (21.0 - 5.0 * x2))))
    return jnp.where(nu <= -SSF_A, -1.0, jnp.where(nu >= SSF_A, 1.0, g))


def _cell_fn(coords, Zs, radii_adjust, partition="becke"):
    """``p -> P_A(p)`` (natom,), the unnormalized cell functions at one
    point, with the pair quantities (R_AB, a_AB) formed once for all points."""
    _check_partition(partition)
    coords = jnp.asarray(coords)
    n_atom = coords.shape[0]
    # Pair quantities, computed once (natom²).
    dAB = coords[:, None, :] - coords[None, :, :]
    # +eye makes the diagonal sqrt(1) with a finite (zero) gradient; the
    # diagonal entries are masked out of the product below.
    RAB = jnp.sqrt(jnp.sum(dAB * dAB, axis=2) + jnp.eye(n_atom))
    a = size_adjustment(np.asarray([bragg_radius(Z) for Z in Zs]), radii_adjust)
    eye = jnp.eye(n_atom, dtype=bool)

    def cell(p):
        # Distances via squared-norm + sqrt; never call norm() on a possibly-
        # zero vector (its gradient is 0/0 = NaN, which leaks even through
        # masking).
        dpA = p[None, :] - coords
        rA = jnp.sqrt(jnp.sum(dpA * dpA, axis=1) + 1e-30)     # (A,)
        mu = (rA[:, None] - rA[None, :]) / RAB                # (A, B)
        nu = mu + a * (1.0 - mu * mu)
        if partition == "stratmann":
            f = stratmann(nu)
        else:
            f = nu
            for _ in range(3):
                f = 1.5 * f - 0.5 * f**3
        s = 0.5 * (1.0 - f)
        # Diagonal (A == B) plays no role: s_AA = 1 leaves it out of the product.
        s = jnp.where(eye, 1.0, s)
        return jnp.prod(s, axis=1)                            # P_A = Π_{B≠A} s(ν_AB)

    return cell


def becke_partition(points, coords, Zs, radii_adjust: str | None = "becke",
                    partition: str = "becke"):
    """Unnormalized cell functions ``P_A(point)``, shape (n_grid, n_atom).

    Vectorized JAX (differentiable w.r.t. ``coords``); includes the heteronuclear
    size adjustment selected by ``radii_adjust`` (see :func:`size_adjustment`)
    and the cell function selected by ``partition`` (``"becke"`` or
    ``"stratmann"``). Points are processed in budget-derived chunks so the
    per-point ``(chunk, A, B)`` transient stays bounded for large molecules
    (checkpointed, so the backward pass used by the forces rematerializes each
    chunk instead of storing them all).
    """
    n_atom = len(Zs)
    chunk = max(1, _PARTITION_BUDGET // (n_atom * n_atom))
    cell = _cell_fn(coords, Zs, radii_adjust, partition)
    return _chunked_vmap(cell, chunk_size=chunk, checkpoint=True)(points)


def becke_weights(points, owners, coords, Zs, radii_adjust: str | None = "becke",
                  partition: str = "becke"):
    """Weight factors ``P_owner(p) / Σ_A P_A(p)`` of points ``p`` owned
    by atoms ``owners`` (int, (n_grid,)), shape (n_grid,).

    One chunked pass over every atom's points with the pair quantities formed
    once, so a molecule's grid is one loop in the compiled program (a pass per
    atom was natom loops, each with its own natom² constants) and the
    (n_grid, n_atom) cell functions are never held. Differentiable w.r.t.
    ``coords``, checkpointed like :func:`becke_partition`. With
    ``partition="stratmann"`` the normalization takes only the cells that can
    be nonzero (:func:`_ssf_weights`).
    """
    _check_partition(partition)
    if partition == "stratmann":
        return _ssf_weights(points, owners, coords, Zs, radii_adjust)
    cell = _cell_fn(coords, Zs, radii_adjust)
    n_atom = len(Zs)

    def weight(p, owner):
        P = cell(p)
        return P[owner] / P.sum()

    chunk = max(1, _PARTITION_BUDGET // (n_atom * n_atom))
    return _chunked_vmap(weight, in_axes=(0, 0), chunk_size=chunk, checkpoint=True)(points, owners)


def _ssf_weights(points, owners, coords, Zs, radii_adjust, cells=SSF_CELLS,
                 budget=_PARTITION_BUDGET):
    """SSF weight factors, each point's from its owner's cell and the cells
    that can be nonzero there.

    Where ``P_O(p) > 0`` (O the owner) another cell ``P_A`` holds the factor
    ``s(ν_AO) = 1 - s(ν_OA)``, so it is nonzero only if ``|ν_OA| < a``: A is
    one of the point's candidates. Where ``P_O(p) = 0`` the weight is 0
    whatever the others. A chunk of points evaluates the union of its points'
    candidates, padded to the smallest of ``cells`` that holds it (every cell
    past the last), as one dense block, and each point sums its own. Exact:
    the cells left out are zero, and so are their derivatives. Checkpointed
    per chunk and per block of points.
    """
    coords = jnp.asarray(coords)
    points = jnp.asarray(points)
    owners = jnp.asarray(owners)
    n_atom, n = coords.shape[0], points.shape[0]
    if n_atom == 1:
        return jnp.ones(n, points.dtype)
    dAB = coords[:, None, :] - coords[None, :, :]
    RAB = jnp.sqrt(jnp.sum(dAB * dAB, axis=2) + jnp.eye(n_atom))
    a = size_adjustment(np.asarray([bragg_radius(Z) for Z in Zs]), radii_adjust)
    cols = jnp.arange(n_atom)
    sizes = sorted({min(int(c), n_atom) for c in cells} | {n_atom})
    P = max(1, budget // (next(m for m in sizes if m > 0) * n_atom))
    P = n if P >= n else 1 << (P.bit_length() - 1)            # a power of two below n

    def cell_sum(r, cand, idx):                               # Σ_A P_A over candidates A in idx
        R, aa, rA, c = ((RAB, a, r, cand) if idx is None
                        else (RAB[idx], a[idx], r[:, idx], cand[:, idx]))
        diag = (cols if idx is None else idx)[:, None] == cols
        mu = (rA[:, :, None] - r[:, None, :]) / R
        s = 0.5 * (1.0 - stratmann(mu + aa * (1.0 - mu * mu)))
        return jnp.sum(jnp.where(c, jnp.prod(jnp.where(diag, 1.0, s), axis=-1), 0.0), axis=1)

    def tier(m):
        cap = max(1, budget // (m * n_atom)) if m else P
        b = next(d for d in range(min(cap, P), 0, -1) if P % d == 0)

        def f(r, cand, u):
            if m == 0:
                return jnp.zeros(r.shape[0], r.dtype)
            idx = None if m == n_atom else lax.top_k(u.astype(r.dtype), m)[1]
            if b == P:
                return cell_sum(r, cand, idx)
            blocks = (r.reshape(-1, b, n_atom), cand.reshape(-1, b, n_atom))
            return lax.map(jax.checkpoint(lambda x: cell_sum(*x, idx)), blocks).reshape(-1)
        return f

    def chunk(x):
        p, own = x
        d = p[:, None, :] - coords[None]
        r = jnp.sqrt(jnp.sum(d * d, axis=2) + 1e-30)          # (P, natom)
        mu = (jnp.take_along_axis(r, own[:, None], 1) - r) / RAB[own]
        nu = mu + a[own] * (1.0 - mu * mu)
        off = own[:, None] != cols
        P_own = jnp.prod(jnp.where(off, 0.5 * (1.0 - stratmann(nu)), 1.0), axis=1)
        cand = off & (jnp.abs(nu) < SSF_A) & (P_own > 0.0)[:, None]
        u = jnp.any(cand, axis=0)
        k = sum((jnp.sum(u) > m).astype(jnp.int32) for m in sizes[:-1])
        Z = P_own + lax.switch(k, [tier(m) for m in sizes], r, cand, u)
        return P_own / jnp.where(Z > 0.0, Z, 1.0)             # Z = 0 only where P_own = 0

    pad = (-n) % P
    pts = jnp.concatenate([points, jnp.broadcast_to(points[:1], (pad, 3))]).reshape(-1, P, 3)
    own = jnp.concatenate([owners, jnp.broadcast_to(owners[:1], (pad,))]).reshape(-1, P)
    return lax.map(jax.checkpoint(chunk), (pts, own)).reshape(-1)[:n]
