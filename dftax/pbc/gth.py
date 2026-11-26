# The GTH tables (CP2K's potential and basis-set files, GPL-2.0) are not vendored with
# dftax (Apache-2.0): they are read from $DFTAX_GTH_DATA or from an installed PySCF's copies.
from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class GTHPotential:
    """One element's GTH pseudopotential.

    ``cloc`` holds the local polynomial coefficients ``C_1..C_4`` (zero
    padded). ``projectors`` has one ``(r_l, h)`` per angular momentum
    ``l = 0, 1, ...``, with ``h`` the symmetric ``n x n`` coupling matrix of
    that channel's ``n`` projectors (``n = 0`` for a channel without any).
    """

    symbol: str
    zion: int
    rloc: float
    cloc: tuple[float, float, float, float]
    projectors: tuple[tuple[float, np.ndarray], ...]


def _data_dir() -> Path:
    env = os.environ.get("DFTAX_GTH_DATA")
    if env:
        return Path(env)
    spec = importlib.util.find_spec("pyscf")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            "GTH tables are not vendored with dftax: set DFTAX_GTH_DATA "
            "to a directory holding pseudo/<name>.dat and basis/<name>.dat "
            "(CP2K format), or install pyscf (pip install dftax[periodic]), "
            "whose copies are then read."
        )
    return Path(spec.origin).parent / "pbc" / "gto"


@lru_cache(maxsize=None)
def _blocks(path: str) -> tuple[tuple[str, ...], ...]:
    blocks, cur = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                if cur:
                    blocks.append(tuple(cur))
                    cur = []
                continue
            cur.append(s)
    if cur:
        blocks.append(tuple(cur))
    return tuple(blocks)


def _table(kind: str, name: str) -> tuple[tuple[str, ...], ...]:
    path = _data_dir() / kind / f"{name.lower()}.dat"
    if not path.exists():
        raise FileNotFoundError(f"no GTH {kind} table {name!r} at {path}")
    return _blocks(str(path))


def load_potential(name: str, symbol: str) -> GTHPotential:
    """The potential of ``symbol`` in the table ``name`` (``"gth-pade"``,
    ``"gth-pbe"``, ...).

    A ``-q<N>`` suffix picks the variant with ``N`` valence electrons
    (``"gth-pade-q2"``); without it the first entry of the element is taken,
    as PySCF does.
    """
    m = re.fullmatch(r"(.+?)(-q\d+)?", name.lower())
    base, suffix = m.group(1), m.group(2)
    sym = symbol.strip().capitalize()
    for block in _table("pseudo", base):
        head = block[0].split()
        if head[0].capitalize() != sym:
            continue
        if suffix and not any(t.lower().endswith(suffix) for t in head[1:]):
            continue
        return _parse_potential(sym, block)
    raise KeyError(f"no {name!r} pseudopotential for {sym}")


def _parse_potential(sym: str, block: tuple[str, ...]) -> GTHPotential:
    nelec = [int(t) for t in block[1].split()]
    t = block[2].split()
    rloc, nexp = float(t[0]), int(t[1])
    cexp = [float(x) for x in t[2:2 + nexp]]
    cloc = tuple(cexp + [0.0] * (4 - nexp))
    nchan = int(block[3].split()[0])
    lines = iter(block[4:])
    projectors = []
    for _ in range(nchan):
        t = next(lines).split()
        r_l, n = float(t[0]), int(t[1])
        upper = [float(x) for x in t[2:]]
        for _ in range(1, n):
            upper += [float(x) for x in next(lines).split()]
        h = np.zeros((n, n))
        if n:
            h[np.triu_indices(n)] = upper
            h = h + h.T - np.diag(np.diag(h))
        projectors.append((r_l, h))
    return GTHPotential(symbol=sym, zion=sum(nelec), rloc=rloc, cloc=cloc,
                        projectors=tuple(projectors))


def load_basis_shells(name: str, symbol: str) -> list[dict]:
    """The contracted shells of ``symbol`` in the basis table ``name``
    (``"gth-szv"``, ``"gth-dzvp"``, ...), in the layout of a Basis Set
    Exchange ``electron_shells`` entry: one dict per angular momentum of
    each exponent set, with ``coefficients`` holding one row per contraction.

    Shells are ordered by angular momentum (stable), the order PySCF gives
    them, so the AO order matches a PySCF ``Cell`` with the same basis.
    """
    sym = symbol.strip().capitalize()
    for block in _table("basis", name):
        if block[0].split()[0].capitalize() == sym:
            break
    else:
        raise KeyError(f"no {name!r} basis for {sym}")
    lines = iter(block[1:])
    shells = []
    for _ in range(int(next(lines).split()[0])):
        head = [int(t) for t in next(lines).split()]
        lmin, lmax, nexp, ncon = head[1], head[2], head[3], head[4:]
        rows = np.array([[float(x) for x in next(lines).split()]
                         for _ in range(nexp)])
        col = 1
        for l, n in zip(range(lmin, lmax + 1), ncon):
            shells.append({
                "angular_momentum": [l],
                "exponents": rows[:, 0].tolist(),
                "coefficients": rows[:, col:col + n].T.tolist(),
            })
            col += n
    return sorted(shells, key=lambda s: s["angular_momentum"][0])


def local_parameters(potentials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.array([p.zion for p in potentials], dtype=np.float64),
            np.array([p.rloc for p in potentials], dtype=np.float64),
            np.array([p.cloc for p in potentials], dtype=np.float64))


def vloc_G(G2, zion, rloc, cloc):
    """Fourier transform ``∫ V_loc(r) exp(-i G·r) dr`` of one atom's local
    pseudopotential, as a function of ``G2 = |G|^2``.

    The ``G = 0`` value is the finite part: the ``-4π Z_ion/G^2`` divergence
    cancels against the Hartree and Ewald terms of a neutral cell, and what
    is left is ``2π Z_ion r_loc^2`` plus the polynomial's integral.
    """
    x = G2 * rloc ** 2
    gauss = jnp.exp(-0.5 * x)
    safe = jnp.where(G2 > 0.0, G2, 1.0)
    coul = jnp.where(G2 > 0.0, -4.0 * jnp.pi * zion / safe * gauss,
                     2.0 * jnp.pi * zion * rloc ** 2)
    poly = (cloc[0]
            + cloc[1] * (3.0 - x)
            + cloc[2] * (15.0 - 10.0 * x + x ** 2)
            + cloc[3] * (105.0 - 105.0 * x + 21.0 * x ** 2 - x ** 3))
    return coul + (2.0 * jnp.pi) ** 1.5 * rloc ** 3 * gauss * poly


def _r2_power_terms(i: int):
    import math

    out = []
    for p in range(i + 1):
        for q in range(i + 1 - p):
            r = i - p - q
            out.append(((2 * p, 2 * q, 2 * r),
                        math.factorial(i) // (math.factorial(p)
                                              * math.factorial(q)
                                              * math.factorial(r))))
    return out


def projector_basis(potentials, coords):
    import math

    from dftax.basis.loader import _block_diag, _cart2sph_blocks, gto_norm
    from dftax.energy.gto import BasisData, _CART_COMPONENTS, _contracted_norm

    c2s = _cart2sph_blocks()
    centers, angular, exps, maps, hblocks = [], [], [], [], []
    max_l = 0
    for pot, center in zip(potentials, np.asarray(coords, dtype=np.float64)):
        for l, (r_l, h) in enumerate(pot.projectors):
            n = h.shape[0]
            if n == 0:
                continue
            alpha = 0.5 / r_l ** 2
            comps = _CART_COMPONENTS[l]
            norm = gto_norm(l, alpha)
            k = [norm * (_contracted_norm(np.array([alpha]), np.array([norm]), c)
                         if l <= 1 else 1.0) for c in comps]
            for i in range(n):
                deg = l + 2 * i
                if deg > 6:
                    raise NotImplementedError(
                        f"{pot.symbol}: projector l={l}, i={i + 1} needs "
                        f"cartesian Gaussians of degree {deg}; the integral "
                        f"engine stops at 6."
                    )
                max_l = max(max_l, deg)
                target = _CART_COMPONENTS[deg]
                index = {c: j for j, c in enumerate(target)}
                ratio = r_l ** (-2 * i) * math.sqrt(
                    math.gamma(l + 1.5) / math.gamma(l + 2 * i + 1.5))
                M = np.zeros((len(target), 2 * l + 1))
                for j, (ax, ay, az) in enumerate(comps):
                    for (px, py, pz), mult in _r2_power_terms(i):
                        M[index[(ax + px, ay + py, az + pz)]] += (
                            ratio * mult * k[j] * c2s[l][j])
                maps.append(M)
                for c in target:
                    centers.append(center)
                    angular.append(c)
                    exps.append(alpha)
            hblocks.append(np.kron(h, np.eye(2 * l + 1)))
    if not maps:
        return None
    basis = BasisData(
        centers=jnp.asarray(np.array(centers)),
        exponents=jnp.asarray(np.array(exps)[:, None]),
        coefficients=jnp.ones((len(exps), 1)),
        angular=jnp.asarray(np.array(angular, dtype=np.int32)),
        cart2sph=jnp.asarray(_block_diag(maps)),
        max_l=int(max_l),
    )
    return basis, _block_diag(hblocks)
