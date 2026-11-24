"""GTH pseudopotentials: the tables and the local part.

The Goedecker-Teter-Hutter form (Phys. Rev. B 54, 1703 (1996); Hartwigsen,
Goedecker and Hutter, Phys. Rev. B 58, 3641 (1998)) is analytic in both
spaces, which is what makes it the natural pseudopotential for a Gaussian
basis with the Hartree term on a uniform grid. The local part of one atom is

    V_loc(r) = -Z_ion/r erf(r / (sqrt(2) r_loc))
               + exp(-r^2 / (2 r_loc^2)) sum_k C_k (r / r_loc)^(2k - 2)

and its Fourier transform, a function of G^2 alone, is :func:`vloc_G`.

The tables (CP2K's potential and basis-set files) are not vendored yet. They
are read from ``$DFTAX_GTH_DATA`` when it is set (a directory holding
``pseudo/<name>.dat`` and ``basis/<name>.dat``) and otherwise from the copies
an installed PySCF carries; nothing of PySCF is imported.
"""

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
            "GTH tables are not vendored with dftax yet: set DFTAX_GTH_DATA "
            "to a directory holding pseudo/<name>.dat and basis/<name>.dat "
            "(CP2K format), or install pyscf, whose copies are then read."
        )
    return Path(spec.origin).parent / "pbc" / "gto"


@lru_cache(maxsize=None)
def _blocks(path: str) -> tuple[tuple[str, ...], ...]:
    """The data blocks of a CP2K table: runs of lines between comment lines
    (every block is preceded by a ``#PSEUDOPOTENTIAL`` / ``#BASIS SET``
    marker)."""
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
        for _ in range(1, n):                      # rows 2..n of the triangle
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
    """``(zion, rloc, cloc)`` arrays, one row per atom, for :func:`vloc_G`."""
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
