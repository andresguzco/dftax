"""The plane-wave basis of a periodic cell.

A Bloch orbital at wave vector ``k`` is expanded as

    ψ_nk(r) = Ω^(-1/2) sum_G c_nkG exp(i (k + G)·r),   |k + G|^2 / 2 <= ecut,

so its coefficients live on the integer triples ``m`` of ``G = m·b`` inside
a sphere (one per ``k``). The spheres of all ``k`` are padded to one size
with a mask, so every ``k`` runs the same program. The density, a sum of
``|ψ|^2``, holds waves up to ``4 ecut`` and lives on the uniform mesh that
holds them (no aliasing), where the Hartree, local-pseudopotential and XC
energies are the ones the Gaussian engine already takes on its grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from dftax.pbc.lattice import mesh_for_cutoff


@dataclass(frozen=True)
class PlaneWaves:
    """The plane-wave basis of a cell (see :func:`planewaves`)."""

    ecut: float
    mesh: tuple[int, int, int] | None = None


def planewaves(ecut: float, *,
               mesh: tuple[int, int, int] | None = None) -> PlaneWaves:
    """Plane waves up to the kinetic energy ``ecut`` (Hartree).

    Args:
        ecut: the wavefunction cutoff, ``|k + G|^2 / 2 <= ecut``.
        mesh: the density mesh; default, the smallest FFT-friendly one
            holding ``4 ecut`` (the density of two waves of ``ecut``).

    Example:
        ```python
        res = scf_pw(cell, LDA(), planewaves(40.0))
        ```
    """
    if ecut <= 0:
        raise ValueError(f"ecut must be positive, got {ecut}")
    if mesh is not None:
        mesh = tuple(int(n) for n in mesh)
    return PlaneWaves(ecut=float(ecut), mesh=mesh)


def monkhorst_pack(kmesh, shift=(0, 0, 0), time_reversal: bool = True):
    """A Monkhorst-Pack mesh in fractional coordinates of the reciprocal
    lattice, ``(nk, 3)``, with its weights ``(nk,)`` (summing to 1).

    ``shift = 0`` (default) centres the mesh on Gamma, which an even mesh
    otherwise misses: a Gamma-centred ``n1 x n2 x n3`` mesh is exactly the
    Gamma point of the ``n1 x n2 x n3`` supercell. With ``time_reversal``
    (default) ``k`` and ``-k`` are one point of twice the weight: their
    densities and band energies are equal for a real (spin-free) potential.
    No other symmetry is used.
    """
    n = np.array([int(x) for x in kmesh])
    s = np.array([float(x) for x in shift])
    axes = [(np.arange(n[d]) + 0.5 * s[d]) / n[d] for d in range(3)]
    k = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    w = np.full(k.shape[0], 1.0 / k.shape[0])
    if time_reversal:
        # integer labels 2 n k (mod 2 n) find each point's partner -k
        lab = np.round(2 * n * k).astype(int) % (2 * n)
        where = {tuple(x): i for i, x in enumerate(lab)}
        keep, wk, seen = [], [], set()
        for i, x in enumerate(lab):
            if i in seen:
                continue
            j = where[tuple((-x) % (2 * n))]
            seen.update((i, j))
            keep.append(i)
            wk.append(w[i] * (1.0 if j == i else 2.0))
        k, w = k[keep], np.asarray(wk)
    k = k - np.round(k)                       # into (-1/2, 1/2]
    return k, w


@dataclass(frozen=True, eq=False)
class PWPlan:
    """The plane-wave spheres of every ``k``, fixed by the concrete lattice.

    ``miller`` are the integer triples ``m`` (``G = m·b``), padded to one
    length per ``k`` (``mask`` 0 on the padding); ``fft_idx`` is the flat
    index of each wave on the density mesh (C order, negative indices
    wrapped), the box a wavefunction is put into before its FFT.
    """

    ecut: float
    mesh: tuple[int, int, int]
    kfrac: Float[Array, "nk 3"]
    weights: Float[Array, "nk"]
    miller: Float[Array, "nk npw 3"]          # integers, as floats
    mask: Float[Array, "nk npw"]
    fft_idx: Int[Array, "nk npw"]
    npw: tuple[int, ...]                       # true sphere sizes

    @property
    def nk(self) -> int:
        return int(self.kfrac.shape[0])


def plan_planewaves(a, pw: PlaneWaves, kfrac=None, weights=None) -> PWPlan:
    """The spheres of the cell's lattice ``a`` (concrete) at the k-points
    ``kfrac`` (fractional, default Gamma)."""
    a = np.asarray(a, dtype=np.float64)
    b = 2.0 * np.pi * np.linalg.inv(a).T
    if kfrac is None:
        kfrac, weights = np.zeros((1, 3)), np.ones(1)
    kfrac = np.asarray(kfrac, dtype=np.float64).reshape(-1, 3)
    weights = (np.full(kfrac.shape[0], 1.0 / kfrac.shape[0]) if weights is None
               else np.asarray(weights, dtype=np.float64))
    mesh = pw.mesh if pw.mesh is not None else mesh_for_cutoff(a, 4.0 * pw.ecut)
    n = np.array(mesh)

    # every triple whose wave can fall inside the sphere of some k (|k| is
    # at most half a reciprocal vector per axis, hence the +1)
    gmax = np.sqrt(2.0 * pw.ecut)
    nmax = np.floor(gmax * np.linalg.norm(a, axis=1) / (2.0 * np.pi)).astype(int) + 1
    m = np.stack(np.meshgrid(*[np.arange(-x, x + 1) for x in nmax],
                             indexing="ij"), axis=-1).reshape(-1, 3)
    spheres = []
    for k in kfrac:
        q = (k + m) @ b
        spheres.append(m[0.5 * np.sum(q * q, axis=1) <= pw.ecut])
    npw = [len(s) for s in spheres]
    width = max(npw)
    # the mesh must at least hold every wave (the default, 4 ecut, also
    # holds every product of two, so the density is not aliased)
    reach = np.max([np.abs(s).max(0) for s in spheres], axis=0)
    if np.any(2 * reach + 1 > n):
        raise ValueError(f"mesh {mesh} is too small for ecut {pw.ecut}")

    miller = np.zeros((len(spheres), width, 3))
    mask = np.zeros((len(spheres), width))
    fft_idx = np.zeros((len(spheres), width), dtype=np.int64)
    for i, s in enumerate(spheres):
        miller[i, :len(s)] = s
        mask[i, :len(s)] = 1.0
        w = np.mod(s, n)
        fft_idx[i, :len(s)] = (w[:, 0] * n[1] + w[:, 1]) * n[2] + w[:, 2]
    return PWPlan(
        ecut=float(pw.ecut), mesh=tuple(int(x) for x in mesh),
        kfrac=jnp.asarray(kfrac), weights=jnp.asarray(weights),
        miller=jnp.asarray(miller), mask=jnp.asarray(mask),
        fft_idx=jnp.asarray(fft_idx, dtype=jnp.int32), npw=tuple(npw))
