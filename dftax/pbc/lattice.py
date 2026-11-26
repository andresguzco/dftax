from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np


def reciprocal_vectors(a):
    """Rows ``b_i`` with ``a_i · b_j = 2π δ_ij``."""
    return 2.0 * jnp.pi * jnp.linalg.inv(a).T


def volume(a):
    """Cell volume ``|det a|``."""
    return jnp.abs(jnp.linalg.det(a))


def fft_indices(mesh) -> np.ndarray:
    axes = [np.fft.fftfreq(int(n), 1.0 / int(n)) for n in mesh]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def gvectors(a, mesh):
    """Plane-wave vectors ``G`` of a mesh, ``(N, 3)``, in FFT order."""
    return jnp.asarray(fft_indices(mesh)) @ reciprocal_vectors(a)


def uniform_grid(a, mesh):
    """Points and quadrature weights of the uniform grid of a mesh:
    ``(N, 3)`` coordinates and ``(N,)`` weights ``volume / N``."""
    axes = [np.arange(int(n)) / int(n) for n in mesh]
    frac = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    n = frac.shape[0]
    return jnp.asarray(frac) @ a, jnp.full((n,), 1.0 / n) * volume(a)


def lattice_images(a, rcut: float) -> np.ndarray:
    """Every integer triple ``n`` with ``|n·a| <= rcut``, ``(nimg, 3)``.

    ``a`` must be concrete.
    """
    a = np.asarray(a, dtype=np.float64)
    b = 2.0 * np.pi * np.linalg.inv(a).T
    nmax = np.ceil(rcut * np.linalg.norm(b, axis=1) / (2.0 * np.pi)).astype(int)
    n = np.array(list(itertools.product(
        *[range(-m, m + 1) for m in nmax])), dtype=np.int64)
    keep = np.linalg.norm(n @ a, axis=1) <= rcut
    return n[keep]


def _fft_friendly(n: int) -> int:
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def mesh_for_cutoff(a, ke_cutoff: float) -> tuple[int, int, int]:
    """The smallest FFT-friendly mesh holding every plane wave with
    ``|G|^2 / 2 <= ke_cutoff`` (Hartree). ``a`` must be concrete.
    """
    a = np.asarray(a, dtype=np.float64)
    gmax = np.sqrt(2.0 * ke_cutoff)
    nmax = np.floor(gmax * np.linalg.norm(a, axis=1) / (2.0 * np.pi)).astype(int)
    return tuple(_fft_friendly(int(2 * m + 1)) for m in nmax)
