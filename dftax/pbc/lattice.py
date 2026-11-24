"""Lattice geometry: reciprocal vectors, plane-wave vectors, uniform grids,
image translations.

Lattice vectors are the rows of ``a`` (Bohr). Whatever is an integer pattern
(the FFT index grid, the list of image translations) is built eagerly with
numpy from concrete inputs; what is geometry (``G = n·b``, ``r = f·a``,
``T = n·a``) is jax and stays a differentiable function of ``a``, which is
what the stress needs.

Conventions are PySCF's, so arrays compare element for element: ``b = 2π
(a⁻¹)ᵀ``, grid points ``(i/n1) a1 + (j/n2) a2 + (k/n3) a3`` in C order, G
vectors in FFT order (``numpy.fft.fftfreq``) in C order.
"""

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
    """Integer plane-wave indices of a mesh in FFT order, ``(N, 3)``."""
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

    ``a`` must be concrete. The bound per axis is the number of lattice
    planes within ``rcut``: planes normal to ``b_i`` are ``2π/|b_i|`` apart.
    """
    a = np.asarray(a, dtype=np.float64)
    b = 2.0 * np.pi * np.linalg.inv(a).T
    nmax = np.ceil(rcut * np.linalg.norm(b, axis=1) / (2.0 * np.pi)).astype(int)
    n = np.array(list(itertools.product(
        *[range(-m, m + 1) for m in nmax])), dtype=np.int64)
    keep = np.linalg.norm(n @ a, axis=1) <= rcut
    return n[keep]


def _fft_friendly(n: int) -> int:
    """Smallest integer ``>= n`` whose only prime factors are 2, 3, 5, 7."""
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

    A wave of index ``n_i`` along axis ``i`` has ``n_i = G·a_i / 2π``, at
    most ``|G| |a_i| / 2π``; the mesh needs both signs and zero.
    """
    a = np.asarray(a, dtype=np.float64)
    gmax = np.sqrt(2.0 * ke_cutoff)
    nmax = np.floor(gmax * np.linalg.norm(a, axis=1) / (2.0 * np.pi)).astype(int)
    return tuple(_fft_friendly(int(2 * m + 1)) for m in nmax)
