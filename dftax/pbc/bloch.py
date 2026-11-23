"""Gaussian basis functions under periodic boundary conditions (Gamma point).

The periodic image of an atomic orbital is its lattice sum,

    φ_μ^pbc(r) = sum_T φ_μ(r - T),

so a one-electron matrix between two of them is one lattice sum of the
molecular integrals, ``sum_T <μ | O | ν shifted by T>``, and their values on
a grid are the molecular values summed over image-shifted centers. Both are
written on the molecular primitives (:mod:`dftax.integrals.overlap`,
:mod:`dftax.energy.gto`) with the list of translations fixed eagerly from a
concrete geometry: the integer images are static, their Cartesian vectors
``n·a`` are not, so everything stays differentiable in the coordinates and
in the lattice.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from dftax.energy.gto import BasisData, eval_gto_and_grad
from dftax.integrals.overlap import (
    _contracted_integral, _kinetic_primitive_3d, _overlap_primitive_3d,
)
from dftax.pbc.lattice import lattice_images
from dftax.utils.vmap import vmap as chunked_vmap


def _min_exponent(basis: BasisData) -> float:
    """Smallest exponent carrying a nonzero coefficient (concrete basis)."""
    e = np.asarray(basis.exponents)
    c = np.asarray(basis.coefficients)
    return float(e[c != 0.0].min())


def _spread(centers) -> float:
    c = np.unique(np.asarray(centers), axis=0)
    return float(np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1).max())


def pair_images(basis: BasisData, a, precision: float = 1e-12,
                other: BasisData | None = None) -> np.ndarray:
    """Integer translations whose image of ``other`` (default: ``basis``
    itself) can overlap ``basis``, ``(nT, 3)``.

    Two primitives ``exp(-α r^2)`` and ``exp(-β r^2)`` a distance ``d`` apart
    overlap as ``exp(-αβ/(α+β) d^2)``; the list reaches the distance where
    the most diffuse pair falls below ``precision``, plus the largest
    separation between centers.
    """
    alpha = _min_exponent(basis)
    beta = alpha if other is None else _min_exponent(other)
    rcut = np.sqrt(-np.log(precision) * (alpha + beta) / (alpha * beta))
    centers = (basis.centers if other is None
               else np.concatenate([np.asarray(basis.centers),
                                    np.asarray(other.centers)]))
    return lattice_images(a, rcut + _spread(centers))


def value_images(basis: BasisData, a, precision: float = 1e-12) -> np.ndarray:
    """Integer translations whose image of a basis function can reach the
    home cell.

    Values fall as ``exp(-α r^2)``, so an image matters only when its center
    lies within the cutoff of the cell; the cell is bounded by the sphere
    through its corners, and a translation is kept when it brings at least
    one atom within the cutoff of that sphere.
    """
    a_np = np.asarray(a, dtype=np.float64)
    rcut = np.sqrt(-np.log(precision) / _min_exponent(basis))
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1)
                        for k in (0, 1)], float) @ a_np
    middle = corners.mean(axis=0)
    radius = float(np.linalg.norm(corners - middle, axis=1).max())
    centers = np.unique(np.asarray(basis.centers), axis=0)
    reach = float(np.linalg.norm(centers - middle, axis=1).max())
    cand = lattice_images(a_np, rcut + radius + reach)
    d = np.linalg.norm(centers[None, :, :] + (cand @ a_np)[:, None, :] - middle,
                       axis=-1).min(axis=1)
    return cand[d <= rcut + radius]


def _lattice_sum(basis_a, basis_b, T, primitive_fn, chunk):
    """``sum_T <a_i | O | b_j shifted by T>`` over cartesian AOs, ``(na, nb)``."""
    def element(i, j):
        def one(t):
            return _contracted_integral(
                basis_a.exponents[i], basis_a.coefficients[i],
                basis_a.centers[i], basis_a.angular[i],
                basis_b.exponents[j], basis_b.coefficients[j],
                basis_b.centers[j] + t, basis_b.angular[j], primitive_fn)
        return jnp.sum(jax.vmap(one)(T))

    jdx = jnp.arange(basis_b.centers.shape[0])

    def row(i):
        return jax.vmap(element, in_axes=(None, 0))(i, jdx)

    return chunked_vmap(row, chunk_size=chunk)(
        jnp.arange(basis_a.centers.shape[0]))


def _to_final(M, basis_a, basis_b):
    if basis_a.cart2sph is not None:
        M = basis_a.cart2sph.T @ M
    if basis_b.cart2sph is not None:
        M = M @ basis_b.cart2sph
    return M


@eqx.filter_jit
def overlap_kinetic(basis: BasisData, a, images, chunk: int = 4):
    """Lattice-summed overlap and kinetic matrices at the Gamma point.

    ``images`` is the integer list of :func:`pair_images`. ``chunk`` rows of
    the matrix are in flight at once (each holds ``nao x nT x nprim^2``
    primitive integrals).
    """
    T = jnp.asarray(images, dtype=a.dtype) @ a
    S = _lattice_sum(basis, basis, T, _overlap_primitive_3d, chunk)
    K = _lattice_sum(basis, basis, T, _kinetic_primitive_3d, chunk)
    return _to_final(S, basis, basis), _to_final(K, basis, basis)


@eqx.filter_jit
def cross_overlap(basis_a: BasisData, basis_b: BasisData, a, images,
                  chunk: int = 4):
    """Lattice-summed overlap ``<a_i | sum_T b_j(r - T)>`` between two bases
    (the orbital basis and the pseudopotential projectors)."""
    T = jnp.asarray(images, dtype=a.dtype) @ a
    return _to_final(
        _lattice_sum(basis_a, basis_b, T, _overlap_primitive_3d, chunk),
        basis_a, basis_b)


@eqx.filter_jit
def ao_on_grid(basis: BasisData, a, images, coords, chunk: int = 8192):
    """Periodic AO values and gradients on grid points: ``(ng, nao)`` and
    ``(ng, nao, 3)``.

    ``images`` is the integer list of :func:`value_images`. The image sum is
    a scan (one image's values in flight at a time) inside a chunked map
    over the points, over the cartesian functions: the spherical transform
    is applied once per point, after the sum, not once per image.
    """
    T = jnp.asarray(images, dtype=a.dtype) @ a
    return chunked_vmap(lambda r: ao_at_point(basis, T, r),
                        chunk_size=chunk)(coords)


def ao_at_point(basis: BasisData, shifts, r):
    """Periodic AO values and gradients at one point, ``(nao,)`` and
    ``(nao, 3)``, summed over the image translations ``shifts`` (Cartesian,
    ``(nT, 3)``)."""
    cart = BasisData(
        centers=basis.centers, exponents=basis.exponents,
        coefficients=basis.coefficients, angular=basis.angular,
        cart2sph=None, max_l=basis.max_l)
    ncart = basis.centers.shape[0]

    def add(acc, t):
        # the function centered at R + t, evaluated at r
        v, g = eval_gto_and_grad(cart, r - t)
        return (acc[0] + v, acc[1] + g), None

    (v, g), _ = jax.lax.scan(
        add, (jnp.zeros((ncart,), r.dtype), jnp.zeros((ncart, 3), r.dtype)),
        shifts)
    if basis.cart2sph is not None:
        return v @ basis.cart2sph, jnp.einsum("cx,cs->sx", g, basis.cart2sph)
    return v, g
