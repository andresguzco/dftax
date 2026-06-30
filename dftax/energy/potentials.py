"""Pointwise real-space potential primitives for grid-based energy terms."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from dftax.energy.xc import XCFunctional
from dftax.utils import vmap


def coulomb_potential(
    r: Float[Array, "3"],                        # type: ignore
    source_coords: Float[Array, "n_sources 3"],  # type: ignore
    source_charges: Float[Array, "n_sources"],   # type: ignore
) -> Scalar:
    """Electrostatic potential at ``r`` due to a set of point charges."""
    d2 = jnp.sum((source_coords - r) ** 2, axis=-1)
    return source_charges @ jax.lax.rsqrt(d2.clip(min=1e-10))


def xc_potential(
    xc: XCFunctional,
    rho: Float[Array, "n"],              # type: ignore
    chunk_size: int | None = None,
    grad_rho: Float[Array, "n 3"] | None = None,  # type: ignore
) -> Float[Array, "n"]:                  # type: ignore
    """Per-electron XC energy density ``ε_xc`` evaluated pointwise on a grid."""
    if xc.xc_type == "LDA":
        return vmap(xc)(rho)
    elif xc.xc_type == "GGA":
        if grad_rho is None:
            raise ValueError("GGA requires grad_rho")
        return vmap(xc, in_axes=(0, 0), chunk_size=chunk_size)(rho, grad_rho)
    else:
        raise NotImplementedError(f"XC type {xc.xc_type} not implemented.")
