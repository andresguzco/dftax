"""Grid-based exchange-correlation energy integration."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from dftax.energy.xc import XCFunctional
from dftax.energy.potentials import xc_potential


def xc_energy(
    xc: XCFunctional,
    rho: Float[Array, "n"],              # type: ignore
    weights: Float[Array, "n"],          # type: ignore
    chunk_size: int | None = None,
    grad_rho: Float[Array, "n 3"] | None = None,  # type: ignore
    rho_thresh: float = 1e-10,
) -> Scalar:
    """Exchange-correlation energy ``∫ ε_xc(ρ, ∇ρ) ρ dr`` on a quadrature grid.

    ``rho``/``grad_rho`` are the density (and, for GGA, its gradient) sampled at
    the grid points; ``weights`` are the quadrature weights.

    Grid points with ``ρ < rho_thresh`` are masked out with a nan-safe
    double-``where``: the functional is evaluated on a clamped density and the
    contribution (and its gradient) is forced to zero. This keeps the GGA
    reduced-gradient terms, whose derivatives diverge as ρ→0, from producing
    NaNs on the far, vanishing-density tail of an unpruned grid.
    """
    mask = rho > rho_thresh
    safe_rho = jnp.where(mask, rho, 1.0)
    if xc.xc_type == "GGA" and grad_rho is not None:
        safe_grad = jnp.where(mask[:, None], grad_rho, 0.0)
        eps = xc_potential(xc, safe_rho, chunk_size, grad_rho=safe_grad)
    else:
        eps = xc_potential(xc, safe_rho, chunk_size, grad_rho=grad_rho)
    contrib = jnp.where(mask, weights * eps * rho, 0.0)
    return jnp.sum(contrib)
