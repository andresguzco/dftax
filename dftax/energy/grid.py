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
    tau: Float[Array, "n"] | None = None,         # type: ignore
    rho_thresh: float = 1e-10,
) -> Scalar:
    mask = rho > rho_thresh
    safe_rho = jnp.where(mask, rho, 1.0)
    if xc.xc_type == "MGGA":
        safe_grad = jnp.where(mask[:, None], grad_rho, 0.0)
        safe_tau = jnp.where(mask, tau, 1.0)
        eps = xc_potential(xc, safe_rho, chunk_size, grad_rho=safe_grad,
                           tau=safe_tau)
    elif xc.xc_type == "GGA" and grad_rho is not None:
        safe_grad = jnp.where(mask[:, None], grad_rho, 0.0)
        eps = xc_potential(xc, safe_rho, chunk_size, grad_rho=safe_grad)
    else:
        eps = xc_potential(xc, safe_rho, chunk_size, grad_rho=grad_rho)
    contrib = jnp.where(mask, weights * eps * rho, 0.0)
    return jnp.sum(contrib)
