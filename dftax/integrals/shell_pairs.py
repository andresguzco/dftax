"""Shell-pair data structure for vectorized integral computation.

Groups AOs by shell. AOs within the same shell share centers, exponents,
and contraction coefficients, differing only in angular momentum components.
This allows batching integrals by shell pair instead of AO pair.
"""

import numpy as np

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array

from dftax.energy.gto import BasisData, _CART_COMPONENTS


# Precomputed Cartesian components as JAX arrays, keyed by total angular momentum
CART_COMPONENTS_ARRAYS = {
    l: jnp.array(_CART_COMPONENTS[l], dtype=jnp.int32)  # (n_comp, 3)
    for l in range(7)  # l = 0..6 (h/i appear only in auxiliary DF bases)
}

# Number of Cartesian components per angular momentum
N_CART = {l: len(_CART_COMPONENTS[l]) for l in range(7)}


class ShellData(eqx.Module):
    """Precomputed shell-level basis data for vectorized integrals.

    AOs are grouped by shell: each shell has a single center, set of
    exponents/coefficients, and total angular momentum l. The individual
    AO components (lx, ly, lz) with lx+ly+lz=l are enumerated separately.

    Attributes:
        centers:      Shell centres, shape (n_shells, 3).
        exponents:    Primitive exponents (zero-padded), shape (n_shells, max_prim).
        coefficients: Contraction coefficients (zero-padded), shape (n_shells, max_prim).
        l_values:     Total angular momentum per shell, shape (n_shells,).
        shell_offsets: Starting AO index (in Cartesian basis) per shell, shape (n_shells,).
        n_shells:     Total number of shells (static).
        nao_cart:     Total number of Cartesian AOs (static).
        cart2sph:     Cartesian→spherical transformation, or None.
    """
    centers: Float[Array, "n_shells 3"]
    exponents: Float[Array, "n_shells max_prim"]
    coefficients: Float[Array, "n_shells max_prim"]
    l_values: Int[Array, "n_shells"]
    shell_offsets: Int[Array, "n_shells"]
    n_shells: int = eqx.field(static=True)
    nao_cart: int = eqx.field(static=True)
    cart2sph: Float[Array, "nao_cart nao_sph"] | None


def extract_shell_data(basis: BasisData) -> ShellData:
    """Extract shell-level data from a per-AO BasisData.

    Groups consecutive AOs that share the same center and exponents into
    shells. Within each shell, AOs differ only in their angular momentum
    components (lx, ly, lz) with the same total l = lx + ly + lz.

    NOTE: This function uses np.array() on basis arrays and cannot be called
    inside jax.grad/jit tracing. Use the _prepare_* helpers which precompute
    static structure separately from dynamic centers.

    Args:
        basis: Per-AO BasisData from extract_basis_data(mol).

    Returns:
        ShellData with deduplicated shell-level arrays.
    """
    # Work with numpy for the grouping logic (CPU setup, called once)
    centers_np = np.array(basis.centers)
    exponents_np = np.array(basis.exponents)
    coefficients_np = np.array(basis.coefficients)
    angular_np = np.array(basis.angular)

    nao_cart = centers_np.shape[0]

    shell_centers = []
    shell_exponents = []
    shell_coefficients = []
    shell_l_values = []
    shell_offsets = []

    i = 0
    while i < nao_cart:
        # This AO starts a new shell
        center = centers_np[i]
        exps = exponents_np[i]
        coeffs = coefficients_np[i]
        ang = angular_np[i]
        l_total = int(ang.sum())
        n_comp = N_CART[l_total]

        shell_centers.append(center)
        shell_exponents.append(exps)
        shell_coefficients.append(coeffs)
        shell_l_values.append(l_total)
        shell_offsets.append(i)

        # Skip over all components of this shell
        i += n_comp

    n_shells = len(shell_centers)

    return ShellData(
        centers=jnp.array(np.array(shell_centers)),
        exponents=jnp.array(np.array(shell_exponents)),
        coefficients=jnp.array(np.array(shell_coefficients)),
        l_values=jnp.array(np.array(shell_l_values), dtype=jnp.int32),
        shell_offsets=jnp.array(np.array(shell_offsets), dtype=jnp.int32),
        n_shells=n_shells,
        nao_cart=nao_cart,
        cart2sph=basis.cart2sph,
    )


def extract_shell_structure(basis: BasisData):
    """Extract static shell structure from a BasisData (CPU-side, one-time).

    Returns arrays that describe which AO indices form each shell, suitable
    for use inside jax.grad. The key insight is that shell grouping depends
    only on angular momenta (static ints), not on centers (which are traced).

    Returns:
        shell_ao_idx: (n_shells,) int, index of the first AO in each shell
        shell_exp_idx: (n_shells,) int, same, used to index exponents/coefficients
        l_values: (n_shells,) int, total angular momentum per shell
        n_shells: int
        nao_cart: int
    """
    angular_np = np.array(basis.angular)
    nao_cart = angular_np.shape[0]

    shell_ao_idx = []
    shell_l_values = []

    i = 0
    while i < nao_cart:
        ang = angular_np[i]
        l_total = int(ang.sum())
        shell_ao_idx.append(i)
        shell_l_values.append(l_total)
        i += N_CART[l_total]

    return (
        np.array(shell_ao_idx, dtype=np.int32),
        np.array(shell_l_values, dtype=np.int32),
        len(shell_ao_idx),
        nao_cart,
    )
