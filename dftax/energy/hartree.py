import jax
import numpy as np
import equinox as eqx
import jax.numpy as jnp

from functools import partial
from jaxtyping import Float, Array, Scalar
from dftax.utils import vmap


def _get_j(mol, dm):
    """Reference Coulomb matrix J via PySCF (test oracle only).

    PySCF is imported lazily here so that importing this module does not pull
    PySCF into the engine's compute-path dependency graph.
    """
    from pyscf import scf

    def aux(dm):
        in_dtype = dm.dtype
        dm = np.asarray(dm)
        vj = scf.jk.get_jk(mol, dm, "ijkl,ji->kl", aosym="s8")
        return vj.astype(in_dtype)

    result_shape_dtypes = jax.eval_shape(lambda: dm)
    return jax.pure_callback(aux, result_shape_dtypes, dm, vmap_method="broadcast_all")


class PairwiseHartree(eqx.Module):
    """O(N^2) pairwise Hartree energy estimator.

    Callable ``(r, nw) -> E_H`` interface, interchangeable with
    ``DensityFittedHartree``.
    """

    chunk_size: int | None = eqx.field(default=None, static=True)
    checkpoint: bool = eqx.field(default=True, static=True)

    def __call__(
        self,
        r: Float[Array, "n 3"],
        nw: Float[Array, "n"],
    ) -> Scalar:

        @partial(vmap, in_axes=(0, 0), chunk_size=self.chunk_size, checkpoint=self.checkpoint)
        def row_potential(i, ri):
            d2 = jnp.sum((r - ri) ** 2, axis=-1)
            rinv = jax.lax.rsqrt(d2 + 1e-12).at[i].set(0.0)
            return jnp.dot(rinv, nw)

        idx = jnp.arange(r.shape[0])
        pwise_vals = row_potential(idx, r)
        return 0.5 * jnp.dot(nw, pwise_vals)