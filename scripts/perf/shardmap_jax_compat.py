"""Minimal repro of the jax 0.11 failure in the sharded RI-K contraction.

Mirrors ShardedDFCoulomb.exchange: a shard_map over an "aux" axis where the
3-center slab (nao, nao, slab) is contracted against a replicated matrix,
`mnP,PX->mnX`. jax 0.11 rejects it with a 3-D sharding on the 2-D bitcast the
dot_general takes as its operand. Each formulation below is a candidate fix.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 python repro_shardmap.py
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map

jax.config.update("jax_enable_x64", True)

devs = jax.devices()
ndev = len(devs)
mesh = jax.sharding.Mesh(np.asarray(devs), ("aux",))
spec = jax.sharding.PartitionSpec
nao, slab = 7, 6
nauxp = slab * ndev

key = jax.random.PRNGKey(0)
t3 = jax.random.normal(key, (nao, nao, nauxp))
L = jax.random.normal(key, (nauxp, nauxp))


def make(kind):
    def exchange(t3x, Lfull):
        my = jax.lax.axis_index("aux")
        rows = jax.lax.dynamic_slice_in_dim(Lfull, my * slab, slab, axis=0)
        W = jnp.zeros_like(t3x)
        for d in range(ndev):
            block = rows[:, d * slab:(d + 1) * slab]
            if kind == "einsum":                       # what dftax does today
                part = jnp.einsum("mnP,PX->mnX", t3x, block)
            elif kind == "reshape_matmul":             # explicit 2-D matmul
                m, n, p = t3x.shape
                part = (t3x.reshape(m * n, p) @ block).reshape(m, n, -1)
            elif kind == "tensordot":
                part = jnp.tensordot(t3x, block, axes=((2,), (0,)))
            elif kind == "dot_general":
                part = jax.lax.dot_general(
                    t3x, block, (((2,), (0,)), ((), ())))
            W = jnp.where(my == d, jax.lax.psum(part, "aux"), W)
        return jax.lax.psum(jnp.sum(W * W), "aux")

    return shard_map(
        exchange, mesh=mesh,
        in_specs=(spec(None, None, "aux"), spec()),
        out_specs=spec(), check_vma=False,
    )


print(f"jax {jax.__version__}, {ndev} devices")
for kind in ("einsum", "reshape_matmul", "tensordot", "dot_general"):
    try:
        val = float(jax.jit(make(kind))(t3, L))
        print(f"  {kind:16s} OK    {val:.8f}")
    except Exception as exc:                                   # noqa: BLE001
        first = str(exc).strip().splitlines()[0][:130]
        print(f"  {kind:16s} FAIL  {first}")
sys.exit(0)
