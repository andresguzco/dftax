# Multi-device execution

Sharding is a build-time value. The mesh spec shards the XC quadrature over
grid points and the DF 3-center tensor over auxiliary slabs; the sharded
terms are listed for reference; they are constructed by the builder, not by
hand.

::: dftax.ks.shard.mesh
::: dftax.ks.shard.MeshSpec
::: dftax.ks.terms.ShardedGridXC
::: dftax.ks.terms.ShardedDFCoulomb

## Across nodes

One process per node runs the same program; `distributed()` joins them into a
process group, after which the mesh spans every GPU of every node and nothing
else about the calculation changes. Call it before any other JAX work (dftax
itself starts no backend at import, so importing it first is fine).

::: dftax.ks.distributed.distributed
::: dftax.ks.distributed.is_coordinator
::: dftax.ks.distributed.barrier
