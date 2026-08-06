# Multi-device execution

Sharding is a build-time value. The mesh spec shards the XC quadrature over
grid points and the DF 3-center tensor over auxiliary slabs; the sharded
terms are listed for reference; they are constructed by the builder, not by
hand.

`mesh()` composes with either DF backend. The materialized one gives each
device a `nao²·naux/ndev` slab, which is the capacity path until the slab
itself stops fitting; past that, `df(chunk=...)` with `mesh=` streams instead,
each device taking its own slice of the auxiliary range and never holding the
tensor at all. The streamed combination covers RI-J; a hybrid needs the
materialized backend, since streamed RI-K has no sharded form yet.

::: dftax.ks.shard.mesh
::: dftax.ks.shard.MeshSpec
::: dftax.ks.terms.ShardedGridXC
::: dftax.ks.terms.ShardedDFCoulomb
::: dftax.ks.terms.ShardedStreamedDFCoulomb

## Across nodes

One process per node runs the same program; `distributed()` joins them into a
process group, after which the mesh spans every GPU of every node and nothing
else about the calculation changes. Call it before any other JAX work (dftax
itself starts no backend at import, so importing it first is fine).

::: dftax.ks.distributed.distributed
::: dftax.ks.distributed.is_coordinator
::: dftax.ks.distributed.barrier
