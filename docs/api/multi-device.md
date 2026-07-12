# Multi-device execution

Sharding is a build-time value. The mesh spec shards the XC quadrature over
grid points and the DF 3-center tensor over auxiliary slabs; the sharded
terms are listed for reference — they are constructed by the builder, not by
hand.

::: dftax.ks.shard.mesh
::: dftax.ks.shard.MeshSpec
::: dftax.ks.terms.ShardedGridXC
::: dftax.ks.terms.ShardedDFCoulomb
