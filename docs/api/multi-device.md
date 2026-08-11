# Multi-device execution

Sharding is a build-time value. The mesh spec shards the XC quadrature over
grid points and the DF 3-center tensor over auxiliary slabs; the sharded
terms are listed for reference; they are constructed by the builder, not by
hand.

`mesh()` composes with either DF backend. The materialized one gives each
device a `nao²·naux/ndev` slab, which is the capacity path until the slab
itself stops fitting; past that, `df(chunk=...)` with `mesh=` streams instead,
each device taking its own slice of the auxiliary range and never holding the
tensor at all.

The two shard on different axes. RI-J splits the auxiliary range, since that is
what `γ` is indexed by and all that has to cross devices. RI-K splits the
occupied set instead: both `E_K` and the exchange kernel are sums over occupied
orbitals, so each device scans its own slice and the partials are `psum`-reduced.
Padding that axis needs no masking, a zero orbital column contributing nothing
to either sum. Both halves of the `custom_vjp` shard identically, which is what
keeps the gradient exact: the exchange Fock the backward returns is the `psum`
of per-device partial kernels, i.e. the single-device matrix. Range-separated
exchange rides the same path on the attenuated metric.

The one combination that raises is VV10 (`WB97XV`) on a sharded grid, and it is
a structural limit rather than a gap: its double-grid pair quadrature is
nonlocal across shards and cannot be evaluated shard by shard.

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
