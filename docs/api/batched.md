# Batched evaluation

One `vmap`ped solve over a stack of geometries. Orbital-sized outputs are
opt-in; `mesh=` shards the batch axis across devices.

::: dftax.ks.batched.scf_batched
::: dftax.ks.batched.BatchedResult

## Batched direct minimization

Different molecules padded to common sizes and minimized in one compiled
program, closed and open shells, pure and hybrid. A system left unconverged
or in a non-aufbau state is re-solved alone by `gdm`
(`BatchedGDMResult.fallback`).

::: dftax.ks.batched_gdm.gdm_batched
::: dftax.ks.batched_gdm.BatchedGDMResult
