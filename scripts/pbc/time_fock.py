"""Time the build and one Fock evaluation of a periodic cell in dftax.

    python scripts/pbc/time_fock.py quartz72 --mesh 122 122 134

The quantity that decides whether a cell is practical: the build is paid
once per geometry, the Fock evaluation once per SCF iteration. Reports which
grid path was taken (held AO values or the streamed pass) and the peak GPU
memory. Runs on a GPU node of the allocation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from systems import ANGSTROM_TO_BOHR, systems  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("system", choices=sorted(systems()))
    p.add_argument("--mesh", type=int, nargs=3, required=True)
    p.add_argument("--basis", default="gth-dzvp")
    p.add_argument("--pseudo", default="gth-pade")
    p.add_argument("--xc", default="lda")
    p.add_argument("--chunk", default="auto")
    p.add_argument("--repeat", type=int, default=2)
    args = p.parse_args()

    from dftax import KS
    from dftax.energy.xc import functional
    from dftax.ks.guess import density_from_guess
    from dftax.ks.scf import canonical_orthonormalizer
    from dftax.pbc import Cell, uniform

    spec = systems()[args.system]
    cell = Cell(list(spec.symbols), spec.cart * ANGSTROM_TO_BOHR,
                spec.a * ANGSTROM_TO_BOHR, args.basis, args.pseudo)
    chunk = args.chunk if args.chunk == "auto" else (
        None if args.chunk == "none" else int(args.chunk))

    t0 = time.perf_counter()
    ks = KS(cell, functional(args.xc), grid=uniform(mesh=args.mesh, chunk=chunk))
    jax.block_until_ready(ks.hcore)
    plan = getattr(ks.xc_term, "plan", None)
    screen = ("" if plan is None else
              f", blocks {plan.fun.shape[0]} x {plan.block} points, "
              f"{plan.fun.shape[1]} (row, image) pairs per block, "
              f"{plan.images.shape[0]} images")
    print(f"build {time.perf_counter() - t0:.1f} s: nao {ks.S.shape[0]}, "
          f"grid {ks.xc_term.weights.shape[0]}, {type(ks.xc_term).__name__}, "
          f"chunk {getattr(ks.xc_term, 'chunk', None)}{screen}", flush=True)

    P = density_from_guess(ks, None, canonical_orthonormalizer(ks.S))
    fock = eqx.filter_jit(lambda k, Q: k.energy_and_fock(Q))
    for i in range(args.repeat):
        t0 = time.perf_counter()
        e, F = fock(ks, P)
        jax.block_until_ready(F)
        print(f"fock {i}: {time.perf_counter() - t0:.1f} s, E(guess) {float(e):.9f}",
              flush=True)
    stats = jax.local_devices()[0].memory_stats() or {}
    print(f"peak device memory {stats.get('peak_bytes_in_use', 0) / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
