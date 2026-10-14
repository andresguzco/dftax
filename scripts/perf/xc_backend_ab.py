"""Materialized against streamed XC: what one Kohn-Sham iteration costs.

    source scripts/perf/env.sh
    python scripts/perf/xc_backend_ab.py --mol cubane bicyclo222octane

The same KS built twice, once with the AO values on the grid materialized
(``becke(chunk=None)``, :class:`~dftax.ks.terms.GridXC`) and once streamed
(``chunk="auto"`` as the size policy resolves it today), timing
``KS.energy_and_fock`` warm, which is what every SCF and gdm iteration calls,
and recording the device peak and the size of the stored AO values.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from profile_terms import git_sha, read_xyz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mol", nargs="*", default=["cubane"])
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--xc", nargs="*", default=["PBE"])
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import equinox as eqx
    import jax

    jax.config.update("jax_enable_x64", True)
    from dftax import KS, Molecule, becke, minao
    from dftax.energy import xc as xcmod
    from dftax.ks.guess import density_from_guess
    from dftax.ks.scf import canonical_orthonormalizer

    enfock = eqx.filter_jit(lambda k, P: k.energy_and_fock(P))
    for name in args.mol:
        mol = Molecule.from_xyz(read_xyz(name), args.basis, spherical=True)
        for xc in args.xc:
            row = dict(sha=git_sha(), mol=name, basis=args.basis, xc=xc)
            for arm, chunk in (("materialized", None), ("streamed", "auto")):
                ks = KS(mol, getattr(xcmod, xc)(), grid=becke(chunk=chunk))
                P = density_from_guess(ks, minao(),
                                       canonical_orthonormalizer(ks.S))
                jax.block_until_ready(enfock(ks, P))
                walls = []
                for _ in range(args.repeat):
                    t0 = time.perf_counter()
                    jax.block_until_ready(enfock(ks, P))
                    walls.append(time.perf_counter() - t0)
                ao_bytes = sum(int(x.size) * 8 for x in (
                    getattr(ks.xc_term, "ao", None),
                    getattr(ks.xc_term, "dao", None)) if x is not None)
                peak = max(int((d.memory_stats() or {}).get(
                    "peak_bytes_in_use", 0)) for d in jax.devices())
                row[arm] = dict(term=type(ks.xc_term).__name__,
                                warm_ms=statistics.median(walls) * 1e3,
                                ao_gib=ao_bytes / 2**30, peak_gib=peak / 2**30)
                del ks
            m, s = row["materialized"], row["streamed"]
            print(f"{name:17s} {xc:5s} materialized {m['warm_ms']:7.2f} ms "
                  f"(AO {m['ao_gib']:.2f} GiB)  streamed {s['warm_ms']:7.2f} ms"
                  f"  ({s['term']})  x{s['warm_ms'] / m['warm_ms']:.2f}",
                  flush=True)
            if args.out:
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
