"""The hard set through gdm and gdm_batched, with and without the fallback.

    source scripts/perf/env.sh
    python scripts/perf/batched_hard.py

The cases :mod:`solver_ab` singles out (stretched N2, radicals, a triplet, the
Fe atom, Cr2), each a batch of one: :func:`~dftax.ks.gdm.gdm`, the batched
loop alone (``fallback=False``), and :func:`~dftax.ks.batched_gdm.gdm_batched`
as shipped, which re-solves an unconverged or non-aufbau system with gdm.
Reports iterations, convergence, the fallback flag and energies against gdm.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from solver_ab import HARD  # noqa: E402


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from dftax import KS, Molecule, gdm
    from dftax.energy.xc import PBE, PBE0
    from dftax.ks.batched_gdm import gdm_batched

    def tag(conv):
        return "conv" if conv else "NC  "

    for xc in (PBE, PBE0):
        for name, (atoms, basis, spin) in HARD.items():
            mol = Molecule.from_xyz(atoms, basis, spin=spin)
            ks = KS(mol, xc(), spin=spin if spin else None)
            g = gdm(ks)
            raw = gdm_batched([ks], fallback=False)
            fb = gdm_batched([ks])
            print(f"{xc.__name__:5s} {name:11s} gdm {g.n_iter:4d} {tag(g.converged)}"
                  f" | batched {int(raw.n_iter[0]):4d} {tag(raw.converged[0])} "
                  f"dE {float(raw.e_tot[0]) - g.e_tot:+.1e} | shipped "
                  f"{'fallback' if fb.fallback[0] else 'batched '} "
                  f"{tag(fb.converged[0])} dE {float(fb.e_tot[0]) - g.e_tot:+.1e}",
                  flush=True)


if __name__ == "__main__":
    main()
