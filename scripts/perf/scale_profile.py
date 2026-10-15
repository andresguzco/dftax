"""Where a warm solve of a large molecule goes, on the default KS.

    source scripts/perf/env.sh
    python scripts/perf/scale_profile.py --mol cholesterol --xc PBE0

Builds ``KS(mol, xc)`` with the defaults (whatever DF storage, screening and
XC backend they pick at this size), then times, warm (median of
``--repeat``): a rebuild at the same geometry, one whole Fock build
(``energy_and_fock``), the Coulomb term alone (RI-J and RI-K,
``energy_and_potential``), the XC term alone, and the whole ``scf``. The
point is to split a slow warm solve into the piece that is slow.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mol", default="cholesterol")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--xc", default="PBE0")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import equinox as eqx
    import jax
    jax.config.update("jax_enable_x64", True)
    from profile_terms import read_xyz

    from dftax import KS, Molecule, scf
    from dftax.energy import xc as xcmod

    def wall(fn):
        jax.block_until_ready(fn())
        ws = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            jax.block_until_ready(fn())
            ws.append(time.perf_counter() - t0)
        return statistics.median(ws)

    mol = Molecule.from_xyz(read_xyz(args.mol), args.basis)
    xc = getattr(xcmod, args.xc)()
    t0 = time.perf_counter()
    ks = KS(mol, xc)
    jax.block_until_ready(ks)
    cold = time.perf_counter() - t0
    c, x = ks.coulomb, ks.xc_term
    print(f"{args.mol} {args.xc}: nao {ks.S.shape[0]}, coulomb "
          f"{type(c).__name__}, xc {type(x).__name__}; cold build {cold:.0f} s",
          flush=True)
    t_build = wall(lambda: KS(mol, xc))
    r = scf(ks)
    P = r.P
    fock = eqx.filter_jit(lambda k, P: k.energy_and_fock(P))
    coul = eqx.filter_jit(
        lambda k, P: k.coulomb.energy_and_potential(P, k.S, k.nocc))
    xcf = eqx.filter_jit(lambda k, P: k.xc_term.energy_and_potential(P))
    t_fock = wall(lambda: fock(ks, P))
    t_coul = wall(lambda: coul(ks, P))
    t_xc = wall(lambda: xcf(ks, P))
    t_scf = wall(lambda: scf(ks).e_tot)
    print(f"  warm rebuild {t_build:.2f} s | one Fock {t_fock * 1e3:.0f} ms "
          f"(Coulomb {t_coul * 1e3:.0f} ms, XC {t_xc * 1e3:.0f} ms) | scf "
          f"{t_scf:.2f} s over {r.n_iter} iterations", flush=True)


if __name__ == "__main__":
    main()
