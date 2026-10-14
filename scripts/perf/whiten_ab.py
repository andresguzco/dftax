"""Whitened DF tensor against the plain one: build, memory, evaluations, solves.

    source scripts/perf/env.sh
    python scripts/perf/whiten_ab.py --mol cubane bicyclo --xc PBE PBE0

Each (molecule, functional, whiten) runs in its own process, so the device
peak is that variant's alone. Reports the KS build (cold, then warm on a
second identical build), the device peak after it, the warm Fock build
(``energy_and_fock``) and orbital-route evaluation
(``energy_and_grad_orbitals``) at the converged density, and warm scf and gdm
solves with their energies.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _median(fn, repeat):
    import jax
    jax.block_until_ready(fn())
    walls = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        walls.append(time.perf_counter() - t0)
    return statistics.median(walls) * 1e3


def child(mol_name, basis, xc_name, whiten, repeat):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from profile_terms import read_xyz

    from dftax import KS, Molecule, gdm, scf
    from dftax.energy import xc as xcmod
    from dftax.ks import df

    mol = Molecule.from_xyz(read_xyz(mol_name), basis)
    xc = getattr(xcmod, xc_name)()
    spec = df(chunk=None, whiten=whiten)
    t0 = time.perf_counter()
    ks = KS(mol, xc, coulomb=spec)
    jax.block_until_ready(ks)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    ks = KS(mol, xc, coulomb=spec)
    jax.block_until_ready(ks)
    warm = time.perf_counter() - t0
    dev = jax.devices()[0]
    peak = (dev.memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30

    r = scf(ks)
    t_scf = _median(lambda: scf(ks).e_tot, repeat)
    g = gdm(ks)
    t_gdm = _median(lambda: gdm(ks).e_tot, repeat)
    P = jnp.asarray(r.P)
    fock = eqx.filter_jit(lambda k, P: k.energy_and_fock(P))
    t_fock = _median(lambda: fock(ks, P), repeat)
    C = jnp.asarray(r.mo_coeff)                     # (nspin, nao, nmo)
    Cs = tuple(C[s, :, :n] for s, n in enumerate(ks.nocc))
    orb = eqx.filter_jit(lambda k, Cs: k.energy_and_grad_orbitals(Cs))
    t_orb = _median(lambda: orb(ks, Cs), repeat)
    return dict(mol=mol_name, xc=xc_name, whiten=whiten,
                term=type(ks.coulomb).__name__, nao=int(ks.S.shape[0]),
                build_cold_s=cold, build_warm_s=warm, peak_gib=peak,
                fock_ms=t_fock, orb_ms=t_orb, scf_ms=t_scf, gdm_ms=t_gdm,
                e_scf=float(r.e_tot), e_gdm=float(g.e_tot),
                it_scf=int(r.n_iter), it_gdm=int(g.n_iter))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mol", nargs="*", default=["water", "cubane"])
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--xc", nargs="*", default=["PBE", "PBE0"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.child:
        m, x, w = args.child.split(",")
        print(json.dumps(child(m, args.basis, x, w == "1", args.repeat)))
        return
    rows = []
    for m in args.mol:
        for x in args.xc:
            for w in (0, 1):
                out = subprocess.run(
                    [sys.executable, __file__, "--child", f"{m},{x},{w}",
                     "--basis", args.basis, "--repeat", str(args.repeat)],
                    capture_output=True, text=True)
                line = [ln for ln in out.stdout.splitlines()
                        if ln.startswith("{")]
                if out.returncode or not line:
                    print(f"{m} {x} whiten={w}: FAILED\n{out.stderr[-2000:]}")
                    continue
                r = json.loads(line[-1])
                rows.append(r)
                print(f"{m:10s} {x:6s} {r['term']:17s} nao {r['nao']:4d} "
                      f"build {r['build_cold_s']:6.1f}/{r['build_warm_s']:5.2f} s "
                      f"peak {r['peak_gib']:5.2f} GiB  fock {r['fock_ms']:7.2f} "
                      f"orb {r['orb_ms']:7.2f}  scf {r['scf_ms']:7.1f} "
                      f"({r['it_scf']}) gdm {r['gdm_ms']:7.1f} ({r['it_gdm']}) ms "
                      f"E {r['e_scf']:.10f} / {r['e_gdm']:.10f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
