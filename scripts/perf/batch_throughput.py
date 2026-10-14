"""Throughput on a dataset of small molecules: batched gdm against one at a time.

    source scripts/perf/env.sh
    python scripts/perf/batch_throughput.py --copies 4 --basis 6-31g

The dataset is a dozen small molecules, each at ``--copies`` randomly
distorted geometries (so the KS builds of one molecule share their compiled
integrals; the builds are not what is measured). Three ways to solve it:
sequential ``scf``, sequential ``gdm``, and one :func:`gdm_batched` over the
whole set. Reported: solves per second, warm (the batched program compiled
once beforehand) and the batched cold compile, plus the largest energy
difference against ``scf``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from profile_terms import git_sha  # noqa: E402

# Approximate geometries (Angstrom); minima are not needed, only molecules.
MOLECULES = {
    "water": "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
    "ammonia": "N 0 0 0.12; H 0 0.94 -0.27; H 0.81 -0.47 -0.27; "
               "H -0.81 -0.47 -0.27",
    "methane": "C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; "
               "H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63",
    "hf": "F 0 0 0; H 0 0 0.92",
    "co": "C 0 0 0; O 0 0 1.13",
    "n2": "N 0 0 0; N 0 0 1.10",
    "hcn": "H 0 0 -1.07; C 0 0 0; N 0 0 1.16",
    "formaldehyde": "C 0 0 0; O 0 0 1.21; H 0 0.94 -0.59; H 0 -0.94 -0.59",
    "methanol": "C -0.05 0.66 0; O -0.05 -0.76 0; H -1.09 0.98 0; "
                "H 0.44 1.07 0.89; H 0.44 1.07 -0.89; H 0.86 -1.06 0",
    "acetylene": "C 0 0 0.60; C 0 0 -0.60; H 0 0 1.66; H 0 0 -1.66",
    "ethylene": "C 0 0 0.67; C 0 0 -0.67; H 0 0.92 1.23; H 0 -0.92 1.23; "
                "H 0 0.92 -1.23; H 0 -0.92 -1.23",
    "fluoromethane": "C 0 0 0; F 0 0 1.38; H 1.03 0 -0.36; "
                     "H -0.52 0.89 -0.36; H -0.52 -0.89 -0.36",
}


def distorted(atom: str, rng, sigma: float) -> str:
    out = []
    for part in atom.split(";"):
        sym, *xyz = part.split()
        xyz = [float(v) + sigma * rng.standard_normal() for v in xyz]
        out.append(f"{sym} {xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}")
    return "; ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    from dftax import KS, Molecule, becke, gdm, scf
    from dftax.energy.xc import PBE
    from dftax.ks.batched_gdm import gdm_batched

    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    names, kss = [], []
    t0 = time.perf_counter()
    for name, atom in MOLECULES.items():
        for c in range(args.copies):
            geo = atom if c == 0 else distorted(atom, rng, args.sigma)
            kss.append(KS(Molecule.from_xyz(geo, args.basis), PBE(),
                          grid=becke(chunk=None)))
            names.append(name)
    jax.block_until_ready([ks.S for ks in kss])
    build_s = time.perf_counter() - t0
    n = len(kss)

    def timed(fn):
        t0 = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - t0

    # warm each path once on the whole set (sequential paths compile per
    # molecule shape; the batched one compiles once), then time a second pass
    seq_scf = lambda: [scf(ks) for ks in kss]                    # noqa: E731
    seq_gdm = lambda: [gdm(ks) for ks in kss]                    # noqa: E731
    batched = lambda: gdm_batched(kss)                           # noqa: E731
    _, cold_scf = timed(seq_scf)
    _, cold_gdm = timed(seq_gdm)
    _, cold_batch = timed(batched)
    r_scf, t_scf = timed(seq_scf)
    r_gdm, t_gdm = timed(seq_gdm)
    r_b, t_b = timed(batched)

    e_scf = np.array([r.e_tot for r in r_scf])
    de_gdm = np.max(np.abs(np.array([r.e_tot for r in r_gdm]) - e_scf))
    de_b = np.max(np.abs(r_b.e_tot - e_scf))
    row = dict(sha=git_sha(), basis=args.basis, n=n, build_s=build_s,
               seq_scf_s=t_scf, seq_gdm_s=t_gdm, batched_s=t_b,
               cold_seq_scf_s=cold_scf, cold_seq_gdm_s=cold_gdm,
               cold_batched_s=cold_batch,
               conv_scf=int(sum(r.converged for r in r_scf)),
               conv_gdm=int(sum(r.converged for r in r_gdm)),
               conv_batched=int(r_b.converged.sum()),
               max_de_gdm=float(de_gdm), max_de_batched=float(de_b),
               device=jax.devices()[0].device_kind)
    print(f"{n} molecules ({args.basis}, PBE), builds {build_s:.1f} s")
    for label, t, cold, conv in (
            ("sequential scf", t_scf, cold_scf, row["conv_scf"]),
            ("sequential gdm", t_gdm, cold_gdm, row["conv_gdm"]),
            ("batched gdm", t_b, cold_batch, row["conv_batched"])):
        print(f"  {label:15s} warm {t:7.2f} s  ({n / t:7.1f} mol/s)  "
              f"first pass {cold:7.2f} s  converged {conv}/{n}")
    print(f"  max |E - E_scf|: gdm {de_gdm:.1e}, batched {de_b:.1e}")
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
