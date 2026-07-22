"""Generate the vendored D3(BJ) data tables (``dftax/energy/data/d3bj.npz``).

Build-time only (the pattern of ``gen_lebedev.py``): the reference C6
coefficients, reference coordination numbers, D3 covalent radii, and
sqrt(Q) = sqrt(r4/r2) expectation values are extracted from `tad-dftd3
<https://github.com/dftd3/tad-dftd3>`_ (Apache-2.0; torch is used only here,
never at dftax runtime), which itself carries Grimme's published D3 data.
BJ-damping parameters per functional are the published Grimme group values.

Run with a throwaway environment that has torch + tad-dftd3:

    uv venv /tmp/d3venv && uv pip install --python /tmp/d3venv/bin/python \
        torch tad-dftd3
    /tmp/d3venv/bin/python scripts/gen_d3_data.py

The script also prints reference dispersion energies (computed with
tad-dftd3 itself) for the molecules used in tests/unit/test_d3.py.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from tad_dftd3 import dftd3
from tad_dftd3.reference import Reference
from tad_dftd3.data import r4r2 as r4r2_mod  # noqa: F401 (module probe)
from tad_dftd3.data import R4R2

OUT = Path(__file__).resolve().parents[1] / "dftax" / "energy" / "data" / "d3bj.npz"

# Published D3(BJ) parameters (s6, a1, s8, a2) per method (Grimme group tables).
DAMPING = {
    "pbe":      (1.0, 0.4289, 0.7875, 4.4407),
    "pbe0":     (1.0, 0.4145, 1.2177, 4.8593),
    "b3lyp":    (1.0, 0.3981, 1.9889, 4.4211),
    "cam-b3lyp": (1.0, 0.3708, 2.0674, 5.4743),
    "r2scan":   (1.0, 0.4948, 0.6018, 5.7308),
}


def main() -> int:
    ref = Reference(dtype=torch.float64)
    c6 = ref.c6.numpy()                       # (104, 104, 7, 7)
    cn = ref.cn.numpy()                       # (104, 7); -1 marks unused slots

    # sqrt(Q) table (R4R2() already returns sqrt-scaled values; see its doc).
    r4r2 = R4R2(dtype=torch.float64).numpy()

    # D3 covalent radii (used by the CN counting function). tad-dftd3 routes
    # these through tad_mctc; take exactly what its ncoord uses.
    from tad_mctc.data.radii import COV_D3

    rcov = COV_D3(dtype=torch.float64).numpy()

    # Pairwise van-der-Waals radii: the ATM three-body damping radii
    # (srvdw = rs9 * vdw in tad-dftd3's dispersion_atm; the driver gathers
    # radii.VDW_PAIRWISE[zi, zj]).
    from tad_mctc.data.radii import VDW_PAIRWISE

    vdw = VDW_PAIRWISE(dtype=torch.float64).numpy()

    tables = {
        "c6": c6, "cn_ref": cn, "r4r2": r4r2, "rcov": rcov, "vdw": vdw,
        "methods": np.array(sorted(DAMPING)),
        "params": np.array([DAMPING[m] for m in sorted(DAMPING)]),
    }
    np.savez_compressed(OUT, **tables)
    print(f"wrote {OUT}: c6{c6.shape} cn{cn.shape} r4r2{r4r2.shape} "
          f"rcov{rcov.shape} vdw{vdw.shape}")

    # Reference energies for tests (two-body D3(BJ), no ATM: s9=0).
    systems = {
        "water": (
            torch.tensor([8, 1, 1]),
            torch.tensor([[0.0, 0.0, 0.0],
                          [1.43349, 0.0, 0.95297],
                          [1.43349, 0.0, -0.95297]], dtype=torch.float64),
        ),
        "ethanol": (
            torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1]),
            torch.tensor([[-1.67619, 0.31561, -0.03213],
                          [0.87500, -0.98268, 0.02646],
                          [2.72311, 0.89762, 0.50833],
                          [-1.80471, 1.67623, 1.51744],
                          [-3.18453, -1.08281, 0.14175],
                          [-1.91430, 1.34561, -1.80667],
                          [0.94491, -2.41145, 1.51744],
                          [1.28893, -1.92565, -1.76888],
                          [4.32570, 0.05669, 0.34016]], dtype=torch.float64),
        ),
    }
    for method, (s6, a1, s8, a2) in sorted(DAMPING.items()):
        param = {
            "s6": torch.tensor(s6, dtype=torch.float64),
            "s8": torch.tensor(s8, dtype=torch.float64),
            "a1": torch.tensor(a1, dtype=torch.float64),
            "a2": torch.tensor(a2, dtype=torch.float64),
            "s9": torch.tensor(0.0, dtype=torch.float64),
        }
        for name, (numbers, positions) in systems.items():
            e = dftd3(numbers, positions, param).sum()
            print(f"E_disp[{method:9s}][{name:7s}] = {float(e):+.12e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
