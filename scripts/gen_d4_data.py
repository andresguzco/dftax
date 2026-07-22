"""Generate the vendored D4 data tables (``dftax/energy/data/d4.npz``).

Build-time only (the pattern of ``gen_d3_data.py``): everything the JAX D4
implementation needs is extracted from `tad-dftd4
<https://github.com/dftd4/tad-dftd4>`_ and `tad-multicharge` (Apache-2.0;
torch is used only here, never at dftax runtime), which carry Grimme's
published D4/EEQ data.

Vendored, all indexed directly by atomic number (row 0 unused):
- ``rc6``: Casimir-Polder reference C6 (nelem, nelem, 7, 7) from the model's
  precomputed tensor (trapezoid over 23 imaginary frequencies).
- ``refcovcn``/``refc``/``refq``: reference coordination numbers, Gaussian
  counts (1 or 3), and EEQ reference charges (nelem, 7).
- ``zeff``/``gam``: effective nuclear charges and chemical hardnesses for the
  zeta charge-scaling (nelem,).
- ``rcov``/``en``: D3 covalent radii and Pauling electronegativities for the
  D4 coordination number (nelem,).
- ``eeq_chi``/``eeq_eta``/``eeq_kcn``/``eeq_rad``: EEQ 2019 parameters.
- ``methods``/``params``: D4(BJ) damping parameters (s6, a1, s8, a2) per
  functional (published Grimme group values).

Run with a throwaway environment that has torch + tad-dftd4:

    uv venv /tmp/d4venv && uv pip install --python /tmp/d4venv/bin/python \
        torch tad-dftd4
    /tmp/d4venv/bin/python scripts/gen_d4_data.py

The script also prints reference EEQ charges and D4 energies for the
molecules used in tests/unit/test_d4.py.
"""

from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parents[1] / "dftax" / "energy" / "data" / "d4.npz"

# Published D4(BJ) parameters (s6, a1, s8, a2) per method.
DAMPING = {
    "pbe":       (1.0, 0.38574991, 0.95948085, 4.80688534),
    "pbe0":      (1.0, 0.40085597, 1.20065498, 5.02928789),
    "b3lyp":     (1.0, 0.40868035, 2.02929367, 4.53807137),
    "cam-b3lyp": (1.0, 0.40147577, 1.66041301, 5.17350180),
    "r2scan":    (1.0, 0.49484001, 0.60187490, 5.73408312),
}

NELEM = 104   # direct atomic-number indexing, row 0 unused


def main() -> int:
    dd = dict(dtype=torch.float64)

    zmax = 86  # tad-dftd4 references cover H..Rn
    zs = torch.arange(1, zmax + 1)

    from tad_dftd4.model import D4Model

    m = D4Model(zs, **dd)
    rc6_small = m.rc6.numpy()                       # (86, 86, 7, 7)
    rc6 = np.zeros((NELEM, NELEM, 7, 7))
    rc6[1:zmax + 1, 1:zmax + 1] = rc6_small

    from tad_dftd4.reference import d4 as d4ref
    from tad_dftd4.reference.d4.charge_eeq import clsq

    def by_z(tbl):
        a = tbl.to(torch.float64).numpy() if isinstance(tbl, torch.Tensor) else np.asarray(tbl)
        a = a[:NELEM]                       # some tables extrapolate past Z=103
        out = np.zeros((NELEM,) + a.shape[1:])
        out[: a.shape[0]] = a
        return out

    refcovcn = by_z(d4ref.refcovcn)                 # (nelem, 7)
    refc = by_z(d4ref.refc)                         # (nelem, 7) ints
    refq = by_z(clsq)                               # (nelem, 7)

    from tad_dftd4 import data

    zeff = by_z(data.ZEFF(dtype=torch.float64))
    gam = by_z(data.GAM(**dd))

    from tad_mctc.data.radii import COV_D3
    from tad_mctc.data.en import PAULING

    rcov = by_z(COV_D3(**dd))
    en = by_z(PAULING(**dd))

    from tad_multicharge.model.eeq import EEQModel

    eeq = EEQModel.param2019(**dd)
    eeq_chi = by_z(eeq.chi)
    eeq_eta = by_z(eeq.eta)
    eeq_kcn = by_z(eeq.kcn)
    eeq_rad = by_z(eeq.rad)

    tables = {
        "rc6": rc6, "refcovcn": refcovcn, "refc": refc, "refq": refq,
        "zeff": zeff, "gam": gam, "rcov": rcov, "en": en,
        "eeq_chi": eeq_chi, "eeq_eta": eeq_eta, "eeq_kcn": eeq_kcn,
        "eeq_rad": eeq_rad,
        "methods": np.array(sorted(DAMPING)),
        "params": np.array([DAMPING[mth] for mth in sorted(DAMPING)]),
    }
    np.savez_compressed(OUT, **tables)
    print(f"wrote {OUT}: " + " ".join(f"{k}{v.shape}" for k, v in tables.items()
                                      if isinstance(v, np.ndarray) and v.ndim))

    # Reference values for tests/unit/test_d4.py.
    from tad_dftd4.disp import dftd4
    from tad_dftd4.damping import Param
    from tad_multicharge.model import eeq as eeq_mod

    systems = {
        "water": (
            torch.tensor([8, 1, 1]),
            torch.tensor([[0.0, 0.0, 0.0],
                          [1.43349, 0.0, 0.95297],
                          [1.43349, 0.0, -0.95297]], **dd),
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
                          [4.32570, 0.05669, 0.34016]], **dd),
        ),
    }
    for name, (z, xyz) in systems.items():
        chg = torch.tensor(0.0, **dd)
        q = eeq_mod.get_charges(z, xyz, chg)
        print(f"EEQ q ({name}): {[f'{v:.12e}' for v in q.numpy()]}")
        for mth in sorted(DAMPING):
            s6, a1, s8, a2 = DAMPING[mth]
            par = Param(**{k: torch.tensor(v, **dd) for k, v in
                           dict(s6=s6, a1=a1, s8=s8, a2=a2, s9=1.0).items()})
            e = float(dftd4(z, xyz, chg, par).sum())
            print(f"REF ({mth},{name}): {e:.15e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
