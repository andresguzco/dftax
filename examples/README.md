# Examples

Runnable from the repo root (`uv run python examples/<file>.py`):

| file | what |
|---|---|
| `01_water_rks.py` | closed-shell RKS on water, all four functionals (LDA/PBE/PBE0/B3LYP) |
| `02_ch3_uks.py` | open-shell UKS on the CH₃ radical (PBE, B3LYP) |
| `03_forces_h2.py` | analytic nuclear forces vs finite difference |
| `04_density_fitting.py` | exact ERI vs RI density fitting vs streamed + screened DF |
| `05_batched.py` | batched KS over many geometries (vmap), energies + analytic forces |
| `06_properties.py` | response properties: dipole, polarizability, IR spectrum, alchemy |

All are PySCF-free. For GPU, install the CUDA extra (`uv sync --extra cuda12`).
