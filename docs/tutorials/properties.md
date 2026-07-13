# Response properties

`dftax.ks.properties` is a PySCF-runtime-free property layer: dipole, polarizability,
Hessian → vibrational frequencies → IR / Raman, and alchemical derivatives. The
geometric quantities are finite differences of the analytic Pulay-free forces; the
dipole and polarizability are exact (matrix trace and finite field). Everything is
validated against PySCF and/or finite difference.

```python
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from dftax import Molecule, becke, dipole, polarizability, ir_spectrum, alchemical_deriv
from dftax.energy.xc import PBE

mol = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")

mu = np.asarray(dipole(mol, PBE(), debye=True))
print("dipole:", np.linalg.norm(mu), "Debye")            # ~1.92 (vs PySCF to 6e-7 a.u.)

alpha = np.asarray(polarizability(mol, PBE()))           # field finite difference
print("isotropic α:", np.trace(alpha) / 3, "a.u.")

ir = ir_spectrum(mol, PBE(), grid=becke(60, 194))
for f, a in sorted(zip(np.asarray(ir.frequencies), np.asarray(ir.intensities))):
    if f > 100:                                          # skip ~0 translations/rotations
        print(f"{f:8.1f} cm^-1   {a:6.2f} km/mol")       # freqs match PySCF to ~1 cm^-1

print("dE/dZ:", np.asarray(alchemical_deriv(mol, PBE())))  # Hellmann-Feynman alchemy
```

Every property takes `(mol, xc)` and runs its own SCF internally; grid quality is a
`grid=becke(...)` spec (the properties rebuild the grid at displaced geometries), and
extra keyword arguments are forwarded to `scf`. `vibrations`, `ir_spectrum`, and
`raman_spectrum` return NamedTuples (`.frequencies`, `.modes`/`.cart_modes`,
`.intensities`, `.activities`).

## What's available

| Function | Quantity | Method |
|---|---|---|
| `dipole` | `μ = Σ Z_A R_A − Tr(P r)` | exact (dipole integrals) |
| `polarizability` | `α_ij = ∂μ_i/∂E_j` | finite field, or `method="analytic"` (CPHF) |
| `hessian` | `∂²E/∂R∂R'` | FD of analytic forces |
| `vibrations` | harmonic frequencies + normal modes | mass-weighted Hessian, Eckart-projected |
| `ir_spectrum` | frequencies + IR intensities | `|dμ/dQ|²` |
| `raman_spectrum` | frequencies + Raman activities | `dα/dQ` (expensive) |
| `alchemical_deriv` | `∂E/∂Z_A` | Hellmann-Feynman (fixed-density autodiff) |

The harmonic analysis projects out translations and rotations (Eckart), so the six
external modes come back at ~0 and the vibrational frequencies match PySCF's
`harmonic_analysis` (water/sto-3g: <5 cm⁻¹). For the exact CPHF polarizability and the
status of the analytic Hessian, see [implicit differentiation](implicit-diff.md).
