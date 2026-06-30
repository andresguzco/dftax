"""Molecular response properties: dipole, polarizability, IR spectrum, alchemy.

All PySCF-runtime-free. The geometric quantities are finite differences of the
analytic Pulay-free forces / dipole; the dipole and polarizability are exact (matrix
trace and finite field). See dftax.ks.properties for the full API.
"""
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from dftax import dipole, polarizability, ir_spectrum, alchemical_deriv
from dftax.system import Molecule
from dftax.energy.xc import PBE

mol = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")

mu = np.asarray(dipole(mol, PBE(), debye=True))
print(f"dipole moment: {np.linalg.norm(mu):.4f} Debye  {np.round(mu, 4)}")

alpha = np.asarray(polarizability(mol, PBE()))                    # finite field
print(f"isotropic polarizability (FD):       {np.trace(alpha) / 3:.4f} a.u.")
alpha_a = np.asarray(polarizability(mol, PBE(), method="analytic"))  # implicit-diff CPHF
print(f"isotropic polarizability (analytic): {np.trace(alpha_a) / 3:.4f} a.u.")

ir = ir_spectrum(mol, PBE(), n_radial=60, lebedev=194)
freq = np.asarray(ir["frequencies"]); inten = np.asarray(ir["intensities"])
print("\nIR-active modes (vibrational):")
for f, a in sorted(zip(freq, inten)):
    if f > 100:  # skip the ~0 translations/rotations
        print(f"  {f:8.1f} cm^-1   {a:7.2f} km/mol")

dEdZ = np.asarray(alchemical_deriv(mol, PBE()))
print("\nalchemical gradient dE/dZ (Ha/charge):", np.round(dEdZ, 4))
