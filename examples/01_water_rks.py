"""Closed-shell KS-DFT on water across the four functionals (no PySCF needed)."""
import jax; jax.config.update("jax_enable_x64", True)
from dftax import KS, Molecule, scf
from dftax.energy.xc import LDA, PBE, PBE0, B3LYP

water = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
for xc in (LDA(), PBE(), PBE0(), B3LYP()):
    res = scf(KS(water, xc))
    print(f"{xc.name:6s}  E = {res.e_tot:.8f} Ha   converged={res.converged}")
