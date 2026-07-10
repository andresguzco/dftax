"""Open-shell KS on the CH3 radical (spin = 2S = 1 -> spin-polarized channels)."""
import jax; jax.config.update("jax_enable_x64", True)
from dftax import KS, Molecule, scf
from dftax.energy.xc import PBE, B3LYP

ch3 = Molecule.from_xyz(
    "C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0", "sto-3g", spin=1
)
for xc in (PBE(), B3LYP()):
    res = scf(KS(ch3, xc))  # spin inferred from the molecule -> alpha/beta channels
    print(f"{xc.name:6s}  E = {res.e_tot:.8f} Ha   converged={res.converged}")
