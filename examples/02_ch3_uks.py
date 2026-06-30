"""Open-shell UKS on the CH3 radical (spin = 2S = 1 -> dispatches to UKS)."""
import jax; jax.config.update("jax_enable_x64", True)
from dftax import run_ks
from dftax.system import Molecule
from dftax.energy.xc import PBE, B3LYP

ch3 = Molecule.from_xyz(
    "C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0", "sto-3g", spin=1
)
for xc in (PBE(), B3LYP()):
    res = run_ks(ch3, xc)
    print(f"{xc.name:6s}  E = {res.e_tot:.8f} Ha   converged={res.converged}")
