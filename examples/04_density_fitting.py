"""Density fitting vs the exact 4-center ERI, and the memory-light streamed path."""
import jax; jax.config.update("jax_enable_x64", True)
from dftax import KS, Molecule, becke, df, scf
from dftax.energy.xc import PBE

mol = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
AUX = "def2-universal-jkfit"
e_exact = scf(KS(mol, PBE())).e_tot
e_df = scf(KS(mol, PBE(), coulomb=df(AUX))).e_tot
e_stream = scf(KS(mol, PBE(),
                  coulomb=df(AUX, chunk=64, screen=1e-10),
                  grid=becke(chunk=20_000))).e_tot
print(f"exact         E = {e_exact:.8f}")
print(f"RI-J (DF)     E = {e_df:.8f}   |Δ| = {abs(e_df - e_exact):.1e}")
print(f"streamed+scr  E = {e_stream:.8f}   |Δ| = {abs(e_stream - e_exact):.1e}")
