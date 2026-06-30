"""Density fitting vs the exact 4-center ERI, and the memory-light streamed path."""
import jax; jax.config.update("jax_enable_x64", True)
from dftax import run_rks
from dftax.system import Molecule
from dftax.energy.xc import PBE

mol = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
AUX = "def2-universal-jkfit"
e_exact = run_rks(mol, PBE()).e_tot
e_df = run_rks(mol, PBE(), auxbasis=AUX).e_tot
e_stream = run_rks(mol, PBE(), auxbasis=AUX, df_chunk=64, df_screen=1e-10, grid_chunk=20000).e_tot
print(f"exact         E = {e_exact:.8f}")
print(f"RI-J (DF)     E = {e_df:.8f}   |Δ| = {abs(e_df - e_exact):.1e}")
print(f"streamed+scr  E = {e_stream:.8f}   |Δ| = {abs(e_stream - e_exact):.1e}")
