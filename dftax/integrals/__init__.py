"""Pure-JAX one-electron integral matrix builders for KS-DFT.

Provides analytical overlap (S), kinetic (T), nuclear attraction (V),
and nuclear repulsion (V_nn) integrals via Obara-Saika recurrence,
fully differentiable and compatible with jit/vmap.
"""

from dftax.integrals.overlap import (
    overlap_matrix,
    cross_overlap_matrix,
    kinetic_matrix,
)
from dftax.integrals.nuclear_attraction import (
    nuclear_attraction_matrix,
)
from dftax.integrals.nuclear_repulsion import nuclear_repulsion
from dftax.integrals.eri2c import eri2c_matrix
from dftax.integrals.eri3c import eri3c_matrix
from dftax.integrals.multipole import dipole_matrices

__all__ = [
    "overlap_matrix",
    "cross_overlap_matrix",
    "kinetic_matrix",
    "nuclear_attraction_matrix",
    "nuclear_repulsion",
    "eri2c_matrix",
    "eri3c_matrix",
    "dipole_matrices",
]
