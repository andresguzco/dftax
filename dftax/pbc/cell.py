from __future__ import annotations

import numpy as np

from dftax.pbc.gth import GTHPotential, load_potential
from dftax.system.molecule import ANGSTROM_TO_BOHR, _parse_atom_string, symbol_to_Z


class Cell:
    """Atoms in a periodic cell with GTH pseudopotentials.

    Coordinates and lattice vectors (the rows of ``a``) are stored in Bohr.
    ``spherical`` defaults to True: the GTH basis sets are defined, and used
    by PySCF and CP2K, with spherical harmonics.
    """

    def __init__(
        self,
        symbols: list[str],
        coords_bohr: np.ndarray,
        a_bohr: np.ndarray,
        basis: str,
        pseudo: str = "gth-pade",
        charge: int = 0,
        spin: int = 0,
        spherical: bool = True,
    ):
        self.symbols = [s.strip().capitalize() for s in symbols]
        self.coords = np.asarray(coords_bohr, dtype=np.float64).reshape(-1, 3)
        self.a = np.asarray(a_bohr, dtype=np.float64).reshape(3, 3)
        self.basis = basis
        self.pseudo = pseudo
        self.charge = int(charge)
        self.spin = int(spin)
        self.spherical = bool(spherical)
        if len(self.symbols) != self.coords.shape[0]:
            raise ValueError("symbols and coords length mismatch")
        if abs(np.linalg.det(self.a)) < 1e-12:
            raise ValueError("the lattice vectors are linearly dependent")
        by_element = {s: load_potential(pseudo, s) for s in set(self.symbols)}
        self.potentials: list[GTHPotential] = [by_element[s] for s in self.symbols]
        nelec = self.nelectron
        if (nelec + self.spin) % 2 != 0:
            raise ValueError(
                f"Inconsistent (nelec={nelec}, spin={self.spin}): nelec+spin "
                f"must be even; give the cell its actual spin (= 2S)."
            )
        if nelec - self.spin < 0:
            raise ValueError(f"spin={self.spin} too large for nelec={nelec} (nβ<0).")

    @classmethod
    def from_xyz(
        cls,
        atom: str,
        a,
        basis: str,
        pseudo: str = "gth-pade",
        *,
        unit: str = "angstrom",
        charge: int = 0,
        spin: int = 0,
        spherical: bool = True,
    ) -> "Cell":
        """Build from a PySCF-style atom string and lattice rows ``a``, both
        in ``unit`` (Angstrom by default)."""
        symbols, coords = _parse_atom_string(atom)
        a = np.asarray(a, dtype=np.float64).reshape(3, 3)
        if unit.lower().startswith("ang"):
            coords, a = coords * ANGSTROM_TO_BOHR, a * ANGSTROM_TO_BOHR
        elif not unit.lower().startswith("b"):
            raise ValueError(f"unit must be 'angstrom' or 'bohr', got {unit!r}")
        return cls(symbols, coords, a, basis, pseudo, charge=charge, spin=spin,
                   spherical=spherical)

    def atom_coords(self) -> np.ndarray:
        """Nuclear coordinates in Bohr, shape (n_atoms, 3)."""
        return self.coords

    def lattice_vectors(self) -> np.ndarray:
        """Lattice vectors in Bohr, one per row, shape (3, 3)."""
        return self.a

    def atom_charges(self) -> np.ndarray:
        """Ionic charges: the valence charge of each atom's pseudopotential."""
        return np.array([p.zion for p in self.potentials], dtype=np.float64)

    def atomic_numbers(self) -> np.ndarray:
        """Atomic numbers (the elements, not the ionic charges)."""
        return np.array([symbol_to_Z(s) for s in self.symbols], dtype=np.int64)

    @property
    def nelectron(self) -> int:
        """Valence electron count (ionic charges minus the total charge)."""
        return int(sum(p.zion for p in self.potentials) - self.charge)

    @property
    def volume(self) -> float:
        """Cell volume in Bohr^3."""
        return float(abs(np.linalg.det(self.a)))
