"""A minimal, PySCF-free molecular system specification.

``Molecule`` holds element symbols, nuclear coordinates (atomic units / Bohr),
total charge and spin, plus the basis-set name. It exposes the small surface the
dftax KS driver needs (``atom_coords``, ``atom_charges``, ``nelectron``) so the
engine can run with no PySCF object anywhere in the pipeline.
"""

from __future__ import annotations

import numpy as np
import periodictable

# CODATA 2018 Bohr radius: 1 Bohr = 0.529177210903 Angstrom.
ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def symbol_to_Z(symbol: str) -> int:
    """Atomic number for an element symbol (e.g. ``"O"`` -> 8)."""
    return periodictable.elements.symbol(symbol.strip().capitalize()).number


def _parse_atom_string(atom: str) -> tuple[list[str], np.ndarray]:
    """Parse a PySCF-style atom string into symbols and coordinates.

    Accepts atoms separated by ``;`` or newlines, e.g.
    ``"O 0 0 0; H 0.757 0 0.587; H -0.757 0 0.587"``.
    """
    symbols: list[str] = []
    coords: list[list[float]] = []
    for token in atom.replace("\n", ";").split(";"):
        token = token.strip()
        if not token:
            continue
        parts = token.split()
        if len(parts) != 4:
            raise ValueError(f"Cannot parse atom line: {token!r}")
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:]])
    return symbols, np.asarray(coords, dtype=np.float64)


class Molecule:
    """A molecular system: atoms, geometry, charge/spin, and a basis name."""

    def __init__(
        self,
        symbols: list[str],
        coords_bohr: np.ndarray,
        basis: str,
        charge: int = 0,
        spin: int = 0,
        spherical: bool = False,
    ):
        self.symbols = [s.strip().capitalize() for s in symbols]
        self.coords = np.asarray(coords_bohr, dtype=np.float64).reshape(-1, 3)
        self.basis = basis
        self.charge = int(charge)
        self.spin = int(spin)  # 2S (number of unpaired electrons)
        # Spherical-harmonic AOs ((2l+1) per shell) vs Cartesian; spherical is
        # the standard convention for cc-pVXZ/def2 and required to match a
        # spherical reference for l >= 2 bases.
        self.spherical = bool(spherical)
        if len(self.symbols) != self.coords.shape[0]:
            raise ValueError("symbols and coords length mismatch")
        # Fail at construction, not at the KS build: nα/nβ must be integers.
        nelec = self.nelectron
        if (nelec + self.spin) % 2 != 0:
            raise ValueError(
                f"Inconsistent (nelec={nelec}, spin={self.spin}): nelec+spin "
                f"must be even; give the molecule its actual spin (= 2S)."
            )
        if nelec - self.spin < 0:
            raise ValueError(f"spin={self.spin} too large for nelec={nelec} (nβ<0).")

    @classmethod
    def from_xyz(
        cls,
        atom: str,
        basis: str,
        *,
        unit: str = "angstrom",
        charge: int = 0,
        spin: int = 0,
        spherical: bool = False,
    ) -> "Molecule":
        """Build from a PySCF-style atom string (Angstrom by default)."""
        symbols, coords = _parse_atom_string(atom)
        if unit.lower().startswith("ang"):
            coords = coords * ANGSTROM_TO_BOHR
        elif not unit.lower().startswith("b"):
            raise ValueError(f"unit must be 'angstrom' or 'bohr', got {unit!r}")
        return cls(symbols, coords, basis, charge=charge, spin=spin, spherical=spherical)

    def atom_coords(self) -> np.ndarray:
        """Nuclear coordinates in Bohr, shape (n_atoms, 3)."""
        return self.coords

    def atom_charges(self) -> np.ndarray:
        """Nuclear charges (atomic numbers), shape (n_atoms,)."""
        return np.array([symbol_to_Z(s) for s in self.symbols], dtype=np.float64)

    @property
    def nelectron(self) -> int:
        """Total electron count (sum of Z minus the total charge)."""
        return int(sum(symbol_to_Z(s) for s in self.symbols) - self.charge)
