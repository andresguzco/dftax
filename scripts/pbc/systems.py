"""The cells the periodic work is developed and gated on.

Three materials, each with a small cell for tests and a medium one as the
baseline:

    si2, si8, si64          diamond silicon (covalent semiconductor)
    quartz9, quartz72       alpha-quartz SiO2 (oxide insulator, hexagonal)
    al1, al4, al32          fcc aluminium (metal)

Plain numpy on purpose: the reference scripts for PySCF, GPAW and VASP and
the dftax gates all build their inputs from here, so none of them depends on
another's cell type. Lengths are in Angstrom; lattice vectors are the rows of
``a``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

A_SI = 5.431
A_AL = 4.050
A_QUARTZ, C_QUARTZ = 4.916, 5.4054
U_SI = 0.4697
X_O, Y_O, Z_O = 0.4135, 0.2669, 0.1191


@dataclass(frozen=True)
class CellSpec:
    name: str
    symbols: tuple[str, ...]
    a: np.ndarray
    frac: np.ndarray

    @property
    def cart(self) -> np.ndarray:
        return self.frac @ self.a

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.a)))

    def repeat(self, n: tuple[int, int, int], name: str | None = None) -> "CellSpec":
        n = np.asarray(n, dtype=int)
        shifts = np.array([(i, j, k) for i in range(n[0])
                           for j in range(n[1]) for k in range(n[2])], float)
        frac = (self.frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3) / n
        return CellSpec(
            name=name or f"{self.name}x{n[0]}{n[1]}{n[2]}",
            symbols=self.symbols * len(shifts),
            a=self.a * n[:, None], frac=frac,
        )

    def scaled(self, volume_factor: float) -> "CellSpec":
        return replace(self, a=self.a * volume_factor ** (1.0 / 3.0))

    def rattled(self, amplitude: float = 0.05, seed: int = 0) -> "CellSpec":
        d = np.random.default_rng(seed).normal(scale=amplitude,
                                               size=self.frac.shape)
        return replace(self, frac=self.frac + d @ np.linalg.inv(self.a))

    def pyscf_atom(self) -> list:
        return [[s, tuple(map(float, r))] for s, r in zip(self.symbols, self.cart)]


def si2() -> CellSpec:
    a = 0.5 * A_SI * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return CellSpec("si2", ("Si", "Si"), a,
                    np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))


def si8() -> CellSpec:
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    frac = np.concatenate([fcc, fcc + 0.25])
    return CellSpec("si8", ("Si",) * 8, A_SI * np.eye(3), frac)


def al1() -> CellSpec:
    a = 0.5 * A_AL * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return CellSpec("al1", ("Al",), a, np.zeros((1, 3)))


def al4() -> CellSpec:
    frac = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    return CellSpec("al4", ("Al",) * 4, A_AL * np.eye(3), frac)


def quartz9() -> CellSpec:
    a = np.array([[A_QUARTZ, 0.0, 0.0],
                  [-0.5 * A_QUARTZ, 0.5 * np.sqrt(3.0) * A_QUARTZ, 0.0],
                  [0.0, 0.0, C_QUARTZ]])

    def screw(p):
        x, y, z = p
        return (-y, x - y, z + 2.0 / 3.0)

    si = [(U_SI, 0.0, 0.0)]
    ox = [(X_O, Y_O, Z_O), (X_O - Y_O, -Y_O, -Z_O)]
    for _ in range(2):
        si.append(screw(si[-1]))
        ox += [screw(ox[-2]), screw(ox[-1])]
    frac = np.mod(np.array(si + ox), 1.0)
    spec = CellSpec("quartz9", ("Si",) * 3 + ("O",) * 6, a, frac)
    _check_quartz(spec)
    return spec


def _check_quartz(spec: CellSpec) -> None:
    cart = spec.cart
    images = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
                       for k in (-1, 0, 1)], float) @ spec.a
    sym = np.array(spec.symbols)
    ox = cart[sym == "O"]
    for r in cart[sym == "Si"]:
        d = np.linalg.norm(ox[None, :, :] + images[:, None, :] - r, axis=-1)
        bonds = np.sort(d.ravel())[:5]
        if not (np.all(np.abs(bonds[:4] - 1.61) < 0.02) and bonds[4] > 2.5):
            raise AssertionError(f"quartz Si-O distances {bonds}")


def systems() -> dict[str, CellSpec]:
    return {
        "si2": si2(),
        "si8": si8(),
        "si64": si8().repeat((2, 2, 2), "si64"),
        "quartz9": quartz9(),
        "quartz72": quartz9().repeat((2, 2, 2), "quartz72"),
        "al1": al1(),
        "al4": al4(),
        "al32": al4().repeat((2, 2, 2), "al32"),
    }


BASELINES = ("si64", "quartz72", "al32")

EOS_VOLUME_FACTORS = tuple(np.linspace(0.94, 1.06, 7))
