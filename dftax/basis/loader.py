"""Build :class:`~dftax.energy.gto.BasisData` from Basis Set Exchange data.

This is the PySCF-free counterpart of ``extract_basis_data``: it fetches the
contracted-GTO definition for each element from the ``basis_set_exchange``
library and assembles the same normalized Cartesian-AO arrays the integral
engine consumes. Normalization matches ``extract_basis_data`` exactly
(primitive ``gto_norm`` times, for l<=1, a contracted norm; libcint's
per-shell convention for l>=2).

Emits Cartesian GTOs by default (``cart2sph=None``); with ``spherical=True`` it
also builds the Cartesian->spherical transform (vendored per-l blocks under
``data/cart2sph.npz``) so spherical d/f bases (cc-pVDZ, cc-pVTZ, def2-*) match a
spherical PySCF reference. For l<=1 bases the two spans coincide.
"""

from __future__ import annotations

import math
from functools import lru_cache
from importlib.resources import files

import numpy as np
import jax.numpy as jnp
import basis_set_exchange as bse

from dftax.energy.gto import BasisData, _CART_COMPONENTS, _contracted_norm
from dftax.system.molecule import symbol_to_Z


@lru_cache(maxsize=1)
def _cart2sph_blocks() -> dict[int, np.ndarray]:
    """Vendored per-l Cartesian->spherical transform blocks (ncart x nsph)."""
    path = files("dftax.basis").joinpath("data", "cart2sph.npz")
    with path.open("rb") as fh:
        npz = np.load(fh)
        return {int(k[1:]): np.asarray(npz[k], dtype=np.float64) for k in npz.files}


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    """Assemble a block-diagonal matrix from a list of 2-D blocks (no SciPy)."""
    nr = sum(b.shape[0] for b in blocks)
    nc = sum(b.shape[1] for b in blocks)
    out = np.zeros((nr, nc), dtype=np.float64)
    r = c = 0
    for b in blocks:
        out[r : r + b.shape[0], c : c + b.shape[1]] = b
        r += b.shape[0]
        c += b.shape[1]
    return out


def gto_norm(l: int, alpha: float) -> float:
    """Radial normalization of a primitive GTO (matches ``pyscf.gto.gto_norm``).

    ``N`` such that the primitive ``N r^l e^{-alpha r^2}`` has unit self-overlap
    in the radial/spherical sense.
    """
    f = (
        2 ** (2 * l + 3)
        * math.factorial(l + 1)
        * (2 * alpha) ** (l + 1.5)
        / (math.factorial(2 * l + 2) * math.sqrt(math.pi))
    )
    return math.sqrt(f)


def _iter_contractions(shell):
    """Yield ``(l, coeff_vector)`` for each contracted function in a BSE shell.

    - single angular momentum: each coefficient block is an independent
      (general) contraction of that L;
    - multiple angular momenta (e.g. an ``sp`` shell): block ``i`` belongs to
      ``angular_momentum[i]``.
    """
    ams = shell["angular_momentum"]
    cblocks = shell["coefficients"]
    if len(ams) == 1:
        for c in cblocks:
            yield ams[0], np.asarray(c, dtype=np.float64)
    else:
        for l, c in zip(ams, cblocks):
            yield l, np.asarray(c, dtype=np.float64)


def build_basis_data(
    symbols: list[str],
    coords_bohr: np.ndarray,
    basis_name: str,
    *,
    spherical: bool = False,
    return_atom_index: bool = False,
):
    """Assemble :class:`BasisData` for a molecule from a named basis set.

    Args:
        symbols: element symbols, one per atom.
        coords_bohr: nuclear coordinates in Bohr, shape (n_atoms, 3).
        basis_name: a Basis Set Exchange basis name (e.g. ``"sto-3g"``).
        spherical: if True, attach the Cartesian->spherical transform so the
            basis uses (2l+1) spherical harmonics (standard for cc-pVXZ/def2);
            otherwise emit Cartesian GTOs (``cart2sph=None``). For l<=1 the two
            coincide.
        return_atom_index: if True, also return an int array mapping each
            Cartesian AO to its owning atom (needed to rebuild differentiable
            centers from nuclear coordinates, e.g. for forces).
    """
    coords = np.asarray(coords_bohr, dtype=np.float64).reshape(-1, 3)
    Zs = [symbol_to_Z(s) for s in symbols]
    bdata = bse.get_basis(basis_name, elements=sorted(set(Zs)), header=False)
    elements = bdata["elements"]

    # Gather every contracted shell: (l, exponents, raw coefficients, center, atom).
    raw_shells: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for atom_idx, (Z, center) in enumerate(zip(Zs, coords)):
        shells = elements[str(Z)]["electron_shells"]
        for shell in shells:
            exps = np.asarray(shell["exponents"], dtype=np.float64)
            for l, c_raw in _iter_contractions(shell):
                raw_shells.append((l, exps, c_raw, center, atom_idx))

    max_prim = max(len(exps) for (_, exps, _, _, _) in raw_shells)

    centers, all_exps, all_coeffs, angular, atom_index = [], [], [], [], []
    for l, exps, c_raw, center, atom_idx in raw_shells:
        prim_norms = np.array([gto_norm(l, e) for e in exps])
        c_prim = c_raw * prim_norms
        for ang in _CART_COMPONENTS[l]:
            if l <= 1:
                c_final = c_prim * _contracted_norm(exps, c_prim, ang)
            else:
                c_final = c_prim.copy()
            pad_e = np.zeros(max_prim, dtype=np.float64)
            pad_c = np.zeros(max_prim, dtype=np.float64)
            pad_e[: len(exps)] = exps
            pad_c[: len(c_final)] = c_final
            centers.append(center)
            all_exps.append(pad_e)
            all_coeffs.append(pad_c)
            angular.append(ang)
            atom_index.append(atom_idx)

    cart2sph = None
    if spherical:
        # One cart2sph(l) block per contracted shell, in build order.
        blocks = _cart2sph_blocks()
        cart2sph = jnp.asarray(
            _block_diag([blocks[l] for (l, _, _, _, _) in raw_shells])
        )

    basis = BasisData(
        centers=jnp.asarray(np.array(centers, dtype=np.float64)),
        exponents=jnp.asarray(np.array(all_exps, dtype=np.float64)),
        coefficients=jnp.asarray(np.array(all_coeffs, dtype=np.float64)),
        angular=jnp.asarray(np.array(angular, dtype=np.int32)),
        cart2sph=cart2sph,
        max_l=int(max(l for (l, _, _, _, _) in raw_shells)),
    )
    if return_atom_index:
        return basis, np.array(atom_index, dtype=np.int64)
    return basis
