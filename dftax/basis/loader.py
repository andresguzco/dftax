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
    path = files("dftax.basis").joinpath("data", "cart2sph.npz")
    with path.open("rb") as fh:
        npz = np.load(fh)
        return {int(k[1:]): np.asarray(npz[k], dtype=np.float64) for k in npz.files}


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
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
    f = (
        2 ** (2 * l + 3)
        * math.factorial(l + 1)
        * (2 * alpha) ** (l + 1.5)
        / (math.factorial(2 * l + 2) * math.sqrt(math.pi))
    )
    return math.sqrt(f)


def _iter_contractions(shell):
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
    coords = np.asarray(coords_bohr, dtype=np.float64).reshape(-1, 3)
    Zs = [symbol_to_Z(s) for s in symbols]
    gth = basis_name.lower().startswith("gth-")
    if gth:
        from dftax.pbc.gth import load_basis_shells

        elements = {
            str(Z): {"electron_shells": load_basis_shells(basis_name, s)}
            for s, Z in set(zip(symbols, Zs))
        }
    else:
        bdata = bse.get_basis(basis_name, elements=sorted(set(Zs)), header=False)
        elements = bdata["elements"]

    raw_shells: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for atom_idx, (Z, center) in enumerate(zip(Zs, coords)):
        shells = elements[str(Z)]["electron_shells"]
        for shell in shells:
            exps = np.asarray(shell["exponents"], dtype=np.float64)
            for l, c_raw in _iter_contractions(shell):
                if gth and l >= 2 and np.count_nonzero(c_raw) > 1:
                    raise NotImplementedError(
                        f"{basis_name!r} has a contracted l={l} shell; only "
                        f"single-primitive d and higher shells are supported "
                        f"for GTH basis sets (the DZVP/TZV2P/QZV3P families)."
                    )
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
        blocks = _cart2sph_blocks()
        cart2sph = jnp.asarray(
            _block_diag([blocks[l] for (l, _, _, _, _) in raw_shells])
        )

    ang_np = np.array(angular, dtype=np.int32)
    exp_np = np.array(all_exps, dtype=np.float64)
    basis = BasisData(
        centers=jnp.asarray(np.array(centers, dtype=np.float64)),
        exponents=jnp.asarray(exp_np),
        coefficients=jnp.asarray(np.array(all_coeffs, dtype=np.float64)),
        angular=jnp.asarray(ang_np),
        cart2sph=cart2sph,
        max_l=int(max(l for (l, _, _, _, _) in raw_shells)),
    )
    if return_atom_index:
        return basis, np.array(atom_index, dtype=np.int64)
    return basis
