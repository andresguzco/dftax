import numpy as np

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array


_CART_COMPONENTS = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
    3: [
        (3, 0, 0), (2, 1, 0), (2, 0, 1),
        (1, 2, 0), (1, 1, 1), (1, 0, 2),
        (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3),
    ],
    4: [
        (4, 0, 0), (3, 1, 0), (3, 0, 1),
        (2, 2, 0), (2, 1, 1), (2, 0, 2),
        (1, 3, 0), (1, 2, 1), (1, 1, 2), (1, 0, 3),
        (0, 4, 0), (0, 3, 1), (0, 2, 2), (0, 1, 3), (0, 0, 4),
    ],
    5: [
        (5, 0, 0), (4, 1, 0), (4, 0, 1),
        (3, 2, 0), (3, 1, 1), (3, 0, 2),
        (2, 3, 0), (2, 2, 1), (2, 1, 2), (2, 0, 3),
        (1, 4, 0), (1, 3, 1), (1, 2, 2), (1, 1, 3), (1, 0, 4),
        (0, 5, 0), (0, 4, 1), (0, 3, 2), (0, 2, 3), (0, 1, 4), (0, 0, 5),
    ],
    6: [
        (6, 0, 0), (5, 1, 0), (5, 0, 1),
        (4, 2, 0), (4, 1, 1), (4, 0, 2),
        (3, 3, 0), (3, 2, 1), (3, 1, 2), (3, 0, 3),
        (2, 4, 0), (2, 3, 1), (2, 2, 2), (2, 1, 3), (2, 0, 4),
        (1, 5, 0), (1, 4, 1), (1, 3, 2), (1, 2, 3), (1, 1, 4), (1, 0, 5),
        (0, 6, 0), (0, 5, 1), (0, 4, 2), (0, 3, 3), (0, 2, 4), (0, 1, 5),
        (0, 0, 6),
    ],
}


def _angular_overlap_factor(ang: tuple[int, int, int], alpha: float) -> float:
    factor = 1.0
    for lk in ang:
        if lk == 1:
            factor /= 2.0 * alpha
        elif lk == 2:
            factor *= 3.0 / (4.0 * alpha**2)
        elif lk == 3:
            factor *= 15.0 / (8.0 * alpha**3)
        elif lk == 4:
            factor *= 105.0 / (16.0 * alpha**4)
        elif lk == 5:
            factor *= 945.0 / (32.0 * alpha**5)
        elif lk == 6:
            factor *= 10395.0 / (64.0 * alpha**6)
    return factor


def _contracted_norm(
    exps: np.ndarray,
    c_prim: np.ndarray,
    ang: tuple[int, int, int],
) -> float:
    S = 0.0
    for ai, ci in zip(exps, c_prim):
        for aj, cj in zip(exps, c_prim):
            alpha = ai + aj
            ovlp = (np.pi / alpha) ** 1.5 * _angular_overlap_factor(ang, alpha)
            S += ci * cj * ovlp
    return 1.0 / np.sqrt(S)


class BasisData(eqx.Module):
    """Precomputed GTO basis data extracted from a PySCF mol object.

    All arrays are JAX arrays; construction is done once on CPU via
    extract_basis_data(mol).

    Attributes:
        centers:      Atom centre for each AO, shape (nao_cart, 3).
        exponents:    Primitive exponents, zero-padded, shape (nao_cart, max_prim).
        coefficients: Final contraction coefficients (primitive-normed *
                      contracted-normed), zero-padded, shape (nao_cart, max_prim).
        angular:      Cartesian angular momenta (lx, ly, lz), shape (nao_cart, 3).
        cart2sph:     Cartesian-to-spherical transformation, shape (nao_cart, nao_sph),
                      or None when the molecule uses Cartesian GTOs.
        max_l:        Maximum total angular momentum in the basis (static int).
                      Lets the integral builders size their recursion to the
                      molecule instead of the global g-type cap.
    """

    centers: Float[Array, "nao_cart 3"]
    exponents: Float[Array, "nao_cart max_prim"]
    coefficients: Float[Array, "nao_cart max_prim"]
    angular: Int[Array, "nao_cart 3"]
    cart2sph: Float[Array, "nao_cart nao_sph"] | None
    max_l: int = eqx.field(static=True, default=4)


def extract_basis_data(mol: "object") -> BasisData:
    """Extract GTO basis data from a PySCF Mole object.

    This is a CPU-side function called once at setup.  It returns a
    BasisData whose arrays live on the JAX default device.

    Args:
        mol: A built PySCF gto.Mole object.

    Returns:
        BasisData suitable for eval_gto.
    """
    from pyscf import gto as pyscf_gto

    atom_coords = mol.atom_coords()
    max_prim = max(mol.bas_nprim(i) for i in range(mol.nbas))

    all_centers: list[np.ndarray] = []
    all_exps: list[np.ndarray] = []
    all_coeffs: list[np.ndarray] = []
    all_angular: list[tuple[int, int, int]] = []

    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        atom_idx = int(mol._bas[i, pyscf_gto.mole.ATOM_OF])
        center = atom_coords[atom_idx]

        exps = mol.bas_exp(i)
        raw_c = mol.bas_ctr_coeff(i)

        for ctr in range(raw_c.shape[1]):
            c_raw = raw_c[:, ctr]

            prim_norms = np.array([pyscf_gto.gto_norm(l, e) for e in exps])
            c_prim = c_raw * prim_norms

            for ang in _CART_COMPONENTS.get(l, []):
                if l <= 1:
                    cont_norm = _contracted_norm(exps, c_prim, ang)
                    c_final = c_prim * cont_norm
                else:
                    c_final = c_prim.copy()

                pad_e = np.zeros(max_prim, dtype=np.float64)
                pad_c = np.zeros(max_prim, dtype=np.float64)
                n = len(exps)
                pad_e[:n] = exps
                pad_c[:n] = c_final

                all_centers.append(center)
                all_exps.append(pad_e)
                all_coeffs.append(pad_c)
                all_angular.append(ang)

    c2s = None
    if not mol.cart:
        c2s_np = mol.cart2sph_coeff()
        if hasattr(c2s_np, 'toarray'):
            c2s_np = c2s_np.toarray()
        c2s = jnp.array(np.asarray(c2s_np, dtype=np.float64))

    max_l = int(max(sum(ang) for ang in all_angular))

    return BasisData(
        centers=jnp.array(np.array(all_centers, dtype=np.float64)),
        exponents=jnp.array(np.array(all_exps, dtype=np.float64)),
        coefficients=jnp.array(np.array(all_coeffs, dtype=np.float64)),
        angular=jnp.array(np.array(all_angular, dtype=np.int32)),
        cart2sph=c2s,
        max_l=max_l,
    )


class _PrimGroup(eqx.Module):
    centers: Float[Array, "n 3"]
    exponents: Float[Array, "n"]
    coefficients: Float[Array, "n"]
    ao_indices: Int[Array, "n"]


class CoulombBasisData(eqx.Module):
    n_ao: int
    groups: dict[tuple[int, int, int], _PrimGroup | None]


def extract_coulomb_basis_data(mol) -> CoulombBasisData:
    from pyscf import gto as pyscf_gto

    atom_coords = mol.atom_coords()

    groups_raw: dict[tuple[int, int, int], list] = {}
    for ang_key in _CART_COMPONENTS.values():
        for ang in ang_key:
            groups_raw[ang] = {"centers": [], "exponents": [], "coefficients": [], "ao_indices": []}

    ao_idx = 0
    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        atom_idx = int(mol._bas[i, pyscf_gto.mole.ATOM_OF])
        center = atom_coords[atom_idx]
        exps = mol.bas_exp(i)
        raw_c = mol.bas_ctr_coeff(i)

        for ctr in range(raw_c.shape[1]):
            c_raw = raw_c[:, ctr]
            prim_norms = np.array([pyscf_gto.gto_norm(l, e) for e in exps])
            c_prim = c_raw * prim_norms

            for ang in _CART_COMPONENTS.get(l, []):
                if l <= 1:
                    cont_norm = _contracted_norm(exps, c_prim, ang)
                    c_final = c_prim * cont_norm
                else:
                    c_final = c_prim.copy()

                g = groups_raw[ang]
                for k in range(len(exps)):
                    g["centers"].append(center)
                    g["exponents"].append(exps[k])
                    g["coefficients"].append(c_final[k])
                    g["ao_indices"].append(ao_idx)

                ao_idx += 1

    n_ao = ao_idx
    groups = {}
    for ang, g in groups_raw.items():
        if len(g["centers"]) == 0:
            groups[ang] = None
        else:
            groups[ang] = _PrimGroup(
                centers=jnp.array(np.array(g["centers"], dtype=np.float64)),
                exponents=jnp.array(np.array(g["exponents"], dtype=np.float64)),
                coefficients=jnp.array(np.array(g["coefficients"], dtype=np.float64)),
                ao_indices=jnp.array(np.array(g["ao_indices"], dtype=np.int32)),
            )

    return CoulombBasisData(n_ao=n_ao, groups=groups)


def shell_records(angular, exponents) -> tuple:
    ang = np.asarray(angular)
    ex = np.asarray(exponents)
    ltot = ang.sum(1)
    n = ang.shape[0]
    starts = [i for i in range(n)
              if ang[i, 0] == ltot[i] and ang[i, 1] == 0 and ang[i, 2] == 0]
    if not starts or starts[0] != 0:
        raise ValueError("basis rows do not start shells at (l,0,0); "
                         "shell_records assumes gto.py row order")
    bounds = starts + [n]
    return tuple(
        (int(ltot[s]), int(s), int(e - s), max(1, int((ex[s] != 0).sum())))
        for s, e in zip(bounds[:-1], bounds[1:])
    )


def static_fingerprint(basis) -> tuple:
    return (
        np.ascontiguousarray(np.asarray(basis.angular)).tobytes(),
        np.ascontiguousarray(np.asarray(basis.exponents)).tobytes(),
        np.ascontiguousarray(np.asarray(basis.coefficients)).tobytes(),
        None if basis.cart2sph is None else tuple(basis.cart2sph.shape),
        int(basis.max_l),
    )


def _axis_powers(x, l: int):
    cols = [jnp.ones_like(x)]
    for _ in range(l):
        cols.append(cols[-1] * x)
    return jnp.stack(cols, axis=-1)


def _eval_gto_flat_grad(basis: BasisData, r: Float[Array, "3"]):
    dr = r[None, :] - basis.centers
    r2 = jnp.sum(dr ** 2, axis=-1)
    E = jnp.exp(-basis.exponents * r2[:, None])
    R = jnp.sum(basis.coefficients * E, axis=-1)
    Rp = jnp.sum(basis.coefficients * basis.exponents * E, axis=-1)

    L = int(basis.max_l)
    lk = [basis.angular[:, k] for k in range(3)]
    pw = [_axis_powers(dr[:, k], L) for k in range(3)]
    P = [jnp.take_along_axis(pw[k], lk[k][:, None], axis=1)[:, 0]
         for k in range(3)]
    A = P[0] * P[1] * P[2]
    ao_cart = A * R

    d = []
    for k in range(3):
        lower = jnp.take_along_axis(
            pw[k], jnp.maximum(lk[k] - 1, 0)[:, None], axis=1)[:, 0]
        dA = lk[k] * lower * P[(k + 1) % 3] * P[(k + 2) % 3]
        d.append(dA * R - 2.0 * dr[:, k] * A * Rp)
    dao_cart = jnp.stack(d, axis=-1)

    if basis.cart2sph is not None:
        return (ao_cart @ basis.cart2sph,
                jnp.einsum("cx,cs->sx", dao_cart, basis.cart2sph))
    return ao_cart, dao_cart


def safe_int_pow(x, n):
    result = jnp.ones_like(x)
    result = jnp.where(n >= 1, result * x, result)
    result = jnp.where(n >= 2, result * x, result)
    result = jnp.where(n >= 3, result * x, result)
    result = jnp.where(n >= 4, result * x, result)
    result = jnp.where(n >= 5, result * x, result)
    result = jnp.where(n >= 6, result * x, result)
    return result


def eval_gto(basis: BasisData, r: Float[Array, "3"]) -> Float[Array, "nao"]:
    """Evaluate contracted GTO basis functions at point r.

    Pure JAX, no PySCF calls, no pure_callback.  Fully differentiable via
    JAX autodiff and compatible with jit and vmap.

    Args:
        basis: Precomputed basis data from extract_basis_data(mol).
        r:     3D coordinate in Bohr, shape (3,).

    Returns:
        AO values, shape (nao,), matching dft.numint.eval_ao(mol, r[None])[0].
    """
    return _eval_gto_flat(basis, r)


def eval_gto_and_grad(basis: BasisData, r: Float[Array, "3"]):
    return _eval_gto_flat_grad(basis, r)


def _eval_gto_flat(basis: BasisData, r: Float[Array, "3"]) -> Float[Array, "nao"]:
    dr = r[None, :] - basis.centers

    r2 = jnp.sum(dr**2, axis=-1)

    angular = (
        safe_int_pow(dr[:, 0], basis.angular[:, 0])
        * safe_int_pow(dr[:, 1], basis.angular[:, 1])
        * safe_int_pow(dr[:, 2], basis.angular[:, 2])
    )

    radial = jnp.sum(
        basis.coefficients * jnp.exp(-basis.exponents * r2[:, None]),
        axis=-1,
    )

    ao_cart = angular * radial

    if basis.cart2sph is not None:
        return ao_cart @ basis.cart2sph
    return ao_cart


def _contracted_norm_partial(
    exps: np.ndarray,
    c_prim: np.ndarray,
    ang_partial: tuple[int, ...],
    ndim: int,
) -> float:
    S = 0.0
    for ai, ci in zip(exps, c_prim):
        for aj, cj in zip(exps, c_prim):
            alpha = ai + aj
            ovlp = (np.pi / alpha) ** (ndim / 2.0) * _angular_overlap_factor(ang_partial, alpha)
            S += ci * cj * ovlp
    return 1.0 / np.sqrt(S) if S > 0.0 else 0.0


class PartialBasisData(eqx.Module):
    centers: Float[Array, "n_partial d"]
    exponents: Float[Array, "n_partial max_prim"]
    coefficients: Float[Array, "n_partial max_prim"]
    angular: Int[Array, "n_partial d"]
    dim_indices: tuple[int, ...] = eqx.field(static=True)


def extract_partial_basis_data(mol, dim_indices: tuple[int, ...]) -> PartialBasisData:
    from pyscf import gto as pyscf_gto

    dim_set = set(dim_indices)
    ndim = len(dim_indices)
    atom_coords = mol.atom_coords()
    max_prim = max(mol.bas_nprim(i) for i in range(mol.nbas))

    all_centers = []
    all_exps = []
    all_coeffs = []
    all_angular = []

    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        atom_idx = int(mol._bas[i, pyscf_gto.mole.ATOM_OF])
        center = atom_coords[atom_idx]
        exps = mol.bas_exp(i)
        raw_c = mol.bas_ctr_coeff(i)

        for ctr in range(raw_c.shape[1]):
            c_raw = raw_c[:, ctr]
            prim_norms = np.array([pyscf_gto.gto_norm(l, e) for e in exps])
            c_prim = c_raw * prim_norms

            for ang in _CART_COMPONENTS.get(l, []):
                skip = False
                for k in range(3):
                    if k not in dim_set and ang[k] != 0:
                        skip = True
                        break
                if skip:
                    continue

                ang_partial = tuple(ang[k] for k in dim_indices)
                center_partial = center[list(dim_indices)]

                cont_norm = _contracted_norm_partial(exps, c_prim, ang_partial, ndim)
                c_final = c_prim * cont_norm

                pad_e = np.zeros(max_prim, dtype=np.float64)
                pad_c = np.zeros(max_prim, dtype=np.float64)
                n = len(exps)
                pad_e[:n] = exps
                pad_c[:n] = c_final

                all_centers.append(center_partial)
                all_exps.append(pad_e)
                all_coeffs.append(pad_c)
                all_angular.append(ang_partial)

    return PartialBasisData(
        centers=jnp.array(np.array(all_centers, dtype=np.float64)),
        exponents=jnp.array(np.array(all_exps, dtype=np.float64)),
        coefficients=jnp.array(np.array(all_coeffs, dtype=np.float64)),
        angular=jnp.array(np.array(all_angular, dtype=np.int32)),
        dim_indices=tuple(int(d) for d in dim_indices),
    )


def eval_gto_partial(
    basis: PartialBasisData, r: Float[Array, "d"]
) -> Float[Array, "n_partial"]:
    dr = r[None, :] - basis.centers
    r2 = jnp.sum(dr ** 2, axis=-1)

    angular = jnp.ones(basis.centers.shape[0])
    for k in range(len(basis.dim_indices)):
        angular = angular * safe_int_pow(dr[:, k], basis.angular[:, k])

    radial = jnp.sum(
        basis.coefficients * jnp.exp(-basis.exponents * r2[:, None]),
        axis=-1,
    )
    return angular * radial
