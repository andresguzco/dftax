"""Pure-JAX Gaussian Type Orbital (GTO) evaluator.

Replaces PySCF's dft.numint.eval_ao CPU callback with a pure-JAX
implementation that is fully differentiable via autodiff and
compatible with jit and vmap.

Usage::

    from dftax.energy.gto import extract_basis_data, eval_gto

    # One-time CPU setup (from PySCF mol object):
    basis = extract_basis_data(mol)

    # Pure-JAX evaluation at any point r (3,):
    ao_vals = eval_gto(basis, r)          # (nao,)

    # Gradient via autodiff (no custom_jvp needed):
    jac = jax.jacobian(eval_gto, argnums=1)(basis, r)  # (nao, 3)
"""

import numpy as np

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array


# ---------------------------------------------------------------------------
# Cartesian angular momentum components (in PySCF order)
# ---------------------------------------------------------------------------

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
}


# ---------------------------------------------------------------------------
# Normalization helpers (CPU, called only at setup)
# ---------------------------------------------------------------------------

def _angular_overlap_factor(ang: tuple[int, int, int], alpha: float) -> float:
    """Extra factor in the primitive-primitive overlap integral.

    The overlap of two primitive GTOs (same angular, exponents a_i and a_j)
    evaluated at alpha = a_i + a_j is:

        S_ij = (pi / alpha)^(3/2) * _angular_overlap_factor(ang, alpha)

    where:
        lk=0  -> 1
        lk=1  -> 1 / (2 * alpha)
        lk=2  -> 3 / (4 * alpha^2)
    """
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
    return factor


def _contracted_norm(
    exps: np.ndarray,
    c_prim: np.ndarray,
    ang: tuple[int, int, int],
) -> float:
    """Normalization constant for a contracted Cartesian GTO.

    Args:
        exps: Primitive exponents, shape (n_prim,).
        c_prim: Primitive-normalised contraction coefficients, shape (n_prim,).
        ang: Cartesian angular momentum (lx, ly, lz).

    Returns:
        Scalar N such that N * sum_k c_prim_k * GTO_k has unit norm.
    """
    S = 0.0
    for ai, ci in zip(exps, c_prim):
        for aj, cj in zip(exps, c_prim):
            alpha = ai + aj
            ovlp = (np.pi / alpha) ** 1.5 * _angular_overlap_factor(ang, alpha)
            S += ci * cj * ovlp
    return 1.0 / np.sqrt(S)


# ---------------------------------------------------------------------------
# BasisData: precomputed JAX arrays for the entire AO basis
# ---------------------------------------------------------------------------

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

    atom_coords = mol.atom_coords()  # (n_atoms, 3) in Bohr
    max_prim = max(mol.bas_nprim(i) for i in range(mol.nbas))

    all_centers: list[np.ndarray] = []
    all_exps: list[np.ndarray] = []
    all_coeffs: list[np.ndarray] = []
    all_angular: list[tuple[int, int, int]] = []

    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        atom_idx = int(mol._bas[i, pyscf_gto.mole.ATOM_OF])
        center = atom_coords[atom_idx]          # (3,)

        exps = mol.bas_exp(i)                   # (n_prim,)
        raw_c = mol.bas_ctr_coeff(i)            # (n_prim, n_ctr)

        for ctr in range(raw_c.shape[1]):
            c_raw = raw_c[:, ctr]               # (n_prim,)

            # 1. Apply primitive normalization (gto_norm normalises each
            #    primitive so that its self-overlap equals 1).
            prim_norms = np.array([pyscf_gto.gto_norm(l, e) for e in exps])
            c_prim = c_raw * prim_norms         # (n_prim,)

            # 2. For each Cartesian angular component of this shell,
            #    compute the contracted normalization and store.
            #
            #    NOTE: PySCF with cart=True applies Cartesian contracted
            #    normalization for l=0 (s) and l=1 (p), but uses only the
            #    primitive norm for l>=2 (d, f, ...).  This is because the
            #    libcint Cartesian convention normalises d+ functions in the
            #    spherical sense (one norm per shell), not per-component.
            #    For l=0,1 the two conventions coincide; for l>=2 they differ.
            for ang in _CART_COMPONENTS.get(l, []):
                if l <= 1:
                    cont_norm = _contracted_norm(exps, c_prim, ang)
                    c_final = c_prim * cont_norm
                else:
                    # l >= 2: no Cartesian contracted norm, so use c_prim directly
                    c_final = c_prim.copy()

                # Zero-pad to max_prim
                pad_e = np.zeros(max_prim, dtype=np.float64)
                pad_c = np.zeros(max_prim, dtype=np.float64)
                n = len(exps)
                pad_e[:n] = exps
                pad_c[:n] = c_final

                all_centers.append(center)
                all_exps.append(pad_e)
                all_coeffs.append(pad_c)
                all_angular.append(ang)

    # Cartesian-to-spherical transformation (only when mol uses spherical GTOs)
    c2s = None
    if not mol.cart:
        c2s_np = mol.cart2sph_coeff()  # (nao_cart, nao_sph), sparse → dense
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


# ---------------------------------------------------------------------------
# CoulombBasisData: primitives grouped by angular momentum for batched eval
# ---------------------------------------------------------------------------

class _PrimGroup(eqx.Module):
    """A group of primitives sharing the same angular momentum type."""
    centers: Float[Array, "n 3"]
    exponents: Float[Array, "n"]
    coefficients: Float[Array, "n"]
    ao_indices: Int[Array, "n"]


class CoulombBasisData(eqx.Module):
    """Basis data grouped by angular momentum for efficient Coulomb evaluation.

    Instead of zero-padded (nao, max_prim) arrays with a 35-way jnp.where
    cascade, primitives are pre-grouped by (lx, ly, lz) so each group can
    be evaluated with a single vectorized formula call, with no conditionals.

    Built once on CPU via extract_coulomb_basis_data(mol).
    """
    n_ao: int
    groups: dict[tuple[int, int, int], _PrimGroup | None]


def extract_coulomb_basis_data(mol) -> CoulombBasisData:
    """Extract basis data grouped by angular momentum for batched Coulomb eval.

    Args:
        mol: A built PySCF gto.Mole object (typically an auxiliary basis mol).

    Returns:
        CoulombBasisData with primitives grouped by (lx, ly, lz).
    """
    from pyscf import gto as pyscf_gto

    atom_coords = mol.atom_coords()

    # Collect all primitives grouped by angular momentum
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


# ---------------------------------------------------------------------------
# Safe integer power (avoids NaN Hessians from JAX's lax.pow at x=0)
# ---------------------------------------------------------------------------

def safe_int_pow(x, n):
    """x^n for small non-negative integer n, safe for autodiff at x=0.

    JAX's lax.pow uses the float power rule d/dx[x^a] = a*x^(a-1) even
    for integer exponents stored as arrays, producing NaN second derivatives
    at x=0 (known issues: JAX #14397, #17995).

    This implements x^n via repeated multiplication and jnp.where, which
    only uses * (correct derivatives to all orders) and costs ~0.02ms extra.

    Supports n in {0, 1, 2, 3, 4} (sufficient for g-orbital GTOs).
    """
    result = jnp.ones_like(x)
    result = jnp.where(n >= 1, result * x, result)
    result = jnp.where(n >= 2, result * x, result)
    result = jnp.where(n >= 3, result * x, result)
    result = jnp.where(n >= 4, result * x, result)
    return result


# ---------------------------------------------------------------------------
# Core evaluation (pure JAX)
# ---------------------------------------------------------------------------

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
    # Displacement from each AO centre: (nao, 3)
    dr = r[None, :] - basis.centers

    # Squared distance for each AO: (nao,)
    r2 = jnp.sum(dr**2, axis=-1)

    # Angular factor: prod_k (r_k - A_k)^{l_k}
    # Uses safe_int_pow to avoid NaN Hessians from JAX's lax.pow at x=0.
    angular = (
        safe_int_pow(dr[:, 0], basis.angular[:, 0])
        * safe_int_pow(dr[:, 1], basis.angular[:, 1])
        * safe_int_pow(dr[:, 2], basis.angular[:, 2])
    )  # (nao,)

    # Radial contraction: sum_k c_k exp(-alpha_k r^2)   (nao,)
    # basis.exponents and .coefficients are zero-padded; zero-coefficient
    # slots contribute 0 to the sum, zero exponent contributes c*exp(0)=c,
    # but those slots already have c=0 so no issue.
    radial = jnp.sum(
        basis.coefficients * jnp.exp(-basis.exponents * r2[:, None]),
        axis=-1,
    )

    ao_cart = angular * radial

    # Apply Cartesian→spherical transformation if needed (l>=2 shells)
    if basis.cart2sph is not None:
        return ao_cart @ basis.cart2sph
    return ao_cart


# ---------------------------------------------------------------------------
# Partial (reduced-dimensional) GTO evaluation
# ---------------------------------------------------------------------------

def _contracted_norm_partial(
    exps: np.ndarray,
    c_prim: np.ndarray,
    ang_partial: tuple[int, ...],
    ndim: int,
) -> float:
    """Normalization constant for a contracted GTO projected to d dimensions.

    Same as ``_contracted_norm`` but uses ``(pi/alpha)^(ndim/2)`` and only
    the angular components along the kept axes.
    """
    S = 0.0
    for ai, ci in zip(exps, c_prim):
        for aj, cj in zip(exps, c_prim):
            alpha = ai + aj
            ovlp = (np.pi / alpha) ** (ndim / 2.0) * _angular_overlap_factor(ang_partial, alpha)
            S += ci * cj * ovlp
    return 1.0 / np.sqrt(S) if S > 0.0 else 0.0


class PartialBasisData(eqx.Module):
    """Precomputed GTO basis data projected to a subset of spatial dimensions.

    Attributes:
        centers:      Projected atom centres, shape (n_partial, d).
        exponents:    Primitive exponents, zero-padded, shape (n_partial, max_prim).
        coefficients: Renormalized contraction coefficients, shape (n_partial, max_prim).
        angular:      Angular momenta along kept dims, shape (n_partial, d).
        dim_indices:  Which Cartesian axes are kept (static).
    """
    centers: Float[Array, "n_partial d"]
    exponents: Float[Array, "n_partial max_prim"]
    coefficients: Float[Array, "n_partial max_prim"]
    angular: Int[Array, "n_partial d"]
    dim_indices: tuple[int, ...] = eqx.field(static=True)


def extract_partial_basis_data(mol, dim_indices: tuple[int, ...]) -> PartialBasisData:
    """Extract GTO basis data projected onto a subset of Cartesian axes.

    Keeps only AOs whose angular momentum is zero along all axes NOT in
    ``dim_indices``, then projects centres and angular momenta to the kept
    axes and renormalizes for the reduced-dimensional overlap integral.

    Args:
        mol: A built PySCF gto.Mole object.
        dim_indices: Which Cartesian axes to keep, e.g. ``(0,)`` for x-only
                     or ``(0, 1)`` for x-y.

    Returns:
        PartialBasisData suitable for eval_gto_partial.
    """
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
                # Filter: skip AOs with angular momentum in dropped dimensions
                skip = False
                for k in range(3):
                    if k not in dim_set and ang[k] != 0:
                        skip = True
                        break
                if skip:
                    continue

                # Project angular momentum to kept dims
                ang_partial = tuple(ang[k] for k in dim_indices)
                center_partial = center[list(dim_indices)]

                # Renormalize for the d-dimensional overlap
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
    """Evaluate reduced-dimensional GTOs at point r.

    Args:
        basis: Partial basis data from extract_partial_basis_data.
        r: Coordinate in the kept dimensions, shape (d,).

    Returns:
        Partial AO values, shape (n_partial,).
    """
    dr = r[None, :] - basis.centers                          # (n_partial, d)
    r2 = jnp.sum(dr ** 2, axis=-1)                           # (n_partial,)

    # Angular factor: product of (r_k - A_k)^{l_k} over kept dims
    angular = jnp.ones(basis.centers.shape[0])
    for k in range(len(basis.dim_indices)):
        angular = angular * safe_int_pow(dr[:, k], basis.angular[:, k])

    # Radial contraction
    radial = jnp.sum(
        basis.coefficients * jnp.exp(-basis.exponents * r2[:, None]),
        axis=-1,
    )
    return angular * radial
