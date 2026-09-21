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
    # l = 5, 6 (h, i): same lexicographic order (lx descending, then ly).
    # Needed only for *auxiliary* DF bases (def2-universal-jkfit carries h/i
    # for the heavier elements); the primary AO basis stays capped at g.
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
        # (2k-1)!! / (2·alpha)^k per axis.
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

def shell_records(angular, exponents) -> tuple:
    """``(l, row0, ncomp, nprim)`` per shell, from static basis metadata.

    A shell starts wherever the canonical Cartesian component sequence
    restarts at ``(l, 0, 0)``, so two same-l shells on one atom split by row
    order alone. ``nprim`` is the shell's true contraction length, not the
    padded row width.
    """
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
    """Hashable digest of the metadata that fixes a basis's structure.

    Angular momenta, exponents, coefficients and whether a spherical transform
    is attached. Centers are excluded, so results keyed on this fingerprint
    stay valid as the geometry moves. Eager only: the arrays must be concrete.
    """
    return (
        np.ascontiguousarray(np.asarray(basis.angular)).tobytes(),
        np.ascontiguousarray(np.asarray(basis.exponents)).tobytes(),
        np.ascontiguousarray(np.asarray(basis.coefficients)).tobytes(),
        None if basis.cart2sph is None else tuple(basis.cart2sph.shape),
        int(basis.max_l),
    )


def _axis_powers(x, l: int):
    """``(n, l+1)`` with column ``i`` equal to ``x**i``, by multiplication.

    Products rather than ``lax.pow``, whose float power rule gives NaN second
    derivatives at ``x = 0`` (the same reason :func:`safe_int_pow` exists).
    """
    cols = [jnp.ones_like(x)]
    for _ in range(l):
        cols.append(cols[-1] * x)
    return jnp.stack(cols, axis=-1)


def _eval_gto_flat_grad(basis: BasisData, r: Float[Array, "3"]):
    """Per-AO values *and* analytic gradient, in the flat vectorized layout.

    The middle option between the two above, and the one that isolates which
    half of the shell-blocked rewrite actually pays on a GPU. It keeps the
    per-AO layout (so the redundant exponentials stay, but so does the single
    wide elementwise kernel that makes them nearly free when the evaluation is
    bandwidth-bound) and drops only the ``jacfwd``, which was three extra
    tangent passes over the whole basis.
    """
    dr = r[None, :] - basis.centers
    r2 = jnp.sum(dr ** 2, axis=-1)
    E = jnp.exp(-basis.exponents * r2[:, None])
    R = jnp.sum(basis.coefficients * E, axis=-1)
    Rp = jnp.sum(basis.coefficients * basis.exponents * E, axis=-1)

    # One power ladder per axis, indexed twice, rather than six independent
    # safe_int_pow chains. safe_int_pow is a six-deep `where` over the whole
    # (nao,) vector, and the gradient needs both x^l and x^{l-1}; spelling
    # that as two chains doubles the widest intermediate in the kernel, which
    # is the wrong thing to double when the evaluation is bandwidth-bound.
    L = int(basis.max_l)
    lk = [basis.angular[:, k] for k in range(3)]
    pw = [_axis_powers(dr[:, k], L) for k in range(3)]        # (nao, L+1)
    P = [jnp.take_along_axis(pw[k], lk[k][:, None], axis=1)[:, 0]
         for k in range(3)]
    A = P[0] * P[1] * P[2]
    ao_cart = A * R

    d = []
    for k in range(3):
        lower = jnp.take_along_axis(
            pw[k], jnp.maximum(lk[k] - 1, 0)[:, None], axis=1)[:, 0]
        # the lk factor is zero exactly where the clamped index would be
        # wrong, so the clamp never contributes
        dA = lk[k] * lower * P[(k + 1) % 3] * P[(k + 2) % 3]
        # second term is -2 dr_k * A * R', with the *angular* factor A, not
        # the AO value A*R
        d.append(dA * R - 2.0 * dr[:, k] * A * Rp)
    dao_cart = jnp.stack(d, axis=-1)

    if basis.cart2sph is not None:
        return (ao_cart @ basis.cart2sph,
                jnp.einsum("cx,cs->sx", dao_cart, basis.cart2sph))
    return ao_cart, dao_cart


def safe_int_pow(x, n):
    """x^n for small non-negative integer n, safe for autodiff at x=0.

    JAX's lax.pow uses the float power rule d/dx[x^a] = a*x^(a-1) even
    for integer exponents stored as arrays, producing NaN second derivatives
    at x=0 (known issues: JAX #14397, #17995).

    This implements x^n via repeated multiplication and jnp.where, which
    only uses * (correct derivatives to all orders) and costs ~0.02ms extra.

    Supports n in {0, .., 6}, the engine's i-shell ceiling. This cap MUST
    match the integral engines' orbital ceiling: an exponent beyond the
    unroll silently returns x^6, which poisons only the grid AO values (the
    integral recursions are separate), i.e. the XC term at a density with
    weight on the affected shells -- an error every matrix-level oracle
    misses (this is how the 5Z enablement initially produced a spurious
    85 mHa SCF minimum from wrong h-orbital grid values).
    """
    result = jnp.ones_like(x)
    result = jnp.where(n >= 1, result * x, result)
    result = jnp.where(n >= 2, result * x, result)
    result = jnp.where(n >= 3, result * x, result)
    result = jnp.where(n >= 4, result * x, result)
    result = jnp.where(n >= 5, result * x, result)
    result = jnp.where(n >= 6, result * x, result)
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
    return _eval_gto_flat(basis, r)


def eval_gto_and_grad(basis: BasisData, r: Float[Array, "3"]):
    """AO values and their spatial gradients ``(nao,), (nao, 3)`` at ``r``.

    One pass for both, with the gradient in closed form. Prefer this to
    ``eval_gto`` beside ``jacfwd(eval_gto)``, which evaluates the basis five
    times over.

    The gradient is written in the per-AO layout rather than the shell-blocked
    one, which is the opposite of what the FLOP count suggests: shell-blocking
    does 4-6x fewer exponentials and still measures slower, because the
    evaluation is bandwidth-bound. See ``scripts/perf/ao_bench.py``.
    """
    return _eval_gto_flat_grad(basis, r)


def _eval_gto_flat(basis: BasisData, r: Float[Array, "3"]) -> Float[Array, "nao"]:
    """The original per-AO evaluation; kept for A/B validation."""
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
