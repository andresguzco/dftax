"""Pure-JAX Coulomb potential integrals for density fitting.

Replaces PySCF's ``fakemol_for_charges`` + ``int2c2e_cart`` CPU callback
with a pure-JAX implementation that is fully differentiable via autodiff
and compatible with jit and vmap.

The core integral is the Coulomb potential of an auxiliary GTO η_P at a
field point r_g:

    V_P(r_g) = ∫ η_P(r) / |r_g − r| dr

Computed via the Obara-Saika nuclear-attraction recurrence for a single-
center GTO.  With P = A (center), XPA = 0, D_i = C_i − A_i = (r_g − A)_i:

    (a+1_i|V)^{(n)} = D_i · (a|V)^{(n+1)}
                     + a_i / (2ζ) · [(a-1_i|V)^{(n)} − (a-1_i|V)^{(n+1)}]

Base:  (0|V)^{(n)} = R_n ≡ (2π/ζ) F_n(T),  T = ζ|D|²

This gives closed-form expressions for each Cartesian angular momentum
type up to l=4 (g-type), as required for the Weigend auxiliary basis.

Usage::

    from dftax.energy.jax_df_integrals import eval_coulomb_potential
    from dftax.energy.gto import extract_basis_data

    aux_basis = extract_basis_data(auxmol)  # auxmol built with cart=True
    V = eval_coulomb_potential(aux_basis, r_g)          # (n_aux,)
    V_all = jax.vmap(lambda r: eval_coulomb_potential(aux_basis, r))(grid_pts)
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from dftax.energy.boys import boys
from dftax.energy.gto import BasisData, CoulombBasisData


# ---------------------------------------------------------------------------
# OS recurrence formulas, one per Cartesian angular momentum type
# All formulas at auxiliary index n=0.
# R[k] = (2π/ζ) * F_k(T),   i2z = 1/(2ζ),   D = (D_x, D_y, D_z).
# ---------------------------------------------------------------------------

def _V_000(D, R, i2z):
    return R[0]

def _V_100(D, R, i2z): return D[0] * R[1]
def _V_010(D, R, i2z): return D[1] * R[1]
def _V_001(D, R, i2z): return D[2] * R[1]

def _V_200(D, R, i2z): return D[0]**2 * R[2] + i2z * (R[0] - R[1])
def _V_110(D, R, i2z): return D[0] * D[1] * R[2]
def _V_101(D, R, i2z): return D[0] * D[2] * R[2]
def _V_020(D, R, i2z): return D[1]**2 * R[2] + i2z * (R[0] - R[1])
def _V_011(D, R, i2z): return D[1] * D[2] * R[2]
def _V_002(D, R, i2z): return D[2]**2 * R[2] + i2z * (R[0] - R[1])

def _V_300(D, R, i2z): return D[0]**3 * R[3] + 3*i2z * D[0] * (R[1] - R[2])
def _V_210(D, R, i2z): return D[0]**2 * D[1] * R[3] + i2z * D[1] * (R[1] - R[2])
def _V_201(D, R, i2z): return D[0]**2 * D[2] * R[3] + i2z * D[2] * (R[1] - R[2])
def _V_120(D, R, i2z): return D[0] * D[1]**2 * R[3] + i2z * D[0] * (R[1] - R[2])
def _V_111(D, R, i2z): return D[0] * D[1] * D[2] * R[3]
def _V_102(D, R, i2z): return D[0] * D[2]**2 * R[3] + i2z * D[0] * (R[1] - R[2])
def _V_030(D, R, i2z): return D[1]**3 * R[3] + 3*i2z * D[1] * (R[1] - R[2])
def _V_021(D, R, i2z): return D[1]**2 * D[2] * R[3] + i2z * D[2] * (R[1] - R[2])
def _V_012(D, R, i2z): return D[1] * D[2]**2 * R[3] + i2z * D[1] * (R[1] - R[2])
def _V_003(D, R, i2z): return D[2]**3 * R[3] + 3*i2z * D[2] * (R[1] - R[2])

def _V_400(D, R, i2z):
    return D[0]**4 * R[4] + 6*i2z * D[0]**2 * (R[2]-R[3]) + 3*i2z**2 * (R[0]-2*R[1]+R[2])
def _V_310(D, R, i2z):
    return D[0]**3 * D[1] * R[4] + 3*i2z * D[0] * D[1] * (R[2]-R[3])
def _V_301(D, R, i2z):
    return D[0]**3 * D[2] * R[4] + 3*i2z * D[0] * D[2] * (R[2]-R[3])
def _V_220(D, R, i2z):
    return D[0]**2 * D[1]**2 * R[4] + i2z * (D[0]**2 + D[1]**2) * (R[2]-R[3]) + i2z**2 * (R[0]-2*R[1]+R[2])
def _V_211(D, R, i2z):
    return D[0]**2 * D[1] * D[2] * R[4] + i2z * D[1] * D[2] * (R[2]-R[3])
def _V_202(D, R, i2z):
    return D[0]**2 * D[2]**2 * R[4] + i2z * (D[0]**2 + D[2]**2) * (R[2]-R[3]) + i2z**2 * (R[0]-2*R[1]+R[2])
def _V_130(D, R, i2z):
    return D[0] * D[1]**3 * R[4] + 3*i2z * D[0] * D[1] * (R[2]-R[3])
def _V_121(D, R, i2z):
    return D[0] * D[1]**2 * D[2] * R[4] + i2z * D[0] * D[2] * (R[2]-R[3])
def _V_112(D, R, i2z):
    return D[0] * D[1] * D[2]**2 * R[4] + i2z * D[0] * D[1] * (R[2]-R[3])
def _V_103(D, R, i2z):
    return D[0] * D[2]**3 * R[4] + 3*i2z * D[0] * D[2] * (R[2]-R[3])
def _V_040(D, R, i2z):
    return D[1]**4 * R[4] + 6*i2z * D[1]**2 * (R[2]-R[3]) + 3*i2z**2 * (R[0]-2*R[1]+R[2])
def _V_031(D, R, i2z):
    return D[1]**3 * D[2] * R[4] + 3*i2z * D[1] * D[2] * (R[2]-R[3])
def _V_022(D, R, i2z):
    return D[1]**2 * D[2]**2 * R[4] + i2z * (D[1]**2 + D[2]**2) * (R[2]-R[3]) + i2z**2 * (R[0]-2*R[1]+R[2])
def _V_013(D, R, i2z):
    return D[1] * D[2]**3 * R[4] + 3*i2z * D[1] * D[2] * (R[2]-R[3])
def _V_004(D, R, i2z):
    return D[2]**4 * R[4] + 6*i2z * D[2]**2 * (R[2]-R[3]) + 3*i2z**2 * (R[0]-2*R[1]+R[2])


# Ordered list of (lx, ly, lz, formula) for all Cartesian types up to l=4.
# Order must match PySCF's Cartesian convention (decreasing lx, then ly, then lz).
_COULOMB_FORMULAS = [
    # l=0
    ((0, 0, 0), _V_000),
    # l=1
    ((1, 0, 0), _V_100), ((0, 1, 0), _V_010), ((0, 0, 1), _V_001),
    # l=2
    ((2, 0, 0), _V_200), ((1, 1, 0), _V_110), ((1, 0, 1), _V_101),
    ((0, 2, 0), _V_020), ((0, 1, 1), _V_011), ((0, 0, 2), _V_002),
    # l=3
    ((3, 0, 0), _V_300), ((2, 1, 0), _V_210), ((2, 0, 1), _V_201),
    ((1, 2, 0), _V_120), ((1, 1, 1), _V_111), ((1, 0, 2), _V_102),
    ((0, 3, 0), _V_030), ((0, 2, 1), _V_021), ((0, 1, 2), _V_012), ((0, 0, 3), _V_003),
    # l=4
    ((4, 0, 0), _V_400), ((3, 1, 0), _V_310), ((3, 0, 1), _V_301),
    ((2, 2, 0), _V_220), ((2, 1, 1), _V_211), ((2, 0, 2), _V_202),
    ((1, 3, 0), _V_130), ((1, 2, 1), _V_121), ((1, 1, 2), _V_112), ((1, 0, 3), _V_103),
    ((0, 4, 0), _V_040), ((0, 3, 1), _V_031), ((0, 2, 2), _V_022),
    ((0, 1, 3), _V_013), ((0, 0, 4), _V_004),
]


# ---------------------------------------------------------------------------
# Per-primitive Coulomb potential (selects formula via jnp.where cascade)
# ---------------------------------------------------------------------------

def _eval_coulomb_primitive(
    exponent: Float[Array, ""],
    coefficient: Float[Array, ""],
    ang: Float[Array, "3"],   # (lx, ly, lz) as float
    D: Float[Array, "3"],     # r_g - A
) -> Float[Array, ""]:
    """Coulomb potential from one primitive GTO at displacement D = r_g − A.

    Returns coefficient * (2π/ζ) * OS_recurrence_result.
    Zero-padded slots (exponent=0, coefficient=0) return 0.
    """
    # Guard against division by zero from zero-padded slots.
    safe_zeta = jnp.where(exponent == 0.0, 1.0, exponent)
    T = safe_zeta * jnp.sum(D ** 2)
    i2z = 0.5 / safe_zeta
    prefactor = coefficient * (2.0 * jnp.pi / safe_zeta)

    # Precompute R_n = F_n(T) for n = 0..4 (max needed for l=4)
    R = [boys(n, T) for n in range(5)]

    # Evaluate the formula for every known (lx, ly, lz) and select based on ang.
    # Build a jnp.where chain: first matching (lx==A) & (ly==B) & (lz==C) wins.
    lx, ly, lz = ang[0], ang[1], ang[2]

    result = jnp.zeros_like(T)
    for (ax, ay, az), formula in reversed(_COULOMB_FORMULAS):
        cond = (lx == float(ax)) & (ly == float(ay)) & (lz == float(az))
        result = jnp.where(cond, formula(D, R, i2z), result)

    return prefactor * result


# ---------------------------------------------------------------------------
# Contracted Coulomb potential at a field point
# ---------------------------------------------------------------------------

def eval_coulomb_potential(
    basis: BasisData,
    r_g: Float[Array, "3"],
) -> Float[Array, "nao"]:
    """Coulomb potential of each contracted auxiliary GTO at field point r_g.

    Pure JAX, no PySCF calls, fully differentiable, jit/vmap compatible.

    Evaluates V_P(r_g) = ∫ η_P(r) / |r_g − r| dr for each AO P using the
    Obara-Saika nuclear-attraction recurrence (closed-form for s through g).

    Args:
        basis: BasisData for the auxiliary basis (from extract_basis_data,
               with the auxmol built using cart=True).
        r_g:   Field point in Bohr, shape (3,).

    Returns:
        V of shape (nao,), matching PySCF's ``int2c2e_cart`` with
        ``fakemol_for_charges([r_g])``.
    """
    # Displacement from each AO centre to the field point: (nao, 3)
    D = r_g[None, :] - basis.centers       # (nao, 3)
    ang = basis.angular.astype(r_g.dtype)  # (nao, 3)

    def _ao_coulomb(D_i, exps_i, coeffs_i, ang_i):
        """Sum primitive Coulomb contributions for one contracted AO."""
        prim_vals = jax.vmap(
            lambda zeta, c: _eval_coulomb_primitive(zeta, c, ang_i, D_i)
        )(exps_i, coeffs_i)
        return jnp.sum(prim_vals)

    return jax.vmap(_ao_coulomb)(D, basis.exponents, basis.coefficients, ang)


# ---------------------------------------------------------------------------
# Batched Coulomb potential (grouped by angular momentum, no conditionals)
# ---------------------------------------------------------------------------

# Map (lx, ly, lz) → formula function, for quick lookup
_FORMULA_MAP = {ang: fn for ang, fn in _COULOMB_FORMULAS}


def eval_coulomb_potential_batched(
    basis: CoulombBasisData,
    r_g: Float[Array, "3"],
) -> Float[Array, "nao"]:
    """Batched Coulomb potential: primitives grouped by angular momentum.

    Same result as eval_coulomb_potential but much faster: each angular
    momentum group is evaluated with a single vectorized formula call
    (pure array ops), eliminating the 35-way jnp.where cascade and
    computing only the needed Boys function orders per group.

    Args:
        basis: CoulombBasisData from extract_coulomb_basis_data (auxmol cart=True).
        r_g:   Field point in Bohr, shape (3,).

    Returns:
        V of shape (n_ao,).
    """
    result = jnp.zeros(basis.n_ao)

    for ang, formula in _COULOMB_FORMULAS:
        group = basis.groups[ang]
        if group is None:
            continue

        lx, ly, lz = ang
        max_boys_order = lx + ly + lz  # highest Boys order needed

        # Displacement from each primitive's centre to the field point
        D = r_g[None, :] - group.centers          # (n_prims, 3)
        zeta = group.exponents                     # (n_prims,)
        T = zeta * jnp.sum(D ** 2, axis=-1)        # (n_prims,)
        i2z = 0.5 / zeta                           # (n_prims,)
        prefactor = group.coefficients * (2.0 * jnp.pi / zeta)  # (n_prims,)

        # Boys function values: only the orders this group actually needs
        R = [boys(n, T) for n in range(max_boys_order + 1)]

        # D components as (3, n_prims) for the formula functions
        Dt = (D[:, 0], D[:, 1], D[:, 2])

        # Apply formula (vectorised over all prims in this group)
        vals = prefactor * formula(Dt, R, i2z)     # (n_prims,)

        # Scatter-add back to the AO dimension
        result = result.at[group.ao_indices].add(vals)

    return result
