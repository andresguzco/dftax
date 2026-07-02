"""Differentiable restricted Kohn-Sham total energy as a function of P.

The :class:`RKS` module precomputes the one-electron Hamiltonian, the
two-electron integrals, the nuclear repulsion, and the AO values/gradients on a
quadrature grid, then exposes ``electronic(P)`` / ``total(P)``, a single,
``jit``/``grad``-friendly energy functional of the closed-shell density matrix
``P``. The KS Fock matrix is obtained downstream as ``sym(∂E/∂P)`` (see
:mod:`dftax.ks.scf`), so no exchange-correlation potential matrix is hand-coded.

Two Coulomb/exchange backends:

- **exact** (default): the full 4-center ERI tensor (``eri4c``), tight against
  PySCF, but O(N⁴) memory, so only for small systems.
- **density fitting** (pass ``auxbasis=...``): RI-J / RI-K via 3- and 2-center
  integrals, O(N³) memory, for larger systems. The fit is the robust Dunlap
  (Coulomb-metric) form; the RI error vs the exact path is sub-mHa with a
  standard JK-fitting auxiliary basis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Scalar

from dftax.energy.gto import BasisData, extract_basis_data, eval_gto
from dftax.energy.grid import xc_energy
from dftax.energy.xc import XCFunctional
from dftax.utils.vmap import vmap as _chunked_vmap
from dftax.integrals import (
    overlap_matrix,
    kinetic_matrix,
    nuclear_attraction_matrix,
    nuclear_repulsion,
    eri3c_matrix,
    eri2c_matrix,
)
from dftax.integrals.eri4c import (
    eri4c_matrix,
    screened_quartets,
    coulomb_j_4c,
    exchange_k_4c,
    significant_pairs,
)
from dftax.integrals.eri3c import _contracted_eri3c, _eri3c_sizes


def _screen_eri(basis, aux_basis, eri_screen):
    """Pre-screened exact-ERI quartet list + orbit map, or ``(None, None)``.

    Cauchy-Schwarz screening applies only to the exact 4-center path (no
    auxiliary basis) and depends on concrete integral values, so it is resolved
    here in the eager constructors (the basis is materialised) rather than inside
    the jitted ``_build_integrals``.
    """
    if eri_screen is None or aux_basis is not None:
        return None, None
    quartets, qof = screened_quartets(basis, float(eri_screen))
    return jnp.asarray(quartets), jnp.asarray(qof)


def ao_on_grid(
    basis: BasisData,
    coords: Float[Array, "ng 3"],
) -> tuple[Float[Array, "ng nao"], Float[Array, "ng nao 3"]]:
    """Atomic-orbital values and spatial gradients at grid points (pure JAX)."""
    ao = jax.vmap(lambda r: eval_gto(basis, r))(coords)
    dao = jax.vmap(lambda r: jax.jacfwd(eval_gto, argnums=1)(basis, r))(coords)
    return ao, dao


@eqx.filter_jit
def _build_integrals(
    basis, coords, charges, grid_coords, aux_basis, materialize_ao, materialize_int3c,
    eri_quartets=None, eri_qof=None, stream_exact=False,
):
    """Build all integral arrays in one jitted pass.

    Jitting fuses the builders (eager mode dispatches each op unfused, e.g.
    eri4c is ~2x slower eager than jitted). ``aux_basis is None`` selects the
    exact 4-center ERI; otherwise RI-J/RI-K density-fitting tensors. When
    ``materialize_ao`` is False the AO grid values/gradients are not precomputed
    (the XC grid is streamed instead; see ``_streamed_e_xc``). ``eri_quartets``/
    ``eri_qof`` optionally supply a pre-screened (Cauchy-Schwarz) quartet list +
    orbit map for the exact ERI. Composes with grad (used by forces), where jit
    is traced inline.
    """
    S = overlap_matrix(basis)
    T = kinetic_matrix(basis)
    V = nuclear_attraction_matrix(basis, coords, charges)
    ao, dao = ao_on_grid(basis, grid_coords) if materialize_ao else (None, None)
    e_nn = nuclear_repulsion(coords, charges)

    if aux_basis is None:
        # stream_exact: skip the O(N⁴) tensor; J/K are contracted on the fly
        # in _coulomb_exchange (coulomb_j_4c / exchange_k_4c).
        eri = None if stream_exact else eri4c_matrix(basis, quartets=eri_quartets, qof=eri_qof)
        int3c = None
        int2c_inv = None
    else:
        # int3c (nao²×naux) is the big DF tensor; skip it when streaming RI-J.
        int3c = eri3c_matrix(basis, aux_basis) if materialize_int3c else None
        int2c = eri2c_matrix(aux_basis)                   # (naux, naux); jit/grad-safe
        # Symmetric pseudo-inverse of the Coulomb metric, dropping near-null
        # directions. Standard JK-fitting auxiliary sets are heavily overcomplete
        # for small orbital bases (metric condition number ~1e12), so a loose
        # cutoff leaves ~1e7 amplification in the inverse that injects noise into
        # the Fock and makes the SCF limit-cycle. A 1e-7 relative cutoff keeps
        # the metric well-conditioned; the dropped directions are redundant so
        # the RI error stays sub-mHa.
        w, U = jnp.linalg.eigh(int2c)
        inv_w = jnp.where(w > 1e-7 * w[-1], 1.0 / w, 0.0)
        int2c_inv = (U * inv_w) @ U.T
        eri = None

    return S, T + V, ao, dao, e_nn, eri, int3c, int2c_inv


def _streamed_e_xc(xc, basis, coords, weights, P, chunk):
    """XC energy ``∫ ε_xc ρ`` streamed over grid-point chunks.

    AO values (and gradients for GGA) are recomputed per chunk and rematerialized
    in the backward pass, so memory is O(chunk·nao) rather than O(ng·nao), and
    the Fock (grad wrt P) and forces (grad wrt coords) stay memory-light. The
    nan-safe density threshold mirrors ``grid.xc_energy``.
    """
    gga = xc.xc_type == "GGA"

    def point(r, w):
        ao_g = eval_gto(basis, r)                       # (nao,)
        rho = ao_g @ P @ ao_g
        mask = rho > 1e-10
        safe_rho = jnp.where(mask, rho, 1.0)
        if gga:
            dao_g = jax.jacfwd(eval_gto, argnums=1)(basis, r)   # (nao, 3)
            grad = 2.0 * (ao_g @ P) @ dao_g             # (3,)
            eps = xc(safe_rho, jnp.where(mask, grad, 0.0))
        else:
            eps = xc(safe_rho)
        return jnp.where(mask, w * eps * rho, 0.0)

    contribs = _chunked_vmap(
        point, in_axes=(0, 0), chunk_size=chunk, checkpoint=True
    )(coords, weights)
    return jnp.sum(contribs)


def _eri3c_elem(basis, aux_basis, i, j, k):
    ml, mt, mm = _eri3c_sizes(basis, aux_basis)   # per-molecule recursion sizes
    return _contracted_eri3c(
        basis.exponents[i], basis.coefficients[i], basis.centers[i], basis.angular[i],
        basis.exponents[j], basis.coefficients[j], basis.centers[j], basis.angular[j],
        aux_basis.exponents[k], aux_basis.coefficients[k],
        aux_basis.centers[k], aux_basis.angular[k],
        ml, mt, mm,
    )


def _streamed_df_rij(basis, aux_basis, int2c_inv, P, chunk, pairs=None):
    """RI-J Coulomb energy ``½ γᵀ V⁻¹ γ`` streamed over auxiliary chunks.

    ``γ_P = Σ_μν (μν|P) P_μν`` is formed without materializing the (nao²×naux)
    3-center tensor: each auxiliary function's 3-center block is recomputed (and
    rematerialized in the backward pass) and contracted with the density on the
    fly, so DF memory is O(chunk·nao²) instead of O(nao²·naux).

    ``pairs`` (``(pi, pj, w)`` from :func:`~dftax.integrals.eri4c.significant_pairs`)
    restricts the bra sum to the significant Schwarz pairs ``i<=j`` (with the i<->j
    weight ``w``), turning the per-aux contraction from O(nao²) to O(N) for extended
    systems. When ``None`` the full nao² grid is used (dense, exact).
    """
    Ptil = basis.cart2sph @ P @ basis.cart2sph.T if basis.cart2sph is not None else P
    naux = aux_basis.centers.shape[0]

    if pairs is None:
        n = basis.centers.shape[0]
        idx = jnp.arange(n)

        def gamma_k(k):
            block = jax.vmap(
                lambda i: jax.vmap(lambda j: _eri3c_elem(basis, aux_basis, i, j, k))(idx)
            )(idx)                                              # (n, n)
            return jnp.sum(block * Ptil)
    else:
        pi, pj, w = pairs
        Pw = Ptil[pi, pj] * w                                  # (n_sig,) folded i<->j
        pidx = jnp.arange(pi.shape[0])

        def gamma_k(k):
            vals = jax.vmap(lambda p: _eri3c_elem(basis, aux_basis, pi[p], pj[p], k))(pidx)
            return jnp.sum(vals * Pw)                          # (n_sig,) -> scalar

    gamma = _chunked_vmap(gamma_k, chunk_size=chunk, checkpoint=True)(jnp.arange(naux))
    return 0.5 * jnp.dot(gamma, int2c_inv @ gamma)


# ---------------------------------------------------------------------------
# Streamed RI-K (exchange): orbital-chunk, O(nao·naux) memory, custom_vjp
# ---------------------------------------------------------------------------

def _rik_occ_orbitals(P, S, nocc, dscale=0.5):
    """Occupied MO coefficients (S-orthonormal) from a density ``P = (1/dscale) C Cᵀ``.

    In a non-orthogonal AO basis the orbitals are recovered via the symmetric
    orthonormalizer: ``S^{1/2}(dscale·P)S^{1/2}`` is a projector whose top-``nocc``
    eigenvectors give ``C`` (back-transformed by ``S^{-1/2}``). ``dscale=0.5`` for a
    closed shell (``P=2CCᵀ``); ``dscale=1`` for one spin channel (``P_σ=CCᵀ``). Used
    only inside the RI-K custom_vjp forward; its gradient is supplied analytically,
    so this eigh is never differentiated (avoids the degenerate-occupation blow-up).
    """
    sval, svec = jnp.linalg.eigh(S)
    sval = jnp.clip(sval, 1e-12, None)
    s_ih = (svec / jnp.sqrt(sval)) @ svec.T
    s_h = (svec * jnp.sqrt(sval)) @ svec.T
    _, evec = jnp.linalg.eigh(s_h @ (dscale * P) @ s_h)
    # Index from the right edge, NOT evec[:, -nocc:]: for an empty spin channel
    # (nocc==0, e.g. the β channel of a one-electron UKS system) `-0 == 0` would
    # select ALL columns instead of none. `nocc` is static, so this slice is fixed.
    ncol = evec.shape[1]
    return s_ih @ evec[:, ncol - nocc:]                    # (nao, nocc)


def _rik_cholesky(int2c_inv):
    """Factor ``L`` with ``V⁻¹ = L Lᵀ`` (from the symmetric eigendecomposition)."""
    w, U = jnp.linalg.eigh(int2c_inv)
    return U * jnp.sqrt(jnp.clip(w, 0.0, None))            # (naux, naux)


def _rik_bmj(basis, aux_basis, Lf, cj, n, naux):
    """Metric-fitted half-transformed 3-center for one occupied orbital (cartesian).

    ``B_mx = Σ_P (mj|P) L_Px`` with ``(mj|P) = Σ_l c_j[l] (ml|P)``; the 3-center is
    recomputed (never stored), so memory is O(nao·naux) per orbital.
    """
    idx = jnp.arange(n)
    aux_idx = jnp.arange(naux)

    def entry(m, k):
        col = jax.vmap(lambda l: _eri3c_elem(basis, aux_basis, m, l, k))(idx)
        return col @ cj
    M = jax.vmap(lambda m: jax.vmap(lambda k: entry(m, k))(aux_idx))(idx)   # (n, naux)
    return M @ Lf                                                            # (n, naux)


def _rik_energy(basis, aux_basis, int2c_inv, Cocc):
    """Raw exchange sum ``Σ_ijx B_ijx²`` streamed over occupied orbitals (the energy
    is ``energy_pref · this``; O(nao·naux) memory)."""
    Lf = _rik_cholesky(int2c_inv)
    c2s = basis.cart2sph
    Cc = c2s @ Cocc if c2s is not None else Cocc           # (n_cart, nocc)
    n, naux = Cc.shape[0], Lf.shape[0]

    def body(acc, cj):
        Bij = Cc.T @ _rik_bmj(basis, aux_basis, Lf, cj, n, naux)    # (nocc, naux)
        return acc + jnp.sum(Bij * Bij), None
    ek, _ = jax.lax.scan(jax.checkpoint(body), jnp.array(0.0), Cc.T)
    return ek


def _rik_kmatrix(basis, aux_basis, int2c_inv, Cocc):
    """Raw exchange kernel ``KK_mn = Σ_jx B_mjx B_njx`` (spherical); the analytic
    gradient is ``grad_pref · KK``."""
    Lf = _rik_cholesky(int2c_inv)
    c2s = basis.cart2sph
    Cc = c2s @ Cocc if c2s is not None else Cocc
    n, naux = Cc.shape[0], Lf.shape[0]

    def body(Ka, cj):
        B = _rik_bmj(basis, aux_basis, Lf, cj, n, naux)            # (n, naux)
        return Ka + (B @ B.T), None
    Kc, _ = jax.lax.scan(jax.checkpoint(body), jnp.zeros((n, n)), Cc.T)
    return c2s.T @ Kc @ c2s if c2s is not None else Kc


def _streamed_df_rik(basis, aux_basis, int2c_inv, S, nocc, P,
                     dscale, energy_pref, grad_pref):
    """Streamed RI-K exchange energy with an exact analytic gradient.

    Orbital-chunk RI-K: ``E_K = energy_pref · Σ_ijx (Σ_P (ij|P) L_Px)²`` (``V⁻¹=LLᵀ``),
    streamed over occupied orbitals so the nao²×naux 3-center is never materialized
    (O(nao·naux) memory, O(nao²·naux·nocc) compute). The occupied orbitals are
    extracted from ``P`` (``dscale·P`` projector) inside the forward; a ``custom_vjp``
    avoids differentiating through that extraction by returning the exact exchange
    Fock ``∂E_K/∂P = grad_pref · KK`` as the gradient.

    Closed shell: ``dscale=0.5, energy_pref=grad_pref=-a_x``. One spin channel:
    ``dscale=1, energy_pref=-½a_x, grad_pref=-a_x``.

    Gradient semantics (read before differentiating this). The ``custom_vjp``
    supplies ``∂E_K/∂P = grad_pref·KK``, the frozen-orbital exchange Fock. This
    equals the true derivative only at an **idempotent** density (``P = C Cᵀ`` for a
    spin channel, ``2 C Cᵀ`` closed-shell), i.e. the SCF density, which is the only
    intended use (computing the KS Fock via ``∂E/∂P``). Off idempotency the forward
    re-extracts orbitals from ``P`` while the backward holds them fixed, so the two
    describe different functions, so do **not** finite-difference or differentiate this
    energy at a non-stationary ``P``. The vjp is wrt ``P`` only: gradients wrt the
    basis/nuclear coordinates are **not** propagated, so geometry derivatives
    (forces) must use the materialized DF or exact path, not the streamed RI-K.
    """
    @jax.custom_vjp
    def rik(P):
        Cocc = _rik_occ_orbitals(P, S, nocc, dscale)
        return energy_pref * _rik_energy(basis, aux_basis, int2c_inv, Cocc)

    def fwd(P):
        Cocc = _rik_occ_orbitals(P, S, nocc, dscale)
        return energy_pref * _rik_energy(basis, aux_basis, int2c_inv, Cocc), Cocc

    def bwd(Cocc, g):
        KK = _rik_kmatrix(basis, aux_basis, int2c_inv, Cocc)
        return (g * grad_pref * KK,)

    rik.defvjp(fwd, bwd)
    return rik(P)


class RKS(eqx.Module):
    """Closed-shell KS total energy as a differentiable function of ``P``."""

    S: Float[Array, "nao nao"]
    hcore: Float[Array, "nao nao"]
    weights: Float[Array, "ng"]
    e_nn: Scalar
    basis: BasisData
    grid_coords: Float[Array, "ng 3"]
    # AO values/grads precomputed on the grid (None when streaming the XC grid):
    ao: Float[Array, "ng nao"] | None
    dao: Float[Array, "ng nao 3"] | None
    # Exact Coulomb backend (None when density fitting):
    eri: Float[Array, "nao nao nao nao"] | None
    # Density-fitting backend (None when exact); int3c is None when streaming RI-J.
    int3c: Float[Array, "nao nao naux"] | None
    int2c_inv: Float[Array, "naux naux"] | None
    aux_basis: BasisData | None

    nelec: int = eqx.field(static=True)
    xc: XCFunctional = eqx.field(static=True)
    hf_coeff: float = eqx.field(static=True)
    density_fit: bool = eqx.field(static=True)
    grid_chunk: int | None = eqx.field(static=True, default=None)
    df_chunk: int | None = eqx.field(static=True, default=None)
    exact_stream: bool = eqx.field(static=True, default=False)
    # Significant Schwarz bra pairs for screened streamed RI-J (None = dense):
    df_pi: Array | None = None
    df_pj: Array | None = None
    df_w: Float[Array, "npair"] | None = None

    @classmethod
    def _assemble(
        cls,
        basis: BasisData,
        coords: Float[Array, "n_atoms 3"],
        charges: Float[Array, "n_atoms"],
        nelec: int,
        xc: XCFunctional,
        grid_coords: Float[Array, "ng 3"],
        grid_weights: Float[Array, "ng"],
        aux_basis: BasisData | None = None,
        grid_chunk: int | None = None,
        df_chunk: int | None = None,
        eri_quartets=None,
        eri_qof=None,
        exact_stream: bool = False,
        df_pairs=None,
    ) -> "RKS":
        """Assemble the energy functional from a basis + geometry + grid.

        If ``aux_basis`` is given, use density fitting; otherwise the exact
        4-center ERI. ``grid_chunk`` streams the XC grid (O(chunk·nao) grid
        memory); ``df_chunk`` streams RI-J (and, for hybrids, RI-K) over auxiliary
        chunks, avoiding the nao²×naux 3-center tensor.
        ``eri_quartets``/``eri_qof`` optionally supply a pre-screened
        (Cauchy-Schwarz) exact-ERI quartet list + orbit map (exact path only).
        """
        coords = jnp.asarray(coords)
        charges = jnp.asarray(charges, dtype=coords.dtype)
        grid_coords = jnp.asarray(grid_coords)
        S, hcore, ao, dao, e_nn, eri, int3c, int2c_inv = _build_integrals(
            basis, coords, charges, grid_coords, aux_basis,
            grid_chunk is None, df_chunk is None,
            eri_quartets, eri_qof, exact_stream,
        )
        return cls(
            S=S,
            hcore=hcore,
            weights=jnp.asarray(grid_weights),
            e_nn=e_nn,
            basis=basis,
            grid_coords=grid_coords,
            ao=ao,
            dao=dao,
            eri=eri,
            int3c=int3c,
            int2c_inv=int2c_inv,
            aux_basis=aux_basis,
            nelec=int(nelec),
            xc=xc,
            hf_coeff=float(xc.hf_coeff),
            density_fit=(aux_basis is not None),
            grid_chunk=grid_chunk,
            df_chunk=df_chunk,
            exact_stream=exact_stream,
            df_pi=(None if df_pairs is None else df_pairs[0]),
            df_pj=(None if df_pairs is None else df_pairs[1]),
            df_w=(None if df_pairs is None else df_pairs[2]),
        )

    @classmethod
    def from_pyscf(
        cls,
        mol,
        xc: XCFunctional,
        grid_coords: Float[Array, "ng 3"],
        grid_weights: Float[Array, "ng"],
        auxbasis: str | None = None,
        grid_chunk: int | None = None,
        df_chunk: int | None = None,
        eri_screen: float | None = None,
        exact_stream: bool = False,
        df_screen: float | None = None,
    ) -> "RKS":
        """Build from a PySCF ``Mole`` (setup only) and a quadrature grid.

        PySCF is used here solely to parse the orbital basis and supply nuclear
        geometry/charges; nothing PySCF enters the compute path. ``auxbasis``
        (a basis-set name) enables density fitting; ``grid_chunk`` streams the XC
        grid and ``df_chunk`` streams RI-J (both memory-light for large systems).
        ``eri_screen`` (exact path) sets a Cauchy-Schwarz threshold to skip
        negligible ERI quartets.
        """
        basis = extract_basis_data(mol)
        aux_basis = None
        if auxbasis is not None:
            from dftax.basis.loader import build_basis_data

            symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
            aux_basis = build_basis_data(symbols, mol.atom_coords(), auxbasis)
        eri_quartets, eri_qof = _screen_eri(basis, aux_basis, eri_screen)
        df_pairs = None
        if df_screen is not None and aux_basis is not None and df_chunk is not None:
            df_pairs = significant_pairs(basis, float(df_screen))  # screened streamed RI-J
        return cls._assemble(
            basis,
            mol.atom_coords(),
            mol.atom_charges(),
            mol.nelectron,
            xc,
            grid_coords,
            grid_weights,
            aux_basis=aux_basis,
            grid_chunk=grid_chunk,
            df_chunk=df_chunk,
            eri_quartets=eri_quartets,
            eri_qof=eri_qof,
            exact_stream=exact_stream,
            df_pairs=df_pairs,
        )

    @classmethod
    def from_molecule(
        cls,
        mol,
        xc: XCFunctional,
        grid_coords: Float[Array, "ng 3"],
        grid_weights: Float[Array, "ng"],
        auxbasis: str | None = None,
        spherical: bool = False,
        grid_chunk: int | None = None,
        df_chunk: int | None = None,
        eri_screen: float | None = None,
        exact_stream: bool = False,
        df_screen: float | None = None,
    ) -> "RKS":
        """Build from a native :class:`~dftax.system.molecule.Molecule` (no PySCF).

        ``auxbasis`` (a basis-set name, e.g. ``"def2-universal-jkfit"``) enables
        density fitting. ``spherical=True`` uses spherical-harmonic orbitals
        (standard for cc-pVXZ/def2; required to match a spherical reference for
        l>=2 bases). ``grid_chunk`` streams the XC grid and ``df_chunk`` streams
        RI-J (both memory-light). ``eri_screen`` (exact path) sets a
        Cauchy-Schwarz threshold to skip negligible ERI quartets.
        """
        from dftax.basis.loader import build_basis_data

        basis = build_basis_data(
            mol.symbols, mol.atom_coords(), mol.basis, spherical=spherical
        )
        aux_basis = None
        if auxbasis is not None:
            aux_basis = build_basis_data(mol.symbols, mol.atom_coords(), auxbasis)
        eri_quartets, eri_qof = _screen_eri(basis, aux_basis, eri_screen)
        df_pairs = None
        if df_screen is not None and aux_basis is not None and df_chunk is not None:
            df_pairs = significant_pairs(basis, float(df_screen))  # screened streamed RI-J
        return cls._assemble(
            basis,
            mol.atom_coords(),
            mol.atom_charges(),
            mol.nelectron,
            xc,
            grid_coords,
            grid_weights,
            aux_basis=aux_basis,
            grid_chunk=grid_chunk,
            df_chunk=df_chunk,
            eri_quartets=eri_quartets,
            eri_qof=eri_qof,
            exact_stream=exact_stream,
            df_pairs=df_pairs,
        )

    # -- density on the grid ------------------------------------------------

    def density(
        self, P: Float[Array, "nao nao"]
    ) -> tuple[Float[Array, "ng"], Float[Array, "ng 3"]]:
        """Electron density and its gradient on the grid from ``P``."""
        rho = jnp.einsum("gm,mn,gn->g", self.ao, P, self.ao)
        grad_rho = 2.0 * jnp.einsum("gm,mn,gnx->gx", self.ao, P, self.dao)
        return rho, grad_rho

    def e_xc(self, P: Float[Array, "nao nao"]) -> Scalar:
        """Exchange-correlation energy ``∫ ε_xc ρ`` (DFT part only).

        Uses precomputed AO grid values when materialized; otherwise streams the
        grid in chunks (recomputing AO per chunk) for O(chunk·nao) memory.
        """
        if self.ao is not None:
            rho, grad_rho = self.density(P)
            gr = grad_rho if self.xc.xc_type == "GGA" else None
            return xc_energy(self.xc, rho, self.weights, grad_rho=gr)
        return _streamed_e_xc(
            self.xc, self.basis, self.grid_coords, self.weights, P, self.grid_chunk
        )

    # -- Coulomb + exact exchange ------------------------------------------

    def _coulomb_exchange(self, P: Float[Array, "nao nao"]) -> Scalar:
        """E_J + a_x·E_x^exact via the exact ERI or density fitting."""
        if self.density_fit:
            if self.int3c is None:  # streamed RI-J (+ streamed RI-K for hybrids)
                pairs = None if self.df_pi is None else (self.df_pi, self.df_pj, self.df_w)
                e_j = _streamed_df_rij(
                    self.basis, self.aux_basis, self.int2c_inv, P, self.df_chunk, pairs
                )
                if self.hf_coeff != 0.0:                  # closed shell: P = 2 C Cᵀ
                    e_j = e_j + _streamed_df_rik(
                        self.basis, self.aux_basis, self.int2c_inv, self.S,
                        self.nelec // 2, P, 0.5, -self.hf_coeff, -self.hf_coeff,
                    )
                return e_j
            gamma = jnp.einsum("mnP,mn->P", self.int3c, P)        # (P|ρ)
            e_j = 0.5 * jnp.dot(gamma, self.int2c_inv @ gamma)    # ½ γᵀ V⁻¹ γ
            if self.hf_coeff != 0.0:
                K = jnp.einsum(
                    "mlP,PQ,nsQ,ls->mn", self.int3c, self.int2c_inv, self.int3c, P
                )
                e_j = e_j - 0.25 * self.hf_coeff * jnp.sum(P * K)
            return e_j
        if self.exact_stream:                                # exact J/K, contracted on the fly
            J = coulomb_j_4c(P, self.basis)
            e_j = 0.5 * jnp.sum(P * J)
            if self.hf_coeff != 0.0:
                K = exchange_k_4c(P, self.basis)
                e_j = e_j - 0.25 * self.hf_coeff * jnp.sum(P * K)
            return e_j
        J = jnp.einsum("ijkl,kl->ij", self.eri, P)
        e_j = 0.5 * jnp.sum(P * J)
        if self.hf_coeff != 0.0:
            K = jnp.einsum("ikjl,kl->ij", self.eri, P)
            e_j = e_j - 0.25 * self.hf_coeff * jnp.sum(P * K)
        return e_j

    # -- energy -------------------------------------------------------------

    def electronic(self, P: Float[Array, "nao nao"]) -> Scalar:
        """Electronic energy ``Tr(P·Hcore) + E_J + a_x·E_x^exact + E_xc``."""
        e1 = jnp.sum(P * self.hcore)
        return e1 + self._coulomb_exchange(P) + self.e_xc(P)

    def total(self, P: Float[Array, "nao nao"]) -> Scalar:
        """Total KS energy (electronic + nuclear repulsion)."""
        return self.electronic(P) + self.e_nn
