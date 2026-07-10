"""Differentiable Kohn-Sham total energy as a function of the density matrix.

The :class:`KS` module precomputes the one-electron Hamiltonian and the nuclear
repulsion, and holds one Coulomb term and one XC term (see
:mod:`dftax.ks.terms`) that carry exactly the integral arrays their backend
needs. It exposes ``electronic(P)`` / ``total(P)``, a single,
``jit``/``grad``-friendly energy functional of the **spin-stacked** density
``P`` of shape ``(nspin, nao, nao)`` (see the convention in
:mod:`dftax.ks.terms`). The KS Fock matrices are obtained downstream as
``F_σ = sym(∂E/∂P_σ)`` (see :mod:`dftax.ks.scf`), so no exchange-correlation
potential matrix is hand-coded.

The spin structure is the static tuple ``nocc`` of per-channel occupied
counts: ``(nelec//2,)`` for a closed shell (one doubly-occupied channel),
``(nα, nβ)`` for a spin-polarized system (unit occupation per channel). All
energy formulas are shared; the closed shell is simply ``nspin == 1``.

:class:`RKS` and :class:`UKS` are thin facades over :class:`KS` that keep the
historical call signatures (a single ``P`` for RKS, ``(Pα, Pβ)`` for UKS) and
the flag-based constructors; both are slated to fold into the unified build
API. The Coulomb/exchange backend is chosen with the
:func:`~dftax.ks.terms.exact` / :func:`~dftax.ks.terms.df` factories:

- **exact** (default): the full 4-center ERI tensor (``eri4c``), tight against
  PySCF, but O(N⁴) memory, so only for small systems.
- **density fitting** (``df(auxbasis)``): RI-J / RI-K via 3- and 2-center
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
from dftax.energy.xc import XCFunctional
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
    significant_pairs,
)
from dftax.ks.terms import (
    CoulombTerm,
    DFSpec,
    ExactSpec,
    GridXC,
    StreamedGridXC,
    XCTerm,
    _make_coulomb,
    _metric_pinv,  # noqa: F401  (re-exported; also used by _build_integrals)
    df,
    exact,
)


def _spin_counts(nelec: int, spin: int) -> tuple[int, int]:
    """(nα, nβ) from electron count and ``spin = 2S = nα - nβ``."""
    if (nelec + spin) % 2 != 0:
        raise ValueError(
            f"Inconsistent (nelec={nelec}, spin={spin}): nelec+spin must be even."
        )
    n_alpha = (nelec + spin) // 2
    n_beta = (nelec - spin) // 2
    if n_beta < 0:
        raise ValueError(f"spin={spin} too large for nelec={nelec} (nβ<0).")
    return n_alpha, n_beta


def _nocc_tuple(nelec: int, spin: int | None) -> tuple[int, ...]:
    """Per-channel occupied counts: ``spin=None`` → one doubly-occupied channel."""
    if spin is None:
        return (int(nelec) // 2,)
    return _spin_counts(int(nelec), int(spin))


def _resolve_screening(spec, basis):
    """Eager Cauchy-Schwarz resolution for a Coulomb spec.

    Returns ``(quartets, qof, pairs)``: a pre-screened exact-ERI quartet list +
    orbit map (materialized exact path), or the significant Schwarz bra pairs
    (screened streamed RI-J). Screening depends on concrete integral values, so
    it is resolved here in the eager constructors (the basis is materialised)
    rather than inside the jitted ``_build_integrals``.
    """
    if isinstance(spec, DFSpec):
        if spec.screen is not None and spec.chunk is not None:
            return None, None, significant_pairs(basis, float(spec.screen))
        return None, None, None
    if spec.screen is not None and not spec.stream:
        quartets, qof = screened_quartets(basis, float(spec.screen))
        return jnp.asarray(quartets), jnp.asarray(qof), None
    return None, None, None


def _spec_from_flags(auxbasis, df_chunk, df_screen, eri_screen, exact_stream, load_aux):
    """Translate the legacy flag cluster into a Coulomb spec (constructor shim).

    Mirrors the historical flag semantics exactly: ``df_screen`` is honored only
    with ``auxbasis`` + ``df_chunk``, ``eri_screen`` only on the materialized
    exact path — combinations that used to be silently inert stay inert here.
    The strict factories (:func:`~dftax.ks.terms.exact` /
    :func:`~dftax.ks.terms.df`) reject those combinations for direct users.
    """
    if auxbasis is not None:
        aux = auxbasis if isinstance(auxbasis, BasisData) else load_aux(auxbasis)
        return df(
            aux,
            chunk=df_chunk,
            screen=df_screen if df_chunk is not None else None,
        )
    return exact(
        screen=None if exact_stream else eri_screen,
        stream=exact_stream,
    )


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
    (the XC grid is streamed instead; see ``terms._streamed_e_xc``). ``eri_quartets``/
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
        # in StreamedExactCoulomb (coulomb_j_4c / exchange_k_4c).
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
        int2c_inv = _metric_pinv(int2c)
        eri = None

    return S, T + V, ao, dao, e_nn, eri, int3c, int2c_inv


class KS(eqx.Module):
    """Spin-stacked KS total energy as a differentiable function of ``P``.

    ``P`` has shape ``(nspin, nao, nao)`` with ``nspin = len(self.nocc)``:
    one doubly-occupied channel (``P[0] = 2ΣCCᵀ``) for a closed shell, two
    unit-occupation channels (``P[σ] = ΣC_σC_σᵀ``) for a spin-polarized system.
    """

    S: Float[Array, "nao nao"]
    hcore: Float[Array, "nao nao"]
    e_nn: Scalar
    basis: BasisData
    coulomb: CoulombTerm
    xc_term: XCTerm
    nelec: int = eqx.field(static=True)
    nocc: tuple[int, ...] = eqx.field(static=True)

    def e_xc(self, P: Float[Array, "nspin nao nao"]) -> Scalar:
        """Exchange-correlation energy ``∫ ε_xc ρ`` (DFT part only)."""
        return self.xc_term.energy(P)

    def electronic(self, P: Float[Array, "nspin nao nao"]) -> Scalar:
        """Electronic energy ``Tr(P_tot·Hcore) + E_J + a_x·E_x^exact + E_xc``."""
        e1 = jnp.sum(jnp.sum(P, axis=0) * self.hcore)
        e2 = self.coulomb.energy(P, self.S, self.nocc)
        return e1 + e2 + self.xc_term.energy(P)

    def total(self, P: Float[Array, "nspin nao nao"]) -> Scalar:
        """Total KS energy (electronic + nuclear repulsion)."""
        # KS.electronic explicitly: the RKS/UKS facades override ``electronic``
        # with their historical signatures, and this must not dispatch there.
        return KS.electronic(self, P) + self.e_nn


def _assemble_ks(
    cls,
    basis: BasisData,
    coords: Float[Array, "n_atoms 3"],
    charges: Float[Array, "n_atoms"],
    nelec: int,
    spin: int | None,
    xc: XCFunctional,
    grid_coords: Float[Array, "ng 3"],
    grid_weights: Float[Array, "ng"],
    coulomb: ExactSpec | DFSpec | None,
    grid_chunk: int | None,
):
    """Assemble a :class:`KS` (or facade subclass) from basis + geometry + grid.

    ``spin=None`` builds the closed-shell channel structure; an integer builds
    the (nα, nβ) channels. ``coulomb`` selects the Coulomb/exchange backend (a
    spec from :func:`~dftax.ks.terms.exact` / :func:`~dftax.ks.terms.df`, with
    any auxiliary basis already resolved to ``BasisData``); ``None`` means the
    materialized exact ERI. ``grid_chunk`` streams the XC grid (O(chunk·nao)
    grid memory) instead of materializing the AO values.
    """
    spec = exact() if coulomb is None else coulomb
    nocc = _nocc_tuple(nelec, spin)
    coords = jnp.asarray(coords)
    charges = jnp.asarray(charges, dtype=coords.dtype)
    grid_coords = jnp.asarray(grid_coords)
    weights = jnp.asarray(grid_weights)

    quartets, qof, pairs = _resolve_screening(spec, basis)
    is_df = isinstance(spec, DFSpec)
    aux_basis = spec.auxbasis if is_df else None
    S, hcore, ao, dao, e_nn, eri, int3c, int2c_inv = _build_integrals(
        basis, coords, charges, grid_coords, aux_basis,
        grid_chunk is None, not (is_df and spec.chunk is not None),
        quartets, qof, (not is_df) and spec.stream,
    )
    coulomb_term = _make_coulomb(
        spec, basis, eri, int3c, int2c_inv, pairs, float(xc.hf_coeff)
    )
    if grid_chunk is None:
        xc_term = GridXC(ao=ao, dao=dao, weights=weights, xc=xc)
    else:
        xc_term = StreamedGridXC(
            basis=basis, grid_coords=grid_coords, weights=weights,
            chunk=grid_chunk, xc=xc,
        )
    return cls(
        S=S, hcore=hcore, e_nn=e_nn, basis=basis,
        coulomb=coulomb_term, xc_term=xc_term,
        nelec=int(nelec), nocc=nocc,
    )


def _from_pyscf(
    cls, mol, xc, grid_coords, grid_weights, spin,
    auxbasis, grid_chunk, df_chunk, eri_screen, exact_stream, df_screen,
):
    """Shared PySCF-``Mole`` constructor body (setup only, nothing PySCF in the
    compute path): parse the basis, translate the legacy flags, assemble."""
    basis = extract_basis_data(mol)

    def load_aux(name):
        from dftax.basis.loader import build_basis_data

        symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        return build_basis_data(symbols, mol.atom_coords(), name)

    spec = _spec_from_flags(
        auxbasis, df_chunk, df_screen, eri_screen, exact_stream, load_aux
    )
    return _assemble_ks(
        cls, basis, mol.atom_coords(), mol.atom_charges(), mol.nelectron, spin,
        xc, grid_coords, grid_weights, spec, grid_chunk,
    )


def _from_molecule(
    cls, mol, xc, grid_coords, grid_weights, spin,
    auxbasis, spherical, grid_chunk, df_chunk, eri_screen, exact_stream, df_screen,
):
    """Shared native-:class:`~dftax.system.molecule.Molecule` constructor body."""
    from dftax.basis.loader import build_basis_data

    basis = build_basis_data(
        mol.symbols, mol.atom_coords(), mol.basis, spherical=spherical
    )
    spec = _spec_from_flags(
        auxbasis, df_chunk, df_screen, eri_screen, exact_stream,
        lambda name: build_basis_data(mol.symbols, mol.atom_coords(), name),
    )
    return _assemble_ks(
        cls, basis, mol.atom_coords(), mol.atom_charges(), mol.nelectron, spin,
        xc, grid_coords, grid_weights, spec, grid_chunk,
    )


class RKS(KS):
    """Closed-shell facade over :class:`KS`: energies of a single ``P`` (nao, nao)."""

    @classmethod
    def _assemble(
        cls, basis, coords, charges, nelec, xc, grid_coords, grid_weights,
        *, coulomb: ExactSpec | DFSpec | None = None, grid_chunk: int | None = None,
    ) -> "RKS":
        """Assemble the closed-shell energy functional (see :func:`_assemble_ks`)."""
        return _assemble_ks(
            cls, basis, coords, charges, nelec, None, xc,
            grid_coords, grid_weights, coulomb, grid_chunk,
        )

    @classmethod
    def from_pyscf(
        cls, mol, xc: XCFunctional, grid_coords, grid_weights,
        auxbasis: str | None = None, grid_chunk: int | None = None,
        df_chunk: int | None = None, eri_screen: float | None = None,
        exact_stream: bool = False, df_screen: float | None = None,
    ) -> "RKS":
        """Build from a PySCF ``Mole`` (setup only) and a quadrature grid.

        ``auxbasis`` (a basis-set name) enables density fitting; ``grid_chunk``
        streams the XC grid and ``df_chunk`` streams RI-J (both memory-light for
        large systems). ``eri_screen`` (exact path) sets a Cauchy-Schwarz
        threshold to skip negligible ERI quartets.
        """
        return _from_pyscf(
            cls, mol, xc, grid_coords, grid_weights, None,
            auxbasis, grid_chunk, df_chunk, eri_screen, exact_stream, df_screen,
        )

    @classmethod
    def from_molecule(
        cls, mol, xc: XCFunctional, grid_coords, grid_weights,
        auxbasis: str | None = None, spherical: bool = False,
        grid_chunk: int | None = None, df_chunk: int | None = None,
        eri_screen: float | None = None, exact_stream: bool = False,
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
        return _from_molecule(
            cls, mol, xc, grid_coords, grid_weights, None,
            auxbasis, spherical, grid_chunk, df_chunk,
            eri_screen, exact_stream, df_screen,
        )

    def density(
        self, P: Float[Array, "nao nao"]
    ) -> tuple[Float[Array, "ng"], Float[Array, "ng 3"]]:
        """Electron density and its gradient on the grid from ``P``."""
        if not isinstance(self.xc_term, GridXC):
            raise NotImplementedError(
                "density on the grid requires materialized AO values; "
                "this RKS streams the XC grid (grid_chunk)."
            )
        return self.xc_term.density(P[None])

    def e_xc(self, P: Float[Array, "nao nao"]) -> Scalar:
        return KS.e_xc(self, P[None])

    def electronic(self, P: Float[Array, "nao nao"]) -> Scalar:
        return KS.electronic(self, P[None])

    def total(self, P: Float[Array, "nao nao"]) -> Scalar:
        return KS.total(self, P[None])


class UKS(KS):
    """Open-shell facade over :class:`KS`: energies of ``(Pα, Pβ)``."""

    @property
    def nalpha(self) -> int:
        return self.nocc[0]

    @property
    def nbeta(self) -> int:
        return self.nocc[1]

    @classmethod
    def _assemble(
        cls, basis, coords, charges, nelec, spin, xc, grid_coords, grid_weights,
        *, coulomb: ExactSpec | DFSpec | None = None, grid_chunk: int | None = None,
    ) -> "UKS":
        """Assemble the open-shell energy functional (see :func:`_assemble_ks`)."""
        return _assemble_ks(
            cls, basis, coords, charges, nelec, int(spin), xc,
            grid_coords, grid_weights, coulomb, grid_chunk,
        )

    @classmethod
    def from_pyscf(
        cls, mol, xc: XCFunctional, grid_coords, grid_weights,
        auxbasis: str | None = None, spin: int | None = None,
        eri_screen: float | None = None, df_chunk: int | None = None,
        df_screen: float | None = None, grid_chunk: int | None = None,
    ) -> "UKS":
        """Build from a PySCF ``Mole``; ``spin`` (= 2S) defaults to ``mol.spin``."""
        spin = int(mol.spin if spin is None else spin)
        return _from_pyscf(
            cls, mol, xc, grid_coords, grid_weights, spin,
            auxbasis, grid_chunk, df_chunk, eri_screen, False, df_screen,
        )

    @classmethod
    def from_molecule(
        cls, mol, xc: XCFunctional, grid_coords, grid_weights,
        auxbasis: str | None = None, spherical: bool = False,
        spin: int | None = None, eri_screen: float | None = None,
        df_chunk: int | None = None, df_screen: float | None = None,
        grid_chunk: int | None = None,
    ) -> "UKS":
        """Build from a native ``Molecule``; ``spin`` (= 2S) defaults to ``mol.spin``."""
        spin = int(mol.spin if spin is None else spin)
        return _from_molecule(
            cls, mol, xc, grid_coords, grid_weights, spin,
            auxbasis, spherical, grid_chunk, df_chunk,
            eri_screen, False, df_screen,
        )

    def e_xc(self, Pa: Float[Array, "nao nao"], Pb: Float[Array, "nao nao"]) -> Scalar:
        return KS.e_xc(self, jnp.stack([Pa, Pb]))

    def electronic(
        self, Pa: Float[Array, "nao nao"], Pb: Float[Array, "nao nao"]
    ) -> Scalar:
        return KS.electronic(self, jnp.stack([Pa, Pb]))

    def total(
        self, Pa: Float[Array, "nao nao"], Pb: Float[Array, "nao nao"]
    ) -> Scalar:
        return KS.total(self, jnp.stack([Pa, Pb]))
