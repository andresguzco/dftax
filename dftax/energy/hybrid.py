"""Hartree-Fock exchange matrix via density fitting.

The hybrid DFT exchange term is
    E_X^HF[P] = -(1/2) * Tr(P · K),   K_{μν} = Σ_{λσ} P_{λσ} (μλ | νσ).

Under DF:
    (μλ | νσ) ≈ Σ_{PQ} (μλ | P) (P|Q)^{-1} (Q | νσ)

so
    K_{μν} = Σ_{PQ} (μλ | P) (P|Q)^{-1} (νσ | Q) P_{λσ}

Given the same ``int3c`` (shape (nao, nao, n_aux)) and ``int2c_inv`` that
``EnergyFunctional`` precomputes for the analytical Hartree term, K can be
assembled with a single einsum.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def df_k_matrix(
    P: Float[Array, "nao nao"],
    int3c: Float[Array, "nao nao n_aux"],
    int2c_inv: Float[Array, "n_aux n_aux"],
) -> Float[Array, "nao nao"]:
    """Build the DF-K matrix:

        K_{μν} = Σ_{PQ} (μλ|P) J^{-1}_{PQ} (νσ|Q) P_{λσ}

    with int3c storing the 3-center integrals (μν|P) in the orbital basis and
    int2c_inv the inverse of the 2-center Coulomb matrix J_{PQ} = (P|Q).
    """
    # einsum plan:
    #   int3c[μ, λ, P]   (mlP)
    #   int2c_inv[P, Q]  (PQ)
    #   int3c[ν, σ, Q]   (nsQ)
    #   P[λ, σ]          (ls)
    # → K[μ, ν]         (mn)
    return jnp.einsum("mlP,PQ,nsQ,ls->mn", int3c, int2c_inv, int3c, P)


def df_exchange_energy(
    P: Float[Array, "nao nao"],
    int3c: Float[Array, "nao nao n_aux"],
    int2c_inv: Float[Array, "n_aux n_aux"],
) -> Float[Array, ""]:
    """Closed-shell HF exchange energy.

    For closed-shell RKS, P is the **total** density matrix (P = 2 · P_α).
    The exchange energy is
        E_X^HF = -(1/4) Tr(P · K[P])
    (see Szabo & Ostlund §3.4). This is equivalent to the UKS form
    -(1/2) Σ_σ Tr(P_σ · K[P_σ]) since K is linear in its argument.
    """
    K = df_k_matrix(P, int3c, int2c_inv)
    return -0.25 * jnp.sum(P * K)
