"""Exchange-correlation functionals for KS-DFT."""

import abc
import equinox as eqx
import jax

from jax import numpy as jnp
from jax import Array
from jax.scipy.special import erf as _jerf
from typing import ClassVar
from jaxtyping import Float, Scalar


ScalarFeature = Float[Array, "s"] | Scalar
VectorFeature = Float[Array, "i s"] | Float[Array, "i"]


def _has_spin(density) -> bool:
    """Return True if density appears spin-polarized (last axis size 2)."""
    return jnp.atleast_1d(density).shape[-1] == 2


class DensityFunctional(eqx.Module):
    """Base class for exchange-correlation density functionals."""

    name: eqx.AbstractClassVar[str]
    xc_type: eqx.AbstractClassVar[str]

    @abc.abstractmethod
    def __call__(self, *args: ScalarFeature | VectorFeature) -> Scalar:
        raise NotImplementedError


class XCFunctional(DensityFunctional):
    """Composite functional combining exchange and correlation components.

    ``hf_coeff`` is the fraction of exact (Hartree-Fock) exchange; it is 0 for
    pure DFT functionals and non-zero for hybrids (e.g. 0.20 for B3LYP). The
    ``__call__`` method returns only the density-local part of ε_xc. The HF K
    contribution is computed in ``EnergyFunctional`` when ``hf_coeff != 0``.
    """

    exchange: eqx.AbstractClassVar[DensityFunctional]
    correlation: eqx.AbstractClassVar[DensityFunctional]
    hf_coeff: ClassVar[float] = 0.0
    # Range-separated hybrids: the exact-exchange split is
    # K_hf = hf_coeff·K + hf_coeff_lr·K_lr(omega), with K_lr built from
    # erf(ω·r₁₂)/r₁₂ integrals (see dftax.ks.terms). Both are 0 for global
    # hybrids and pure functionals.
    hf_coeff_lr: ClassVar[float] = 0.0
    omega: ClassVar[float] = 0.0
    # Nonlocal (VV10) correlation parameters; 0 means none. A functional
    # declaring nlc_b adds the VV10 double-grid term (see energy/vv10.py).
    nlc_b: ClassVar[float] = 0.0
    nlc_c: ClassVar[float] = 0.0

    def __call__(self, *args: ScalarFeature | VectorFeature) -> Scalar:
        return self.exchange(*args) + self.correlation(*args)


class LDAExchange(DensityFunctional):
    """Local density approximation (LDA) exchange energy density."""

    name: ClassVar[str] = "LDA"
    xc_type: ClassVar[str] = "LDA"

    @staticmethod
    def _unpolarized(density: ScalarFeature) -> Scalar:
        kF = (3 * jnp.pi**2 * density + 1e-30) ** (1 / 3)
        return -3 * kF / (4 * jnp.pi)

    def __call__(self, density: ScalarFeature) -> Scalar:
        if _has_spin(density):
            # Spin-scaling relation: E_x[n↑,n↓] = ½(E_x[2n↑] + E_x[2n↓]); the
            # per-electron density mixes the two channels by their populations.
            n_up, n_dn = jnp.unstack(density, axis=-1)
            n_tot = jnp.clip(n_up + n_dn, 1e-30)
            return (n_up / n_tot) * self._unpolarized(2 * n_up) + (
                n_dn / n_tot
            ) * self._unpolarized(2 * n_dn)
        return self._unpolarized(density)


class PBEExchange(DensityFunctional):
    """PBE GGA exchange energy density."""

    name: ClassVar[str] = "PBE"
    xc_type: ClassVar[str] = "GGA"

    kappa: ClassVar[float] = 0.804
    # μ = β·π²/3 with β = 0.06672455060314922 (the PBE/PW92 β, cf. PBECorrelation.beta);
    # this is libxc's exact MU_PBE. The earlier 0.21951 truncation (rel. err 2.3e-5) was
    # the *sole* source of the documented ~1e-5 Ha PBE/PBE0 gap vs libxc; full precision
    # restores machine-precision agreement (verified on water: 1.5e-5 → 2e-11 Ha).
    mu: ClassVar[float] = 0.2195149727645171

    def _unpolarized(self, density: ScalarFeature, grad_density: VectorFeature) -> Scalar:
        """Exchange for spin-unpolarized densities."""

        # eps = jnp.finfo(density.dtype).eps
        grad_n_norm = jnp.linalg.norm(grad_density + 1e-30, axis=-1)

        # +1e-30 floor (as in LDAExchange) keeps kF and its ρ→0 derivative finite;
        # ρ^{1/3}'s gradient is otherwise -inf at exactly ρ=0.
        kF = (3 * jnp.pi**2 * density + 1e-30) ** (1 / 3)
        s = grad_n_norm / jnp.clip(2 * kF * density, 1e-30)
        eps_x_unif = -3 * kF / (4 * jnp.pi)
        f = 1 + self.kappa - self.kappa / (1 + self.mu * s**2 / self.kappa)

        return eps_x_unif * f

    def __call__(self, density: ScalarFeature, grad_density: VectorFeature) -> Scalar:
        """Exchange for spin-unpolarized or spin-polarized densities."""

        if _has_spin(density):
            if density.shape != (2,):
                raise ValueError("Density must have shape (..., 2) for spin-polarized case.")

            if grad_density.shape != (3, 2):
                raise ValueError("Gradient density must have shape (3, 2) for spin-polarized case.")

            n_up, n_dn = jnp.unstack(density, axis=-1)
            grad_up, grad_dn = jnp.unstack(grad_density, axis=-1)
            n_tot = jnp.clip(n_up + n_dn, 1e-30)

            exc_up = self._unpolarized(2 * n_up, 2 * grad_up)
            exc_dn = self._unpolarized(2 * n_dn, 2 * grad_dn)

            return (n_up / n_tot) * exc_up + (n_dn / n_tot) * exc_dn

        else:
            if grad_density.shape != (3,):
                raise ValueError("Gradient density must have shape (3,) for spin-unpolarized case.")

            return self._unpolarized(density, grad_density)


class PBECorrelation(DensityFunctional):
    """PBE correlation energy density with PW92 base."""

    name: ClassVar[str] = "PBE"
    xc_type: ClassVar[str] = "GGA"

    gamma: ClassVar[float] = (1.0 - jnp.log(2.0)) / (jnp.pi**2)
    beta: ClassVar[float] = 0.06672455060314922
    fz20: ClassVar[float] = 1.709920934161365617563962776245

    # --- Constants ---
    # PW92 Modified Parameters
    # Rows: [Para, Ferro, Alpha] | Cols: [A, alpha1, beta1, beta2, beta3, beta4]
    pw_params: ClassVar[Float[Array, "3 6"]] = jnp.array(
        [
            [0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294],  # Paramagnetic
            [0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517],  # Ferromagnetic
            [0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671],  # Alpha
        ]
    )

    def _pw92_correlation(self, rs, zeta):
        """PW92 LDA correlation used as PBE baseline."""

        A, a1, b1, b2, b3, b4 = jnp.unstack(self.pw_params, axis=1)

        rs_sqrt = jnp.sqrt(rs)
        denom = 2.0 * A * (b1 * rs_sqrt + b2 * rs + b3 * rs * rs_sqrt + b4 * rs**2)
        g1, g2, g3 = -2.0 * A * (1.0 + a1 * rs) * jnp.log1p(1 / denom.clip(1e-20))

        f_z = ((1.0 + zeta) ** (4 / 3) + (1.0 - zeta) ** (4 / 3) - 2.0) / (2.0 ** (4 / 3) - 2.0)

        return g1 + (zeta**4) * f_z * (g2 - g1 + g3 / self.fz20) - f_z * g3 / self.fz20

    def _pbe_correction(self, eps_pw, phi, t):
        """Calculate the PBE gradient correction term (H)."""
        # A = beta / [ gamma * (exp( -eps_pw / (gamma * phi^3) ) - 1) ]
        # Note: eps_pw is negative, so -eps_pw is positive.
        gamma_phi3 = self.gamma * (phi**3)

        # Use expm1 for stability when exponent is small (low density limit)
        denom_A = self.gamma * jnp.expm1(-eps_pw / gamma_phi3)
        A = self.beta / denom_A.clip(1e-20)

        t2, t4 = t**2, t**4
        f1 = t2 + A * t4

        # H term structure: gamma * phi^3 * log(1 + beta/gamma * f1 / (1 + A*f1))
        denom_f2 = self.gamma * (1.0 + A * f1)
        fraction = (self.beta * f1) / denom_f2.clip(1e-20)

        return gamma_phi3 * jnp.log1p(fraction)

    def _pbe_inputs(self, density, grad_density):
        """Compute reduced density variables for PBE correlation."""

        if _has_spin(density):

            rho_up, rho_dn = jnp.clip(density, 1e-20)
            rho_tot = rho_up + rho_dn

            # Zeta: Clip inside (-1, 1) to prevent log divergences in derivatives
            zeta = (rho_up - rho_dn) / rho_tot
            zeta = jnp.clip(zeta, -1.0 + 1e-7, 1.0 - 1e-7)

            # Wigner-Seitz radius: rs = (3 / (4*pi*n))^(1/3)
            rs = (3 / (4 * jnp.pi * rho_tot) + 1e-20) ** (1 / 3)

            # Phi (Spin scaling)
            phi = ((1.0 + zeta) ** (2 / 3) + (1.0 - zeta) ** (2 / 3)) / 2
            phi = jnp.clip(phi, 1e-20)

            grad_density = jnp.sum(grad_density, axis=-1)

        else:
            rho_tot = jnp.clip(density, 1e-20)
            rs = (3 / (4 * jnp.pi * rho_tot) + 1e-20) ** (1 / 3)
            zeta, phi = 0.0, 1.0

        kf = (3 * jnp.pi**2 * rho_tot + 1e-20) ** (1 / 3)
        ks = jnp.sqrt(4 * kf / jnp.pi)

        # t = |grad n| / (2 * phi * ks * n)
        grad_norm = jnp.sqrt(jnp.sum(grad_density**2, axis=-1) + 1e-30)
        denom_t = 2 * phi * ks * rho_tot
        t = grad_norm / denom_t.clip(1e-20)

        return rs, zeta, phi, t

    def __call__(self, density: ScalarFeature, grad_density: VectorFeature) -> Scalar:
        """Calculate PBE correlation energy density."""

        rs, zeta, phi, t = self._pbe_inputs(density, grad_density)
        eps_pw = self._pw92_correlation(rs, zeta)
        H = self._pbe_correction(eps_pw, phi, t)

        return eps_pw + H


class PBE(XCFunctional):
    """Convenience wrapper for the full PBE XC functional."""
    name: ClassVar[str] = "PBE"
    xc_type: ClassVar[str] = "GGA"
    exchange: ClassVar[DensityFunctional] = PBEExchange()
    correlation: ClassVar[DensityFunctional] = PBECorrelation()


# ---------------------------------------------------------------------------
#  B88 exchange  (Becke, Phys. Rev. A 38, 3098 (1988))
# ---------------------------------------------------------------------------


class B88Exchange(DensityFunctional):
    """Becke 1988 GGA exchange.

    Per spin channel:
        ε_x^{B88,σ}(ρ_σ, |∇ρ_σ|) =
            -C_σ ρ_σ^{1/3}
            - β ρ_σ^{1/3} x_σ^2 / (1 + 6 β x_σ sinh^{-1}(x_σ))
    with x_σ = |∇ρ_σ| / ρ_σ^{4/3}, β = 0.0042, C_σ = (3/4)(6/π)^{1/3}.
    """

    name: ClassVar[str] = "B88"
    xc_type: ClassVar[str] = "GGA"

    beta: ClassVar[float] = 0.0042

    @staticmethod
    def _per_spin(rho_sigma: ScalarFeature, grad_norm_sigma: ScalarFeature) -> Scalar:
        rho = jnp.clip(rho_sigma, 1e-20)
        rho13 = rho ** (1 / 3)
        C_sigma = (3 / 4) * (6 / jnp.pi) ** (1 / 3)
        eps_lda = -C_sigma * rho13

        x = grad_norm_sigma / jnp.clip(rho ** (4 / 3), 1e-20)
        denom = 1.0 + 6 * B88Exchange.beta * x * jnp.arcsinh(x)
        correction = -B88Exchange.beta * rho13 * x**2 / jnp.clip(denom, 1e-20)
        return eps_lda + correction

    def __call__(self, density: ScalarFeature, grad_density: VectorFeature) -> Scalar:
        if _has_spin(density):
            n_up, n_dn = jnp.unstack(density, axis=-1)
            grad_up, grad_dn = jnp.unstack(grad_density, axis=-1)
            gn_up = jnp.linalg.norm(grad_up + 1e-30, axis=-1)
            gn_dn = jnp.linalg.norm(grad_dn + 1e-30, axis=-1)
            n_tot = jnp.clip(n_up + n_dn, 1e-20)
            eps_up = self._per_spin(n_up, gn_up)
            eps_dn = self._per_spin(n_dn, gn_dn)
            return (n_up / n_tot) * eps_up + (n_dn / n_tot) * eps_dn
        grad_norm = jnp.linalg.norm(grad_density + 1e-30, axis=-1)
        # Closed shell: ρ_σ = ρ/2, |∇ρ_σ| = |∇ρ|/2
        return self._per_spin(density / 2, grad_norm / 2)


# ---------------------------------------------------------------------------
#  VWN5 correlation  (Vosko-Wilk-Nusair, Can. J. Phys. 58, 1200 (1980), fit V)
# ---------------------------------------------------------------------------


def _vwn5_branch(x, A, b, c, x0):
    """VWN5 branch function (paramagnetic, ferromagnetic, or spin stiffness).

    ε_c = A [ ln(x²/X(x)) + 2b/Q · arctan(Q/(2x+b))
              - b x₀/X(x₀) · ( ln((x-x₀)²/X(x)) + 2(b+2x₀)/Q · arctan(Q/(2x+b)) ) ]
    with X(x) = x² + b x + c, Q = sqrt(4c - b²), x = √r_s.
    """
    X_x = x * x + b * x + c
    X_x0 = x0 * x0 + b * x0 + c
    Q = jnp.sqrt(4.0 * c - b * b)
    Q2x_b = Q / jnp.clip(2.0 * x + b, 1e-20)
    atan_term = jnp.arctan(Q2x_b)
    term1 = jnp.log(x * x / jnp.clip(X_x, 1e-20))
    term2 = (2.0 * b / Q) * atan_term
    term3 = -(b * x0 / X_x0) * (
        jnp.log((x - x0) ** 2 / jnp.clip(X_x, 1e-20))
        + (2.0 * (b + 2.0 * x0) / Q) * atan_term
    )
    return A * (term1 + term2 + term3)


class VWN5Correlation(DensityFunctional):
    """VWN5 LDA correlation (parameterization V from VWN 1980)."""

    name: ClassVar[str] = "VWN5"
    xc_type: ClassVar[str] = "LDA"

    # Paramagnetic parameters (in Hartree).
    A_p: ClassVar[float] = 0.0310907
    b_p: ClassVar[float] = 3.72744
    c_p: ClassVar[float] = 12.9352
    x0_p: ClassVar[float] = -0.10498
    # Ferromagnetic
    A_f: ClassVar[float] = 0.01554535
    b_f: ClassVar[float] = 7.06042
    c_f: ClassVar[float] = 18.0578
    x0_f: ClassVar[float] = -0.32500
    # Spin stiffness (alpha_c)
    A_a: ClassVar[float] = -1.0 / (6.0 * jnp.pi ** 2)  # = -0.01688685
    b_a: ClassVar[float] = 1.13107
    c_a: ClassVar[float] = 13.0045
    x0_a: ClassVar[float] = -0.0047584

    # Spin interpolation
    fz20: ClassVar[float] = 1.709920934161365617563962776245

    def _unpolarized(self, density: ScalarFeature) -> Scalar:
        rho = jnp.clip(density, 1e-20)
        r_s = (3.0 / (4.0 * jnp.pi * rho)) ** (1 / 3)
        x = jnp.sqrt(r_s)
        return _vwn5_branch(x, self.A_p, self.b_p, self.c_p, self.x0_p)

    def __call__(self, density: ScalarFeature, *args) -> Scalar:
        if _has_spin(density):
            n_up, n_dn = jnp.unstack(density, axis=-1)
            n_tot = jnp.clip(n_up + n_dn, 1e-20)
            zeta = jnp.clip((n_up - n_dn) / n_tot, -1.0 + 1e-7, 1.0 - 1e-7)
            r_s = (3.0 / (4.0 * jnp.pi * n_tot)) ** (1 / 3)
            x = jnp.sqrt(r_s)
            eps_p = _vwn5_branch(x, self.A_p, self.b_p, self.c_p, self.x0_p)
            eps_f = _vwn5_branch(x, self.A_f, self.b_f, self.c_f, self.x0_f)
            eps_a = _vwn5_branch(x, self.A_a, self.b_a, self.c_a, self.x0_a)
            f_z = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2.0) / (
                2.0 ** (4 / 3) - 2.0
            )
            z4 = zeta ** 4
            return (
                eps_p
                + eps_a * f_z * (1.0 - z4) / self.fz20
                + (eps_f - eps_p) * f_z * z4
            )
        return self._unpolarized(density)


class VWN_RPACorrelation(DensityFunctional):
    """VWN_RPA (parameterization III / RPA) LDA correlation.

    Same closed form as VWN5 but with different paramagnetic parameters.
    This is what libxc calls ``vwn_rpa`` (aka VWN3) and is the correlation
    component of Gaussian/libxc's default ``b3lyp`` functional.
    """

    name: ClassVar[str] = "VWN_RPA"
    xc_type: ClassVar[str] = "LDA"

    # Paramagnetic (VWN RPA, table 5 parameters).
    A_p: ClassVar[float] = 0.0310907
    b_p: ClassVar[float] = 13.0720
    c_p: ClassVar[float] = 42.7198
    x0_p: ClassVar[float] = -0.409286
    # Ferromagnetic (VWN RPA)
    A_f: ClassVar[float] = 0.01554535
    b_f: ClassVar[float] = 20.1231
    c_f: ClassVar[float] = 101.578
    x0_f: ClassVar[float] = -0.743294

    def _unpolarized(self, density: ScalarFeature) -> Scalar:
        rho = jnp.clip(density, 1e-20)
        r_s = (3.0 / (4.0 * jnp.pi * rho)) ** (1 / 3)
        x = jnp.sqrt(r_s)
        return _vwn5_branch(x, self.A_p, self.b_p, self.c_p, self.x0_p)

    def __call__(self, density: ScalarFeature, *args) -> Scalar:
        if _has_spin(density):
            # VWN3 (RPA) spin interpolation: ε_p + (ε_f − ε_p) f(ζ). Unlike VWN5,
            # the RPA form uses the simple para→ferro interpolation with no
            # spin-stiffness term (matches libxc LDA_C_VWN_RPA, B3LYP's correlation).
            n_up, n_dn = jnp.unstack(density, axis=-1)
            n_tot = jnp.clip(n_up + n_dn, 1e-20)
            zeta = jnp.clip((n_up - n_dn) / n_tot, -1.0 + 1e-7, 1.0 - 1e-7)
            x = jnp.sqrt((3.0 / (4.0 * jnp.pi * n_tot)) ** (1 / 3))
            eps_p = _vwn5_branch(x, self.A_p, self.b_p, self.c_p, self.x0_p)
            eps_f = _vwn5_branch(x, self.A_f, self.b_f, self.c_f, self.x0_f)
            f_z = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2.0) / (
                2.0 ** (4 / 3) - 2.0
            )
            return eps_p + (eps_f - eps_p) * f_z
        return self._unpolarized(density)


# ---------------------------------------------------------------------------
#  LYP correlation  (Lee, Yang, Parr, PRB 37, 785 (1988); Miehlich form)
# ---------------------------------------------------------------------------


class LYPCorrelation(DensityFunctional):
    """LYP GGA correlation energy density.

    Closed-shell expression (ρ_↑ = ρ_↓ = ρ/2) from Miehlich et al. (1989):

        ε_c^LYP = -a · g(ρ)
                 - a·b·ω(ρ) · [ 2^{11/3} C_F · (ρ/2)^{8/3}/ρ
                                + (1/36) · (47 - 7 δ) · γ/ρ
                                - (5/4) · (γ_↑↑ + γ_↓↓) · (1/ρ)   (γ_σσ = |∇ρ_σ|² = γ/4)
                                - ... ]

    The exact algebraic form has been re-derived for ρ_↑ = ρ_↓ = ρ/2 and
    validated against ``pyscf.dft.libxc.eval_xc(',LYP', ...)``.
    """

    name: ClassVar[str] = "LYP"
    xc_type: ClassVar[str] = "GGA"

    a: ClassVar[float] = 0.04918
    b: ClassVar[float] = 0.132
    c: ClassVar[float] = 0.2533
    d: ClassVar[float] = 0.349
    # Thomas-Fermi kinetic coefficient: C_F = (3/10)(3π²)^(2/3)
    C_F: ClassVar[float] = 0.3 * (3.0 * jnp.pi ** 2) ** (2 / 3)

    @staticmethod
    def _closed_shell(rho: ScalarFeature, gamma: ScalarFeature) -> Scalar:
        """Closed-shell LYP energy density per electron.

        Derived for ρ_↑ = ρ_↓ = ρ/2 by substituting into the Miehlich 1989 form
        of LYP (|∇ρ_σ|² = γ/4, ρ_σ = ρ/2). After collecting terms:

            ε_c^LYP = -a·g - a·b·ω·ρ · ( C_F · ρ^{8/3}  -  γ·(3 + 7δ)/72 )

        g = 1/(1 + d ρ^{-1/3})
        ω = e^{-c ρ^{-1/3}} · g · ρ^{-11/3}
        δ = c ρ^{-1/3} + d ρ^{-1/3} · g
        """
        a, b, c, d = LYPCorrelation.a, LYPCorrelation.b, LYPCorrelation.c, LYPCorrelation.d
        C_F = LYPCorrelation.C_F

        rho = jnp.clip(rho, 1e-20)
        rho_inv_third = rho ** (-1 / 3)
        g = 1.0 / (1.0 + d * rho_inv_third)
        omega = jnp.exp(-c * rho_inv_third) * g * rho ** (-11 / 3)
        delta = c * rho_inv_third + d * rho_inv_third * g

        term_ff = C_F * rho ** (8 / 3) - gamma * (3.0 + 7.0 * delta) / 72.0
        eps_a = -a * g
        eps_b = -a * b * omega * rho * term_ff
        return eps_a + eps_b

    @staticmethod
    def _open_shell(n_up, n_dn, gaa, gbb, gab) -> Scalar:
        """Open-shell LYP energy density per electron (Miehlich et al. 1989).

        γ_σσ' = ∇ρ_σ·∇ρ_σ'; |∇ρ|² = γ_aa + 2γ_ab + γ_bb. Validated against
        ``pyscf.dft.libxc.eval_xc('GGA_C_LYP', …, spin=1)`` to machine precision.
        """
        a, b, c, d = LYPCorrelation.a, LYPCorrelation.b, LYPCorrelation.c, LYPCorrelation.d
        C_F = LYPCorrelation.C_F
        ra = jnp.clip(n_up, 1e-20)
        rb = jnp.clip(n_dn, 1e-20)
        rho = ra + rb
        g_tot = gaa + 2.0 * gab + gbb
        r13 = rho ** (-1 / 3)
        g = 1.0 / (1.0 + d * r13)
        omega = jnp.exp(-c * r13) * rho ** (-11 / 3) * g
        delta = c * r13 + d * r13 * g
        e1 = -a * 4.0 * ra * rb / rho * g
        e2 = -a * b * omega * (
            ra * rb * (
                2.0 ** (11 / 3) * C_F * (ra ** (8 / 3) + rb ** (8 / 3))
                + (47.0 / 18.0 - 7.0 * delta / 18.0) * g_tot
                - (5.0 / 2.0 - delta / 18.0) * (gaa + gbb)
                - (delta - 11.0) / 9.0 * (ra / rho * gaa + rb / rho * gbb)
            )
            - 2.0 / 3.0 * rho ** 2 * g_tot
            + (2.0 / 3.0 * rho ** 2 - ra ** 2) * gbb
            + (2.0 / 3.0 * rho ** 2 - rb ** 2) * gaa
        )
        return (e1 + e2) / rho

    def __call__(self, density: ScalarFeature, grad_density: VectorFeature) -> Scalar:
        if _has_spin(density):
            n_up, n_dn = jnp.unstack(density, axis=-1)
            ga, gb = grad_density[:, 0], grad_density[:, 1]     # ∇ρ_a, ∇ρ_b
            gaa = jnp.sum(ga * ga)
            gbb = jnp.sum(gb * gb)
            gab = jnp.sum(ga * gb)
            return self._open_shell(n_up, n_dn, gaa, gbb, gab)
        grad_norm_sq = jnp.sum(grad_density ** 2, axis=-1)
        return self._closed_shell(density, grad_norm_sq)


# ---------------------------------------------------------------------------
#  B3LYP  (Becke 1993 three-parameter exchange-correlation)
# ---------------------------------------------------------------------------


class B3LYP(XCFunctional):
    """Gaussian / libxc default B3LYP functional (uses VWN_RPA).

    Total XC energy density (per electron):
      ε_xc = 0.08 · ε_x^LDA + 0.72 · ε_x^B88 + 0.19 · ε_c^VWN_RPA + 0.81 · ε_c^LYP

    The HF exchange coefficient (0.20) is exposed as ``hf_coeff``. The
    ``EnergyFunctional`` is responsible for building the K matrix and adding
    -0.5 · hf_coeff · Tr(P K) to the total energy.
    """

    name: ClassVar[str] = "B3LYP"
    xc_type: ClassVar[str] = "GGA"
    hf_coeff: ClassVar[float] = 0.20

    ax: ClassVar[float] = 0.08   # Slater exchange mix
    bx: ClassVar[float] = 0.72   # B88 exchange mix
    ac: ClassVar[float] = 0.19   # VWN_RPA correlation mix
    bc: ClassVar[float] = 0.81   # LYP correlation mix

    # Stored as ClassVar so XCFunctional's interface is satisfied but only used
    # via the weighted __call__ below.
    exchange: ClassVar[DensityFunctional] = B88Exchange()
    correlation: ClassVar[DensityFunctional] = LYPCorrelation()

    _lda_x: ClassVar[DensityFunctional] = LDAExchange()
    _b88_x: ClassVar[DensityFunctional] = B88Exchange()
    _vwn_rpa: ClassVar[DensityFunctional] = VWN_RPACorrelation()
    _lyp: ClassVar[DensityFunctional] = LYPCorrelation()

    def __call__(self, density: ScalarFeature, grad_density: VectorFeature) -> Scalar:
        eps_lda_x = self._lda_x(density)
        eps_b88_x = self._b88_x(density, grad_density)
        eps_vwn = self._vwn_rpa(density)
        eps_lyp = self._lyp(density, grad_density)
        return (
            self.ax * eps_lda_x
            + self.bx * eps_b88_x
            + self.ac * eps_vwn
            + self.bc * eps_lyp
        )


# ---------------------------------------------------------------------------
#  LDA (Slater exchange + VWN5 correlation, "SVWN5") and PBE0 hybrid
# ---------------------------------------------------------------------------


class LDA(XCFunctional):
    """Local density approximation: Slater exchange + VWN5 correlation."""

    name: ClassVar[str] = "LDA"
    xc_type: ClassVar[str] = "LDA"
    exchange: ClassVar[DensityFunctional] = LDAExchange()
    correlation: ClassVar[DensityFunctional] = VWN5Correlation()


class PBE0(XCFunctional):
    """PBE0 global hybrid: ε_xc = 0.75·ε_x^PBE + ε_c^PBE, plus 25% exact exchange.

    The HF exchange fraction (0.25) is exposed as ``hf_coeff``; the exact-K
    contribution is added by the energy driver (see :mod:`dftax.ks.energy`).
    """

    name: ClassVar[str] = "PBE0"
    xc_type: ClassVar[str] = "GGA"
    hf_coeff: ClassVar[float] = 0.25

    exchange: ClassVar[DensityFunctional] = PBEExchange()
    correlation: ClassVar[DensityFunctional] = PBECorrelation()

    def __call__(self, density: ScalarFeature, grad_density: VectorFeature) -> Scalar:
        return 0.75 * self.exchange(density, grad_density) + self.correlation(
            density, grad_density
        )


# ---------------------------------------------------------------------------
#  Range-separated hybrids: erf attenuation, PW92, SR-B88 (ITYH), B97 series
# ---------------------------------------------------------------------------


def _erf_attenuation(a: Scalar) -> Scalar:
    """Short-range fraction ``F(a)`` of an erfc-attenuated exchange hole.

    ``F(a) = 1 - (8/3)·a·[√π·erf(1/2a) - 3a + 4a³ + (2a - 4a³)·e^{-1/(4a²)}]``
    (Gill/Adamson; the common kernel of SR-LDA exchange and the ITYH SR-GGA
    scheme). The direct form loses ~a⁶ digits to cancellation as a grows, so
    beyond ``a = 8`` the asymptote ``F → 1/(36a²)`` takes over (relative
    switch error ~1e-4 there, in a region reached only by masked ρ→0 points).
    Both branches are evaluated clamped to their own region so the unused
    branch cannot inject NaNs through autodiff (the boys.py double-``where``
    pattern).
    """
    a = jnp.clip(a, 1e-10)
    a_small = jnp.minimum(a, 8.0)
    direct = 1.0 - (8.0 / 3.0) * a_small * (
        jnp.sqrt(jnp.pi) * _jerf(1.0 / (2.0 * a_small))
        - 3.0 * a_small
        + 4.0 * a_small**3
        + (2.0 * a_small - 4.0 * a_small**3)
        * jnp.exp(-1.0 / (4.0 * a_small**2))
    )
    asym = 1.0 / (36.0 * jnp.maximum(a, 8.0) ** 2)
    return jnp.where(a < 8.0, direct, asym)


class PW92Correlation(DensityFunctional):
    """Perdew-Wang 1992 LSDA correlation (the B97 family's LSDA backbone).

    ``ε_c(r_s, ζ) = ε_0 - α_c·f(ζ)/f''(0)·(1-ζ⁴) + (ε_1-ε_0)·f(ζ)·ζ⁴`` with
    the three G-function fits of PW92 (the parameters fit −α_c, hence the
    sign). Validated against ``libxc LDA_C_PW`` to ~4e-10 (libxc carries one
    more digit on the A constants).
    """

    name: ClassVar[str] = "PW92"
    xc_type: ClassVar[str] = "LDA"

    @staticmethod
    def _G(rs, A, a1, b1, b2, b3, b4):
        srs = jnp.sqrt(rs)
        den = 2.0 * A * (b1 * srs + b2 * rs + b3 * rs * srs + b4 * rs * rs)
        return -2.0 * A * (1.0 + a1 * rs) * jnp.log1p(1.0 / jnp.clip(den, 1e-30))

    @staticmethod
    def eps(rho_a: Scalar, rho_b: Scalar) -> Scalar:
        """ε_c per electron for one (ρ_α, ρ_β) point."""
        ra = jnp.clip(rho_a, 0.0)
        rb = jnp.clip(rho_b, 0.0)
        rho = jnp.clip(ra + rb, 1e-20)
        zeta = jnp.clip((ra - rb) / rho, -1.0, 1.0)
        rs = (3.0 / (4.0 * jnp.pi * rho)) ** (1.0 / 3.0)
        ec0 = PW92Correlation._G(rs, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
        ec1 = PW92Correlation._G(rs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
        mac = PW92Correlation._G(rs, 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
        fz = ((1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0) - 2.0) / (
            2.0 ** (4.0 / 3.0) - 2.0
        )
        d2f0 = 4.0 / (9.0 * (2.0 ** (1.0 / 3.0) - 1.0))
        return ec0 - mac * fz * (1.0 - zeta**4) / d2f0 + (ec1 - ec0) * fz * zeta**4

    def __call__(self, density: ScalarFeature) -> Scalar:
        if _has_spin(density):
            n_up, n_dn = jnp.unstack(density, axis=-1)
            return self.eps(n_up, n_dn)
        return self.eps(density / 2.0, density / 2.0)


class ITYHB88Exchange(DensityFunctional):
    """Short-range B88 exchange via the ITYH scheme at ω = 0.33 (CAM-B3LYP).

    Per spin: ``ε_x^{sr} = ε_x^{B88}·F(a_σ)`` with the GGA-adapted momentum
    ``a_σ = ω·√K_σ / (6√π·ρ_σ^{1/3})``, ``K_σ = -2·ε_x^{B88}/ρ_σ^{1/3}``
    (Iikura-Tsuneda-Yanai-Hirao 2001). Matches ``libxc GGA_X_ITYH`` at
    ω = 0.33 to machine precision.
    """

    name: ClassVar[str] = "ITYH-B88"
    xc_type: ClassVar[str] = "GGA"
    omega: ClassVar[float] = 0.33

    @classmethod
    def _per_spin(cls, rho_s, grad_norm_s):
        rho_s = jnp.clip(rho_s, 1e-20)
        eps = B88Exchange._per_spin(rho_s, grad_norm_s)      # < 0
        K = jnp.clip(-2.0 * eps / rho_s ** (1.0 / 3.0), 1e-20)
        a = cls.omega * jnp.sqrt(K) / (6.0 * jnp.sqrt(jnp.pi) * rho_s ** (1.0 / 3.0))
        return eps * _erf_attenuation(a)

    def __call__(self, density: ScalarFeature, grad_density: VectorFeature) -> Scalar:
        if _has_spin(density):
            n_up, n_dn = jnp.unstack(density, axis=-1)
            grad_up, grad_dn = jnp.unstack(grad_density, axis=-1)
            gn_up = jnp.linalg.norm(grad_up + 1e-30, axis=-1)
            gn_dn = jnp.linalg.norm(grad_dn + 1e-30, axis=-1)
            n_tot = jnp.clip(n_up + n_dn, 1e-20)
            return (n_up / n_tot) * self._per_spin(n_up, gn_up) + (
                n_dn / n_tot
            ) * self._per_spin(n_dn, gn_dn)
        grad_norm = jnp.linalg.norm(grad_density + 1e-30, axis=-1)
        return self._per_spin(density / 2.0, grad_norm / 2.0)


class CAMB3LYP(XCFunctional):
    """CAM-B3LYP (Yanai-Tew-Handy 2004) range-separated hybrid.

    DFT part (identified against libxc ``HYB_GGA_XC_CAM_B3LYP`` to machine
    precision): ``0.35·B88 + 0.46·SR-B88(ITYH, ω=0.33) + 0.81·LYP +
    0.19·VWN5``. Exact exchange: ``0.19·K + 0.46·K_lr(ω=0.33)`` (PySCF
    ``rsh_coeff`` convention (0.33, 0.65, -0.46) re-expressed as
    full-range + long-range).
    """

    name: ClassVar[str] = "CAM-B3LYP"
    xc_type: ClassVar[str] = "GGA"
    hf_coeff: ClassVar[float] = 0.19
    hf_coeff_lr: ClassVar[float] = 0.46
    omega: ClassVar[float] = 0.33

    exchange: ClassVar[DensityFunctional] = B88Exchange()
    correlation: ClassVar[DensityFunctional] = LYPCorrelation()

    _b88: ClassVar[DensityFunctional] = B88Exchange()
    _sr_b88: ClassVar[DensityFunctional] = ITYHB88Exchange()
    _lyp: ClassVar[DensityFunctional] = LYPCorrelation()
    _vwn5: ClassVar[DensityFunctional] = VWN5Correlation()

    def __call__(self, density: ScalarFeature, grad_density: VectorFeature) -> Scalar:
        return (
            0.35 * self._b88(density, grad_density)
            + 0.46 * self._sr_b88(density, grad_density)
            + 0.81 * self._lyp(density, grad_density)
            + 0.19 * self._vwn5(density)
        )


class _B97Pieces:
    """Shared machinery for the wB97X power series (Chai-Head-Gordon 2008).

    Every piece is an *energy density* per spin assembled from
    ``u = γ·s²/(1+γ·s²)`` enhancement series over the SR-LDA exchange and the
    Stoll-partitioned PW92 correlation. The series coefficients below were
    recovered from ``libxc HYB_GGA_XC_WB97X`` by exact linear fit
    (residual ~1e-10) and equal the published wB97X parameters.
    """

    GAMMA_X: ClassVar[float] = 0.004
    GAMMA_SS: ClassVar[float] = 0.2
    GAMMA_AB: ClassVar[float] = 0.006

    @staticmethod
    def u(s2, gamma):
        return gamma * s2 / (1.0 + gamma * s2)

    @staticmethod
    def series(u, coeffs):
        acc = coeffs[0] * jnp.ones_like(u)
        up = u
        for c in coeffs[1:]:
            acc = acc + c * up
            up = up * u
        return acc

    @staticmethod
    def ex_sr_lda_density(rho_s, omega):
        """SR-LDA exchange energy *density* of one spin channel."""
        rho_s = jnp.clip(rho_s, 1e-20)
        kF = (6.0 * jnp.pi**2 * rho_s) ** (1.0 / 3.0)
        C_sigma = (3.0 / 4.0) * (6.0 / jnp.pi) ** (1.0 / 3.0)
        return rho_s * (-C_sigma * rho_s ** (1.0 / 3.0)) * _erf_attenuation(
            omega / (2.0 * kF)
        )


class WB97X(XCFunctional):
    """ωB97X (Chai-Head-Gordon 2008) range-separated hybrid GGA.

    Exchange: B97 series over SR-LDA (ω = 0.3); correlation: B97 series over
    Stoll-partitioned PW92 (same-spin / opposite-spin). Exact exchange:
    ``0.157706·K + 0.842294·K_lr(ω=0.3)``. The series coefficients match
    libxc ``HYB_GGA_XC_WB97X`` (recovered by exact linear fit, residual
    ~1e-10) and the published parameters.
    """

    name: ClassVar[str] = "wB97X"
    xc_type: ClassVar[str] = "GGA"
    hf_coeff: ClassVar[float] = 0.157706
    hf_coeff_lr: ClassVar[float] = 0.842294
    omega: ClassVar[float] = 0.3

    CX: ClassVar[tuple] = (0.842294, 0.726479, 1.04760, -5.70635, 13.2794)
    CSS: ClassVar[tuple] = (1.0, -4.33879, 18.2308, -31.7430, 17.2901)
    CAB: ClassVar[tuple] = (1.0, 2.37031, -11.3995, 6.58405, -3.78132)

    # Interface stubs (assembly happens in the weighted __call__ below).
    exchange: ClassVar[DensityFunctional] = LDAExchange()
    correlation: ClassVar[DensityFunctional] = PW92Correlation()

    @classmethod
    def _energy_density(cls, rho_a, rho_b, gn_a, gn_b):
        """Total XC energy density from per-spin (ρ, |∇ρ|)."""
        P = _B97Pieces
        rho_a = jnp.clip(rho_a, 1e-20)
        rho_b = jnp.clip(rho_b, 1e-20)
        s2a = (gn_a / rho_a ** (4.0 / 3.0)) ** 2
        s2b = (gn_b / rho_b ** (4.0 / 3.0)) ** 2

        ex = P.ex_sr_lda_density(rho_a, cls.omega) * P.series(
            P.u(s2a, P.GAMMA_X), cls.CX
        ) + P.ex_sr_lda_density(rho_b, cls.omega) * P.series(
            P.u(s2b, P.GAMMA_X), cls.CX
        )

        # Stoll partition of PW92: e_c^{σσ} = e_c(ρ_σ, 0); the αβ part is the
        # remainder. Energy densities (ρ·ε).
        ec_aa = rho_a * PW92Correlation.eps(rho_a, jnp.zeros_like(rho_a))
        ec_bb = rho_b * PW92Correlation.eps(rho_b, jnp.zeros_like(rho_b))
        ec_tot = (rho_a + rho_b) * PW92Correlation.eps(rho_a, rho_b)
        ec_ab = ec_tot - ec_aa - ec_bb

        css = ec_aa * P.series(P.u(s2a, P.GAMMA_SS), cls.CSS) + ec_bb * P.series(
            P.u(s2b, P.GAMMA_SS), cls.CSS
        )
        cab = ec_ab * P.series(P.u(0.5 * (s2a + s2b), P.GAMMA_AB), cls.CAB)
        return ex + css + cab

    def __call__(self, density: ScalarFeature, grad_density: VectorFeature) -> Scalar:
        if _has_spin(density):
            n_up, n_dn = jnp.unstack(density, axis=-1)
            grad_up, grad_dn = jnp.unstack(grad_density, axis=-1)
            gn_up = jnp.linalg.norm(grad_up + 1e-30, axis=-1)
            gn_dn = jnp.linalg.norm(grad_dn + 1e-30, axis=-1)
            e_dens = self._energy_density(n_up, n_dn, gn_up, gn_dn)
            return e_dens / jnp.clip(n_up + n_dn, 1e-20)
        gn = jnp.linalg.norm(grad_density + 1e-30, axis=-1)
        e_dens = self._energy_density(
            density / 2.0, density / 2.0, gn / 2.0, gn / 2.0
        )
        return e_dens / jnp.clip(density, 1e-20
        )


class WB97XV(WB97X):
    """ωB97X-V (Mardirossian-Head-Gordon 2014): 10-parameter RSH GGA + VV10.

    The B97 series is deliberately short (3 exchange, 2 + 2 correlation
    terms); nonlocal correlation is VV10 with ``b = 6.0``, ``C = 0.01``
    (declared via ``nlc_b``/``nlc_c``; the KS builder adds the double-grid
    term). Exact exchange ``0.167·K + 0.833·K_lr(ω = 0.3)``. Series
    coefficients verified pointwise against libxc ``HYB_GGA_XC_WB97X_V``.
    """

    name: ClassVar[str] = "wB97X-V"
    xc_type: ClassVar[str] = "GGA"
    hf_coeff: ClassVar[float] = 0.167
    hf_coeff_lr: ClassVar[float] = 0.833
    omega: ClassVar[float] = 0.3
    nlc_b: ClassVar[float] = 6.0
    nlc_c: ClassVar[float] = 0.01

    CX: ClassVar[tuple] = (0.833, 0.603, 1.194, 0.0, 0.0)
    CSS: ClassVar[tuple] = (0.556, -0.257, 0.0, 0.0, 0.0)
    CAB: ClassVar[tuple] = (1.219, -1.850, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
#  r2SCAN meta-GGA (Furness, Kaplan, Ning, Perdew, Sun, JPCL 11, 8208 (2020))
# ---------------------------------------------------------------------------
#
# Ported from the libxc maple sources (mgga_x_r2scan.mpl / mgga_c_r2scan.mpl
# and their SCAN/rSCAN includes); every branch validated against libxc via
# PySCF to machine precision (exchange ~4e-16, correlation ~1e-16, closed and
# spin-polarized). Note r2SCAN uses the *modified* PW92 constants.

_R2S_ETA = 0.001
_R2S_DP2 = 0.361
_R2S_K1 = 0.065
_R2S_H0X = 1.174
_R2S_A1 = 4.9479
_R2S_C1X, _R2S_C2X, _R2S_DXP = 0.667, 0.8, 1.24
_R2S_C1C, _R2S_C2C, _R2S_DCP = 0.64, 1.5, 0.7
# rSCAN switching polynomials, highest order first (c7 ... c0).
_R2S_FX = (-0.023185843322, 0.234528941479, -0.887998041597, 1.451297044490,
           -0.663086601049, -0.4445555, -0.667, 1.0)
_R2S_FC = (-0.051848879792, 0.516884468372, -1.915710236206, 3.061560252175,
           -1.535685604549, -0.4352, -0.64, 1.0)
_R2S_GAMMA = 0.031090690869654895  # (1 - ln 2)/pi^2
_R2S_B1C, _R2S_B2C, _R2S_B3C = 0.0285764, 0.0889, 0.125541
_R2S_CHI_INF = 0.12802585262625815
_R2S_G_CNST = 2.363
_X2S = 1.0 / (2.0 * (6.0 * jnp.pi**2) ** (1.0 / 3.0))
_XT2S = 1.0 / (2.0 * (3.0 * jnp.pi**2) ** (1.0 / 3.0))
_K_FACTOR_C = 0.3 * (6.0 * jnp.pi**2) ** (2.0 / 3.0)
_MU_GE = 10.0 / 81.0


def _r2scan_f_alpha(a, ff, c1, c2, d):
    """r2SCAN interpolation: exp (a<=0), 7th-order poly (0<a<=2.5), decay.

    Each branch is evaluated clamped to its own region (the boys.py
    double-``where`` pattern), so autodiff never sees the singular a -> 1
    of the unused exponential branches.
    """
    a_neg = jnp.minimum(a, 0.0)
    neg = jnp.exp(-c1 * a_neg / (1.0 - a_neg))
    small = jnp.polyval(jnp.asarray(ff), jnp.clip(a, 0.0, 2.5))
    a_big = jnp.maximum(a, 2.5 + 1e-12)
    large = -d * jnp.exp(c2 / (1.0 - a_big))
    return jnp.where(a <= 0.0, neg, jnp.where(a <= 2.5, small, large))


class R2SCANExchange(DensityFunctional):
    """r2SCAN exchange (spin-scaled per channel)."""

    name: ClassVar[str] = "r2SCAN-X"
    xc_type: ClassVar[str] = "MGGA"

    @staticmethod
    def _per_spin(rho_s, gn_s, tau_s):
        rho_s = jnp.clip(rho_s, 1e-20)
        xs = gn_s / rho_s ** (4.0 / 3.0)
        ts = jnp.clip(tau_s, 0.0) / rho_s ** (5.0 / 3.0)
        p = (_X2S * xs) ** 2
        alpha = (ts - xs * xs / 8.0) / (_K_FACTOR_C + _R2S_ETA * xs * xs / 8.0)

        Cn = 20.0 / 27.0 + _R2S_ETA * 5.0 / 3.0
        # C2 = -sum_i i*c_i * (1 - h0x) over the switching polynomial.
        idx = jnp.arange(1, 9)
        C2 = -jnp.sum(idx * jnp.asarray(_R2S_FX)[8 - idx]) * (1.0 - _R2S_H0X)
        y = (Cn * C2 * jnp.exp(-(p * p) / _R2S_DP2**4) + _MU_GE) * p
        h1x = 1.0 + _R2S_K1 * y / (_R2S_K1 + y)
        s = jnp.sqrt(jnp.clip(p, 1e-30))
        gx = -jnp.expm1(-_R2S_A1 / jnp.sqrt(s))
        fa = _r2scan_f_alpha(alpha, _R2S_FX, _R2S_C1X, _R2S_C2X, _R2S_DXP)
        F = (h1x + fa * (_R2S_H0X - h1x)) * gx
        eps_unif = (
            -(3.0 / 4.0) * (6.0 / jnp.pi) ** (1.0 / 3.0) * rho_s ** (1.0 / 3.0)
        )
        return eps_unif * F

    def __call__(self, density, grad_density, tau) -> Scalar:
        if _has_spin(density):
            n_up, n_dn = jnp.unstack(density, axis=-1)
            grad_up, grad_dn = jnp.unstack(grad_density, axis=-1)
            t_up, t_dn = jnp.unstack(tau, axis=-1)
            gn_up = jnp.linalg.norm(grad_up + 1e-30, axis=-1)
            gn_dn = jnp.linalg.norm(grad_dn + 1e-30, axis=-1)
            n_tot = jnp.clip(n_up + n_dn, 1e-20)
            return (n_up / n_tot) * self._per_spin(n_up, gn_up, t_up) + (
                n_dn / n_tot
            ) * self._per_spin(n_dn, gn_dn, t_dn)
        gn = jnp.linalg.norm(grad_density + 1e-30, axis=-1)
        return self._per_spin(density / 2.0, gn / 2.0, tau / 2.0)


class R2SCANCorrelation(DensityFunctional):
    """r2SCAN correlation (built on the modified-PW92 LSDA)."""

    name: ClassVar[str] = "r2SCAN-C"
    xc_type: ClassVar[str] = "MGGA"

    @staticmethod
    def _pw92_mod(rs, zeta):
        def G(rs, A, a1, b1, b2, b3, b4):
            srs = jnp.sqrt(rs)
            den = 2.0 * A * (b1 * srs + b2 * rs + b3 * rs * srs + b4 * rs * rs)
            return -2.0 * A * (1.0 + a1 * rs) * jnp.log1p(1.0 / jnp.clip(den, 1e-30))

        ec0 = G(rs, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
        ec1 = G(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
        mac = G(rs, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
        fz = (
            (1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0) - 2.0
        ) / (2.0 ** (4.0 / 3.0) - 2.0)
        d2f0 = 4.0 / (9.0 * (2.0 ** (1.0 / 3.0) - 1.0))
        return ec0 - mac * fz * (1.0 - zeta**4) / d2f0 + (ec1 - ec0) * fz * zeta**4

    @staticmethod
    def _eclda0(rs):
        return -_R2S_B1C / (1.0 + _R2S_B2C * jnp.sqrt(rs) + _R2S_B3C * rs)

    @staticmethod
    def _Gc(zeta):
        fz = (
            (1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0) - 2.0
        ) / (2.0 ** (4.0 / 3.0) - 2.0)
        return (1.0 - _R2S_G_CNST * (2.0 ** (1.0 / 3.0) - 1.0) * fz) * (
            1.0 - zeta**12
        )

    @staticmethod
    def _phi(zeta):
        return ((1.0 + zeta) ** (2.0 / 3.0) + (1.0 - zeta) ** (2.0 / 3.0)) / 2.0

    @classmethod
    def _ec0(cls, rs, zeta, s):
        one_minus_ginf = -jnp.expm1(
            -0.25 * jnp.log1p(4.0 * _R2S_CHI_INF * s * s)
        )
        H0 = _R2S_B1C * jnp.log1p(
            jnp.expm1(-cls._eclda0(rs) / _R2S_B1C) * one_minus_ginf
        )
        return (cls._eclda0(rs) + H0) * cls._Gc(zeta)

    @classmethod
    def _ec1(cls, rs, zeta, s, t):
        phi = cls._phi(zeta)
        eps_pw = cls._pw92_mod(rs, zeta)
        w1 = jnp.expm1(-eps_pw / (_R2S_GAMMA * phi**3))
        beta = 0.066724550603149220 * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs)
        y = beta * t * t / (_R2S_GAMMA * jnp.clip(w1, 1e-30))

        # Single-water-regime correction Delta-y (eq S34); the rs-derivatives
        # of the two LSDA limits come from autodiff.
        idx = jnp.arange(1, 8)
        dfc2 = jnp.sum(idx * jnp.asarray(_R2S_FC)[7 - idx])
        dz = ((1.0 + zeta) ** (5.0 / 3.0) + (1.0 - zeta) ** (5.0 / 3.0)) / 2.0
        elsda0_f = lambda r: cls._eclda0(r) * cls._Gc(zeta)
        elsda1_f = lambda r: cls._pw92_mod(r, zeta)
        delsda0 = jax.grad(elsda0_f)(rs)
        delsda1 = jax.grad(elsda1_f)(rs)
        dy = (
            dfc2
            / (27.0 * _R2S_GAMMA * dz * phi**3 * jnp.clip(w1, 1e-30))
            * (
                20.0 * rs * (delsda0 - delsda1)
                - 45.0 * _R2S_ETA * (elsda0_f(rs) - elsda1_f(rs))
            )
            * s
            * s
            * jnp.exp(-(s**4) / _R2S_DP2**4)
        )

        one_minus_g = -jnp.expm1(-0.25 * jnp.log1p(4.0 * (y - dy)))
        return eps_pw + _R2S_GAMMA * phi**3 * jnp.log1p(w1 * one_minus_g)

    @classmethod
    def _eps(cls, rho_a, rho_b, gn_tot, tau_a, tau_b):
        rho = jnp.clip(rho_a + rho_b, 1e-20)
        zeta = jnp.clip((rho_a - rho_b) / rho, -0.9999999999, 0.9999999999)
        rs = (3.0 / (4.0 * jnp.pi * rho)) ** (1.0 / 3.0)
        xt = gn_tot / rho ** (4.0 / 3.0)
        s = _XT2S * xt
        t = xt / (4.0 * 2.0 ** (1.0 / 3.0) * cls._phi(zeta) * jnp.sqrt(rs))
        ts0 = jnp.clip(tau_a, 0.0) / jnp.clip(rho_a, 1e-20) ** (5.0 / 3.0)
        ts1 = jnp.clip(tau_b, 0.0) / jnp.clip(rho_b, 1e-20) ** (5.0 / 3.0)

        def t_total(a, b):
            # (ts0 (1+z)^{5/3} + ts1 (1-z)^{5/3}) / 2^{5/3}: tau/rho^{5/3}
            return (
                (1.0 + zeta) ** (5.0 / 3.0) * a
                + (1.0 - zeta) ** (5.0 / 3.0) * b
            ) / 2.0 ** (5.0 / 3.0)

        alpha = (t_total(ts0, ts1) - xt * xt / 8.0) / (
            _K_FACTOR_C * t_total(1.0, 1.0) + _R2S_ETA * xt * xt / 8.0
        )
        fa = _r2scan_f_alpha(alpha, _R2S_FC, _R2S_C1C, _R2S_C2C, _R2S_DCP)
        e1 = cls._ec1(rs, zeta, s, t)
        return e1 + fa * (cls._ec0(rs, zeta, s) - e1)

    def __call__(self, density, grad_density, tau) -> Scalar:
        if _has_spin(density):
            n_up, n_dn = jnp.unstack(density, axis=-1)
            grad_up, grad_dn = jnp.unstack(grad_density, axis=-1)
            t_up, t_dn = jnp.unstack(tau, axis=-1)
            gn_tot = jnp.linalg.norm(grad_up + grad_dn + 1e-30, axis=-1)
            return self._eps(n_up, n_dn, gn_tot, t_up, t_dn)
        gn = jnp.linalg.norm(grad_density + 1e-30, axis=-1)
        return self._eps(density / 2.0, density / 2.0, gn, tau / 2.0, tau / 2.0)


class R2SCAN(XCFunctional):
    """r2SCAN meta-GGA (regularized-restored SCAN; the modern default mGGA).

    Needs the kinetic-energy density τ on the grid (``xc_type = "MGGA"``);
    the KS terms provide it and the Fock matrix comes from autodiff of the
    energy, so no vxc code exists anywhere. Validated pointwise against
    libxc ``mgga_x_r2scan``/``mgga_c_r2scan`` to machine precision.
    """

    name: ClassVar[str] = "r2SCAN"
    xc_type: ClassVar[str] = "MGGA"
    exchange: ClassVar[DensityFunctional] = R2SCANExchange()
    correlation: ClassVar[DensityFunctional] = R2SCANCorrelation()

    def __call__(self, density, grad_density, tau) -> Scalar:
        return self.exchange(density, grad_density, tau) + self.correlation(
            density, grad_density, tau
        )
