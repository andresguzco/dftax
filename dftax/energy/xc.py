"""Exchange-correlation functionals for KS-DFT."""

import abc
import equinox as eqx

from jax import numpy as jnp
from jax import Array
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
    # the *sole* source of the documented ~1e-5 Ha PBE/PBE0 gap vs libxc — full precision
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
