Yes. I would organise your Python implementation into **six cases**, starting with the simplest calculation and gradually introducing the probability distributions.

Your attached document explains why the joint generalized-Gamma distribution needs to be connected to a physical model before it is used for fog–aerosol optics. In particular, it distinguishes a joint probability density from a particle number-size distribution and explains why ordinary extinction and backscatter contributions from two separate populations can be added. 

The code below follows that approach. It uses the official `miepython` interface for Mie efficiencies and SciPy for numerical integration and special functions. ([miepython.readthedocs.io][1])

# 1. The six cases the program will cover

| Case  | What you want to calculate                                 | What you need                                                                            |
| ----- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **A** | \(Q_{\mathrm{ext}}\) and \(Q_b\) for one particle          | Diameter, wavelength and refractive index                                                |
| **B** | Efficiencies for a range of particle diameters             | A diameter grid and optical properties                                                   |
| **C** | Bulk extinction and backscatter for one population         | Its number-size distribution                                                             |
| **D** | Bulk optics from a generalized-Gamma marginal distribution | The distribution parameters and total number concentration                               |
| **E** | Total optics from separate fog and aerosol populations     | Separate distributions for fog and aerosol                                               |
| **F** | A joint generalized-Gamma model                            | The joint distribution parameters and a physically meaningful paired-size interpretation |

Cases A–E will cover most of the ordinary Mie-scattering calculations. Case F is for when you actually need the joint distribution.

One important clarification: **a joint probability distribution does not directly define a unique combined efficiency \(Q_{\mathrm{ext}}\) or \(Q_b\).** Those are single-particle quantities. For a population or a pair of particles, you generally need to calculate cross-sections and then integrate them over the appropriate size distribution.

---

# 2. Before running the code

Open a terminal or Command Prompt and install the required libraries:

```bash
python -m pip install numpy scipy miepython matplotlib
```

The program below uses the current `miepython` 3.x API.

**Save all the code in the next section as `fog_aerosol_mie.py`.** You can then import its functions into another Python program or run it directly.

All lengths in this implementation are expressed in **metres**, including diameter and wavelength. That keeps the population-level optical coefficients in SI units.

The refractive-index convention follows `miepython`: use \(m=n-ik\) for a material with real refractive index \(n\) and nonnegative absorption index \(k\). See the [official `miepython` documentation](https://miepython.readthedocs.io/en/stable/02_efficiencies.html). ([miepython.readthedocs.io][2])

# 3. Complete Python program

```python
"""
FOG–AEROSOL MIE SCATTERING
==========================

Cases implemented
-----------------
A. Q_ext and Q_back for one homogeneous spherical particle.
B. Mie efficiencies over a diameter grid.
C. Bulk optical coefficients for a supplied number-size distribution.
D. Bulk optical coefficients using a generalized-Gamma marginal PDF.
E. Total optics for separate fog and aerosol populations.
F. Joint generalized-Gamma PDF and a paired-population calculation.

Units used throughout
---------------------
Diameter             : metres
Wavelength           : metres
Refractive index     : complex number, convention n - 1j*k
Number concentration : particles/m^3
Number-size density  : particles/m^4 (dN/dD)
Q_ext, Q_sca, Q_back : dimensionless
Cross-sections       : m^2
Extinction coefficient: m^-1

IMPORTANT
---------
For two distinct populations sharing the same atmospheric volume, calculate
their optical coefficients separately and add them. A joint PDF is not
normally necessary for this standard additive calculation.

If an aerosol nucleus is physically inside a fog droplet, a coated-sphere
scattering model may be required. This code does not implement coated spheres.
"""

import math

import numpy as np
import miepython as mie

from scipy.integrate import trapezoid
from scipy.special import gammaln, ive
from scipy.stats import gamma as gamma_distribution


# ============================================================
# A. SINGLE-PARTICLE MIE EFFICIENCIES
# ============================================================

def mie_efficiencies_one_particle(
    diameter_m,
    wavelength_m,
    refractive_index,
    n_env=1.0,
):
    """
    Calculate Q_ext, Q_sca, Q_back and g for one spherical particle.

    Parameters
    ----------
    diameter_m : float
        Diameter of the particle in metres.

    wavelength_m : float
        Vacuum wavelength in metres.

    refractive_index : complex
        Complex refractive index of the sphere.
        miepython convention: n - 1j*k.

    n_env : float
        Real refractive index of the surrounding medium.
        Default is 1.0 (air approximately).

    Returns
    -------
    Dictionary of dimensionless Mie efficiencies.
    """

    if diameter_m <= 0:
        raise ValueError("Diameter must be positive.")

    if wavelength_m <= 0:
        raise ValueError("Wavelength must be positive.")

    if n_env <= 0:
        raise ValueError("Environmental refractive index must be positive.")

    qext, qsca, qback, g = mie.efficiencies(
        refractive_index,
        diameter_m,
        wavelength_m,
        n_env=n_env,
    )

    return {
        "Q_ext": float(qext),
        "Q_sca": float(qsca),
        "Q_back": float(qback),
        "g": float(g),
    }


# ============================================================
# B. EFFICIENCIES OVER A RANGE OF DIAMETERS
# ============================================================

def mie_efficiencies_over_grid(
    diameter_m,
    wavelength_m,
    refractive_index,
    n_env=1.0,
):
    """
    Calculate Mie efficiencies for every diameter in a 1-D array.
    """

    D = np.asarray(diameter_m, dtype=float)

    if D.ndim != 1 or D.size < 2:
        raise ValueError("Diameters must be a 1-D array with at least 2 values.")

    if np.any(D <= 0) or not np.all(np.isfinite(D)):
        raise ValueError("Diameters must be positive and finite.")

    if not np.all(np.diff(D) > 0):
        raise ValueError("Diameters must be strictly increasing.")

    qext, qsca, qback, g = mie.efficiencies(
        refractive_index,
        D,
        wavelength_m,
        n_env=n_env,
    )

    return {
        "Q_ext": np.asarray(qext),
        "Q_sca": np.asarray(qsca),
        "Q_back": np.asarray(qback),
        "g": np.asarray(g),
    }


# ============================================================
# CONVERT EFFICIENCIES INTO CROSS-SECTIONS
# ============================================================

def cross_sections(diameter_m, qext, qsca, qback):
    """
    Convert dimensionless Mie efficiencies into cross-sections.

    C_ext  = pi * radius^2 * Q_ext
    C_sca  = pi * radius^2 * Q_sca
    C_back = pi * radius^2 * Q_back

    All cross-sections are in m^2.
    """

    D = np.asarray(diameter_m, dtype=float)
    geometric_area = np.pi * (D / 2.0)**2

    return {
        "C_ext_m2": geometric_area * np.asarray(qext),
        "C_sca_m2": geometric_area * np.asarray(qsca),
        "C_back_m2": geometric_area * np.asarray(qback),
    }


# ============================================================
# C. BULK OPTICS FROM A SUPPLIED NUMBER-SIZE DISTRIBUTION
# ============================================================

def population_optics_from_number_distribution(
    diameter_m,
    number_size_density_m4,
    wavelength_m,
    refractive_index,
    n_env=1.0,
):
    """
    Calculate bulk optical coefficients for one particle population.

    number_size_density_m4 means dN/dD:
        particles per m^3 of air per metre of particle diameter.

    Its units are m^-4.

    Extinction:
        alpha_ext = integral [n(D) * C_ext(D)] dD

    Differential backscatter convention:
        beta_back = integral [n(D) * C_back(D)/(4*pi)] dD

    The function also returns the alternative area-valued backscatter
    coefficient, integral [n(D)*C_back(D)] dD, to avoid convention ambiguity.
    """

    D = np.asarray(diameter_m, dtype=float)
    nD = np.asarray(number_size_density_m4, dtype=float)

    if D.ndim != 1 or nD.ndim != 1:
        raise ValueError("Diameter and number-size density must be 1-D arrays.")

    if D.size != nD.size or D.size < 2:
        raise ValueError("The arrays must have equal lengths of at least 2.")

    if np.any(D <= 0) or not np.all(np.isfinite(D)):
        raise ValueError("Diameters must be positive and finite.")

    if np.any(nD < 0) or not np.all(np.isfinite(nD)):
        raise ValueError("Number-size density must be nonnegative and finite.")

    if not np.all(np.diff(D) > 0):
        raise ValueError("Diameters must be strictly increasing.")

    # Calculate single-particle Mie efficiencies.
    q = mie_efficiencies_over_grid(
        D,
        wavelength_m,
        refractive_index,
        n_env,
    )

    # Convert efficiencies to single-particle cross-sections.
    C = cross_sections(
        D,
        q["Q_ext"],
        q["Q_sca"],
        q["Q_back"],
    )

    # Integrate over particle size.
    alpha_ext = trapezoid(nD * C["C_ext_m2"], x=D)

    alpha_sca = trapezoid(nD * C["C_sca_m2"], x=D)

    # miepython uses C_back = pi*r^2*Q_back.
    # The differential cross-section at 180 degrees is C_back/(4*pi).
    beta_back = trapezoid(
        nD * C["C_back_m2"] / (4.0 * np.pi),
        x=D,
    )

    # Alternative backscatter coefficient convention.
    backscatter_area_coefficient = trapezoid(
        nD * C["C_back_m2"],
        x=D,
    )

    # This should be close to the actual concentration if the diameter
    # grid captures nearly all of the size distribution.
    integrated_concentration = trapezoid(nD, x=D)

    return {
        "Q_ext": q["Q_ext"],
        "Q_sca": q["Q_sca"],
        "Q_back": q["Q_back"],
        "g": q["g"],
        "C_ext_m2": C["C_ext_m2"],
        "C_sca_m2": C["C_sca_m2"],
        "C_back_m2": C["C_back_m2"],
        "alpha_ext_m1": float(alpha_ext),
        "alpha_sca_m1": float(alpha_sca),
        "beta_back_m1_sr1": float(beta_back),
        "backscatter_area_coefficient_m1": float(
            backscatter_area_coefficient
        ),
        "integrated_number_concentration_m3": float(
            integrated_concentration
        ),
    }


# ============================================================
# D. GENERALIZED-GAMMA MARGINAL PDF
# ============================================================

def generalized_gamma_pdf(
    diameter_m,
    m_shape,
    eta,
    alpha,
):
    r"""
    Generalized-Gamma marginal used in the attached draft:

    f(D) = 2*eta*m^m/[Gamma(m)*alpha^m]
           * D^(2*eta*m - 1)
           * exp[-m*D^(2*eta)/alpha]

    The marginal PDF integrates to 1 over D in (0, infinity).

    Its parameters satisfy:
        E[D^(2*eta)] = alpha

    If D is measured in metres, alpha has units m^(2*eta).
    """

    if m_shape <= 0:
        raise ValueError("m_shape must be positive.")

    if eta <= 0:
        raise ValueError("eta must be positive.")

    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    D = np.asarray(diameter_m, dtype=float)

    if np.any(D <= 0):
        raise ValueError("Diameters must be strictly positive.")

    # Compute the PDF in the log domain to improve numerical stability.
    log_pdf = (
        math.log(2.0 * eta)
        + m_shape * math.log(m_shape)
        - gammaln(m_shape)
        - m_shape * math.log(alpha)
        + (2.0 * eta * m_shape - 1.0) * np.log(D)
        - (m_shape / alpha) * D**(2.0 * eta)
    )

    return np.exp(log_pdf)


def alpha_from_mean_diameter(
    mean_diameter_m,
    m_shape,
    eta,
):
    """
    Calculate alpha so that the generalized-Gamma distribution has a specified
    arithmetic mean diameter.

    This is a convenience for demonstrations. In research, fit the parameters
    to measurements or use verified parameters from your reference paper.
    """

    if mean_diameter_m <= 0 or m_shape <= 0 or eta <= 0:
        raise ValueError("Mean diameter, m_shape and eta must be positive.")

    log_alpha = (
        math.log(m_shape)
        + 2.0 * eta * (
            math.log(mean_diameter_m)
            + gammaln(m_shape)
            - gammaln(m_shape + 1.0 / (2.0 * eta))
        )
    )

    return math.exp(log_alpha)


def generalized_gamma_diameter_grid(
    m_shape,
    eta,
    alpha,
    n_points=1200,
    q_min=1e-6,
    q_max=1.0 - 1e-6,
):
    """
    Create a diameter grid that spans most of a generalized-Gamma marginal.

    The transformation Y = D^(2*eta) follows a Gamma distribution with
    shape m_shape and scale alpha/m_shape.

    A finite grid necessarily truncates tiny tails, so check the integrated
    number concentration after calculating the optical properties.
    """

    if not (0 < q_min < q_max < 1):
        raise ValueError("Require 0 < q_min < q_max < 1.")

    if n_points < 20:
        raise ValueError("Use at least 20 diameter points.")

    probabilities = np.linspace(q_min, q_max, n_points)

    transformed = gamma_distribution.ppf(
        probabilities,
        a=m_shape,
        scale=alpha / m_shape,
    )

    D = transformed**(1.0 / (2.0 * eta))

    return np.unique(D)


def population_optics_from_generalized_gamma(
    total_number_concentration_m3,
    m_shape,
    eta,
    alpha,
    wavelength_m,
    refractive_index,
    n_points=1200,
    n_env=1.0,
):
    """
    Use a generalized-Gamma marginal PDF to calculate population optics.

    A normalized PDF is multiplied by total number concentration:

        n(D) = N_total * f(D)

    Then n(D) is integrated against the Mie cross-sections.
    """

    if total_number_concentration_m3 < 0:
        raise ValueError("Number concentration cannot be negative.")

    D = generalized_gamma_diameter_grid(
        m_shape,
        eta,
        alpha,
        n_points=n_points,
    )

    pdf = generalized_gamma_pdf(D, m_shape, eta, alpha)

    number_size_density = total_number_concentration_m3 * pdf

    result = population_optics_from_number_distribution(
        D,
        number_size_density,
        wavelength_m,
        refractive_index,
        n_env=n_env,
    )

    result["diameter_m"] = D
    result["pdf_D_per_m"] = pdf
    result["number_size_density_m4"] = number_size_density
    result["specified_number_concentration_m3"] = (
        total_number_concentration_m3
    )

    return result


# ============================================================
# E. ADD THE OPTICS OF TWO SEPARATE POPULATIONS
# ============================================================

def add_population_optics(fog_result, aerosol_result):
    """
    Add optical coefficients from separate fog and aerosol populations.

    Use this for the ordinary mixture model in which fog droplets and aerosol
    particles are distinct particles that independently contribute to optics.

    No joint PDF is needed for this additive calculation.
    """

    return {
        "alpha_ext_total_m1": (
            fog_result["alpha_ext_m1"]
            + aerosol_result["alpha_ext_m1"]
        ),
        "alpha_sca_total_m1": (
            fog_result["alpha_sca_m1"]
            + aerosol_result["alpha_sca_m1"]
        ),
        "beta_back_total_m1_sr1": (
            fog_result["beta_back_m1_sr1"]
            + aerosol_result["beta_back_m1_sr1"]
        ),
        "backscatter_area_coefficient_total_m1": (
            fog_result["backscatter_area_coefficient_m1"]
            + aerosol_result["backscatter_area_coefficient_m1"]
        ),
    }


# ============================================================
# F. JOINT GENERALIZED-GAMMA PDF
# ============================================================

def joint_generalized_gamma_pdf(
    x1,
    x2,
    m_shape,
    eta,
    alpha1,
    alpha2,
    rho,
):
    r"""
    Joint generalized-Gamma PDF from the correlated-Gaussian construction.

    rho has the interpretation:
        rho = Corr(X1^(2*eta), X2^(2*eta))

    It is NOT generally Corr(X1, X2).

    Parameter conditions:
        m_shape >= 1/2
        eta > 0
        alpha1 > 0
        alpha2 > 0
        0 <= rho < 1

    The function uses the scaled modified Bessel function ive to reduce
    numerical overflow. The independent case rho=0 is handled directly as
    the product of the two marginal PDFs.

    For a 2-D grid, pass x1[:, None] and x2[None, :].
    """

    m = float(m_shape)
    eta = float(eta)
    alpha1 = float(alpha1)
    alpha2 = float(alpha2)
    rho = float(rho)

    if m < 0.5:
        raise ValueError("This model requires m_shape >= 0.5.")

    if eta <= 0 or alpha1 <= 0 or alpha2 <= 0:
        raise ValueError("eta, alpha1 and alpha2 must be positive.")

    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must satisfy 0 <= rho < 1.")

    X1, X2 = np.broadcast_arrays(
        np.asarray(x1, dtype=float),
        np.asarray(x2, dtype=float),
    )

    if np.any(X1 <= 0) or np.any(X2 <= 0):
        raise ValueError("Both variables must be strictly positive.")

    # Independence: exact product of marginals.
    if rho == 0.0:
        return (
            generalized_gamma_pdf(X1, m, eta, alpha1)
            * generalized_gamma_pdf(X2, m, eta, alpha2)
        )

    log_x1 = np.log(X1)
    log_x2 = np.log(X2)
    log_a1 = math.log(alpha1)
    log_a2 = math.log(alpha2)
    log_rho = math.log(rho)

    # Argument of the Bessel function:
    #
    # z = 2*m*sqrt(rho)/(1-rho)
    #     * (x1*x2)^eta / sqrt(alpha1*alpha2)
    #
    log_z = (
        math.log(2.0 * m)
        - math.log1p(-rho)
        + 0.5 * log_rho
        + eta * (log_x1 + log_x2)
        - 0.5 * (log_a1 + log_a2)
    )

    nu = m - 1.0

    # Work with flat arrays so scalar inputs and grids are both supported.
    log_z_flat = np.ravel(log_z)
    log_bessel_flat = np.empty_like(log_z_flat)

    # For small z:
    # I_(m-1)(z) approximately (z/2)^(m-1)/Gamma(m).
    small = log_z_flat < math.log(1e-4)

    log_bessel_flat[small] = (
        nu * (log_z_flat[small] - math.log(2.0))
        - gammaln(m)
    )

    # For larger z, use the exponentially scaled Bessel function:
    # ive(nu,z) = I_nu(z)*exp(-z) for positive real z.
    if np.any(~small):

        if np.any(log_z_flat[~small] > 700):
            raise FloatingPointError(
                "Bessel argument is too large. Check parameter units/ranges."
            )

        z = np.exp(log_z_flat[~small])
        scaled_bessel = ive(nu, z)

        if np.any(scaled_bessel <= 0) or not np.all(
            np.isfinite(scaled_bessel)
        ):
            raise FloatingPointError(
                "Bessel evaluation failed. Check parameters and size units."
            )

        log_bessel_flat[~small] = (
            np.log(scaled_bessel) + z
        )

    log_bessel = log_bessel_flat.reshape(log_z.shape)

    # Logarithm of the constant in the joint density.
    log_constant = (
        math.log(4.0)
        + 2.0 * math.log(eta)
        + (m + 1.0) * math.log(m)
        + 0.5 * (1.0 - m) * log_rho
        - math.log1p(-rho)
        - gammaln(m)
        - 0.5 * (m + 1.0) * (log_a1 + log_a2)
    )

    term1 = np.exp(2.0 * eta * log_x1 - log_a1)
    term2 = np.exp(2.0 * eta * log_x2 - log_a2)

    log_pdf = (
        log_constant
        + (eta * (m + 1.0) - 1.0) * (log_x1 + log_x2)
        - (m / (1.0 - rho)) * (term1 + term2)
        + log_bessel
    )

    return np.exp(log_pdf)


# ============================================================
# INTEGRATE A 2-D JOINT DISTRIBUTION
# ============================================================

def integrate_2d_grid(values, x1_grid, x2_grid):
    """
    Numerically integrate values[i,j] over x1 and x2.

    Rows correspond to x1; columns correspond to x2.
    """

    values = np.asarray(values, dtype=float)
    x1_grid = np.asarray(x1_grid, dtype=float)
    x2_grid = np.asarray(x2_grid, dtype=float)

    if values.shape != (x1_grid.size, x2_grid.size):
        raise ValueError("The response grid shape does not match the axes.")

    integral_over_x2 = trapezoid(
        values,
        x=x2_grid,
        axis=1,
    )

    return float(
        trapezoid(
            integral_over_x2,
            x=x1_grid,
        )
    )


def joint_pdf_expectation(
    x1_grid,
    x2_grid,
    response_grid,
    m_shape,
    eta,
    alpha1,
    alpha2,
    rho,
):
    """
    Calculate E[response(X1,X2)] numerically under the joint PDF.

    response_grid[i,j] must represent the response at
    (x1_grid[i], x2_grid[j]).

    Also returns the probability mass captured by the finite numerical grid.
    """

    x1_grid = np.asarray(x1_grid, dtype=float)
    x2_grid = np.asarray(x2_grid, dtype=float)
    response_grid = np.asarray(response_grid, dtype=float)

    J = joint_generalized_gamma_pdf(
        x1_grid[:, None],
        x2_grid[None, :],
        m_shape,
        eta,
        alpha1,
        alpha2,
        rho,
    )

    captured_mass = integrate_2d_grid(
        J,
        x1_grid,
        x2_grid,
    )

    expected_response = integrate_2d_grid(
        J * response_grid,
        x1_grid,
        x2_grid,
    )

    return {
        "captured_joint_probability_mass": captured_mass,
        "expected_response": expected_response,
    }


# ============================================================
# JOINT MODEL: A POPULATION OF PHYSICALLY MEANINGFUL PAIRS
# ============================================================

def paired_population_optics(
    diameter1_m,
    diameter2_m,
    m_shape,
    eta,
    alpha1,
    alpha2,
    rho,
    pair_number_concentration_m3,
    wavelength_m,
    refractive_index1,
    refractive_index2,
    n_env=1.0,
):
    """
    Special case: each counted entity consists of TWO DISTINCT homogeneous
    spheres with a meaningful joint size distribution.

    Assumptions:
    - Each pair contains one sphere of type 1 and one sphere of type 2.
    - Their cross-sections add.
    - The two spheres are optically independent.
    - The number concentration of pairs is known.

    This is NOT the function to use for an ordinary unpaired fog+aerosol mixture.

    The expected cross-section of an additive pair is:
        E[C_pair] = E[C_1(D_1)] + E[C_2(D_2)]

    Because of that linearity, rho does not change the exact expectation when
    the marginals are held fixed. A finite numerical grid can cause small
    numerical differences.
    """

    D1 = np.asarray(diameter1_m, dtype=float)
    D2 = np.asarray(diameter2_m, dtype=float)

    if pair_number_concentration_m3 < 0:
        raise ValueError("Pair concentration cannot be negative.")

    q1 = mie_efficiencies_over_grid(
        D1,
        wavelength_m,
        refractive_index1,
        n_env,
    )

    q2 = mie_efficiencies_over_grid(
        D2,
        wavelength_m,
        refractive_index2,
        n_env,
    )

    C1 = cross_sections(
        D1,
        q1["Q_ext"],
        q1["Q_sca"],
        q1["Q_back"],
    )

    C2 = cross_sections(
        D2,
        q2["Q_ext"],
        q2["Q_sca"],
        q2["Q_back"],
    )

    J = joint_generalized_gamma_pdf(
        D1[:, None],
        D2[None, :],
        m_shape,
        eta,
        alpha1,
        alpha2,
        rho,
    )

    # One pair contains two distinct spheres.
    Cext_pair = (
        C1["C_ext_m2"][:, None]
        + C2["C_ext_m2"][None, :]
    )

    Cback_pair = (
        C1["C_back_m2"][:, None]
        + C2["C_back_m2"][None, :]
    )

    captured_mass = integrate_2d_grid(J, D1, D2)

    mean_pair_Cext = integrate_2d_grid(
        J * Cext_pair,
        D1,
        D2,
    )

    mean_pair_Cback = integrate_2d_grid(
        J * Cback_pair,
        D1,
        D2,
    )

    Npairs = pair_number_concentration_m3

    return {
        "captured_joint_probability_mass": captured_mass,
        "mean_pair_Cext_m2": mean_pair_Cext,
        "mean_pair_Cback_m2": mean_pair_Cback,
        "alpha_ext_m1": Npairs * mean_pair_Cext,
        "beta_back_m1_sr1": (
            Npairs * mean_pair_Cback / (4.0 * np.pi)
        ),
        "backscatter_area_coefficient_m1": (
            Npairs * mean_pair_Cback
        ),
    }
```

The single-particle calls and cross-section conversions follow the documented `miepython` definitions. Its \(Q_{\mathrm{back}}\) convention uses \(C_{\mathrm{back}}=\pi r^2Q_{\mathrm{back}}\); the differential cross-section at \(180^\circ\) is \(C_{\mathrm{back}}/(4\pi)\). Make sure this convention matches your research paper. ([miepython.readthedocs.io][2])

---

# 4. How to run each case

The code above defines the functions. These examples show how to use them.

## Case A — Calculate \(Q_{\mathrm{ext}}\) and \(Q_b\) for one fog droplet

A wavelength of 532 nm is used here simply as an example. Replace it with your wavelength.

```python
from fog_aerosol_mie import mie_efficiencies_one_particle

wavelength = 532e-9       # 532 nm, expressed in metres
diameter = 10e-6          # 10 micrometres, expressed in metres

water_index = 1.333 - 0.0j  # Example only; replace with appropriate data

result = mie_efficiencies_one_particle(
    diameter_m=diameter,
    wavelength_m=wavelength,
    refractive_index=water_index,
)

print("Q_ext  =", result["Q_ext"])
print("Q_back =", result["Q_back"])
```

This calculates the efficiencies for **one homogeneous spherical particle**. It does not need a probability distribution.

For an aerosol particle, use its diameter and its own wavelength-appropriate complex refractive index.

## Case B — Calculate the efficiencies for many diameters

```python
import numpy as np

from fog_aerosol_mie import mie_efficiencies_over_grid

diameters = np.geomspace(0.05e-6, 30e-6, 500)

result = mie_efficiencies_over_grid(
    diameter_m=diameters,
    wavelength_m=532e-9,
    refractive_index=1.333 - 0.0j,
)

Q_ext = result["Q_ext"]
Q_back = result["Q_back"]

print("Number of calculated diameters:", len(diameters))
print("First Q_ext value:", Q_ext[0])
print("First Q_back value:", Q_back[0])
```

Here, `diameters` contains 500 different diameters. The code returns an efficiency value for each one.

You can plot `Q_ext` and `Q_back` against diameter using Matplotlib.

## Case C — Calculate bulk optics from a known number-size distribution

This is the important case if you already have experimental or literature data for the particle-size distribution.

Suppose your measured data provide:

* `diameters`: a sorted array of diameters in metres.
* `number_size_density`: the number-size distribution \(dN/dD\), in \(\mathrm m^{-4}\).

Then run:

```python
from fog_aerosol_mie import population_optics_from_number_distribution

result = population_optics_from_number_distribution(
    diameter_m=diameters,
    number_size_density_m4=number_size_density,
    wavelength_m=532e-9,
    refractive_index=1.333 - 0.0j,
)

print("Extinction coefficient =", result["alpha_ext_m1"], "m^-1")
print("Backscatter coefficient =", result["beta_back_m1_sr1"], "m^-1 sr^-1")
```

The extinction calculation is

$$
\alpha_{\mathrm{ext}}
=
\int n_D(D)\,\frac{\pi D^2}{4}\,
Q_{\mathrm{ext}}(D)\,dD.
$$

The function also returns the alternative backscatter coefficient using the area-valued \(C_{\mathrm{back}}\) convention. You should compare whichever quantity your research paper defines.

**Do not confuse a normalized probability density with a number-size distribution.** A normalized PDF integrates to one; a number-size distribution integrates to the actual particle concentration. Your draft explicitly distinguishes these quantities. 

## Case D — Calculate bulk optics using a generalized-Gamma marginal

Suppose you want to use the marginal generalized-Gamma model from your document.

First, calculate a scale parameter \(\alpha\) for an illustrative mean diameter. In real work, use parameters fitted to measured sizes or reported in your reference paper.

```python
from fog_aerosol_mie import (
    alpha_from_mean_diameter,
    population_optics_from_generalized_gamma,
)

m_shape = 2.0
eta = 1.0

mean_diameter = 8e-6  # Illustrative mean diameter: 8 micrometres

alpha = alpha_from_mean_diameter(
    mean_diameter_m=mean_diameter,
    m_shape=m_shape,
    eta=eta,
)

# Illustrative concentration: 100 particles/cm^3 = 100 million particles/m^3
fog_concentration = 100e6

fog_result = population_optics_from_generalized_gamma(
    total_number_concentration_m3=fog_concentration,
    m_shape=m_shape,
    eta=eta,
    alpha=alpha,
    wavelength_m=532e-9,
    refractive_index=1.333 - 0.0j,
)

print("Extinction coefficient:", fog_result["alpha_ext_m1"], "m^-1")
print("Backscatter coefficient:", fog_result["beta_back_m1_sr1"], "m^-1 sr^-1")
print(
    "Concentration captured by numerical grid:",
    fog_result["integrated_number_concentration_m3"],
    "m^-3",
)
```

The last printed quantity is a useful numerical check. The integration uses a finite diameter grid, whereas the theoretical generalized-Gamma distribution extends over all positive diameters. Increasing the grid's width and resolution helps you verify that the calculated coefficients converge.

The parameters `m_shape`, `eta` and `alpha` in this example are illustrative. **They are not validated parameters for your particular fog.**

---

# 5. The ordinary fog–aerosol mixture

This is the case I would implement first if your goal is to model fog droplets and aerosol particles as two separate populations sharing the same atmosphere.

Calculate each population independently.

```python
from fog_aerosol_mie import (
    population_optics_from_generalized_gamma,
    add_population_optics,
)

# ---------------- FOG ----------------
# Use the fog's actual fitted/reported parameters in real research.

fog_result = population_optics_from_generalized_gamma(
    total_number_concentration_m3=100e6,  # Illustrative only
    m_shape=2.0,
    eta=1.0,
    alpha=alpha,                         # From Case D
    wavelength_m=532e-9,
    refractive_index=1.333 - 0.0j,
)

# ---------------- AEROSOL ----------------
# These are independent illustrative parameters, not validated aerosol data.

aerosol_mean_diameter = 0.15e-6
aerosol_m = 2.0
aerosol_eta = 1.0

aerosol_alpha = alpha_from_mean_diameter(
    aerosol_mean_diameter,
    aerosol_m,
    aerosol_eta,
)

aerosol_result = population_optics_from_generalized_gamma(
    total_number_concentration_m3=1e10,  # Illustrative only
    m_shape=aerosol_m,
    eta=aerosol_eta,
    alpha=aerosol_alpha,
    wavelength_m=532e-9,
    refractive_index=1.50 - 0.01j,      # Illustrative only
)

# ---------------- TOTAL MIXTURE ----------------

mixture = add_population_optics(
    fog_result,
    aerosol_result,
)

print("Total extinction:", mixture["alpha_ext_total_m1"], "m^-1")
print(
    "Total differential backscatter:",
    mixture["beta_back_total_m1_sr1"],
    "m^-1 sr^-1",
)
```

The total extinction coefficient is simply

$$
\boxed{
\alpha_{\mathrm{ext,total}}
=
\alpha_{\mathrm{ext,fog}}
+
\alpha_{\mathrm{ext,aerosol}}
}
$$

and the corresponding backscatter coefficients are added using the same convention.

This calculation does **not** require you to specify a correlation parameter \(\rho\) between fog and aerosol diameters.

That is not an omission. It follows from the physical model: the two separate populations contribute additive optical cross-sections under the ordinary independent-scattering approximation.

---

# 6. The joint generalized-Gamma case

Now suppose your research specifically requires a joint generalized-Gamma model.

You can calculate its PDF using:

```python
from fog_aerosol_mie import joint_generalized_gamma_pdf

joint_density = joint_generalized_gamma_pdf(
    x1=8e-6,
    x2=0.15e-6,
    m_shape=2.0,
    eta=1.0,
    alpha1=alpha,
    alpha2=aerosol_alpha,
    rho=0.5,
)

print("Joint density:", joint_density)
```

The parameter \(\rho\) has the meaning specified by your model:

$$
\rho=\operatorname{Corr}(X_1^{2\eta},X_2^{2\eta}).
$$

It is not generally the Pearson correlation between \(X_1\) and \(X_2\).

For the independent case, set

```python
rho = 0.0
```

The function then returns the product of the two marginal PDFs exactly.

For a dependent case, such as \(\rho=0.5\), use that parameter only if the dependence is justified by your model or data.

**The joint PDF describes the probability density of a pair of size values. It does not tell you how many pairs exist per cubic metre.** A physical paired-population calculation also requires the concentration of those pairs.

## What if I want to integrate an optical response over the joint PDF?

The general calculation is

$$
E[G(D_1,D_2)]
=
\int_0^\infty\int_0^\infty
f_{D_1,D_2}(d_1,d_2)
G(d_1,d_2)\,dd_2\,dd_1.
$$

You must define \(G(D_1,D_2)\) according to the physical meaning of the pair.

For example, if each pair consists of two distinct spheres whose cross-sections add,

$$
G_{\mathrm{ext}}(D_1,D_2)
=
C_{\mathrm{ext},1}(D_1)+C_{\mathrm{ext},2}(D_2).
$$

In this case, the expected pair cross-section is independent of \(\rho\) when the two marginals are held fixed. The correlation affects the joint distribution of the pair, but it does not change the expectation of this additive response.

The `paired_population_optics()` function in the complete program implements that special case.

If, instead, \(D_1\) describes a dry aerosol nucleus and \(D_2\) describes the surrounding wet fog droplet, you may be modelling **one coated particle**, rather than two separate spheres. The homogeneous-sphere code above is not a valid substitute for a coated-sphere Mie calculation.

---

# 7. How to verify that your numerical results are reliable

Before using results in a paper, I recommend checking the following.

**Check 1: A single particle.** Confirm that the program returns finite \(Q_{\mathrm{ext}}\) and \(Q_b\) values for realistic sizes and wavelength.

**Check 2: The marginal PDF.** Integrate a generalized-Gamma PDF over a sufficiently broad diameter grid. Its integral should be close to one.

**Check 3: The independent joint distribution.** For \(\rho=0\), confirm that the joint PDF agrees with the product of the marginal PDFs.

**Check 4: The joint probability mass.** Integrate the joint PDF over your finite two-dimensional grid. A result below one indicates that the grid truncates probability or that the grid resolution is inadequate.

**Check 5: Grid convergence.** Increase the number of diameter points and widen the diameter ranges. The integrated optical coefficients should stop changing materially.

**Check 6: Backscatter convention.** Compare the exact definition of \(Q_b\) or \(Q_{\mathrm{back}}\) in your target paper with the `miepython` convention. Do not silently add or remove the factor of \(4\pi\). ([miepython.readthedocs.io][2])

I have not been able to execute the complete program in this turn's runtime, so I would not describe it as execution-tested. The function calls and equations follow the current documented interfaces, but the first run should be treated as a test run before its output is used scientifically.

## My recommendation

Start with **Case A**, then Case C, then the separate fog–aerosol mixture. Those will establish the core Mie calculation, the conversion from efficiencies to cross-sections, and the integration over particle populations.

Add the joint generalized-Gamma calculation only after you have established what the two random variables represent physically and why their dependence must be included. That is the same distinction your document reaches in its discussion of the practical LiDAR application. 

[1]: https://miepython.readthedocs.io/en/latest/miepython.html?utm_source=chatgpt.com "API for miepython package — miepython 3.3.0 documentation"
[2]: https://miepython.readthedocs.io/en/stable/02_efficiencies.html "Mie Scattering Efficiencies — miepython 3.3.0 documentation"
