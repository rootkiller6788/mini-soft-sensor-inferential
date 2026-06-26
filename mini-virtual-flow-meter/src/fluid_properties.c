/**
 * @file fluid_properties.c
 * @brief Fluid property model implementations
 *
 * Implements gas laws, viscosity models, density corrections,
 * compressibility factor, and oil property correlations.
 *
 * @module mini-virtual-flow-meter
 */

#include "fluid_properties.h"
#include <math.h>
#include <stdio.h>

/* ==========================================================================
 * L2: Gas Properties -- Ideal and Real Gas
 * ========================================================================== */

double gas_density_ideal(double pressure, double temperature,
                          double mol_weight)
{
    /*
     * Ideal gas law: PV = nRT
     * rho = m/V = P*M / (R*T)
     *
     * Assumes Z = 1 (ideal gas). Accurate to ~1% for air at STP,
     * but errors reach 5-10% for natural gas at pipeline pressures (>50 bar).
     */

    if (temperature <= 0.0 || mol_weight <= 0.0) return 0.0;
    if (pressure < 0.0) return 0.0;

    return pressure * mol_weight / (FLUID_R_GAS * temperature);
}

double gas_density_real(double pressure, double temperature,
                         double mol_weight, double z)
{
    /*
     * Real gas law: PV = ZnRT
     * rho = P*M / (Z*R*T)
     *
     * Z = PV/RT is the compressibility factor, accounting for
     * non-ideal behavior. Z < 1 at moderate pressures (attractive
     * forces dominate), Z > 1 at very high pressures (repulsive).
     */

    if (temperature <= 0.0 || mol_weight <= 0.0 || z <= 0.0) return 0.0;
    if (pressure < 0.0) return 0.0;

    return pressure * mol_weight / (z * FLUID_R_GAS * temperature);
}

/* Dranchuk-Abou-Kassem (DAK) correlation constants */
#define DAK_A1  0.3265
#define DAK_A2 -1.0700
#define DAK_A3 -0.5339
#define DAK_A4  0.01569
#define DAK_A5 -0.05165
#define DAK_A6  0.5475
#define DAK_A7 -0.7361
#define DAK_A8  0.1844
#define DAK_A9  0.1056
#define DAK_A10 0.6134
#define DAK_A11 0.7210

double gas_compressibility_dak(double Pr, double Tr, double tol, int max_iter)
{
    /*
     * Dranchuk-Abou-Kassem (1975) correlation for natural gas
     * compressibility factor Z.
     *
     * The DAK equation expresses Z as a function of reduced density rho_r:
     *   rho_r = 0.27 * Pr / (Z * Tr)
     *
     * Since Z depends on rho_r and rho_r depends on Z, this is an
     * implicit equation solved iteratively:
     *
     * 1. Guess Z0 = 1.0
     * 2. Compute rho_r = 0.27 * Pr / (Z0 * Tr)
     * 3. Compute Z from DAK equation using rho_r
     * 4. Update Z = Z_new
     * 5. Repeat until convergence
     *
     * DAK equation (11-parameter):
     * Z = 1 + (A1+A2/Tr+A3/Tr^3+A4/Tr^4+A5/Tr^5)*rho_r
     *       + (A6+A7/Tr+A8/Tr^2)*rho_r^2
     *       - A9*(A7/Tr+A8/Tr^2)*rho_r^5
     *       + A10*(1+A11*rho_r^2)*(rho_r^2/Tr^3)*exp(-A11*rho_r^2)
     */

    if (Pr < 0.0) return 1.0;  /* Negative pressure invalid */

    double Z = 1.0;  /* Start with ideal gas guess */
    double rho_r;

    int i;
    for (i = 0; i < max_iter; i++) {
        rho_r = 0.27 * Pr / (Z * Tr);

        double rho_r2 = rho_r * rho_r;
        double rho_r5 = rho_r2 * rho_r2 * rho_r;

        double Tr_inv  = 1.0 / Tr;
        double Tr_inv2 = Tr_inv * Tr_inv;
        double Tr_inv3 = Tr_inv2 * Tr_inv;
        double Tr_inv4 = Tr_inv3 * Tr_inv;
        double Tr_inv5 = Tr_inv4 * Tr_inv;

        /* Term 1: linear in rho_r */
        double T1_coeff = DAK_A1 + DAK_A2*Tr_inv + DAK_A3*Tr_inv3
                         + DAK_A4*Tr_inv4 + DAK_A5*Tr_inv5;
        double T1 = T1_coeff * rho_r;

        /* Term 2: quadratic in rho_r */
        double T2_coeff = DAK_A6 + DAK_A7*Tr_inv + DAK_A8*Tr_inv2;
        double T2 = T2_coeff * rho_r2;

        /* Term 3: quintic in rho_r (negative) */
        double T3_coeff = DAK_A9 * (DAK_A7*Tr_inv + DAK_A8*Tr_inv2);
        double T3 = -T3_coeff * rho_r5;

        /* Term 4: exponential term */
        double exp_arg = -DAK_A11 * rho_r2;
        double exp_term = exp(exp_arg);
        double T4 = DAK_A10 * (1.0 + DAK_A11*rho_r2) * (rho_r2 * Tr_inv3) * exp_term;

        double Z_new = 1.0 + T1 + T2 + T3 + T4;

        if (fabs(Z_new - Z) < tol) {
            return Z_new;
        }
        Z = Z_new;

        if (Z <= 0.01) {
            /* Diverging — return best estimate */
            return Z;
        }
    }

    /* Non-convergence — return current estimate */
    return Z;
}

void gas_pseudocritical_sutton(double sg, double *Ppc, double *Tpc)
{
    /*
     * Sutton's (1985) correlation for natural gas pseudocritical
     * properties from specific gravity.
     *
     * Ppc [psia] = 756.8 - 131.0*sg - 3.6*sg^2
     * Tpc [R]    = 169.2 + 349.5*sg - 74.0*sg^2
     *
     * Reference: Sutton, R.P. (1985) "Compressibility Factors for
     * High-Molecular-Weight Reservoir Gases", SPE 14274.
     *
     * Convert to SI: Ppc [Pa] = Ppc [psia] * 6894.757
     *               Tpc [K]  = Tpc [R] / 1.8
     */

    if (!Ppc || !Tpc) return;
    if (sg <= 0.0) sg = 0.6;  /* Typical natural gas */

    double sg2 = sg * sg;
    double Ppc_psia = 756.8 - 131.0*sg - 3.6*sg2;
    double Tpc_R    = 169.2 + 349.5*sg - 74.0*sg2;

    *Ppc = Ppc_psia * 6894.757;
    *Tpc = Tpc_R / 1.8;

    /* Sanity checks */
    if (*Ppc < 1.0e5) *Ppc = 4.6e6;  /* ~670 psia default */
    if (*Tpc < 100.0) *Tpc = 210.0;  /* ~378 R default */
}

double gas_specific_gravity(double mol_weight)
{
    if (mol_weight <= 0.0) return 0.0;
    return mol_weight / FLUID_MW_AIR;
}

/* ==========================================================================
 * L3: Viscosity Models
 * ========================================================================== */

double gas_viscosity_sutherland(double temperature, double mu0,
                                 double T0, double S)
{
    /*
     * Sutherland's formula (1893):
     * mu = mu0 * (T/T0)^(3/2) * (T0 + S) / (T + S)
     *
     * Based on kinetic theory of gases. The (T/T0)^(3/2) term comes
     * from the mean molecular velocity, and the (T0+S)/(T+S) term
     * accounts for the temperature dependence of the collision cross-section.
     *
     * For dry air: mu0 = 1.716e-5 Pa*s, T0 = 273.15 K, S = 110.4 K
     * For natural gas (methane): mu0 = 1.03e-5, T0 = 273.15, S = 162 K
     */

    if (temperature <= 0.0 || mu0 <= 0.0 || T0 <= 0.0) return 0.0;

    double ratio = temperature / T0;
    return mu0 * pow(ratio, 1.5) * (T0 + S) / (temperature + S);
}

double liquid_viscosity_andrade(double temperature, double A, double B)
{
    /*
     * Andrade (Arrhenius-type) correlation for liquid viscosity:
     * mu = A * exp(B / T)
     *
     * Based on the idea that viscosity is a thermally activated process.
     * A: pre-exponential factor [Pa*s]
     * B: activation energy / R [K]
     *
     * For water: A = 2.414e-5 Pa*s, B = 570.6 K (approximate)
     * For crude oil: A ~ 1e-8 to 1e-5, B ~ 2000 to 5000 (temp-dependent)
     */

    if (temperature <= 0.0 || A <= 0.0) return 0.0;
    if (B / temperature > 50.0) return 1e10;  /* Prevent overflow */

    return A * exp(B / temperature);
}

double water_viscosity_iapws(double temperature)
{
    /*
     * Simplified IAPWS water viscosity correlation.
     * Valid: 273.15 K < T < 373.15 K (0 C to 100 C).
     *
     * mu [Pa*s] = A * 10^(B/(T-C))
     *
     * A = 2.414e-5, B = 247.8 K, C = 140 K
     *
     * Accuracy: ~2% within the valid range.
     */

    if (temperature <= 0.0) return 0.0;

    double A = 2.414e-5;
    double B = 247.8;
    double C = 140.0;

    return A * pow(10.0, B / (temperature - C));
}

double viscosity_dynamic_to_kinematic(double mu, double rho)
{
    /*
     * nu = mu / rho
     *
     * nu [m^2/s] = mu [Pa*s] / rho [kg/m^3]
     *
     * This is the ratio of momentum diffusivity to density.
     * Crucial for Reynolds number computation.
     */

    if (rho <= 0.0) return 0.0;
    return mu / rho;
}

/* ==========================================================================
 * L4: Liquid Density Temperature Correction
 * ========================================================================== */

double liquid_density_temp_correct(double rho_ref, double T_ref,
                                    double T_actual, double beta)
{
    /*
     * First-order thermal expansion:
     * V(T) = V_ref * (1 + beta * (T - T_ref))
     * rho(T) = rho_ref / (1 + beta * (T - T_ref))
     *
     * beta: volumetric thermal expansion coefficient [1/K]
     *   Water at 20 C: beta = 2.07e-4
     *   Ethanol:       beta = 1.10e-3
     *   Crude oil:     beta = 7e-4 to 9e-4 (API-dependent)
     */

    if (rho_ref <= 0.0 || T_ref <= 0.0) return 0.0;

    double dT = T_actual - T_ref;
    double V_ratio = 1.0 + beta * dT;

    if (V_ratio <= 0.0) return 0.0;  /* Negative volume impossible */

    return rho_ref / V_ratio;
}

double water_density(double temperature)
{
    /*
     * Water density correlation (Kell, 1975 — simplified).
     *
     * rho(T) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4
     *
     * where t = T [deg C] = T [K] - 273.15
     *
     * Coefficients for 0 to 100 C:
     *   a0 = 999.84, a1 = 6.745e-2, a2 = -8.575e-3,
     *   a3 = 6.630e-5, a4 = -4.030e-7
     *
     * Reference: Kell, G.S. (1975) J. Chem. Eng. Data 20(1), 97-105.
     */

    double t = temperature - 273.15;

    double a0 =  999.84;
    double a1 =    0.06745;
    double a2 =   -8.575e-3;
    double a3 =    6.630e-5;
    double a4 =   -4.030e-7;

    double t2 = t * t;
    double t3 = t2 * t;
    double t4 = t3 * t;

    double rho = a0 + a1*t + a2*t2 + a3*t3 + a4*t4;

    if (rho < 950.0) rho = 958.3;  /* Clamp to physically valid range */
    if (rho > 1005.0) rho = 1005.0;

    return rho;
}

/* ==========================================================================
 * L5: Oil Properties (Standing Correlation)
 * ========================================================================== */

double oil_density_from_api(double api_gravity)
{
    /*
     * API gravity to density conversion:
     * API = 141.5 / SG - 131.5
     * SG = rho_oil / rho_water (both at 60 F, 1 atm)
     *
     * => rho_oil = rho_water * 141.5 / (131.5 + API)
     *
     * rho_water at 60 F = 999.016 kg/m^3
     * Standard condition: 15.56 C, 101.325 kPa
     */

    if (api_gravity <= 0.0) return 999.016;  /* Heavy as water */

    double rho_water_std = 999.016;
    return rho_water_std * 141.5 / (131.5 + api_gravity);
}

double oil_bubble_point_standing(double Rs, double sg_gas,
                                  double api_gravity, double temperature)
{
    /*
     * Standing's (1947) bubble-point pressure correlation.
     *
     * Pb [psia] = 18.2 * [(Rs/sg_gas)^0.83 * 10^(0.00091*Tf - 0.0125*API) - 1.4]
     *
     * where:
     *   Rs  = solution gas-oil ratio [scf/STB]
     *   Tf  = temperature [deg F]
     *   API = oil API gravity
     *
     * Convert to SI:
     *   Rs [m^3/m^3] -> Rs [scf/STB]: multiply by 5.615
     *   T [K] -> Tf [F]: Tf = (T - 273.15) * 9/5 + 32
     *   Pb [psia] -> Pb [Pa]: multiply by 6894.757
     */

    if (Rs < 0.0) Rs = 0.0;
    if (sg_gas <= 0.0) sg_gas = 0.65;
    if (api_gravity <= 5.0) api_gravity = 30.0;

    /* Convert to field units */
    double Rs_scf_stb = Rs * 5.615;
    double Tf = (temperature - 273.15) * 9.0 / 5.0 + 32.0;

    double exponent = 0.00091 * Tf - 0.0125 * api_gravity;
    double term = pow(Rs_scf_stb / sg_gas, 0.83) * pow(10.0, exponent);

    double Pb_psia = 18.2 * (term - 1.4);

    if (Pb_psia < 0.0) return 101325.0;  /* Atmospheric at minimum */

    return Pb_psia * 6894.757;  /* Convert to Pa */
}

double oil_live_density(double api_gravity, double Rs, double sg_gas,
                         double temperature, double pressure)
{
    /*
     * Live oil density calculation combining:
     * 1. Dead oil density from API gravity
     * 2. Temperature correction (thermal expansion)
     * 3. Dissolved gas effect (oil swelling)
     *
     * Step 1: Dead oil density at standard temperature (60 F = 288.71 K)
     * Step 2: Correct for reservoir temperature
     * Step 3: Add dissolved gas contribution (below bubble point)
     *
     * The dissolved gas reduces oil density through two mechanisms:
     * - Swelling (volume increase)
     * - Gas is less dense than oil
     *
     * This simplified implementation uses Standing's approach.
     */

    if (api_gravity <= 0.0) return 0.0;

    /* Dead oil density at standard conditions */
    double rho_dead = oil_density_from_api(api_gravity);

    /* Temperature correction — assume beta for oil ~ 7.5e-4 1/K */
    double beta_oil = 7.5e-4;
    double T_std = 288.71;  /* 60 F in K */
    double rho_temp_corrected = liquid_density_temp_correct(
        rho_dead, T_std, temperature, beta_oil);

    (void)sg_gas;
    (void)pressure;
    /* Dissolved gas correction: oil density decreases with dissolved gas.
     * Approximate: rho_live = rho_temp * (1 - 0.15 * Rs / Rs_max)
     * where Rs_max is a rough estimate of max GOR at the given pressure.
     *
     * Using a simplified form:
     * rho_live ~= rho_temp / (1 + 0.0005 * Rs)
     * (This captures the swelling effect — oil volume increases with Rs)
     */
    double swelling_factor = 1.0 + 0.0005 * Rs;
    double rho_live = rho_temp_corrected / swelling_factor;

    return rho_live;
}

/* ==========================================================================
 * L6: Two-Phase Mixture Properties
 * ========================================================================== */

double mixture_density_homogeneous(double gas_fraction,
                                    double rho_gas, double rho_liq)
{
    /*
     * Homogeneous (no-slip) mixture density model:
     * rho_mix = alpha * rho_gas + (1 - alpha) * rho_liq
     *
     * where alpha = gas void fraction = A_gas / A_total
     *
     * This assumes gas and liquid travel at the same velocity (no slip).
     * Valid for dispersed bubble flow; overestimates mixture density
     * for slug/annular flow where gas travels faster than liquid.
     */

    if (gas_fraction < 0.0) gas_fraction = 0.0;
    if (gas_fraction > 1.0) gas_fraction = 1.0;

    if (rho_gas < 0.0) rho_gas = 0.0;
    if (rho_liq <= 0.0) return 0.0;

    return gas_fraction * rho_gas + (1.0 - gas_fraction) * rho_liq;
}

double mixture_viscosity_mcadams(double quality, double mu_gas,
                                  double mu_liq)
{
    /*
     * McAdams et al. (1942) two-phase viscosity correlation:
     * 1 / mu_mix = x / mu_gas + (1 - x) / mu_liq
     *
     * where x = mass quality (gas mass fraction).
     *
     * This is equivalent to series (harmonic mean) averaging of
     * viscosities. Alternative models include:
     *   - Cicchitti: mu_mix = x*mu_gas + (1-x)*mu_liq (mass-weighted)
     *   - Dukler: mu_mix = alpha*mu_gas + (1-alpha)*mu_liq (volume-weighted)
     *
     * McAdams is most commonly used for two-phase pressure drop
     * calculations in boiling/condensation applications.
     */

    if (quality < 0.0) quality = 0.0;
    if (quality > 1.0) quality = 1.0;

    if (mu_gas <= 0.0 || mu_liq <= 0.0) return 0.0;

    double inv_mix = quality / mu_gas + (1.0 - quality) / mu_liq;

    if (inv_mix <= 0.0) return 0.0;

    return 1.0 / inv_mix;
}