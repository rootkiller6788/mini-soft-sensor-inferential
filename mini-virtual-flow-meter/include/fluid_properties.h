/**
 * @file fluid_properties.h
 * @brief Fluid property models for virtual flow metering
 *
 * Knowledge Coverage:
 *   L1 Definitions: Density, viscosity, compressibility factor, ideal/real gas
 *   L2 Core Concepts: Phase behavior, equation of state
 *   L3 Engineering Structures: Fluid property computation pipeline
 *   L4 Engineering Laws: Ideal gas law, AGA-8, Sutherland viscosity,
 *       Andrade viscosity, Standing correlation
 *
 * Fluid properties are essential inputs to any flow model. Errors in
 * density or viscosity propagate directly to flow estimation errors.
 * This module implements temperature/pressure corrections for common
 * industrial fluids: natural gas, crude oil, water, and air.
 *
 * References:
 *   AGA Report No. 8 (1992) Compressibility Factors of Natural Gas
 *   Sutherland, W. (1893) The viscosity of gases and molecular force
 *   Standing, M.B. (1947) A pressure-volume-temperature correlation
 *     for mixtures of California oils and gases
 *
 * @module mini-virtual-flow-meter
 */

#ifndef FLUID_PROPERTIES_H
#define FLUID_PROPERTIES_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * L1: Core Constants
 * ========================================================================== */

/** Universal gas constant [J/(mol*K)] */
#define FLUID_R_GAS 8.314462618

/** Standard atmospheric pressure [Pa] */
#define FLUID_P_ATM 101325.0

/** Standard temperature [K] (0 deg C) */
#define FLUID_T_STD 273.15

/** Gravitational acceleration [m/s^2] (ISO 9806) */
#define FLUID_G 9.80665

/** Molecular weight of dry air [kg/mol] */
#define FLUID_MW_AIR 0.0289644

/** Molecular weight of water [kg/mol] */
#define FLUID_MW_WATER 0.01801528

/** Molecular weight of methane [kg/mol] */
#define FLUID_MW_CH4 0.0160425

/* ==========================================================================
 * L2: Gas Properties -- Ideal and Real Gas
 * ========================================================================== */

/**
 * @brief Compute ideal gas density from pressure and temperature.
 *
 * rho = P * M / (R * T)
 *
 * Assumes ideal gas behavior (Z = 1). Valid for low-pressure,
 * moderate-temperature conditions. Errors exceed 5% for natural gas
 * above 10 bar.
 *
 * @param pressure   Absolute pressure [Pa]
 * @param temperature Absolute temperature [K]
 * @param mol_weight Molecular weight [kg/mol]
 * @return Density [kg/m^3]
 */
double gas_density_ideal(double pressure, double temperature,
                          double mol_weight);

/**
 * @brief Compute real gas density with compressibility factor.
 *
 * rho = P * M / (Z * R * T)
 *
 * where Z is the compressibility factor, a function of reduced
 * pressure Pr = P/Pc and reduced temperature Tr = T/Tc.
 *
 * @param pressure   Absolute pressure [Pa]
 * @param temperature Absolute temperature [K]
 * @param mol_weight Molecular weight [kg/mol]
 * @param z          Compressibility factor Z = PV/RT
 * @return Density [kg/m^3]
 */
double gas_density_real(double pressure, double temperature,
                         double mol_weight, double z);

/**
 * @brief Compute compressibility factor using the Dranchuk-Abou-Kassem
 *        correlation for natural gas (AGA-8 simplified).
 *
 * This is a curve-fit of the Standing-Katz chart widely used when
 * full AGA-8 compositional analysis is not available.
 *
 * Z = 1 + (A1 + A2/Tr + A3/Tr^3 + A4/Tr^4 + A5/Tr^5) * rho_r
 *       + (A6 + A7/Tr + A8/Tr^2) * rho_r^2
 *       - A9 * (A7/Tr + A8/Tr^2) * rho_r^5
 *       + A10 * (1 + A11*rho_r^2) * (rho_r^2/Tr^3) * exp(-A11*rho_r^2)
 *
 * where rho_r = 0.27 * Pr / (Z * Tr) (reduced density, solved iteratively)
 *       Pr = P/Pc, Tr = T/Tc (reduced pressure/temperature)
 *
 * @param Pr  Reduced pressure P/Pc
 * @param Tr  Reduced temperature T/Tc
 * @param tol Convergence tolerance
 * @param max_iter Maximum iterations
 * @return Compressibility factor Z, -1 on non-convergence
 */
double gas_compressibility_dak(double Pr, double Tr, double tol, int max_iter);

/**
 * @brief Compute pseudocritical properties for natural gas from
 *        specific gravity using Sutton's correlation (SPE 14274).
 *
 * Ppc [psi] = 756.8 - 131.0*sg - 3.6*sg^2
 * Tpc [R]   = 169.2 + 349.5*sg - 74.0*sg^2
 *
 * Then converted to SI: Ppc [Pa] = Ppc [psi] * 6894.757
 *                       Tpc [K]  = Tpc [R] / 1.8
 *
 * @param sg           Specific gravity (relative to air)
 * @param Ppc          Output: pseudocritical pressure [Pa]
 * @param Tpc          Output: pseudocritical temperature [K]
 */
void gas_pseudocritical_sutton(double sg, double *Ppc, double *Tpc);

/**
 * @brief Compute specific gravity from molecular weight.
 *
 * SG = MW_gas / MW_air
 *
 * @param mol_weight  Molecular weight [kg/mol]
 * @return Specific gravity (dimensionless)
 */
double gas_specific_gravity(double mol_weight);

/* ==========================================================================
 * L3: Viscosity Models
 * ========================================================================== */

/**
 * @brief Sutherland's formula for gas dynamic viscosity.
 *
 * mu = mu0 * (T/T0)^(3/2) * (T0 + S) / (T + S)
 *
 * where S is the Sutherland constant (temperature-dependent correction
 * for intermolecular forces). For air: S = 110.4 K, mu0 = 1.716e-5 Pa*s
 * at T0 = 273.15 K.
 *
 * Valid for temperatures 100 K < T < 1900 K.
 *
 * @param temperature  Absolute temperature [K]
 * @param mu0          Reference viscosity at T0 [Pa*s]
 * @param T0           Reference temperature [K]
 * @param S            Sutherland constant [K]
 * @return Dynamic viscosity [Pa*s]
 */
double gas_viscosity_sutherland(double temperature, double mu0,
                                 double T0, double S);

/**
 * @brief Andrade correlation for liquid dynamic viscosity.
 *
 * mu = A * exp(B / T)
 *
 * where A and B are fluid-specific constants.
 * For water: A = 2.414e-5 Pa*s, B = 570.6 K (approx)
 *
 * Named after E.N. da C. Andrade (1930).
 *
 * @param temperature  Absolute temperature [K]
 * @param A            Pre-exponential factor [Pa*s]
 * @param B            Activation temperature [K]
 * @return Dynamic viscosity [Pa*s]
 */
double liquid_viscosity_andrade(double temperature, double A, double B);

/**
 * @brief Compute dynamic viscosity for water (IAPWS simplified).
 *
 * Correlation valid for 273 K < T < 373 K.
 *
 * mu = A * 10^(B/(T-C))
 * A = 2.414e-5 Pa*s, B = 247.8 K, C = 140 K
 *
 * @param temperature  Absolute temperature [K]
 * @return Dynamic viscosity of water [Pa*s]
 */
double water_viscosity_iapws(double temperature);

/**
 * @brief Convert dynamic viscosity to kinematic viscosity.
 *
 * nu = mu / rho
 *
 * @param mu   Dynamic viscosity [Pa*s]
 * @param rho  Density [kg/m^3]
 * @return Kinematic viscosity [m^2/s]
 */
double viscosity_dynamic_to_kinematic(double mu, double rho);

/* ==========================================================================
 * L4: Liquid Density Temperature Correction
 * ========================================================================== */

/**
 * @brief Correct liquid density for temperature using thermal
 *        expansion coefficient.
 *
 * rho(T) = rho_ref / (1 + beta * (T - T_ref))
 *
 * where beta is the volumetric thermal expansion coefficient.
 * For water at 20 C: beta = 2.07e-4 1/K
 * For crude oil:    beta = 7e-4 to 9e-4 1/K (API-dependent)
 *
 * @param rho_ref   Density at reference temperature [kg/m^3]
 * @param T_ref     Reference temperature [K]
 * @param T_actual  Actual temperature [K]
 * @param beta      Thermal expansion coefficient [1/K]
 * @return Density at actual temperature [kg/m^3]
 */
double liquid_density_temp_correct(double rho_ref, double T_ref,
                                    double T_actual, double beta);

/**
 * @brief Compute water density at a given temperature (simplified).
 *
 * Third-order polynomial fit valid 273 K < T < 373 K.
 * rho(T) = a0 + a1*t + a2*t^2 + a3*t^3
 * where t = T [deg C] and coefficients from Kell (1975).
 *
 * @param temperature  Temperature [K]
 * @return Density [kg/m^3]
 */
double water_density(double temperature);

/* ==========================================================================
 * L5: Oil Properties (Standing Correlation)
 * ========================================================================== */

/**
 * @brief Compute bubble-point pressure using Standing correlation.
 *
 * Pb = 18.2 * ((Rsb/sg_gas)^0.83 * 10^(0.00091*Tf - 0.0125*API) - 1.4)
 *
 * where Pb = bubble-point pressure [psia], Rsb = solution GOR [scf/STB],
 * Tf = temperature [deg F], API = oil gravity [deg API].
 *
 * Converted to SI for internal computation.
 *
 * @param Rs         Solution gas-oil ratio [m^3/m^3]
 * @param sg_gas     Gas specific gravity (air = 1)
 * @param api_gravity Oil API gravity [deg API]
 * @param temperature Temperature [K]
 * @return Bubble-point pressure [Pa]
 */
double oil_bubble_point_standing(double Rs, double sg_gas,
                                  double api_gravity, double temperature);

/**
 * @brief Compute dead oil density from API gravity.
 *
 * rho_dead = 141.5 * rho_water / (131.5 + API)
 *
 * @param api_gravity  Oil API gravity [deg API]
 * @return Dead oil density at standard conditions [kg/m^3]
 */
double oil_density_from_api(double api_gravity);

/**
 * @brief Compute live oil density (below bubble point) using
 *        Standing's correlation with dissolved gas correction.
 *
 * rho_live = rho_dead_corrected - dRho_gas
 *
 * Accounts for oil swelling by dissolved gas and density reduction.
 *
 * @param api_gravity  Oil API gravity [deg API]
 * @param Rs           Solution GOR [m^3/m^3]
 * @param sg_gas       Gas specific gravity
 * @param temperature  Temperature [K]
 * @param pressure     Pressure [Pa]
 * @return Live oil density [kg/m^3]
 */
double oil_live_density(double api_gravity, double Rs, double sg_gas,
                         double temperature, double pressure);

/* ==========================================================================
 * L6: Two-Phase Mixture Properties
 * ========================================================================== */

/**
 * @brief Compute gas-liquid mixture density (homogeneous model).
 *
 * rho_mix = alpha * rho_gas + (1 - alpha) * rho_liq
 *
 * where alpha = gas void fraction.
 *
 * @param gas_fraction  Gas volume fraction [0..1]
 * @param rho_gas       Gas density [kg/m^3]
 * @param rho_liq       Liquid density [kg/m^3]
 * @return Mixture density [kg/m^3]
 */
double mixture_density_homogeneous(double gas_fraction,
                                    double rho_gas, double rho_liq);

/**
 * @brief Compute gas-liquid mixture viscosity.
 *
 * Using Dukler et al. (1964) correlation for two-phase flow.
 * mu_mix = alpha * mu_gas + (1-alpha) * mu_liq  (homogeneous)
 * 1/mu_mix = x/mu_gas + (1-x)/mu_liq  (McAdams, for mass quality x)
 *
 * @param quality    Mass quality (gas mass fraction) [0..1]
 * @param mu_gas     Gas dynamic viscosity [Pa*s]
 * @param mu_liq     Liquid dynamic viscosity [Pa*s]
 * @return Mixture dynamic viscosity [Pa*s]
 */
double mixture_viscosity_mcadams(double quality, double mu_gas,
                                  double mu_liq);

#ifdef __cplusplus
}
#endif

#endif /* FLUID_PROPERTIES_H */