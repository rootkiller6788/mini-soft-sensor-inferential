/**
 * @file vfm_uncertainty.h
 * @brief Uncertainty quantification for Virtual Flow Meter
 *
 * Knowledge Coverage:
 *   L1 Definitions: Standard uncertainty, expanded uncertainty, coverage factor
 *   L2 Core Concepts: Error propagation, sensitivity analysis, GUM framework
 *   L3 Engineering Structures: Uncertainty budget, Monte Carlo simulation
 *   L4 Engineering Laws: GUM (JCGM 100:2008), Taylor series propagation
 *   L5 Algorithms: Monte Carlo uncertainty propagation (GUM Supplement 1)
 *
 * Every VFM estimate must be accompanied by a defensible uncertainty
 * statement. This module implements the Guide to the Expression of
 * Uncertainty in Measurement (GUM) framework adapted for flow estimation.
 *
 * References:
 *   JCGM 100:2008 GUM -- Evaluation of measurement data
 *   JCGM 101:2008 GUM Supplement 1 -- Propagation of distributions
 *     using a Monte Carlo method
 *   ISO 5168:2005 Measurement of fluid flow -- Procedures for
 *     the evaluation of uncertainties
 *
 * @module mini-virtual-flow-meter
 */

#ifndef VFM_UNCERTAINTY_H
#define VFM_UNCERTAINTY_H

#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * L1: Core Definitions -- Uncertainty Types
 * ========================================================================== */

/**
 * @brief Uncertainty type classification per GUM.
 *
 * Type A: evaluated by statistical analysis of repeated measurements
 * Type B: evaluated by other means (calibration certs, specs, experience)
 */
typedef enum {
    VFM_UNCERT_TYPE_A = 0,    /**< Statistical (standard deviation of mean) */
    VFM_UNCERT_TYPE_B = 1     /**< Non-statistical (calibration, specs)     */
} vfm_uncertainty_type_t;

/**
 * @brief Uncertainty distribution shape for Type B evaluation.
 */
typedef enum {
    VFM_DIST_NORMAL    = 0,   /**< Normal (Gaussian) distribution           */
    VFM_DIST_RECTANGULAR= 1,  /**< Uniform distribution (no preference)     */
    VFM_DIST_TRIANGULAR = 2,  /**< Triangular distribution                  */
    VFM_DIST_U_SHAPED   = 3   /**< U-shaped (arcsine) distribution          */
} vfm_distribution_t;

/**
 * @brief Single uncertainty component in the uncertainty budget.
 */
typedef struct {
    char     source[64];            /**< Description of uncertainty source  */
    double   standard_uncertainty;  /**< Standard uncertainty u(x_i)        */
    double   sensitivity_coeff;     /**< Sensitivity coefficient c_i        */
    double   contribution;          /**< c_i * u(x_i)                       */
    int      degrees_of_freedom;    /**< Effective degrees of freedom       */
    vfm_uncertainty_type_t type;    /**< Type A or Type B                   */
    vfm_distribution_t distribution;/**< Assumed distribution               */
} vfm_uncertainty_component_t;

/**
 * @brief Complete uncertainty budget for a VFM estimate.
 */
typedef struct {
    vfm_uncertainty_component_t components[16]; /**< Up to 16 components    */
    int    num_components;         /**< Number of active components         */
    double combined_std_uncert;    /**< Combined standard uncertainty u_c   */
    double expanded_uncertainty;   /**< U = k * u_c                         */
    double coverage_factor;        /**< Coverage factor k (usually 2)       */
    double confidence_level;       /**< Confidence level [0..1]             */
    int    effective_dof;          /**< Welch-Satterthwaite effective DOF   */
} vfm_uncertainty_budget_t;

/* ==========================================================================
 * L2: Core Uncertainty Operations
 * ========================================================================== */

/**
 * @brief Initialize an empty uncertainty budget.
 *
 * @param budget    Budget to initialize
 * @param conf_level Confidence level [0..1] (e.g., 0.95)
 * @return 0 on success
 */
int vfm_uncertainty_budget_init(vfm_uncertainty_budget_t *budget,
                                 double conf_level);

/**
 * @brief Add an uncertainty component to the budget.
 *
 * @param budget        Budget to add to
 * @param source        Description (e.g., "DP transmitter accuracy")
 * @param std_uncert    Standard uncertainty of the input quantity
 * @param sensitivity   Sensitivity coefficient dQ/dx_i
 * @param dof           Degrees of freedom (>=1 for Type A, large for Type B)
 * @param type          Type A or Type B
 * @param dist          Distribution shape
 * @return 0 on success, -1 if budget full
 */
int vfm_uncertainty_add_component(vfm_uncertainty_budget_t *budget,
                                   const char *source,
                                   double std_uncert, double sensitivity,
                                   int dof, vfm_uncertainty_type_t type,
                                   vfm_distribution_t dist);

/**
 * @brief Compute combined standard uncertainty (RSS of contributions).
 *
 * u_c = sqrt(sum_i (c_i * u_i)^2)
 *
 * Assumes independent input quantities (no covariance terms).
 *
 * @param budget  Uncertainty budget (updated with combined uncertainty)
 * @return Combined standard uncertainty u_c
 */
double vfm_uncertainty_combine(vfm_uncertainty_budget_t *budget);

/**
 * @brief Compute expanded uncertainty U = k * u_c.
 *
 * The coverage factor k is determined from:
 * 1. Effective degrees of freedom (Welch-Satterthwaite formula)
 * 2. Desired confidence level
 * 3. Student's t-distribution
 *
 * @param budget  Uncertainty budget (updated with expanded uncertainty)
 * @return Expanded uncertainty U
 */
double vfm_uncertainty_expand(vfm_uncertainty_budget_t *budget);

/* ==========================================================================
 * L3: Taylor Series Propagation (Analytical)
 * ========================================================================== */

/**
 * @brief Compute combined uncertainty using first-order Taylor series.
 *
 * For a measurement model y = f(x1, x2, ..., xn):
 *
 *   u_c^2(y) = sum_i (df/dx_i)^2 * u^2(x_i)
 *            + 2 * sum_i sum_{j>i} (df/dx_i)*(df/dx_j) * u(x_i,x_j)
 *
 * where u(x_i,x_j) is the covariance between x_i and x_j.
 * If all inputs are independent, covariance terms are zero.
 *
 * This function takes pre-computed sensitivity coefficients and
 * input uncertainties, assuming independent inputs.
 *
 * @param sensitivity     Array of df/dx_i [n_inputs]
 * @param uncertainties   Array of u(x_i) [n_inputs]
 * @param n_inputs        Number of input quantities
 * @return Combined standard uncertainty u_c(y)
 */
double vfm_taylor_combined_uncertainty(const double *sensitivity,
                                        const double *uncertainties,
                                        int n_inputs);

/**
 * @brief Compute sensitivity coefficient numerically.
 *
 * df/dx_i ~= (f(x + h*e_i) - f(x - h*e_i)) / (2*h)
 *
 * Central difference with step size h.
 * Assumes f is a function of x[] with one scalar output.
 *
 * @param f          Function to differentiate (takes x array, returns scalar)
 * @param x          Point at which to evaluate [n_params]
 * @param n_params   Dimension of x
 * @param param_idx  Which parameter to differentiate with respect to (0-indexed)
 * @param h          Step size
 * @return Sensitivity coefficient df/dx_param_idx
 */
double vfm_numerical_sensitivity(double (*f)(const double *, int),
                                  const double *x, int n_params,
                                  int param_idx, double h);

/* ==========================================================================
 * L4: Welch-Satterthwaite Effective Degrees of Freedom
 * ========================================================================== */

/**
 * @brief Compute effective degrees of freedom.
 *
 * Welch-Satterthwaite formula (ISO 5168):
 *
 *   nu_eff = u_c^4 / sum_i (c_i^4 * u_i^4 / nu_i)
 *
 * where nu_i is the degrees of freedom for each component.
 * This is used to select the appropriate t-distribution coverage factor.
 *
 * @param budget  Uncertainty budget with combined uncertainty
 * @return Effective degrees of freedom (capped at 1000 for practical use)
 */
int vfm_welch_satterthwaite_dof(const vfm_uncertainty_budget_t *budget);

/**
 * @brief Get Student's t coverage factor.
 *
 * k = t_{nu_eff, (1+CL)/2}
 *
 * Approximates the t-distribution quantile using:
 *   t = z + (z + z^3)/(4*nu) + (z + 5*z^3 + 3*z^5)/(96*nu^2) + ...
 *
 * where z is the normal quantile for the same confidence level.
 *
 * @param dof   Effective degrees of freedom
 * @param conf_level  Confidence level [0..1]
 * @return Coverage factor k
 */
double vfm_t_distribution_coverage(int dof, double conf_level);

/* ==========================================================================
 * L5: Monte Carlo Uncertainty Propagation (GUM Supplement 1)
 * ========================================================================== */

/**
 * @brief Monte Carlo simulation for uncertainty propagation.
 *
 * GUM Supplement 1 method: instead of analytical Taylor propagation,
 * Monte Carlo draws random samples from input distributions and
 * evaluates the measurement model for each sample, building the
 * output distribution empirically.
 *
 * Algorithm:
 *   For i = 1 to M:
 *     1. Draw random sample x* from each input distribution
 *     2. Compute y_i = f(x*)
 *   Sort y_i values
 *   Compute mean and quantile-based confidence interval
 *
 * @param model_func    Measurement model function
 * @param x_nominal     Nominal input values [n_inputs]
 * @param x_uncert      Standard uncertainties [n_inputs]
 * @param n_inputs      Number of input quantities
 * @param n_samples     Number of Monte Carlo trials (e.g., 100000)
 * @param confidence    Confidence level [0..1]
 * @param y_mean        Output: mean of output distribution
 * @param y_std         Output: standard deviation of output
 * @param y_lower       Output: lower confidence bound
 * @param y_upper       Output: upper confidence bound
 * @return 0 on success, -1 on allocation failure
 */
int vfm_monte_carlo_uncertainty(
    double (*model_func)(const double *, int, void *),
    const double *x_nominal, const double *x_uncert,
    int n_inputs, int n_samples, double confidence,
    double *y_mean, double *y_std, double *y_lower, double *y_upper,
    void *user_data);

/* ==========================================================================
 * L6: Uncertainty Propagation for Specific VFM Models
 * ========================================================================== */

/**
 * @brief Compute orifice flow uncertainty analytically.
 *
 * For Q = Cd * A * sqrt(2*dP/rho) / sqrt(1-beta^4):
 *
 * (u_Q/Q)^2 = (u_Cd/Cd)^2 + (u_A/A)^2
 *           + (1/4)*(u_dP/dP)^2 + (1/4)*(u_rho/rho)^2
 *           + (2*beta^3/(1-beta^4))^2 * (u_beta)^2
 *
 * @param Q            Flow rate [m^3/s]
 * @param Cd           Discharge coefficient
 * @param u_Cd         Uncertainty in Cd
 * @param dP           Differential pressure [Pa]
 * @param u_dP         Uncertainty in dP
 * @param rho          Density [kg/m^3]
 * @param u_rho        Uncertainty in rho
 * @param beta         Beta ratio
 * @param u_beta       Uncertainty in beta
 * @param D            Pipe diameter [m]
 * @param u_D          Uncertainty in D
 * @return Relative standard uncertainty u_Q/Q [0..1]
 */
double vfm_orifice_relative_uncertainty(double Q, double Cd, double u_Cd,
                                         double dP, double u_dP,
                                         double rho, double u_rho,
                                         double beta, double u_beta,
                                         double D, double u_D);

#ifdef __cplusplus
}
#endif

#endif /* VFM_UNCERTAINTY_H */