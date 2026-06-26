/**
 * @file vfm_uncertainty.c
 * @brief Uncertainty quantification implementations for VFM
 *
 * Implements GUM uncertainty budget, Taylor propagation,
 * Monte Carlo simulation, Welch-Satterthwaite DOF, orifice uncertainty.
 *
 * @module mini-virtual-flow-meter
 */

#include "vfm_uncertainty.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ==========================================================================
 * L2: Uncertainty Budget Management
 * ========================================================================== */

int vfm_uncertainty_budget_init(vfm_uncertainty_budget_t *budget,
                                 double conf_level)
{
    if (!budget) return -1;

    budget->num_components       = 0;
    budget->combined_std_uncert  = 0.0;
    budget->expanded_uncertainty = 0.0;
    budget->coverage_factor      = 2.0;
    budget->confidence_level     = conf_level;
    budget->effective_dof        = 50;  /* Default: large DOF for Type B dominant */

    /* Zero all components */
    memset(budget->components, 0, sizeof(budget->components));

    return 0;
}

int vfm_uncertainty_add_component(vfm_uncertainty_budget_t *budget,
                                   const char *source,
                                   double std_uncert, double sensitivity,
                                   int dof, vfm_uncertainty_type_t type,
                                   vfm_distribution_t dist)
{
    if (!budget || !source) return -1;
    if (budget->num_components >= 16) return -1;
    if (std_uncert < 0.0) return -1;
    if (dof < 1) dof = 1;

    int idx = budget->num_components;

    /* Copy source name, truncating to 63 chars */
    size_t len = strlen(source);
    if (len > 63) len = 63;
    memcpy(budget->components[idx].source, source, len);
    budget->components[idx].source[len] = '\0';

    budget->components[idx].standard_uncertainty = std_uncert;
    budget->components[idx].sensitivity_coeff    = sensitivity;
    budget->components[idx].contribution = sensitivity * std_uncert;
    budget->components[idx].degrees_of_freedom   = dof;
    budget->components[idx].type        = type;
    budget->components[idx].distribution = dist;

    budget->num_components++;

    return 0;
}

double vfm_uncertainty_combine(vfm_uncertainty_budget_t *budget)
{
    /*
     * Combined standard uncertainty: RSS (Root Sum of Squares).
     *
     * u_c = sqrt(sum_i (c_i * u_i)^2)
     *
     * Assumes independent input quantities. If correlations exist,
     * covariance terms must be added:
     * u_c^2 = sum_i (c_i*u_i)^2 + 2*sum_{i<j} c_i*c_j*u_i*u_j*r_ij
     *
     * where r_ij is the correlation coefficient between inputs i and j.
     */

    if (!budget) return 0.0;

    double sum_sq = 0.0;
    int i;
    for (i = 0; i < budget->num_components; i++) {
        double contrib = budget->components[i].contribution;
        sum_sq += contrib * contrib;
    }

    budget->combined_std_uncert = sqrt(sum_sq);
    return budget->combined_std_uncert;
}

double vfm_uncertainty_expand(vfm_uncertainty_budget_t *budget)
{
    /*
     * Expanded uncertainty: U = k * u_c
     *
     * The coverage factor k is determined from the effective degrees
     * of freedom and the desired confidence level using Student's
     * t-distribution.
     *
     * Step 1: Compute combined standard uncertainty u_c
     * Step 2: Compute effective DOF (Welch-Satterthwaite)
     * Step 3: Get t-distribution coverage factor k
     * Step 4: U = k * u_c
     */

    if (!budget) return 0.0;

    /* Ensure combined uncertainty is computed */
    vfm_uncertainty_combine(budget);

    /* Compute effective degrees of freedom */
    budget->effective_dof = vfm_welch_satterthwaite_dof(budget);

    /* Get coverage factor */
    budget->coverage_factor = vfm_t_distribution_coverage(
        budget->effective_dof, budget->confidence_level);

    budget->expanded_uncertainty = budget->coverage_factor
                                 * budget->combined_std_uncert;

    return budget->expanded_uncertainty;
}

/* ==========================================================================
 * L3: Taylor Series Propagation
 * ========================================================================== */

double vfm_taylor_combined_uncertainty(const double *sensitivity,
                                        const double *uncertainties,
                                        int n_inputs)
{
    /*
     * First-order Taylor series uncertainty propagation.
     *
     * For y = f(x1, ..., xn), assuming independent inputs:
     *   u_c^2(y) = sum_i (df/dx_i)^2 * u^2(x_i)
     *
     * This is exact for linear models. For nonlinear models, it is
     * a first-order approximation valid when:
     *   - Nonlinearity is moderate over the uncertainty range
     *   - Input uncertainties are small relative to the function curvature
     *
     * If these conditions are not met, use Monte Carlo (GUM Supplement 1).
     */

    if (!sensitivity || !uncertainties || n_inputs <= 0) return 0.0;

    double sum_sq = 0.0;
    int i;
    for (i = 0; i < n_inputs; i++) {
        double term = sensitivity[i] * uncertainties[i];
        sum_sq += term * term;
    }

    return sqrt(sum_sq);
}

double vfm_numerical_sensitivity(double (*f)(const double *, int),
                                  const double *x, int n_params,
                                  int param_idx, double h)
{
    /*
     * Central difference numerical differentiation.
     *
     * df/dx_i = (f(x + h_i*e_i) - f(x - h_i*e_i)) / (2*h_i)
     *
     * where h_i = max(|x_i| * sqrt(eps), h_min) and e_i is the i-th
     * unit vector.
     *
     * Central difference is second-order accurate (O(h^2)) vs.
     * forward difference which is O(h).
     *
     * Choose h ~ sqrt(epsilon_machine) * |x| for optimal tradeoff
     * between truncation error (decreases with h) and roundoff error
     * (increases as 1/h).
     */

    if (!f || !x || param_idx < 0 || param_idx >= n_params) return 0.0;

    /* Adaptive step size */
    double eps = 1.4901161193847656e-8;  /* sqrt(DBL_EPSILON) */
    double h_actual = fmax(h, eps * fabs(x[param_idx]));
    if (h_actual < 1e-12) h_actual = 1e-8;

    /* Create perturbed copies of x */
    double *x_plus  = (double *)malloc((size_t)n_params * sizeof(double));
    double *x_minus = (double *)malloc((size_t)n_params * sizeof(double));
    if (!x_plus || !x_minus) {
        free(x_plus);
        free(x_minus);
        return 0.0;
    }

    int i;
    for (i = 0; i < n_params; i++) {
        x_plus[i]  = x[i];
        x_minus[i] = x[i];
    }

    x_plus[param_idx]  += h_actual;
    x_minus[param_idx] -= h_actual;

    double f_plus  = f(x_plus, n_params);
    double f_minus = f(x_minus, n_params);

    free(x_plus);
    free(x_minus);

    return (f_plus - f_minus) / (2.0 * h_actual);
}

/* ==========================================================================
 * L4: Welch-Satterthwaite and t-distribution
 * ========================================================================== */

int vfm_welch_satterthwaite_dof(const vfm_uncertainty_budget_t *budget)
{
    /*
     * Welch-Satterthwaite formula for effective degrees of freedom.
     *
     * nu_eff = u_c^4 / sum_i (c_i^4 * u_i^4 / nu_i)
     *
     * where:
     *   u_c = combined standard uncertainty
     *   c_i = sensitivity coefficient for component i
     *   u_i = standard uncertainty of component i
     *   nu_i = degrees of freedom for component i
     *
     * Practical limits:
     *   nu_eff >= 1 (minimum)
     *   nu_eff <= 1000 (effectively infinite, normal distribution)
     *
     * For Type B evaluations with rectangular distributions,
     * nu_i -> infinity (perfect knowledge of bounds).
     * We cap nu_i at 1e6 to prevent overflow.
     *
     * Reference: Welch, B.L. (1947), Satterthwaite, F.E. (1946)
     */

    if (!budget || budget->num_components == 0) return 50;

    double uc = budget->combined_std_uncert;
    double uc4 = uc * uc * uc * uc;

    if (uc4 < 1e-30) return 50;  /* No meaningful uncertainty */

    double denom = 0.0;
    int i;
    for (i = 0; i < budget->num_components; i++) {
        double c = budget->components[i].sensitivity_coeff;
        double u = budget->components[i].standard_uncertainty;
        double nu = (double)budget->components[i].degrees_of_freedom;
        if (nu < 1.0) nu = 1.0;
        if (nu > 1e6) nu = 1e6;  /* Cap for Type B with rectangular dist */

        double c4 = c * c;
        c4 = c4 * c4;  /* c^4 */
        double u4 = u * u;
        u4 = u4 * u4;  /* u^4 */

        denom += c4 * u4 / nu;
    }

    if (denom < 1e-30) return 1000;

    double nu_eff = uc4 / denom;

    /* Clamp to practical range */
    if (nu_eff < 1.0) nu_eff = 1.0;
    if (nu_eff > 1000.0) nu_eff = 1000.0;

    return (int)(nu_eff + 0.5);
}

double vfm_t_distribution_coverage(int dof, double conf_level)
{
    /*
     * Student's t-distribution coverage factor.
     *
     * For large DOF (>100), t approaches the normal distribution.
     * k = t_{nu, (1+CL)/2}
     *
     * Uses the approximation:
     * t = z * (1 + (z^2+1)/(4*nu) + (5*z^4+16*z^2+3)/(96*nu^2)
     *          + (3*z^6+19*z^4+17*z^2-15)/(384*nu^3))
     *
     * where z = normal quantile for same confidence level.
     *
     * This expansion is accurate to O(1/nu^4).
     */

    if (dof < 1) dof = 1;
    if (conf_level <= 0.0 || conf_level >= 1.0) return 2.0;

    /* Normal quantile approximation (Abramowitz-Stegun) */
    double p = (1.0 + conf_level) / 2.0;
    double x = p;
    if (x > 0.5) x = 1.0 - x;

    double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;

    double t_val = sqrt(-2.0 * log(x));
    double z = t_val - (c0 + c1*t_val + c2*t_val*t_val)
                     / (1.0 + d1*t_val + d2*t_val*t_val + d3*t_val*t_val*t_val);
    if (p < 0.5) z = -z;

    /* For large DOF, normal approximation is sufficient */
    if (dof > 120) return z;

    /* t = z * [1 + correction terms] */
    double nu = (double)dof;
    double z2 = z * z;
    double z3 = z2 * z;
    double z4 = z3 * z;
    double z5 = z4 * z;
    double z6 = z5 * z;

    double term1 = (z2 + 1.0) / (4.0 * nu);
    double term2 = (5.0*z4 + 16.0*z2 + 3.0) / (96.0 * nu * nu);
    double term3 = (3.0*z6 + 19.0*z4 + 17.0*z2 - 15.0) / (384.0 * nu * nu * nu);

    double k = z * (1.0 + term1 + term2 + term3);

    return k;
}

/* ==========================================================================
 * L5: Monte Carlo Uncertainty Propagation
 * ========================================================================== */

/**
 * Box-Muller transform: generate two independent standard normal
 * random variables from two uniform [0,1] random numbers.
 */
static void box_muller(double u1, double u2, double *n1, double *n2)
{
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    *n1 = r * cos(theta);
    *n2 = r * sin(theta);
}

/**
 * Simple linear congruential generator for deterministic
 * pseudo-random numbers (for reproducibility in Monte Carlo).
 */
static double lcg_random(unsigned int *seed)
{
    *seed = (*seed) * 1103515245U + 12345U;
    return (double)((*seed) >> 16) / 32768.0;
}

/**
 * Comparison function for qsort (ascending order).
 */
static int cmp_double_asc(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

int vfm_monte_carlo_uncertainty(
    double (*model_func)(const double *, int, void *),
    const double *x_nominal, const double *x_uncert,
    int n_inputs, int n_samples, double confidence,
    double *y_mean, double *y_std, double *y_lower, double *y_upper,
    void *user_data)
{
    /*
     * Monte Carlo uncertainty propagation per GUM Supplement 1.
     *
     * Algorithm:
     *   1. Allocate output array of size n_samples
     *   2. For each trial:
     *      a. Generate random input vector X* from N(x_i, u_i^2)
     *         (assuming independent normal distributions)
     *      b. Evaluate Y_i = f(X*)
     *   3. Sort Y_i ascending
     *   4. Compute mean, std, and confidence bounds from order statistics
     *
     * Coverage probability for confidence interval:
     *   alpha = 1 - confidence
     *   lower index = floor(n_samples * alpha/2)
     *   upper index = ceil(n_samples * (1 - alpha/2)) - 1
     *
     * With n_samples = 1e6, the Monte Carlo error in the bounds is
     * approximately u_c / sqrt(n_samples) ≈ 0.001 * u_c.
     */

    if (!model_func || !x_nominal || !x_uncert || n_inputs <= 0 ||
        n_samples < 100 || !y_mean || !y_std || !y_lower || !y_upper) {
        return -1;
    }

    double *outputs = (double *)malloc((size_t)n_samples * sizeof(double));
    if (!outputs) return -1;

    /* Pseudo-random seed (fixed for reproducibility) */
    unsigned int seed = 12345U;
    int i;

    for (i = 0; i < n_samples; i++) {
        /* Generate independent normal perturbations for each input */
        int j;
        for (j = 0; j < n_inputs; j += 2) {
            double u1 = lcg_random(&seed);
            double u2 = lcg_random(&seed);
            double n1, n2;
            box_muller(u1, u2, &n1, &n2);

            /* No need to store the full input vector — evaluate directly? */
            /* But we need the perturbed vector for model_func */
        }
    }

    /* Second pass: generate inputs and evaluate */
    seed = 12345U;
    double *x_pert = (double *)malloc((size_t)n_inputs * sizeof(double));
    if (!x_pert) { free(outputs); return -1; }

    for (i = 0; i < n_samples; i++) {
        int j;
        for (j = 0; j < n_inputs; j++) {
            double u1 = lcg_random(&seed);
            double u2 = lcg_random(&seed);
            double n1, n2;
            box_muller(u1, u2, &n1, &n2);

            /* Perturb input j */
            x_pert[j] = x_nominal[j] + n1 * x_uncert[j];
        }

        outputs[i] = model_func(x_pert, n_inputs, user_data);
    }

    free(x_pert);

    /* Sort outputs for quantile computation */
    qsort(outputs, (size_t)n_samples, sizeof(double), cmp_double_asc);

    /* Compute mean and standard deviation */
    double sum = 0.0, sum_sq = 0.0;
    for (i = 0; i < n_samples; i++) {
        sum    += outputs[i];
        sum_sq += outputs[i] * outputs[i];
    }
    *y_mean = sum / (double)n_samples;
    double variance = sum_sq / (double)n_samples - (*y_mean) * (*y_mean);
    if (variance < 0.0) variance = 0.0;
    *y_std = sqrt(variance);

    /* Confidence bounds: symmetric percentile interval */
    double alpha = 1.0 - confidence;
    int idx_lower = (int)((double)n_samples * alpha / 2.0);
    int idx_upper = (int)((double)n_samples * (1.0 - alpha / 2.0)) - 1;
    if (idx_lower < 0) idx_lower = 0;
    if (idx_upper >= n_samples) idx_upper = n_samples - 1;

    *y_lower = outputs[idx_lower];
    *y_upper = outputs[idx_upper];

    free(outputs);
    return 0;
}

/* ==========================================================================
 * L6: Orifice Flow Uncertainty Analytical Derivation
 * ========================================================================== */

double vfm_orifice_relative_uncertainty(double Q, double Cd, double u_Cd,
                                         double dP, double u_dP,
                                         double rho, double u_rho,
                                         double beta, double u_beta,
                                         double D, double u_D)
{
    /*
     * Analytical uncertainty propagation for orifice flow measurement.
     *
     * From the orifice equation:
     *   Q = Cd * A * sqrt(2*dP/rho) / sqrt(1-beta^4)
     *   A = pi/4 * d^2 = pi/4 * (beta*D)^2
     *
     * Taking logarithms and differentiating:
     *   ln(Q) = ln(Cd) + 2*ln(beta) + 2*ln(D) + ln(const)
     *         + 0.5*ln(dP) - 0.5*ln(rho) - 0.5*ln(1-beta^4)
     *
     * dQ/Q = dCd/Cd + 2*d(beta)/beta * (1 + beta^4/(1-beta^4))
     *      + 2*dD/D + 0.5*d(dP)/dP - 0.5*d(rho)/rho
     *
     * Therefore relative uncertainty (independent inputs):
     * (u_Q/Q)^2 = (u_Cd/Cd)^2
     *           + [2*(1 + beta^4/(1-beta^4))]^2 * (u_beta/beta)^2
     *           + 4*(u_D/D)^2
     *           + 0.25*(u_dP/dP)^2
     *           + 0.25*(u_rho/rho)^2
     */

    if (Q <= 0.0 || Cd <= 0.0 || dP <= 0.0 || rho <= 0.0 ||
        beta <= 0.0 || D <= 0.0) {
        return 1.0;  /* Invalid inputs => 100% uncertainty */
    }

    /* Component 1: Discharge coefficient */
    double term_Cd = (u_Cd / Cd);
    double sum_sq = term_Cd * term_Cd;

    /* Component 2: Beta ratio (diameter ratio) */
    double beta4 = beta * beta;
    beta4 = beta4 * beta4;
    double beta_coeff = 1.0 + beta4 / (1.0 - beta4);
    double term_beta = 2.0 * beta_coeff * (u_beta / beta);
    sum_sq += term_beta * term_beta;

    /* Component 3: Pipe diameter */
    double term_D = 2.0 * (u_D / D);
    sum_sq += term_D * term_D;

    /* Component 4: Differential pressure */
    double term_dP = 0.5 * (u_dP / dP);
    sum_sq += term_dP * term_dP;

    /* Component 5: Density */
    double term_rho = 0.5 * (u_rho / rho);
    sum_sq += term_rho * term_rho;

    return sqrt(sum_sq);
}