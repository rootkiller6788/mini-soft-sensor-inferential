/**
 * pca_monitoring.c -- PCA-based Multivariate Statistical Process Monitoring
 *
 * Knowledge: L1 T2/SPE, L2 FDD, L4 Jackson-Mudholkar, L5 contribution plots
 * Reference: MacGregor & Kourti (1995), Qin (2003), Jackson & Mudholkar (1979)
 */
#include "pca_monitoring.h"
#include "pca_decomposition.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Normal quantile via Abramowitz & Stegun 26.2.23 rational approx */
static double normal_quantile(double p)
{
    double t, num, den, z;
    const double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    const double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
    if (p >= 1.0) return 8.0;
    if (p <= 0.0) return -8.0;
    if (p > 0.5) return -normal_quantile(1.0 - p);
    t = sqrt(-2.0 * log(p));
    num = c0 + c1 * t + c2 * t * t;
    den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    z = t - num / den;
    return -z;
}

/* Wilson-Hilferty chi-squared quantile approximation */
static double chi2_quantile(double p, double df)
{
    double z = normal_quantile(p);
    double a = 2.0 / (9.0 * df);
    return df * pow(1.0 - a + z * sqrt(a), 3.0);
}

/* F-distribution quantile via chi-squared relationship */
static double f_quantile(double alpha, double d1, double d2)
{
    double chi2_d1 = chi2_quantile(1.0 - alpha, d1);
    double chi2_d2 = chi2_quantile(1.0 - alpha, d2);
    double f_approx = (d2 / d1) * (chi2_d1 / chi2_d2);
    if (d2 > 4.0) f_approx *= d2 / (d2 - 2.0);
    return f_approx > 0.0 ? f_approx : 1.0;
}

/* ===================================================================
 * L1: T2 statistic -- Mahalanobis distance in PC subspace
 * T2 = sum_{a=1..A} (t_a^2 / lambda_a), t_a = x * p_a
 * =================================================================== */
double pca_t2_statistic(const double *x, const pca_matrix *loadings,
                        const double *eigenvalues, size_t n_retained)
{
    size_t a, j, M;
    double t2 = 0.0;
    if (!x || !loadings || !loadings->data || !eigenvalues) return 0.0;
    M = loadings->cols;
    for (a = 0; a < n_retained; a++) {
        double t_a = 0.0;
        for (j = 0; j < M; j++)
            t_a += x[j] * loadings->data[j * M + a];
        double lam = eigenvalues[a];
        if (lam < 1e-12) lam = 1e-12;
        t2 += (t_a * t_a) / lam;
    }
    return t2;
}

/* L4: T2 threshold via F-distribution
 * T2_lim = A*(N-1)*(N+1) / (N*(N-A)) * F(alpha, A, N-A) */
double pca_t2_threshold_f(size_t n_retained, size_t n_samples, double alpha)
{
    double d1, d2, f_val, t2_lim;
    if (n_retained == 0 || n_samples == 0) return 0.0;
    d1 = (double)n_retained;
    d2 = (double)(n_samples - n_retained);
    if (d2 <= 0.0) return pca_t2_threshold_chi2(n_retained, alpha);
    f_val = f_quantile(alpha, d1, d2);
    t2_lim = d1 * (double)(n_samples - 1) * (double)(n_samples + 1)
             / ((double)n_samples * d2) * f_val;
    return t2_lim;
}

double pca_t2_threshold_chi2(size_t n_retained, double alpha)
{
    return chi2_quantile(1.0 - alpha, (double)n_retained);
}

/* ===================================================================
 * L1: SPE (Q) statistic -- squared reconstruction error
 * x_hat = reconstruction using A PCs, SPE = ||x - x_hat||^2
 * =================================================================== */
double pca_spe_statistic(const double *x, const pca_matrix *loadings,
                         size_t n_retained, size_t n_vars)
{
    size_t a, j;
    double *x_hat, spe;
    if (!x || !loadings || !loadings->data) return 0.0;
    x_hat = (double*)calloc(n_vars, sizeof(double));
    if (!x_hat) return -1.0;
    for (a = 0; a < n_retained; a++) {
        double t_a = 0.0;
        for (j = 0; j < n_vars; j++)
            t_a += x[j] * loadings->data[j * loadings->cols + a];
        for (j = 0; j < n_vars; j++)
            x_hat[j] += t_a * loadings->data[j * loadings->cols + a];
    }
    spe = 0.0;
    for (j = 0; j < n_vars; j++) {
        double e = x[j] - x_hat[j];
        spe += e * e;
    }
    free(x_hat);
    return spe;
}

/* L4: SPE threshold via Jackson-Mudholkar (1979) approximation
 * theta_i = sum_{j=A+1..M} lambda_j^i for i=1,2,3
 * h0 = 1 - 2*theta1*theta3/(3*theta2^2)
 * SPE_lim = theta1*(z_alpha*sqrt(2*theta2*h0^2)/theta1 + 1 + theta2*h0*(h0-1)/theta1^2)^(1/h0) */
double pca_spe_threshold(const double *eigenvalues, size_t n_retained,
                         size_t n_vars, double alpha)
{
    double theta1 = 0.0, theta2 = 0.0, theta3 = 0.0;
    double h0, z_alpha, term1, term2, spe_lim;
    size_t j;
    if (!eigenvalues || n_vars == 0) return 0.0;
    if (n_retained >= n_vars) return 0.0;
    for (j = n_retained; j < n_vars; j++) {
        double lam = eigenvalues[j];
        theta1 += lam;
        theta2 += lam * lam;
        theta3 += lam * lam * lam;
    }
    if (theta2 < 1e-15) return 0.0;
    h0 = 1.0 - 2.0 * theta1 * theta3 / (3.0 * theta2 * theta2);
    z_alpha = normal_quantile(1.0 - alpha);
    if (z_alpha > 8.0) z_alpha = 1.645;
    term1 = z_alpha * sqrt(2.0 * theta2 * h0 * h0) / theta1;
    term2 = theta2 * h0 * (h0 - 1.0) / (theta1 * theta1);
    spe_lim = theta1 * pow(1.0 + term1 + term2, 1.0 / h0);
    return spe_lim;
}

/* L5: T2 contribution plots -- per-variable decomposition of T2 */
void pca_t2_contributions(const double *x, const pca_matrix *loadings,
                          const double *eigenvalues, size_t n_retained,
                          size_t n_vars, double *contribs)
{
    size_t a, j;
    if (!x || !loadings || !loadings->data || !eigenvalues || !contribs) return;
    for (j = 0; j < n_vars; j++) contribs[j] = 0.0;
    for (a = 0; a < n_retained; a++) {
        double t_a = 0.0;
        for (j = 0; j < n_vars; j++)
            t_a += x[j] * loadings->data[j * loadings->cols + a];
        double lam = eigenvalues[a];
        if (lam < 1e-12) lam = 1e-12;
        for (j = 0; j < n_vars; j++)
            contribs[j] += (t_a * loadings->data[j * loadings->cols + a] / lam) * x[j];
    }
}

/* L5: SPE contribution plots -- per-variable squared reconstruction error */
void pca_spe_contributions(const double *x, const pca_matrix *loadings,
                           size_t n_retained, size_t n_vars, double *contribs)
{
    size_t a, j;
    double *x_hat;
    if (!x || !loadings || !loadings->data || !contribs) return;
    x_hat = (double*)calloc(n_vars, sizeof(double));
    if (!x_hat) return;
    for (a = 0; a < n_retained; a++) {
        double t_a = 0.0;
        for (j = 0; j < n_vars; j++)
            t_a += x[j] * loadings->data[j * loadings->cols + a];
        for (j = 0; j < n_vars; j++)
            x_hat[j] += t_a * loadings->data[j * loadings->cols + a];
    }
    for (j = 0; j < n_vars; j++) {
        double e = x[j] - x_hat[j];
        contribs[j] = e * e;
    }
    free(x_hat);
}

/* L2: Combined index phi = T2/T2_lim + SPE/SPE_lim */
double pca_combined_index(const double *x, const pca_matrix *loadings,
                          const double *eigenvalues, size_t n_retained,
                          size_t n_vars, size_t n_samples, double alpha)
{
    double t2 = pca_t2_statistic(x, loadings, eigenvalues, n_retained);
    double spe = pca_spe_statistic(x, loadings, n_retained, n_vars);
    double t2_lim = pca_t2_threshold_f(n_retained, n_samples, alpha);
    double spe_lim = pca_spe_threshold(eigenvalues, n_retained, n_vars, alpha);
    if (t2_lim < 1e-12) t2_lim = 1.0;
    if (spe_lim < 1e-12) spe_lim = 1.0;
    return t2 / t2_lim + spe / spe_lim;
}

/* L5: Combined index threshold via moment matching (g*chi2(h) approximation) */
double pca_combined_threshold(const double *eigenvalues, size_t n_retained,
                              size_t n_vars, size_t n_samples, double alpha)
{
    double t2_lim = pca_t2_threshold_f(n_retained, n_samples, alpha);
    double spe_lim_val = pca_spe_threshold(eigenvalues, n_retained, n_vars, alpha);
    double mu_T2, var_T2, mu_SPE, var_SPE, mu_phi, var_phi, g, h;
    size_t j;
    if (t2_lim < 1e-12) t2_lim = 1.0;
    if (spe_lim_val < 1e-12) spe_lim_val = 1.0;
    mu_T2 = (double)n_retained;
    var_T2 = 2.0 * (double)n_retained;
    mu_SPE = 0.0; var_SPE = 0.0;
    for (j = n_retained; j < n_vars; j++) {
        double lam = eigenvalues[j];
        mu_SPE += lam;
        var_SPE += 2.0 * lam * lam;
    }
    mu_phi = mu_T2 / t2_lim + mu_SPE / spe_lim_val;
    var_phi = var_T2 / (t2_lim * t2_lim) + var_SPE / (spe_lim_val * spe_lim_val);
    if (mu_phi < 1e-12 || var_phi < 1e-12) return 1.0;
    g = var_phi / (2.0 * mu_phi);
    h = 2.0 * mu_phi * mu_phi / var_phi;
    return g * chi2_quantile(1.0 - alpha, h);
}

/* L5: Full fault detection for one observation */
pca_fault_result* pca_monitor_observation(const double *x,
    const pca_matrix *loadings, const double *eigenvalues,
    size_t n_retained, size_t n_vars, size_t n_samples, double alpha)
{
    pca_fault_result *result;
    if (!x || !loadings || !eigenvalues) return NULL;
    result = (pca_fault_result*)calloc(1, sizeof(pca_fault_result));
    if (!result) return NULL;
    result->n_vars = n_vars;
    result->t2_value = pca_t2_statistic(x, loadings, eigenvalues, n_retained);
    result->t2_threshold = pca_t2_threshold_f(n_retained, n_samples, alpha);
    result->t2_alarm = (result->t2_value > result->t2_threshold) ? 1 : 0;
    result->spe_value = pca_spe_statistic(x, loadings, n_retained, n_vars);
    result->spe_threshold = pca_spe_threshold(eigenvalues, n_retained, n_vars, alpha);
    result->spe_alarm = (result->spe_value > result->spe_threshold) ? 1 : 0;
    result->combined_value = pca_combined_index(x, loadings, eigenvalues,
        n_retained, n_vars, n_samples, alpha);
    result->combined_threshold = pca_combined_threshold(eigenvalues, n_retained,
        n_vars, n_samples, alpha);
    result->combined_alarm = (result->combined_value > result->combined_threshold) ? 1 : 0;
    result->t2_contribs = (double*)calloc(n_vars, sizeof(double));
    result->spe_contribs = (double*)calloc(n_vars, sizeof(double));
    if (result->t2_contribs)
        pca_t2_contributions(x, loadings, eigenvalues, n_retained, n_vars, result->t2_contribs);
    if (result->spe_contribs)
        pca_spe_contributions(x, loadings, n_retained, n_vars, result->spe_contribs);
    return result;
}

void pca_fault_result_free(pca_fault_result *result)
{
    if (!result) return;
    free(result->t2_contribs); result->t2_contribs = NULL;
    free(result->spe_contribs); result->spe_contribs = NULL;
    free(result);
}

void pca_fault_result_print(const pca_fault_result *result)
{
    size_t j;
    if (!result) { printf("Fault result: (null)\n"); return; }
    printf("=== PCA Fault Detection ===\n");
    printf("T2:  %10.4f  lim=%10.4f  alarm=%d\n",
           result->t2_value, result->t2_threshold, result->t2_alarm);
    printf("SPE: %10.4f  lim=%10.4f  alarm=%d\n",
           result->spe_value, result->spe_threshold, result->spe_alarm);
    printf("Phi: %10.4f  lim=%10.4f  alarm=%d\n",
           result->combined_value, result->combined_threshold, result->combined_alarm);
    printf("T2-contrib:  ");
    for (j = 0; j < result->n_vars && j < 10; j++)
        printf("%8.4f ", result->t2_contribs[j]);
    printf("\nSPE-contrib: ");
    for (j = 0; j < result->n_vars && j < 10; j++)
        printf("%8.4f ", result->spe_contribs[j]);
    printf("\n");
}
