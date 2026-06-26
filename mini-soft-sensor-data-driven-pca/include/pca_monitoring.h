/**
 * pca_monitoring.h ¡ª Multivariate Statistical Process Monitoring via PCA
 *
 * Knowledge Coverage:
 *   L1 Definitions: Hotelling's T^2 statistic, SPE (Squared Prediction Error)
 *                   also known as Q statistic, contribution plots.
 *   L2 Core Concepts: Fault detection and diagnosis (FDD) using MSPC,
 *                     model plane vs residual space decomposition.
 *   L4 Engineering Laws: T^2 threshold via F-distribution,
 *                         SPE threshold via Jackson-Mudholkar approximation.
 *
 * Reference: MacGregor & Kourti (1995), Qin (2003), Jackson & Mudholkar (1979)
 * Course Alignment: MIT 6.302, Stanford ENGR205, Georgia Tech ECE 6550
 */

#ifndef PCA_MONITORING_H
#define PCA_MONITORING_H

#include "pca_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* T^2 Statistic: t2 = sum (t_a^2 / lambda_a) over retained PCs */
double pca_t2_statistic(const double *x, const pca_matrix *loadings,
                        const double *eigenvalues, size_t n_retained);

/* T^2 threshold via F-distribution: A*(N-1)*(N+1)/(N*(N-A)) * F(alpha,A,N-A) */
double pca_t2_threshold_f(size_t n_retained, size_t n_samples, double alpha);

/* T^2 threshold via chi-squared approximation */
double pca_t2_threshold_chi2(size_t n_retained, double alpha);

/* SPE (Q) statistic: ||x - x_hat||^2, residual squared norm */
double pca_spe_statistic(const double *x, const pca_matrix *loadings,
                         size_t n_retained, size_t n_vars);

/* SPE threshold via Jackson-Mudholkar (1979) approximation */
double pca_spe_threshold(const double *eigenvalues, size_t n_retained,
                         size_t n_vars, double alpha);

/* T^2 per-variable contribution: cont_j = sum_a (t_a * p_{a,j} / lambda_a) * x_j */
void pca_t2_contributions(const double *x, const pca_matrix *loadings,
                          const double *eigenvalues, size_t n_retained,
                          size_t n_vars, double *contribs);

/* SPE per-variable contribution: cont_j = (x_j - x_hat_j)^2 */
void pca_spe_contributions(const double *x, const pca_matrix *loadings,
                           size_t n_retained, size_t n_vars, double *contribs);

/* Combined index: phi = T^2/T^2_lim + SPE/SPE_lim */
double pca_combined_index(const double *x, const pca_matrix *loadings,
                          const double *eigenvalues, size_t n_retained,
                          size_t n_vars, size_t n_samples, double alpha);

/* Combined index threshold via moment-matching chi-squared approximation */
double pca_combined_threshold(const double *eigenvalues, size_t n_retained,
                              size_t n_vars, size_t n_samples, double alpha);

/* Fault detection result structure */
typedef struct {
    int t2_alarm;
    int spe_alarm;
    int combined_alarm;
    double t2_value;
    double spe_value;
    double t2_threshold;
    double spe_threshold;
    double combined_value;
    double combined_threshold;
    double *t2_contribs;
    double *spe_contribs;
    size_t n_vars;
} pca_fault_result;

/* Full fault detection for one observation */
pca_fault_result* pca_monitor_observation(const double *x,
    const pca_matrix *loadings, const double *eigenvalues,
    size_t n_retained, size_t n_vars, size_t n_samples, double alpha);

void pca_fault_result_free(pca_fault_result *result);
void pca_fault_result_print(const pca_fault_result *result);

#ifdef __cplusplus
}
#endif

#endif /* PCA_MONITORING_H */
