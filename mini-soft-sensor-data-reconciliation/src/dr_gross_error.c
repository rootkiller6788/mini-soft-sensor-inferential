/**
 * @file dr_gross_error.c
 * @brief Gross error detection, identification, and robust reconciliation.
 */

#include "dr_gross_error.h"
#include "dr_core.h"
#include "dr_matrix.h"
#include "dr_measurement.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static int compute_V_cholesky(const dr_problem_t *prob, double *V, int m) {
    if (dr_compute_V_matrix(prob, V) != DR_OK) return -1;
    if (dr_mat_cholesky_raw(V, m) != 0) return -1;
    return 0;
}

int dr_ge_global_test(const dr_problem_t *prob, dr_ge_test_result_t *result,
                      double alpha) {
    int i, j, m, n;
    double *V, *r, z;
    if (!prob || !result) return DR_ERR_NULL_POINTER;
    m = prob->ncon; n = prob->nvar;
    result->test_type = 0;
    if (m == 0) {
        result->statistic = 0.0; result->critical_value = 0.0;
        result->p_value = 1.0; result->reject_H0 = 0;
        result->suspect_index = -1; return DR_OK;
    }
    V = (double *)malloc((size_t)m * (size_t)m * sizeof(double));
    r = (double *)malloc((size_t)m * sizeof(double));
    if (!V || !r) { free(V); free(r); return DR_ERR_NULL_POINTER; }
    if (compute_V_cholesky(prob, V, m) != 0) {
        free(V); free(r); result->reject_H0 = 0; return DR_ERR_SINGULAR;
    }
    for (i = 0; i < m; i++) {
        double sum = 0.0;
        for (j = 0; j < n; j++)
            sum += prob->constraint_A[i * n + j] * prob->measurements[j].value;
        r[i] = sum - prob->constraint_b[i];
    }
    double *y = (double *)malloc((size_t)m * sizeof(double));
    if (!y) { free(V); free(r); return DR_ERR_NULL_POINTER; }
    if (dr_mat_cholesky_solve_raw(V, r, y, m) != 0) {
        free(y); free(V); free(r); return DR_ERR_SINGULAR;
    }
    z = 0.0;
    for (i = 0; i < m; i++) z += r[i] * y[i];
    double chi2_crit = dr_meas_chi2_critical(m, alpha);
    result->statistic = z;
    result->critical_value = chi2_crit;
    result->reject_H0 = (z > chi2_crit) ? 1 : 0;
    result->suspect_index = -1;
    {
        double nu = (double)m;
        double zn = (pow(z / nu, 1.0/3.0) - (1.0 - 2.0/(9.0*nu)))
                     / sqrt(2.0/(9.0*nu));
        result->p_value = 1.0 - 0.5 * (1.0 + erf(zn / sqrt(2.0)));
        if (result->p_value < 0.0) result->p_value = 0.0;
        if (result->p_value > 1.0) result->p_value = 1.0;
    }
    free(y); free(V); free(r);
    return DR_OK;
}
static double huber_weight(double r, double c) {
    double ar = fabs(r);
    return (ar <= c) ? 1.0 : (c / ar);
}

static double biweight_weight(double r, double c) {
    double ar = fabs(r);
    if (ar <= c) {
        double t = 1.0 - (ar * ar) / (c * c);
        return t * t;
    }
    return 0.0;
}

static double cauchy_weight(double r, double c) {
    return 1.0 / (1.0 + (r * r) / (c * c));
}

static double fair_weight(double r, double c) {
    return 1.0 / (1.0 + fabs(r) / c);
}

static int irls_solve(const dr_problem_t *prob, double *x_out,
                      double (*weight_fn)(double, double),
                      double c, int max_iter, double tol) {
    int iter, i, m = prob->ncon, n = prob->nvar;
    double *x_curr = (double *)malloc((size_t)n * sizeof(double));
    double *x_prev = (double *)malloc((size_t)n * sizeof(double));
    if (!x_curr || !x_prev) {
        free(x_curr); free(x_prev); return DR_ERR_NULL_POINTER;
    }
    dr_result_t *init = dr_result_create(n, m);
    if (!init) { free(x_prev); free(x_curr); return DR_ERR_NULL_POINTER; }
    if (dr_solve(prob, init, DR_SOLVER_LAGRANGE) != DR_OK) {
        dr_result_free(init); free(x_prev); free(x_curr);
        return DR_ERR_SINGULAR;
    }
    for (i = 0; i < n; i++) x_curr[i] = init->x_reconciled[i];
    dr_result_free(init);
    dr_problem_t *work = dr_problem_create(n, m);
    if (!work) { free(x_prev); free(x_curr); return DR_ERR_NULL_POINTER; }
    for (i = 0; i < m; i++)
        dr_set_constraint(work, i, &prob->constraint_A[i * n],
                          prob->constraint_b[i], prob->constraint_types[i]);
    for (iter = 0; iter < max_iter; iter++) {
        for (i = 0; i < n; i++) {
            if (!prob->measurements[i].is_present) continue;
            double r = (prob->measurements[i].value - x_curr[i])
                       / prob->measurements[i].stddev;
            double w = weight_fn(r, c);
            if (w < 1e-10) w = 1e-10;
            double sig_eff = prob->measurements[i].stddev / sqrt(w);
            dr_set_measurement(work, i, prob->measurements[i].value,
                               sig_eff, prob->measurements[i].tag_id);
        }
        dr_result_t *res = dr_result_create(n, m);
        if (!res) break;
        if (dr_solve(work, res, DR_SOLVER_LAGRANGE) != DR_OK) {
            dr_result_free(res); break;
        }
        double dmax = 0.0;
        for (i = 0; i < n; i++) {
            double d = fabs(res->x_reconciled[i] - x_curr[i]);
            if (d > dmax) dmax = d;
        }
        memcpy(x_prev, x_curr, (size_t)n * sizeof(double));
        for (i = 0; i < n; i++) x_curr[i] = res->x_reconciled[i];
        dr_result_free(res);
        if (dmax < tol) break;
    }
    for (i = 0; i < n; i++) x_out[i] = x_curr[i];
    dr_problem_free(work);
    free(x_prev); free(x_curr);
    return DR_OK;
}

int dr_ge_robust_huber(const dr_problem_t *prob, double *x_out,
                       const dr_ge_robust_params_t *params) {
    if (!prob || !x_out) return DR_ERR_NULL_POINTER;
    double cu = params ? params->tuning_constant : 1.345;
    int mi = params ? params->max_iterations : 50;
    double tl = params ? params->convergence_tol : 1e-6;
    return irls_solve(prob, x_out, huber_weight, cu, mi, tl);
}

int dr_ge_robust_biweight(const dr_problem_t *prob, double *x_out,
                          double c, int max_iter, double tol) {
    if (!prob || !x_out) return DR_ERR_NULL_POINTER;
    if (c <= 0.0) c = 4.685;
    return irls_solve(prob, x_out, biweight_weight, c, max_iter, tol);
}

int dr_ge_robust_cauchy(const dr_problem_t *prob, double *x_out,
                        double c, int max_iter, double tol) {
    if (!prob || !x_out) return DR_ERR_NULL_POINTER;
    if (c <= 0.0) c = 2.385;
    return irls_solve(prob, x_out, cauchy_weight, c, max_iter, tol);
}

int dr_ge_robust_fair(const dr_problem_t *prob, double *x_out,
                      double c, int max_iter, double tol) {
    if (!prob || !x_out) return DR_ERR_NULL_POINTER;
    if (c <= 0.0) c = 1.400;
    return irls_solve(prob, x_out, fair_weight, c, max_iter, tol);
}

void dr_ge_robust_params_init(dr_ge_robust_params_t *params) {
    if (!params) return;
    params->max_iterations = 50;
    params->convergence_tol = 1e-6;
    params->tuning_constant = 1.345;
    params->robust_function = 0;
    params->use_median_init = 1;
}
int dr_ge_test_power(const dr_problem_t *prob, const double *delta,
                     double alpha, double *power_out) {
    int i, j, m = prob->ncon, n = prob->nvar;
    if (!prob || !delta || !power_out) return DR_ERR_NULL_POINTER;
    if (m == 0) { *power_out = 0.0; return DR_OK; }
    double *r0 = (double *)calloc((size_t)m, sizeof(double));
    if (!r0) return DR_ERR_NULL_POINTER;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            r0[i] += prob->constraint_A[i * n + j] * delta[j];
    double *V = (double *)malloc((size_t)m * (size_t)m * sizeof(double));
    if (!V) { free(r0); return DR_ERR_NULL_POINTER; }
    dr_compute_V_matrix(prob, V);
    if (dr_mat_cholesky_raw(V, m) != 0) {
        free(V); free(r0); *power_out = 0.5; return DR_OK;
    }
    double *y = (double *)malloc((size_t)m * sizeof(double));
    if (!y) { free(V); free(r0); return DR_ERR_NULL_POINTER; }
    dr_mat_cholesky_solve_raw(V, r0, y, m);
    double lambda = 0.0;
    for (i = 0; i < m; i++) lambda += r0[i] * y[i];
    double chi2_crit = dr_meas_chi2_critical(m, alpha);
    double mu_nc = m + lambda;
    double var_nc = 2.0 * (m + 2.0 * lambda);
    if (var_nc < 1e-10) var_nc = 1e-10;
    double z = (chi2_crit - mu_nc) / sqrt(var_nc);
    double power = 1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)));
    if (power < 0.0) power = 0.0;
    if (power > 1.0) power = 1.0;
    *power_out = power;
    free(y); free(V); free(r0);
    return DR_OK;
}

int dr_ge_expected_value_test(const dr_problem_t *prob, const double *x_hat,
                              dr_ge_test_result_t *results, double alpha) {
    int i, n = prob->nvar;
    if (!prob || !x_hat || !results) return DR_ERR_NULL_POINTER;
    double z_crit = dr_meas_normal_critical(alpha / 2.0);
    for (i = 0; i < n; i++) {
        results[i].test_type = 3;
        if (!prob->measurements[i].is_present) {
            results[i].statistic = 0.0; results[i].critical_value = z_crit;
            results[i].reject_H0 = 0; results[i].suspect_index = -1;
            results[i].p_value = 1.0; continue;
        }
        double adj = x_hat[i] - prob->measurements[i].value;
        double z_evt = fabs(adj) / prob->measurements[i].stddev;
        results[i].statistic = z_evt;
        results[i].critical_value = z_crit;
        results[i].reject_H0 = (z_evt > z_crit) ? 1 : 0;
        results[i].suspect_index = i;
        double pv = 2.0 * (1.0 - 0.5 * (1.0 + erf(z_evt / sqrt(2.0))));
        results[i].p_value = (pv < 0.0) ? 0.0 : ((pv > 1.0) ? 1.0 : pv);
    }
    return DR_OK;
}

int dr_ge_pca_detection(const dr_problem_t *prob, int *n_components,
                        double *scores, double alpha) {
    int i, j, k, m = prob->ncon, n = prob->nvar;
    (void)alpha;
    if (!prob || !n_components || !scores) return DR_ERR_NULL_POINTER;
    if (m == 0) { *n_components = 0; return DR_OK; }
    double *V = (double *)malloc((size_t)m * (size_t)m * sizeof(double));
    if (!V) return DR_ERR_NULL_POINTER;
    dr_compute_V_matrix(prob, V);
    double *r = (double *)malloc((size_t)m * sizeof(double));
    if (!r) { free(V); return DR_ERR_NULL_POINTER; }
    for (i = 0; i < m; i++) {
        double sum = 0.0;
        for (j = 0; j < n; j++)
            sum += prob->constraint_A[i * n + j] * prob->measurements[j].value;
        r[i] = sum - prob->constraint_b[i];
    }
    double *v = (double *)malloc((size_t)m * sizeof(double));
    if (!v) { free(r); free(V); return DR_ERR_NULL_POINTER; }
    double norm_r = 0.0;
    for (i = 0; i < m; i++) norm_r += r[i] * r[i];
    if (norm_r < 1e-15) {
        double inv_sqrt_m = 1.0 / sqrt((double)m);
        for (i = 0; i < m; i++) v[i] = inv_sqrt_m;
    } else {
        double inv_norm = 1.0 / sqrt(norm_r);
        for (i = 0; i < m; i++) v[i] = r[i] * inv_norm;
    }
    for (k = 0; k < 20; k++) {
        double *w = (double *)calloc((size_t)m, sizeof(double));
        if (!w) break;
        for (i = 0; i < m; i++)
            for (j = 0; j < m; j++)
                w[i] += V[i * m + j] * v[j];
        double norm_w = 0.0;
        for (i = 0; i < m; i++) norm_w += w[i] * w[i];
        if (norm_w < 1e-15) { free(w); break; }
        double inv_norm = 1.0 / sqrt(norm_w);
        for (i = 0; i < m; i++) v[i] = w[i] * inv_norm;
        free(w);
    }
    double pc_score = 0.0;
    for (i = 0; i < m; i++) pc_score += r[i] * v[i];
    for (i = 0; i < m; i++) scores[i] = v[i] * pc_score;
    double *Vv = (double *)calloc((size_t)m, sizeof(double));
    double lambda_est = 0.0;
    if (Vv) {
        for (i = 0; i < m; i++)
            for (j = 0; j < m; j++)
                Vv[i] += V[i * m + j] * v[j];
        for (i = 0; i < m; i++) lambda_est += v[i] * Vv[i];
        free(Vv);
    }
    *n_components = (lambda_est > 0.01) ? 1 : 0;
    free(v); free(r); free(V);
    return DR_OK;
}
int dr_ge_nodal_test(const dr_problem_t *prob, dr_ge_test_result_t *results,
                     double alpha) {
    int i, j, m, n;
    double *V, *r;
    if (!prob || !results) return DR_ERR_NULL_POINTER;
    m = prob->ncon; n = prob->nvar;
    if (m == 0) return DR_OK;
    V = (double *)malloc((size_t)m * (size_t)m * sizeof(double));
    r = (double *)malloc((size_t)m * sizeof(double));
    if (!V || !r) { free(V); free(r); return DR_ERR_NULL_POINTER; }
    if (dr_compute_V_matrix(prob, V) != DR_OK) {
        free(V); free(r); return DR_ERR_SINGULAR;
    }
    for (i = 0; i < m; i++) {
        double sum = 0.0;
        for (j = 0; j < n; j++)
            sum += prob->constraint_A[i * n + j] * prob->measurements[j].value;
        r[i] = sum - prob->constraint_b[i];
    }
    double alpha_bonf = alpha / (double)m;
    double z_crit = dr_meas_normal_critical(alpha_bonf / 2.0);
    for (i = 0; i < m; i++) {
        double v_ii = V[i * m + i];
        results[i].test_type = 1;
        if (v_ii < 1e-15) {
            results[i].statistic = 0.0; results[i].critical_value = z_crit;
            results[i].reject_H0 = 0; results[i].suspect_index = -1;
            results[i].p_value = 1.0; continue;
        }
        double stat = fabs(r[i]) / sqrt(v_ii);
        results[i].statistic = stat;
        results[i].critical_value = z_crit;
        results[i].reject_H0 = (stat > z_crit) ? 1 : 0;
        results[i].suspect_index = i;
        double pv = 2.0 * (1.0 - 0.5 * (1.0 + erf(stat / sqrt(2.0))));
        results[i].p_value = (pv < 0.0) ? 0.0 : ((pv > 1.0) ? 1.0 : pv);
    }
    free(V); free(r);
    return DR_OK;
}
int dr_ge_measurement_test(const dr_problem_t *prob,
                           dr_ge_test_result_t *results, double alpha) {
    int i, j, k, p, q;
    int m = prob->ncon, n = prob->nvar;
    if (!prob || !results) return DR_ERR_NULL_POINTER;
    if (m == 0) {
        for (i = 0; i < n; i++) {
            results[i].test_type = 2; results[i].statistic = 0.0;
            results[i].critical_value = 0.0; results[i].reject_H0 = 0;
            results[i].suspect_index = -1; results[i].p_value = 1.0;
        }
        return DR_OK;
    }
    double *Sigma = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
    for (i = 0; i < n; i++)
        Sigma[i * n + i] = prob->measurements[i].is_present
            ? prob->measurements[i].stddev * prob->measurements[i].stddev : 1e10;
    double *V = (double *)malloc((size_t)m * (size_t)m * sizeof(double));
    if (!V) { free(Sigma); return DR_ERR_NULL_POINTER; }
    dr_compute_V_matrix(prob, V);
    if (dr_mat_cholesky_raw(V, m) != 0) {
        free(V); free(Sigma);
        for (i = 0; i < n; i++) { results[i].reject_H0 = 0; results[i].statistic = 0.0; }
        return DR_ERR_SINGULAR;
    }
    double *r = (double *)malloc((size_t)m * sizeof(double));
    if (!r) { free(V); free(Sigma); return DR_ERR_NULL_POINTER; }
    for (i = 0; i < m; i++) {
        double sum = 0.0;
        for (j = 0; j < n; j++)
            sum += prob->constraint_A[i * n + j] * prob->measurements[j].value;
        r[i] = sum - prob->constraint_b[i];
    }
    double *lambda = (double *)malloc((size_t)m * sizeof(double));
    if (!lambda) { free(r); free(V); free(Sigma); return DR_ERR_NULL_POINTER; }
    dr_mat_cholesky_solve_raw(V, r, lambda, m);
    double *Vinv = (double *)malloc((size_t)m * (size_t)m * sizeof(double));
    if (!Vinv) { free(lambda); free(r); free(V); free(Sigma); return DR_ERR_NULL_POINTER; }
    for (i = 0; i < m * m; i++) Vinv[i] = 0.0;
    for (i = 0; i < m; i++) Vinv[i * m + i] = 1.0;
    for (j = 0; j < m; j++) {
        double *b = (double *)calloc((size_t)m, sizeof(double));
        double *x = (double *)malloc((size_t)m * sizeof(double));
        if (!b || !x) { free(x); free(b); break; }
        b[j] = 1.0;
        dr_mat_cholesky_solve_raw(V, b, x, m);
        for (i = 0; i < m; i++) Vinv[i * m + j] = x[i];
        free(x); free(b);
    }
    double z_crit = dr_meas_normal_critical(alpha / 2.0);
    for (i = 0; i < n; i++) {
        results[i].test_type = 2;
        if (!prob->measurements[i].is_present) {
            results[i].statistic = 0.0; results[i].critical_value = z_crit;
            results[i].reject_H0 = 0; results[i].suspect_index = -1;
            results[i].p_value = 1.0; continue;
        }
        double d_i = 0.0;
        for (k = 0; k < m; k++)
            d_i += Sigma[i * n + i] * prob->constraint_A[k * n + i] * lambda[k];
        double w_ii = 0.0;
        for (p = 0; p < m; p++) {
            double a_pi = prob->constraint_A[p * n + i];
            if (fabs(a_pi) < 1e-12) continue;
            for (q = 0; q < m; q++) {
                double a_qi = prob->constraint_A[q * n + i];
                if (fabs(a_qi) < 1e-12) continue;
                w_ii += a_pi * Vinv[p * m + q] * a_qi;
            }
        }
        w_ii *= Sigma[i * n + i] * Sigma[i * n + i];
        if (w_ii < 1e-15) {
            results[i].statistic = 0.0; results[i].critical_value = z_crit;
            results[i].reject_H0 = 0; results[i].suspect_index = -1;
            results[i].p_value = 1.0; continue;
        }
        double z_mt = fabs(d_i) / sqrt(w_ii);
        results[i].statistic = z_mt; results[i].critical_value = z_crit;
        results[i].reject_H0 = (z_mt > z_crit) ? 1 : 0;
        results[i].suspect_index = i;
        double pv = 2.0 * (1.0 - 0.5 * (1.0 + erf(z_mt / sqrt(2.0))));
        results[i].p_value = (pv < 0.0) ? 0.0 : ((pv > 1.0) ? 1.0 : pv);
    }
    free(Vinv); free(lambda); free(r); free(V); free(Sigma);
    return DR_OK;
}

dr_ge_identification_t *dr_ge_identification_create(int nvar) {
    dr_ge_identification_t *ident;
    if (nvar <= 0) return NULL;
    ident = (dr_ge_identification_t *)calloc(1, sizeof(dr_ge_identification_t));
    if (!ident) return NULL;
    ident->error_indices = (int *)calloc((size_t)nvar, sizeof(int));
    ident->error_magnitudes = (double *)calloc((size_t)nvar, sizeof(double));
    ident->error_stddevs = (double *)calloc((size_t)nvar, sizeof(double));
    if (!ident->error_indices || !ident->error_magnitudes || !ident->error_stddevs) {
        dr_ge_identification_free(ident); return NULL;
    }
    return ident;
}

void dr_ge_identification_free(dr_ge_identification_t *ident) {
    if (!ident) return;
    free(ident->error_indices);
    free(ident->error_magnitudes);
    free(ident->error_stddevs);
    free(ident);
}

int dr_ge_serial_elimination(const dr_problem_t *prob,
                             dr_ge_identification_t *ident, double alpha) {
    int i, k, m = prob->ncon, n = prob->nvar, n_errors = 0;
    if (!prob || !ident) return DR_ERR_NULL_POINTER;
    dr_problem_t *work = dr_problem_create(n, m);
    if (!work) return DR_ERR_NULL_POINTER;
    for (i = 0; i < n; i++)
        dr_set_measurement(work, i, prob->measurements[i].value,
                           prob->measurements[i].stddev, prob->measurements[i].tag_id);
    for (i = 0; i < m; i++)
        dr_set_constraint(work, i, &prob->constraint_A[i * n],
                          prob->constraint_b[i], prob->constraint_types[i]);
    int *eliminated = (int *)calloc((size_t)n, sizeof(int));
    if (!eliminated) { dr_problem_free(work); return DR_ERR_NULL_POINTER; }
    for (k = 0; k < n; k++) {
        dr_ge_test_result_t gt_result;
        if (dr_ge_global_test(work, &gt_result, alpha) != DR_OK) break;
        if (!gt_result.reject_H0) break;
        dr_ge_test_result_t *mt_results =
            (dr_ge_test_result_t *)malloc((size_t)n * sizeof(dr_ge_test_result_t));
        if (!mt_results) break;
        if (dr_ge_measurement_test(work, mt_results, alpha) != DR_OK) {
            free(mt_results); break;
        }
        int worst_idx = -1; double worst_stat = -1.0;
        for (i = 0; i < n; i++) {
            if (eliminated[i]) continue;
            if (!work->measurements[i].is_present) continue;
            if (mt_results[i].statistic > worst_stat) {
                worst_stat = mt_results[i].statistic; worst_idx = i;
            }
        }
        free(mt_results);
        if (worst_idx < 0) break;
        ident->error_indices[n_errors] = worst_idx;
        ident->error_magnitudes[n_errors] = work->measurements[worst_idx].value;
        ident->error_stddevs[n_errors] = work->measurements[worst_idx].stddev;
        n_errors++;
        eliminated[worst_idx] = 1;
        work->measurements[worst_idx].is_present = 0;
        work->measurements[worst_idx].stddev = 1e5;
        work->nmeas--;
    }
    ident->n_errors_found = n_errors;
    ident->iterations = k; ident->final_objective = 0.0;
    free(eliminated); dr_problem_free(work);
    return DR_OK;
}

int dr_ge_simultaneous_estimation(const dr_problem_t *prob,
                                  dr_ge_identification_t *ident,
                                  const int *suspects, int n_suspects) {
    int s, n;
    if (!prob || !ident || !suspects || n_suspects <= 0)
        return DR_ERR_NULL_POINTER;
    n = prob->nvar;
    dr_ge_test_result_t *mt = (dr_ge_test_result_t *)
        malloc((size_t)n * sizeof(dr_ge_test_result_t));
    if (!mt) return DR_ERR_NULL_POINTER;
    if (dr_ge_measurement_test(prob, mt, 0.05) != DR_OK) {
        free(mt); return DR_ERR_SINGULAR;
    }
    ident->n_errors_found = n_suspects;
    for (s = 0; s < n_suspects; s++) {
        int idx = suspects[s];
        if (idx < 0 || idx >= n) continue;
        ident->error_indices[s] = idx;
        ident->error_magnitudes[s] = mt[idx].statistic * prob->measurements[idx].stddev;
        ident->error_stddevs[s] = prob->measurements[idx].stddev;
    }
    ident->iterations = 1; ident->final_objective = 0.0;
    free(mt); return DR_OK;
}
