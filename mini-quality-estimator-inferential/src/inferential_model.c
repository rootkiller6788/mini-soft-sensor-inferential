/**
 * @file inferential_model.c
 * @brief Implementation of inferential model structures: FPM, ARX, state-space, hybrid, LW-PLS, MHE.
 */

#include "inferential_model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*===========================================================================
 * L2: First-Principles Model
 *===========================================================================*/

void fpm_init(fpm_model_t *model,
              double (*evaluate)(const double *, int, const double *, int),
              const double *params, int n_params, int n_states)
{
    if (!model) return;
    memset(model, 0, sizeof(fpm_model_t));
    model->evaluate = evaluate;
    model->n_params = (n_params > 64) ? 64 : n_params;
    model->n_states = (n_states > 16) ? 16 : n_states;
    if (params && n_params > 0) {
        memcpy(model->params, params, model->n_params * sizeof(double));
    }
}

double fpm_evaluate(const fpm_model_t *model, const double *inputs, int n_inputs)
{
    if (!model || !model->evaluate || !inputs || n_inputs <= 0) {
        return NAN;
    }
    return model->evaluate(inputs, n_inputs, model->params, model->n_params);
}

void fpm_sensitivity(const fpm_model_t *model, const double *inputs, int n_inputs,
                     double *jacobian, double h)
{
    if (!model || !inputs || !jacobian || n_inputs <= 0) return;

    double *x_pert = (double *)malloc(n_inputs * sizeof(double));
    if (!x_pert) return;

    double f_center = fpm_evaluate(model, inputs, n_inputs);
    if (isnan(f_center)) { free(x_pert); return; }

    if (h <= 0.0) h = 0.001;  /* Default step */

    for (int i = 0; i < n_inputs; i++) {
        memcpy(x_pert, inputs, n_inputs * sizeof(double));
        x_pert[i] += h;
        double f_plus  = fpm_evaluate(model, x_pert, n_inputs);
        x_pert[i] = inputs[i] - h;
        double f_minus = fpm_evaluate(model, x_pert, n_inputs);

        if (!isnan(f_plus) && !isnan(f_minus)) {
            jacobian[i] = (f_plus - f_minus) / (2.0 * h);
        } else {
            /* Fall back to forward differences */
            x_pert[i] = inputs[i] + h;
            f_plus = fpm_evaluate(model, x_pert, n_inputs);
            if (!isnan(f_plus)) {
                jacobian[i] = (f_plus - f_center) / h;
            } else {
                jacobian[i] = 0.0;
            }
        }
    }
    free(x_pert);
}

/*===========================================================================
 * L3: State-Space Model
 *===========================================================================*/

void ss_model_alloc(ss_model_t *model, int n_states, int n_inputs, int n_outputs)
{
    if (!model) return;
    memset(model, 0, sizeof(ss_model_t));
    if (n_states <= 0) return;

    model->n_states = n_states;
    model->n_inputs = n_inputs;
    model->n_outputs = n_outputs;

    int ns2 = n_states * n_states;

    model->A = (double *)calloc(ns2, sizeof(double));
    model->B = (double *)calloc(n_states * n_inputs, sizeof(double));
    model->C = (double *)calloc(n_outputs * n_states, sizeof(double));
    model->D = (double *)calloc(n_outputs * n_inputs, sizeof(double));
    model->Q = (double *)calloc(ns2, sizeof(double));
    model->R = (double *)calloc(n_outputs * n_outputs, sizeof(double));
    model->x = (double *)calloc(n_states, sizeof(double));
    model->P = (double *)calloc(ns2, sizeof(double));
    model->x_prior = (double *)calloc(n_states, sizeof(double));
    model->P_prior = (double *)calloc(ns2, sizeof(double));

    /* Set A to identity */
    if (model->A) {
        for (int i = 0; i < n_states; i++) {
            model->A[i * n_states + i] = 1.0;
        }
    }
    /* Set P to large diagonal (high initial uncertainty) */
    if (model->P) {
        for (int i = 0; i < n_states; i++) {
            model->P[i * n_states + i] = 1000.0;
        }
    }
}

void ss_model_free(ss_model_t *model)
{
    if (!model) return;
    free(model->A); free(model->B); free(model->C); free(model->D);
    free(model->Q); free(model->R); free(model->x); free(model->P);
    free(model->x_prior); free(model->P_prior);
    memset(model, 0, sizeof(ss_model_t));
}

void ss_model_predict(ss_model_t *model, const double *u)
{
    if (!model || !model->A || !model->x) return;

    int n = model->n_states;
    int m = model->n_inputs;

    /* x_prior = A * x */
    memset(model->x_prior, 0, n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            model->x_prior[i] += model->A[i * n + j] * model->x[j];
        }
    }

    /* x_prior += B * u */
    if (u && m > 0 && model->B) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                model->x_prior[i] += model->B[i * m + j] * u[j];
            }
        }
    }

    /* P_prior = A * P * A^T + Q */
    /* Compute temp = P * A^T, then A * temp, then add Q */
    double *temp = (double *)calloc(n * n, sizeof(double));
    if (!temp) return;

    /* temp = P * A^T */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += model->P[i * n + k] * model->A[j * n + k]; /* A^T[k,j] = A[j,k] */
            }
            temp[i * n + j] = sum;
        }
    }
    /* P_prior = A * temp */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += model->A[i * n + k] * temp[k * n + j];
            }
            model->P_prior[i * n + j] = sum;
        }
    }
    /* P_prior += Q */
    if (model->Q) {
        for (int i = 0; i < n * n; i++) {
            model->P_prior[i] += model->Q[i];
        }
    }
    free(temp);
}

void ss_model_output(const ss_model_t *model, const double *u, double *y)
{
    if (!model || !y) return;
    int n = model->n_states;
    int m = model->n_inputs;
    int p = model->n_outputs;

    /* y = C * x */
    if (model->C && model->x) {
        for (int i = 0; i < p; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += model->C[i * n + j] * model->x[j];
            }
            y[i] = sum;
        }
    } else {
        y[0] = model->x ? model->x[0] : 0.0;
    }

    /* y += D * u */
    if (u && m > 0 && model->D) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < m; j++) {
                y[i] += model->D[i * m + j] * u[j];
            }
        }
    }
}

/*===========================================================================
 * L3: ARX Model
 *===========================================================================*/

void arx_init(arx_model_t *model, int na, int nb, int n_inputs, int nk)
{
    if (!model) return;
    memset(model, 0, sizeof(arx_model_t));
    model->na = (na >= 0 && na < QEST_MAX_LAGS) ? na : 0;
    model->nb = (nb >= 0 && nb < QEST_MAX_LAGS) ? nb : 0;
    model->n_inputs = (n_inputs > 0 && n_inputs <= QEST_MAX_INPUT_VARS) ? n_inputs : 1;
    model->nk = (nk >= 0) ? nk : 0;
}

void arx_set_coeffs(arx_model_t *model, const double *a, const double *b)
{
    if (!model) return;
    if (a) memcpy(model->a_coeffs, a, model->na * sizeof(double));
    if (b) memcpy(model->b_coeffs, b,
                  model->n_inputs * (model->nb + 1) * sizeof(double));
}

double arx_predict(arx_model_t *model, const double *u)
{
    if (!model || !u) return 0.0;

    double y_pred = 0.0;

    /* Sum of b*u terms (current + past) */
    for (int j = 0; j < model->n_inputs; j++) {
        int b_offset = j * (model->nb + 1);
        /* Current u (delay nk applied when storing) */
        /* For simplicity, we apply nk as a buffer offset */
        int base_pos = model->u_buffer_pos;
        for (int i = 0; i <= model->nb; i++) {
            int hist_idx = (base_pos - 1 - i - model->nk);
            while (hist_idx < 0) hist_idx += QEST_MAX_LAGS;
            hist_idx = hist_idx % QEST_MAX_LAGS;
            y_pred += model->b_coeffs[b_offset + i]
                    * model->u_past[hist_idx * QEST_MAX_INPUT_VARS + j];
        }
    }

    /* Sum of a*y terms (past outputs) */
    for (int i = 0; i < model->na; i++) {
        int hist_idx = (model->y_buffer_pos - 1 - i);
        while (hist_idx < 0) hist_idx += QEST_MAX_LAGS;
        y_pred -= model->a_coeffs[i] * model->y_past[hist_idx];
    }

    return y_pred;
}

void arx_update_buffers(arx_model_t *model, double y_true, const double *u)
{
    if (!model) return;

    /* Store output */
    model->y_past[model->y_buffer_pos] = y_true;
    model->y_buffer_pos = (model->y_buffer_pos + 1) % QEST_MAX_LAGS;

    /* Store inputs */
    if (u) {
        for (int j = 0; j < model->n_inputs; j++) {
            model->u_past[model->u_buffer_pos * QEST_MAX_INPUT_VARS + j] = u[j];
        }
        model->u_buffer_pos = (model->u_buffer_pos + 1) % QEST_MAX_LAGS;
    }

    if (!model->y_buffer_filled && model->y_buffer_pos == 0) {
        model->y_buffer_filled = 1;
    }
}

/*===========================================================================
 * L2: Hybrid Grey-Box Model
 *===========================================================================*/

void hybrid_init(hybrid_model_t *model, const fpm_model_t *fpm)
{
    if (!model) return;
    memset(model, 0, sizeof(hybrid_model_t));
    if (fpm) {
        memcpy(&model->fpm, fpm, sizeof(fpm_model_t));
    }
    model->use_correction = 1;
    model->corr_bias  = 0.0;
    model->corr_scale = 0.0;
}

double hybrid_evaluate(const hybrid_model_t *model, const double *inputs, int n_inputs)
{
    if (!model || !inputs) return NAN;

    double y_fpm = fpm_evaluate(&model->fpm, inputs, n_inputs);
    if (isnan(y_fpm)) return NAN;

    if (!model->use_correction) return y_fpm;

    double y_correction = model->corr_bias;
    /* Linear correction term */
    y_correction += linear_model_evaluate(&model->corr_linear, inputs, n_inputs)
                  - model->corr_linear.intercept;  /* Remove intercept (already in bias) */

    return y_fpm * (1.0 + model->corr_scale) + y_correction;
}

void hybrid_set_correction(hybrid_model_t *model, double bias, double scale)
{
    if (!model) return;
    model->corr_bias  = bias;
    model->corr_scale = scale;
}

/*===========================================================================
 * L3: Moving Horizon Estimation
 *===========================================================================*/

void mhe_init(mhe_buffer_t *buf, int horizon, double arr_cost, double meas_wt)
{
    if (!buf) return;
    memset(buf, 0, sizeof(mhe_buffer_t));
    buf->horizon_length = (horizon > 0 && horizon < QEST_MAX_LAGS) ? horizon : 10;
    buf->arrival_cost_weight = arr_cost;
    buf->measurement_noise_weight = meas_wt;
}

void mhe_push(mhe_buffer_t *buf, const double *inputs, int n_in,
              double y_lab, const qest_timestamp_t *ts)
{
    if (!buf || !inputs) return;

    int idx = buf->current_index;
    int h   = buf->horizon_length;

    /* Store inputs */
    for (int i = 0; i < n_in && i < QEST_MAX_INPUT_VARS; i++) {
        buf->x_buffer[idx * QEST_MAX_INPUT_VARS + i] = inputs[i];
    }
    /* Store lab value */
    buf->y_buffer[idx] = y_lab;
    /* Store timestamp */
    if (ts) memcpy(&buf->t_buffer[idx], ts, sizeof(qest_timestamp_t));

    buf->current_index = (idx + 1) % h;
    if (buf->n_samples_stored < h) buf->n_samples_stored++;
}

void mhe_estimate(const mhe_buffer_t *buf, const linear_model_t *model,
                  int n_inputs, double *estimate, double *variance)
{
    if (!buf || !model || !estimate) return;

    int h = buf->horizon_length;
    int n = buf->n_samples_stored;
    if (n == 0) { *estimate = 0.0; if (variance) *variance = 1.0; return; }

    /* Simple MHE: compute weighted average of residuals */
    double sum_w = 0.0;
    double sum_wy = 0.0;
    double sum_wr2 = 0.0;

    for (int k = 0; k < n; k++) {
        int idx = (buf->current_index - 1 - k + h) % h;
        double y_lab = buf->y_buffer[idx];
        /* Model prediction at this point */
        double y_mod = model->intercept;
        for (int i = 0; i < n_inputs && i < model->n_inputs; i++) {
            y_mod += model->coefficients[i] * buf->x_buffer[idx * QEST_MAX_INPUT_VARS + i];
        }

        /* Exponential weighting: older data weighted less */
        double weight = exp(-(double)k * buf->measurement_noise_weight / (double)h);
        sum_w  += weight;
        sum_wy += weight * y_lab;
        double residual = y_lab - y_mod;
        sum_wr2 += weight * residual * residual;
    }

    *estimate = sum_wy / sum_w;

    if (variance) {
        *variance = sum_wr2 / sum_w;
        if (*variance < 1e-10) *variance = 1e-10;
    }
}

/*===========================================================================
 * L4: Model Validation Functions
 *===========================================================================*/

void compute_residuals(const double *y_model, const double *y_lab,
                       double *residuals, int n)
{
    if (!y_model || !y_lab || !residuals) return;
    for (int i = 0; i < n; i++) {
        residuals[i] = y_lab[i] - y_model[i];
    }
}

void compute_regression_stats(const double *y_model, const double *y_lab,
                              int n, qest_performance_t *stats)
{
    if (!y_model || !y_lab || !stats || n < 2) return;

    double sum_e = 0.0, sum_e2 = 0.0, sum_abs_e = 0.0, sum_ape = 0.0;
    double sum_y = 0.0, sum_y2 = 0.0, sum_ym = 0.0, sum_ym2 = 0.0, sum_ym_y = 0.0;

    for (int i = 0; i < n; i++) {
        double e = y_lab[i] - y_model[i];
        sum_e   += e;
        sum_e2  += e * e;
        sum_abs_e += fabs(e);
        sum_ape += (fabs(y_lab[i]) > 1e-10) ? fabs(e / y_lab[i]) * 100.0 : 0.0;

        sum_y  += y_lab[i];
        sum_y2 += y_lab[i] * y_lab[i];
        sum_ym += y_model[i];
        sum_ym2 += y_model[i] * y_model[i];
        sum_ym_y += y_model[i] * y_lab[i];
    }

    double inv_n = 1.0 / (double)n;
    stats->mse  = sum_e2 * inv_n;
    stats->mae  = sum_abs_e * inv_n;
    stats->rmse = sqrt(stats->mse);
    stats->mape = sum_ape * inv_n;

    /* R-squared */
    double ss_res = sum_e2;
    double y_mean = sum_y * inv_n;
    double ss_tot = 0.0;
    for (int i = 0; i < n; i++) {
        double d = y_lab[i] - y_mean;
        ss_tot += d * d;
    }
    stats->r_squared = (ss_tot > 1e-10) ? 1.0 - ss_res / ss_tot : 0.0;
}

double durbin_watson_test(const double *residuals, int n)
{
    if (!residuals || n < 2) return 2.0; /* Default: no autocorrelation */

    double sum_diff2 = 0.0, sum_e2 = 0.0;
    for (int i = 1; i < n; i++) {
        double diff = residuals[i] - residuals[i - 1];
        sum_diff2 += diff * diff;
    }
    for (int i = 0; i < n; i++) {
        sum_e2 += residuals[i] * residuals[i];
    }

    if (sum_e2 < 1e-10) return 2.0;
    return sum_diff2 / sum_e2;
}

int t_test_zero_mean(const double *residuals, int n, double *t_stat)
{
    if (!residuals || n < 2) { if (t_stat) *t_stat = 0.0; return 1; }

    /* Compute mean and standard deviation */
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        sum   += residuals[i];
        sum_sq += residuals[i] * residuals[i];
    }
    double mean = sum / (double)n;
    double var  = (sum_sq - sum * sum / (double)n) / (double)(n - 1);
    if (var < 1e-10) { if (t_stat) *t_stat = 0.0; return 1; }
    double se = sqrt(var / (double)n);

    double t = mean / se;
    if (t_stat) *t_stat = t;

    /* Critical value for 95% confidence, n-1 degrees of freedom.
       Approximate: 1.96 for large n, using Student's t approximation.
       For simplicity, use z-test threshold for n > 30, t-table for smaller n.
    */
    double t_crit;
    if (n > 30) {
        t_crit = 1.96;
    } else if (n > 10) {
        t_crit = 2.0 + 0.5 / sqrt((double)n);
    } else {
        t_crit = 2.5 + 1.0 / sqrt((double)n);
    }

    return (fabs(t) < t_crit) ? 1 : 0;  /* 1 = cannot reject H0 (mean=0) */
}

int grubbs_outlier_test(double y_new, double y_pred, double residuals_stddev,
                        int n_hist, double alpha)
{
    if (residuals_stddev < 1e-10) return 0; /* No variability = no outlier test possible */
    if (n_hist < 3) return 0; /* Too few samples */

    double residual = y_new - y_pred;
    double G = fabs(residual) / residuals_stddev;

    /* Grubbs critical value approximation:
       G_crit = (n-1)/sqrt(n) * sqrt(t^2_{alpha/(2n), n-2} / (n-2 + t^2_{alpha/(2n), n-2}))
       Simplified for typical process control applications:
    */
    double t_alpha;
    if (alpha <= 0.01) t_alpha = 3.0;
    else if (alpha <= 0.05) t_alpha = 2.0;
    else t_alpha = 1.5;

    int df = n_hist - 2;
    if (df < 1) df = 1;
    double G_crit = ((double)(n_hist - 1) / sqrt((double)n_hist))
                  * sqrt((t_alpha * t_alpha) / (df + t_alpha * t_alpha));

    return (G > G_crit) ? 1 : 0;
}
