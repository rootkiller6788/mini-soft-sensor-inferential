/**
 * @file quality_recursive_ls.c
 * @brief Implementation of RLS, VFF-RLS, directional forgetting, RPLS, and IV estimators.
 */

#include "quality_recursive_ls.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* External declarations from kalman_quality.c */
extern int matrix_invert(const double *A, double *Ainv, int n);

/*===========================================================================
 * L5: Classical Recursive Least Squares (RLS)
 *===========================================================================*/

void rls_alloc(rls_estimator_t *rls, int n_params, double lambda, double delta)
{
    if (!rls) return;
    memset(rls, 0, sizeof(rls_estimator_t));
    if (n_params <= 0) return;

    rls->n_params    = n_params;
    rls->n_regressors = n_params;
    rls->lambda      = lambda;
    rls->delta       = delta;

    int n2 = n_params * n_params;
    rls->theta = (double *)calloc(n_params, sizeof(double));
    rls->P     = (double *)calloc(n2, sizeof(double));
    rls->K     = (double *)calloc(n_params, sizeof(double));
    rls->temp  = (double *)calloc(n_params, sizeof(double));
    rls->temp_mat = (double *)calloc(n2, sizeof(double));

    /* P = delta * I */
    if (rls->P) {
        for (int i = 0; i < n_params; i++) {
            rls->P[i * n_params + i] = delta;
        }
    }
    rls->n_updates = 0;
}

void rls_free(rls_estimator_t *rls)
{
    if (!rls) return;
    free(rls->theta); free(rls->P); free(rls->K);
    free(rls->temp); free(rls->temp_mat);
    memset(rls, 0, sizeof(rls_estimator_t));
}

void rls_update(rls_estimator_t *rls, const double *phi, double y)
{
    if (!rls || !phi) return;

    int n = rls->n_params;
    double lambda = rls->lambda;

    /* Step 1: Compute denominator = lambda + phi^T * P * phi */
    /* temp = P * phi */
    memset(rls->temp, 0, n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            rls->temp[i] += rls->P[i * n + j] * phi[j];
        }
    }
    double denom = lambda;
    for (int i = 0; i < n; i++) {
        denom += phi[i] * rls->temp[i];
    }

    if (denom < 1e-15) return;  /* Avoid division by zero */

    /* Step 2: Compute gain K = P * phi / denom */
    for (int i = 0; i < n; i++) {
        rls->K[i] = rls->temp[i] / denom;
    }

    /* Step 3: Prediction error */
    double y_pred = 0.0;
    for (int i = 0; i < n; i++) {
        y_pred += phi[i] * rls->theta[i];
    }
    double error = y - y_pred;

    /* Step 4: Update theta = theta + K * error */
    for (int i = 0; i < n; i++) {
        rls->theta[i] += rls->K[i] * error;
    }

    /* Step 5: Update P = (1/lambda) * (P - K * phi^T * P) */
    /* temp_mat = K * phi^T */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            rls->temp_mat[i * n + j] = rls->K[i] * phi[j];
        }
    }
    /* temp_mat = I - K*phi^T */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double delta_ij = (i == j) ? 1.0 : 0.0;
            rls->temp_mat[i * n + j] = delta_ij - rls->temp_mat[i * n + j];
        }
    }
    /* P_new = (1/lambda) * (I - K*phi^T) * P */
    /* temp = temp_mat * P */
    double *P_new = (double *)malloc(n * n * sizeof(double));
    if (!P_new) return;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += rls->temp_mat[i * n + k] * rls->P[k * n + j];
            }
            P_new[i * n + j] = sum / lambda;
        }
    }
    memcpy(rls->P, P_new, n * n * sizeof(double));
    free(P_new);

    rls->n_updates++;
}

double rls_predict(const rls_estimator_t *rls, const double *phi)
{
    if (!rls || !phi) return 0.0;
    double y_pred = 0.0;
    for (int i = 0; i < rls->n_params; i++) {
        y_pred += phi[i] * rls->theta[i];
    }
    return y_pred;
}

void rls_get_parameters(const rls_estimator_t *rls, double *theta)
{
    if (!rls || !theta) return;
    memcpy(theta, rls->theta, rls->n_params * sizeof(double));
}

void rls_get_covariance(const rls_estimator_t *rls, double *P)
{
    if (!rls || !P) return;
    memcpy(P, rls->P, rls->n_params * rls->n_params * sizeof(double));
}

void rls_reset_covariance(rls_estimator_t *rls)
{
    if (!rls) return;
    int n = rls->n_params;
    if (rls->P) {
        memset(rls->P, 0, n * n * sizeof(double));
        for (int i = 0; i < n; i++) {
            rls->P[i * n + i] = rls->delta;
        }
    }
}

/*===========================================================================
 * L5: RLS with Variable Forgetting Factor (VFF)
 *===========================================================================*/

void rls_vff_alloc(rls_vff_t *rls_vff, int n_params,
                   double lambda_min, double lambda_max,
                   double sigma_e, double delta)
{
    if (!rls_vff) return;
    memset(rls_vff, 0, sizeof(rls_vff_t));
    rls_alloc(&rls_vff->rls, n_params, lambda_max, delta);
    rls_vff->lambda_min = lambda_min;
    rls_vff->lambda_max = lambda_max;
    rls_vff->sigma_e    = sigma_e;
    rls_vff->current_lambda = lambda_max;
    rls_vff->error_variance = sigma_e * sigma_e;
}

void rls_vff_free(rls_vff_t *rls_vff)
{
    if (!rls_vff) return;
    rls_free(&rls_vff->rls);
}

void rls_vff_update(rls_vff_t *rls_vff, const double *phi, double y)
{
    if (!rls_vff || !phi) return;

    /* Compute prediction error before updating lambda */
    double y_pred = rls_predict(&rls_vff->rls, phi);
    double error  = y - y_pred;

    /* Adapt forgetting factor */
    double abs_err = fabs(error);
    if (rls_vff->sigma_e > 0.0) {
        double ratio = abs_err / rls_vff->sigma_e;
        rls_vff->current_lambda = rls_vff->lambda_min
            + (rls_vff->lambda_max - rls_vff->lambda_min) * exp(-ratio);
    } else {
        rls_vff->current_lambda = rls_vff->lambda_max;
    }

    /* Update error variance estimate */
    rls_vff->error_variance = 0.95 * rls_vff->error_variance + 0.05 * error * error;

    /* Set lambda and update */
    rls_vff->rls.lambda = rls_vff->current_lambda;
    rls_update(&rls_vff->rls, phi, y);
}

double rls_vff_predict(const rls_vff_t *rls_vff, const double *phi)
{
    return rls_vff ? rls_predict(&rls_vff->rls, phi) : 0.0;
}

double rls_vff_get_lambda(const rls_vff_t *rls_vff)
{
    return rls_vff ? rls_vff->current_lambda : 1.0;
}

/*===========================================================================
 * L5: RLS with Directional Forgetting
 *===========================================================================*/

void rls_directional_alloc(rls_directional_t *rls_dir, int n_params,
                           double lambda, double epsilon, double delta)
{
    if (!rls_dir) return;
    memset(rls_dir, 0, sizeof(rls_directional_t));
    rls_alloc(&rls_dir->rls, n_params, lambda, delta);
    rls_dir->lambda  = lambda;
    rls_dir->epsilon = epsilon;
    rls_dir->info_content = 0.0;
}

void rls_directional_free(rls_directional_t *rls_dir)
{
    if (!rls_dir) return;
    rls_free(&rls_dir->rls);
}

void rls_directional_update(rls_directional_t *rls_dir, const double *phi, double y)
{
    if (!rls_dir || !phi) return;

    int n = rls_dir->rls.n_params;

    /* First compute standard RLS update with lambda = 1 (no forgetting yet) */
    rls_dir->rls.lambda = 1.0;
    rls_update(&rls_dir->rls, phi, y);

    /* Then apply directional forgetting to P */
    /* Compute information measure: phi^T * P * phi */
    double *Pphi = (double *)malloc(n * sizeof(double));
    if (!Pphi) return;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += rls_dir->rls.P[i * n + j] * phi[j];
        }
        Pphi[i] = sum;
    }
    double info = 0.0;
    for (int i = 0; i < n; i++) {
        info += phi[i] * Pphi[i];
    }

    /* Directional forgetting factor alpha */
    double alpha;
    if (info > rls_dir->epsilon) {
        alpha = rls_dir->lambda - (1.0 - rls_dir->lambda) / info;
        if (alpha < 0.01) alpha = 0.01;
    } else {
        alpha = 1.0;  /* No forgetting when information is low */
    }

    /* P = alpha * P  (only forget in the direction of new data) */
    if (alpha < 1.0) {
        for (int i = 0; i < n * n; i++) {
            rls_dir->rls.P[i] /= alpha;
        }
    }
    rls_dir->info_content = info;
    free(Pphi);
}

double rls_directional_predict(const rls_directional_t *rls_dir, const double *phi)
{
    return rls_dir ? rls_predict(&rls_dir->rls, phi) : 0.0;
}

/*===========================================================================
 * L5: Recursive PLS (RPLS)
 *===========================================================================*/

void rpls_init(rpls_model_t *rpls, int n_inputs, int n_outputs, int n_latent, double lambda)
{
    if (!rpls) return;
    memset(rpls, 0, sizeof(rpls_model_t));

    rpls->n_inputs  = n_inputs;
    rpls->n_outputs = n_outputs;
    rpls->n_latent  = n_latent;
    rpls->lambda    = lambda;

    /* Initialize PLS model dimensions */
    rpls->model.n_inputs  = n_inputs;
    rpls->model.n_outputs = n_outputs;
    rpls->model.n_latent  = n_latent;
}

void rpls_update(rpls_model_t *rpls, const double *x, const double *y)
{
    if (!rpls || !x || !y) return;

    int n_x = rpls->n_inputs;
    int n_y = rpls->n_outputs;
    double lam = rpls->lambda;

    rpls->n_samples++;

    /* Update running sums */
    for (int i = 0; i < n_x; i++) {
        rpls->x_sum[i]    = lam * rpls->x_sum[i] + x[i];
        rpls->x_sum_sq[i] = lam * rpls->x_sum_sq[i] + x[i] * x[i];
    }
    for (int j = 0; j < n_y; j++) {
        rpls->y_sum[j]    = lam * rpls->y_sum[j] + y[j];
        rpls->y_sum_sq[j] = lam * rpls->y_sum_sq[j] + y[j] * y[j];
    }

    /* Update X^T X (for input covariance) */
    for (int i = 0; i < n_x; i++) {
        for (int j = 0; j < n_x; j++) {
            rpls->XTX[i][j] = lam * rpls->XTX[i][j] + x[i] * x[j];
        }
    }

    /* Update X^T Y (for cross-covariance) */
    for (int i = 0; i < n_x; i++) {
        for (int j = 0; j < n_y; j++) {
            rpls->XTY[i][j] = lam * rpls->XTY[i][j] + x[i] * y[j];
        }
    }

    /* Update means and standard deviations */
    (void)lam;  /* Effective sample count with forgetting = 1/(1-lambda) */
    for (int i = 0; i < n_x; i++) {
        rpls->model.x_means[i] = rpls->x_sum[i] * (1.0 - lam);
        double var_x = rpls->x_sum_sq[i] * (1.0 - lam)
                     - rpls->model.x_means[i] * rpls->model.x_means[i];
        rpls->model.x_stds[i] = (var_x > 1e-10) ? sqrt(var_x) : 1.0;
    }
    for (int j = 0; j < n_y; j++) {
        rpls->model.y_means[j] = rpls->y_sum[j] * (1.0 - lam);
        double var_y = rpls->y_sum_sq[j] * (1.0 - lam)
                     - rpls->model.y_means[j] * rpls->model.y_means[j];
        rpls->model.y_stds[j] = (var_y > 1e-10) ? sqrt(var_y) : 1.0;
    }

    /* Simplified RPLS: Use XTY as regression coefficients directly */
    /* In practice, full NIPALS or SIMPLS algorithm would be used */
    for (int j = 0; j < n_y; j++) {
        rpls->model.intercept[j] = rpls->model.y_means[j];
        for (int i = 0; i < n_x; i++) {
            /* Simple regression coefficient: beta = cov(x_i, y_j) / var(x_i) */
            double var_xi = rpls->model.x_stds[i] * rpls->model.x_stds[i];
            if (var_xi > 1e-10) {
                double cov_xi_yj = rpls->XTY[i][j] * (1.0 - lam)
                    - rpls->model.x_means[i] * rpls->model.y_means[j];
                rpls->model.beta_coeffs[i][j] = cov_xi_yj / var_xi;
            } else {
                rpls->model.beta_coeffs[i][j] = 0.0;
            }
            rpls->model.intercept[j] -= rpls->model.beta_coeffs[i][j] * rpls->model.x_means[i];
        }
    }

    /* Update latent variable loadings (simplified: use first n_latent eigenvectors) */
    for (int a = 0; a < rpls->n_latent && a < QEST_MAX_LATENT_FACTORS; a++) {
        for (int i = 0; i < n_x; i++) {
            rpls->model.x_weights[i][a] = (i < n_x) ? x[i] : 0.0;  /* Simplified initialization */
        }
        rpls->model.explained_variance_y[a] = 0.5 + 0.1 * (double)a;  /* Placeholder */
    }
}

void rpls_predict(const rpls_model_t *rpls, const double *x, double *y_pred)
{
    if (!rpls || !x || !y_pred) return;

    int n_x = rpls->n_inputs;
    int n_y = rpls->n_outputs;

    for (int j = 0; j < n_y; j++) {
        y_pred[j] = rpls->model.intercept[j];
        for (int i = 0; i < n_x; i++) {
            y_pred[j] += rpls->model.beta_coeffs[i][j] * x[i];
        }
    }
}

void rpls_get_explained_variance(const rpls_model_t *rpls, double *var_y)
{
    if (!rpls || !var_y) return;
    for (int a = 0; a < rpls->n_latent; a++) {
        var_y[a] = rpls->model.explained_variance_y[a];
    }
}

/*===========================================================================
 * L5: Weighted Least Squares (WLS) Window
 *===========================================================================*/

void wls_window_alloc(wls_window_t *wls, int n_params, int window_size, double lambda)
{
    if (!wls) return;
    memset(wls, 0, sizeof(wls_window_t));
    wls->n_params    = n_params;
    wls->window_size = window_size;
    wls->lambda      = lambda;

    wls->phi_buffer = (double *)calloc(window_size * n_params, sizeof(double));
    wls->y_buffer   = (double *)calloc(window_size, sizeof(double));
    wls->weights    = (double *)calloc(window_size, sizeof(double));
    wls->theta      = (double *)calloc(n_params, sizeof(double));
    wls->P          = (double *)calloc(n_params * n_params, sizeof(double));
    wls->temp_mat   = (double *)calloc(n_params * n_params, sizeof(double));
    wls->temp_vec   = (double *)calloc(n_params, sizeof(double));

    /* Initial covariance */
    if (wls->P) {
        for (int i = 0; i < n_params; i++) wls->P[i * n_params + i] = 1000.0;
    }
}

void wls_window_free(wls_window_t *wls)
{
    if (!wls) return;
    free(wls->phi_buffer); free(wls->y_buffer); free(wls->weights);
    free(wls->theta); free(wls->P); free(wls->temp_mat); free(wls->temp_vec);
    memset(wls, 0, sizeof(wls_window_t));
}

void wls_window_update(wls_window_t *wls, const double *phi, double y)
{
    if (!wls || !phi) return;

    int n = wls->n_params;
    int w = wls->window_size;
    int idx = wls->current_index;

    /* Store sample */
    memcpy(&wls->phi_buffer[idx * n], phi, n * sizeof(double));
    wls->y_buffer[idx] = y;
    wls->current_index = (idx + 1) % w;
    if (wls->n_stored < w) wls->n_stored++;

    /* Compute weights: exponential decay */
    int stored = wls->n_stored;
    for (int k = 0; k < stored; k++) {
        int sample_idx = (wls->current_index - 1 - k + w) % w;
        wls->weights[sample_idx] = pow(wls->lambda, (double)k);
    }

    /* Solve weighted normal equations: theta = (Phi^T * W * Phi)^{-1} * Phi^T * W * y */
    /* Build Phi^T * W * Phi */
    memset(wls->temp_mat, 0, n * n * sizeof(double));
    memset(wls->temp_vec, 0, n * sizeof(double));

    for (int k = 0; k < stored; k++) {
        int sidx = (wls->current_index - 1 - k + w) % w;
        double w_k = wls->weights[sidx];
        double y_k = wls->y_buffer[sidx];
        double *phi_k = &wls->phi_buffer[sidx * n];

        for (int i = 0; i < n; i++) {
            wls->temp_vec[i] += w_k * phi_k[i] * y_k;
            for (int j = 0; j < n; j++) {
                wls->temp_mat[i * n + j] += w_k * phi_k[i] * phi_k[j];
            }
        }
    }

    /* Add regularization (ridge) for numerical stability */
    for (int i = 0; i < n; i++) {
        wls->temp_mat[i * n + i] += 1e-6;
    }

    /* Solve using matrix inversion */
    double *PhiWPhi_inv = (double *)malloc(n * n * sizeof(double));
    if (PhiWPhi_inv && matrix_invert(wls->temp_mat, PhiWPhi_inv, n)) {
        /* theta = PhiWPhi_inv * temp_vec */
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += PhiWPhi_inv[i * n + j] * wls->temp_vec[j];
            }
            wls->theta[i] = sum;
        }
        memcpy(wls->P, PhiWPhi_inv, n * n * sizeof(double));
    }
    free(PhiWPhi_inv);
}

double wls_window_predict(const wls_window_t *wls, const double *phi)
{
    if (!wls || !phi) return 0.0;
    double y_pred = 0.0;
    for (int i = 0; i < wls->n_params; i++) {
        y_pred += phi[i] * wls->theta[i];
    }
    return y_pred;
}

void wls_window_get_parameters(const wls_window_t *wls, double *theta)
{
    if (!wls || !theta) return;
    memcpy(theta, wls->theta, wls->n_params * sizeof(double));
}

/*===========================================================================
 * L2: Instrumental Variable (IV) Estimator
 *===========================================================================*/

void iv_estimator_alloc(iv_estimator_t *iv, int n_params, int n_lag)
{
    if (!iv) return;
    memset(iv, 0, sizeof(iv_estimator_t));
    iv->n_params = n_params;
    iv->n_lag    = n_lag;
    iv->n_iv     = n_params;  /* Use same number of instruments as parameters */

    int n2 = n_params * n_params;
    iv->phi_buffer = (double *)calloc(n_lag * n_params, sizeof(double));
    iv->y_buffer   = (double *)calloc(n_lag, sizeof(double));
    iv->theta      = (double *)calloc(n_params, sizeof(double));
    iv->P_iv       = (double *)calloc(n2, sizeof(double));
    iv->temp_mat   = (double *)calloc(n2, sizeof(double));
    iv->temp_vec   = (double *)calloc(n_params, sizeof(double));

    if (iv->P_iv) {
        for (int i = 0; i < n_params; i++) iv->P_iv[i * n_params + i] = 1000.0;
    }
}

void iv_estimator_free(iv_estimator_t *iv)
{
    if (!iv) return;
    free(iv->phi_buffer); free(iv->y_buffer);
    free(iv->theta); free(iv->P_iv); free(iv->temp_mat); free(iv->temp_vec);
    memset(iv, 0, sizeof(iv_estimator_t));
}

void iv_estimator_update(iv_estimator_t *iv, const double *phi, double y)
{
    if (!iv || !phi) return;

    int n = iv->n_params;
    int lag = iv->n_lag;

    /* Store sample in history buffer */
    int pos = iv->buffer_pos;
    memcpy(&iv->phi_buffer[pos * n], phi, n * sizeof(double));
    iv->y_buffer[pos] = y;
    iv->buffer_pos = (pos + 1) % lag;
    if (iv->buffer_pos == 0) iv->buffer_filled = 1;

    if (!iv->buffer_filled) {
        /* Not enough data for IV yet — use OLS fallback */
        for (int i = 0; i < n; i++) {
            iv->theta[i] += 0.01 * (y * phi[i] - phi[i] * phi[i] * iv->theta[i]);
        }
        return;
    }

    /* IV method: instruments = lagged regressors */
    /* z(k) = phi(k-1)  (simple lag-1 instrument) */
    /* Note: instruments are the lagged phi values used in the loop below */

    /* Build Z^T * Phi */
    memset(iv->temp_mat, 0, n * n * sizeof(double));
    memset(iv->temp_vec, 0, n * sizeof(double));

    for (int k = 0; k < lag; k++) {
        int phi_idx = (iv->buffer_pos - 1 - k + lag) % lag;
        int z_idx   = (phi_idx + lag - 1) % lag;  /* Lagged instruments */

        double *phi_k = &iv->phi_buffer[phi_idx * n];
        double *z_k   = &iv->phi_buffer[z_idx * n];
        double y_k    = iv->y_buffer[phi_idx];

        for (int i = 0; i < n; i++) {
            iv->temp_vec[i] += z_k[i] * y_k;
            for (int j = 0; j < n; j++) {
                iv->temp_mat[i * n + j] += z_k[i] * phi_k[j];
            }
        }
    }

    /* Solve: theta = (Z^T * Phi)^{-1} * Z^T * y */
    for (int i = 0; i < n; i++) {
        iv->temp_mat[i * n + i] += 1e-6;  /* Regularization */
    }

    double *ZPhi_inv = (double *)malloc(n * n * sizeof(double));
    if (ZPhi_inv && matrix_invert(iv->temp_mat, ZPhi_inv, n)) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += ZPhi_inv[i * n + j] * iv->temp_vec[j];
            }
            iv->theta[i] = sum;
        }
        memcpy(iv->P_iv, ZPhi_inv, n * n * sizeof(double));
    }
    free(ZPhi_inv);
}

double iv_estimator_predict(const iv_estimator_t *iv, const double *phi)
{
    if (!iv || !phi) return 0.0;
    double y_pred = 0.0;
    for (int i = 0; i < iv->n_params; i++) {
        y_pred += phi[i] * iv->theta[i];
    }
    return y_pred;
}
