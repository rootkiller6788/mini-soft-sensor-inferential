/**
 * @file vfm_state_estimation.c
 * @brief State estimation algorithm implementations for VFM
 *
 * Implements Kalman filter, RLS, MHE, sensor fusion, CUSUM drift detection.
 *
 * @module mini-virtual-flow-meter
 */

#include "vfm_state_estimation.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ==========================================================================
 * L2: Kalman Filter Implementation
 * ========================================================================== */

/**
 * Matrix utility: set 3x3 identity matrix (row-major).
 */
static void mat3x3_identity(double *M)
{
    M[0] = 1.0; M[1] = 0.0; M[2] = 0.0;
    M[3] = 0.0; M[4] = 1.0; M[5] = 0.0;
    M[6] = 0.0; M[7] = 0.0; M[8] = 1.0;
}

/**
 * Matrix multiplication: C = A * B for 3x3 * 3x3 (row-major).
 */
static void mat3x3_mul(const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            double sum = 0.0;
            for (k = 0; k < 3; k++) {
                sum += A[i*3 + k] * B[k*3 + j];
            }
            C[i*3 + j] = sum;
        }
    }
}

/**
 * Matrix multiplication: C = A * B^T for 3x3 * 3x3 (row-major).
 * B^T is the transpose of B.
 */
static void mat3x3_mul_at_bt(const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            double sum = 0.0;
            for (k = 0; k < 3; k++) {
                sum += A[i*3 + k] * B[j*3 + k];  /* B transposed: use B[j*3+k] */
            }
            C[i*3 + j] = sum;
        }
    }
}

/**
 * Matrix addition: C = A + B (3x3 row-major).
 */
static void mat3x3_add(const double *A, const double *B, double *C)
{
    int i;
    for (i = 0; i < 9; i++) C[i] = A[i] + B[i];
}

/**
 * Vector-matrix multiply: y = H * x (1x3 * 3x1 => scalar).
 */
static double vec3_dot(const double *a, const double *b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

int vfm_kalman_init(vfm_kalman_t *kf, double proc_noise, double meas_noise)
{
    if (!kf) return -1;

    /* Initial state: zero flow, zero bias, zero drift */
    kf->x[0] = 0.0;  /* flow rate */
    kf->x[1] = 0.0;  /* bias */
    kf->x[2] = 0.0;  /* drift rate */

    /* Initial covariance: large diagonal (high uncertainty) */
    int i;
    for (i = 0; i < 9; i++) kf->P[i] = 0.0;
    kf->P[0] = 1.0;       /* Flow variance = 1.0 */
    kf->P[4] = 0.01;      /* Bias variance = 0.01 */
    kf->P[8] = 1e-6;      /* Drift variance = 1e-6 */

    /* State transition matrix A:
     * x[k+1] = A * x[k] = [1, 1, dt;
     *                       0, 1, dt;
     *                       0, 0, 1]
     *
     * Flow = previous_flow + bias + drift*dt
     * Bias = previous_bias + drift*dt
     * Drift = previous_drift (random walk)
     *
     * For dt=1 (unit sampling):
     */
    mat3x3_identity(kf->A);
    kf->A[1] = 1.0;   /* A[0][1] = 1: flow gets bias contribution */
    kf->A[2] = 1.0;   /* A[0][2] = dt: flow gets drift (dt=1) */
    kf->A[5] = 1.0;   /* A[1][2] = dt: bias gets drift (dt=1) */

    /* Measurement matrix H = [1, 1, 0]:
     * z = flow + bias (model-predicted flow corrected by bias)
     */
    kf->H[0] = 1.0;
    kf->H[1] = 1.0;
    kf->H[2] = 0.0;

    /* Process noise covariance Q:
     * Q = diag([sigma_flow^2, sigma_bias^2, sigma_drift^2])
     * scaled by process_noise^2.
     */
    double ps2 = proc_noise * proc_noise;
    for (i = 0; i < 9; i++) kf->Q[i] = 0.0;
    kf->Q[0] = ps2;         /* Flow process noise */
    kf->Q[4] = ps2 * 0.01;  /* Bias changes slowly */
    kf->Q[8] = ps2 * 1e-6;  /* Drift changes very slowly */

    /* Measurement noise variance */
    kf->R = meas_noise * meas_noise;

    /* Kalman gain initialized to zero */
    kf->K[0] = kf->K[1] = kf->K[2] = 0.0;
    kf->innovation     = 0.0;
    kf->innovation_cov = 0.0;

    kf->initialized = 1;
    kf->steps       = 0;

    return 0;
}

int vfm_kalman_predict(vfm_kalman_t *kf)
{
    if (!kf || !kf->initialized) return -1;

    /* State prediction: x_pred = A * x */
    double x_pred[3];
    x_pred[0] = kf->A[0]*kf->x[0] + kf->A[1]*kf->x[1] + kf->A[2]*kf->x[2];
    x_pred[1] = kf->A[3]*kf->x[0] + kf->A[4]*kf->x[1] + kf->A[5]*kf->x[2];
    x_pred[2] = kf->A[6]*kf->x[0] + kf->A[7]*kf->x[1] + kf->A[8]*kf->x[2];

    /* Covariance prediction: P_pred = A * P * A^T + Q */
    double P_AT[9];
    mat3x3_mul_at_bt(kf->P, kf->A, P_AT);   /* temp = P * A^T */
    double P_pred[9];
    mat3x3_mul(kf->A, P_AT, P_pred);         /* P_pred = A * temp */
    mat3x3_add(P_pred, kf->Q, kf->P);        /* P = P_pred + Q */

    /* Update state */
    kf->x[0] = x_pred[0];
    kf->x[1] = x_pred[1];
    kf->x[2] = x_pred[2];

    return 0;
}

int vfm_kalman_update(vfm_kalman_t *kf, double measurement)
{
    if (!kf || !kf->initialized) return -1;

    /*
     * Innovation: e = z - H * x
     * Innovation covariance: S = H * P * H^T + R
     * Kalman gain: K = P * H^T / S
     * State update: x = x + K * e
     * Covariance update: P = (I - K*H) * P
     */

    /* Predicted measurement: z_pred = H * x */
    double z_pred = vec3_dot(kf->H, kf->x);
    kf->innovation = measurement - z_pred;

    /* Compute P * H^T (3x1 vector) */
    double PHt[3];
    PHt[0] = kf->P[0]*kf->H[0] + kf->P[1]*kf->H[1] + kf->P[2]*kf->H[2];
    PHt[1] = kf->P[3]*kf->H[0] + kf->P[4]*kf->H[1] + kf->P[5]*kf->H[2];
    PHt[2] = kf->P[6]*kf->H[0] + kf->P[7]*kf->H[1] + kf->P[8]*kf->H[2];

    /* Innovation covariance S = H * PHt + R */
    kf->innovation_cov = vec3_dot(kf->H, PHt) + kf->R;

    if (kf->innovation_cov < 1e-15) {
        return -1;  /* Prevent division by zero */
    }

    /* Kalman gain K = PHt / S */
    kf->K[0] = PHt[0] / kf->innovation_cov;
    kf->K[1] = PHt[1] / kf->innovation_cov;
    kf->K[2] = PHt[2] / kf->innovation_cov;

    /* State update: x = x + K * e */
    kf->x[0] += kf->K[0] * kf->innovation;
    kf->x[1] += kf->K[1] * kf->innovation;
    kf->x[2] += kf->K[2] * kf->innovation;

    /* Covariance update: P = (I - K*H) * P
     * I - K*H is 3x3: identity minus outer product K * H
     */
    double I_KH[9];
    mat3x3_identity(I_KH);
    I_KH[0] -= kf->K[0] * kf->H[0];
    I_KH[1] -= kf->K[0] * kf->H[1];
    I_KH[2] -= kf->K[0] * kf->H[2];
    I_KH[3] -= kf->K[1] * kf->H[0];
    I_KH[4] -= kf->K[1] * kf->H[1];
    I_KH[5] -= kf->K[1] * kf->H[2];
    I_KH[6] -= kf->K[2] * kf->H[0];
    I_KH[7] -= kf->K[2] * kf->H[1];
    I_KH[8] -= kf->K[2] * kf->H[2];

    double P_new[9];
    mat3x3_mul(I_KH, kf->P, P_new);

    /* Copy back */
    int i;
    for (i = 0; i < 9; i++) kf->P[i] = P_new[i];

    kf->steps++;
    return 0;
}

int vfm_kalman_get_estimate(const vfm_kalman_t *kf, double *flow,
                             double *uncert)
{
    if (!kf || !kf->initialized) return -1;
    if (!flow || !uncert) return -1;

    *flow   = kf->x[0];
    *uncert = sqrt(kf->P[0]);  /* sqrt of flow variance */

    if (*uncert < 0.0) *uncert = 0.0;
    return 0;
}

int vfm_kalman_get_bias(const vfm_kalman_t *kf, double *bias)
{
    if (!kf || !kf->initialized || !bias) return -1;
    *bias = kf->x[1];
    return 0;
}

double vfm_kalman_adapt_noise(vfm_kalman_t *kf, double gamma)
{
    /*
     * Adaptive noise estimation using innovation sequence.
     *
     * Theory: If the filter is well-tuned, the innovation sequence
     * should be zero-mean with variance S = H*P*H^T + R.
     *
     * If actual innovation variance > S:  measurement noise underestimated → increase R
     * If actual innovation variance < S:  measurement noise overestimated → decrease R
     *
     * Update: R_new = R_old + gamma * (e^2 - S)
     *
     * Uses an exponential moving average for robustness.
     */

    if (!kf || !kf->initialized) return 0.0;

    /* Actual squared innovation */
    double e2 = kf->innovation * kf->innovation;

    /* Predicted innovation variance */
    double S_pred = kf->innovation_cov;

    if (gamma < 0.0) gamma = 0.0;
    if (gamma > 1.0) gamma = 1.0;

    /* Exponential moving average update */
    double R_new = kf->R + gamma * (e2 - S_pred);

    /* Clamp: R must stay positive */
    if (R_new < 1e-10) R_new = 1e-10;
    if (R_new > 1e10)  R_new = 1e10;

    kf->R = R_new;
    return R_new;
}

/* ==========================================================================
 * L3: Recursive Least Squares (RLS) Implementation
 * ========================================================================== */

int vfm_rls_init(vfm_rls_t *rls, int n_params, double lambda,
                  const double *theta0)
{
    if (!rls || !theta0) return -1;
    if (n_params < 1 || n_params > 4) return -1;

    rls->n_params = n_params;
    rls->lambda   = lambda;
    if (lambda < 0.95) lambda = 0.95;
    if (lambda > 1.0)  lambda = 1.0;

    rls->prediction_error = 0.0;
    rls->initialized = 1;
    rls->steps       = 0;

    /* Copy initial parameter estimates */
    int i;
    for (i = 0; i < n_params; i++) {
        rls->theta[i] = theta0[i];
    }

    /* Initialize covariance P = large * I (high initial uncertainty) */
    int n2 = n_params * n_params;
    for (i = 0; i < n2; i++) rls->P[i] = 0.0;
    double large_val = 1e6;
    for (i = 0; i < n_params; i++) {
        rls->P[i * n_params + i] = large_val;
    }

    return 0;
}

int vfm_rls_update(vfm_rls_t *rls, const double *phi, double y)
{
    /*
     * RLS update for single-output linear regression:
     *   y = phi^T * theta
     *
     * Step 1: prediction error e = y - phi^T * theta
     * Step 2: compute P*phi (vector of size n_params)
     * Step 3: denominator = lambda + phi^T * P * phi
     * Step 4: gain K = (P*phi) / denominator
     * Step 5: theta += K * e
     * Step 6: P = (1/lambda) * (P - K*(phi^T*P))
     *
     * Note: phi^T*P is a row vector; K*(phi^T*P) is outer product.
     */

    if (!rls || !phi || !rls->initialized) return -1;

    int n = rls->n_params;

    /* Step 1: prediction and error */
    double y_pred = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        y_pred += phi[i] * rls->theta[i];
    }
    rls->prediction_error = y - y_pred;

    /* Step 2: compute v = P * phi (size n x 1) */
    double v[4] = {0.0, 0.0, 0.0, 0.0};
    for (i = 0; i < n; i++) {
        double sum = 0.0;
        int j;
        for (j = 0; j < n; j++) {
            sum += rls->P[i * n + j] * phi[j];
        }
        v[i] = sum;
    }

    /* Step 3: denominator = lambda + phi^T * v */
    double phiT_v = 0.0;
    for (i = 0; i < n; i++) {
        phiT_v += phi[i] * v[i];
    }
    double denom = rls->lambda + phiT_v;

    if (fabs(denom) < 1e-15) return -1;

    /* Step 4: gain K = v / denom */
    double K[4];
    for (i = 0; i < n; i++) {
        K[i] = v[i] / denom;
    }

    /* Step 5: theta += K * e */
    for (i = 0; i < n; i++) {
        rls->theta[i] += K[i] * rls->prediction_error;
    }

    /* Step 6: P = (1/lambda) * (P - K*v^T)
     * where K*v^T is the outer product
     */
    double one_over_lambda = 1.0 / rls->lambda;
    for (i = 0; i < n; i++) {
        int j;
        for (j = 0; j < n; j++) {
            rls->P[i * n + j] = one_over_lambda * (
                rls->P[i * n + j] - K[i] * phi[j] * v[j]);
        }
    }

    rls->steps++;
    return 0;
}

int vfm_rls_get_params(const vfm_rls_t *rls, double *theta)
{
    if (!rls || !theta || !rls->initialized) return -1;
    int i;
    for (i = 0; i < rls->n_params; i++) {
        theta[i] = rls->theta[i];
    }
    return 0;
}

/* ==========================================================================
 * L4: Moving Horizon Estimation Implementation
 * ========================================================================== */

int vfm_mhe_init(vfm_mhe_t *mhe, int horizon, double q_weight,
                  double r_weight)
{
    if (!mhe || horizon < 2 || horizon > 100) return -1;

    mhe->horizon    = horizon;
    mhe->Q_weight   = q_weight;
    mhe->R_weight   = r_weight;
    mhe->current_idx = 0;
    mhe->filled      = 0;

    /* Allocate history buffers */
    mhe->x_history = (double *)calloc((size_t)horizon * 2, sizeof(double));
    mhe->y_history = (double *)calloc((size_t)horizon, sizeof(double));

    if (!mhe->x_history || !mhe->y_history) {
        free(mhe->x_history);
        free(mhe->y_history);
        mhe->x_history = NULL;
        mhe->y_history = NULL;
        return -1;
    }

    mhe->x_estimate[0] = 0.0;
    mhe->x_estimate[1] = 0.0;

    /* Arrival cost: large initial uncertainty */
    mhe->arrival_cost_P[0] = 1e6;
    mhe->arrival_cost_P[1] = 0.0;
    mhe->arrival_cost_P[2] = 0.0;
    mhe->arrival_cost_P[3] = 1e6;

    return 0;
}

int vfm_mhe_step(vfm_mhe_t *mhe, double measurement, double model_pred,
                  double *estimate)
{
    /*
     * Simplified MHE: uses a weighted moving average over the horizon
     * with exponential forgetting. This provides a computationally
     * cheap approximation of full MHE for online VFM applications.
     *
     * The estimate is a weighted combination of:
     *   - Model prediction (from physics-based VFM model)
     *   - Recent measurements (weighted by recency)
     *
     * Cost: J = sum_{k} w_k * (y_k - x)^2 + q * (x - x_model)^2
     * Optimal: x* = (sum w_k*y_k + q*x_model) / (sum w_k + q)
     */

    if (!mhe || !estimate) return -1;

    int idx = mhe->current_idx;
    int N = mhe->horizon;

    /* Store new measurement and model prediction */
    mhe->y_history[idx] = measurement;
    mhe->x_history[idx * 2]     = model_pred;
    mhe->x_history[idx * 2 + 1] = measurement;

    /* Update circular buffer index */
    mhe->current_idx = (idx + 1) % N;
    if (mhe->current_idx == 0) mhe->filled = 1;

    int usable = mhe->filled ? N : (idx + 1);

    /* Weighted fusion: more recent measurements weighted higher */
    double sum_w = mhe->Q_weight;  /* Prior weight */
    double sum_wy = mhe->Q_weight * model_pred;  /* Prior * model */

    int i;
    for (i = 0; i < usable; i++) {
        /* Weight decays exponentially with age */
        int age = usable - 1 - i;
        double w = mhe->R_weight * exp(-0.3 * (double)age);

        sum_w  += w;
        sum_wy += w * mhe->y_history[i];
    }

    if (sum_w < 1e-12) {
        *estimate = model_pred;
        return 0;
    }

    *estimate = sum_wy / sum_w;
    mhe->x_estimate[0] = *estimate;

    return 0;
}

void vfm_mhe_free(vfm_mhe_t *mhe)
{
    if (!mhe) return;
    free(mhe->x_history);
    free(mhe->y_history);
    mhe->x_history = NULL;
    mhe->y_history = NULL;
}

/* ==========================================================================
 * L5: Sensor Fusion — Weighted Least Squares
 * ========================================================================== */

int vfm_sensor_fusion_wls(const double *estimates, const double *variances,
                           int n, double *fused_estimate,
                           double *fused_variance)
{
    /*
     * Inverse-variance weighting (optimal for independent Gaussian errors).
     *
     * x_fused = sum(x_i / sigma_i^2) / sum(1 / sigma_i^2)
     * sigma_fused^2 = 1 / sum(1 / sigma_i^2)
     *
     * This is the BLUE (Best Linear Unbiased Estimator) and also
     * the MLE (Maximum Likelihood Estimate) for Gaussian errors.
     *
     * Derived from minimizing the weighted sum of squares:
     *   J(x) = sum_i (x - x_i)^2 / sigma_i^2
     */

    if (!estimates || !variances || !fused_estimate || !fused_variance) {
        return -1;
    }
    if (n <= 0) return -1;

    double sum_w = 0.0;
    double sum_wx = 0.0;

    int i;
    for (i = 0; i < n; i++) {
        if (variances[i] <= 0.0) return -2;  /* Invalid variance */
        double w = 1.0 / variances[i];
        sum_w  += w;
        sum_wx += w * estimates[i];
    }

    if (sum_w <= 0.0) return -1;

    *fused_estimate = sum_wx / sum_w;
    *fused_variance = 1.0 / sum_w;

    return 0;
}

double vfm_consistency_chi2(const double *estimates, const double *variances,
                             int n, double fused)
{
    /*
     * Chi-squared consistency test:
     * chi2 = sum_i (x_i - x_fused)^2 / sigma_i^2
     *
     * Under H0 (all sensors measure same true value), chi2 follows
     * a chi-squared distribution with (n-1) degrees of freedom.
     *
     * For n=2: critical value at 95% confidence is 3.84
     * For n=3: critical value at 95% confidence is 5.99
     * For n=4: critical value at 95% confidence is 7.81
     *
     * Large chi2 => at least one sensor is biased/faulty.
     */

    if (!estimates || !variances || n < 2) return 0.0;

    double chi2 = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        if (variances[i] <= 0.0) continue;
        double diff = estimates[i] - fused;
        chi2 += (diff * diff) / variances[i];
    }

    return chi2;
}

/* ==========================================================================
 * L6: CUSUM Drift Detection
 * ========================================================================== */

int vfm_cusum_drift_detect(double x, double mu0, double delta, double H,
                            double *s_high, double *s_low)
{
    /*
     * CUSUM (Cumulative Sum) algorithm for detecting mean shifts.
     *
     * Originally developed by E.S. Page (1954) for statistical
     * quality control. Widely used in industrial process monitoring
     * for detecting sensor drift and process upsets.
     *
     * For positive drift detection (mean increasing):
     *   S_high[k] = max(0, S_high[k-1] + (x[k] - mu0 - delta/2))
     *
     * For negative drift detection (mean decreasing):
     *   S_low[k] = max(0, S_low[k-1] - (x[k] - mu0 - delta/2))
     *
     * Actually, the standard formulation is:
     *   S_high[k] = max(0, S_high[k-1] + x[k] - mu0 - K)
     *   S_low[k]  = max(0, S_low[k-1] + mu0 - K - x[k])
     *   where K = delta/2 (allowable slack)
     *
     * Alarm when S_high > H or S_low > H.
     *
     * H = decision interval (typically 4-5 times sigma)
     * delta = minimum shift magnitude to detect
     */

    if (!s_high || !s_low) return 0;

    double K = delta / 2.0;  /* Reference value / allowable slack */

    /* Positive drift */
    double s_high_new = *s_high + (x - mu0 - K);
    if (s_high_new < 0.0) s_high_new = 0.0;
    *s_high = s_high_new;

    /* Negative drift */
    double s_low_new = *s_low - (x - mu0 + K);
    if (s_low_new < 0.0) s_low_new = 0.0;
    *s_low = s_low_new;

    /* Check for alarm */
    if (*s_high > H || *s_low > H) {
        return 1;  /* Drift detected */
    }

    return 0;
}