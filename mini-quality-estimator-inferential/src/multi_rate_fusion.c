/**
 * @file multi_rate_fusion.c
 * @brief Implementation of multi-rate Kalman fusion, interpolation, and Cubature KF.
 */

#include "multi_rate_fusion.h"
#include "kalman_quality.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Forward declarations from kalman_quality.c */
extern int matrix_invert(const double *A, double *Ainv, int n);
static void matrix_mult_local(double *C, const double *A, const double *B,
                               int rows_A, int cols_A, int cols_B);
static void matrix_transpose_local(double *B, const double *A, int rows, int cols);

static void matrix_mult_local(double *C, const double *A, const double *B,
                               int rows_A, int cols_A, int cols_B)
{
    if (!C || !A || !B) return;
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols_A; k++) {
                sum += A[i * cols_A + k] * B[k * cols_B + j];
            }
            C[i * cols_B + j] = sum;
        }
    }
}

static void matrix_transpose_local(double *B, const double *A, int rows, int cols)
{
    if (!B || !A) return;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

/*===========================================================================
 * L3: Multi-Rate Kalman Fusion
 *===========================================================================*/

void mrf_kalman_alloc(mrf_kalman_t *mrf, int n_states, int n_fast,
                      int n_medium, int n_slow, int buffer_size)
{
    if (!mrf) return;
    memset(mrf, 0, sizeof(mrf_kalman_t));
    if (n_states <= 0) return;

    mrf->n_states      = n_states;
    mrf->n_fast_inputs  = n_fast;
    mrf->n_medium_meas  = n_medium;
    mrf->n_slow_meas    = n_slow;
    mrf->buffer_size    = (buffer_size > 0) ? buffer_size : 1000;

    int n2 = n_states * n_states;
    mrf->A = (double *)calloc(n2, sizeof(double));
    mrf->B = (double *)calloc(n_states * n_fast, sizeof(double));
    mrf->C_medium = (double *)calloc(n_medium * n_states, sizeof(double));
    mrf->C_slow   = (double *)calloc(n_slow * n_states, sizeof(double));
    mrf->Q = (double *)calloc(n2, sizeof(double));
    mrf->R_medium = (double *)calloc(n_medium * n_medium, sizeof(double));
    mrf->R_slow   = (double *)calloc(n_slow * n_slow, sizeof(double));
    mrf->x = (double *)calloc(n_states, sizeof(double));
    mrf->P = (double *)calloc(n2, sizeof(double));

    mrf->x_buffer = (double *)calloc(mrf->buffer_size * n_states, sizeof(double));
    mrf->P_buffer = (double *)calloc(mrf->buffer_size * n2, sizeof(double));
    mrf->t_buffer = (double *)calloc(mrf->buffer_size, sizeof(double));

    mrf->temp_nn  = (double *)calloc(n2, sizeof(double));
    mrf->temp_n   = (double *)calloc(n_states, sizeof(double));
    mrf->temp_mm_m = (double *)calloc(n_medium * n_medium, sizeof(double));
    mrf->temp_ss_s = (double *)calloc(n_slow * n_slow, sizeof(double));

    /* Initialize A to identity */
    if (mrf->A) {
        for (int i = 0; i < n_states; i++) mrf->A[i * n_states + i] = 1.0;
    }
    /* Initialize P to large diagonal */
    if (mrf->P) {
        for (int i = 0; i < n_states; i++) mrf->P[i * n_states + i] = 100.0;
    }
    /* Initialize R matrices to identity */
    if (mrf->R_medium) {
        for (int i = 0; i < n_medium; i++) mrf->R_medium[i * n_medium + i] = 1.0;
    }
    if (mrf->R_slow) {
        for (int i = 0; i < n_slow; i++) mrf->R_slow[i * n_slow + i] = 1.0;
    }
}

void mrf_kalman_free(mrf_kalman_t *mrf)
{
    if (!mrf) return;
    free(mrf->A); free(mrf->B); free(mrf->C_medium); free(mrf->C_slow);
    free(mrf->Q); free(mrf->R_medium); free(mrf->R_slow);
    free(mrf->x); free(mrf->P);
    free(mrf->x_buffer); free(mrf->P_buffer); free(mrf->t_buffer);
    free(mrf->temp_nn); free(mrf->temp_n); free(mrf->temp_mm_m); free(mrf->temp_ss_s);
    memset(mrf, 0, sizeof(mrf_kalman_t));
}

void mrf_set_matrices(mrf_kalman_t *mrf, const double *A, const double *B,
                      const double *C_medium, const double *C_slow)
{
    if (!mrf) return;
    int n  = mrf->n_states;
    int m  = mrf->n_fast_inputs;
    int pm = mrf->n_medium_meas;
    int ps = mrf->n_slow_meas;
    if (A) memcpy(mrf->A, A, n * n * sizeof(double));
    if (B && m > 0) memcpy(mrf->B, B, n * m * sizeof(double));
    if (C_medium && pm > 0) memcpy(mrf->C_medium, C_medium, pm * n * sizeof(double));
    if (C_slow && ps > 0) memcpy(mrf->C_slow, C_slow, ps * n * sizeof(double));
}

void mrf_set_noise(mrf_kalman_t *mrf, const double *Q,
                   const double *R_medium, const double *R_slow)
{
    if (!mrf) return;
    int n  = mrf->n_states;
    int pm = mrf->n_medium_meas;
    int ps = mrf->n_slow_meas;
    if (Q) memcpy(mrf->Q, Q, n * n * sizeof(double));
    if (R_medium && pm > 0) memcpy(mrf->R_medium, R_medium, pm * pm * sizeof(double));
    if (R_slow && ps > 0) memcpy(mrf->R_slow, R_slow, ps * ps * sizeof(double));
}

void mrf_fast_step(mrf_kalman_t *mrf, const double *u, double t)
{
    if (!mrf) return;

    int n = mrf->n_states;
    int m = mrf->n_fast_inputs;

    /* Predict: x = A*x + B*u */
    double *x_pred = (double *)calloc(n, sizeof(double));
    if (!x_pred) return;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            x_pred[i] += mrf->A[i * n + j] * mrf->x[j];
        }
    }
    if (u && m > 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                x_pred[i] += mrf->B[i * m + j] * u[j];
            }
        }
    }

    /* P = A*P*A^T + Q */
    double *AT = (double *)malloc(n * n * sizeof(double));
    double *temp1 = (double *)malloc(n * n * sizeof(double));
    if (AT && temp1) {
        matrix_transpose_local(AT, mrf->A, n, n);
        matrix_mult_local(temp1, mrf->P, AT, n, n, n);
        matrix_mult_local(mrf->temp_nn, mrf->A, temp1, n, n, n);
        for (int i = 0; i < n * n; i++) {
            mrf->P[i] = mrf->temp_nn[i] + mrf->Q[i];
        }
    }
    free(AT); free(temp1);

    /* Store new state */
    memcpy(mrf->x, x_pred, n * sizeof(double));

    /* Store in buffer for delayed measurement updates */
    if (mrf->buffer_size > 0 && mrf->x_buffer) {
        int idx = mrf->buffer_head;
        memcpy(&mrf->x_buffer[idx * n], mrf->x, n * sizeof(double));
        memcpy(&mrf->P_buffer[idx * n * n], mrf->P, n * n * sizeof(double));
        mrf->t_buffer[idx] = t;
        mrf->buffer_head = (idx + 1) % mrf->buffer_size;
        if (mrf->buffer_count < mrf->buffer_size) mrf->buffer_count++;
    }

    mrf->t_current = t;
    free(x_pred);
}

void mrf_medium_update(mrf_kalman_t *mrf, const double *y_medium, double t_meas)
{
    if (!mrf || !y_medium) return;

    int n  = mrf->n_states;
    int pm = mrf->n_medium_meas;
    if (pm <= 0) return;

    /* Find the closest buffered state to t_meas */
    int best_idx = -1;
    double best_dt = 1e30;
    int count = mrf->buffer_count;
    for (int k = 0; k < count; k++) {
        int idx = (mrf->buffer_head - 1 - k + mrf->buffer_size) % mrf->buffer_size;
        double dt = fabs(mrf->t_buffer[idx] - t_meas);
        if (dt < best_dt) {
            best_dt = dt;
            best_idx = idx;
        }
    }

    if (best_idx < 0) return;  /* No buffered state */

    /* Use the buffered state for the measurement update */
    double *x_buf = &mrf->x_buffer[best_idx * n];
    double *P_buf = &mrf->P_buffer[best_idx * n * n];

    /* Innovation: y - C_medium * x_buf */
    double *innov = (double *)malloc(pm * sizeof(double));
    if (!innov) return;
    for (int i = 0; i < pm; i++) {
        double Cx = 0.0;
        for (int j = 0; j < n; j++) {
            Cx += mrf->C_medium[i * n + j] * x_buf[j];
        }
        innov[i] = y_medium[i] - Cx;
    }

    /* Kalman gain: K = P_buf * C^T * (C * P_buf * C^T + R_medium)^{-1} */
    double *PCT = (double *)malloc(n * pm * sizeof(double));
    double *S_med = (double *)malloc(pm * pm * sizeof(double));
    double *S_inv = (double *)malloc(pm * pm * sizeof(double));
    double *K   = (double *)malloc(n * pm * sizeof(double));

    if (PCT && S_med && S_inv && K) {
        double *CT = (double *)malloc(pm * n * sizeof(double));
        if (CT) {
            matrix_transpose_local(CT, mrf->C_medium, pm, n);
            matrix_mult_local(PCT, P_buf, CT, n, n, pm);
            matrix_mult_local(S_med, mrf->C_medium, PCT, pm, n, pm);
            for (int i = 0; i < pm * pm; i++) {
                S_med[i] += mrf->R_medium[i];
            }
            if (matrix_invert(S_med, S_inv, pm)) {
                matrix_mult_local(K, PCT, S_inv, n, pm, pm);

                /* Update state */
                for (int i = 0; i < n; i++) {
                    double k_innov = 0.0;
                    for (int j = 0; j < pm; j++) {
                        k_innov += K[i * pm + j] * innov[j];
                    }
                    mrf->x[i] = x_buf[i] + k_innov;
                }

                /* Update covariance: P = (I - K*C)*P_buf */
                double *KC = (double *)malloc(n * n * sizeof(double));
                if (KC) {
                    matrix_mult_local(KC, K, mrf->C_medium, n, pm, n);
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n; j++) {
                            double delta = (i == j) ? 1.0 : 0.0;
                            KC[i * n + j] = delta - KC[i * n + j];
                        }
                    }
                    matrix_mult_local(mrf->P, KC, P_buf, n, n, n);
                    free(KC);
                }
            }
            free(CT);
        }
    }

    free(PCT); free(S_med); free(S_inv); free(K); free(innov);
    mrf->t_last_medium = t_meas;
}

void mrf_slow_update(mrf_kalman_t *mrf, const double *y_slow, double t_meas)
{
    if (!mrf || !y_slow) return;

    int n  = mrf->n_states;
    int ps = mrf->n_slow_meas;
    if (ps <= 0) return;

    /* Find closest buffered state */
    int best_idx = -1;
    double best_dt = 1e30;
    int count = mrf->buffer_count;
    for (int k = 0; k < count; k++) {
        int idx = (mrf->buffer_head - 1 - k + mrf->buffer_size) % mrf->buffer_size;
        double dt = fabs(mrf->t_buffer[idx] - t_meas);
        if (dt < best_dt) {
            best_dt = dt;
            best_idx = idx;
        }
    }

    if (best_idx < 0) return;

    double *x_buf = &mrf->x_buffer[best_idx * n];
    double *P_buf = &mrf->P_buffer[best_idx * n * n];

    /* Innovation */
    double *innov = (double *)malloc(ps * sizeof(double));
    if (!innov) return;
    for (int i = 0; i < ps; i++) {
        double Cx = 0.0;
        for (int j = 0; j < n; j++) {
            Cx += mrf->C_slow[i * n + j] * x_buf[j];
        }
        innov[i] = y_slow[i] - Cx;
    }

    /* Kalman gain computation (same pattern as medium update) */
    double *PCT = (double *)malloc(n * ps * sizeof(double));
    double *S_slow = (double *)malloc(ps * ps * sizeof(double));
    double *S_inv = (double *)malloc(ps * ps * sizeof(double));
    double *K = (double *)malloc(n * ps * sizeof(double));

    if (PCT && S_slow && S_inv && K) {
        double *CT = (double *)malloc(ps * n * sizeof(double));
        if (CT) {
            matrix_transpose_local(CT, mrf->C_slow, ps, n);
            matrix_mult_local(PCT, P_buf, CT, n, n, ps);
            matrix_mult_local(S_slow, mrf->C_slow, PCT, ps, n, ps);
            for (int i = 0; i < ps * ps; i++) S_slow[i] += mrf->R_slow[i];
            if (matrix_invert(S_slow, S_inv, ps)) {
                matrix_mult_local(K, PCT, S_inv, n, ps, ps);
                for (int i = 0; i < n; i++) {
                    double k_innov = 0.0;
                    for (int j = 0; j < ps; j++) {
                        k_innov += K[i * ps + j] * innov[j];
                    }
                    mrf->x[i] = x_buf[i] + k_innov;
                }
            }
            free(CT);
        }
    }

    free(PCT); free(S_slow); free(S_inv); free(K); free(innov);
    mrf->t_last_slow = t_meas;
}

void mrf_get_state(const mrf_kalman_t *mrf, double *x)
{
    if (!mrf || !x || !mrf->x) return;
    memcpy(x, mrf->x, mrf->n_states * sizeof(double));
}

double mrf_get_quality(const mrf_kalman_t *mrf)
{
    if (!mrf || !mrf->x) return 0.0;
    return mrf->x[0];
}

double mrf_get_quality_variance(const mrf_kalman_t *mrf)
{
    if (!mrf || !mrf->P) return 1.0;
    double var = mrf->P[0];
    return (var > 0.0) ? var : 1.0;
}

/*===========================================================================
 * L3: Multi-Rate Interpolation
 *===========================================================================*/

void mrf_interp_init(mrf_interpolator_t *interp, double max_gap)
{
    if (!interp) return;
    memset(interp, 0, sizeof(mrf_interpolator_t));
    interp->max_gap = max_gap;
}

void mrf_interp_feed(mrf_interpolator_t *interp, double y_lab, double t_lab)
{
    if (!interp) return;
    if (!interp->has_prev) {
        interp->y_prev = y_lab;
        interp->t_prev = t_lab;
        interp->has_prev = 1;
        return;
    }
    /* Shift: prev <- next <- new */
    if (interp->has_next) {
        interp->y_prev = interp->y_next;
        interp->t_prev = interp->t_next;
    }
    interp->y_next = y_lab;
    interp->t_next = t_lab;
    interp->has_next = 1;
}

int mrf_interp_evaluate(const mrf_interpolator_t *interp, double t, double *y_est)
{
    if (!interp || !y_est) return 0;

    if (!interp->has_prev) {
        *y_est = 0.0;
        return 0;
    }

    if (!interp->has_next || t > interp->t_next) {
        /* Extrapolation: hold last value */
        *y_est = interp->has_next ? interp->y_next : interp->y_prev;
        return 0;
    }

    if (t <= interp->t_prev) {
        *y_est = interp->y_prev;
        return 1;
    }

    double dt_total = interp->t_next - interp->t_prev;
    if (dt_total <= 0.0 || dt_total > interp->max_gap) {
        *y_est = interp->y_prev;
        return 0;
    }

    double frac = (t - interp->t_prev) / dt_total;
    *y_est = interp->y_prev + frac * (interp->y_next - interp->y_prev);
    return 1;
}

/*===========================================================================
 * L5: Cubature Kalman Filter (CKF)
 *===========================================================================*/

void ckf_alloc(ckf_filter_t *ckf, int n_states, int n_inputs, int n_measurements,
               void (*f)(const double*, const double*, double, double*),
               void (*h)(const double*, const double*, double*))
{
    if (!ckf) return;
    memset(ckf, 0, sizeof(ckf_filter_t));
    if (n_states <= 0) return;

    ckf->n_states       = n_states;
    ckf->n_inputs       = n_inputs;
    ckf->n_measurements = n_measurements;
    ckf->f = f;
    ckf->h = h;
    ckf->weight = 1.0 / (2.0 * (double)n_states);

    int n   = n_states;
    int n2  = n * n;
    int p   = n_measurements;
    int pm  = (p > 0) ? p : 1;
    int np  = n * pm;
    int p2  = pm * pm;

    ckf->cubature_points   = (double *)calloc(2 * n * n, sizeof(double));
    ckf->propagated_points = (double *)calloc(2 * n * n, sizeof(double));
    ckf->meas_pred_points  = (double *)calloc(2 * n * pm, sizeof(double));
    ckf->x = (double *)calloc(n, sizeof(double));
    ckf->P = (double *)calloc(n2, sizeof(double));
    ckf->Q = (double *)calloc(n2, sizeof(double));
    ckf->R = (double *)calloc(p2, sizeof(double));
    ckf->x_prior = (double *)calloc(n, sizeof(double));
    ckf->P_prior = (double *)calloc(n2, sizeof(double));
    ckf->y_prior = (double *)calloc(pm, sizeof(double));
    ckf->P_xy    = (double *)calloc(np, sizeof(double));
    ckf->P_yy    = (double *)calloc(p2, sizeof(double));
    ckf->K       = (double *)calloc(np, sizeof(double));
    ckf->temp_n  = (double *)calloc(n, sizeof(double));
    ckf->temp_nn = (double *)calloc(n2, sizeof(double));
    ckf->temp_mm = (double *)calloc(p2, sizeof(double));
    ckf->temp_nm = (double *)calloc(np, sizeof(double));

    /* P = I (moderate initial uncertainty) */
    if (ckf->P) {
        for (int i = 0; i < n; i++) ckf->P[i * n + i] = 1.0;
    }
    /* R = I */
    if (ckf->R && p > 0) {
        for (int i = 0; i < p; i++) ckf->R[i * p + i] = 1.0;
    }
}

void ckf_free(ckf_filter_t *ckf)
{
    if (!ckf) return;
    free(ckf->cubature_points); free(ckf->propagated_points);
    free(ckf->meas_pred_points);
    free(ckf->x); free(ckf->P); free(ckf->Q); free(ckf->R);
    free(ckf->x_prior); free(ckf->P_prior); free(ckf->y_prior);
    free(ckf->P_xy); free(ckf->P_yy); free(ckf->K);
    free(ckf->temp_n); free(ckf->temp_nn); free(ckf->temp_mm); free(ckf->temp_nm);
    memset(ckf, 0, sizeof(ckf_filter_t));
}

void ckf_setup(ckf_filter_t *ckf, const double *Q, const double *R,
               const double *x0, const double *P0)
{
    if (!ckf) return;
    int n  = ckf->n_states;
    int p  = ckf->n_measurements;
    if (Q) memcpy(ckf->Q, Q, n * n * sizeof(double));
    if (R && p > 0) memcpy(ckf->R, R, p * p * sizeof(double));
    if (x0) memcpy(ckf->x, x0, n * sizeof(double));
    if (P0) memcpy(ckf->P, P0, n * n * sizeof(double));
}

void ckf_step(ckf_filter_t *ckf, const double *u, const double *y, double dt)
{
    if (!ckf) return;

    int n = ckf->n_states;
    int p = ckf->n_measurements;
    double w = ckf->weight;

    /* Generate cubature points around current estimate */
    /* Compute sqrt(P) using Cholesky-like decomposition */
    /* For simplicity: use diagonal sqrt (good for well-conditioned P) */
    for (int i = 0; i < 2 * n; i++) {
        int sign = (i < n) ? 1 : -1;
        int dim  = (i < n) ? i : i - n;
        double sigma = sqrt(fabs(ckf->P[dim * n + dim]));
        if (sigma < 1e-10) sigma = 1e-5;
        for (int j = 0; j < n; j++) {
            ckf->cubature_points[i * n + j] = ckf->x[j]
                + sign * sigma * (j == dim ? 1.0 : 0.0);
        }
    }

    /* Propagate cubature points through state function */
    for (int i = 0; i < 2 * n; i++) {
        if (ckf->f) {
            ckf->f(&ckf->cubature_points[i * n], u, dt,
                   &ckf->propagated_points[i * n]);
        } else {
            memcpy(&ckf->propagated_points[i * n],
                   &ckf->cubature_points[i * n], n * sizeof(double));
        }
    }

    /* Predicted state: average of propagated points */
    memset(ckf->x_prior, 0, n * sizeof(double));
    for (int i = 0; i < 2 * n; i++) {
        for (int j = 0; j < n; j++) {
            ckf->x_prior[j] += w * ckf->propagated_points[i * n + j];
        }
    }

    /* Predicted covariance */
    memset(ckf->P_prior, 0, n * n * sizeof(double));
    for (int i = 0; i < 2 * n; i++) {
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) {
                double da = ckf->propagated_points[i * n + a] - ckf->x_prior[a];
                double db = ckf->propagated_points[i * n + b] - ckf->x_prior[b];
                ckf->P_prior[a * n + b] += w * da * db;
            }
        }
    }
    for (int i = 0; i < n * n; i++) {
        ckf->P_prior[i] += ckf->Q[i];
    }

    /* Measurement update (if measurement available) */
    if (y && p > 0) {
        /* Propagate cubature points through measurement function */
        for (int i = 0; i < 2 * n; i++) {
            if (ckf->h) {
                ckf->h(&ckf->cubature_points[i * n], u,
                       &ckf->meas_pred_points[i * p]);
            } else {
                /* Default: first component = quality */
                for (int j = 0; j < p; j++) {
                    ckf->meas_pred_points[i * p + j] = ckf->cubature_points[i * n + j];
                }
            }
        }

        /* Predicted measurement */
        memset(ckf->y_prior, 0, p * sizeof(double));
        for (int i = 0; i < 2 * n; i++) {
            for (int j = 0; j < p; j++) {
                ckf->y_prior[j] += w * ckf->meas_pred_points[i * p + j];
            }
        }

        /* Cross-covariance P_xy and innovation covariance P_yy */
        memset(ckf->P_xy, 0, n * p * sizeof(double));
        memset(ckf->P_yy, 0, p * p * sizeof(double));
        for (int i = 0; i < 2 * n; i++) {
            for (int a = 0; a < n; a++) {
                double dx = ckf->propagated_points[i * n + a] - ckf->x_prior[a];
                for (int b = 0; b < p; b++) {
                    double dy = ckf->meas_pred_points[i * p + b] - ckf->y_prior[b];
                    ckf->P_xy[a * p + b] += w * dx * dy;
                    ckf->P_yy[b * p + b] += w * dy * dy;
                }
            }
        }
        for (int i = 0; i < p * p; i++) {
            ckf->P_yy[i] += ckf->R[i];
        }

        /* Kalman gain: K = P_xy * P_yy^{-1} */
        double *P_yy_inv = (double *)malloc(p * p * sizeof(double));
        if (P_yy_inv && matrix_invert(ckf->P_yy, P_yy_inv, p)) {
            matrix_mult_local(ckf->K, ckf->P_xy, P_yy_inv, n, p, p);

            /* State update: x = x_prior + K * (y - y_prior) */
            double *innov = (double *)malloc(p * sizeof(double));
            if (innov) {
                for (int i = 0; i < p; i++) {
                    innov[i] = y[i] - ckf->y_prior[i];
                }
                for (int i = 0; i < n; i++) {
                    double k_innov = 0.0;
                    for (int j = 0; j < p; j++) {
                        k_innov += ckf->K[i * p + j] * innov[j];
                    }
                    ckf->x[i] = ckf->x_prior[i] + k_innov;
                }
                /* P = P_prior - K * P_yy * K^T */
                double *KPKT = (double *)malloc(n * n * sizeof(double));
                if (KPKT) {
                    double *KT = (double *)malloc(p * n * sizeof(double));
                    if (KT) {
                        matrix_transpose_local(KT, ckf->K, n, p);
                        matrix_mult_local(ckf->temp_nm, ckf->K, ckf->P_yy, n, p, p);
                        matrix_mult_local(KPKT, ckf->temp_nm, KT, n, p, n);
                        for (int i = 0; i < n * n; i++) {
                            ckf->P[i] = ckf->P_prior[i] - KPKT[i];
                            if (ckf->P[i] < 1e-10) ckf->P[i] = 1e-10;
                        }
                        free(KT);
                    }
                    free(KPKT);
                }
                free(innov);
            }
        }
        free(P_yy_inv);
    } else {
        /* Predict only */
        memcpy(ckf->x, ckf->x_prior, n * sizeof(double));
        memcpy(ckf->P, ckf->P_prior, n * n * sizeof(double));
    }
}

double ckf_get_quality(const ckf_filter_t *ckf)
{
    return (ckf && ckf->x) ? ckf->x[0] : 0.0;
}

double ckf_get_quality_variance(const ckf_filter_t *ckf)
{
    if (!ckf || !ckf->P) return 1.0;
    return (ckf->P[0] > 0.0) ? ckf->P[0] : 1.0;
}
