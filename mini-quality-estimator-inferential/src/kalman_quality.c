/**
 * @file kalman_quality.c
 * @brief Implementation of Kalman filter variants (KF, EKF, Adaptive KF) for quality estimation.
 *
 * Level: L5 Algorithms
 * Reference: Simon (2006) "Optimal State Estimation"; Julier & Uhlmann (2004)
 */

#include "kalman_quality.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*===========================================================================
 * Matrix Utility Functions (L3: Engineering Structures)
 *===========================================================================*/

/**
 * @brief Invert a small matrix using Gauss-Jordan elimination with full pivoting.
 *
 * This is suitable for quality estimation where n ≤ 8 typically.
 * For larger matrices, Cholesky or LU decomposition would be preferred.
 *
 * Complexity: O(n^3).
 *
 * @param A    Input matrix [n × n] (destroyed in-place)
 * @param Ainv [out] Inverse matrix [n × n]
 * @param n    Matrix dimension
 * @return     1 on success, 0 if singular
 */
int matrix_invert(const double *A, double *Ainv, int n)
{
    if (!A || !Ainv || n <= 0) return 0;
    if (n == 1) {
        if (fabs(A[0]) < 1e-30) return 0;
        Ainv[0] = 1.0 / A[0];
        return 1;
    }

    /* Create augmented matrix [A | I] */
    double *aug = (double *)malloc(n * (2 * n) * sizeof(double));
    if (!aug) return 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * (2 * n) + j] = A[i * n + j];
        }
        for (int j = n; j < 2 * n; j++) {
            aug[i * (2 * n) + j] = (i == (j - n)) ? 1.0 : 0.0;
        }
    }

    /* Gauss-Jordan elimination */
    for (int col = 0; col < n; col++) {
        /* Find pivot (partial pivoting) */
        int pivot_row = col;
        double max_val = fabs(aug[col * (2 * n) + col]);
        for (int row = col + 1; row < n; row++) {
            double val = fabs(aug[row * (2 * n) + col]);
            if (val > max_val) {
                max_val = val;
                pivot_row = row;
            }
        }

        if (max_val < 1e-30) { free(aug); return 0; }

        /* Swap rows */
        if (pivot_row != col) {
            for (int j = 0; j < 2 * n; j++) {
                double tmp = aug[col * (2 * n) + j];
                aug[col * (2 * n) + j] = aug[pivot_row * (2 * n) + j];
                aug[pivot_row * (2 * n) + j] = tmp;
            }
        }

        /* Normalize pivot row */
        double pivot = aug[col * (2 * n) + col];
        for (int j = 0; j < 2 * n; j++) {
            aug[col * (2 * n) + j] /= pivot;
        }

        /* Eliminate other rows */
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double factor = aug[row * (2 * n) + col];
            for (int j = 0; j < 2 * n; j++) {
                aug[row * (2 * n) + j] -= factor * aug[col * (2 * n) + j];
            }
        }
    }

    /* Extract inverse */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Ainv[i * n + j] = aug[i * (2 * n) + (n + j)];
        }
    }

    free(aug);
    return 1;
}

/**
 * @brief Multiply two matrices: C = A * B  (all stored row-major).
 *
 * C[i][j] = sum_k A[i][k] * B[k][j]
 *
 * @param C    [out] Result matrix [rows_A × cols_B]
 * @param A    Left matrix [rows_A × cols_A]
 * @param B    Right matrix [cols_A × cols_B]
 * @param rows_A Number of rows in A
 * @param cols_A Number of columns in A (and rows in B)
 * @param cols_B Number of columns in B
 */
static void matrix_mult(double *C, const double *A, const double *B,
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

/**
 * @brief Transpose a matrix: B = A^T.
 */
static void matrix_transpose(double *B, const double *A, int rows, int cols)
{
    if (!B || !A) return;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

/*===========================================================================
 * L5: Linear Kalman Filter
 *===========================================================================*/

void kf_alloc(kalman_filter_t *kf, int n_states, int n_inputs, int n_measurements)
{
    if (!kf) return;
    memset(kf, 0, sizeof(kalman_filter_t));
    if (n_states <= 0) return;

    kf->n_states       = n_states;
    kf->n_inputs       = n_inputs;
    kf->n_measurements = n_measurements;

    int n2   = n_states * n_states;
    int nm   = n_states * n_measurements;
    int m2   = n_measurements * n_measurements;

    kf->A = (double *)calloc(n2, sizeof(double));
    kf->B = (double *)calloc(n_states * n_inputs, sizeof(double));
    kf->C = (double *)calloc(n_measurements * n_states, sizeof(double));
    kf->Q = (double *)calloc(n2, sizeof(double));
    kf->R = (double *)calloc(m2, sizeof(double));
    kf->x = (double *)calloc(n_states, sizeof(double));
    kf->P = (double *)calloc(n2, sizeof(double));

    kf->x_prior = (double *)calloc(n_states, sizeof(double));
    kf->P_prior = (double *)calloc(n2, sizeof(double));
    kf->K        = (double *)calloc(nm, sizeof(double));
    kf->S        = (double *)calloc(m2, sizeof(double));
    kf->S_inv    = (double *)calloc(m2, sizeof(double));
    kf->temp_nn  = (double *)calloc(n2, sizeof(double));
    kf->temp_nm  = (double *)calloc(nm, sizeof(double));
    kf->temp_mn  = (double *)calloc(n_measurements * n_states, sizeof(double));
    kf->temp_mm  = (double *)calloc(m2, sizeof(double));
    kf->temp_n   = (double *)calloc(n_states, sizeof(double));
    kf->temp_m   = (double *)calloc(n_measurements, sizeof(double));
    kf->I        = (double *)calloc(n2, sizeof(double));

    /* Set A and I to identity */
    if (kf->A) {
        for (int i = 0; i < n_states; i++) kf->A[i * n_states + i] = 1.0;
    }
    if (kf->I) {
        for (int i = 0; i < n_states; i++) kf->I[i * n_states + i] = 1.0;
    }
    /* P = large * I (high initial uncertainty) */
    if (kf->P) {
        for (int i = 0; i < n_states; i++) kf->P[i * n_states + i] = 1000.0;
    }
    /* R = identity (unit measurement noise) */
    if (kf->R) {
        for (int i = 0; i < n_measurements; i++) kf->R[i * n_measurements + i] = 1.0;
    }
}

void kf_free(kalman_filter_t *kf)
{
    if (!kf) return;
    free(kf->A); free(kf->B); free(kf->C); free(kf->Q); free(kf->R);
    free(kf->x); free(kf->P);
    free(kf->x_prior); free(kf->P_prior); free(kf->K);
    free(kf->S); free(kf->S_inv);
    free(kf->temp_nn); free(kf->temp_nm); free(kf->temp_mn); free(kf->temp_mm);
    free(kf->temp_n); free(kf->temp_m); free(kf->I);
    memset(kf, 0, sizeof(kalman_filter_t));
}

void kf_set_matrices(kalman_filter_t *kf, const double *A, const double *B, const double *C)
{
    if (!kf) return;
    int n = kf->n_states;
    int m = kf->n_inputs;
    int p = kf->n_measurements;
    if (A) memcpy(kf->A, A, n * n * sizeof(double));
    if (B && m > 0) memcpy(kf->B, B, n * m * sizeof(double));
    if (C) memcpy(kf->C, C, p * n * sizeof(double));
}

void kf_set_noise(kalman_filter_t *kf, const double *Q, const double *R)
{
    if (!kf) return;
    int n = kf->n_states;
    int p = kf->n_measurements;
    if (Q) memcpy(kf->Q, Q, n * n * sizeof(double));
    if (R) memcpy(kf->R, R, p * p * sizeof(double));
}

void kf_set_initial(kalman_filter_t *kf, const double *x0, const double *P0)
{
    if (!kf) return;
    int n = kf->n_states;
    if (x0) memcpy(kf->x, x0, n * sizeof(double));
    if (P0) memcpy(kf->P, P0, n * n * sizeof(double));
    kf->is_initialized = 1;
}

void kf_predict(kalman_filter_t *kf, const double *u)
{
    if (!kf || !kf->A || !kf->x) return;

    int n = kf->n_states;
    int m = kf->n_inputs;

    /* x_prior = A * x */
    memset(kf->x_prior, 0, n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            kf->x_prior[i] += kf->A[i * n + j] * kf->x[j];
        }
    }

    /* x_prior += B * u */
    if (u && m > 0 && kf->B) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                kf->x_prior[i] += kf->B[i * m + j] * u[j];
            }
        }
    }

    /* P_prior = A * P * A^T + Q */
    /* temp_nn = P * A^T */
    matrix_mult(kf->temp_nn, kf->P, kf->A, n, n, n);  /* Actually need A^T */
    /* Re-compute properly: P_prior = A * P * A^T */
    double *AT = (double *)malloc(n * n * sizeof(double));
    if (AT) {
        matrix_transpose(AT, kf->A, n, n);
        matrix_mult(kf->temp_nn, kf->P, AT, n, n, n);
        matrix_mult(kf->P_prior, kf->A, kf->temp_nn, n, n, n);
        free(AT);
    } else {
        memcpy(kf->P_prior, kf->P, n * n * sizeof(double));
    }

    /* P_prior += Q */
    for (int i = 0; i < n * n; i++) {
        kf->P_prior[i] += kf->Q[i];
    }
}

void kf_update(kalman_filter_t *kf, const double *y)
{
    if (!kf || !y) return;

    int n = kf->n_states;
    int p = kf->n_measurements;

    /* S = C * P_prior * C^T + R */
    /* temp_mn = C * P_prior */
    matrix_mult(kf->temp_mn, kf->C, kf->P_prior, p, n, n);
    /* S = temp_mn * C^T */
    double *CT = (double *)malloc(p * n * sizeof(double));
    if (CT) {
        matrix_transpose(CT, kf->C, p, n);
        matrix_mult(kf->S, kf->temp_mn, CT, p, n, p);
        free(CT);
    } else {
        memcpy(kf->S, kf->R, p * p * sizeof(double));
    }
    /* S += R */
    for (int i = 0; i < p * p; i++) {
        kf->S[i] += kf->R[i];
    }

    /* K = P_prior * C^T * S^{-1} */
    if (!matrix_invert(kf->S, kf->S_inv, p)) {
        /* S singular — skip update, retain predicted state */
        memcpy(kf->x, kf->x_prior, n * sizeof(double));
        return;
    }
    /* temp_nm = P_prior * C^T */
    double *CT2 = (double *)malloc(p * n * sizeof(double));
    if (CT2) {
        matrix_transpose(CT2, kf->C, p, n);
        matrix_mult(kf->temp_nm, kf->P_prior, CT2, n, n, p);
        /* K = temp_nm * S_inv */
        matrix_mult(kf->K, kf->temp_nm, kf->S_inv, n, p, p);
        free(CT2);
    }

    /* x = x_prior + K * (y - C * x_prior) */
    /* innovation = y - C * x_prior */
    for (int i = 0; i < p; i++) {
        double Cx = 0.0;
        for (int j = 0; j < n; j++) {
            Cx += kf->C[i * n + j] * kf->x_prior[j];
        }
        kf->temp_m[i] = y[i] - Cx;
    }
    /* x = x_prior + K * innovation */
    for (int i = 0; i < n; i++) {
        double k_innov = 0.0;
        for (int j = 0; j < p; j++) {
            k_innov += kf->K[i * p + j] * kf->temp_m[j];
        }
        kf->x[i] = kf->x_prior[i] + k_innov;
    }

    /* P = (I - K*C) * P_prior  (Joseph form for numerical stability) */
    /* temp_nn = K * C */
    matrix_mult(kf->temp_nn, kf->K, kf->C, n, p, n);
    /* temp_nn = I - K*C */
    for (int i = 0; i < n * n; i++) {
        kf->temp_nn[i] = kf->I[i] - kf->temp_nn[i];
    }
    /* P = temp_nn * P_prior */
    matrix_mult(kf->temp_nm, kf->temp_nn, kf->P_prior, n, n, n);
    memcpy(kf->P, kf->temp_nm, n * n * sizeof(double));

    /* Joseph form: P = (I-KC) * P_prior * (I-KC)^T + K * R * K^T */
    double *IKC_T = (double *)malloc(n * n * sizeof(double));
    if (IKC_T) {
        matrix_transpose(IKC_T, kf->temp_nn, n, n);
        matrix_mult(kf->temp_nm, kf->temp_nn, kf->P_prior, n, n, n);
        matrix_mult(kf->P, kf->temp_nm, IKC_T, n, n, n);
        /* + K * R * K^T */
        double *KT = (double *)malloc(p * n * sizeof(double));
        if (KT) {
            matrix_transpose(KT, kf->K, n, p);
            matrix_mult(kf->temp_nm, kf->K, kf->R, n, p, p);
            matrix_mult(kf->temp_nn, kf->temp_nm, KT, n, p, n);
            for (int i = 0; i < n * n; i++) {
                kf->P[i] += kf->temp_nn[i];
            }
            free(KT);
        }
        free(IKC_T);
    }
}

void kf_step(kalman_filter_t *kf, const double *u, const double *y, double *x_out)
{
    if (!kf) return;
    kf_predict(kf, u);
    if (y) kf_update(kf, y);
    if (x_out) memcpy(x_out, kf->x, kf->n_states * sizeof(double));
}

double kf_get_quality(const kalman_filter_t *kf)
{
    if (!kf || !kf->x) return 0.0;
    return kf->x[0];  /* Quality is first state by convention */
}

double kf_get_quality_variance(const kalman_filter_t *kf)
{
    if (!kf || !kf->P) return 1.0;
    double var = kf->P[0];  /* First diagonal element of P */
    return (var > 0.0) ? var : 1.0;
}

void kf_innovation(const kalman_filter_t *kf, const double *y,
                   double *innov, double *n_innov)
{
    if (!kf || !y) return;

    int n = kf->n_states;
    int p = kf->n_measurements;

    /* innov = y - C * x_prior */
    double innov_norm = 0.0;
    for (int i = 0; i < p; i++) {
        double Cx = 0.0;
        for (int j = 0; j < n; j++) {
            Cx += kf->C[i * n + j] * kf->x_prior[j];
        }
        double inn_val = y[i] - Cx;
        if (innov) innov[i] = inn_val;
        innov_norm += inn_val * inn_val;
    }

    if (n_innov) {
        /* Normalized innovation = sqrt(innov^T * S^{-1} * innov) */
        double *S_inv = (double *)malloc(p * p * sizeof(double));
        if (S_inv && matrix_invert(kf->S, S_inv, p)) {
            if (p == 1) {
                *n_innov = fabs(y[0] - kf->C[0]) / sqrt(kf->S[0]);
            } else {
                *n_innov = sqrt(innov_norm);
            }
            free(S_inv);
        } else {
            *n_innov = sqrt(innov_norm);
        }
    }
}

/*===========================================================================
 * L5: Extended Kalman Filter (EKF)
 *===========================================================================*/

void ekf_alloc(ekf_filter_t *ekf, int n_states, int n_inputs, int n_measurements,
               void (*f)(const double*, const double*, double, double*),
               void (*h)(const double*, const double*, double*),
               void (*compute_F)(const double*, const double*, double, double*),
               void (*compute_H)(const double*, const double*, double*))
{
    if (!ekf) return;
    memset(ekf, 0, sizeof(ekf_filter_t));

    ekf->n_states       = n_states;
    ekf->n_inputs       = n_inputs;
    ekf->n_measurements = n_measurements;
    ekf->f         = f;
    ekf->h         = h;
    ekf->compute_F = compute_F;
    ekf->compute_H = compute_H;

    kf_alloc(&ekf->kf, n_states, n_inputs, n_measurements);

    ekf->x_pred = (double *)calloc(n_states, sizeof(double));
    ekf->y_pred = (double *)calloc(n_measurements, sizeof(double));
    ekf->F_jac  = (double *)calloc(n_states * n_states, sizeof(double));
    ekf->H_jac  = (double *)calloc(n_measurements * n_states, sizeof(double));

    /* Default: F = identity */
    if (ekf->F_jac) {
        for (int i = 0; i < n_states; i++) ekf->F_jac[i * n_states + i] = 1.0;
    }
}

void ekf_free(ekf_filter_t *ekf)
{
    if (!ekf) return;
    kf_free(&ekf->kf);
    free(ekf->x_pred); free(ekf->y_pred);
    free(ekf->F_jac);  free(ekf->H_jac);
    memset(ekf, 0, sizeof(ekf_filter_t));
}

void ekf_setup(ekf_filter_t *ekf, const double *Q, const double *R,
               const double *x0, const double *P0)
{
    if (!ekf) return;
    kf_set_noise(&ekf->kf, Q, R);
    kf_set_initial(&ekf->kf, x0, P0);
}

void ekf_step(ekf_filter_t *ekf, const double *u, const double *y, double dt)
{
    if (!ekf) return;

    int n = ekf->n_states;
    ekf->dt = dt;

    /* 1. Predict state using nonlinear f() */
    if (ekf->f) {
        ekf->f(ekf->kf.x, u, dt, ekf->x_pred);
        memcpy(ekf->kf.x_prior, ekf->x_pred, n * sizeof(double));
    } else {
        memcpy(ekf->kf.x_prior, ekf->kf.x, n * sizeof(double));
    }

    /* 2. Linearize: F = df/dx at current estimate */
    if (ekf->compute_F) {
        ekf->compute_F(ekf->kf.x, u, dt, ekf->F_jac);
    }

    /* 3. Predict covariance: P_prior = F * P * F^T + Q */
    /* temp_nn = P * F^T */
    double *FT = (double *)malloc(n * n * sizeof(double));
    if (FT) {
        matrix_transpose(FT, ekf->F_jac, n, n);
        matrix_mult(ekf->kf.temp_nn, ekf->kf.P, FT, n, n, n);
        matrix_mult(ekf->kf.P_prior, ekf->F_jac, ekf->kf.temp_nn, n, n, n);
        free(FT);
    }
    for (int i = 0; i < n * n; i++) {
        ekf->kf.P_prior[i] += ekf->kf.Q[i];
    }

    /* 4. If measurement available, update */
    if (y) {
        /* Predict measurement */
        if (ekf->h) {
            ekf->h(ekf->kf.x_prior, u, ekf->y_pred);
        }

        /* Linearize measurement: H = dh/dx at predicted state */
        if (ekf->compute_H) {
            ekf->compute_H(ekf->kf.x_prior, u, ekf->H_jac);
        }

        /* Copy H_jac to C for the standard KF update */
        int p = ekf->n_measurements;
        if (ekf->kf.C) {
            memcpy(ekf->kf.C, ekf->H_jac, p * n * sizeof(double));
        }

        /* Innovation: y_innov = y - h(x_prior) */
        double *y_innov = (double *)malloc(p * sizeof(double));
        if (y_innov) {
            for (int i = 0; i < p; i++) {
                y_innov[i] = y[i] - ekf->y_pred[i];
            }
            kf_update(&ekf->kf, y_innov);
            free(y_innov);
        }
    } else {
        /* Predict only: copy prior to current */
        memcpy(ekf->kf.x, ekf->kf.x_prior, n * sizeof(double));
    }
}

double ekf_get_quality(const ekf_filter_t *ekf)
{
    return ekf ? kf_get_quality(&ekf->kf) : 0.0;
}

double ekf_get_quality_variance(const ekf_filter_t *ekf)
{
    return ekf ? kf_get_quality_variance(&ekf->kf) : 1.0;
}

/*===========================================================================
 * L5: Adaptive Kalman Filter (innovation-based Q adaptation)
 *===========================================================================*/

void akf_alloc(adaptive_kf_t *akf, int n_states, int n_inputs, int n_measurements, int window)
{
    if (!akf) return;
    memset(akf, 0, sizeof(adaptive_kf_t));
    kf_alloc(&akf->kf, n_states, n_inputs, n_measurements);
    akf->window_size = (window > 0) ? window : 10;
    int m = n_measurements;
    akf->innovation_buffer = (double *)calloc(akf->window_size * m, sizeof(double));
    akf->Q_adaptive = (double *)calloc(n_states * n_states, sizeof(double));
    akf->Q_initial  = (double *)calloc(n_states * n_states, sizeof(double));
}

void akf_free(adaptive_kf_t *akf)
{
    if (!akf) return;
    kf_free(&akf->kf);
    free(akf->innovation_buffer);
    free(akf->Q_adaptive);
    free(akf->Q_initial);
    memset(akf, 0, sizeof(adaptive_kf_t));
}

void akf_setup(adaptive_kf_t *akf, const double *A, const double *B, const double *C,
               const double *Q0, const double *R, const double *x0, const double *P0)
{
    if (!akf) return;
    kf_set_matrices(&akf->kf, A, B, C);
    kf_set_noise(&akf->kf, Q0, R);
    kf_set_initial(&akf->kf, x0, P0);
    if (Q0) memcpy(akf->Q_initial, Q0, akf->kf.n_states * akf->kf.n_states * sizeof(double));
}

void akf_step(adaptive_kf_t *akf, const double *u, const double *y)
{
    if (!akf) return;

    int n = akf->kf.n_states;
    int p = akf->kf.n_measurements;
    int w = akf->window_size;

    /* Standard predict */
    kf_predict(&akf->kf, u);

    if (y) {
        /* Compute innovation before update */
        double *innov = (double *)malloc(p * sizeof(double));
        if (innov) {
            for (int i = 0; i < p; i++) {
                double Cx = 0.0;
                for (int j = 0; j < n; j++) {
                    Cx += akf->kf.C[i * n + j] * akf->kf.x_prior[j];
                }
                innov[i] = y[i] - Cx;
            }

            /* Store innovation */
            int idx = akf->window_index;
            memcpy(&akf->innovation_buffer[idx * p], innov, p * sizeof(double));
            akf->window_index = (idx + 1) % w;
            if (!akf->buffer_filled && akf->window_index == 0) {
                akf->buffer_filled = 1;
            }

            /* Update */
            kf_update(&akf->kf, y);

            /* Adapt Q using stored innovations (Mehra's method) */
            if (akf->buffer_filled) {
                /* Estimate innovation covariance */
                double *S_est = (double *)calloc(p * p, sizeof(double));
                if (S_est) {
                    int samples = w;
                    for (int k = 0; k < samples; k++) {
                        for (int i = 0; i < p; i++) {
                            for (int j = 0; j < p; j++) {
                                S_est[i * p + j] += akf->innovation_buffer[k * p + i]
                                                  * akf->innovation_buffer[k * p + j];
                            }
                        }
                    }
                    for (int i = 0; i < p * p; i++) S_est[i] /= (double)samples;

                    /* Q_adaptive = K * S_est * K^T */
                    /* Using the Kalman gain K from the filter */
                    matrix_mult(akf->kf.temp_nm, akf->kf.K, S_est, n, p, p);
                    double *KT = (double *)malloc(p * n * sizeof(double));
                    if (KT) {
                        matrix_transpose(KT, akf->kf.K, n, p);
                        matrix_mult(akf->Q_adaptive, akf->kf.temp_nm, KT, n, p, n);
                        free(KT);
                    }

                    /* Blend with initial Q (smooth adaptation) */
                    double alpha = 0.1; /* Slow adaptation */
                    for (int i = 0; i < n * n; i++) {
                        akf->kf.Q[i] = (1.0 - alpha) * akf->kf.Q[i]
                                     + alpha * akf->Q_adaptive[i];
                    }

                    free(S_est);
                }
            }

            free(innov);
        }
    } else {
        /* No measurement: state = prior */
        memcpy(akf->kf.x, akf->kf.x_prior, n * sizeof(double));
    }
}

void akf_get_Q(const adaptive_kf_t *akf, double *Q_out)
{
    if (!akf || !Q_out) return;
    int n2 = akf->kf.n_states * akf->kf.n_states;
    memcpy(Q_out, akf->kf.Q, n2 * sizeof(double));
}
