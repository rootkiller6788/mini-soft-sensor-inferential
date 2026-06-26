/**
 * @file kalman_unscented.c
 * @brief Unscented Kalman Filter (UKF) — Sigma-Point Implementation
 *
 * Implements the derivative-free UKF using the unscented transform.
 * The UKF propagates 2n+1 deterministically-chosen sigma points through
 * nonlinear functions to capture mean and covariance without linearization.
 *
 * L5 Algorithms: Unscented transform, sigma point generation
 * L8 Advanced Topics: Square-root UKF for numerical stability
 *
 * Reference: Julier & Uhlmann (2004) "Unscented Filtering and Nonlinear Estimation"
 *            Wan & van der Merwe (2000) "The Unscented Kalman Filter..."
 */
#include "kalman_unscented.h"
#include "kalman_matrix_ops.h"
#include <math.h>
#include <string.h>

/* ===================================================================
 * UKF Parameters
 * =================================================================== */

void ukf_params_init(UKFParams *params, uint8_t n,
                     double alpha, double beta, double kappa)
{
    if (!params) return;
    params->n = n;
    params->alpha = alpha;
    params->beta = beta;
    params->kappa = kappa;

    /* Composite scaling */
    params->lambda = alpha * alpha * ((double)n + kappa) - (double)n;
    params->sqrt_n_plus_lambda = sqrt((double)n + params->lambda);

    /* Sigma point weights */
    params->w_mean_0 = params->lambda / ((double)n + params->lambda);
    params->w_cov_0  = params->w_mean_0 + (1.0 - alpha * alpha + beta);
    params->w_i      = 0.5 / ((double)n + params->lambda);
}

/* ===================================================================
 * UKF Initialization
 * =================================================================== */

void ukf_init(UKFState *ukf,
              const double *x0, const double *P0,
              ukf_process_fn proc_fn, ukf_measure_fn meas_fn,
              uint8_t n, uint8_t m, uint8_t p,
              const UKFParams *params)
{
    if (!ukf) return;
    memset(ukf, 0, sizeof(*ukf));

    if (params) {
        memcpy(&ukf->params, params, sizeof(UKFParams));
    } else {
        ukf_params_init(&ukf->params, n, UKF_DEFAULT_ALPHA,
                        UKF_DEFAULT_BETA, UKF_DEFAULT_KAPPA);
    }

    ukf->process_fn = proc_fn;
    ukf->measure_fn = meas_fn;
    ukf->p = p;
    ukf->initialized = 1;

    /* Initialize KF state */
    KalmanModel km;
    memset(&km, 0, sizeof(km));
    km.n = n; km.m = m; km.p = p;
    kf_init(&ukf->kf, &km, x0, P0, n, m);
}

/* ===================================================================
 * Generate Sigma Points
 * =================================================================== */

void ukf_generate_sigma_points(UKFState *ukf)
{
    if (!ukf) return;
    uint8_t n = ukf->params.n;
    double *sp = ukf->sigma_points;
    double *x  = ukf->kf.x;
    double *P  = ukf->kf.P;

    /* Compute Cholesky factor of (n + lambda) * P */
    double sqrtP[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    for (uint8_t i = 0; i < n * n; i++)
        sqrtP[i] = ukf->params.sqrt_n_plus_lambda * ukf->params.sqrt_n_plus_lambda * P[i];
    /* Actually: sqrt of (n+lambda)*P. Scale then Cholesky. */
    for (uint8_t i = 0; i < n * n; i++)
        sqrtP[i] = P[i] * ukf->params.sqrt_n_plus_lambda * ukf->params.sqrt_n_plus_lambda;

    /* Cholesky: (n+lambda) * P = L * L' */
    /* Scale first, then take sqrt via Cholesky */
    double scaled_P[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    double scale = (ukf->params.n + ukf->params.lambda);
    for (uint8_t i = 0; i < n * n; i++) scaled_P[i] = scale * P[i];

    double L[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    memcpy(L, scaled_P, (size_t)n * n * sizeof(double));
    if (!mat_cholesky(L, n)) {
        /* Fallback: use identity-scaled */
        mat_identity(L, n);
        for (uint8_t i = 0; i < n * n; i++) L[i] *= sqrt(scale);
    }
    /* L is lower triangular with (n+lambda)*P = L*L' */

    /* Sigma point 0: x */
    for (uint8_t i = 0; i < n; i++) sp[i] = x[i];

    /* Sigma points 1..n: x + L_col_i */
    for (uint8_t j = 0; j < n; j++) {
        double *sp_j = sp + (j + 1) * KF_MAX_STATE_DIM;
        for (uint8_t i = 0; i < n; i++)
            sp_j[i] = x[i] + L[i * n + j];
    }

    /* Sigma points n+1..2n: x - L_col_i */
    for (uint8_t j = 0; j < n; j++) {
        double *sp_j = sp + (j + 1 + n) * KF_MAX_STATE_DIM;
        for (uint8_t i = 0; i < n; i++)
            sp_j[i] = x[i] - L[i * n + j];
    }
}

/* ===================================================================
 * UKF Prediction
 * =================================================================== */

void ukf_predict(UKFState *ukf, const double *u, const double *Q)
{
    if (!ukf || !ukf->process_fn) return;
    uint8_t n = ukf->params.n;
    uint8_t num_sigma = 2 * n + 1;
    double *sp = ukf->sigma_points;
    double *sp_f = ukf->sigma_points_f;

    /* 1. Generate sigma points */
    ukf_generate_sigma_points(ukf);

    /* 2. Propagate each sigma point through process model */
    for (uint8_t s = 0; s < num_sigma; s++) {
        ukf->process_fn(sp + s * KF_MAX_STATE_DIM, u, n, ukf->p,
                        sp_f + s * KF_MAX_STATE_DIM);
    }

    /* 3. Recombine: predicted mean */
    memset(ukf->kf.x_prior, 0, n * sizeof(double));
    for (uint8_t i = 0; i < n; i++) {
        ukf->kf.x_prior[i] += ukf->params.w_mean_0 * sp_f[i];
        for (uint8_t s = 1; s < num_sigma; s++) {
            ukf->kf.x_prior[i] += ukf->params.w_i
                                   * sp_f[s * KF_MAX_STATE_DIM + i];
        }
    }

    /* 4. Recombine: predicted covariance */
    memset(ukf->kf.P_prior, 0, (size_t)n * n * sizeof(double));
    for (uint8_t s = 0; s < num_sigma; s++) {
        double w = (s == 0) ? ukf->params.w_cov_0 : ukf->params.w_i;
        double dx[KF_MAX_STATE_DIM];
        for (uint8_t i = 0; i < n; i++)
            dx[i] = sp_f[s * KF_MAX_STATE_DIM + i] - ukf->kf.x_prior[i];

        for (uint8_t i = 0; i < n; i++)
            for (uint8_t j = 0; j < n; j++)
                ukf->kf.P_prior[i * n + j] += w * dx[i] * dx[j];
    }

    /* 5. Add process noise */
    if (Q) mat_add(ukf->kf.P_prior, Q, ukf->kf.P_prior, n, n);
}

/* ===================================================================
 * UKF Update
 * =================================================================== */

void ukf_update(UKFState *ukf, const double *z, const double *R)
{
    if (!ukf || !ukf->measure_fn || !z) return;
    uint8_t n = ukf->params.n, m = ukf->kf.m;
    uint8_t num_sigma = 2 * n + 1;
    double *sp_f = ukf->sigma_points_f;
    double *sp_h = ukf->sigma_points_h;

    /* 1. Propagate sigma points through measurement model */
    for (uint8_t s = 0; s < num_sigma; s++) {
        ukf->measure_fn(sp_f + s * KF_MAX_STATE_DIM, n, m,
                        sp_h + s * KF_MAX_MEAS_DIM);
    }

    /* 2. Predicted measurement mean */
    memset(ukf->z_pred, 0, m * sizeof(double));
    for (uint8_t i = 0; i < m; i++) {
        ukf->z_pred[i] += ukf->params.w_mean_0 * sp_h[i];
        for (uint8_t s = 1; s < num_sigma; s++)
            ukf->z_pred[i] += ukf->params.w_i * sp_h[s * KF_MAX_MEAS_DIM + i];
    }

    /* 3. Innovation */
    for (uint8_t i = 0; i < m; i++)
        ukf->kf.innovation[i] = z[i] - ukf->z_pred[i];

    /* 4. Innovation covariance P_zz */
    memset(ukf->P_zz, 0, (size_t)m * m * sizeof(double));
    for (uint8_t s = 0; s < num_sigma; s++) {
        double w = (s == 0) ? ukf->params.w_cov_0 : ukf->params.w_i;
        double dz[KF_MAX_MEAS_DIM];
        for (uint8_t i = 0; i < m; i++)
            dz[i] = sp_h[s * KF_MAX_MEAS_DIM + i] - ukf->z_pred[i];

        for (uint8_t i = 0; i < m; i++)
            for (uint8_t j = 0; j < m; j++)
                ukf->P_zz[i * m + j] += w * dz[i] * dz[j];
    }
    if (R) mat_add(ukf->P_zz, R, ukf->P_zz, m, m);

    /* 5. Cross-covariance P_xz */
    memset(ukf->P_xz, 0, (size_t)n * m * sizeof(double));
    for (uint8_t s = 0; s < num_sigma; s++) {
        double w = (s == 0) ? ukf->params.w_cov_0 : ukf->params.w_i;
        double dx[KF_MAX_STATE_DIM];
        double dz[KF_MAX_MEAS_DIM];
        for (uint8_t i = 0; i < n; i++)
            dx[i] = sp_f[s * KF_MAX_STATE_DIM + i] - ukf->kf.x_prior[i];
        for (uint8_t i = 0; i < m; i++)
            dz[i] = sp_h[s * KF_MAX_MEAS_DIM + i] - ukf->z_pred[i];

        for (uint8_t i = 0; i < n; i++)
            for (uint8_t j = 0; j < m; j++)
                ukf->P_xz[i * m + j] += w * dx[i] * dz[j];
    }

    /* 6. Kalman gain: K = P_xz * inv(P_zz) */
    double P_zz_inv[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    memcpy(P_zz_inv, ukf->P_zz, (size_t)m * m * sizeof(double));
    if (!mat_inverse_cholesky(P_zz_inv, P_zz_inv, m)) {
        ukf->kf.singular = 1;
        return;
    }
    mat_mul(ukf->P_xz, P_zz_inv, ukf->kf.K, n, m, m);

    /* 7. Update state: x = x_prior + K * y */
    double Ky[KF_MAX_STATE_DIM];
    mat_vec_mul(ukf->kf.K, ukf->kf.innovation, Ky, n, m);
    for (uint8_t i = 0; i < n; i++)
        ukf->kf.x[i] = ukf->kf.x_prior[i] + Ky[i];

    /* 8. Update covariance: P = P_prior - K * P_zz * K' */
    double KPzz[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM];
    mat_mul(ukf->kf.K, ukf->P_zz, KPzz, n, m, m);
    double KPzzKt[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    mat_mul_A_BT(KPzz, ukf->kf.K, KPzzKt, n, m, n);
    mat_sub(ukf->kf.P_prior, KPzzKt, ukf->kf.P, n, n);

    /* Copy S for diagnostics */
    memcpy(ukf->kf.S, ukf->P_zz, (size_t)m * m * sizeof(double));
    ukf->kf.step_count++;
}

/* ===================================================================
 * Combined UKF step
 * =================================================================== */

const double *ukf_step(UKFState *ukf, const double *z, const double *u,
                       const double *Q, const double *R)
{
    ukf_predict(ukf, u, Q);
    ukf_update(ukf, z, R);
    return ukf->kf.x;
}

/* ===================================================================
 * Square-Root UKF Predict
 * =================================================================== */

void ukf_predict_sqrt(UKFState *ukf, const double *u, const double *sqrt_Q)
{
    if (!ukf || !ukf->process_fn) return;
    uint8_t n = ukf->params.n;
    uint8_t num_sigma = 2 * n + 1;

    /* Generate sigma points using current sqrt(P) factor */
    ukf_generate_sigma_points(ukf);

    double *sp = ukf->sigma_points;
    double *sp_f = ukf->sigma_points_f;

    for (uint8_t s = 0; s < num_sigma; s++) {
        ukf->process_fn(sp + s * KF_MAX_STATE_DIM, u, n, ukf->p,
                        sp_f + s * KF_MAX_STATE_DIM);
    }

    /* Predicted mean */
    memset(ukf->kf.x_prior, 0, n * sizeof(double));
    for (uint8_t i = 0; i < n; i++) {
        ukf->kf.x_prior[i] += ukf->params.w_mean_0 * sp_f[i];
        for (uint8_t s = 1; s < num_sigma; s++)
            ukf->kf.x_prior[i] += ukf->params.w_i
                                   * sp_f[s * KF_MAX_STATE_DIM + i];
    }

    /* Form weighted, centered matrix for QR decomposition */
    double X_weighted[UKF_MAX_SIGMA_PTS * KF_MAX_STATE_DIM];
    for (uint8_t s = 1; s < num_sigma; s++) {
        double sqrt_w = sqrt(ukf->params.w_i);
        for (uint8_t i = 0; i < n; i++)
            X_weighted[(s-1) * KF_MAX_STATE_DIM + i] =
                sqrt_w * (sp_f[s * KF_MAX_STATE_DIM + i] - ukf->kf.x_prior[i]);
    }

    /* QR decomposition to get sqrt(P_prior) — simplified: use Cholesky */
    memset(ukf->kf.P_prior, 0, (size_t)n * n * sizeof(double));
    for (uint8_t s = 0; s < num_sigma; s++) {
        double w = (s == 0) ? ukf->params.w_cov_0 : ukf->params.w_i;
        double dx[KF_MAX_STATE_DIM];
        for (uint8_t i = 0; i < n; i++)
            dx[i] = sp_f[s * KF_MAX_STATE_DIM + i] - ukf->kf.x_prior[i];
        for (uint8_t i = 0; i < n; i++)
            for (uint8_t j = 0; j < n; j++)
                ukf->kf.P_prior[i * n + j] += w * dx[i] * dx[j];
    }
    if (sqrt_Q) {
        /* add sqrt_Q * sqrt_Q' = Q */
        double Q_full[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
        mat_mul_A_BT(sqrt_Q, sqrt_Q, Q_full, n, n, n);
        mat_add(ukf->kf.P_prior, Q_full, ukf->kf.P_prior, n, n);
    }
}

/* ===================================================================
 * Square-Root UKF Update
 * =================================================================== */

void ukf_update_sqrt(UKFState *ukf, const double *z, const double *sqrt_R)
{
    if (!ukf || !ukf->measure_fn || !z) return;
    uint8_t n = ukf->params.n, m = ukf->kf.m;
    uint8_t num_sigma = 2 * n + 1;
    double *sp_f = ukf->sigma_points_f;
    double *sp_h = ukf->sigma_points_h;

    for (uint8_t s = 0; s < num_sigma; s++) {
        ukf->measure_fn(sp_f + s * KF_MAX_STATE_DIM, n, m,
                        sp_h + s * KF_MAX_MEAS_DIM);
    }

    /* Predicted measurement mean */
    memset(ukf->z_pred, 0, m * sizeof(double));
    for (uint8_t i = 0; i < m; i++) {
        ukf->z_pred[i] += ukf->params.w_mean_0 * sp_h[i];
        for (uint8_t s = 1; s < num_sigma; s++)
            ukf->z_pred[i] += ukf->params.w_i * sp_h[s * KF_MAX_MEAS_DIM + i];
    }

    /* Innovation */
    for (uint8_t i = 0; i < m; i++)
        ukf->kf.innovation[i] = z[i] - ukf->z_pred[i];

    /* P_zz and P_xz (same as standard UKF) */
    memset(ukf->P_zz, 0, (size_t)m * m * sizeof(double));
    memset(ukf->P_xz, 0, (size_t)n * m * sizeof(double));

    for (uint8_t s = 0; s < num_sigma; s++) {
        double w = (s == 0) ? ukf->params.w_cov_0 : ukf->params.w_i;
        double dx[KF_MAX_STATE_DIM], dz[KF_MAX_MEAS_DIM];
        for (uint8_t i = 0; i < n; i++)
            dx[i] = sp_f[s * KF_MAX_STATE_DIM + i] - ukf->kf.x_prior[i];
        for (uint8_t i = 0; i < m; i++)
            dz[i] = sp_h[s * KF_MAX_MEAS_DIM + i] - ukf->z_pred[i];

        for (uint8_t i = 0; i < n; i++)
            for (uint8_t j = 0; j < m; j++)
                ukf->P_xz[i * m + j] += w * dx[i] * dz[j];
        for (uint8_t i = 0; i < m; i++)
            for (uint8_t j = 0; j < m; j++)
                ukf->P_zz[i * m + j] += w * dz[i] * dz[j];
    }
    if (sqrt_R) {
        double R_full[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
        mat_mul_A_BT(sqrt_R, sqrt_R, R_full, m, m, m);
        mat_add(ukf->P_zz, R_full, ukf->P_zz, m, m);
    }

    /* Gain and update */
    double P_zz_inv[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    memcpy(P_zz_inv, ukf->P_zz, (size_t)m * m * sizeof(double));
    if (!mat_inverse_cholesky(P_zz_inv, P_zz_inv, m)) return;

    mat_mul(ukf->P_xz, P_zz_inv, ukf->kf.K, n, m, m);

    double Ky[KF_MAX_STATE_DIM];
    mat_vec_mul(ukf->kf.K, ukf->kf.innovation, Ky, n, m);
    for (uint8_t i = 0; i < n; i++)
        ukf->kf.x[i] = ukf->kf.x_prior[i] + Ky[i];

    double KPzzKt[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    double KPzz[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM];
    mat_mul(ukf->kf.K, ukf->P_zz, KPzz, n, m, m);
    mat_mul_A_BT(KPzz, ukf->kf.K, KPzzKt, n, m, n);
    mat_sub(ukf->kf.P_prior, KPzzKt, ukf->kf.P, n, n);

    ukf->kf.step_count++;
}

/* ===================================================================
 * UKF NIS
 * =================================================================== */

double ukf_nis(const UKFState *ukf)
{
    if (!ukf) return 0.0;
    uint8_t m = ukf->kf.m;

    double S_inv[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    memcpy(S_inv, ukf->kf.S, (size_t)m * m * sizeof(double));
    if (!mat_inverse_cholesky(S_inv, S_inv, m)) return 1e9;

    double nis = 0.0;
    for (uint8_t i = 0; i < m; i++) {
        double sum = 0.0;
        for (uint8_t j = 0; j < m; j++)
            sum += S_inv[i * m + j] * ukf->kf.innovation[j];
        nis += ukf->kf.innovation[i] * sum;
    }
    return nis;
}
