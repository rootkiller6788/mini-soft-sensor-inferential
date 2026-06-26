/**
 * @file kalman_core.c
 * @brief Standard Linear Kalman Filter — Core Implementation
 *
 * Implements the five Kalman filter equations plus supporting routines
 * for initialization, prediction, update, diagnostics, and smoothing.
 *
 * L1-L5: Complete Kalman filter algorithm with numerical safeguards.
 *
 * References:
 *   Kalman (1960) "A New Approach to Linear Filtering..."
 *   Gelb (1974) "Applied Optimal Estimation"
 *   Simon (2006) "Optimal State Estimation"
 */
#include "kalman_core.h"
#include "kalman_matrix_ops.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* ---- helper: ensure min/max bounds ---- */
static double clamp_val(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ===================================================================
 * Initialization
 * =================================================================== */

void kf_init(KalmanFilterState *kf, KalmanModel *model,
             const double *x0, const double *P0,
             uint8_t n, uint8_t m)
{
    if (!kf || !model) return;
    if (n == 0 || n > KF_MAX_STATE_DIM) return;
    if (m == 0 || m > KF_MAX_MEAS_DIM)  return;

    memset(kf, 0, sizeof(*kf));
    kf->n = n;
    kf->m = m;
    kf->step_count = 0;
    kf->initialized = 1;
    kf->converged = 0;
    kf->singular = 0;
    kf->diverged = 0;

    /* copy state */
    if (x0) memcpy(kf->x, x0, n * sizeof(double));
    if (P0) memcpy(kf->P, P0, (size_t)n * n * sizeof(double));
    else    mat_identity(kf->P, n);  /* default: identity */

    /* mark model */
    model->n = n;
    model->m = m;
    model->time_varying = 0;
    model->is_linear = 1;
}

/* ===================================================================
 * Prediction step
 * =================================================================== */

void kf_predict(KalmanFilterState *kf, const KalmanModel *model,
                const double *u)
{
    if (!kf || !model || !kf->initialized) return;
    uint8_t n = kf->n;
    uint8_t m = kf->m;
    uint8_t p = model->p;

    /* --- 1. state prediction: x_prior = F * x + B * u --- */
    if (u && p > 0) {
        /* x_prior = F * x */
        mat_vec_mul(model->F, kf->x, kf->x_prior, n, n);
        /* x_prior += B * u */
        double Bu[KF_MAX_STATE_DIM] = {0};
        mat_vec_mul(model->B, u, Bu, n, p);
        for (uint8_t i = 0; i < n; i++) kf->x_prior[i] += Bu[i];
    } else {
        mat_vec_mul(model->F, kf->x, kf->x_prior, n, n);
    }

    /* --- 2. covariance prediction: P_prior = F * P * F' + Q --- */
    double FP[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM] = {0};
    mat_mul(model->F, kf->P, FP, n, n, n);
    mat_mul_A_BT(FP, model->F, kf->P_prior, n, n, n);
    mat_add(kf->P_prior, model->Q, kf->P_prior, n, n);
}

/* ===================================================================
 * Update step — standard (simpler) form
 * =================================================================== */

void kf_update(KalmanFilterState *kf, const KalmanModel *model,
               const double *z)
{
    if (!kf || !model || !z || !kf->initialized) return;
    uint8_t n = kf->n, m = kf->m;

    /* --- 1. innovation: y = z - H * x_prior --- */
    double Hx[KF_MAX_MEAS_DIM] = {0};
    mat_vec_mul(model->H, kf->x_prior, Hx, m, n);
    for (uint8_t i = 0; i < m; i++) kf->innovation[i] = z[i] - Hx[i];

    /* --- 2. innovation covariance: S = H * P_prior * H' + R --- */
    double HP[KF_MAX_MEAS_DIM * KF_MAX_STATE_DIM] = {0};
    mat_mul(model->H, kf->P_prior, HP, m, n, n);
    mat_mul_A_BT(HP, model->H, kf->S, m, n, m);
    mat_add(kf->S, model->R, kf->S, m, m);

    /* --- 3. Kalman gain: K = P_prior * H' * inv(S) --- */
    double PHT[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM] = {0};
    mat_mul_A_BT(kf->P_prior, model->H, PHT, n, n, m);

    double S_inv[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    memcpy(S_inv, kf->S, (size_t)m * m * sizeof(double));
    int ok = mat_inverse_cholesky(S_inv, S_inv, m);
    if (!ok) { kf->singular = 1; return; }

    mat_mul(PHT, S_inv, kf->K, n, m, m);

    /* --- 4. state update: x = x_prior + K * y --- */
    double Ky[KF_MAX_STATE_DIM] = {0};
    mat_vec_mul(kf->K, kf->innovation, Ky, n, m);
    for (uint8_t i = 0; i < n; i++) kf->x[i] = kf->x_prior[i] + Ky[i];

    /* --- 5. covariance update: P = (I - K*H) * P_prior --- */
    double KH[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM] = {0};
    mat_mul(kf->K, model->H, KH, n, m, n);
    double I_KH[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    mat_identity(I_KH, n);
    mat_sub(I_KH, KH, I_KH, n, n);
    mat_mul(I_KH, kf->P_prior, kf->P, n, n, n);

    kf->step_count++;
}

/* ===================================================================
 * Combined step
 * =================================================================== */

const double *kf_step(KalmanFilterState *kf, const KalmanModel *model,
                      const double *z, const double *u)
{
    kf_predict(kf, model, u);
    kf_update(kf, model, z);
    return kf->x;
}

/* ===================================================================
 * Joseph-form update — symmetrical, guaranteed PSD
 * =================================================================== */

void kf_joseph_update(KalmanFilterState *kf, const KalmanModel *model)
{
    if (!kf || !model) return;
    uint8_t n = kf->n, m = kf->m;

    /* P = (I-KH)*P_prior*(I-KH)' + K*R*K' */
    double KH[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM] = {0};
    mat_mul(kf->K, model->H, KH, n, m, n);

    /* T1 = I - K*H */
    double I_mat[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    mat_identity(I_mat, n);
    double T1[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    mat_sub(I_mat, KH, T1, n, n);

    /* T2 = T1 * P_prior */
    double T2[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM] = {0};
    mat_mul(T1, kf->P_prior, T2, n, n, n);

    /* Part1 = T2 * T1' */
    double Part1[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM] = {0};
    mat_mul_A_BT(T2, T1, Part1, n, n, n);

    /* Part2 = K * R * K' */
    double KR[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM] = {0};
    mat_mul(kf->K, model->R, KR, n, m, m);
    double Part2[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM] = {0};
    mat_mul_A_BT(KR, kf->K, Part2, n, m, n);

    /* P = Part1 + Part2 */
    mat_add(Part1, Part2, kf->P, n, n);
}

/* ===================================================================
 * Sequential scalar update — no matrix inversion needed
 * =================================================================== */

void kf_sequential_update(KalmanFilterState *kf, const KalmanModel *model,
                           const double *z)
{
    if (!kf || !model || !z) return;
    uint8_t n = kf->n, m = kf->m;

    /* Process each measurement channel independently */
    for (uint8_t j = 0; j < m; j++) {
        /* Innovation for this channel: y_j = z_j - H_row_j * x */
        double y_j = z[j];
        for (uint8_t i = 0; i < n; i++)
            y_j -= model->H[j * n + i] * kf->x_prior[i];
        kf->innovation[j] = y_j;

        /* Innovation variance: s = h * P * h' + r_jj */
        double Ph[KF_MAX_STATE_DIM] = {0};
        for (uint8_t i = 0; i < n; i++) {
            double sum = 0.0;
            for (uint8_t kk = 0; kk < n; kk++)
                sum += kf->P_prior[i * n + kk] * model->H[j * n + kk];
            Ph[i] = sum;
        }
        double s = model->R[j * m + j];
        for (uint8_t i = 0; i < n; i++)
            s += model->H[j * n + i] * Ph[i];

        if (s < MAT_EPSILON) continue;  /* skip degenerate */

        /* Kalman gain for this channel: K = P * h' / s */
        double kk_j[KF_MAX_STATE_DIM];
        for (uint8_t i = 0; i < n; i++) kk_j[i] = Ph[i] / s;

        /* State update */
        for (uint8_t i = 0; i < n; i++)
            kf->x_prior[i] += kk_j[i] * y_j;

        /* Covariance update: P = P - (K * K') * s */
        for (uint8_t i = 0; i < n; i++)
            for (uint8_t ik = 0; ik < n; ik++)
                kf->P_prior[i * n + ik] -= kk_j[i] * kk_j[ik] * s;
    }

    /* Copy updated prior to posterior */
    memcpy(kf->x, kf->x_prior, n * sizeof(double));
    memcpy(kf->P, kf->P_prior, (size_t)n * n * sizeof(double));
    kf->step_count++;
}

/* ===================================================================
 * Solve DARE for steady-state gain
 * =================================================================== */

int kf_solve_dare(KalmanFilterState *kf, const KalmanModel *model,
                  uint32_t max_iter)
{
    if (!kf || !model) return -1;
    if (max_iter == 0) max_iter = 1000;
    uint8_t n = kf->n, m = kf->m;

    /* Initialize P with Q (or identity) */
    memcpy(kf->P, model->Q, (size_t)n * n * sizeof(double));
    double P_new[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    double FPFt[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    for (uint32_t iter = 1; iter <= max_iter; iter++) {
        /* F * P * F' */
        double FP[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
        mat_mul(model->F, kf->P, FP, n, n, n);
        mat_mul_A_BT(FP, model->F, FPFt, n, n, n);

        /* H * P * H' + R */
        double HP[KF_MAX_MEAS_DIM * KF_MAX_STATE_DIM];
        mat_mul(model->H, kf->P, HP, m, n, n);
        double HPHtR[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
        mat_mul_A_BT(HP, model->H, HPHtR, m, n, m);
        mat_add(HPHtR, model->R, HPHtR, m, m);

        /* inv(HPHtR) */
        double invS[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
        memcpy(invS, HPHtR, (size_t)m * m * sizeof(double));
        if (!mat_inverse_cholesky(invS, invS, m)) return -1;

        /* K = F * P * H' * inv(HPHtR) */
        double FPHt[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM];
        mat_mul_A_BT(FP, model->H, FPHt, n, n, m);
        mat_mul(FPHt, invS, kf->K, n, m, m);

        /* P_new = FPFt - K * HPHtR * K' + Q */
        double KSKt[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
        double KS[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM];
        mat_mul(kf->K, HPHtR, KS, n, m, m);
        mat_mul_A_BT(KS, kf->K, KSKt, n, m, n);
        mat_sub(FPFt, KSKt, P_new, n, n);
        mat_add(P_new, model->Q, P_new, n, n);

        /* Check convergence */
        double diff = 0.0;
        for (uint8_t i = 0; i < n; i++)
            for (uint8_t j = 0; j < n; j++) {
                double d = P_new[i * n + j] - kf->P[i * n + j];
                diff += d * d;
            }

        memcpy(kf->P, P_new, (size_t)n * n * sizeof(double));

        if (diff < KF_CONVERGENCE_TOL * KF_CONVERGENCE_TOL)
            return (int)iter;
    }
    return -1;
}

/* ===================================================================
 * Fading memory prediction
 * =================================================================== */

void kf_fading_memory_predict(KalmanFilterState *kf, const KalmanModel *model,
                               const double *u, double lambda)
{
    if (!kf || !model) return;
    if (lambda <= 0.0 || lambda > 1.0) lambda = 0.95;
    uint8_t n = kf->n;

    /* Standard prediction first */
    kf_predict(kf, model, u);

    /* Scale F*P*F' by 1/lambda */
    double FP[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    mat_mul(model->F, kf->P, FP, n, n, n);
    double FPFt[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    mat_mul_A_BT(FP, model->F, FPFt, n, n, n);

    double scale = 1.0 / lambda;
    for (uint8_t i = 0; i < n * n; i++) {
        kf->P_prior[i] = scale * FPFt[i] + model->Q[i];
    }
}

/* ===================================================================
 * Convergence check
 * =================================================================== */

int kf_has_converged(const KalmanFilterState *kf, double tol)
{
    if (!kf || kf->step_count < 2) return 0;
    double diff = 0.0;
    for (uint8_t i = 0; i < (uint8_t)(kf->n * kf->m); i++) {
        double d = kf->K[i] - kf->K_prev[i];
        diff += d * d;
    }
    return (sqrt(diff) < tol) ? 1 : 0;
}

/* ===================================================================
 * Accessors
 * =================================================================== */

const double *kf_get_state(const KalmanFilterState *kf)     { return kf ? kf->x : NULL; }
const double *kf_get_covariance(const KalmanFilterState *kf){ return kf ? kf->P : NULL; }
const double *kf_get_gain(const KalmanFilterState *kf)      { return kf ? kf->K : NULL; }
const double *kf_get_innovation(const KalmanFilterState *kf){ return kf ? kf->innovation : NULL; }

double kf_get_state_element(const KalmanFilterState *kf, uint8_t idx) {
    if (!kf || idx >= kf->n) return 0.0;
    return kf->x[idx];
}

/* ===================================================================
 * Log-likelihood
 * =================================================================== */

double kf_log_likelihood(const KalmanFilterState *kf)
{
    if (!kf) return -DBL_MAX;
    uint8_t m = kf->m;

    /* log p = -0.5 * (m*log(2*pi) + log|S| + y' * inv(S) * y) */
    double S_copy[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    memcpy(S_copy, kf->S, (size_t)m * m * sizeof(double));
    double log_det_S = mat_logdet_cholesky(S_copy, m);

    double S_inv[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    memcpy(S_inv, kf->S, (size_t)m * m * sizeof(double));
    if (!mat_inverse_cholesky(S_inv, S_inv, m)) return -DBL_MAX;

    double quad = 0.0;
    for (uint8_t i = 0; i < m; i++) {
        double sum = 0.0;
        for (uint8_t j = 0; j < m; j++)
            sum += S_inv[i * m + j] * kf->innovation[j];
        quad += kf->innovation[i] * sum;
    }

    return -0.5 * (m * log(2.0 * M_PI) + log_det_S + quad);
}

/* ===================================================================
 * RTS smoother (single backward step)
 * =================================================================== */

void kf_rts_smooth(const KalmanFilterState *kf_history,
                   const KalmanModel *model,
                   uint32_t N,
                   double *x_smoothed,
                   double *P_smoothed)
{
    if (!kf_history || !model || N < 2) return;
    uint8_t n = model->n;

    /* Copy final filtered estimate */
    uint32_t idx_last = N - 1;
    if (x_smoothed)
        memcpy(x_smoothed + idx_last * n, kf_history[idx_last].x, n * sizeof(double));
    if (P_smoothed)
        memcpy(P_smoothed + idx_last * n * n, kf_history[idx_last].P,
               (size_t)n * n * sizeof(double));

    /* Backward recursion */
    for (uint32_t k_int = N - 1; k_int > 0; k_int--) {
        uint32_t kk = k_int;
        uint8_t k = (uint8_t)(kk - 1);  /* index for k-1 */

        /* G = P[k] * F' * inv(P_prior[k+1]) */
        double PFT[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
        mat_mul_A_BT(kf_history[k].P, model->F, PFT, n, n, n);

        double P_prior_inv[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
        memcpy(P_prior_inv, kf_history[kk].P_prior, (size_t)n * n * sizeof(double));
        if (!mat_inverse_cholesky(P_prior_inv, P_prior_inv, n)) continue;

        double G[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
        mat_mul(PFT, P_prior_inv, G, n, n, n);

        /* x_smooth[k] = x[k] + G * (x_smooth[k+1] - x_prior[k+1]) */
        double dx[KF_MAX_STATE_DIM];
        for (uint8_t i = 0; i < n; i++)
            dx[i] = x_smoothed[kk * n + i] - kf_history[kk].x_prior[i];

        if (x_smoothed) {
            for (uint8_t i = 0; i < n; i++) {
                double corr = 0.0;
                for (uint8_t j = 0; j < n; j++) corr += G[i * n + j] * dx[j];
                x_smoothed[k * n + i] = kf_history[k].x[i] + corr;
            }
        }

        /* P_smooth[k] = P[k] + G * (P_smooth[k+1] - P_prior[k+1]) * G' */
        if (P_smoothed) {
            double dP[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
            double *Ps_kp1 = P_smoothed + kk * n * n;
            mat_sub(Ps_kp1, kf_history[kk].P_prior, dP, n, n);
            double GdP[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
            mat_mul(G, dP, GdP, n, n, n);
            double GdPGt[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
            mat_mul_A_BT(GdP, G, GdPGt, n, n, n);
            mat_add(kf_history[k].P, GdPGt, P_smoothed + k * n * n, n, n);
        }
    }
}

/* ===================================================================
 * Diagnostics
 * =================================================================== */

void kf_diagnostics(const KalmanFilterState *kf, KalmanDiagnostics *diag)
{
    if (!kf || !diag) return;
    uint8_t n = kf->n, m = kf->m;

    diag->total_measurements = kf->step_count;

    /* NIS = y' * inv(S) * y */
    double S_inv[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    memcpy(S_inv, kf->S, (size_t)m * m * sizeof(double));
    if (mat_inverse_cholesky(S_inv, S_inv, m)) {
        diag->nis = 0.0;
        for (uint8_t i = 0; i < m; i++) {
            double sum = 0.0;
            for (uint8_t j = 0; j < m; j++)
                sum += S_inv[i * m + j] * kf->innovation[j];
            diag->nis += kf->innovation[i] * sum;
        }
    } else {
        diag->nis = 1e9;
    }

    /* NEES = (x_true - x_est)' * inv(P) * (x_true - x_est)
     * Since we usually don't have x_true, set to NIS approximation */
    diag->nees = diag->nis;

    /* Mahalanobis distance */
    diag->mahalanobis_dist = sqrt(diag->nis);

    /* Condition numbers */
    double P_copy[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    memcpy(P_copy, kf->P, (size_t)n * n * sizeof(double));
    diag->P_cond = mat_cond_estimate_sym(P_copy, n, 20);

    double S_copy[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    memcpy(S_copy, kf->S, (size_t)m * m * sizeof(double));
    diag->S_cond = mat_cond_estimate_sym(S_copy, m, 20);

    /* P_trace */
    diag->P_trace = mat_trace(kf->P, n);

    /* SNR ratio: estimated from innovation and P trace.
     * Innovation magnitude vs state uncertainty gives a signal-quality indicator. */
    double innov_norm = 0.0;
    for (uint8_t i = 0; i < m; i++) innov_norm += kf->innovation[i] * kf->innovation[i];
    double p_trace = mat_trace(kf->P, n);
    diag->snr_ratio = (p_trace > MAT_EPSILON) ? innov_norm / p_trace : 0.0;

    /* Fault detection */
    double chi2_95[8] = {3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507};
    double threshold = (m <= 8) ? chi2_95[m-1] : chi2_95[7];
    diag->nis_fault = (diag->nis > threshold) ? 1 : 0;
    diag->cov_fault = (diag->P_cond > 1e6) ? 1 : 0;
    diag->divergence = (diag->P_trace > 1e9) ? 1 : 0;
    diag->residual_fault = 0;
}

/* ===================================================================
 * Reset & Consistency
 * =================================================================== */

void kf_reset(KalmanFilterState *kf, const double *x0, const double *P0)
{
    if (!kf) return;
    uint8_t n = kf->n;
    if (x0) memcpy(kf->x, x0, n * sizeof(double));
    if (P0) memcpy(kf->P, P0, (size_t)n * n * sizeof(double));
    kf->step_count = 0;
    kf->converged = 0;
    kf->singular = 0;
    kf->diverged = 0;
}

int kf_is_consistent(const KalmanFilterState *kf)
{
    if (!kf) return 0;
    uint8_t n = kf->n;

    /* Check symmetry */
    if (!mat_is_symmetric(kf->P, n, 1e-10)) return 0;
    if (!mat_is_symmetric(kf->P_prior, n, 1e-10)) return 0;

    /* Check positive definiteness (approximate: trace > 0 and not singular) */
    double tr = mat_trace(kf->P, n);
    if (tr <= 0.0 || tr > 1e12) return 0;

    /* Check state for NaN/Inf */
    for (uint8_t i = 0; i < n; i++) {
        if (!isfinite(kf->x[i])) return 0;
    }

    return 1;
}
