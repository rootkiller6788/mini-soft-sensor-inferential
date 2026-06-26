/**
 * @file kalman_extended.c
 * @brief Extended Kalman Filter (EKF) Implementation
 *
 * Implements the first-order EKF for nonlinear systems by linearizing
 * the nonlinear dynamics and measurement models using Jacobians evaluated
 * at the current state estimate.
 *
 * L5 Algorithms: EKF predict/update, iterated EKF, second-order correction
 *
 * Reference: Jazwinski (1970) "Stochastic Processes and Filtering Theory"
 */
#include "kalman_extended.h"
#include "kalman_matrix_ops.h"
#include <math.h>
#include <string.h>

/* ===================================================================
 * EKF Initialization
 * =================================================================== */

void ekf_init(EKFState *ekf, const EKFModel *model,
              const double *x0, const double *P0,
              uint8_t n, uint8_t m, uint8_t p)
{
    if (!ekf || !model) return;
    memset(ekf, 0, sizeof(*ekf));
    ekf->model = model;
    ekf->use_iterative = 0;
    ekf->use_second_order = 0;
    ekf->iter_count = 0;

    /* Initialize the underlying linear KF */
    KalmanModel km;
    memset(&km, 0, sizeof(km));
    km.n = n; km.m = m; km.p = p;
    km.time_varying = 1;
    km.is_linear = 0;

    kf_init(&ekf->kf, &km, x0, P0, n, m);
}

/* ===================================================================
 * EKF Prediction
 * =================================================================== */

void ekf_predict(EKFState *ekf, const double *u)
{
    if (!ekf || !ekf->model || !ekf->kf.initialized) return;
    const EKFModel *m = ekf->model;
    uint8_t n = ekf->kf.n;

    if (!m->f || !m->F_jacobian) return;

    /* 1. Compute Jacobian F = df/dx at current state */
    m->F_jacobian(ekf->kf.x, u, n, m->p, ekf->F_jac_workspace);

    /* 2. Propagate state through nonlinear f */
    m->f(ekf->kf.x, u, n, m->p, ekf->kf.x_prior);

    /* 3. Propagate covariance: P_prior = F * P * F' + Q */
    double FP[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    mat_mul(ekf->F_jac_workspace, ekf->kf.P, FP, n, n, n);
    mat_mul_A_BT(FP, ekf->F_jac_workspace, ekf->kf.P_prior, n, n, n);
    mat_add(ekf->kf.P_prior, m->Q, ekf->kf.P_prior, n, n);
}

/* ===================================================================
 * EKF Update
 * =================================================================== */

void ekf_update(EKFState *ekf, const double *z)
{
    if (!ekf || !ekf->model || !z || !ekf->kf.initialized) return;
    const EKFModel *m = ekf->model;
    uint8_t n = ekf->kf.n, mm = ekf->kf.m;

    if (!m->h || !m->H_jacobian) return;

    /* 1. Predict measurement: z_pred = h(x_prior) */
    m->h(ekf->kf.x_prior, n, mm, ekf->z_pred);

    /* 2. Innovation: y = z - z_pred */
    for (uint8_t i = 0; i < mm; i++)
        ekf->kf.innovation[i] = z[i] - ekf->z_pred[i];

    /* 3. Compute measurement Jacobian H = dh/dx at x_prior */
    m->H_jacobian(ekf->kf.x_prior, n, mm, ekf->H_jac_workspace);

    /* 4. Innovation covariance: S = H * P_prior * H' + R */
    double HP[KF_MAX_MEAS_DIM * KF_MAX_STATE_DIM];
    mat_mul(ekf->H_jac_workspace, ekf->kf.P_prior, HP, mm, n, n);
    mat_mul_A_BT(HP, ekf->H_jac_workspace, ekf->kf.S, mm, n, mm);
    mat_add(ekf->kf.S, m->R, ekf->kf.S, mm, mm);

    /* 5. Kalman gain: K = P_prior * H' * inv(S) */
    double PHT[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM];
    mat_mul_A_BT(ekf->kf.P_prior, ekf->H_jac_workspace, PHT, n, n, mm);

    double S_inv[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    memcpy(S_inv, ekf->kf.S, (size_t)mm * mm * sizeof(double));
    if (!mat_inverse_cholesky(S_inv, S_inv, mm)) {
        ekf->kf.singular = 1;
        return;
    }
    mat_mul(PHT, S_inv, ekf->kf.K, n, mm, mm);

    /* 6. State update: x = x_prior + K * y */
    double Ky[KF_MAX_STATE_DIM];
    mat_vec_mul(ekf->kf.K, ekf->kf.innovation, Ky, n, mm);
    for (uint8_t i = 0; i < n; i++)
        ekf->kf.x[i] = ekf->kf.x_prior[i] + Ky[i];

    /* 7. Covariance update: P = (I - K*H) * P_prior */
    double KH[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    mat_mul(ekf->kf.K, ekf->H_jac_workspace, KH, n, mm, n);
    double I_KH[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    mat_identity(I_KH, n);
    mat_sub(I_KH, KH, I_KH, n, n);
    mat_mul(I_KH, ekf->kf.P_prior, ekf->kf.P, n, n, n);

    ekf->kf.step_count++;
}

/* ===================================================================
 * Combined EKF step
 * =================================================================== */

const double *ekf_step(EKFState *ekf, const double *z, const double *u)
{
    ekf_predict(ekf, u);
    ekf_update(ekf, z);
    return ekf->kf.x;
}

/* ===================================================================
 * Iterated EKF Update — re-linearize H at updated estimate
 * =================================================================== */

void ekf_update_iterated(EKFState *ekf, const double *z, uint8_t max_iter)
{
    if (!ekf || !ekf->model || !z || max_iter == 0) return;
    const EKFModel *m = ekf->model;
    uint8_t n = ekf->kf.n, mm = ekf->kf.m;

    /* Save original x_prior */
    double x_prior_saved[KF_MAX_STATE_DIM];
    memcpy(x_prior_saved, ekf->kf.x_prior, n * sizeof(double));

    /* Initial estimate: x_prior */
    double x_iter[KF_MAX_STATE_DIM];
    memcpy(x_iter, ekf->kf.x_prior, n * sizeof(double));

    for (uint8_t iter = 0; iter < max_iter; iter++) {
        /* Re-linearize H at current iterated estimate */
        m->H_jacobian(x_iter, n, mm, ekf->H_jac_workspace);

        /* Predicted measurement at iterated state */
        m->h(x_iter, n, mm, ekf->z_pred);

        /* Innovation with linearized correction */
        double y_iter[KF_MAX_MEAS_DIM];
        for (uint8_t i = 0; i < mm; i++)
            y_iter[i] = z[i] - ekf->z_pred[i]
                        + (/* H*(x_iter - x_prior) correction already in state */ 0.0);

        /* Actually: y = z - h(x_iter) + H*(x_iter - x_prior) */
        double H_dx[KF_MAX_MEAS_DIM];
        double dx[KF_MAX_STATE_DIM];
        for (uint8_t i = 0; i < n; i++) dx[i] = x_iter[i] - x_prior_saved[i];
        mat_vec_mul(ekf->H_jac_workspace, dx, H_dx, mm, n);
        for (uint8_t i = 0; i < mm; i++)
            y_iter[i] = z[i] - ekf->z_pred[i] + H_dx[i];

        /* Standard KF update with this linearization */
        double HP[KF_MAX_MEAS_DIM * KF_MAX_STATE_DIM];
        mat_mul(ekf->H_jac_workspace, ekf->kf.P_prior, HP, mm, n, n);
        mat_mul_A_BT(HP, ekf->H_jac_workspace, ekf->kf.S, mm, n, mm);
        mat_add(ekf->kf.S, m->R, ekf->kf.S, mm, mm);

        double PHT[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM];
        mat_mul_A_BT(ekf->kf.P_prior, ekf->H_jac_workspace, PHT, n, n, mm);

        double S_inv[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
        memcpy(S_inv, ekf->kf.S, (size_t)mm * mm * sizeof(double));
        if (!mat_inverse_cholesky(S_inv, S_inv, mm)) break;

        mat_mul(PHT, S_inv, ekf->kf.K, n, mm, mm);

        double Ky[KF_MAX_STATE_DIM];
        mat_vec_mul(ekf->kf.K, y_iter, Ky, n, mm);

        /* Update iterated estimate */
        double x_new[KF_MAX_STATE_DIM];
        for (uint8_t i = 0; i < n; i++)
            x_new[i] = x_prior_saved[i] + Ky[i];

        /* Check convergence */
        double diff = 0.0;
        for (uint8_t i = 0; i < n; i++) {
            double d = x_new[i] - x_iter[i];
            diff += d * d;
        }
        memcpy(x_iter, x_new, n * sizeof(double));
        if (diff < 1e-12) break;
    }

    /* Finalize: update state and covariance */
    memcpy(ekf->kf.x, x_iter, n * sizeof(double));

    double KH[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    mat_mul(ekf->kf.K, ekf->H_jac_workspace, KH, n, mm, n);
    double I_KH[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    mat_identity(I_KH, n);
    mat_sub(I_KH, KH, I_KH, n, n);
    mat_mul(I_KH, ekf->kf.P_prior, ekf->kf.P, n, n, n);

    ekf->kf.step_count++;
    ekf->iter_count = max_iter;
}

/* ===================================================================
 * Second-Order EKF correction (Hessian term)
 * =================================================================== */

void ekf_second_order_correction(EKFState *ekf,
                                  const double *meas_hessian)
{
    if (!ekf || !meas_hessian) return;
    uint8_t n = ekf->kf.n, m = ekf->kf.m;

    /* Add Hessian contribution to innovation covariance:
     * S = S + 0.5 * tr(Hessian_i * P * Hessian_j * P)
     * This increases innovation covariance to account for nonlinearity bias.
     */
    for (uint8_t i = 0; i < m; i++) {
        for (uint8_t j = 0; j < m; j++) {
            /* Hessian_i and Hessian_j are n x n */
            const double *Hi = meas_hessian + i * n * n;
            const double *Hj = meas_hessian + j * n * n;

            /* Compute tr(Hi * P * Hj * P) */
            double HiP[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
            mat_mul(Hi, ekf->kf.P_prior, HiP, n, n, n);
            double HjP[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
            mat_mul(Hj, ekf->kf.P_prior, HjP, n, n, n);

            double trace_val = 0.0;
            for (uint8_t r = 0; r < n; r++) {
                for (uint8_t c = 0; c < n; c++)
                    trace_val += HiP[r * n + c] * HjP[c * n + r];
            }

            ekf->kf.S[i * m + j] += 0.5 * trace_val;
        }
    }
}

/* ===================================================================
 * EKF NIS for consistency monitoring
 * =================================================================== */

double ekf_nis(const EKFState *ekf)
{
    if (!ekf) return 0.0;
    uint8_t m = ekf->kf.m;

    double S_inv[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    memcpy(S_inv, ekf->kf.S, (size_t)m * m * sizeof(double));
    if (!mat_inverse_cholesky(S_inv, S_inv, m)) return 1e9;

    double nis = 0.0;
    for (uint8_t i = 0; i < m; i++) {
        double sum = 0.0;
        for (uint8_t j = 0; j < m; j++)
            sum += S_inv[i * m + j] * ekf->kf.innovation[j];
        nis += ekf->kf.innovation[i] * sum;
    }
    return nis;
}

/* ===================================================================
 * EKF consistency check
 * =================================================================== */

int ekf_is_consistent(const EKFState *ekf)
{
    if (!ekf) return 0;

    /* Check basic KF consistency */
    if (!kf_is_consistent(&ekf->kf)) return 0;

    /* Check NIS threshold for chi^2(m) at 95% */
    double nis = ekf_nis(ekf);
    double chi2_95[] = {3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507};
    uint8_t m = ekf->kf.m;
    if (m > 0 && m <= 8 && nis > chi2_95[m-1]) return 0;

    return 1;
}
