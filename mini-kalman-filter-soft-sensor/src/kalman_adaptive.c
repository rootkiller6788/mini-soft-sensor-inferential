/**
 * @file kalman_adaptive.c
 * @brief Adaptive Kalman Filtering — Online Noise Covariance Estimation
 *
 * Implements methods for estimating unknown or time-varying process noise Q
 * and measurement noise R from the innovation sequence.
 *
 * L5 Algorithms: Covariance matching, innovation correlation, Bayesian MAP
 * L8 Advanced Topics: IMM for multi-model estimation
 *
 * References:
 *   Mehra (1970, 1972) approaches to adaptive filtering
 *   Blom & Bar-Shalom (1988) IMM algorithm
 */
#include "kalman_adaptive.h"
#include "kalman_matrix_ops.h"
#include <math.h>
#include <string.h>

/* ===================================================================
 * Adaptive KF Initialization
 * =================================================================== */

void akf_init(AKFState *akf, const double *x0, const double *P0,
              const double *Q0, const double *R0,
              uint8_t n, uint8_t m,
              uint8_t adapt_q, uint8_t adapt_r,
              uint8_t window_sz)
{
    if (!akf) return;
    memset(akf, 0, sizeof(*akf));

    akf->n = n; akf->m = m;
    akf->adapt_q = adapt_q;
    akf->adapt_r = adapt_r;
    akf->use_cov_matching = 1;
    akf->use_bayesian = 0;

    /* Init core KF */
    KalmanModel km;
    memset(&km, 0, sizeof(km));
    km.n = n; km.m = m;

    kf_init(&akf->kf, &km, x0, P0, n, m);

    /* Store initial noise estimates */
    if (Q0) memcpy(akf->Q_est, Q0, (size_t)n * n * sizeof(double));
    else    mat_identity(akf->Q_est, n);
    if (R0) memcpy(akf->R_est, R0, (size_t)m * m * sizeof(double));
    else    mat_identity(akf->R_est, m);

    /* Init innovation stats */
    akf->innov_stats.m = m;
    akf->innov_stats.window_size = window_sz > AKF_DEFAULT_WINDOW
                                    ? AKF_DEFAULT_WINDOW : window_sz;
    if (akf->innov_stats.window_size == 0)
        akf->innov_stats.window_size = AKF_DEFAULT_WINDOW;
    akf->innov_stats.forgetting_factor = AKF_DEFAULT_FORGET_FACTOR;
    akf->innov_stats.window_count = 0;
    akf->innov_stats.window_full = 0;
}

/* ===================================================================
 * A-KF Predict with adaptive Q
 * =================================================================== */

void akf_predict(AKFState *akf, const KalmanModel *model, const double *u)
{
    if (!akf || !model) return;
    uint8_t n = akf->n;

    /* Use estimated Q instead of model's Q */
    KalmanModel mod_copy = *model;
    memcpy(mod_copy.Q, akf->Q_est, (size_t)n * n * sizeof(double));

    kf_predict(&akf->kf, &mod_copy, u);
}

/* ===================================================================
 * A-KF Update with adaptive R
 * =================================================================== */

void akf_update(AKFState *akf, const KalmanModel *model, const double *z)
{
    if (!akf || !model || !z) return;
    uint8_t m = akf->m;

    /* Use estimated R instead of model's R */
    KalmanModel mod_copy = *model;
    memcpy(mod_copy.R, akf->R_est, (size_t)m * m * sizeof(double));

    kf_update(&akf->kf, &mod_copy, z);

    /* Update innovation statistics */
    akf_update_innovation_stats(akf);

    /* Run adaptation */
    if (akf->adapt_q && akf->use_cov_matching)
        akf_estimate_Q_covariance_matching(akf, model);
    if (akf->adapt_r)
        akf_estimate_R_innovation(akf, model);
    if (akf->use_bayesian)
        akf_estimate_bayesian(akf, model);
}

/* ===================================================================
 * Combined step
 * =================================================================== */

const double *akf_step(AKFState *akf, const KalmanModel *model,
                       const double *z, const double *u)
{
    akf_predict(akf, model, u);
    akf_update(akf, model, z);
    return akf->kf.x;
}

/* ===================================================================
 * Update Innovation Statistics Buffer
 * =================================================================== */

void akf_update_innovation_stats(AKFState *akf)
{
    if (!akf) return;
    AKFInnovationStats *st = &akf->innov_stats;
    uint8_t m = st->m;
    uint8_t w = st->window_size;

    /* Store current innovation in circular buffer */
    uint8_t idx = st->window_count % w;
    memcpy(st->innov_history + idx * KF_MAX_MEAS_DIM,
           akf->kf.innovation, m * sizeof(double));

    st->window_count++;
    if (st->window_count >= w) {
        st->window_full = 1;
        st->window_count = w;  /* clamp */
    }

    /* Update exponential weighted estimate: C = lambda*C + (1-lambda)*y*y' */
    double lambda = st->forgetting_factor;
    double one_minus_lambda = 1.0 - lambda;

    /* If first update, initialize with y*y' */
    if (st->window_count == 1) {
        for (uint8_t i = 0; i < m; i++)
            for (uint8_t j = 0; j < m; j++)
                st->innov_cov_est[i * m + j] =
                    akf->kf.innovation[i] * akf->kf.innovation[j];
    } else {
        for (uint8_t i = 0; i < m; i++)
            for (uint8_t j = 0; j < m; j++)
                st->innov_cov_est[i * m + j] =
                    lambda * st->innov_cov_est[i * m + j]
                    + one_minus_lambda * akf->kf.innovation[i]
                      * akf->kf.innovation[j];
    }
}

/* ===================================================================
 * Q Estimation via Covariance Matching
 * =================================================================== */

void akf_estimate_Q_covariance_matching(AKFState *akf,
                                         const KalmanModel *model)
{
    if (!akf || !model) return;
    uint8_t n = akf->n, m = akf->m;

    /* Q_est = K * C_innov * K' where C_innov is estimated innovation cov */
    double KCKt[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    double KC[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM];
    mat_mul(akf->kf.K, akf->innov_stats.innov_cov_est, KC, n, m, m);
    mat_mul_A_BT(KC, akf->kf.K, KCKt, n, m, n);

    /* Blend with existing estimate using exponential forgetting */
    double alpha = 0.1;  /* learning rate */
    for (uint8_t i = 0; i < n * n; i++)
        akf->Q_est[i] = (1.0 - alpha) * akf->Q_est[i] + alpha * KCKt[i];

    /* Ensure positive definiteness: add small diagonal if needed */
    for (uint8_t i = 0; i < n; i++)
        if (akf->Q_est[i * n + i] < MAT_EPSILON)
            akf->Q_est[i * n + i] = MAT_EPSILON;
}

/* ===================================================================
 * R Estimation via Innovation Correlation
 * =================================================================== */

void akf_estimate_R_innovation(AKFState *akf, const KalmanModel *model)
{
    if (!akf || !model) return;
    uint8_t n = akf->n, m = akf->m;

    /* R_est = C_innov - H * P_prior * H' */
    double HPHt[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];

    double HP[KF_MAX_MEAS_DIM * KF_MAX_STATE_DIM];
    mat_mul(model->H, akf->kf.P_prior, HP, m, n, n);
    mat_mul_A_BT(HP, model->H, HPHt, m, n, m);

    double R_new[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    mat_sub(akf->innov_stats.innov_cov_est, HPHt, R_new, m, m);

    /* Blend with existing estimate */
    double alpha = 0.1;
    for (uint8_t i = 0; i < m * m; i++)
        akf->R_est[i] = (1.0 - alpha) * akf->R_est[i] + alpha * R_new[i];

    /* Ensure positive definiteness */
    for (uint8_t i = 0; i < m; i++)
        if (akf->R_est[i * m + i] < MAT_EPSILON)
            akf->R_est[i * m + i] = MAT_EPSILON;
}

/* ===================================================================
 * Bayesian MAP estimation of Q and R
 * =================================================================== */

void akf_estimate_bayesian(AKFState *akf, const KalmanModel *model)
{
    if (!akf || !model) return;
    uint8_t n = akf->n, m = akf->m;

    /* Bayesian update using inverse-Wishart conjugate prior.
     * Simplified: weight innovation-based estimates by sample count.
     *
     * Q_post = (nu_0*Q_0 + N * Q_ml) / (nu_0 + N)
     * Where Q_ml comes from covariance matching.
     */

    /* Effective sample count */
    double N_eff = (double)akf->innov_stats.window_count;
    double nu_0 = 5.0;  /* prior strength */

    /* Q estimate (already computed in cov matching; blend here) */
    for (uint8_t i = 0; i < n * n; i++) {
        akf->Q_est[i] = (nu_0 * akf->Q_est[i] + N_eff * akf->Q_est[i])
                        / (nu_0 + N_eff);
    }
    /* R estimate */
    for (uint8_t i = 0; i < m * m; i++) {
        akf->R_est[i] = (nu_0 * akf->R_est[i] + N_eff * akf->R_est[i])
                        / (nu_0 + N_eff);
    }
}

/* ===================================================================
 * IMM Initialization
 * =================================================================== */

void imm_init(AKFMultiModel *imm,
              const KalmanModel *models,
              const double *x0, const double *P0,
              const double *mu0, const double *Pi,
              uint8_t num_models, uint8_t n, uint8_t m)
{
    if (!imm || !models || num_models == 0 || num_models > AKF_MAX_MODELS)
        return;
    memset(imm, 0, sizeof(*imm));
    imm->num_models = num_models;
    imm->n = n;
    imm->m = m;
    imm->initialized = 1;

    /* Initialize each filter */
    for (uint8_t r = 0; r < num_models; r++) {
        KalmanModel mcopy = models[r];
        kf_init(&imm->filters[r], &mcopy, x0, P0, n, m);
        memcpy(&imm->models[r], &models[r], sizeof(KalmanModel));
    }

    /* Initialize model probabilities */
    if (mu0) {
        memcpy(imm->mu, mu0, num_models * sizeof(double));
    } else {
        double p = 1.0 / num_models;
        for (uint8_t r = 0; r < num_models; r++) imm->mu[r] = p;
    }

    /* Transition matrix */
    if (Pi) {
        memcpy(imm->Pi, Pi, (size_t)num_models * num_models * sizeof(double));
    } else {
        /* Default: high probability of staying in same model */
        for (uint8_t r = 0; r < num_models; r++) {
            for (uint8_t c = 0; c < num_models; c++)
                imm->Pi[r * num_models + c] = (r == c) ? 0.9 : 0.1 / (num_models - 1);
        }
    }
}

/* ===================================================================
 * IMM Step
 * =================================================================== */

void imm_step(AKFMultiModel *imm, const double *z, const double *u)
{
    if (!imm || !z || !imm->initialized) return;
    uint8_t r, n = imm->n, m = imm->m, N = imm->num_models;

    /* Phase 1: Interaction (mixing) */
    double cbar[AKF_MAX_MODELS];  /* normalization */
    double mu_mix[AKF_MAX_MODELS * AKF_MAX_MODELS];  /* mu_{i|j} */

    /* Mixing probabilities: mu_{i|j} = Pi[i][j] * mu[i] / cbar[j] */
    for (uint8_t j = 0; j < N; j++) {
        cbar[j] = 0.0;
        for (uint8_t i = 0; i < N; i++)
            cbar[j] += imm->Pi[i * N + j] * imm->mu[i];
    }

    for (uint8_t i = 0; i < N; i++) {
        for (uint8_t j = 0; j < N; j++) {
            mu_mix[i * N + j] = (cbar[j] > MAT_EPSILON)
                ? imm->Pi[i * N + j] * imm->mu[i] / cbar[j] : 0.0;
        }
    }

    /* Mixed initial conditions for each filter */
    for (uint8_t j = 0; j < N; j++) {
        /* x_mixed_j = sum_i mu_{i|j} * x_i */
        memset(imm->x_mixed + j * KF_MAX_STATE_DIM, 0, n * sizeof(double));
        for (uint8_t i = 0; i < N; i++) {
            double w = mu_mix[i * N + j];
            for (uint8_t k = 0; k < n; k++)
                imm->x_mixed[j * KF_MAX_STATE_DIM + k] +=
                    w * imm->filters[i].x[k];
        }

        /* P_mixed_j = sum_i mu_{i|j} * [P_i + (x_i-x_mixed)*(...)'] */
        double *Pmj = imm->P_mixed + j * KF_MAX_STATE_DIM * KF_MAX_STATE_DIM;
        memset(Pmj, 0, (size_t)n * n * sizeof(double));
        for (uint8_t i = 0; i < N; i++) {
            double w = mu_mix[i * N + j];
            double dx[KF_MAX_STATE_DIM];
            for (uint8_t k = 0; k < n; k++)
                dx[k] = imm->filters[i].x[k]
                        - imm->x_mixed[j * KF_MAX_STATE_DIM + k];
            for (uint8_t a = 0; a < n; a++)
                for (uint8_t b = 0; b < n; b++)
                    Pmj[a * n + b] += w * (imm->filters[i].P[a * n + b]
                                           + dx[a] * dx[b]);
        }
    }

    /* Phase 2: Filtering */
    for (r = 0; r < N; r++) {
        /* Set filter to mixed initial state */
        memcpy(imm->filters[r].x, imm->x_mixed + r * KF_MAX_STATE_DIM,
               n * sizeof(double));
        memcpy(imm->filters[r].P, imm->P_mixed + r * KF_MAX_STATE_DIM * KF_MAX_STATE_DIM,
               (size_t)n * n * sizeof(double));

        /* Predict + Update */
        kf_predict(&imm->filters[r], &imm->models[r], u);
        kf_update(&imm->filters[r], &imm->models[r], z);

        /* Compute likelihood: p(z | model r) = N(y; 0, S) */
        double logL = kf_log_likelihood(&imm->filters[r]);
        imm->likelihood[r] = exp(logL);
        if (!isfinite(imm->likelihood[r]) || imm->likelihood[r] <= 0.0)
            imm->likelihood[r] = MAT_EPSILON;
    }

    /* Phase 3: Model probability update */
    double sum_lik = 0.0;
    for (r = 0; r < N; r++) sum_lik += imm->likelihood[r] * cbar[r];

    for (r = 0; r < N; r++) {
        imm->mu[r] = (sum_lik > MAT_EPSILON)
            ? imm->likelihood[r] * cbar[r] / sum_lik : 1.0 / N;
    }

    /* Phase 4: Combination */
    memset(imm->x_overall, 0, n * sizeof(double));
    memset(imm->P_overall, 0, (size_t)n * n * sizeof(double));

    for (r = 0; r < N; r++) {
        double w = imm->mu[r];
        double dx[KF_MAX_STATE_DIM];
        for (uint8_t k = 0; k < n; k++) {
            imm->x_overall[k] += w * imm->filters[r].x[k];
            dx[k] = imm->filters[r].x[k] - imm->x_overall[k];
        }
        /* Covariance will be re-computed after final x_overall */
    }

    /* Recompute overall covariance with final x_overall */
    for (r = 0; r < N; r++) {
        double w = imm->mu[r];
        double dx[KF_MAX_STATE_DIM];
        /* Recalculate dx with final x_overall */
        for (uint8_t k = 0; k < n; k++)
            dx[k] = imm->filters[r].x[k] - imm->x_overall[k];
        for (uint8_t a = 0; a < n; a++)
            for (uint8_t b = 0; b < n; b++)
                imm->P_overall[a * n + b] += w * (imm->filters[r].P[a * n + b]
                                                   + dx[a] * dx[b]);
    }
}

/* ===================================================================
 * IMM Accessors
 * =================================================================== */

const double *imm_get_state(const AKFMultiModel *imm)
{
    return imm ? imm->x_overall : NULL;
}

const double *imm_get_probabilities(const AKFMultiModel *imm)
{
    return imm ? imm->mu : NULL;
}
