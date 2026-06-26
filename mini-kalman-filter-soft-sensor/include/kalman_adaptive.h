/**
 * @file kalman_adaptive.h
 * @brief Adaptive Kalman Filtering — Noise Covariance Estimation
 *
 * L5 Algorithms: Innovation-based adaptation, multiple-model estimation
 * L8 Advanced Topics: Adaptive filtering, covariance matching
 *
 * When process noise Q and measurement noise R are unknown or time-varying,
 * adaptive Kalman filtering estimates them online from the innovation sequence.
 *
 * Reference: Mehra, R.K. (1972) "Approaches to Adaptive Filtering."
 *   IEEE Trans. Automatic Control, 17(5), 693-698.
 *
 * Course alignment: MIT 6.302, CMU 24-677, Georgia Tech AE 6530
 */
#ifndef KALMAN_ADAPTIVE_H
#define KALMAN_ADAPTIVE_H

#include "kalman_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * L1: Adaptive KF type definitions
 * -------------------------------------------------------------------------- */

/** Window size for innovation covariance estimation */
#define AKF_DEFAULT_WINDOW 20

/** Maximum number of models in multi-model estimation */
#define AKF_MAX_MODELS 8

/** Forgetting factor for exponential weighted innovation covariance */
#define AKF_DEFAULT_FORGET_FACTOR 0.95

/**
 * L1: AKFInnovationStats — innovation-based statistics for adaptation
 */
typedef struct {
    /** Running estimate of innovation covariance: E[y*y''], m x m */
    double innov_cov_est[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];

    /** Theoretical innovation covariance: S = H*P*H'' + R, m x m */
    double innov_cov_theory[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];

    /** Innovation history buffer, window_size x m, row-major */
    double innov_history[AKF_DEFAULT_WINDOW * KF_MAX_MEAS_DIM];

    /** Current window fill count */
    uint8_t window_count;

    /** Window size (N) */
    uint8_t window_size;

    /** Forgetting factor (0 < lambda <= 1) */
    double forgetting_factor;

    /** Measurement dimension */
    uint8_t m;

    /** Flags */
    uint8_t window_full : 1;
    uint8_t reserved    : 7;
} AKFInnovationStats;

/**
 * L1: AKFState — Adaptive Kalman Filter state
 */
typedef struct {
    /** Core KF state */
    KalmanFilterState kf;

    /** Innovation statistics */
    AKFInnovationStats innov_stats;

    /** Current estimated Q, n x n, row-major */
    double Q_est[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Current estimated R, m x m, row-major */
    double R_est[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];

    /** State residual history for Q estimation, window_size x n */
    double state_residual_history[AKF_DEFAULT_WINDOW * KF_MAX_STATE_DIM];

    /** Process noise estimation enabled */
    uint8_t adapt_q : 1;

    /** Measurement noise estimation enabled */
    uint8_t adapt_r : 1;

    /** Use covariance matching method (vs innovation correlation) */
    uint8_t use_cov_matching : 1;

    /** Use Bayesian estimation method */
    uint8_t use_bayesian : 1;

    /** State dimension */
    uint8_t n;

    /** Measurement dimension */
    uint8_t m;

    uint8_t reserved : 3;
} AKFState;

/**
 * L1: AKFMultiModel — Interacting Multiple Model (IMM) estimation
 *
 * IMM runs multiple Kalman filters in parallel, each with a different
 * model hypothesis, and blends their estimates based on model probabilities.
 *
 * Applications: maneuvering target tracking (constant velocity vs turn),
 *   fault detection (normal vs faulty model), regime-switching processes.
 *
 * Reference: Blom & Bar-Shalom (1988) "The Interacting Multiple Model Algorithm..."
 */
typedef struct {
    /** Array of Kalman filter states, one per model */
    KalmanFilterState filters[AKF_MAX_MODELS];

    /** Array of Kalman models */
    KalmanModel models[AKF_MAX_MODELS];

    /** Model probabilities mu[i] = P(model i | measurements) */
    double mu[AKF_MAX_MODELS];

    /** Model transition probability matrix Pi[i][j] = P(model j at k | model i at k-1) */
    double Pi[AKF_MAX_MODELS * AKF_MAX_MODELS];

    /** Mixed initial states for each model, AKF_MAX_MODELS x n */
    double x_mixed[AKF_MAX_MODELS * KF_MAX_STATE_DIM];

    /** Mixed initial covariances, AKF_MAX_MODELS x (n x n) */
    double P_mixed[AKF_MAX_MODELS * KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Blended (overall) state estimate, size n */
    double x_overall[KF_MAX_STATE_DIM];

    /** Blended covariance estimate, n x n */
    double P_overall[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Number of active models */
    uint8_t num_models;

    /** State dimension */
    uint8_t n;

    /** Measurement dimension */
    uint8_t m;

    /** Likelihood of each model for latest measurement */
    double likelihood[AKF_MAX_MODELS];

    uint8_t initialized : 1;
    uint8_t reserved    : 7;
} AKFMultiModel;

/* --------------------------------------------------------------------------
 * L5: Adaptive KF API
 * -------------------------------------------------------------------------- */

/**
 * Initialize adaptive Kalman filter.
 *
 * @param akf        Adaptive KF state
 * @param x0         Initial state, size n
 * @param P0         Initial covariance, n x n, row-major
 * @param Q0         Initial process noise estimate
 * @param R0         Initial measurement noise estimate
 * @param n          State dimension
 * @param m          Measurement dimension
 * @param adapt_q    Enable Q adaptation
 * @param adapt_r    Enable R adaptation
 * @param window_sz  Innovation history window size
 *
 * Complexity: O(n^2 + m^2)
 */
void akf_init(AKFState *akf, const double *x0, const double *P0,
              const double *Q0, const double *R0,
              uint8_t n, uint8_t m,
              uint8_t adapt_q, uint8_t adapt_r,
              uint8_t window_sz);

/**
 * Adaptive prediction step with current Q estimate.
 *
 * @param akf   Adaptive KF state
 * @param model System model (H, F used; Q comes from akf->Q_est)
 * @param u     Control input (NULL if none)
 */
void akf_predict(AKFState *akf, const KalmanModel *model, const double *u);

/**
 * Adaptive update step with current R estimate.
 *
 * @param akf   Adaptive KF state
 * @param model System model
 * @param z     Measurement vector
 */
void akf_update(AKFState *akf, const KalmanModel *model, const double *z);

/**
 * Single adaptive predict+update step.
 */
const double *akf_step(AKFState *akf, const KalmanModel *model,
                       const double *z, const double *u);

/**
 * Covariance-matching based Q estimation.
 *
 * Estimates Q from the discrepancy between actual innovation covariance
 * and the theoretical innovation covariance:
 *
 *   Q_est = K * E[y*y'] * K'
 *
 * Uses a sliding window of innovation vectors.
 *
 * Complexity: O(n^3 + window * m^3)
 * Reference: Myers & Tapley (1976) "Adaptive Sequential Estimation..."
 */
void akf_estimate_Q_covariance_matching(AKFState *akf, const KalmanModel *model);

/**
 * Innovation-correlation based R estimation.
 *
 * Estimates R from the innovation autocorrelation:
 *
 *   R_est = E[y*y'] - H*P*H'
 *
 * Complexity: O(n^3 + m^3)
 * Reference: Mehra (1970) "On the Identification of Variances..."
 */
void akf_estimate_R_innovation(AKFState *akf, const KalmanModel *model);

/**
 * Bayesian maximum a posteriori (MAP) estimation of Q and R.
 *
 * Uses prior distributions on Q and R and updates them based on
 * the likelihood of observed innovations.
 *
 * Complexity: O(n^3 + m^3)
 * Reference: Sarkka & Nummenmaa (2009) "Recursive Noise Adaptive Kalman..."
 */
void akf_estimate_bayesian(AKFState *akf, const KalmanModel *model);

/**
 * Update the innovation statistics buffer.
 *
 * @param akf  Adaptive KF state (after update step)
 *
 * Complexity: O(m^2)
 */
void akf_update_innovation_stats(AKFState *akf);

/**
 * Initialize IMM estimator.
 *
 * @param imm        IMM state
 * @param models     Array of Kalman models (one per hypothesis)
 * @param x0         Initial state for all models
 * @param P0         Initial covariance for all models
 * @param mu0        Initial model probabilities (NULL = uniform)
 * @param Pi         Model transition matrix (NULL = identity-like)
 * @param num_models Number of models
 * @param n          State dimension
 * @param m          Measurement dimension
 *
 * Complexity: O(num_models * n^2)
 */
void imm_init(AKFMultiModel *imm,
              const KalmanModel *models,
              const double *x0, const double *P0,
              const double *mu0, const double *Pi,
              uint8_t num_models, uint8_t n, uint8_t m);

/**
 * IMM step: interaction, filtering, model probability update, combination.
 *
 * 1. Interaction (mixing): compute mixed initial conditions for each filter
 * 2. Filtering: run each KF predict+update
 * 3. Model probability update: compute likelihoods, update mu
 * 4. Combination: blend estimates by model probabilities
 *
 * @param imm  IMM state
 * @param z    Measurement vector
 * @param u    Control input (NULL if none)
 *
 * Complexity: O(num_models * (n^3 + n^2*m + n*m^2 + m^3))
 * Reference: Blom & Bar-Shalom (1988)
 */
void imm_step(AKFMultiModel *imm, const double *z, const double *u);

/**
 * Get the overall (blended) state estimate from IMM.
 *
 * @param imm  IMM state
 * @return Pointer to x_overall, size n
 */
const double *imm_get_state(const AKFMultiModel *imm);

/**
 * Get model probabilities from IMM.
 *
 * @param imm  IMM state
 * @return Pointer to mu array, size num_models
 */
const double *imm_get_probabilities(const AKFMultiModel *imm);

#ifdef __cplusplus
}
#endif
#endif /* KALMAN_ADAPTIVE_H */
