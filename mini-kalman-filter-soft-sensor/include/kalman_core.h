/**
 * @file kalman_core.h
 * @brief Standard Linear Kalman Filter — Core Definitions and API
 *
 * L1 Definitions: State-space model, Gaussian noise, Kalman gain
 * L2 Core Concepts: Prediction-correction cycle, MMSE optimality
 * L4 Engineering Laws: Kalman 5 equations, Joseph form, Riccati convergence
 *
 * Reference: Kalman, R.E. (1960). "A New Approach to Linear Filtering
 *   and Prediction Problems." Trans. ASME J. Basic Eng., 82, 35-45.
 *
 * Course alignment: MIT 6.302, Stanford ENGR205, Berkeley ME233,
 *   CMU 24-677, Georgia Tech AE 6530
 */
#ifndef KALMAN_CORE_H
#define KALMAN_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * L1: Core type definitions — Kalman filter state, measurement, and model
 * -------------------------------------------------------------------------- */

/** Maximum supported state dimension (compile-time constant) */
#define KF_MAX_STATE_DIM 12

/** Maximum supported measurement dimension */
#define KF_MAX_MEAS_DIM  8

/** Convergence tolerance for checking filter steady-state */
#define KF_CONVERGENCE_TOL 1e-8

/**
 * L1: KalmanFilterState — the core filter state
 *
 * The Kalman filter maintains:
 *   x[k|k]  — posterior state estimate (after incorporating measurement z[k])
 *   P[k|k]  — posterior error covariance matrix
 *   x[k|k-1]— prior state prediction
 *   P[k|k-1]— prior error covariance
 *
 * Gaussian assumption: x ~ N(x_hat, P), v ~ N(0, R), w ~ N(0, Q)
 */
typedef struct {
    /** Current posterior state estimate x[k|k], size n x 1 */
    double x[KF_MAX_STATE_DIM];

    /** Posterior error covariance P[k|k], size n x n, stored row-major */
    double P[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Prior state prediction x[k|k-1], size n x 1 */
    double x_prior[KF_MAX_STATE_DIM];

    /** Prior error covariance P[k|k-1], size n x n, stored row-major */
    double P_prior[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Kalman gain K[k], size n x m, stored row-major */
    double K[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM];

    /** Innovation (measurement residual) y[k] = z[k] - H*x[k|k-1], size m x 1 */
    double innovation[KF_MAX_MEAS_DIM];

    /** Innovation covariance S[k] = H*P*H'' + R, size m x m */
    double S[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];

    /** State dimension n */
    uint8_t n;

    /** Measurement dimension m */
    uint8_t m;

    /** Update counter for convergence detection */
    uint32_t step_count;

    /** Previous Kalman gain for convergence check, stored n x m row-major */
    double K_prev[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM];

    /** Filter status flags */
    uint8_t initialized : 1;
    uint8_t converged   : 1;
    uint8_t singular    : 1;
    uint8_t diverged    : 1;
    uint8_t reserved    : 4;
} KalmanFilterState;

/**
 * L1: KalmanModel — linear state-space model
 *
 * System:  x[k] = F * x[k-1] + B * u[k-1] + w[k-1],  w ~ N(0, Q)
 * Measure: z[k] = H * x[k] + v[k],                     v ~ N(0, R)
 *
 * F: state transition matrix (n x n)
 * B: control input matrix (n x p)
 * H: measurement/observation matrix (m x n)
 * Q: process noise covariance (n x n) — must be positive semi-definite
 * R: measurement noise covariance (m x m) — must be positive definite
 */
typedef struct {
    /** State transition matrix F, n x n, row-major */
    double F[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Control input matrix B, n x p, row-major */
    double B[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Observation matrix H, m x n, row-major */
    double H[KF_MAX_MEAS_DIM * KF_MAX_STATE_DIM];

    /** Process noise covariance Q, n x n, row-major */
    double Q[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Measurement noise covariance R, m x m, row-major */
    double R[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];

    /** Control input dimension p */
    uint8_t p;

    /** State dimension n */
    uint8_t n;

    /** Measurement dimension m */
    uint8_t m;

    /** Flags: is time-varying? is model linear? */
    uint8_t time_varying : 1;
    uint8_t is_linear    : 1;
    uint8_t reserved     : 6;
} KalmanModel;

/**
 * L1: KalmanDiagnostics — filter health and performance metrics
 *
 * Monitors:
 *   NEES (Normalized Estimation Error Squared)
 *   NIS  (Normalized Innovation Squared)
 *   Log-likelihood of the filter
 */
typedef struct {
    /** Normalized innovation squared (NIS), chi-square distributed */
    double nis;

    /** Normalized estimation error squared (NEES) */
    double nees;

    /** Running log-likelihood */
    double log_likelihood;

    /** Mahalanobis distance of latest innovation */
    double mahalanobis_dist;

    /** Condition number of innovation covariance S */
    double S_cond;

    /** Condition number of error covariance P */
    double P_cond;

    /** Trace of P (total uncertainty) */
    double P_trace;

    /** Trace of Q / Trace of R ratio — signal-to-noise indicator */
    double snr_ratio;

    /** Innovation whiteness test: autocorrelation at lag 1 */
    double innov_autocorr;

    /** Previously stored innovation for autocorrelation, size m */
    double innov_prev[KF_MAX_MEAS_DIM];

    /** Number of NIS exceedances (95% confidence) */
    uint32_t nis_exceedances;

    /** Total measurement count */
    uint32_t total_measurements;

    /** Fault flags */
    uint8_t nis_fault     : 1;
    uint8_t cov_fault     : 1;
    uint8_t divergence    : 1;
    uint8_t residual_fault: 1;
    uint8_t reserved      : 4;
} KalmanDiagnostics;

/* --------------------------------------------------------------------------
 * L2-L5: Core API — standard Kalman filter operations
 * -------------------------------------------------------------------------- */

/**
 * Initialize the Kalman filter state and model.
 *
 * @param kf      Pointer to filter state (must be non-NULL)
 * @param model   Pointer to model definition (must be non-NULL)
 * @param x0      Initial state estimate, size n
 * @param P0      Initial covariance, size n x n, row-major
 * @param n       State dimension
 * @param m       Measurement dimension
 *
 * Complexity: O(n^2) for covariance copy
 * Source: Kalman (1960) equations (1)-(5)
 */
void kf_init(KalmanFilterState *kf, KalmanModel *model,
             const double *x0, const double *P0,
             uint8_t n, uint8_t m);

/**
 * Prediction step: project state and covariance forward in time.
 *
 * x[k|k-1] = F * x[k-1|k-1] + B * u[k-1]
 * P[k|k-1] = F * P[k-1|k-1] * F'' + Q
 *
 * @param kf     Filter state
 * @param model  System model
 * @param u      Control input vector (NULL if no control), size p
 *
 * Complexity: O(n^3) for matrix multiply (n x n) x (n x n)
 * Source: Kalman (1960) eq. (3)-(4)
 */
void kf_predict(KalmanFilterState *kf, const KalmanModel *model,
                const double *u);

/**
 * Update step: incorporate measurement to refine state estimate.
 *
 * y[k]    = z[k] - H * x[k|k-1]              (innovation)
 * S[k]    = H * P[k|k-1] * H'' + R            (innovation covariance)
 * K[k]    = P[k|k-1] * H'' * inv(S[k])        (Kalman gain)
 * x[k|k]  = x[k|k-1] + K[k] * y[k]           (state update)
 * P[k|k]  = (I - K[k] * H) * P[k|k-1]        (covariance update)
 *
 * @param kf     Filter state
 * @param model  System model
 * @param z      Measurement vector, size m
 *
 * Complexity: O(n^3 + n^2*m + n*m^2 + m^3) for matrix operations
 * Source: Kalman (1960) eq. (5)-(9)
 */
void kf_update(KalmanFilterState *kf, const KalmanModel *model,
               const double *z);

/**
 * Single combined predict+update step.
 *
 * @return Pointer to posterior state estimate x[k|k]
 */
const double *kf_step(KalmanFilterState *kf, const KalmanModel *model,
                      const double *z, const double *u);

/**
 * Joseph-form covariance update — numerically stabler than standard form.
 *
 * P[k|k] = (I - K*H) * P[k|k-1] * (I - K*H)'' + K * R * K''
 *
 * Guarantees symmetry and positive semi-definiteness of P
 * even under numerical round-off errors.
 *
 * Complexity: O(n^3)
 * Reference: Bucy & Joseph (1968) "Filtering for Stochastic Processes"
 */
void kf_joseph_update(KalmanFilterState *kf, const KalmanModel *model);

/**
 * Sequential (scalar) measurement update — processes measurements one at a time.
 *
 * Avoids matrix inversion when R is diagonal (uncorrelated measurements).
 * Uses scalar division instead of matrix inversion.
 *
 * Complexity: O(m * n^2) vs O(n^3 + m^3) for batch update
 * Reference: Bierman (1977) "Factorization Methods for Discrete Estimation"
 */
void kf_sequential_update(KalmanFilterState *kf, const KalmanModel *model,
                           const double *z);

/**
 * Compute the steady-state Kalman gain by solving the discrete algebraic
 * Riccati equation (DARE) using iterative method.
 *
 * @param kf       Filter state (receives converged gain and covariance)
 * @param model    System model
 * @param max_iter Maximum iterations (0 = default 1000)
 * @return Number of iterations to converge, -1 if did not converge
 *
 * Complexity: O(n^3 * iterations)
 * Reference: Laub (1979) "A Schur Method for Solving Algebraic Riccati Equations"
 */
int kf_solve_dare(KalmanFilterState *kf, const KalmanModel *model,
                  uint32_t max_iter);

/**
 * Apply fading memory factor lambda (0.95 < lambda <= 1.0) to inflate prior covariance.
 *
 * P[k|k-1] = (1/lambda) * F * P[k-1|k-1] * F'' + Q
 *
 * This gives more weight to recent measurements, useful for tracking
 * slowly time-varying systems.
 *
 * Complexity: O(n^2)
 * Reference: Sorenson & Sacks (1971) "Recursive Fading Memory Filtering"
 */
void kf_fading_memory_predict(KalmanFilterState *kf, const KalmanModel *model,
                               const double *u, double lambda);

/**
 * Check if filter has converged to steady state.
 *
 * Convergence criterion: ||K[k] - K[k-1]||_F < tol
 *
 * @param kf   Filter state
 * @param tol  Convergence tolerance
 * @return 1 if converged, 0 otherwise
 */
int kf_has_converged(const KalmanFilterState *kf, double tol);

/**
 * Get pointer to the current posterior state estimate.
 *
 * @return Pointer to x[k|k] array of length n
 */
const double *kf_get_state(const KalmanFilterState *kf);

/**
 * Get pointer to the current posterior covariance matrix.
 *
 * @return Pointer to P[k|k] matrix, n x n row-major
 */
const double *kf_get_covariance(const KalmanFilterState *kf);

/**
 * Get pointer to the Kalman gain matrix.
 *
 * @return Pointer to K[k] matrix, n x m row-major
 */
const double *kf_get_gain(const KalmanFilterState *kf);

/**
 * Get the innovation vector from the latest update.
 *
 * @return Pointer to y[k] array of length m
 */
const double *kf_get_innovation(const KalmanFilterState *kf);

/**
 * Extract a scalar state component.
 *
 * @param kf    Filter state
 * @param idx   State index (0-based)
 * @return x[idx], or 0.0 if index out of bounds
 */
double kf_get_state_element(const KalmanFilterState *kf, uint8_t idx);

/**
 * Compute the log-likelihood of measurement z under the current filter:
 *
 * log p(z[k] | z^{k-1}) = -0.5 * [m*log(2*pi) + log|S| + y'' * inv(S) * y]
 *
 * Used for model selection, noise parameter tuning, and fault detection.
 *
 * Complexity: O(m^3) for determinant and inverse of S
 * Reference: Schweppe (1965) "Evaluation of Likelihood Functions for Gaussian Signals"
 */
double kf_log_likelihood(const KalmanFilterState *kf);

/**
 * Smooth the state estimates backward using the Rauch-Tung-Striebel (RTS)
 * fixed-interval smoother.
 *
 * Requires: stored forward-pass results x[k|k], P[k|k], x[k|k-1], P[k|k-1]
 * Produces:  smoothed estimates x[k|N], P[k|N] for k = N-1, ..., 0
 *
 * G[k] = P[k|k] * F'' * inv(P[k+1|k])
 * x[k|N] = x[k|k] + G[k] * (x[k+1|N] - x[k+1|k])
 * P[k|N] = P[k|k] + G[k] * (P[k+1|N] - P[k+1|k]) * G[k]''
 *
 * @param kf_history    Array of N filter states (forward pass)
 * @param model         System model
 * @param N             Number of time steps
 * @param x_smoothed    Output: smoothed states, N x n, row-major
 * @param P_smoothed    Output: smoothed covariances, N x (n x n), row-major
 *
 * Complexity: O(N * n^3)
 * Reference: Rauch, Tung, Striebel (1965) "Maximum Likelihood Estimates of
 *   Linear Dynamic Systems" AIAA Journal, 3(8), 1445-1450.
 */
void kf_rts_smooth(const KalmanFilterState *kf_history,
                   const KalmanModel *model,
                   uint32_t N,
                   double *x_smoothed,
                   double *P_smoothed);

/**
 * Update filter diagnostics after each measurement.
 *
 * Computes NIS, NEES, condition numbers, innovation autocorrelation.
 *
 * @param kf    Filter state
 * @param diag  Diagnostics struct to populate
 *
 * Complexity: O(n^3 + m^3)
 */
void kf_diagnostics(const KalmanFilterState *kf, KalmanDiagnostics *diag);

/**
 * Reset the filter state to a new initial estimate.
 *
 * Useful for re-initialization after detected divergence.
 */
void kf_reset(KalmanFilterState *kf, const double *x0, const double *P0);

/**
 * Check filter internal consistency:
 * - Covariance symmetric?
 * - Covariance positive semi-definite? (all eigenvalues >= 0)
 * - State values finite?
 *
 * @return 1 if consistent, 0 otherwise
 */
int kf_is_consistent(const KalmanFilterState *kf);

#ifdef __cplusplus
}
#endif
#endif /* KALMAN_CORE_H */
