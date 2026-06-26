/**
 * @file kalman_quality.h
 * @brief Kalman filter variants for quality estimation — linear KF, EKF, UKF, and error-state KF.
 *
 * Level: L5 Algorithms — Kalman filtering for inferential quality estimation.
 * Reference:
 *   Kalman, R.E. (1960) "A New Approach to Linear Filtering and Prediction Problems" — JBE, 82(1)
 *   Simon, D. (2006) "Optimal State Estimation" — Chapters 5, 13, 14
 *   Julier, Uhlmann (2004) "Unscented filtering and nonlinear estimation" — Proc IEEE, 92(3)
 *   Maybeck, P.S. (1979) "Stochastic Models, Estimation, and Control" — Vol. 1
 *
 * Course mapping:
 *   MIT 6.302/2.171: Kalman filtering, state estimation for control
 *   Stanford AA272: GNSS/position estimation → quality state estimation analogy
 *   CMU 24-677: Advanced state estimation, EKF/UKF
 *   Berkeley ME233: Stochastic estimation, Kalman theory
 */

#ifndef KALMAN_QUALITY_H
#define KALMAN_QUALITY_H

#include "quality_estimator_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
 * L5: Linear Kalman Filter for Quality Estimation
 *===========================================================================*/

/**
 * @brief Linear Kalman Filter (KF) state for quality estimation.
 *
 * The Kalman filter is the optimal recursive state estimator for linear
 * systems with Gaussian noise. For quality estimation, the "state" may
 * include both the quality variable itself and the bias term:
 *
 *   x = [quality; bias]^T
 *
 * This allows simultaneous quality prediction and bias estimation.
 *
 * Equations:
 *   Predict:  x_prior(k) = A * x(k-1) + B * u(k-1)
 *             P_prior(k) = A * P(k-1) * A^T + Q
 *   Update:   K(k) = P_prior * C^T * (C * P_prior * C^T + R)^{-1}
 *             x(k) = x_prior + K(k) * (y_meas - C * x_prior)
 *             P(k) = (I - K(k)*C) * P_prior
 *
 * Complexity: O(n^3) for the update step (matrix inversion), O(n^2) for predict.
 *             For quality estimation, n_states is typically 1-4, making this
 *             easily computable in real-time (microseconds).
 */
typedef struct {
    int     n_states;        /**< Number of state variables */
    int     n_inputs;        /**< Number of input variables (process variables) */
    int     n_measurements;  /**< Number of measurements (quality + possible analyzers) */

    double *A;               /**< State transition matrix [n_states × n_states] */
    double *B;               /**< Input matrix [n_states × n_inputs] */
    double *C;               /**< Measurement matrix [n_measurements × n_states] */
    double *Q;               /**< Process noise covariance [n_states × n_states] */
    double *R;               /**< Measurement noise covariance [n_measurements × n_measurements] */

    double *x;               /**< Current state estimate [n_states] */
    double *P;               /**< State error covariance [n_states × n_states] */

    /* Scratch space for computations (avoid repeated malloc) */
    double *x_prior;         /**< Predicted state [n_states] */
    double *P_prior;         /**< Predicted covariance [n_states × n_states] */
    double *K;               /**< Kalman gain [n_states × n_measurements] */
    double *S;               /**< Innovation covariance [n_measurements × n_measurements] */
    double *S_inv;           /**< Inverse of S [n_measurements × n_measurements] */
    double *temp_nn;         /**< General scratch [n_states × n_states] */
    double *temp_nm;         /**< Scratch [n_states × n_measurements] */
    double *temp_mn;         /**< Scratch [n_measurements × n_states] */
    double *temp_mm;         /**< Scratch [n_measurements × n_measurements] */
    double *temp_n;          /**< Scratch vector [n_states] */
    double *temp_m;          /**< Scratch vector [n_measurements] */

    double *I;               /**< Identity matrix [n_states × n_states] for Joseph form update */

    int     is_initialized;  /**< Flag: filter has been properly initialized */
} kalman_filter_t;

/**
 * @brief Allocate and initialize a linear Kalman filter.
 *
 * Sets x=0, P=large*I (high initial uncertainty), K=0.
 * Matrices A, B, C, Q, R must be set separately after allocation.
 *
 * @param kf             Uninitialized Kalman filter struct
 * @param n_states       Number of state variables
 * @param n_inputs       Number of input variables
 * @param n_measurements Number of measurement variables
 *
 * Complexity: O(n^2 + n*m) for allocation and initialization.
 */
void kf_alloc(kalman_filter_t *kf, int n_states, int n_inputs, int n_measurements);

/**
 * @brief Free all memory allocated for a Kalman filter.
 *
 * @param kf  Kalman filter to free
 */
void kf_free(kalman_filter_t *kf);

/**
 * @brief Set the state-space matrices for the Kalman filter.
 *
 * @param kf  Kalman filter
 * @param A   State transition matrix [n_states × n_states]
 * @param B   Input matrix [n_states × n_inputs]
 * @param C   Measurement matrix [n_measurements × n_states]
 */
void kf_set_matrices(kalman_filter_t *kf, const double *A, const double *B, const double *C);

/**
 * @brief Set the noise covariance matrices.
 *
 * @param kf  Kalman filter
 * @param Q   Process noise covariance [n_states × n_states] (must be positive semi-definite)
 * @param R   Measurement noise covariance [n_measurements × n_measurements] (must be positive definite)
 *
 * Theorem: If Q ≥ 0 and R > 0, and (A, C) is detectable, then the Kalman filter
 * is exponentially stable and the error covariance P converges to a steady-state value.
 */
void kf_set_noise(kalman_filter_t *kf, const double *Q, const double *R);

/**
 * @brief Set the initial state and covariance.
 *
 * @param kf  Kalman filter
 * @param x0  Initial state estimate [n_states]
 * @param P0  Initial error covariance [n_states × n_states]
 */
void kf_set_initial(kalman_filter_t *kf, const double *x0, const double *P0);

/**
 * @brief Kalman filter predict step (time update).
 *
 * x_prior = A * x + B * u
 * P_prior = A * P * A^T + Q
 *
 * @param kf  Kalman filter (updated in-place)
 * @param u   Input (control) vector [n_inputs] — may be NULL if n_inputs=0
 */
void kf_predict(kalman_filter_t *kf, const double *u);

/**
 * @brief Kalman filter update step (measurement update).
 *
 * When a lab measurement y becomes available:
 *   S = C * P_prior * C^T + R
 *   K = P_prior * C^T * S^{-1}
 *   x = x_prior + K * (y - C * x_prior)
 *   P = (I - K*C) * P_prior           (standard form)
 *
 * Uses the numerically stable Joseph form:
 *   P = (I - K*C) * P_prior * (I - K*C)^T + K * R * K^T
 *
 * @param kf  Kalman filter (updated in-place)
 * @param y   Measurement vector [n_measurements]
 */
void kf_update(kalman_filter_t *kf, const double *y);

/**
 * @brief Combined predict + update: run one complete Kalman step.
 *
 * @param kf  Kalman filter (updated in-place)
 * @param u   Input vector [n_inputs] (may be NULL if n_inputs=0)
 * @param y   Measurement vector [n_measurements] (may be NULL to only predict)
 * @param x_out [out] Updated state estimate [n_states] (may be NULL)
 */
void kf_step(kalman_filter_t *kf, const double *u, const double *y, double *x_out);

/**
 * @brief Get the current quality estimate from the Kalman filter state.
 *
 * The quality estimate is typically the first element of the state vector,
 * or a linear combination: quality = C_quality * x.
 *
 * @param kf  Kalman filter
 * @return    Quality estimate value
 */
double kf_get_quality(const kalman_filter_t *kf);

/**
 * @brief Get the quality estimate variance from the error covariance.
 *
 * var(quality) = C_quality * P * C_quality^T
 *
 * @param kf  Kalman filter
 * @return    Quality estimate variance
 */
double kf_get_quality_variance(const kalman_filter_t *kf);

/**
 * @brief Compute the innovation (measurement residual) and its normalized form.
 *
 * innovation = y - C * x_prior
 * normalized_innovation = S^{-1/2} * innovation
 *
 * A normalized innovation > 3σ suggests a bad measurement (outlier).
 *
 * @param kf       Kalman filter (after predict, before update)
 * @param y        Measurement vector [n_measurements]
 * @param innov    [out] Innovation vector [n_measurements]
 * @param n_innov  [out] Normalized innovation norm (scalar)
 */
void kf_innovation(const kalman_filter_t *kf, const double *y,
                   double *innov, double *n_innov);

/*===========================================================================
 * L5: Extended Kalman Filter (EKF) for Nonlinear Quality Models
 *===========================================================================*/

/**
 * @brief Extended Kalman Filter (EKF) state for nonlinear quality estimation.
 *
 * The EKF handles nonlinear process and/or measurement models by linearizing
 * about the current state estimate:
 *
 *   x(k+1) = f(x(k), u(k)) + w(k)
 *   y(k)   = h(x(k), u(k)) + v(k)
 *
 * The Jacobians F_k = ∂f/∂x|_x and H_k = ∂h/∂x|_x are computed at each step.
 *
 * This is essential when the quality relationship is nonlinear, such as:
 *   - Antoine equation for vapor-liquid equilibrium (exponential in T)
 *   - Arrhenius reaction kinetics (exponential in 1/T)
 *   - pH from strong acid/base (logarithmic)
 */
typedef struct {
    int     n_states;        /**< Number of state variables */
    int     n_inputs;        /**< Number of input variables */
    int     n_measurements;  /**< Number of measurements */

    /* Nonlinear model functions */
    /**
     * @brief State transition function: x_next = f(x, u, dt).
     * @param x      Current state [n_states]
     * @param u      Inputs [n_inputs]
     * @param dt     Time step
     * @param x_next [out] Next state [n_states]
     */
    void (*f)(const double *x, const double *u, double dt, double *x_next);

    /**
     * @brief Measurement function: y = h(x, u).
     * @param x  State [n_states]
     * @param u  Inputs [n_inputs]
     * @param y  [out] Predicted measurement [n_measurements]
     */
    void (*h)(const double *x, const double *u, double *y);

    /**
     * @brief State Jacobian: F = ∂f/∂x evaluated at (x, u).
     * @param x  Current state [n_states]
     * @param u  Inputs [n_inputs]
     * @param dt Time step
     * @param F  [out] Jacobian [n_states × n_states]
     */
    void (*compute_F)(const double *x, const double *u, double dt, double *F);

    /**
     * @brief Measurement Jacobian: H = ∂h/∂x evaluated at (x, u).
     * @param x  Current state [n_states]
     * @param u  Inputs [n_inputs]
     * @param H  [out] Jacobian [n_measurements × n_states]
     */
    void (*compute_H)(const double *x, const double *u, double *H);

    /* Internal linear KF */
    kalman_filter_t kf;  /**< Underlying linear Kalman filter (A,B,C updated by Jacobians) */

    /* Nonlinear model context */
    double  dt;          /**< Current time step */
    double *x_pred;      /**< Predicted state from f() [n_states] */
    double *y_pred;      /**< Predicted measurement from h() [n_measurements] */
    double *F_jac;       /**< State Jacobian [n_states × n_states] */
    double *H_jac;       /**< Measurement Jacobian [n_measurements × n_states] */
} ekf_filter_t;

/**
 * @brief Allocate and initialize an Extended Kalman Filter.
 *
 * @param ekf            Uninitialized EKF struct
 * @param n_states       State dimension
 * @param n_inputs       Input dimension
 * @param n_measurements Measurement dimension
 * @param f              State transition function
 * @param h              Measurement function
 * @param compute_F      State Jacobian function
 * @param compute_H      Measurement Jacobian function
 */
void ekf_alloc(ekf_filter_t *ekf, int n_states, int n_inputs, int n_measurements,
               void (*f)(const double*, const double*, double, double*),
               void (*h)(const double*, const double*, double*),
               void (*compute_F)(const double*, const double*, double, double*),
               void (*compute_H)(const double*, const double*, double*));

/**
 * @brief Free EKF memory.
 *
 * @param ekf  EKF to free
 */
void ekf_free(ekf_filter_t *ekf);

/**
 * @brief Set EKF noise and initial conditions (delegates to kf_set_*).
 *
 * @param ekf  EKF
 * @param Q    Process noise covariance [n_states × n_states]
 * @param R    Measurement noise covariance [n_measurements × n_measurements]
 * @param x0   Initial state [n_states]
 * @param P0   Initial covariance [n_states × n_states]
 */
void ekf_setup(ekf_filter_t *ekf, const double *Q, const double *R,
               const double *x0, const double *P0);

/**
 * @brief Run one complete EKF step: predict using f() + update using h().
 *
 * Steps:
 *   1. Predict state: x_prior = f(x, u, dt)
 *   2. Compute F = ∂f/∂x (linearize at current estimate)
 *   3. Predict covariance: P_prior = F*P*F^T + Q
 *   4. If measurement available:
 *      a. Predict measurement: y_pred = h(x_prior, u)
 *      b. Compute H = ∂h/∂x (linearize at predicted state)
 *      c. Compute Kalman gain and update
 *
 * @param ekf  EKF (updated in-place)
 * @param u    Input vector [n_inputs] (may be NULL)
 * @param y    Measurement vector [n_measurements] (may be NULL for predict-only)
 * @param dt   Time step
 */
void ekf_step(ekf_filter_t *ekf, const double *u, const double *y, double dt);

/**
 * @brief Get quality estimate from EKF (delegates to underlying KF).
 *
 * @param ekf  EKF
 * @return     Quality estimate
 */
double ekf_get_quality(const ekf_filter_t *ekf);

/**
 * @brief Get quality estimate variance from EKF.
 *
 * @param ekf  EKF
 * @return     Quality estimate variance
 */
double ekf_get_quality_variance(const ekf_filter_t *ekf);

/*===========================================================================
 * L5: Process Noise Adaptation — Adaptive Kalman Filter
 *===========================================================================*/

/**
 * @brief Adaptive Kalman filter that estimates process noise Q online.
 *
 * Uses the innovation-based covariance matching method (Mehra, 1972):
 *   Q_adaptive = K * S * K^T
 * where S is the innovation covariance estimated from a moving window.
 *
 * Reference: Mehra, R.K. (1972) "Approaches to adaptive filtering"
 *            IEEE Trans Automatic Control, 17(5), 693-698.
 */
typedef struct {
    kalman_filter_t kf;          /**< Underlying Kalman filter */
    int     window_size;         /**< Innovation window for Q estimation */
    int     window_index;        /**< Current position in innovation buffer */
    double *innovation_buffer;   /**< Stored innovations [window_size * n_measurements] */
    int     buffer_filled;       /**< Flag: buffer fully populated */
    double *Q_adaptive;          /**< Current adaptive Q estimate [n_states × n_states] */
    double *Q_initial;           /**< Save initial Q for reference */
} adaptive_kf_t;

/**
 * @brief Allocate and initialize an adaptive Kalman filter.
 *
 * @param akf           Uninitialized adaptive KF struct
 * @param n_states      State dimension
 * @param n_inputs      Input dimension
 * @param n_measurements Measurement dimension
 * @param window        Innovation window size for Q adaptation
 */
void akf_alloc(adaptive_kf_t *akf, int n_states, int n_inputs, int n_measurements, int window);

/**
 * @brief Free adaptive KF memory.
 *
 * @param akf  Adaptive KF to free
 */
void akf_free(adaptive_kf_t *akf);

/**
 * @brief Set adaptive KF parameters (delegates to kf_set_*).
 *
 * @param akf  Adaptive KF
 * @param A    State transition matrix
 * @param B    Input matrix
 * @param C    Measurement matrix
 * @param Q0   Initial process noise (will be adapted online)
 * @param R    Measurement noise (assumed known, not adapted)
 * @param x0   Initial state
 * @param P0   Initial covariance
 */
void akf_setup(adaptive_kf_t *akf, const double *A, const double *B, const double *C,
               const double *Q0, const double *R, const double *x0, const double *P0);

/**
 * @brief Run one adaptive Kalman step with Q adaptation.
 *
 * @param akf  Adaptive KF (updated in-place)
 * @param u    Input vector [n_inputs]
 * @param y    Measurement vector [n_measurements]
 */
void akf_step(adaptive_kf_t *akf, const double *u, const double *y);

/**
 * @brief Get the current adapted process noise estimate.
 *
 * @param akf   Adaptive KF
 * @param Q_out [out] Current Q estimate [n_states × n_states]
 */
void akf_get_Q(const adaptive_kf_t *akf, double *Q_out);

#ifdef __cplusplus
}
#endif

#endif /* KALMAN_QUALITY_H */
