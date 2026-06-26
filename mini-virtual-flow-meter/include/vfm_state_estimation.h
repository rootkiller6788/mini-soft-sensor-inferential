/**
 * @file vfm_state_estimation.h
 * @brief State estimation algorithms for Virtual Flow Meter
 *
 * Knowledge Coverage:
 *   L1 Definitions: State-space model, Kalman filter, RLS, MHE
 *   L2 Core Concepts: Optimal estimation, sensor fusion, bias tracking
 *   L3 Engineering Structures: Recursive estimator with configurable process/measurement noise
 *   L5 Algorithms: Linear Kalman filter, recursive least squares (RLS),
 *       moving horizon estimation (MHE), adaptive noise estimation
 *
 * The VFM state estimator fuses multiple sensor readings (pressure,
 * temperature, valve position, pump speed) to produce an optimal flow
 * estimate with quantified uncertainty. It also tracks systematic bias
 * and sensor health degradation over time.
 *
 * References:
 *   Kalman, R.E. (1960) "A New Approach to Linear Filtering and
 *     Prediction Problems" — ASME Journal of Basic Engineering
 *   Ljung, L. (1999) "System Identification: Theory for the User"
 *   Rawlings, J.B. (2017) "Model Predictive Control" Ch.4 — MHE
 *
 * @module mini-virtual-flow-meter
 */

#ifndef VFM_STATE_ESTIMATION_H
#define VFM_STATE_ESTIMATION_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * L1: Core Definitions -- Estimator State Structures
 * ========================================================================== */

/**
 * @brief Linear Kalman filter state for VFM flow estimation.
 *
 * State vector x = [flow_rate, bias, drift_rate]^T
 *
 * The Kalman filter provides the optimal minimum-variance estimate
 * of the state given noisy measurements, assuming Gaussian noise
 * and linear dynamics. It is the workhorse estimator for industrial VFM.
 *
 * Recursive equations:
 *   Predict:  x_pred = A * x_prev + B * u
 *             P_pred = A * P_prev * A^T + Q
 *   Update:   K = P_pred * H^T * (H * P_pred * H^T + R)^(-1)
 *             x_new = x_pred + K * (z - H * x_pred)
 *             P_new = (I - K * H) * P_pred
 */
typedef struct {
    double x[3];              /**< State vector [flow, bias, drift]         */
    double P[9];              /**< Covariance matrix (3x3, row-major)       */
    double A[9];              /**< State transition matrix (3x3)            */
    double H[3];              /**< Measurement matrix (1x3)                 */
    double Q[9];              /**< Process noise covariance (3x3)           */
    double R;                 /**< Measurement noise variance (scalar)      */
    double K[3];              /**< Kalman gain (3x1)                        */
    double innovation;        /**< Most recent innovation (z - H*x_pred)    */
    double innovation_cov;    /**< Innovation covariance S = H*P*H^T + R    */
    int    initialized;       /**< 1 = filter state has been initialized    */
    int    steps;             /**< Number of update steps performed         */
} vfm_kalman_t;

/**
 * @brief Recursive Least Squares (RLS) for online parameter estimation.
 *
 * Used in VFM for online calibration of discharge coefficients and
 * pipe roughness factors. RLS updates parameter estimates with each
 * new measurement, weighting recent data more heavily via forgetting
 * factor lambda (0.95 < lambda < 1.0).
 *
 * RLS update (for single-output linear model y = phi^T * theta):
 *   e = y - phi^T * theta_prev
 *   K = P * phi / (lambda + phi^T * P * phi)
 *   theta_new = theta_prev + K * e
 *   P_new = (1/lambda) * (I - K * phi^T) * P_prev
 *
 * @public
 */
typedef struct {
    double theta[4];          /**< Parameter estimates [n_params]           */
    double P[16];             /**< Covariance matrix (4x4 row-major)        */
    double phi[4];            /**< Regressor vector                         */
    double lambda;            /**< Forgetting factor [0..1]                 */
    double prediction_error;  /**< Most recent prediction error             */
    int    n_params;          /**< Number of parameters (<=4)               */
    int    initialized;       /**< 1 = initialized                         */
    int    steps;             /**< Number of RLS steps performed            */
} vfm_rls_t;

/**
 * @brief Moving Horizon Estimation (MHE) configuration.
 *
 * MHE solves an optimization problem over a fixed-size window of past
 * measurements to estimate the current state. Unlike Kalman filtering,
 * MHE can naturally handle constraints (non-negative flow, physical
 * bounds) and nonlinear models.
 *
 * Cost function over horizon N:
 *   min sum_{k=t-N}^{t} ||y_k - h(x_k)||^2_R^{-1}
 *        + ||x_{t-N} - x_bar||^2_Pi^{-1}
 *        + sum_{k=t-N}^{t-1} ||x_{k+1} - f(x_k)||^2_Q^{-1}
 *
 * This implementation uses a simplified interior-point approach
 * for the constrained least-squares problem.
 */
typedef struct {
    double *x_history;        /**< State history [horizon * state_dim]      */
    double *y_history;        /**< Measurement history [horizon]            */
    double  x_estimate[2];    /**< Current optimal state estimate           */
    double  arrival_cost_P[4];/**< Arrival cost matrix (2x2)               */
    double  Q_weight;         /**< Process noise weight                     */
    double  R_weight;         /**< Measurement noise weight                 */
    int     horizon;          /**< Moving horizon length N                  */
    int     current_idx;      /**< Current index in circular buffer         */
    int     filled;           /**< 1 = buffer has N samples                */
} vfm_mhe_t;

/* ==========================================================================
 * L2: Kalman Filter API
 * ========================================================================== */

/**
 * @brief Initialize a Kalman filter for VFM flow estimation.
 *
 * Sets up state transition matrix A = identity (random walk model for bias),
 * measurement matrix H = [1, 1, 0], process noise Q, and measurement noise R.
 *
 * Process noise Q: represents uncertainty in the flow dynamics model.
 *   Small Q -> trusts model more -> slower response to changes.
 *   Large Q -> trusts measurements more -> noisy estimate.
 *
 * Measurement noise R: represents sensor noise variance.
 *   Usually estimated from sensor specifications or data.
 *
 * @param kf          Pointer to Kalman filter struct
 * @param proc_noise  Process noise standard deviation [flow units]
 * @param meas_noise  Measurement noise standard deviation [flow units]
 * @return 0 on success
 */
int vfm_kalman_init(vfm_kalman_t *kf, double proc_noise, double meas_noise);

/**
 * @brief Kalman filter predict step (time update).
 *
 * Projects state and covariance forward one time step.
 * x_pred = A * x,  P_pred = A * P * A^T + Q
 *
 * Must be called before each measurement update.
 *
 * @param kf  Kalman filter state
 * @return 0 on success, -1 if not initialized
 */
int vfm_kalman_predict(vfm_kalman_t *kf);

/**
 * @brief Kalman filter update step (measurement update).
 *
 * Incorporates a new measurement into the state estimate.
 * K = P * H^T / (H * P * H^T + R)
 * x = x + K * (z - H*x)
 * P = (I - K*H) * P
 *
 * @param kf           Kalman filter state
 * @param measurement  New flow measurement [m^3/s]
 * @return 0 on success
 */
int vfm_kalman_update(vfm_kalman_t *kf, double measurement);

/**
 * @brief Get current flow estimate from Kalman filter.
 *
 * @param kf  Kalman filter state
 * @param flow Output: estimated flow rate [m^3/s]
 * @param uncert Output: flow uncertainty (std dev) [m^3/s]
 * @return 0 on success
 */
int vfm_kalman_get_estimate(const vfm_kalman_t *kf, double *flow,
                             double *uncert);

/**
 * @brief Get estimated bias from Kalman filter.
 *
 * The bias state tracks systematic offset between the physical
 * model and actual flow. Important for drift detection.
 *
 * @param kf   Kalman filter state
 * @param bias Output: estimated bias [m^3/s]
 * @return 0 on success
 */
int vfm_kalman_get_bias(const vfm_kalman_t *kf, double *bias);

/**
 * @brief Adaptively tune measurement noise R based on innovation sequence.
 *
 * If actual innovation variance > predicted (H*P*H^T + R), increase R.
 * If actual innovation variance < predicted, decrease R.
 * This makes the filter robust to changing sensor noise levels.
 *
 * Uses the ALS (Adaptive Limited memory) approach:
 *   R_new = R_old + gamma * (e^2 - (H*P*H^T + R_old))
 *
 * where gamma is a small learning rate.
 *
 * @param kf     Kalman filter state
 * @param gamma  Adaptation rate [0..0.1]
 * @return Updated R value
 */
double vfm_kalman_adapt_noise(vfm_kalman_t *kf, double gamma);

/* ==========================================================================
 * L3: Recursive Least Squares (RLS) API
 * ========================================================================== */

/**
 * @brief Initialize RLS estimator.
 *
 * Sets theta to initial guesses, P to large diagonal (high initial
 * uncertainty), and stores the forgetting factor lambda.
 *
 * @param rls        RLS structure
 * @param n_params   Number of parameters (<=4)
 * @param lambda     Forgetting factor [0.95, 1.0]
 * @param theta0     Initial parameter guesses [n_params]
 * @return 0 on success, -1 if n_params > 4
 */
int vfm_rls_init(vfm_rls_t *rls, int n_params, double lambda,
                  const double *theta0);

/**
 * @brief Update RLS parameter estimates with new measurement.
 *
 * y = phi^T * theta (linear regression model)
 *
 * Performs one full RLS update step:
 *   1. Compute prediction error e = y - phi^T * theta
 *   2. Compute gain K = P*phi / (lambda + phi^T*P*phi)
 *   3. Update theta += K * e
 *   4. Update P = (1/lambda) * (P - K*phi^T*P)
 *
 * @param rls      RLS structure
 * @param phi      Regressor vector [n_params]
 * @param y        Measured output
 * @return 0 on success
 */
int vfm_rls_update(vfm_rls_t *rls, const double *phi, double y);

/**
 * @brief Get current RLS parameter estimates.
 *
 * @param rls    RLS structure
 * @param theta  Output: parameter estimates [n_params]
 * @return 0 on success
 */
int vfm_rls_get_params(const vfm_rls_t *rls, double *theta);

/* ==========================================================================
 * L4: Moving Horizon Estimation API
 * ========================================================================== */

/**
 * @brief Initialize MHE with given horizon length.
 *
 * @param mhe      MHE structure
 * @param horizon  Number of past measurements to consider
 * @param q_weight Process noise penalty weight
 * @param r_weight Measurement noise penalty weight
 * @return 0 on success, -1 on allocation/memory failure
 */
int vfm_mhe_init(vfm_mhe_t *mhe, int horizon, double q_weight,
                  double r_weight);

/**
 * @brief Add a new measurement to the MHE buffer and re-solve.
 *
 * On each call, the new measurement is stored, and the MHE optimization
 * over the fixed-size horizon is solved to produce the current state
 * estimate.
 *
 * Uses a simplified Gauss-Newton approach for the optimization.
 *
 * @param mhe          MHE structure
 * @param measurement  New flow measurement [m^3/s]
 * @param model_pred   Model-predicted flow (control input) [m^3/s]
 * @param estimate     Output: current state estimate [m^3/s]
 * @return 0 on success
 */
int vfm_mhe_step(vfm_mhe_t *mhe, double measurement, double model_pred,
                  double *estimate);

/**
 * @brief Free MHE allocated memory.
 *
 * @param mhe  MHE structure
 */
void vfm_mhe_free(vfm_mhe_t *mhe);

/* ==========================================================================
 * L5: Sensor Fusion -- Weighted Least Squares
 * ========================================================================== */

/**
 * @brief Fuse multiple flow estimates using inverse-variance weighting.
 *
 * Given N independent flow estimates x_i, each with variance sigma_i^2,
 * the optimal combined estimate (minimum variance, unbiased) is:
 *
 *   x_fused = sum(x_i / sigma_i^2) / sum(1 / sigma_i^2)
 *   sigma_fused^2 = 1 / sum(1 / sigma_i^2)
 *
 * This is equivalent to the maximum-likelihood estimate assuming
 * independent Gaussian errors.
 *
 * @param estimates      Array of N flow estimates [m^3/s]
 * @param variances      Array of N variances [m^6/s^2]
 * @param n              Number of estimates to fuse
 * @param fused_estimate Output: optimal fused estimate
 * @param fused_variance Output: variance of fused estimate
 * @return 0 on success, -1 if n <= 0, -2 if any variance <= 0
 */
int vfm_sensor_fusion_wls(const double *estimates, const double *variances,
                           int n, double *fused_estimate,
                           double *fused_variance);

/**
 * @brief Compute consistency check for multiple sensor estimates.
 *
 * Computes the chi-squared statistic to test the null hypothesis that
 * all sensors are measuring the same true value.
 *
 * chi2 = sum_i (x_i - x_fused)^2 / sigma_i^2
 *
 * If chi2 is large, at least one sensor is biased or faulty.
 *
 * @param estimates   Array of N flow estimates
 * @param variances   Array of N variances
 * @param n           Number of estimates
 * @param fused       Fused estimate (from vfm_sensor_fusion_wls)
 * @return Chi-squared statistic
 */
double vfm_consistency_chi2(const double *estimates, const double *variances,
                             int n, double fused);

/* ==========================================================================
 * L6: Drift Detection
 * ========================================================================== */

/**
 * @brief CUSUM (Cumulative Sum) drift detection.
 *
 * Detects a persistent shift in the mean of a signal. CUSUM accumulates
 * deviations from a target value and triggers when the cumulative sum
 * exceeds a threshold.
 *
 * For detecting positive drift:
 *   S_high[k] = max(0, S_high[k-1] + (x[k] - mu0 - delta/2))
 *
 * For detecting negative drift:
 *   S_low[k] = max(0, S_low[k-1] - (x[k] - mu0 + delta/2))
 *
 * Drift is declared if S_high > H or S_low > H.
 *
 * @param x           Current measurement
 * @param mu0         Target mean (no-drift value)
 * @param delta       Minimum detectable shift magnitude
 * @param H           Detection threshold
 * @param s_high      Input/output: cumulative sum for positive drift
 * @param s_low       Input/output: cumulative sum for negative drift
 * @return 1 if drift detected, 0 otherwise
 */
int vfm_cusum_drift_detect(double x, double mu0, double delta, double H,
                            double *s_high, double *s_low);

#ifdef __cplusplus
}
#endif

#endif /* VFM_STATE_ESTIMATION_H */