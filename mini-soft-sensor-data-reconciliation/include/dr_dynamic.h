/**
 * @file dr_dynamic.h
 * @brief Dynamic data reconciliation using Kalman filtering and
 * Moving Horizon Estimation (MHE).
 *
 * Dynamic data reconciliation extends steady-state DR to time-varying
 * processes where measurements arrive sequentially and the process
 * evolves according to a dynamic model.
 *
 * State-space formulation:
 *   x_{k+1} = F_k * x_k + B_k * u_k + G_k * w_k    (state equation)
 *   y_k     = H_k * x_k + v_k                       (measurement equation)
 *
 * where:
 *   x_k : state vector (process variables)
 *   u_k : known inputs (control actions, feed rates)
 *   y_k : measurement vector
 *   w_k ~ N(0, Q_k) : process noise (model uncertainty)
 *   v_k ~ N(0, R_k) : measurement noise
 *
 * Kalman filter provides the optimal state estimate (minimum variance)
 * for linear Gaussian systems.
 *
 * Three modes of operation:
 *   1. Prediction: Estimate state at k+1 given measurements up to k
 *   2. Filtering: Estimate state at k given measurements up to k
 *   3. Smoothing: Estimate state at k given all measurements (up to N)
 *
 * Moving Horizon Estimation (MHE):
 *   Uses a finite window of past measurements to estimate states,
 *   incorporating constraints (mass/energy balances) directly.
 *   Formulated as a constrained optimization problem over the horizon.
 *
 * References:
 *   [1] Kalman, R.E. (1960). "A New Approach to Linear Filtering and
 *       Prediction Problems." J. Basic Eng., 82(1), 35-45.
 *   [2] Rauch, H.E., Tung, F., Striebel, C.T. (1965). "Maximum Likelihood
 *       Estimates of Linear Dynamic Systems." AIAA Journal, 3(8), 1445-1450.
 *   [3] Robertson, D.G., Lee, J.H., Rawlings, J.B. (1996). "A Moving
 *       Horizon-Based Approach for Least-Squares Estimation." AIChE Journal,
 *       42(8), 2209-2224.
 *   [4] Rao, C.V., Rawlings, J.B., Mayne, D.Q. (2003). "Constrained State
 *       Estimation for Nonlinear Discrete-Time Systems." IEEE Trans. Auto.
 *       Control, 48(2), 246-258.
 *   [5] Gelb, A. (1974). "Applied Optimal Estimation." MIT Press.
 *   [6] Simon, D. (2006). "Optimal State Estimation." Wiley.
 *   [7] Darouach, M., Zasadzinski, M. (1991). "Data Reconciliation in
 *       Generalized Linear Dynamic Systems." AIChE Journal, 37(2), 193-201.
 */

#ifndef DR_DYNAMIC_H
#define DR_DYNAMIC_H

#include "dr_core.h"

/* ============================================================================
 * L1: Dynamic System Structures
 * ============================================================================ */

/**
 * @brief Linear time-invariant (LTI) state-space model for dynamic DR.
 */
typedef struct {
    int      n_states;        /**< Number of state variables (nx) */
    int      n_inputs;        /**< Number of known inputs (nu) */
    int      n_measurements;  /**< Number of measurements (ny) */
    double  *F;               /**< State transition matrix (nx x nx, row-major) */
    double  *B;               /**< Input matrix (nx x nu, row-major) */
    double  *H;               /**< Measurement matrix (ny x nx, row-major) */
    double  *Q;               /**< Process noise covariance (nx x nx) */
    double  *R;               /**< Measurement noise covariance (ny x ny) */
    double  *G;               /**< Process noise input matrix (nx x nw) */
    int      n_noise;         /**< Dimension of process noise vector */
} dr_dyn_model_t;

/**
 * @brief Kalman filter state for dynamic data reconciliation.
 *
 * Maintains the current state estimate and its error covariance,
 * which together represent the complete Gaussian belief state:
 *   x_k ~ N(x_hat_k, P_k)
 */
typedef struct {
    int      n_states;        /**< Number of state variables */
    double  *x_hat;           /**< Current state estimate (length nx) */
    double  *P;               /**< Error covariance matrix (nx x nx, row-major) */
    double  *K;               /**< Kalman gain matrix (nx x ny, row-major) */
    double  *innovation;      /**< Innovation (measurement residual) (length ny) */
    double  *S;               /**< Innovation covariance (ny x ny, row-major) */
    double   log_likelihood;  /**< Accumulated log-likelihood of measurements */
    int      k;               /**< Current time step */
    int      initialized;     /**< 1 if filter has been initialized */
} dr_kf_state_t;

/**
 * @brief Moving Horizon Estimation configuration.
 */
typedef struct {
    int      horizon_length;  /**< Number of past measurements in window (N) */
    int      n_states;        /**< Number of state variables */
    int      n_measurements;  /**< Number of measurements per time step */
    double  *arrival_cost_P;  /**< Arrival cost covariance (nx x nx) */
    double  *arrival_cost_x;  /**< Arrival cost state estimate (length nx) */
    int      use_constraints; /**< 1 = enforce physical constraints in MHE */
} dr_mhe_config_t;

/* ============================================================================
 * L2: Model Lifecycle
 * ============================================================================ */

/**
 * @brief Allocate and initialize a dynamic system model.
 *
 * @param nx  Number of state variables.
 * @param nu  Number of inputs.
 * @param ny  Number of measurements.
 * @param nw  Number of process noise inputs.
 * @return    Pointer to allocated model, or NULL.
 */
dr_dyn_model_t *dr_dyn_model_create(int nx, int nu, int ny, int nw);

/**
 * @brief Free a dynamic system model.
 *
 * @param model  Model to free. Safe to pass NULL.
 */
void dr_dyn_model_free(dr_dyn_model_t *model);

/**
 * @brief Set the state transition matrix F.
 *
 * @param model  Dynamic model.
 * @param F      State transition matrix (row-major, nx x nx).
 * @return       DR_OK or error code.
 */
int dr_dyn_model_set_F(dr_dyn_model_t *model, const double *F);

/**
 * @brief Set the measurement matrix H.
 *
 * @param model  Dynamic model.
 * @param H      Measurement matrix (row-major, ny x nx).
 * @return       DR_OK or error code.
 */
int dr_dyn_model_set_H(dr_dyn_model_t *model, const double *H);

/**
 * @brief Set process and measurement noise covariances.
 *
 * Q and R must be symmetric positive (semi-)definite.
 *
 * @param model  Dynamic model.
 * @param Q      Process noise covariance (row-major, nx x nx).
 * @param R      Measurement noise covariance (row-major, ny x ny).
 * @return       DR_OK or error code.
 */
int dr_dyn_model_set_noise(dr_dyn_model_t *model,
                           const double *Q, const double *R);

/* ============================================================================
 * L3: Kalman Filter Operations
 * ============================================================================ */

/**
 * @brief Initialize the Kalman filter state.
 *
 * Sets the initial state estimate x_hat_0 and error covariance P_0.
 * P_0 should reflect the uncertainty in the initial state estimate.
 *
 * @param kf    Kalman filter state to initialize.
 * @param nx    Number of state variables.
 * @param ny    Number of measurements.
 * @param x0    Initial state estimate.
 * @param P0    Initial error covariance (row-major, nx x nx).
 * @return      DR_OK or error code.
 */
int dr_kf_init(dr_kf_state_t *kf, int nx, int ny,
               const double *x0, const double *P0);

/**
 * @brief Kalman filter prediction step (time update).
 *
 * Projects the state and covariance forward in time:
 *   x̂_{k|k-1} = F_k * x̂_{k-1|k-1} + B_k * u_k
 *   P_{k|k-1}   = F_k * P_{k-1|k-1} * F_k^T + G_k * Q_k * G_k^T
 *
 * @param kf     Kalman filter state.
 * @param model  System model.
 * @param u      Input vector at current time (length nu, or NULL if nu=0).
 * @return       DR_OK or error code.
 *
 * Complexity: O(nx^3) for covariance propagation matrix multiply.
 *
 * Theorem (Kalman, 1960): The predicted estimate is the minimum
 * variance unbiased estimate of x_k given measurements up to k-1.
 */
int dr_kf_predict(dr_kf_state_t *kf, const dr_dyn_model_t *model,
                  const double *u);

/**
 * @brief Kalman filter update step (measurement update).
 *
 * Incorporates new measurement y_k:
 *   y_tilde_k    = y_k - H_k * x̂_{k|k-1}                    (innovation)
 *   S_k          = H_k * P_{k|k-1} * H_k^T + R_k            (innovation covariance)
 *   K_k          = P_{k|k-1} * H_k^T * S_k^{-1}             (Kalman gain)
 *   x̂_{k|k}     = x̂_{k|k-1} + K_k * y_tilde_k              (updated estimate)
 *   P_{k|k}      = (I - K_k * H_k) * P_{k|k-1}              (updated covariance)
 *
 * Joseph form for numerical stability:
 *   P_{k|k} = (I-KH)*P*(I-KH)^T + K*R*K^T
 *
 * @param kf     Kalman filter state.
 * @param model  System model.
 * @param y      Measurement vector at current time (length ny).
 * @return       DR_OK or error code.
 *
 * Complexity: O(ny^3 + nx*ny*nx) for covariance update.
 *
 * Theorem (Kalman, 1960): The updated estimate minimizes the trace
 * of the error covariance matrix among all linear estimators.
 */
int dr_kf_update(dr_kf_state_t *kf, const dr_dyn_model_t *model,
                 const double *y);

/**
 * @brief Free a Kalman filter state.
 *
 * @param kf  Kalman filter state to free.
 */
void dr_kf_free(dr_kf_state_t *kf);

/* ============================================================================
 * L4: Kalman Smoothing
 * ============================================================================ */

/**
 * @brief Rauch-Tung-Striebel (RTS) fixed-interval smoother.
 *
 * Given filtered estimates x̂_{k|k} and P_{k|k} for k = 0..N-1,
 * computes smoothed estimates x̂_{k|N} and P_{k|N} backwards:
 *
 *   C_k     = P_{k|k} * F_k^T * P_{k+1|k}^{-1}             (smoother gain)
 *   x̂_{k|N} = x̂_{k|k} + C_k * (x̂_{k+1|N} - x̂_{k+1|k})     (smoothed state)
 *   P_{k|N}  = P_{k|k} + C_k * (P_{k+1|N} - P_{k+1|k}) * C_k^T
 *
 * @param kf_hist  Array of N Kalman filter states (filtered).
 * @param model    System model.
 * @param N        Number of time steps.
 * @param x_smooth Output: smoothed states (N x nx, row-major per step).
 * @param P_smooth Output: smoothed covariances (N x nx x nx, row-major).
 * @return         DR_OK or error code.
 *
 * Complexity: O(N * nx^3).
 *
 * Reference: Rauch, H.E., Tung, F., Striebel, C.T. (1965). "Maximum
 * Likelihood Estimates of Linear Dynamic Systems." AIAA Journal,
 * 3(8), 1445-1450.
 *
 * Theorem: The RTS smoother provides the minimum variance estimate
 * of x_k given all measurements y_0, ..., y_{N-1}.
 */
int dr_kf_smooth_rts(const dr_kf_state_t *kf_hist,
                     const dr_dyn_model_t *model, int N,
                     double *x_smooth, double *P_smooth);

/* ============================================================================
 * L5: Innovation Analysis
 * ============================================================================ */

/**
 * @brief Test if the Kalman filter innovations are white (uncorrelated).
 *
 * Under optimal filtering, innovations should be zero-mean white noise:
 *   E[y_tilde_k] = 0
 *   E[y_tilde_k * y_tilde_j^T] = 0 for k != j
 *
 * Autocorrelation test: computes the Portmanteau (Ljung-Box) statistic
 * for the first L lags of the normalized innovations.
 *
 * @param innovations   Array of N innovation vectors (each length ny).
 * @param ny            Number of measurements.
 * @param N             Number of time steps.
 * @param max_lag       Maximum lag for autocorrelation test.
 * @param Q_stat_out    Output: Ljung-Box Q statistic.
 * @param is_white_out  Output: 1 if innovations appear white, 0 otherwise.
 * @return              DR_OK or error code.
 *
 * Complexity: O(N * ny * max_lag).
 *
 * Reference: Mehra, R.K. (1970). "On the Identification of Variances
 * and Adaptive Kalman Filtering." IEEE Trans. Auto. Control, 15(2),
 * 175-184.
 */
int dr_kf_innovation_whiteness(const double *innovations, int ny, int N,
                               int max_lag, double *Q_stat_out,
                               int *is_white_out);

/**
 * @brief Compute the steady-state Kalman gain by solving the
 * Discrete Algebraic Riccati Equation (DARE).
 *
 * The steady-state gain K_ss satisfies:
 *   P_ss = F * P_ss * F^T - F * P_ss * H^T * (H * P_ss * H^T + R)^{-1}
 *          * H * P_ss * F^T + G * Q * G^T
 *   K_ss = P_ss * H^T * (H * P_ss * H^T + R)^{-1}
 *
 * Solved via iterative Riccati recursion until convergence.
 *
 * @param model   System model.
 * @param K_ss    Output: steady-state Kalman gain (row-major, nx x ny).
 * @param P_ss    Output: steady-state error covariance (row-major, nx x nx).
 * @param max_iter Maximum iterations.
 * @param tol     Convergence tolerance.
 * @return        DR_OK or error code.
 *
 * Complexity: O(k * nx^3) where k is iterations to convergence.
 */
int dr_kf_steady_gain(const dr_dyn_model_t *model, double *K_ss,
                      double *P_ss, int max_iter, double tol);

/* ============================================================================
 * L6: Moving Horizon Estimation (MHE)
 * ============================================================================ */

/**
 * @brief Initialize MHE configuration with default parameters.
 *
 * @param config    Output: initialized MHE configuration.
 * @param nx        Number of state variables.
 * @param ny        Number of measurements.
 * @param horizon   Horizon length N.
 * @return          DR_OK or error code.
 */
int dr_mhe_init(dr_mhe_config_t *config, int nx, int ny, int horizon);

/**
 * @brief Free MHE configuration resources.
 *
 * @param config  MHE configuration to free.
 */
void dr_mhe_free(dr_mhe_config_t *config);

/**
 * @brief Solve one step of Moving Horizon Estimation.
 *
 * Given the current measurement y_k and past horizon of measurements
 * y_{k-N+1:k}, estimates the state x_{k-N+1} at the start of the
 * horizon by minimizing:
 *
 *   min J = ||x_{k-N+1} - x_bar||^2_{P_bar^{-1}}
 *         + sum_{i=k-N+1}^{k} ||y_i - H * x_i||^2_{R^{-1}}
 *         + sum_{i=k-N+1}^{k-1} ||x_{i+1} - F * x_i - B * u_i||^2_{Q^{-1}}
 *
 * This is solved as an unconstrained least-squares problem by stacking
 * all measurements and the dynamic model into one large system,
 * analogous to steady-state DR but extended over the time horizon.
 *
 * @param model    System model.
 * @param config   MHE configuration.
 * @param y_hist   Measurement history (length horizon * ny).
 * @param u_hist   Input history (length horizon * nu, or NULL).
 * @param x_out    Output: estimated state at current time (length nx).
 * @return         DR_OK or error code.
 *
 * Complexity: O((N * (nx + ny))^3) for the stacked system solve.
 *
 * Reference: Robertson, D.G., Lee, J.H., Rawlings, J.B. (1996).
 * "A Moving Horizon-Based Approach for Least-Squares Estimation."
 * AIChE Journal, 42(8), 2209-2224.
 */
int dr_mhe_solve(const dr_dyn_model_t *model, const dr_mhe_config_t *config,
                 const double *y_hist, const double *u_hist, double *x_out);

/**
 * @brief Update the MHE arrival cost using the EKF covariance update.
 *
 * The arrival cost summarizes the effect of measurements older than
 * the horizon on the current estimate. This uses the extended Kalman
 * filter covariance propagation to approximate the arrival cost.
 *
 * @param config     MHE configuration (arrival cost updated in-place).
 * @param model      System model.
 * @param kf         Current Kalman filter state.
 * @return           DR_OK or error code.
 *
 * Complexity: O(nx^3).
 */
int dr_mhe_update_arrival_cost(dr_mhe_config_t *config,
                               const dr_dyn_model_t *model,
                               const dr_kf_state_t *kf);

#endif /* DR_DYNAMIC_H */
