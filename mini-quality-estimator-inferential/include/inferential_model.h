/**
 * @file inferential_model.h
 * @brief Inferential model structures — first-principles, data-driven, hybrid, and state-space.
 *
 * Level: L2 Core Concepts + L3 Engineering Structures + L4 Engineering Laws
 * Reference:
 *   Seborg, Edgar, Mellichamp (2016) "Process Dynamics and Control" Ch. 20
 *   Fortuna, Graziani, Rizzo, Xibilia (2007) "Soft Sensors for Monitoring and Control of Industrial Processes"
 *   Kadlec, Gabrys (2011) "Review of adaptation mechanisms for data-driven soft sensors" — CACE, 35(1)
 *
 * Course mapping:
 *   MIT 6.302: Feedback Systems — observer design, state estimation
 *   Stanford ENGR205: Process Control — inferential models, soft sensing
 *   Berkeley ME233: Advanced Control — Kalman filtering, state-space models
 */

#ifndef INFERENTIAL_MODEL_H
#define INFERENTIAL_MODEL_H

#include "quality_estimator_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
 * L2: Static Inferential Model — First-Principles
 *===========================================================================*/

/**
 * @brief Initialize a first-principles inferential model.
 *
 * The first-principles model captures the physical relationship between
 * measurable process variables and the unmeasured quality variable:
 *
 *   y_quality = f(x_1, x_2, ..., x_n; theta)
 *
 * where f() encodes mass balances, energy balances, reaction kinetics,
 * phase equilibria, or any physically derived mapping.
 *
 * @param model      Uninitialized first-principles model struct
 * @param evaluate   Model evaluation function pointer
 * @param params     Physical parameter array
 * @param n_params   Number of physical parameters (≤ 64)
 * @param n_states   For dynamic models: number of internal states (0 for static)
 *
 * Theorem: For a well-posed physical model, small perturbations in inputs
 * produce bounded perturbations in output (Lipschitz continuity).
 */
void fpm_init(fpm_model_t *model,
              double (*evaluate)(const double *, int, const double *, int),
              const double *params, int n_params, int n_states);

/**
 * @brief Evaluate a first-principles quality model.
 *
 * @param model    Configured first-principles model
 * @param inputs   Process variable values [n_inputs]
 * @param n_inputs Number of input variables
 * @return         Predicted quality value. Returns NaN if validation fails.
 *
 * Complexity: O(n_inputs * n_params) for evaluation function.
 */
double fpm_evaluate(const fpm_model_t *model, const double *inputs, int n_inputs);

/**
 * @brief Compute first-principles model sensitivity (Jacobian).
 *
 * The sensitivity ∂y/∂x_i quantifies how each process variable influences
 * the quality estimate. This is essential for:
 *   - Measurement system design (which sensors matter most?)
 *   - Sensor fault detection (which sensors to monitor?)
 *   - Input validation (expected rate of change bounds)
 *
 * Uses central finite differences: ∂f/∂x_i ≈ (f(x+h*e_i) - f(x-h*e_i)) / (2h)
 *
 * @param model     Configured first-principles model
 * @param inputs    Current process variable values
 * @param n_inputs  Number of input variables
 * @param jacobian  [out] Sensitivity vector of length n_inputs
 * @param h         Finite difference step size (e.g., 0.001)
 */
void fpm_sensitivity(const fpm_model_t *model, const double *inputs, int n_inputs,
                     double *jacobian, double h);

/*===========================================================================
 * L3: Dynamic State-Space Model for Quality Estimation
 *===========================================================================*/

/**
 * @brief Discrete-time linear state-space model for quality dynamics.
 *
 * Standard form:
 *   x(k+1) = A * x(k) + B * u(k) + w(k)     (state equation)
 *   y(k)   = C * x(k) + D * u(k) + v(k)     (output equation — quality)
 *
 * where:
 *   x = state vector (may include unmeasured quality as a state)
 *   u = known inputs (process variables: temperatures, flows, pressures)
 *   y = estimated quality output
 *   w = process noise ~ N(0, Q)
 *   v = measurement noise ~ N(0, R)
 *
 * Reference: Simon, D. (2006) "Optimal State Estimation" — Ch. 5 Discrete Kalman Filter.
 */
typedef struct {
    int     n_states;        /**< Number of state variables */
    int     n_inputs;        /**< Number of known inputs (process variables) */
    int     n_outputs;       /**< Number of quality outputs */

    double *A;               /**< State transition matrix [n_states × n_states] */
    double *B;               /**< Input matrix [n_states × n_inputs] */
    double *C;               /**< Output (quality) matrix [n_outputs × n_states] */
    double *D;               /**< Feedthrough matrix [n_outputs × n_inputs] */

    double *Q;               /**< Process noise covariance [n_states × n_states] */
    double *R;               /**< Measurement noise covariance [n_outputs × n_outputs] */

    double *x;               /**< Current state estimate [n_states] */
    double *P;               /**< State error covariance [n_states × n_states] */
    double *x_prior;         /**< Prior (predicted) state [n_states] */
    double *P_prior;         /**< Prior error covariance [n_states × n_states] */
} ss_model_t;

/**
 * @brief Allocate and initialize a linear state-space model.
 *
 * Memory is allocated for all matrices based on dimensions.
 * A is initialized to identity, B/C/D/Q/R to zero.
 * P is initialized to a large diagonal (high initial uncertainty).
 *
 * @param model     Uninitialized state-space model pointer
 * @param n_states  State dimension
 * @param n_inputs  Input dimension
 * @param n_outputs Output (quality) dimension
 *
 * Complexity: O(n_states^2 + n_states*n_inputs + n_outputs*n_states) for initialization.
 */
void ss_model_alloc(ss_model_t *model, int n_states, int n_inputs, int n_outputs);

/**
 * @brief Free memory allocated for a state-space model.
 *
 * @param model  State-space model to free
 */
void ss_model_free(ss_model_t *model);

/**
 * @brief State prediction step: x_prior = A*x + B*u.
 *
 * @param model  State-space model
 * @param u      Input vector [n_inputs]
 */
void ss_model_predict(ss_model_t *model, const double *u);

/**
 * @brief Output (quality) prediction: y = C*x + D*u.
 *
 * @param model  State-space model
 * @param u      Input vector [n_inputs]
 * @param y      [out] Predicted quality [n_outputs]
 */
void ss_model_output(const ss_model_t *model, const double *u, double *y);

/*===========================================================================
 * L3: ARX Dynamic Model for Quality Estimation
 *===========================================================================*/

/**
 * @brief Auto-Regressive with eXogenous inputs (ARX) model for dynamic quality.
 *
 * The ARX model captures the dynamic relationship between process variables
 * (inputs) and quality (output) using past values:
 *
 *   y(k) + a_1*y(k-1) + ... + a_na*y(k-na) =
 *         b_0*u(k) + b_1*u(k-1) + ... + b_nb*u(k-nb) + e(k)
 *
 * This is the most widely used linear dynamic model in process identification.
 *
 * Reference: Ljung, L. (1999) "System Identification: Theory for the User" Ch. 4.
 */
typedef struct {
    int     na;               /**< Number of auto-regressive (past output) terms */
    int     nb;               /**< Number of exogenous (input) terms per input */
    int     n_inputs;         /**< Number of input variables */
    int     nk;               /**< Input delay in samples (pure time delay) */

    double  a_coeffs[QEST_MAX_LAGS];                              /**< AR coefficients a_1..a_na */
    double  b_coeffs[QEST_MAX_INPUT_VARS * QEST_MAX_LAGS];        /**< X coefficients b_0..b_nb per input */

    /** Buffers for past values (circular) */
    double  y_past[QEST_MAX_LAGS];     /**< Past quality values y(k-1)..y(k-na) */
    double  u_past[QEST_MAX_INPUT_VARS * QEST_MAX_LAGS]; /**< Past input values */
    int     y_buffer_pos;              /**< Current write position in y buffer */
    int     u_buffer_pos;              /**< Current write position in u buffer */
    int     y_buffer_filled;           /**< Flag: buffer has been fully populated */
} arx_model_t;

/**
 * @brief Initialize an ARX model with given orders.
 *
 * @param model    Uninitialized ARX model struct
 * @param na       AR order (number of past output terms, 0..QEST_MAX_LAGS-1)
 * @param nb       X order (number of past input terms, 0..QEST_MAX_LAGS-1)
 * @param n_inputs Number of input variables
 * @param nk       Input delay (≥ 0)
 */
void arx_init(arx_model_t *model, int na, int nb, int n_inputs, int nk);

/**
 * @brief Set ARX model coefficients.
 *
 * @param model  ARX model to configure
 * @param a      AR coefficients [na] (a_1 through a_na)
 * @param b      X coefficients [n_inputs * (nb+1)] (b_0 through b_nb per input)
 */
void arx_set_coeffs(arx_model_t *model, const double *a, const double *b);

/**
 * @brief Predict quality using the ARX model.
 *
 * y_hat(k) = -sum_{i=1}^{na} a_i*y(k-i) + sum_{j=1}^{n_inputs} sum_{i=0}^{nb} b_{j,i}*u_j(k-i-nk)
 *
 * @param model  Configured ARX model
 * @param u      Current input values [n_inputs]
 * @return       Predicted quality value
 */
double arx_predict(arx_model_t *model, const double *u);

/**
 * @brief Update ARX model buffers with new measurements.
 *
 * After a lab sample y_measured becomes available, call this to shift
 * the output buffer and store the new ground truth.
 *
 * @param model   ARX model
 * @param y_true  Lab-measured quality value
 * @param u       Current input values [n_inputs] (stored for future predictions)
 */
void arx_update_buffers(arx_model_t *model, double y_true, const double *u);

/*===========================================================================
 * L2: Hybrid Grey-Box Model
 *===========================================================================*/

/**
 * @brief Grey-box (hybrid) quality model combining first-principles with data correction.
 *
 * The hybrid model structure:
 *   y_hybrid = y_fpm(x; theta) + delta_y_data(x; phi)
 *
 * where:
 *   y_fpm = first-principles model output (captures known physics)
 *   delta_y_data = data-driven correction (captures unmodeled effects, e.g. fouling, catalyst aging)
 *
 * This combines the extrapolation capability of first-principles models
 * with the accuracy of data-driven models near the training region.
 *
 * Reference: Psichogios, Ungar (1992) "A hybrid neural network-first principles
 *            approach to process modeling" — AIChE J, 38(10).
 */
typedef struct {
    fpm_model_t  fpm;           /**< First-principles component */
    int          use_correction; /**< Flag: enable/disable data correction term */
    double       corr_bias;      /**< Additive correction bias */
    double       corr_scale;     /**< Multiplicative correction scale factor */
    linear_model_t corr_linear;  /**< Linear correction model (if used) */

    /** Performance tracking for each component */
    double       fpm_mse;        /**< FPM-only MSE on validation data */
    double       hybrid_mse;     /**< Full hybrid MSE on validation data */
} hybrid_model_t;

/**
 * @brief Initialize a hybrid (grey-box) quality model.
 *
 * @param model  Uninitialized hybrid model struct
 * @param fpm    Configured first-principles component
 */
void hybrid_init(hybrid_model_t *model, const fpm_model_t *fpm);

/**
 * @brief Evaluate the hybrid model.
 *
 * y = fpm_evaluate(x) + bias_correction + linear_correction(x)
 *
 * @param model    Configured hybrid model
 * @param inputs   Process variable values
 * @param n_inputs Number of inputs
 * @return         Hybrid quality estimate
 */
double hybrid_evaluate(const hybrid_model_t *model, const double *inputs, int n_inputs);

/**
 * @brief Set the data-driven correction parameters.
 *
 * @param model  Hybrid model to configure
 * @param bias   Additive bias correction term
 * @param scale  Multiplicative scale correction
 */
void hybrid_set_correction(hybrid_model_t *model, double bias, double scale);

/*===========================================================================
 * L2: Linear Model Evaluation (declared here for cross-file access)
 *===========================================================================*/

/**
 * @brief Evaluate a linear regression model: y = beta_0 + sum beta_i * x_i
 *
 * This function is defined in quality_estimator_core.c and declared here
 * because it is used by inferential_model.c (hybrid model evaluation).
 */
double linear_model_evaluate(const linear_model_t *model, const double *x, int n_inputs);

/*===========================================================================
 * L3: Locally Weighted PLS (LW-PLS) Model — Just-In-Time Learning
 *===========================================================================*/

/**
 * @brief Locally Weighted PLS for time-varying quality estimation.
 *
 * Traditional global PLS uses all historical data equally. LW-PLS weights
 * historical samples by their similarity to the current query point,
 * giving more influence to similar operating conditions.
 *
 * Weight function: w_i = exp(-d_i^2 / (phi * sigma_d^2))
 * where d_i = ||x_query - x_i|| is the Euclidean distance and phi is a
 * localization parameter (smaller phi = more local).
 *
 * Reference: Schaal, Atkeson, Vijayakumar (2002) "Real-time robot learning
 *            with locally weighted statistical learning" — IEEE ICRA.
 */
typedef struct {
    pls_model_t  global_model;  /**< Base PLS model trained on all data */
    int          use_local;     /**< Flag: enable LW-PLS (1) or use global (0) */
    double       phi;           /**< Localization parameter (typically 0.1..10) */
    double       sigma_dist;    /**< Scaling parameter for distance normalization */
    int          n_local;       /**< Number of nearest neighbors to use */
    int          n_historical;  /**< Number of historical samples in database */
    double      *x_historical;  /**< Historical input data [n_historical * n_inputs] */
    double      *y_historical;  /**< Historical output data [n_historical * n_outputs] */
    double      *distances;     /**< Scratch space for distance computation */
    double      *weights;       /**< Scratch space for sample weights */
} lwpls_model_t;

/**
 * @brief Initialize a Locally Weighted PLS model.
 *
 * @param model       Uninitialized LW-PLS model
 * @param global_pls  Base PLS model
 * @param phi         Localization parameter
 * @param n_historical Max historical samples to store
 * @param n_inputs    Number of input variables
 * @param n_outputs   Number of output variables
 */
void lwpls_init(lwpls_model_t *model, const pls_model_t *global_pls,
                double phi, int n_historical, int n_inputs, int n_outputs);

/**
 * @brief Free LW-PLS allocated memory.
 *
 * @param model  LW-PLS model to free
 */
void lwpls_free(lwpls_model_t *model);

/**
 * @brief Predict quality using LW-PLS with local weighting.
 *
 * Steps:
 *   1. Compute distances from query x to all historical points
 *   2. Compute weights w_i = exp(-d_i^2 / (phi * sigma_d^2))
 *   3. Select n_local nearest neighbors
 *   4. Fit weighted local PLS model
 *   5. Predict y for query point x
 *
 * @param model    Configured LW-PLS model
 * @param x_query  Query input vector [n_inputs]
 * @param n_inputs Number of input variables
 * @param y_pred   [out] Predicted quality [n_outputs]
 */
void lwpls_predict(lwpls_model_t *model, const double *x_query, int n_inputs, double *y_pred);

/**
 * @brief Add a new historical sample to the LW-PLS database.
 *
 * @param model   LW-PLS model
 * @param x       Input sample [n_inputs]
 * @param y       Quality measurement [n_outputs]
 */
void lwpls_add_sample(lwpls_model_t *model, const double *x, const double *y);

/*===========================================================================
 * L3: Moving Horizon Quality Estimation
 *===========================================================================*/

/**
 * @brief Configure a Moving Horizon Estimation buffer.
 *
 * @param buf      Uninitialized MHE buffer
 * @param horizon  Horizon length N (1..QEST_MAX_LAGS)
 * @param arr_cost Arrival cost weight
 * @param meas_wt  Measurement noise weight
 */
void mhe_init(mhe_buffer_t *buf, int horizon, double arr_cost, double meas_wt);

/**
 * @brief Add a new sample (process variables + lab result) to the MHE buffer.
 *
 * @param buf     MHE buffer
 * @param inputs  Process variable values [n_inputs]
 * @param n_in    Number of input variables
 * @param y_lab   Lab-measured quality value
 * @param ts      Time stamp of the sample
 */
void mhe_push(mhe_buffer_t *buf, const double *inputs, int n_in,
              double y_lab, const qest_timestamp_t *ts);

/**
 * @brief Estimate quality using MHE over the stored horizon.
 *
 * For a linear model y = H*x + v, the MHE solves:
 *   min_x  ||x - x_prior||^2_{Pi^-1} + sum_{i} ||y_i - H_i*x||^2_{R_i^-1}
 *
 * @param buf      MHE buffer with stored data
 * @param model    Linear model (regression coefficients)
 * @param n_inputs Number of input variables
 * @param estimate [out] Quality estimate
 * @param variance [out] Estimate variance
 */
void mhe_estimate(const mhe_buffer_t *buf, const linear_model_t *model,
                  int n_inputs, double *estimate, double *variance);

/*===========================================================================
 * L4: Model Validation and Performance Assessment
 *===========================================================================*/

/**
 * @brief Compute prediction residuals between model and lab measurements.
 *
 * residuals[i] = y_lab[i] - y_model[i]
 *
 * @param y_model   Model-predicted quality values [n]
 * @param y_lab     Lab-measured quality values [n]
 * @param residuals [out] Prediction residuals [n]
 * @param n         Number of paired samples
 */
void compute_residuals(const double *y_model, const double *y_lab,
                       double *residuals, int n);

/**
 * @brief Compute comprehensive regression statistics.
 *
 * Computes MSE, RMSE, MAE, MAPE, R-squared, and residual autocorrelation
 * at lag 1 (Durbin-Watson-like statistic).
 *
 * @param y_model  Model predictions [n]
 * @param y_lab    Lab measurements [n]
 * @param n        Number of samples
 * @param stats    [out] Performance statistics (fills mse, rmse, mae, mape, r_squared fields)
 */
void compute_regression_stats(const double *y_model, const double *y_lab,
                              int n, qest_performance_t *stats);

/**
 * @brief Durbin-Watson test for residual autocorrelation.
 *
 * DW ≈ sum_{i=2}^{n} (e_i - e_{i-1})^2 / sum_{i=1}^{n} e_i^2
 *
 * DW ≈ 2 indicates no autocorrelation.
 * DW < 1 indicates positive autocorrelation (model misses dynamics).
 * DW > 3 indicates negative autocorrelation.
 *
 * @param residuals Prediction residuals [n]
 * @param n         Number of samples (n ≥ 2)
 * @return          Durbin-Watson statistic (0..4)
 */
double durbin_watson_test(const double *residuals, int n);

/**
 * @brief Test if prediction residuals are zero-mean (Student's t-test).
 *
 * Null hypothesis H0: mean(residuals) = 0
 * Tests whether the model is unbiased.
 *
 * @param residuals Prediction residuals [n]
 * @param n         Number of samples
 * @param t_stat    [out] Computed t-statistic
 * @return          1 if H0 is NOT rejected at 95% confidence, 0 otherwise
 */
int t_test_zero_mean(const double *residuals, int n, double *t_stat);

/**
 * @brief Grubbs' test for outlier detection in a single new lab sample.
 *
 * Tests whether y_new is an outlier relative to the distribution of
 * recent model residuals. Used to reject bad lab samples before bias update.
 *
 * G = |y_new - y_pred| / sigma_residuals
 *
 * @param y_new    New lab measurement value
 * @param y_pred   Model prediction at the sample time
 * @param residuals_stddev Standard deviation of historical residuals
 * @param n_hist   Number of historical residuals used to estimate stddev
 * @param alpha    Significance level (e.g., 0.05 for 95%)
 * @return         1 if null hypothesis rejected (outlier detected), 0 otherwise
 */
int grubbs_outlier_test(double y_new, double y_pred, double residuals_stddev,
                        int n_hist, double alpha);

#ifdef __cplusplus
}
#endif

#endif /* INFERENTIAL_MODEL_H */
