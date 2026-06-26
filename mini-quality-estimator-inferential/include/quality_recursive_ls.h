/**
 * @file quality_recursive_ls.h
 * @brief Recursive Least Squares (RLS) and related algorithms for adaptive quality models.
 *
 * Level: L5 Algorithms — Recursive parameter estimation for time-varying quality models.
 * Reference:
 *   Ljung, L. (1999) "System Identification: Theory for the User" Ch. 11 — RLS and variants
 *   Astrom, Wittenmark (2008) "Adaptive Control" Ch. 3 — RLS with forgetting factor
 *   Qin, S.J. (1998) "Recursive PLS algorithms for adaptive data modeling"
 *   Dayal, MacGregor (1997) "Recursive exponentially weighted PLS" — J. Chemometrics, 11(1)
 *
 * Course mapping:
 *   MIT 2.171: Digital Control — recursive identification, RLS
 *   Stanford ENGR205: Adaptive process models
 *   Purdue ME 575: Industrial adaptive control
 *   Tsinghua: 过程控制工程 — 在线系统辨识
 */

#ifndef QUALITY_RECURSIVE_LS_H
#define QUALITY_RECURSIVE_LS_H

#include "quality_estimator_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
 * L5: Classical Recursive Least Squares (RLS)
 *===========================================================================*/

/**
 * @brief Recursive Least Squares (RLS) estimator for linear quality models.
 *
 * Estimates theta in the linear model: y(k) = phi(k)^T * theta + e(k)
 *
 * Standard RLS update equations:
 *   K(k) = P(k-1) * phi(k) / (lambda + phi(k)^T * P(k-1) * phi(k))
 *   theta(k) = theta(k-1) + K(k) * (y(k) - phi(k)^T * theta(k-1))
 *   P(k) = (1/lambda) * (I - K(k) * phi(k)^T) * P(k-1)
 *
 * where lambda is the forgetting factor (0 < lambda ≤ 1).
 *   lambda = 1  → no forgetting (infinite memory)
 *   lambda < 1  → older data exponentially weighted down
 *   Typical: lambda = 0.95..0.995 for slow-drifting processes
 *
 * Complexity per update: O(n_params^2) where n_params = number of regressors.
 * For quality estimation, n_params is typically 5-30, giving microsecond updates.
 */
typedef struct {
    int      n_params;         /**< Number of parameters to estimate */
    int      n_regressors;     /**< Number of regressors (equals n_params for RLS) */
    double   lambda;           /**< Forgetting factor (0 < lambda ≤ 1) */
    double  *theta;            /**< Parameter estimate [n_params] */
    double  *P;                /**< Covariance matrix [n_params × n_params] */
    double  *K;                /**< Gain vector [n_params] */
    double  *temp;             /**< Scratch vector [n_params] */
    double  *temp_mat;         /**< Scratch matrix [n_params × n_params] */
    double   delta;            /**< Initial P = delta * I (large = high uncertainty) */
    int      n_updates;        /**< Number of updates applied */
} rls_estimator_t;

/**
 * @brief Allocate and initialize an RLS estimator.
 *
 * @param rls      Uninitialized RLS estimator
 * @param n_params Number of parameters (regressors)
 * @param lambda   Forgetting factor
 * @param delta    Initial covariance diagonal (e.g., 100.0 for high uncertainty)
 */
void rls_alloc(rls_estimator_t *rls, int n_params, double lambda, double delta);

/**
 * @brief Free RLS estimator memory.
 */
void rls_free(rls_estimator_t *rls);

/**
 * @brief Update RLS estimate with a new data pair.
 *
 * @param rls   RLS estimator (updated in-place)
 * @param phi   Regressor vector [n_params]
 * @param y     Measured output (lab quality value)
 *
 * Theorem: Under persistent excitation (phi(k) spans the parameter space),
 * theta converges to the true parameters as k → ∞.
 * Convergence rate: O(1/k) for lambda=1, exponential with lambda<1.
 */
void rls_update(rls_estimator_t *rls, const double *phi, double y);

/**
 * @brief Predict output given regressor vector using current parameters.
 *
 * y_pred = phi^T * theta
 *
 * @param rls  RLS estimator
 * @param phi  Regressor vector [n_params]
 * @return     Predicted output
 */
double rls_predict(const rls_estimator_t *rls, const double *phi);

/**
 * @brief Get the current parameter estimates.
 *
 * @param rls   RLS estimator
 * @param theta [out] Parameter vector [n_params]
 */
void rls_get_parameters(const rls_estimator_t *rls, double *theta);

/**
 * @brief Get the parameter covariance matrix (for uncertainty quantification).
 *
 * @param rls  RLS estimator
 * @param P    [out] Covariance matrix [n_params × n_params]
 */
void rls_get_covariance(const rls_estimator_t *rls, double *P);

/**
 * @brief Reset the RLS estimator covariance to delta * I (for restart).
 *
 * Retains current parameter estimates but resets P to restart adaptation.
 * Useful after a known process change (e.g., feedstock switch).
 *
 * @param rls  RLS estimator
 */
void rls_reset_covariance(rls_estimator_t *rls);

/*===========================================================================
 * L5: RLS with Variable Forgetting Factor
 *===========================================================================*/

/**
 * @brief RLS with adaptive (variable) forgetting factor.
 *
 * The forgetting factor is adjusted based on the prediction error:
 * when prediction error is small, lambda increases toward 1 (more memory);
 * when prediction error is large, lambda decreases to track faster.
 *
 * lambda(k) = lambda_min + (1 - lambda_min) * exp(-|e(k)| / sigma_e)
 *
 * Reference: Fortescue, Kershenbaum, Ydstie (1981) "Implementation of
 *            self-tuning regulators with variable forgetting factors" — Automatica, 17(6).
 */
typedef struct {
    rls_estimator_t rls;         /**< Underlying RLS estimator */
    double   lambda_min;         /**< Minimum forgetting factor */
    double   lambda_max;         /**< Maximum forgetting factor (≤ 1) */
    double   sigma_e;            /**< Expected prediction error magnitude */
    double   current_lambda;     /**< Current adapted forgetting factor */
    double   error_variance;     /**< Running estimate of prediction error variance */
} rls_vff_t;

/**
 * @brief Allocate and initialize an RLS with variable forgetting factor.
 *
 * @param rls_vff   Uninitialized VFF RLS
 * @param n_params  Number of parameters
 * @param lambda_min Minimum forgetting factor (e.g., 0.90)
 * @param lambda_max Maximum forgetting factor (e.g., 0.999)
 * @param sigma_e   Expected prediction error std dev
 * @param delta     Initial covariance
 */
void rls_vff_alloc(rls_vff_t *rls_vff, int n_params,
                   double lambda_min, double lambda_max,
                   double sigma_e, double delta);

/**
 * @brief Free VFF RLS memory.
 */
void rls_vff_free(rls_vff_t *rls_vff);

/**
 * @brief Update VFF RLS with new data, adapting the forgetting factor.
 */
void rls_vff_update(rls_vff_t *rls_vff, const double *phi, double y);

/**
 * @brief Predict output with current VFF RLS parameters.
 */
double rls_vff_predict(const rls_vff_t *rls_vff, const double *phi);

/**
 * @brief Get the current adapted forgetting factor.
 */
double rls_vff_get_lambda(const rls_vff_t *rls_vff);

/*===========================================================================
 * L5: Directional Forgetting RLS
 *===========================================================================*/

/**
 * @brief RLS with directional forgetting to prevent estimator windup.
 *
 * Standard exponential forgetting can cause "estimator windup" (also called
 * "covariance blowup") when the regressor signal is not persistently exciting.
 * Directional forgetting only forgets in the direction of incoming data,
 * preventing P from growing without bound.
 *
 * Reference: Kulhavy, R. (1987) "Restricted exponential forgetting in real-time
 *            identification" — Automatica, 23(5), 589-600.
 *            Bittanti, Bolzern, Campi (1990) "Convergence and exponential convergence
 *            of identification algorithms with directional forgetting factor" — Automatica, 26(5).
 */
typedef struct {
    rls_estimator_t rls;         /**< Underlying RLS estimator */
    double   lambda;             /**< Base forgetting factor */
    double   epsilon;            /**< Conditioning factor (> 0, typically 1e-3) */
    double   info_content;       /**< Current information content measure */
} rls_directional_t;

/**
 * @brief Allocate RLS with directional forgetting.
 */
void rls_directional_alloc(rls_directional_t *rls_dir, int n_params,
                           double lambda, double epsilon, double delta);

/**
 * @brief Free directional forgetting RLS.
 */
void rls_directional_free(rls_directional_t *rls_dir);

/**
 * @brief Update with directional forgetting.
 */
void rls_directional_update(rls_directional_t *rls_dir, const double *phi, double y);

/**
 * @brief Predict output.
 */
double rls_directional_predict(const rls_directional_t *rls_dir, const double *phi);

/*===========================================================================
 * L5: Recursive Partial Least Squares (RPLS)
 *===========================================================================*/

/**
 * @brief Recursive PLS for adaptive multi-input quality estimation.
 *
 * RPLS updates the PLS model online as new lab samples arrive, without
 * requiring complete retraining. This handles:
 *   - Catalyst deactivation (slow drift)
 *   - Feedstock changes (sudden shifts)
 *   - Seasonal effects (cyclic changes)
 *
 * Steps for each new sample pair (x, y):
 *   1. Mean-center and scale (x, y) using running statistics
 *   2. Update X and Y block scores incrementally
 *   3. Update the PLS regression coefficients
 *
 * Reference: Qin, S.J. (1998) "Recursive PLS algorithms for adaptive data modeling"
 *            CACE, 22(4), 503-514.
 */
typedef struct {
    pls_model_t model;           /**< Current PLS model */
    int         n_inputs;        /**< Number of input (X) variables */
    int         n_outputs;       /**< Number of output (Y) variables */
    int         n_latent;        /**< Number of latent variables */

    /* Running statistics for updating means and standard deviations */
    int         n_samples;       /**< Number of samples processed */
    double      x_sum[QEST_MAX_INPUT_VARS];    /**< Running sum of X for mean */
    double      x_sum_sq[QEST_MAX_INPUT_VARS]; /**< Running sum of X^2 for std */
    double      y_sum[8];                     /**< Running sum of Y */
    double      y_sum_sq[8];                  /**< Running sum of Y^2 */

    /* Recursive update accumulators */
    double      XTX[QEST_MAX_INPUT_VARS][QEST_MAX_INPUT_VARS]; /**< X^T * X accumulator */
    double      XTY[QEST_MAX_INPUT_VARS][8];                   /**< X^T * Y accumulator */

    /* Forgetting factor */
    double      lambda;          /**< Forgetting factor for recursive updates */
} rpls_model_t;

/**
 * @brief Initialize a Recursive PLS model.
 *
 * @param rpls      Uninitialized RPLS model
 * @param n_inputs  Number of X variables
 * @param n_outputs Number of Y variables
 * @param n_latent  Number of latent factors
 * @param lambda    Forgetting factor (1 = no forgetting, < 1 = exponential forgetting)
 */
void rpls_init(rpls_model_t *rpls, int n_inputs, int n_outputs, int n_latent, double lambda);

/**
 * @brief Perform one recursive PLS update with a new sample pair.
 *
 * @param rpls  RPLS model (updated in-place)
 * @param x     Input vector [n_inputs]
 * @param y     Output (quality) vector [n_outputs]
 */
void rpls_update(rpls_model_t *rpls, const double *x, const double *y);

/**
 * @brief Predict output using the current RPLS model.
 *
 * @param rpls     RPLS model
 * @param x        Input vector [n_inputs]
 * @param y_pred   [out] Predicted quality [n_outputs]
 */
void rpls_predict(const rpls_model_t *rpls, const double *x, double *y_pred);

/**
 * @brief Get the RPLS model's explained variance for Y.
 *
 * @param rpls  RPLS model
 * @param var_y [out] Cumulative Y variance explained per latent factor [n_latent]
 */
void rpls_get_explained_variance(const rpls_model_t *rpls, double *var_y);

/*===========================================================================
 * L5: Weighted Least Squares (WLS) — alternative to RLS with fixed window
 *===========================================================================*/

/**
 * @brief Weighted Least Squares estimator using exponential weighting.
 *
 * Unlike RLS which updates recursively, WLS stores a window of recent
 * samples and solves the weighted normal equations. Useful when the
 * window size should be controlled explicitly.
 *
 * min_theta  sum_{i=1}^{N} w_i * (y_i - phi_i^T * theta)^2
 *
 * where w_i = lambda^(N-i) for exponential weighting.
 */
typedef struct {
    int      n_params;          /**< Number of parameters */
    int      window_size;       /**< Number of samples in window */
    int      current_index;     /**< Write position in sample buffer */
    int      n_stored;          /**< Number of valid samples stored */
    double  *phi_buffer;        /**< Regressor buffer [window_size × n_params] */
    double  *y_buffer;          /**< Output buffer [window_size] */
    double  *weights;           /**< Weight buffer [window_size] */
    double   lambda;            /**< Exponential weighting factor */
    double  *theta;             /**< Current parameter estimate [n_params] */
    double  *P;                 /**< Parameter covariance [n_params × n_params] */
    double  *temp_mat;          /**< Scratch [n_params × n_params] */
    double  *temp_vec;          /**< Scratch [n_params] */
} wls_window_t;

/**
 * @brief Allocate a windowed WLS estimator.
 */
void wls_window_alloc(wls_window_t *wls, int n_params, int window_size, double lambda);

/**
 * @brief Free windowed WLS memory.
 */
void wls_window_free(wls_window_t *wls);

/**
 * @brief Add a sample and recompute WLS estimate.
 */
void wls_window_update(wls_window_t *wls, const double *phi, double y);

/**
 * @brief Predict output.
 */
double wls_window_predict(const wls_window_t *wls, const double *phi);

/**
 * @brief Get parameters.
 */
void wls_window_get_parameters(const wls_window_t *wls, double *theta);

/*===========================================================================
 * L2: Instrument Variable (IV) method for noisy regressor data
 *===========================================================================*/

/**
 * @brief Instrumental Variable estimator for quality model identification
 *        with noisy process measurements.
 *
 * OLS assumes regressors are noise-free, which is violated when process
 * variables (temperatures, flows) contain measurement noise. IV method
 * uses lagged values as instruments to obtain consistent estimates.
 *
 * Reference: Söderström, Stoica (1983) "Instrumental Variable Methods
 *            for System Identification" — Springer.
 */
typedef struct {
    int      n_params;          /**< Number of parameters */
    int      n_iv;              /**< Number of instrumental variables */
    int      n_lag;             /**< Lag for generating instruments from regressors */
    double  *phi_buffer;        /**< Regressor history [n_lag × n_params] */
    double  *y_buffer;          /**< Output history [n_lag] */
    int      buffer_pos;        /**< Write position */
    int      buffer_filled;     /**< Flag: buffer full */
    double  *theta;             /**< Parameter estimate [n_params] */
    double  *P_iv;              /**< Parameter covariance [n_params × n_params] */
    double  *temp_mat;          /**< Scratch [n_params × n_params] */
    double  *temp_vec;          /**< Scratch [n_params] */
} iv_estimator_t;

/**
 * @brief Allocate an IV estimator.
 */
void iv_estimator_alloc(iv_estimator_t *iv, int n_params, int n_lag);

/**
 * @brief Free IV estimator.
 */
void iv_estimator_free(iv_estimator_t *iv);

/**
 * @brief Update IV estimate with new data.
 */
void iv_estimator_update(iv_estimator_t *iv, const double *phi, double y);

/**
 * @brief Predict with IV parameter estimates.
 */
double iv_estimator_predict(const iv_estimator_t *iv, const double *phi);

#ifdef __cplusplus
}
#endif

#endif /* QUALITY_RECURSIVE_LS_H */
