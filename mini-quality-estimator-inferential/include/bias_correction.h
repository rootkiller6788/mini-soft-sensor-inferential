/**
 * @file bias_correction.h
 * @brief Bias correction and lab sample update strategies for inferential quality estimators.
 *
 * Level: L2 Core Concepts + L3 Engineering Structures + L5 Algorithms
 * Reference:
 *   Seborg, Edgar, Mellichamp (2016) "Process Dynamics and Control" Section 20.4
 *   Qin, S.J. (1998) "Recursive PLS algorithms for adaptive data modeling" — CACE, 22(4)
 *   Kadlec, Gabrys (2011) "Review of adaptation mechanisms for data-driven soft sensors"
 *   Montgomery, D.C. (2019) "Introduction to Statistical Quality Control" Ch. 6-7
 *
 * Course mapping:
 *   Stanford ENGR205: Bias tracking in inferential control
 *   Purdue ME 575: Operator bias update procedures
 *   Tsinghua: 软测量模型在线校正
 *   CMU 24-677: Recursive parameter estimation
 */

#ifndef BIAS_CORRECTION_H
#define BIAS_CORRECTION_H

#include "quality_estimator_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
 * L2: Simple Bias Correction (Additive / Multiplicative)
 *===========================================================================*/

/**
 * @brief State for additive bias correction.
 *
 * y_corrected = y_model + bias
 *
 * This is the most common industrial practice. The bias is updated whenever
 * a lab sample becomes available, typically using EWMA filtering to avoid
 * overreacting to lab measurement noise.
 */
typedef struct {
    double        bias;              /**< Current additive bias value */
    double        bias_previous;     /**< Previous bias value (for rate-of-change check) */
    double        ewma_gain;         /**< EWMA filter gain (0..1, typical 0.1..0.3) */
    double        bias_min;          /**< Minimum allowed bias (for sanity checking) */
    double        bias_max;          /**< Maximum allowed bias (for sanity checking) */
    double        max_bias_change;   /**< Maximum allowed bias change per update */
    int           n_updates;         /**< Number of bias updates applied since init */
    double        lab_history[QEST_MAX_LAB_HISTORY]; /**< Recent lab values for statistics */
    double        resid_history[QEST_MAX_LAB_HISTORY]; /**< Recent residuals */
    int           history_index;      /**< Write position in history buffers */
    int           history_count;      /**< Number of valid entries in history */
    double        bias_stddev;        /**< Standard deviation of recent bias estimates */
    int           is_initialized;     /**< Flag: sufficient history for statistics */
} bias_additive_t;

/**
 * @brief Initialize an additive bias corrector.
 *
 * @param bc     Uninitialized bias corrector
 * @param gain   EWMA filter gain (0 < gain ≤ 1, typical 0.15)
 * @param b_min  Minimum sane bias value
 * @param b_max  Maximum sane bias value
 * @param max_chg Maximum bias change per update (0 = no limit)
 */
void bias_additive_init(bias_additive_t *bc, double gain,
                        double b_min, double b_max, double max_chg);

/**
 * @brief Update additive bias using new lab measurement.
 *
 * residual = y_lab - y_model
 * bias_raw = residual  (for additive correction)
 * bias_new = bias_old + ewma_gain * (bias_raw - bias_old)
 *
 * @param bc      Bias corrector state (updated in-place)
 * @param y_lab   Lab-measured quality value
 * @param y_model Model-predicted quality value at the lab sample time
 * @return        Updated bias value
 *
 * Guard condition: if |bias_raw - bc->bias| > max_bias_change, the update
 * is rejected and the previous bias is retained (lab outlier protection).
 */
double bias_additive_update(bias_additive_t *bc, double y_lab, double y_model);

/**
 * @brief Apply additive bias correction to a model prediction.
 *
 * y_corrected = y_model + bias
 *
 * @param bc       Bias corrector
 * @param y_model  Raw model prediction
 * @return         Bias-corrected quality estimate
 */
double bias_additive_correct(const bias_additive_t *bc, double y_model);

/**
 * @brief Get the current additive bias value.
 *
 * @param bc  Bias corrector
 * @return    Current bias
 */
double bias_additive_get(const bias_additive_t *bc);

/**
 * @brief Reset the additive bias corrector to zero.
 *
 * @param bc  Bias corrector
 */
void bias_additive_reset(bias_additive_t *bc);

/*===========================================================================
 * L2: Multiplicative Bias Correction
 *===========================================================================*/

/**
 * @brief State for multiplicative (scale-factor) bias correction.
 *
 * y_corrected = y_model * (1 + bias)
 *
 * Useful when model error scales with the predicted value (e.g., percent error
 * is more consistent than absolute error). Common in composition estimation
 * where errors are proportional to concentration.
 */
typedef struct {
    double        bias;              /**< Current multiplicative bias factor */
    double        ewma_gain;         /**< EWMA filter gain */
    double        bias_min;          /**< Min multiplicative bias (> -1 typically, -1 = zero output) */
    double        bias_max;          /**< Max multiplicative bias */
    double        max_bias_change;   /**< Max change per update */
    int           n_updates;         /**< Update counter */
    int           is_initialized;    /**< Initialization flag */
} bias_multiplicative_t;

/**
 * @brief Initialize a multiplicative bias corrector.
 */
void bias_mult_init(bias_multiplicative_t *bc, double gain,
                    double b_min, double b_max, double max_chg);

/**
 * @brief Update multiplicative bias using new lab measurement.
 *
 * bias_raw = (y_lab / y_model) - 1  (if y_model != 0)
 * bias_new = bias_old + ewma_gain * (bias_raw - bias_old)
 *
 * @param bc      Bias corrector
 * @param y_lab   Lab measurement
 * @param y_model Model prediction
 * @return        Updated bias factor
 */
double bias_mult_update(bias_multiplicative_t *bc, double y_lab, double y_model);

/**
 * @brief Apply multiplicative bias correction.
 *
 * y_corrected = y_model * (1 + bias)
 *
 * @param bc       Bias corrector
 * @param y_model  Raw prediction
 * @return         Corrected estimate
 */
double bias_mult_correct(const bias_multiplicative_t *bc, double y_model);

/**
 * @brief Get the current multiplicative bias factor.
 */
double bias_mult_get(const bias_multiplicative_t *bc);

/**
 * @brief Reset multiplicative bias to zero (no correction).
 */
void bias_mult_reset(bias_multiplicative_t *bc);

/*===========================================================================
 * L3: EWMA-Filtered Bias with Trend Monitoring
 *===========================================================================*/

/**
 * @brief Enhanced EWMA bias corrector with trend detection and adaptation.
 *
 * Uses a dual EWMA structure:
 *   bias_filtered = lambda * residual + (1-lambda) * bias_filtered
 *   trend_filtered = beta * (bias_filtered - bias_filtered_prev) + (1-beta) * trend_filtered
 *
 * When the trend exceeds a threshold, the filter gain can be temporarily
 * increased to track rapid changes (e.g., after feedstock change).
 *
 * This is patterned after the Holt-Winters forecasting method adapted
 * for bias tracking.
 */
typedef struct {
    double        bias;              /**< Current filtered bias */
    double        trend;             /**< Current bias trend (rate of change) */
    double        lambda;            /**< EWMA gain for bias (0..1) */
    double        beta;              /**< EWMA gain for trend (0..1) */
    double        lambda_boost;      /**< Boosted lambda when trend detected */
    double        trend_threshold;   /**< Trend magnitude that triggers boost */
    double        bias_min;          /**< Bias clamp minimum */
    double        bias_max;          /**< Bias clamp maximum */
    int           n_updates;         /**< Update counter */
    int           in_boost;          /**< Flag: currently in boosted adaptation mode */
    double        prev_bias;         /**< Previous bias for trend computation */
} bias_ewma_trend_t;

/**
 * @brief Initialize an EWMA-trend bias corrector.
 *
 * @param bc       Uninitialized corrector
 * @param lambda   EWMA gain for bias level
 * @param beta     EWMA gain for trend
 * @param boost    Boosted lambda when rapid change detected
 * @param trend_th Threshold for trend-based boost
 */
void bias_ewma_trend_init(bias_ewma_trend_t *bc, double lambda, double beta,
                          double boost, double trend_th);

/**
 * @brief Update bias with trend-adaptive EWMA.
 *
 * @param bc      Bias corrector (updated in-place)
 * @param y_lab   Lab measurement
 * @param y_model Model prediction
 * @return        Updated bias value
 */
double bias_ewma_trend_update(bias_ewma_trend_t *bc, double y_lab, double y_model);

/**
 * @brief Correct model prediction using current trend-aware bias.
 *
 * @param bc       Bias corrector
 * @param y_model  Raw prediction
 * @return         Corrected estimate
 */
double bias_ewma_trend_correct(const bias_ewma_trend_t *bc, double y_model);

/**
 * @brief Get the current bias trend magnitude.
 *
 * @param bc  Bias corrector
 * @return    Trend (units of quality per update interval)
 */
double bias_ewma_trend_get_trend(const bias_ewma_trend_t *bc);

/*===========================================================================
 * L5: CUSUM-Triggered Bias Update
 *===========================================================================*/

/**
 * @brief Cumulative Sum (CUSUM) detector for bias drift triggering.
 *
 * CUSUM detects small but persistent shifts in the model bias:
 *   S_hi(k) = max(0, S_hi(k-1) + residual(k) - mu_0 - K)
 *   S_lo(k) = max(0, S_lo(k-1) + mu_0 - residual(k) - K)
 *
 * When S_hi > H or S_lo > H, a bias shift is detected and the bias
 * is recalculated from recent lab samples.
 *
 * Reference: Page, E.S. (1954) "Continuous inspection schemes" — Biometrika, 41(1).
 *            Montgomery, D.C. (2019) "Statistical Quality Control" Ch. 7.
 */
typedef struct {
    double        cusum_hi;          /**< Upper CUSUM statistic */
    double        cusum_lo;          /**< Lower CUSUM statistic */
    double        mu_0;              /**< Target mean residual (typically 0) */
    double        K;                 /**< Reference value / slack (typically 0.5*delta) */
    double        H;                 /**< Decision interval / threshold */
    double        drift_magnitude;   /**< Minimum drift magnitude to detect (delta) */
    int           alarm_hi;          /**< Flag: upper CUSUM triggered */
    int           alarm_lo;          /**< Flag: lower CUSUM triggered */
    double        last_residual;     /**< Most recent residual value */
    int           n_samples;         /**< Samples processed since last reset */
} cusum_detector_t;

/**
 * @brief Initialize a CUSUM drift detector.
 *
 * @param cd    Uninitialized CUSUM detector
 * @param delta Minimum drift to detect (in units of residual std dev)
 * @param H     Decision interval (typically 4 or 5)
 */
void cusum_init(cusum_detector_t *cd, double delta, double H);

/**
 * @brief Process a new residual through the CUSUM detector.
 *
 * @param cd       CUSUM detector (updated in-place)
 * @param residual New model-vs-lab residual (y_lab - y_model)
 * @param sigma    Standard deviation of residuals (for scaling)
 * @return         1 if either CUSUM alarm triggered, 0 otherwise
 */
int cusum_update(cusum_detector_t *cd, double residual, double sigma);

/**
 * @brief Reset CUSUM statistics after handling an alarm.
 *
 * @param cd  CUSUM detector
 */
void cusum_reset(cusum_detector_t *cd);

/*===========================================================================
 * L5: Kalman-Based Bias State Estimation
 *===========================================================================*/

/**
 * @brief Bias state estimator using a reduced-order Kalman filter.
 *
 * Models the bias as a random walk:
 *   bias(k+1) = bias(k) + w_bias(k)    (w_bias ~ N(0, q_bias))
 *   y_lab = y_model + bias + v_lab     (v_lab ~ N(0, r_lab))
 *
 * This is a scalar Kalman filter (1 state) specialized for bias tracking.
 * It automatically balances the confidence in the model vs. the lab.
 *
 * Advantage over EWMA: automatically adjusts the effective gain based on
 * the relative uncertainty of model predictions vs lab measurements.
 */
typedef struct {
    double        bias;              /**< Estimated bias state */
    double        P;                 /**< Bias estimation error variance */
    double        q_bias;            /**< Process noise variance (bias random walk) */
    double        r_lab;             /**< Lab measurement noise variance */
    double        model_variance;    /**< Model prediction uncertainty */
    int           is_initialized;    /**< Flag: filter initialized */
    int           n_updates;         /**< Update counter */
} bias_kalman_t;

/**
 * @brief Initialize a Kalman-based bias estimator.
 *
 * @param bk     Uninitialized bias estimator
 * @param q_bias Process noise variance for bias random walk (e.g., 0.001)
 * @param r_lab  Lab measurement noise variance
 * @param bias_0 Initial bias estimate (typically 0)
 * @param P0     Initial bias uncertainty (large = 100.0 for first sample)
 */
void bias_kalman_init(bias_kalman_t *bk, double q_bias, double r_lab,
                      double bias_0, double P0);

/**
 * @brief Update Kalman bias estimate with new lab measurement.
 *
 * Predict: bias_prior = bias, P_prior = P + q_bias
 * Update:  K = P_prior / (P_prior + r_lab)
 *          bias = bias_prior + K * (y_lab - y_model - bias_prior)
 *          P = (1 - K) * P_prior
 *
 * @param bk      Kalman bias estimator (updated in-place)
 * @param y_lab   Lab measurement
 * @param y_model Model prediction
 * @return        Updated bias estimate
 */
double bias_kalman_update(bias_kalman_t *bk, double y_lab, double y_model);

/**
 * @brief Correct model prediction using Kalman bias estimate.
 *
 * @param bk       Bias estimator
 * @param y_model  Raw prediction
 * @return         Corrected estimate with uncertainty bounds
 */
double bias_kalman_correct(const bias_kalman_t *bk, double y_model);

/**
 * @brief Get bias estimation uncertainty (standard deviation).
 *
 * @param bk  Bias estimator
 * @return    sqrt(P) — bias uncertainty
 */
double bias_kalman_get_uncertainty(const bias_kalman_t *bk);

/*===========================================================================
 * L3: Lab Sample Validation and Rejection
 *===========================================================================*/

/**
 * @brief Validate a new lab sample before using it for bias update.
 *
 * Performs a sequence of checks:
 *   1. Range check: y_lab within [y_min, y_max] physically possible?
 *   2. Rate-of-change check: |y_lab - y_lab_prev| / dt within limits?
 *   3. Model-consistency check: |y_lab - y_model| < N * sigma_model?
 *   4. Lab precision: is lab_stddev reasonable (< threshold)?
 *
 * @param y_lab             New lab measurement
 * @param y_model           Simultaneous model prediction
 * @param y_lab_prev        Previous lab measurement
 * @param dt                Time since previous lab sample (hours)
 * @param lab_stddev        Reported lab measurement std deviation
 * @param y_min             Physical lower bound for this quality variable
 * @param y_max             Physical upper bound
 * @param max_rate          Maximum physically possible rate of change
 * @param sigma_multiplier  Model-consistency sigma multiplier (e.g., 4.0)
 * @param model_sigma       Standard deviation of model residuals
 * @param max_lab_stddev    Maximum acceptable lab measurement std dev
 * @return                  LAB_QUALITY_GOOD, SUSPECT, or BAD
 */
lab_quality_t lab_sample_validate(double y_lab, double y_model,
                                  double y_lab_prev, double dt,
                                  double lab_stddev,
                                  double y_min, double y_max,
                                  double max_rate,
                                  double sigma_multiplier, double model_sigma,
                                  double max_lab_stddev);

/*===========================================================================
 * L3: Bias Correction Dispatcher
 *===========================================================================*/

/**
 * @brief Unified bias correction context that wraps all strategies.
 *
 * This dispatches to the appropriate strategy based on configuration,
 * allowing runtime switching of bias correction methods.
 */
typedef struct {
    bias_strategy_t       active_strategy; /**< Currently active bias strategy */
    bias_additive_t       additive;        /**< Additive bias state */
    bias_multiplicative_t multiplicative;  /**< Multiplicative bias state */
    bias_ewma_trend_t     ewma_trend;      /**< EWMA + trend state */
    bias_kalman_t         kalman;          /**< Kalman bias state */
    cusum_detector_t      cusum;           /**< CUSUM drift detector */
    qest_config_t         config;          /**< Reference to estimator config */

    /** Per-strategy performance tracking */
    double        additive_mse;            /**< MSE with additive correction */
    double        multiplicative_mse;      /**< MSE with multiplicative correction */
    double        kalman_mse;              /**< MSE with Kalman correction */
} bias_context_t;

/**
 * @brief Initialize the unified bias correction context.
 *
 * @param ctx    Uninitialized bias context
 * @param config Estimator configuration
 */
void bias_context_init(bias_context_t *ctx, const qest_config_t *config);

/**
 * @brief Process a new lab measurement through the active bias strategy.
 *
 * @param ctx     Bias context (updated in-place)
 * @param y_lab   Lab measurement
 * @param y_model Model prediction at lab sample time
 * @return        LAB_QUALITY_GOOD if sample was accepted and used, BAD if rejected
 */
lab_quality_t bias_context_update(bias_context_t *ctx, double y_lab, double y_model);

/**
 * @brief Apply bias correction to a model prediction.
 *
 * @param ctx      Bias context
 * @param y_model  Raw model prediction
 * @return         Bias-corrected quality estimate
 */
double bias_context_correct(const bias_context_t *ctx, double y_model);

/**
 * @brief Get the current bias value from the active strategy.
 *
 * @param ctx  Bias context
 * @return     Current bias value
 */
double bias_context_get_bias(const bias_context_t *ctx);

/**
 * @brief Switch the active bias strategy at runtime.
 *
 * @param ctx       Bias context
 * @param strategy  New strategy to activate
 */
void bias_context_set_strategy(bias_context_t *ctx, bias_strategy_t strategy);

#ifdef __cplusplus
}
#endif

#endif /* BIAS_CORRECTION_H */
