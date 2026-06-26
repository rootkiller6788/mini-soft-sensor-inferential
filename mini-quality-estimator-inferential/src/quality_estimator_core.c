/**
 * @file quality_estimator_core.c
 * @brief Core implementation of the inferential quality estimator framework.
 *
 * Level: L1 Definitions + L2 Core Concepts + L3 Engineering Structures
 * Reference:
 *   Seborg, Edgar, Mellichamp (2016) "Process Dynamics and Control" Ch. 20
 *   Fortuna et al. (2007) "Soft Sensors for Monitoring and Control"
 */

#include "quality_estimator_types.h"
#include "inferential_model.h"
#include "kalman_quality.h"
#include "bias_correction.h"
#include "multi_rate_fusion.h"
#include "quality_recursive_ls.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*===========================================================================
 * L1: Timestamp Utilities
 *===========================================================================*/

/**
 * @brief Set a timestamp to current "wall clock" values.
 *
 * Complexity: O(1).
 */
void qest_timestamp_set(qest_timestamp_t *ts, int year, int month, int day,
                        int hour, int minute, double second)
{
    if (!ts) return;
    ts->year   = year;
    ts->month  = month;
    ts->day    = day;
    ts->hour   = hour;
    ts->minute = minute;
    ts->second = second;
}

/**
 * @brief Compute time difference in seconds between two timestamps.
 *
 * Simplified: assumes 30-day months (adequate for process control where
 * time differences are hours/days, not seconds-level precision).
 *
 * Complexity: O(1).
 */
double qest_timestamp_diff_seconds(const qest_timestamp_t *t1, const qest_timestamp_t *t2)
{
    if (!t1 || !t2) return 0.0;
    double days1 = t1->year * 365.25 + t1->month * 30.4375 + t1->day;
    double days2 = t2->year * 365.25 + t2->month * 30.4375 + t2->day;
    double secs1 = days1 * 86400.0 + t1->hour * 3600.0 + t1->minute * 60.0 + t1->second;
    double secs2 = days2 * 86400.0 + t2->hour * 3600.0 + t2->minute * 60.0 + t2->second;
    return secs1 - secs2;
}

/**
 * @brief Compare two timestamps.
 *
 * @return negative if t1 < t2, 0 if equal, positive if t1 > t2.
 *
 * Complexity: O(1).
 */
int qest_timestamp_compare(const qest_timestamp_t *t1, const qest_timestamp_t *t2)
{
    if (!t1 || !t2) return 0;
    if (t1->year != t2->year)   return t1->year - t2->year;
    if (t1->month != t2->month) return t1->month - t2->month;
    if (t1->day != t2->day)     return t1->day - t2->day;
    if (t1->hour != t2->hour)   return t1->hour - t2->hour;
    if (t1->minute != t2->minute) return t1->minute - t2->minute;
    if (t1->second < t2->second - 0.001) return -1;
    if (t1->second > t2->second + 0.001) return 1;
    return 0;
}

/*===========================================================================
 * L1: Process Variable Utilities
 *===========================================================================*/

/**
 * @brief Scale a process variable from engineering units to [0, 1].
 *
 * scaled = (raw - low) / (high - low)
 *
 * Returns 0.0 if high == low (degenerate range).
 *
 * Complexity: O(1).
 */
static double pv_scale_to_normalized(double raw, double low, double high)
{
    if (high <= low) return 0.0;
    double scaled = (raw - low) / (high - low);
    if (scaled < 0.0) scaled = 0.0;
    if (scaled > 1.0) scaled = 1.0;
    return scaled;
}

/**
 * @brief Update process variable with new raw measurement.
 *
 * Performs scaling, rate-of-change computation, and stuck-sensor detection.
 *
 * @param pv      Process variable state (updated in-place)
 * @param raw_val New raw measurement in engineering units
 *
 * Complexity: O(1).
 */
void pv_update(process_variable_t *pv, double raw_val)
{
    if (!pv) return;
    double prev_raw = pv->raw_value;
    pv->raw_value = raw_val;
    pv->scaled_value = pv_scale_to_normalized(raw_val, pv->low_range, pv->high_range);
    pv->rate_of_change = raw_val - prev_raw;

    /* Stuck sensor detection: zero change for consecutive updates */
    if (fabs(raw_val - prev_raw) < 1e-10) {
        pv->freeze_count++;
    } else {
        pv->freeze_count = 0;
    }
    if (pv->freeze_count > 60) {
        pv->is_faulted = 1;  /* Stuck for > 60 samples = fault */
    }
}

/**
 * @brief Initialize a process variable with range and tag.
 *
 * @param pv        Uninitialized process variable
 * @param tag_name  SCADA/DCS tag name
 * @param low       Instrument range low
 * @param high      Instrument range high
 * @param stddev    Sensor noise standard deviation
 *
 * Complexity: O(1).
 */
void pv_init(process_variable_t *pv, const char *tag_name,
             double low, double high, double stddev)
{
    if (!pv) return;
    memset(pv, 0, sizeof(process_variable_t));
    strncpy(pv->tag_name, tag_name, QEST_TAG_NAME_LEN - 1);
    pv->tag_name[QEST_TAG_NAME_LEN - 1] = '\0';
    pv->low_range  = low;
    pv->high_range = high;
    pv->sensor_stddev = stddev;
    pv->scaled_value = 0.0;
}

/*===========================================================================
 * L1: Quality Estimate Utilities
 *===========================================================================*/

/**
 * @brief Initialize a quality estimate with default values.
 *
 * @param est  Uninitialized estimate
 *
 * Complexity: O(1).
 */
void qest_init_estimate(quality_estimate_t *est)
{
    if (!est) return;
    memset(est, 0, sizeof(quality_estimate_t));
    est->predicted_value      = 0.0;
    est->bias_corrected_value = 0.0;
    est->prediction_variance  = 1.0;   /* High initial uncertainty */
    est->lower_bound_95       = -1.96;
    est->upper_bound_95       = 1.96;
    est->bias_current         = 0.0;
    est->bias_ewma            = 0.0;
    est->mode                 = QEST_MODE_STANDBY;
    est->health               = MAINT_OK;
    est->is_valid             = 0;
}

/**
 * @brief Compute confidence bounds for a quality estimate.
 *
 * lower = y - z * sigma
 * upper = y + z * sigma
 *
 * where sigma = sqrt(variance) and z is the z-score for the desired
 * confidence level (1.96 for 95%, 2.576 for 99%).
 *
 * @param est        Quality estimate (updated in-place)
 * @param z_score    Z-score for desired confidence (e.g., QEST_Z_SCORE_95)
 *
 * Complexity: O(1).
 */
void qest_compute_confidence_bounds(quality_estimate_t *est, double z_score)
{
    if (!est) return;
    double sigma = sqrt(est->prediction_variance);
    if (sigma < 0.0) sigma = 0.0;
    est->lower_bound_95 = est->bias_corrected_value - z_score * sigma;
    est->upper_bound_95 = est->bias_corrected_value + z_score * sigma;
}

/*===========================================================================
 * L2: Quality Estimator Configuration
 *===========================================================================*/

/**
 * @brief Initialize an estimator configuration with sensible defaults.
 *
 * Default configuration:
 *   - Model type: DATA_DRIVEN (most common in practice)
 *   - Data subtype: PLS
 *   - Bias strategy: EWMA filtered (robust, widely used)
 *   - Fast sample period: 10 seconds
 *   - Lab sample period: 4 hours
 *   - Bias filter gain: 0.15
 *   - Confidence: 95%
 *
 * @param config  Uninitialized configuration
 * @param name    Estimator name (e.g., "Distillate Composition")
 * @param tag     Quality tag name (e.g., "AI_TOP_COMP")
 * @param units   Engineering units (e.g., "wt%")
 * @param n_inputs Number of input process variables
 *
 * Complexity: O(1).
 */
void qest_config_init(qest_config_t *config, const char *name, const char *tag,
                      const char *units, int n_inputs)
{
    if (!config) return;
    memset(config, 0, sizeof(qest_config_t));

    if (name) {
        strncpy(config->estimator_name, name, QEST_MODEL_NAME_LEN - 1);
        config->estimator_name[QEST_MODEL_NAME_LEN - 1] = '\0';
    }
    if (tag) {
        strncpy(config->quality_tag, tag, QEST_TAG_NAME_LEN - 1);
        config->quality_tag[QEST_TAG_NAME_LEN - 1] = '\0';
    }
    if (units) {
        strncpy(config->units, units, 15);
        config->units[15] = '\0';
    }

    config->model_type    = QMODEL_DATA_DRIVEN;
    config->data_subtype  = DATA_MODEL_PLS;
    config->bias_strategy = BIAS_EWMA_FILTERED;
    config->n_input_vars  = n_inputs;
    config->n_lags        = 0;
    config->n_latent      = 2;

    config->fast_sample_period = 10.0;    /* 10 seconds */
    config->lab_sample_period  = 14400.0; /* 4 hours */
    config->bias_filter_gain   = 0.15;

    config->bias_warning_limit   = 2.0;   /* 2 sigma warning */
    config->bias_alarm_limit     = 3.0;   /* 3 sigma alarm */
    config->variance_alarm_limit = 9.0;   /* variance = 9 => sigma = 3 */
    config->cusum_threshold      = 5.0;
    config->outlier_sigma        = 4.0;   /* 4-sigma outlier rejection */

    config->require_input_validity   = 1;  /* Require all inputs valid */
    config->max_consecutive_bad_labs  = 3; /* Alarm after 3 bad lab samples */

    config->is_dynamic      = 0;  /* Static model by default */
    config->is_multivariate = 0;  /* Single output by default */
}

/*===========================================================================
 * L2: Complete Quality Estimator Instance
 *===========================================================================*/

/**
 * @brief The main quality estimator runtime structure.
 *
 * This is the central data structure that holds all sub-components:
 * model, bias corrector, performance metrics, and runtime state.
 */
struct quality_estimator {
    qest_config_t      config;           /**< Estimator configuration */
    qest_mode_t        mode;             /**< Current operating mode */
    quality_estimate_t last_estimate;    /**< Most recent quality estimate */

    /* Model components (active one based on config.model_type) */
    fpm_model_t        fpm;              /**< First-principles model */
    linear_model_t     linear;           /**< Linear regression model */
    pls_model_t        pls;              /**< PLS model */
    arx_model_t        arx;              /**< ARX dynamic model */
    ss_model_t         ss;               /**< State-space model */
    kalman_filter_t    kf;               /**< Kalman filter (for KF model type) */

    /* Bias correction */
    bias_context_t     bias_ctx;         /**< Bias correction context */

    /* Multi-rate fusion */
    multi_rate_context_t mr_ctx;         /**< Multi-rate context */

    /* Process variable storage */
    process_variable_t pvs[QEST_MAX_INPUT_VARS]; /**< Input process variables */
    int                n_active_pvs;     /**< Number of active process variables */

    /* Lab sample history */
    lab_sample_t       lab_history[QEST_MAX_LAB_HISTORY];
    int                lab_history_count;
    int                lab_history_index;

    /* Performance monitoring */
    qest_performance_t perf;             /**< Accumulated performance metrics */

    /* Runtime state */
    int                is_configured;    /**< Flag: configuration complete */
    int                n_steps;          /**< Total estimator cycles executed */
};

/**
 * @brief Allocate and initialize a complete quality estimator.
 *
 * Memory is allocated for the estimator and all sub-components.
 *
 * @return Pointer to initialized estimator, or NULL on allocation failure.
 *
 * Complexity: O(1) for fixed-size structures.
 */
quality_estimator_t *qest_alloc(void)
{
    quality_estimator_t *qest = (quality_estimator_t *)calloc(1, sizeof(quality_estimator_t));
    if (!qest) return NULL;
    qest->mode = QEST_MODE_STANDBY;
    qest->is_configured = 0;
    qest->n_steps = 0;
    qest_init_estimate(&qest->last_estimate);
    memset(&qest->perf, 0, sizeof(qest_performance_t));
    qest->perf.health = MAINT_OK;
    return qest;
}

/**
 * @brief Free a quality estimator and all its sub-components.
 *
 * @param qest  Estimator to free
 */
void qest_free(quality_estimator_t *qest)
{
    if (!qest) return;
    if (qest->config.model_type == QMODEL_KALMAN_FILTER) {
        ss_model_free(&qest->ss);
        kf_free(&qest->kf);
    }
    free(qest);
}

/**
 * @brief Configure a quality estimator with the given configuration.
 *
 * Sets up all sub-components based on the model type and bias strategy.
 *
 * @param qest    Estimator to configure
 * @param config  Configuration parameters
 * @return        0 on success, -1 on invalid configuration
 *
 * Complexity: O(n_inputs) for per-input setup.
 */
int qest_configure(quality_estimator_t *qest, const qest_config_t *config)
{
    if (!qest || !config) return -1;
    if (config->n_input_vars <= 0 || config->n_input_vars > QEST_MAX_INPUT_VARS)
        return -1;
    if (config->fast_sample_period <= 0.0)
        return -1;

    memcpy(&qest->config, config, sizeof(qest_config_t));
    qest->n_active_pvs = config->n_input_vars;

    /* Initialize bias correction context */
    bias_context_init(&qest->bias_ctx, config);

    /* Initialize multi-rate context */
    memset(&qest->mr_ctx, 0, sizeof(multi_rate_context_t));
    qest->mr_ctx.n_fast_streams = config->n_input_vars;
    qest->mr_ctx.n_slow_streams = config->is_multivariate ? 2 : 1;
    qest->mr_ctx.fast_period = config->fast_sample_period;
    qest->mr_ctx.slow_period = config->lab_sample_period;

    /* Initialize process variables with identity scaling (low=0, high=1)
     * so that raw = scaled by default. Users can override via pv_init. */
    for (int i = 0; i < config->n_input_vars; i++) {
        pv_init(&qest->pvs[i], "PV", 0.0, 1.0, 0.01);
    }

    qest->is_configured = 1;
    qest->mode = QEST_MODE_ACTIVE;
    return 0;
}

/**
 * @brief Configure a linear regression model for the quality estimator.
 *
 * This is the most common industrial data-driven model type.
 *
 * @param qest          Configured estimator
 * @param intercept     Model intercept (beta_0)
 * @param coefficients  Regression coefficients (beta_1..beta_n)
 * @param n_inputs      Number of inputs
 * @param r_squared     Training R-squared
 * @param rmse          Training RMSE
 * @return              0 on success, -1 on error
 */
int qest_set_linear_model(quality_estimator_t *qest, double intercept,
                          const double *coefficients, int n_inputs,
                          double r_squared, double rmse)
{
    if (!qest || !coefficients || n_inputs <= 0
        || n_inputs > QEST_MAX_INPUT_VARS)
        return -1;

    qest->linear.n_inputs   = n_inputs;
    qest->linear.intercept  = intercept;
    qest->linear.r_squared  = r_squared;
    qest->linear.rmse       = rmse;
    memcpy(qest->linear.coefficients, coefficients, n_inputs * sizeof(double));
    qest->config.model_type = QMODEL_DATA_DRIVEN;
    qest->config.data_subtype = DATA_MODEL_PLS;  /* But we use linear underneath */
    return 0;
}

/**
 * @brief Set process variable values for the current estimation cycle.
 *
 * @param qest    Estimator
 * @param inputs  Raw process variable values [n_inputs]
 * @param n       Number of values (must match n_input_vars in config)
 * @return        0 on success, -1 on error
 */
int qest_set_inputs(quality_estimator_t *qest, const double *inputs, int n)
{
    if (!qest || !inputs || n != qest->config.n_input_vars)
        return -1;

    for (int i = 0; i < n; i++) {
        pv_update(&qest->pvs[i], inputs[i]);
    }
    return 0;
}

/**
 * @brief Run one complete quality estimation cycle.
 *
 * This is the main runtime function. It:
 *   1. Validates all input process variables
 *   2. Evaluates the selected model type
 *   3. Applies bias correction
 *   4. Computes confidence bounds
 *   5. Updates performance metrics
 *
 * @param qest  Estimator (updated in-place)
 * @return      Quality estimate structure (also stored internally)
 *
 * Complexity: depends on model type:
 *   - Linear: O(n_inputs)
 *   - PLS: O(n_inputs * n_latent)
 *   - ARX: O(na + nb*n_inputs)
 *   - Kalman: O(n_states^3)
 */
const quality_estimate_t *qest_step(quality_estimator_t *qest)
{
    if (!qest || !qest->is_configured) return NULL;
    if (qest->mode != QEST_MODE_ACTIVE && qest->mode != QEST_MODE_BIAS_UPDATE)
        return &qest->last_estimate;

    /* Step 1: Validate inputs and build input vector */
    int all_valid = 1;
    double model_inputs[QEST_MAX_INPUT_VARS];
    for (int i = 0; i < qest->config.n_input_vars; i++) {
        /* Use raw values for model evaluation (models are trained on raw EU values) */
        model_inputs[i] = qest->pvs[i].raw_value;
        if (qest->pvs[i].is_faulted) all_valid = 0;
    }
    if (!all_valid && qest->config.require_input_validity) {
        qest->last_estimate.is_valid = 0;
        qest->mode = QEST_MODE_FAULT;
        return &qest->last_estimate;
    }

    /* Step 2: Evaluate model */
    double y_model = 0.0;
    double model_variance = 1.0;

    switch (qest->config.model_type) {
    case QMODEL_FIRST_PRINCIPLES:
        y_model = fpm_evaluate(&qest->fpm, model_inputs, qest->config.n_input_vars);
        model_variance = qest->config.variance_alarm_limit * 0.01;
        break;
    case QMODEL_DATA_DRIVEN: {
        /* Linear model prediction */
        y_model = qest->linear.intercept;
        for (int i = 0; i < qest->linear.n_inputs; i++) {
            y_model += qest->linear.coefficients[i] * model_inputs[i];
        }
        model_variance = qest->linear.rmse * qest->linear.rmse;
        break;
    }
    case QMODEL_KALMAN_FILTER: {
        kf_predict(&qest->kf, model_inputs);
        y_model = kf_get_quality(&qest->kf);
        model_variance = kf_get_quality_variance(&qest->kf);
        break;
    }
    case QMODEL_HYBRID_GREY:
        /* Hybrid: FPM base + linear correction */
        y_model = fpm_evaluate(&qest->fpm, model_inputs, qest->config.n_input_vars);
        break;
    default:
        y_model = 0.0;
        model_variance = 1.0;
        break;
    }

    /* Step 3: Apply bias correction */
    double y_corrected = bias_context_correct(&qest->bias_ctx, y_model);

    /* Step 4: Update estimate */
    qest->last_estimate.predicted_value      = y_model;
    qest->last_estimate.bias_corrected_value = y_corrected;
    qest->last_estimate.prediction_variance  = model_variance;
    qest->last_estimate.bias_current         = bias_context_get_bias(&qest->bias_ctx);
    qest->last_estimate.mode                 = qest->mode;
    qest->last_estimate.is_valid             = 1;

    /* Step 5: Compute confidence bounds */
    qest_compute_confidence_bounds(&qest->last_estimate, QEST_Z_SCORE_95);

    /* Step 6: Update performance counters */
    qest->perf.n_predictions++;
    qest->perf.current_bias = qest->last_estimate.bias_current;
    qest->n_steps++;

    return &qest->last_estimate;
}

/**
 * @brief Process a new lab measurement for bias update.
 *
 * This is called when a lab result becomes available (hours after sampling).
 * The estimator validates the sample, updates the bias, and records metrics.
 *
 * @param qest       Estimator (updated in-place)
 * @param lab_sample Lab measurement sample
 * @return           LAB_QUALITY_GOOD if accepted, BAD if rejected
 */
lab_quality_t qest_process_lab(quality_estimator_t *qest, const lab_sample_t *lab_sample)
{
    if (!qest || !lab_sample) return LAB_QUALITY_BAD;
    if (!qest->is_configured) return LAB_QUALITY_BAD;

    /* Store in history */
    if (qest->lab_history_count < QEST_MAX_LAB_HISTORY) {
        int idx = qest->lab_history_index;
        memcpy(&qest->lab_history[idx], lab_sample, sizeof(lab_sample_t));
        qest->lab_history_index = (idx + 1) % QEST_MAX_LAB_HISTORY;
        qest->lab_history_count++;
    }

    /* Temporarily switch mode for bias update */
    qest_mode_t prev_mode = qest->mode;
    qest->mode = QEST_MODE_BIAS_UPDATE;

    /* Get model prediction at lab sample time */
    /* In a real system, this would be looked up from stored predictions */
    double y_model = qest->last_estimate.predicted_value;

    /* Process through bias context */
    lab_quality_t result = bias_context_update(&qest->bias_ctx,
                                                lab_sample->measured_value, y_model);

    /* Update performance metrics */
    if (result == LAB_QUALITY_GOOD) {
        qest->perf.n_bias_updates++;
        double residual = lab_sample->measured_value - y_model;
        /* Running MSE update (Welford-like) */
        double n = (double)qest->perf.n_bias_updates;
        double delta = residual - qest->perf.mae;
        qest->perf.mae += delta / n;
        qest->perf.mse = ((n - 1.0) * qest->perf.mse + residual * residual) / n;
        qest->perf.rmse = sqrt(qest->perf.mse);
        qest->perf.bias_stddev = sqrt(qest->perf.mse - qest->perf.mae * qest->perf.mae);
        if (qest->perf.bias_stddev < 0.0) qest->perf.bias_stddev = 0.0;
        qest->perf.n_consecutive_bad_labs = 0;
    } else {
        qest->perf.n_lab_rejections++;
        qest->perf.n_consecutive_bad_labs++;
    }

    qest->mode = prev_mode;

    /* Update health based on consecutive rejections */
    if (qest->perf.n_consecutive_bad_labs > qest->config.max_consecutive_bad_labs) {
        qest->perf.health = MAINT_NEEDS_ATTENTION;
    }

    return result;
}

/**
 * @brief Get the current performance metrics.
 *
 * @param qest   Estimator
 * @param perf   [out] Performance metrics
 */
void qest_get_performance(const quality_estimator_t *qest, qest_performance_t *perf)
{
    if (!qest || !perf) return;
    memcpy(perf, &qest->perf, sizeof(qest_performance_t));
}

/**
 * @brief Get the most recent quality estimate.
 *
 * @param qest  Estimator
 * @return      Pointer to last estimate (NULL if not yet computed)
 */
const quality_estimate_t *qest_get_estimate(const quality_estimator_t *qest)
{
    if (!qest) return NULL;
    return &qest->last_estimate;
}

/**
 * @brief Set the estimator operating mode.
 *
 * @param qest  Estimator
 * @param mode  New operating mode
 */
void qest_set_mode(quality_estimator_t *qest, qest_mode_t mode)
{
    if (!qest) return;
    qest->mode = mode;
}

/**
 * @brief Get the current operating mode.
 *
 * @param qest  Estimator
 * @return      Current mode
 */
qest_mode_t qest_get_mode(const quality_estimator_t *qest)
{
    if (!qest) return QEST_MODE_STANDBY;
    return qest->mode;
}

/**
 * @brief Evaluate the linear model directly for a given input vector.
 *
 * y = beta_0 + sum_{i=1}^{n} beta_i * x_i
 *
 * This can be called without going through the full estimator framework.
 *
 * @param model    Linear regression model
 * @param x        Input vector [n_inputs]
 * @param n_inputs Number of inputs
 * @return         Predicted value
 */
double linear_model_evaluate(const linear_model_t *model, const double *x, int n_inputs)
{
    if (!model || !x || n_inputs <= 0) return 0.0;
    double y = model->intercept;
    int n = (n_inputs < model->n_inputs) ? n_inputs : model->n_inputs;
    for (int i = 0; i < n; i++) {
        y += model->coefficients[i] * x[i];
    }
    return y;
}

/**
 * @brief Evaluate PLS model for quality prediction.
 *
 * Steps:
 *   1. Center and scale inputs: x_scaled = (x - mean) / std
 *   2. Compute scores: t_a = x_scaled * w_a for each latent factor a
 *   3. Compute output: y_scaled = sum(t_a * q_a)
 *   4. Unscale output: y = y_scaled * y_std + y_mean
 *
 * @param model    PLS model
 * @param x        Input vector [n_inputs]
 * @param n_inputs Number of input variables
 * @param y_pred   [out] Predicted quality value
 *
 * Complexity: O(n_inputs * n_latent) per prediction.
 */
void pls_model_predict(const pls_model_t *model, const double *x, int n_inputs, double *y_pred)
{
    if (!model || !x || !y_pred || n_inputs <= 0) return;

    double x_scaled[QEST_MAX_INPUT_VARS];
    int n_x = (n_inputs < model->n_inputs) ? n_inputs : model->n_inputs;

    /* Center and scale */
    for (int i = 0; i < n_x; i++) {
        if (model->x_stds[i] > 1e-10) {
            x_scaled[i] = (x[i] - model->x_means[i]) / model->x_stds[i];
        } else {
            x_scaled[i] = x[i] - model->x_means[i];
        }
    }

    /* Compute prediction for each output */
    int n_out = model->n_outputs;
    if (n_out > 8) n_out = 8;

    for (int j = 0; j < n_out; j++) {
        double y_scaled = model->intercept[j];
        /* t = X * W */
        for (int a = 0; a < model->n_latent; a++) {
            double t_a = 0.0;
            for (int i = 0; i < n_x; i++) {
                t_a += x_scaled[i] * model->x_weights[i][a];
            }
            y_scaled += t_a * model->y_loadings[j][a];
        }
        /* Unscale */
        y_pred[j] = y_scaled * model->y_stds[j] + model->y_means[j];
    }
}

/**
 * @brief Compute the Mahalanobis distance from a sample to the PLS model center.
 *
 * D^2 = (x - mean)^T * Sigma^{-1} * (x - mean)
 *
 * Used for detecting extrapolation: a large Mahalanobis distance means the
 * current operating point is far from the training data, so the model
 * prediction may be unreliable.
 *
 * @param model    PLS model
 * @param x        Input sample [n_inputs]
 * @param n_inputs Number of inputs
 * @return         Squared Mahalanobis distance
 */
double pls_mahalanobis_distance(const pls_model_t *model, const double *x, int n_inputs)
{
    if (!model || !x) return 0.0;
    double d2 = 0.0;
    int n = (n_inputs < model->n_inputs) ? n_inputs : model->n_inputs;
    for (int i = 0; i < n; i++) {
        double diff = x[i] - model->x_means[i];
        if (model->x_stds[i] > 1e-10) {
            diff /= model->x_stds[i];
        }
        d2 += diff * diff;
    }
    return d2 / (double)n;
}

/**
 * @brief Compute Hotelling's T^2 statistic for PLS scores.
 *
 * T^2 = sum_{a=1}^{A} (t_a^2 / lambda_a)
 *
 * where lambda_a is the variance of the a-th score in training data.
 *
 * @param model          PLS model
 * @param t_scores       Latent variable scores [n_latent]
 * @param score_variance Variance of each score from training [n_latent]
 * @return               Hotelling's T^2 value
 */
double pls_hotelling_t2(const pls_model_t *model, const double *t_scores,
                        const double *score_variance)
{
    if (!model || !t_scores || !score_variance) return 0.0;
    double t2 = 0.0;
    for (int a = 0; a < model->n_latent; a++) {
        if (score_variance[a] > 1e-10) {
            t2 += (t_scores[a] * t_scores[a]) / score_variance[a];
        }
    }
    return t2;
}

/**
 * @brief Simple outlier detection using median absolute deviation (MAD).
 *
 * MAD = median(|x_i - median(x)|)
 *
 * Robust to outliers: a value is flagged if |x_i - median| > k * MAD,
 * typically k = 3 (very conservative) or k = 5 (moderate).
 *
 * @param data  Data array [n]
 * @param n     Number of data points
 * @param k     Threshold multiplier
 * @param flags [out] Outlier flags (1 = outlier) [n]
 * @return      Number of outliers detected
 */
int mad_outlier_detect(const double *data, int n, double k, int *flags)
{
    if (!data || !flags || n < 3) return 0;

    /* Copy and sort for median */
    double *sorted = (double *)malloc(n * sizeof(double));
    if (!sorted) return 0;
    memcpy(sorted, data, n * sizeof(double));

    /* Simple bubble sort (n is typically small, < 100) */
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                double tmp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = tmp;
            }
        }
    }

    double median;
    if (n % 2 == 0) {
        median = (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
    } else {
        median = sorted[n/2];
    }

    /* Compute absolute deviations from median */
    double *abs_dev = (double *)malloc(n * sizeof(double));
    if (!abs_dev) { free(sorted); return 0; }
    for (int i = 0; i < n; i++) {
        abs_dev[i] = fabs(data[i] - median);
    }

    /* Sort absolute deviations */
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (abs_dev[j] > abs_dev[j + 1]) {
                double tmp = abs_dev[j];
                abs_dev[j] = abs_dev[j + 1];
                abs_dev[j + 1] = tmp;
            }
        }
    }

    double mad;
    if (n % 2 == 0) {
        mad = (abs_dev[n/2 - 1] + abs_dev[n/2]) / 2.0;
    } else {
        mad = abs_dev[n/2];
    }

    /* Scale factor for normally distributed data: 1.4826 */
    double scaled_mad = mad * 1.4826;

    /* Flag outliers */
    int count = 0;
    if (scaled_mad > 1e-10) {
        for (int i = 0; i < n; i++) {
            flags[i] = (fabs(data[i] - median) > k * scaled_mad) ? 1 : 0;
            if (flags[i]) count++;
        }
    } else {
        memset(flags, 0, n * sizeof(int));
    }

    free(sorted);
    free(abs_dev);
    return count;
}

/**
 * @brief Exponential smoothing (EWMA) for a time series.
 *
 * s(k) = alpha * x(k) + (1 - alpha) * s(k-1)
 *
 * Used extensively in bias filtering, trend monitoring, and performance
 * metric smoothing throughout the quality estimation framework.
 *
 * @param current_value  New measurement x(k)
 * @param previous_smoothed Previous smoothed value s(k-1)
 * @param alpha          Smoothing factor (0 < alpha ≤ 1)
 * @return               Updated smoothed value s(k)
 */
double ewma_smooth(double current_value, double previous_smoothed, double alpha)
{
    if (alpha <= 0.0) return previous_smoothed;
    if (alpha >= 1.0) return current_value;
    return alpha * current_value + (1.0 - alpha) * previous_smoothed;
}

/**
 * @brief Standard normal (Gaussian) cumulative distribution function.
 *
 * Uses the Abramowitz and Stegun approximation (error < 7.5e-8).
 * Phi(x) = P(Z ≤ x) for Z ~ N(0, 1).
 *
 * Used for confidence interval computation and statistical tests.
 *
 * @param x  Standard normal deviate
 * @return   Phi(x) in [0, 1]
 */
double normal_cdf(double x)
{
    /* Abramowitz and Stegun 7.1.26 approximation */
    const double b0 = 0.2316419;
    const double b1 = 0.319381530;
    const double b2 = -0.356563782;
    const double b3 = 1.781477937;
    const double b4 = -1.821255978;
    const double b5 = 1.330274429;

    if (x > 6.0) return 1.0;
    if (x < -6.0) return 0.0;

    double t = 1.0 / (1.0 + b0 * fabs(x));
    double phi = exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
    double cdf = 1.0 - phi * (b1*t + b2*t*t + b3*t*t*t + b4*t*t*t*t + b5*t*t*t*t*t);

    return (x >= 0.0) ? cdf : 1.0 - cdf;
}

/**
 * @brief Inverse normal CDF (quantile function).
 *
 * Returns z such that Phi(z) = p for given probability p.
 * Uses the rational approximation from Wichura (1988), error < 1e-15.
 *
 * @param p  Probability in (0, 1)
 * @return   z-score
 */
double normal_quantile(double p)
{
    /* Simple rational approximation for the normal quantile.
     * Uses the algorithm from Wichura (1988) as presented in
     * "Algorithm AS 241: The Percentage Points of the Normal Distribution."
     * Applied Statistics, 37(3), 477-484.
     *
     * Maximum absolute error: ~1e-15 in double precision.
     */
    if (p <= 0.0) return -10.0;
    if (p >= 1.0) return 10.0;

    double q = p - 0.5;
    double r, val;

    if (fabs(q) <= 0.425) {
        /* Central region: use rational approximation in q */
        r = 0.180625 - q * q;
        val = q * (((((((2.5090809287301226727e3 * r
                       + 3.3430575583588128105e4) * r
                       + 6.7265770927008700853e4) * r
                       + 4.5921953931549871457e4) * r
                       + 1.3731693765509461125e4) * r
                       + 1.9715909503065514427e3) * r
                       + 1.3314166789178437745e2) * r
                       + 3.3871328727963666080e0)
          / (((((((5.2264952788528545610e3 * r
                  + 2.8729085735721942674e4) * r
                  + 3.9307895800092710610e4) * r
                  + 2.1213794301586595867e4) * r
                  + 5.3941960214247511077e3) * r
                  + 6.8718700749205790830e2) * r
                  + 4.2313330701600911252e1) * r
                  + 1.0);
    } else {
        /* Tail region */
        r = (q > 0.0) ? 1.0 - p : p;
        r = sqrt(-log(r));

        if (r <= 5.0) {
            r -= 1.6;
            val = (((((((7.74545014278341407640e-4 * r
                       + 2.27238449892691845833e-2) * r
                       + 2.41780725177450611770e-1) * r
                       + 1.27045825245236838258e0) * r
                       + 3.64784832476320460504e0) * r
                       + 5.76949722146069140550e0) * r
                       + 4.63033784615654529590e0) * r
                       + 1.42343711074968357734e0)
            / (((((((1.05075007164441684324e-9 * r
                    + 5.47593808499534494600e-4) * r
                    + 1.51986665636164571966e-2) * r
                    + 1.48103976427480074590e-1) * r
                    + 6.89767334985100004550e-1) * r
                    + 1.67638483018380384940e0) * r
                    + 2.05319162663775882187e0) * r
                    + 1.0);
        } else {
            r -= 5.0;
            val = (((((((2.01033439929228813265e-7 * r
                       + 2.71155556874348757815e-5) * r
                       + 1.24266094738807843860e-3) * r
                       + 2.65321895265761230930e-2) * r
                       + 2.96560571828504891230e-1) * r
                       + 1.78482653991729133580e0) * r
                       + 5.46378491116411436990e0) * r
                       + 6.65790464350110377720e0)
            / (((((((2.04426310338993978564e-15 * r
                    + 1.42151175831644588870e-7) * r
                    + 1.84631831751005468180e-5) * r
                    + 7.86869131145613259100e-4) * r
                    + 1.48753612908506148525e-2) * r
                    + 1.36929880922735805310e-1) * r
                    + 5.99832206555887937690e-1) * r
                    + 1.0);
        }

        if (q < 0.0) val = -val;
    }

    return val;
}
