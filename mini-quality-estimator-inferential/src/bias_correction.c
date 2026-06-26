/**
 * @file bias_correction.c
 * @brief Implementation of bias correction strategies: additive, multiplicative, EWMA-trend, CUSUM, Kalman.
 */

#include "bias_correction.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*===========================================================================
 * L2: Additive Bias Correction
 *===========================================================================*/

void bias_additive_init(bias_additive_t *bc, double gain,
                        double b_min, double b_max, double max_chg)
{
    if (!bc) return;
    memset(bc, 0, sizeof(bias_additive_t));
    bc->bias     = 0.0;
    bc->ewma_gain = gain;
    bc->bias_min  = b_min;
    bc->bias_max  = b_max;
    bc->max_bias_change = max_chg;
    bc->n_updates = 0;
}

double bias_additive_update(bias_additive_t *bc, double y_lab, double y_model)
{
    if (!bc) return 0.0;

    double residual = y_lab - y_model;
    double bias_raw = residual;  /* Additive: bias = residual */

    /* Sanity check: reject if bias_raw outside limits or change too large */
    if (bias_raw < bc->bias_min || bias_raw > bc->bias_max) {
        return bc->bias;  /* Reject, retain previous */
    }

    if (bc->max_bias_change > 0.0) {
        double change = bias_raw - bc->bias;
        if (fabs(change) > bc->max_bias_change) {
            return bc->bias;  /* Too large a change, reject */
        }
    }

    /* EWMA update */
    bc->bias_previous = bc->bias;
    bc->bias = bc->bias + bc->ewma_gain * (bias_raw - bc->bias);

    /* Store in history for statistics */
    if (bc->history_count < QEST_MAX_LAB_HISTORY) {
        bc->lab_history[bc->history_index]   = y_lab;
        bc->resid_history[bc->history_index] = residual;
        bc->history_count++;
    } else {
        bc->lab_history[bc->history_index]   = y_lab;
        bc->resid_history[bc->history_index] = residual;
    }
    bc->history_index = (bc->history_index + 1) % QEST_MAX_LAB_HISTORY;

    /* Update bias stddev */
    if (bc->history_count >= 2) {
        double sum = 0.0, sum_sq = 0.0;
        int n = bc->history_count;
        for (int i = 0; i < n; i++) {
            sum   += bc->resid_history[i];
            sum_sq += bc->resid_history[i] * bc->resid_history[i];
        }
        double mean = sum / (double)n;
        double var  = sum_sq / (double)n - mean * mean;
        bc->bias_stddev = (var > 0.0) ? sqrt(var) : 0.0;
    }

    bc->n_updates++;
    bc->is_initialized = (bc->n_updates >= 3);
    return bc->bias;
}

double bias_additive_correct(const bias_additive_t *bc, double y_model)
{
    if (!bc) return y_model;
    return y_model + bc->bias;
}

double bias_additive_get(const bias_additive_t *bc)
{
    return bc ? bc->bias : 0.0;
}

void bias_additive_reset(bias_additive_t *bc)
{
    if (!bc) return;
    bc->bias = 0.0;
    bc->bias_previous = 0.0;
}

/*===========================================================================
 * L2: Multiplicative Bias Correction
 *===========================================================================*/

void bias_mult_init(bias_multiplicative_t *bc, double gain,
                    double b_min, double b_max, double max_chg)
{
    if (!bc) return;
    memset(bc, 0, sizeof(bias_multiplicative_t));
    bc->bias     = 0.0;
    bc->ewma_gain = gain;
    bc->bias_min  = b_min;
    bc->bias_max  = b_max;
    bc->max_bias_change = max_chg;
    bc->n_updates = 0;
}

double bias_mult_update(bias_multiplicative_t *bc, double y_lab, double y_model)
{
    if (!bc) return 0.0;
    if (fabs(y_model) < 1e-10) return bc->bias;  /* Cannot divide by zero */

    double bias_raw = (y_lab / y_model) - 1.0;

    /* Sanity checks */
    if (bias_raw < bc->bias_min || bias_raw > bc->bias_max) return bc->bias;
    if (bc->max_bias_change > 0.0 && fabs(bias_raw - bc->bias) > bc->max_bias_change) {
        return bc->bias;
    }

    bc->bias = bc->bias + bc->ewma_gain * (bias_raw - bc->bias);
    bc->n_updates++;
    bc->is_initialized = (bc->n_updates >= 3);
    return bc->bias;
}

double bias_mult_correct(const bias_multiplicative_t *bc, double y_model)
{
    if (!bc) return y_model;
    return y_model * (1.0 + bc->bias);
}

double bias_mult_get(const bias_multiplicative_t *bc)
{
    return bc ? bc->bias : 0.0;
}

void bias_mult_reset(bias_multiplicative_t *bc)
{
    if (!bc) return;
    bc->bias = 0.0;
}

/*===========================================================================
 * L3: EWMA-Trend Bias Correction
 *===========================================================================*/

void bias_ewma_trend_init(bias_ewma_trend_t *bc, double lambda, double beta,
                          double boost, double trend_th)
{
    if (!bc) return;
    memset(bc, 0, sizeof(bias_ewma_trend_t));
    bc->lambda  = lambda;
    bc->beta    = beta;
    bc->lambda_boost = boost;
    bc->trend_threshold = trend_th;
    bc->bias_min = -1e6;
    bc->bias_max = 1e6;
}

double bias_ewma_trend_update(bias_ewma_trend_t *bc, double y_lab, double y_model)
{
    if (!bc) return 0.0;

    double residual = y_lab - y_model;
    double eff_lambda = bc->in_boost ? bc->lambda_boost : bc->lambda;

    /* Check if we should enter boost mode */
    if (fabs(bc->trend) > bc->trend_threshold && !bc->in_boost) {
        bc->in_boost = 1;
        eff_lambda = bc->lambda_boost;
    } else if (fabs(bc->trend) < bc->trend_threshold * 0.5 && bc->in_boost) {
        bc->in_boost = 0;
        eff_lambda = bc->lambda;
    }

    /* Update trend and bias using dual EWMA */
    bc->prev_bias = bc->bias;
    double bias_raw = residual;
    bc->bias = bc->bias + eff_lambda * (bias_raw - bc->bias);

    /* Trend update */
    double bias_change = bc->bias - bc->prev_bias;
    bc->trend = bc->trend + bc->beta * (bias_change - bc->trend);

    /* Clamp bias */
    if (bc->bias < bc->bias_min) bc->bias = bc->bias_min;
    if (bc->bias > bc->bias_max) bc->bias = bc->bias_max;

    bc->n_updates++;
    return bc->bias;
}

double bias_ewma_trend_correct(const bias_ewma_trend_t *bc, double y_model)
{
    if (!bc) return y_model;
    return y_model + bc->bias;
}

double bias_ewma_trend_get_trend(const bias_ewma_trend_t *bc)
{
    return bc ? bc->trend : 0.0;
}

/*===========================================================================
 * L5: CUSUM Bias Drift Detector
 *===========================================================================*/

void cusum_init(cusum_detector_t *cd, double delta, double H)
{
    if (!cd) return;
    memset(cd, 0, sizeof(cusum_detector_t));
    cd->drift_magnitude = delta;
    cd->K = delta / 2.0;  /* Reference value = half the drift */
    cd->H = H;
    cd->mu_0 = 0.0;       /* Target: zero-mean residuals */
    cd->cusum_hi = 0.0;
    cd->cusum_lo = 0.0;
    cd->alarm_hi = 0;
    cd->alarm_lo = 0;
}

int cusum_update(cusum_detector_t *cd, double residual, double sigma)
{
    if (!cd) return 0;
    if (sigma < 1e-10) sigma = 1.0;

    cd->last_residual = residual;
    double x = residual / sigma;  /* Standardized residual */

    /* CUSUM upper: positive drift detection */
    cd->cusum_hi += x - cd->K;
    if (cd->cusum_hi < 0.0) cd->cusum_hi = 0.0;
    if (cd->cusum_hi > cd->H) cd->alarm_hi = 1;

    /* CUSUM lower: negative drift detection */
    cd->cusum_lo += -cd->K - x;
    if (cd->cusum_lo < 0.0) cd->cusum_lo = 0.0;
    if (cd->cusum_lo > cd->H) cd->alarm_lo = 1;

    cd->n_samples++;
    return (cd->alarm_hi || cd->alarm_lo) ? 1 : 0;
}

void cusum_reset(cusum_detector_t *cd)
{
    if (!cd) return;
    cd->cusum_hi = 0.0;
    cd->cusum_lo = 0.0;
    cd->alarm_hi = 0;
    cd->alarm_lo = 0;
    cd->n_samples = 0;
}

/*===========================================================================
 * L5: Kalman-Based Bias Estimation
 *===========================================================================*/

void bias_kalman_init(bias_kalman_t *bk, double q_bias, double r_lab,
                      double bias_0, double P0)
{
    if (!bk) return;
    memset(bk, 0, sizeof(bias_kalman_t));
    bk->bias = bias_0;
    bk->P    = P0;
    bk->q_bias = q_bias;
    bk->r_lab  = r_lab;
    bk->model_variance = 1.0;
    bk->is_initialized = 1;
}

double bias_kalman_update(bias_kalman_t *bk, double y_lab, double y_model)
{
    if (!bk || !bk->is_initialized) return 0.0;

    /* Predict: bias_prior = bias, P_prior = P + q_bias */
    double P_prior = bk->P + bk->q_bias;

    /* Update: K = P_prior / (P_prior + r_lab) */
    double total_noise = P_prior + bk->r_lab;
    double K = P_prior / total_noise;

    /* Innovation: y_lab - y_model - bias */
    double innovation = y_lab - y_model - bk->bias;

    /* Update bias estimate */
    bk->bias = bk->bias + K * innovation;

    /* Update error variance */
    bk->P = (1.0 - K) * P_prior;
    if (bk->P < 0.0) bk->P = 1e-10;

    bk->n_updates++;
    return bk->bias;
}

double bias_kalman_correct(const bias_kalman_t *bk, double y_model)
{
    if (!bk) return y_model;
    return y_model + bk->bias;
}

double bias_kalman_get_uncertainty(const bias_kalman_t *bk)
{
    if (!bk) return 1.0;
    return sqrt(bk->P);
}

/*===========================================================================
 * L3: Lab Sample Validation
 *===========================================================================*/

lab_quality_t lab_sample_validate(double y_lab, double y_model,
                                  double y_lab_prev, double dt,
                                  double lab_stddev,
                                  double y_min, double y_max,
                                  double max_rate,
                                  double sigma_multiplier, double model_sigma,
                                  double max_lab_stddev)
{
    /* Check 1: Range check */
    if (y_lab < y_min || y_lab > y_max) {
        return LAB_QUALITY_BAD;
    }

    /* Check 2: Sudden jump check (rate of change) */
    if (dt > 0.0 && max_rate > 0.0) {
        double rate = fabs(y_lab - y_lab_prev) / dt;
        if (rate > max_rate) {
            return LAB_QUALITY_SUSPECT;
        }
    }

    /* Check 3: Model consistency */
    if (model_sigma > 1e-10) {
        double z_score = fabs(y_lab - y_model) / model_sigma;
        if (z_score > sigma_multiplier) {
            return LAB_QUALITY_SUSPECT;  /* Far from model prediction */
        }
    }

    /* Check 4: Lab precision */
    if (lab_stddev > max_lab_stddev && max_lab_stddev > 0.0) {
        return LAB_QUALITY_SUSPECT;
    }

    return LAB_QUALITY_GOOD;
}

/*===========================================================================
 * L3: Unified Bias Context (dispatcher)
 *===========================================================================*/

void bias_context_init(bias_context_t *ctx, const qest_config_t *config)
{
    if (!ctx || !config) return;
    memset(ctx, 0, sizeof(bias_context_t));

    ctx->active_strategy = config->bias_strategy;
    memcpy(&ctx->config, config, sizeof(qest_config_t));

    /* Initialize all sub-strategies regardless of active one */
    bias_additive_init(&ctx->additive, config->bias_filter_gain, -100.0, 100.0, 0.0);
    bias_mult_init(&ctx->multiplicative, config->bias_filter_gain, -0.9, 9.0, 0.0);
    bias_ewma_trend_init(&ctx->ewma_trend, config->bias_filter_gain, 0.1, 0.3, 0.5);
    bias_kalman_init(&ctx->kalman, 0.001, 0.01, 0.0, 1.0);
    cusum_init(&ctx->cusum, config->cusum_threshold * 0.5, config->cusum_threshold);
}

lab_quality_t bias_context_update(bias_context_t *ctx, double y_lab, double y_model)
{
    if (!ctx) return LAB_QUALITY_BAD;

    /* First, validate the lab sample through all available checks */
    /* Simplified validation using context config */
    lab_quality_t quality = LAB_QUALITY_GOOD;

    /* Model sigma */
    double model_sigma = sqrt(ctx->additive.bias_stddev * ctx->additive.bias_stddev + 0.01);

    /* Basic validation */
    if (fabs(y_lab - y_model) > 5.0 * model_sigma) {
        quality = LAB_QUALITY_SUSPECT;
    }

    /* CUSUM check for persistent drift */
    if (ctx->active_strategy == BIAS_CUSUM_TRIGGERED) {
        double residual = y_lab - y_model;
        int drift_detected = cusum_update(&ctx->cusum, residual, model_sigma);
        if (drift_detected) {
            cusum_reset(&ctx->cusum);
            /* Trigger full bias recalculation from recent history */
            quality = LAB_QUALITY_GOOD;  /* Use the lab sample despite drift */
        }
    }

    if (quality == LAB_QUALITY_BAD) return LAB_QUALITY_BAD;

    /* Dispatch to active strategy */
    switch (ctx->active_strategy) {
    case BIAS_ADDITIVE:
        bias_additive_update(&ctx->additive, y_lab, y_model);
        ctx->additive_mse = ctx->additive.bias_stddev * ctx->additive.bias_stddev;
        break;
    case BIAS_MULTIPLICATIVE:
        bias_mult_update(&ctx->multiplicative, y_lab, y_model);
        break;
    case BIAS_KALMAN:
        bias_kalman_update(&ctx->kalman, y_lab, y_model);
        ctx->kalman_mse = ctx->kalman.P;
        break;
    case BIAS_EWMA_FILTERED:
        bias_ewma_trend_update(&ctx->ewma_trend, y_lab, y_model);
        break;
    case BIAS_CUSUM_TRIGGERED:
        /* Already checked above; fall back to additive update */
        bias_additive_update(&ctx->additive, y_lab, y_model);
        break;
    case BIAS_NONE:
    default:
        break;
    }

    return quality;
}

double bias_context_correct(const bias_context_t *ctx, double y_model)
{
    if (!ctx) return y_model;

    switch (ctx->active_strategy) {
    case BIAS_ADDITIVE:
    case BIAS_CUSUM_TRIGGERED:
        return bias_additive_correct(&ctx->additive, y_model);
    case BIAS_MULTIPLICATIVE:
        return bias_mult_correct(&ctx->multiplicative, y_model);
    case BIAS_KALMAN:
        return bias_kalman_correct(&ctx->kalman, y_model);
    case BIAS_EWMA_FILTERED:
        return bias_ewma_trend_correct(&ctx->ewma_trend, y_model);
    case BIAS_NONE:
    default:
        return y_model;
    }
}

double bias_context_get_bias(const bias_context_t *ctx)
{
    if (!ctx) return 0.0;

    switch (ctx->active_strategy) {
    case BIAS_ADDITIVE:
    case BIAS_CUSUM_TRIGGERED:
        return bias_additive_get(&ctx->additive);
    case BIAS_MULTIPLICATIVE:
        return bias_mult_get(&ctx->multiplicative);
    case BIAS_KALMAN:
        return ctx->kalman.bias;
    case BIAS_EWMA_FILTERED:
        return ctx->ewma_trend.bias;
    default:
        return 0.0;
    }
}

void bias_context_set_strategy(bias_context_t *ctx, bias_strategy_t strategy)
{
    if (!ctx) return;
    ctx->active_strategy = strategy;
}
