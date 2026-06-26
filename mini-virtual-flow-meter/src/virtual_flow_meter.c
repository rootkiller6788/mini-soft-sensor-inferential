/**
 * @file virtual_flow_meter.c
 * @brief Core Virtual Flow Meter implementation
 *
 * Implements the lifecycle, sensor management, and main estimation dispatch.
 *
 * @module mini-virtual-flow-meter
 */

#include "virtual_flow_meter.h"
#include "flow_models.h"
#include "fluid_properties.h"
#include "vfm_state_estimation.h"
#include "vfm_uncertainty.h"
#include "pipeline_geometry.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ==========================================================================
 * L2: Lifecycle Management
 * ========================================================================== */

/**
 * ISO 5167 default discharge coefficient for orifice plates (D-D/2 taps)
 * with beta=0.5 and Re>1e5 as a starting value.
 */
#define VFM_DEFAULT_CD_ORIFICE  0.610
#define VFM_DEFAULT_CV_VENTURI  0.985
#define VFM_DEFAULT_BETA         0.50
#define VFM_DEFAULT_PIPE_DIAM    0.100
#define VFM_DEFAULT_ROUGHNESS    0.000045
#define VFM_DEFAULT_CONFIDENCE   0.95
#define VFM_DEFAULT_MAX_SENSORS  8

int vfm_config_init(vfm_config_t *config)
{
    if (!config) return -1;

    /* Set physically meaningful defaults, not zeros */
    config->model_type        = VFM_MODEL_ORIFUSE;
    config->fallback_type     = VFM_MODEL_BERNOULLI;
    config->pipe_diameter     = VFM_DEFAULT_PIPE_DIAM;
    config->pipe_roughness    = VFM_DEFAULT_ROUGHNESS;
    config->orifice_beta      = VFM_DEFAULT_BETA;
    config->orifice_discharge_coeff = VFM_DEFAULT_CD_ORIFICE;
    config->venturi_discharge_coeff = VFM_DEFAULT_CV_VENTURI;
    config->pump_rated_flow   = 0.01;
    config->pump_rated_head   = 30.0;
    config->sampling_period_s = 1.0;
    config->confidence_level  = VFM_DEFAULT_CONFIDENCE;
    config->flow_range_min    = 0.0;
    config->flow_range_max    = 1.0;
    config->max_sensors       = VFM_DEFAULT_MAX_SENSORS;
    config->enable_drift_detection  = 1;
    config->enable_auto_calibration = 0;

    return 0;
}

int vfm_config_validate(const vfm_config_t *config)
{
    if (!config) return -1;

    /* Pipe diameter must be positive and physically reasonable (< 5m) */
    if (config->pipe_diameter <= 0.0 || config->pipe_diameter > 5.0) {
        return -1;
    }

    /* Roughness must be non-negative and less than pipe diameter */
    if (config->pipe_roughness < 0.0 ||
        config->pipe_roughness >= config->pipe_diameter) {
        return -1;
    }

    /* Beta ratio must be in valid range per ISO 5167 */
    if (config->orifice_beta < 0.10 || config->orifice_beta > 0.75) {
        return -1;
    }

    /* Discharge coefficient for orifice must be in [0.55, 0.85] */
    if (config->orifice_discharge_coeff < 0.55 ||
        config->orifice_discharge_coeff > 0.85) {
        return -1;
    }

    /* Venturi discharge coefficient must be in [0.90, 1.05] */
    if (config->venturi_discharge_coeff < 0.90 ||
        config->venturi_discharge_coeff > 1.05) {
        return -1;
    }

    /* Flow range: min < max, both non-negative */
    if (config->flow_range_min < 0.0 ||
        config->flow_range_max <= config->flow_range_min) {
        return -1;
    }

    /* Sampling period must be positive */
    if (config->sampling_period_s <= 0.0) {
        return -1;
    }

    /* Confidence level in (0.5, 1.0) */
    if (config->confidence_level <= 0.5 ||
        config->confidence_level >= 1.0) {
        return -1;
    }

    /* Max sensors must be at least 1 */
    if (config->max_sensors < 1 || config->max_sensors > 64) {
        return -1;
    }

    return 0;
}

int vfm_state_init(vfm_state_t *state)
{
    if (!state) return -1;

    state->flow_rate        = 0.0;
    state->flow_rate_uncert = 1.0;    /* Large initial uncertainty */
    state->bias_correction  = 0.0;
    state->drift_rate       = 0.0;
    state->model_confidence = 0.5;    /* Uncalibrated */
    state->num_active_sensors = 0;
    state->regime           = VFM_REGIME_TURBULENT;
    state->status           = VFM_STATUS_UNCALIBRATED;

    /* Initialize all sensor health to unknown (0.5) */
    int i;
    for (i = 0; i < 8; i++) {
        state->sensor_health[i] = 0.5;
    }

    return 0;
}

int vfm_result_init(vfm_result_t *result)
{
    if (!result) return -1;

    result->flow_rate            = 0.0;
    result->mass_flow_rate       = 0.0;
    result->expanded_uncertainty = 0.0;
    result->relative_uncertainty = 0.0;
    result->upstream_pressure    = 0.0;
    result->downstream_pressure  = 0.0;
    result->fluid_temperature    = 293.15;  /* 20 C default */
    result->reynolds_number      = 0.0;
    result->regime               = VFM_REGIME_TURBULENT;
    result->status               = VFM_STATUS_UNCALIBRATED;
    result->timestamp_ms         = 0;
    result->confidence_interval[0] = 0.0;
    result->confidence_interval[1] = 0.0;
    result->num_samples_used     = 0;

    return 0;
}

/* ==========================================================================
 * L2: Sensor Management
 * ========================================================================== */

int vfm_sensor_register(vfm_state_t *state, const vfm_config_t *config,
                         const char *tag, int *sensor_id)
{
    if (!state || !config || !tag || !sensor_id) return -2;

    if (state->num_active_sensors >= config->max_sensors) {
        return -1;  /* Max sensors exceeded */
    }

    *sensor_id = state->num_active_sensors;
    state->num_active_sensors++;

    /* Initialize health for this sensor */
    if (*sensor_id < 8) {
        state->sensor_health[*sensor_id] = 1.0;
    }

    return 0;
}

int vfm_sensor_update(vfm_sensor_t *sensor, double value, double uncertainty)
{
    if (!sensor) return -1;

    /* Reject NaN and Inf values */
    if (isnan(value) || isinf(value)) return -1;
    if (isnan(uncertainty) || isinf(uncertainty) || uncertainty < 0.0) {
        return -1;
    }

    sensor->value       = value;
    sensor->uncertainty = uncertainty;
    sensor->valid       = 1;

    return 0;
}

int vfm_sensor_mark_fault(vfm_sensor_t *sensor)
{
    if (!sensor) return -1;

    sensor->valid = 0;
    return 0;
}

double vfm_sensor_health_check(const vfm_sensor_t *sensor,
                                int n_history, const double *history)
{
    if (!sensor || !history || n_history < 2) {
        /* Not enough data for health assessment */
        return sensor ? (double)sensor->valid : 0.0;
    }

    if (!sensor->valid) return 0.0;

    /* Compute mean and standard deviation of recent history */
    double sum = 0.0, sum_sq = 0.0;
    int i;
    for (i = 0; i < n_history; i++) {
        sum    += history[i];
        sum_sq += history[i] * history[i];
    }
    double mean = sum / (double)n_history;
    double variance = (sum_sq / (double)n_history) - (mean * mean);
    if (variance < 0.0) variance = 0.0;
    double std_dev = sqrt(variance);

    /* Compute coefficient of variation */
    double cv = (fabs(mean) > 1e-12) ? (std_dev / fabs(mean)) : std_dev;

    /* Check if current value is an outlier (> 3 sigma from mean) */
    double deviation = fabs(sensor->value - mean);
    double outlier_penalty = 0.0;
    if (std_dev > 0.0 && deviation > 3.0 * std_dev) {
        outlier_penalty = 0.5;
    }

    /* Health score: 1.0 if CV is low and no outliers */
    double health = 1.0;
    if (cv > 0.05) {
        health -= (cv - 0.05) * 10.0;  /* Decrease linearly with noise */
    }
    health -= outlier_penalty;

    /* Clamp to [0.0, 1.0] */
    if (health < 0.0) health = 0.0;
    if (health > 1.0) health = 1.0;

    return health;
}

/* ==========================================================================
 * L2: Main Estimation Dispatch
 * ========================================================================== */

/**
 * Helper: find sensor by tag substring match.
 * Returns sensor index, or -1 if not found.
 */
static int find_sensor_by_tag(const vfm_sensor_t *sensors, int n_sensors,
                               const char *substr)
{
    int i;
    for (i = 0; i < n_sensors; i++) {
        if (sensors[i].valid && strstr(sensors[i].tag, substr)) {
            return i;
        }
    }
    return -1;
}

int vfm_estimate(vfm_result_t *result, vfm_state_t *state,
                  const vfm_config_t *config,
                  const vfm_sensor_t *sensors, int n_sensors,
                  const vfm_fluid_t *fluid)
{
    if (!result || !state || !config || !sensors || !fluid) return -1;
    if (n_sensors < 1) return -1;

    /* Find differential pressure sensor */
    int idx_dp = find_sensor_by_tag(sensors, n_sensors, "DP");
    if (idx_dp < 0) idx_dp = find_sensor_by_tag(sensors, n_sensors, "PDT");
    if (idx_dp < 0) {
        /* Fall back to first valid sensor as pressure measurement */
        int i;
        for (i = 0; i < n_sensors; i++) {
            if (sensors[i].valid) { idx_dp = i; break; }
        }
    }
    if (idx_dp < 0) {
        result->status = VFM_STATUS_FAULT;
        return -1;
    }

    double dp_meas = sensors[idx_dp].value;

    /* Compute flow estimate based on configured model type */
    double Q = 0.0;
    orifice_params_t orif;
    switch (config->model_type) {

    case VFM_MODEL_ORIFUSE:
        orifice_params_init(&orif, config->pipe_diameter,
                            config->pipe_diameter * config->orifice_beta);
        orif.discharge_coeff = config->orifice_discharge_coeff;
        Q = orifice_vol_flow(&orif, dp_meas, fluid->density, 1.0);
        result->upstream_pressure = dp_meas + 101325.0;
        result->downstream_pressure = 101325.0;
        break;

    case VFM_MODEL_BERNOULLI:
        Q = bernoulli_flow_rate(dp_meas + 101325.0, 101325.0,
                                 pipe_cross_section_area(config->pipe_diameter),
                                 fluid->density, 0.0, 0.0);
        break;

    case VFM_MODEL_PUMP_CURVE: {
        pump_curve_params_t pump;
        pump_curve_init(&pump, config->pump_rated_flow,
                        config->pump_rated_head,
                        config->pump_rated_head * 1.2,
                        1500.0);
        /* Use DP as head measurement [Pa to m]: head = dP/(rho*g) */
        double head_m = dp_meas / (fluid->density * 9.80665);
        Q = pump_curve_flow_estimate(&pump, head_m, 1500.0);
        if (Q < 0.0) Q = 0.0;
        break;
    }

    case VFM_MODEL_DARCY_HEADLOSS:
        /* Infer flow from head loss measurement */
        /* Compute velocity via Darcy inverse */
        {
            double head_loss = dp_meas / (fluid->density * 9.80665);
            Q = darcy_inverse_flow(config->pipe_diameter,
                                   config->pipe_roughness,
                                   100.0,  /* Assume 100m pipe */
                                   head_loss,
                                   fluid->viscosity_kinematic,
                                   50, 1e-6);
            if (Q < 0.0) Q = 0.0;
        }
        break;

    default:
        /* Default to Bernoulli as simplest model */
        Q = bernoulli_flow_rate(dp_meas + 101325.0, 101325.0,
                                 pipe_cross_section_area(config->pipe_diameter),
                                 fluid->density, 0.0, 0.0);
        break;
    }

    /* Clamp to configured flow range */
    if (Q < config->flow_range_min) Q = config->flow_range_min;
    if (Q > config->flow_range_max) {
        Q = config->flow_range_max;
        result->status = VFM_STATUS_OUT_OF_RANGE;
    }

    /* Compute Reynolds number at flow condition */
    double velocity = Q / pipe_cross_section_area(config->pipe_diameter);
    double re = flow_reynolds_number(velocity, config->pipe_diameter,
                                      fluid->viscosity_kinematic);

    /* Fill result structure */
    result->flow_rate          = Q;
    result->mass_flow_rate     = Q * fluid->density;
    result->reynolds_number    = re;
    result->regime             = flow_regime_classify(re);
    result->fluid_temperature  = 293.15;  /* default, could come from sensor */
    result->num_samples_used   = n_sensors;

    /* Compute uncertainty — simplified: assume 2% relative */
    double std_uncert = 0.02 * Q;
    result->expanded_uncertainty = vfm_expanded_uncertainty(
        std_uncert, config->confidence_level);
    result->relative_uncertainty = (Q > 1e-12) ?
        (result->expanded_uncertainty / Q) : 1.0;

    /* Confidence interval */
    double k = vfm_coverage_factor(config->confidence_level);
    result->confidence_interval[0] = Q - k * std_uncert;
    result->confidence_interval[1] = Q + k * std_uncert;

    /* Update state */
    state->flow_rate       = Q;
    state->flow_rate_uncert = std_uncert;
    state->regime          = result->regime;

    if (result->status == VFM_STATUS_OUT_OF_RANGE) {
        state->status = VFM_STATUS_OUT_OF_RANGE;
    } else if (Q > 1e-12) {
        state->status = VFM_STATUS_OK;
    }

    result->status = state->status;

    return 0;
}

/* ==========================================================================
 * L2: Uncertainty Helper Functions
 * ========================================================================== */

/**
 * Approximate inverse normal CDF using the Abramowitz-Stegun rational
 * approximation (formula 26.2.23). Returns z-score for given probability.
 *
 * p in [0, 1] -> z such that Phi(z) = p.
 * Accuracy: absolute error < 4.5e-4 for 0 < p < 1.
 */
static double inverse_normal_cdf(double p)
{
    if (p <= 0.0) return -10.0;
    if (p >= 1.0) return  10.0;

    double x = p;
    if (x > 0.5) x = 1.0 - x;

    /* Abramowitz and Stegun approximation coefficients */
    double c0 = 2.515517;
    double c1 = 0.802853;
    double c2 = 0.010328;
    double d1 = 1.432788;
    double d2 = 0.189269;
    double d3 = 0.001308;

    double t = sqrt(-2.0 * log(x));
    double z = t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t);

    if (p < 0.5) z = -z;
    return z;
}

double vfm_coverage_factor(double confidence_level)
{
    /* For normal distribution, k is the z-score at (1+CL)/2 quantile. */
    if (confidence_level <= 0.0 || confidence_level >= 1.0) {
        return 2.0;  /* Default: k=2 for ~95% */
    }
    double p = (1.0 + confidence_level) / 2.0;
    return inverse_normal_cdf(p);
}

double vfm_expanded_uncertainty(double std_uncertainty,
                                 double confidence_level)
{
    if (std_uncertainty < 0.0) return 0.0;
    double k = vfm_coverage_factor(confidence_level);
    return k * std_uncertainty;
}