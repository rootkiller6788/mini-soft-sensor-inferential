/**
 * @file virtual_flow_meter.h
 * @brief Virtual Flow Meter (VFM) -- Core Types and API
 *
 * Knowledge Coverage:
 *   L1 Definitions: VFM model types, sensor structures, result types
 *   L2 Core Concepts: Soft sensing, inferential measurement, sensor fusion
 *   L3 Engineer Structures: Hierarchical VFM configuration, multi-sensor arrays
 *
 * A Virtual Flow Meter estimates volumetric/mass flow rate using available
 * process measurements (pressure, temperature, valve position, pump speed, etc.)
 * instead of a physical flow meter. This is a canonical soft sensor /
 * inferential measurement problem in process control engineering.
 *
 * Reference: Seborg, Edgar, Mellichamp (2016) "Process Dynamics and Control"
 *            Ch. 17 - Inferential Control and Soft Sensors
 *
 * @module mini-virtual-flow-meter
 */

#ifndef VIRTUAL_FLOW_METER_H
#define VIRTUAL_FLOW_METER_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * L1: Core Definitions - VFM Model Types
 * ========================================================================== */

typedef enum {
    VFM_MODEL_ORIFUSE       = 0,
    VFM_MODEL_VENTURI       = 1,
    VFM_MODEL_BERNOULLI     = 2,
    VFM_MODEL_PUMP_CURVE    = 3,
    VFM_MODEL_CHOKE_GAS     = 4,
    VFM_MODEL_CHOKE_LIQUID  = 5,
    VFM_MODEL_CORIOLIS_PROXY= 6,
    VFM_MODEL_DARCY_HEADLOSS= 7,
    VFM_MODEL_MULTIPHASE    = 8,
    VFM_MODEL_HYBRID        = 9
} vfm_model_type_t;

typedef enum {
    VFM_REGIME_LAMINAR      = 0,
    VFM_REGIME_TRANSITIONAL = 1,
    VFM_REGIME_TURBULENT    = 2,
    VFM_REGIME_CRITICAL     = 3,
    VFM_REGIME_SLUG         = 4,
    VFM_REGIME_ANNULAR      = 5
} vfm_regime_t;

typedef enum {
    VFM_STATUS_OK            = 0,
    VFM_STATUS_DEGRADED      = 1,
    VFM_STATUS_FAULT         = 2,
    VFM_STATUS_UNCALIBRATED  = 3,
    VFM_STATUS_OUT_OF_RANGE  = 4
} vfm_status_t;

/* L1: Core Data Structures */

typedef struct {
    double   value;
    double   uncertainty;
    double   bias_estimate;
    uint64_t timestamp_ms;
    int      sensor_id;
    int      valid;
    char     tag[32];
} vfm_sensor_t;

typedef struct {
    double density;
    double viscosity_dynamic;
    double viscosity_kinematic;
    double specific_heat_cp;
    double thermal_conductivity;
    double isentropic_exponent;
    double compressibility_z;
    double surface_tension;
    double vapor_pressure;
} vfm_fluid_t;

typedef struct {
    double gas_fraction;
    double oil_fraction;
    double water_fraction;
    double liquid_holdup;
    double mixture_density;
    double mixture_viscosity;
    double slip_velocity;
    int    flow_pattern;
} vfm_multiphase_t;

typedef struct {
    vfm_model_type_t model_type;
    vfm_model_type_t fallback_type;
    double pipe_diameter;
    double pipe_roughness;
    double orifice_beta;
    double orifice_discharge_coeff;
    double venturi_discharge_coeff;
    double pump_rated_flow;
    double pump_rated_head;
    double sampling_period_s;
    double confidence_level;
    double flow_range_min;
    double flow_range_max;
    int    max_sensors;
    int    enable_drift_detection;
    int    enable_auto_calibration;
} vfm_config_t;

typedef struct {
    double flow_rate;
    double flow_rate_uncert;
    double bias_correction;
    double sensor_health[8];
    double drift_rate;
    double model_confidence;
    int    num_active_sensors;
    vfm_regime_t regime;
    vfm_status_t status;
} vfm_state_t;

typedef struct {
    double flow_rate;
    double mass_flow_rate;
    double expanded_uncertainty;
    double relative_uncertainty;
    double upstream_pressure;
    double downstream_pressure;
    double fluid_temperature;
    double reynolds_number;
    vfm_regime_t regime;
    vfm_status_t status;
    uint64_t timestamp_ms;
    double confidence_interval[2];
    int    num_samples_used;
} vfm_result_t;

/* L2: Core VFM API */

int vfm_config_init(vfm_config_t *config);
int vfm_config_validate(const vfm_config_t *config);
int vfm_state_init(vfm_state_t *state);
int vfm_result_init(vfm_result_t *result);
int vfm_sensor_register(vfm_state_t *state, const vfm_config_t *config,
                         const char *tag, int *sensor_id);
int vfm_sensor_update(vfm_sensor_t *sensor, double value, double uncertainty);
int vfm_sensor_mark_fault(vfm_sensor_t *sensor);
double vfm_sensor_health_check(const vfm_sensor_t *sensor,
                                int n_history, const double *history);
int vfm_estimate(vfm_result_t *result, vfm_state_t *state,
                  const vfm_config_t *config,
                  const vfm_sensor_t *sensors, int n_sensors,
                  const vfm_fluid_t *fluid);
double vfm_expanded_uncertainty(double std_uncertainty,
                                 double confidence_level);
double vfm_coverage_factor(double confidence_level);

#ifdef __cplusplus
}
#endif

#endif /* VIRTUAL_FLOW_METER_H */
