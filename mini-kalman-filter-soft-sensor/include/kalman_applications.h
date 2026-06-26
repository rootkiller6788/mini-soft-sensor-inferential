/**
 * @file kalman_applications.h
 * @brief Kalman Filter Applications — Soft Sensors for Industrial Processes
 *
 * L6 Canonical Problems: DC motor estimation, GPS/INS, chemical reactor
 * L7 Industrial Applications: Honeywell Experion, Siemens PLC, OSIsoft PI
 *
 * This header defines application-specific structures and interfaces that
 * connect the generic Kalman filter algorithms to real industrial problems.
 *
 * Reference: Myke King (2016) "Process Control: A Practical Approach"
 * Course alignment: Stanford ENGR205 (applications), RWTH Aachen (PLC-based estimation)
 */
#ifndef KALMAN_APPLICATIONS_H
#define KALMAN_APPLICATIONS_H

#include "kalman_core.h"
#include "kalman_extended.h"
#include "kalman_unscented.h"
#include "kalman_adaptive.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * L6: DC Motor — canonical estimation problem (Berkeley ME233, ECE C128)
 * -------------------------------------------------------------------------- */

/**
 * DC Motor model for state estimation:
 *
 * State: x = [theta (angle), omega (angular velocity)]
 * Input: u = voltage
 *
 * Continuous: d/dt[theta, omega] = [omega, -b/J*omega + Kt/J*u]
 * Discrete (Euler with dt): x[k] = F*x[k-1] + B*u[k-1]
 *
 * F = [[1, dt], [0, 1 - b*dt/J]]
 * B = [[0], [Kt*dt/J]]
 * H = [[1, 0]]  (measure angle only)
 */
#define DC_MOTOR_STATE_DIM 2
#define DC_MOTOR_INPUT_DIM 1
#define DC_MOTOR_MEAS_DIM  1

typedef struct {
    /** Kalman filter and model for DC motor */
    KalmanFilterState kf;
    KalmanModel model;

    /** Motor parameters */
    double resistance;      /**< Armature resistance (Ohm) */
    double inductance;      /**< Armature inductance (H) */
    double back_emf_const;  /**< Back EMF constant Kb (V*s/rad) */
    double torque_const;    /**< Torque constant Kt (N*m/A) */
    double inertia;         /**< Rotor inertia J (kg*m^2) */
    double damping;         /**< Viscous damping b (N*m*s/rad) */
    double dt;              /**< Sampling period (s) */

    /** Estimated states */
    double angle_est;       /**< Estimated angular position (rad) */
    double velocity_est;    /**< Estimated angular velocity (rad/s) */
    double current_est;     /**< Estimated armature current (A) — augmented state */

    /** Filter status */
    uint8_t initialized;
} DCMotorEstimator;

/** Initialize DC motor estimator with motor parameters */
void dc_motor_estimator_init(DCMotorEstimator *est,
                              double resistance, double inductance,
                              double Kb, double Kt,
                              double inertia, double damping,
                              double dt,
                              double process_noise_q,
                              double measure_noise_r);

/** Update DC motor estimator with new voltage input and angle measurement */
void dc_motor_estimator_step(DCMotorEstimator *est,
                              double voltage, double angle_meas);

/** Get estimated velocity from DC motor estimator */
double dc_motor_get_velocity(const DCMotorEstimator *est);

/** Get estimated torque: T = Kt * current */
double dc_motor_get_torque(const DCMotorEstimator *est);

/* --------------------------------------------------------------------------
 * L6: Chemical Reactor — CSTR temperature estimation (Seborg, Edgar, Mellichamp 2016)
 * -------------------------------------------------------------------------- */

/** CSTR state dimension: [concentration, temperature] */
#define CSTR_STATE_DIM 2
#define CSTR_MEAS_DIM  1

typedef struct {
    /** EKF for CSTR nonlinear estimation */
    EKFState ekf;
    EKFModel ekf_model;

    /** CSTR parameters */
    double feed_conc;       /**< Feed concentration Ca0 (mol/L) */
    double feed_temp;       /**< Feed temperature T0 (K) */
    double flow_rate;       /**< Volumetric flow rate q (L/min) */
    double volume;          /**< Reactor volume V (L) */
    double pre_exp_factor;  /**< Pre-exponential factor k0 (1/min) */
    double activation_E_R;  /**< Activation energy / R (K) */
    double delta_H_R;       /**< Heat of reaction -DeltaH (K*L/mol) */
    double rho_Cp;          /**< Density * heat capacity (J/(L*K)) */
    double UA;              /**< Heat transfer coefficient * area (J/(min*K)) */
    double jacket_temp;     /**< Jacket temperature Tc (K) */
    double dt;              /**< Sampling period (min) */

    /** Estimated states */
    double conc_est;        /**< Estimated concentration (mol/L) */
    double temp_est;        /**< Estimated temperature (K) */

    uint8_t initialized;
} CSTREstimator;

/** Initialize CSTR soft sensor estimator */
void cstr_estimator_init(CSTREstimator *est,
                          double Ca0, double T0, double q, double V,
                          double k0, double E_R, double dH_R,
                          double rhoCp, double UA,
                          double dt,
                          double Q_conc, double Q_temp,
                          double R_meas);

/** EKF predict+update step for CSTR */
void cstr_estimator_step(CSTREstimator *est,
                          double Tc, double temp_meas);

/** Get estimated conversion: X = (Ca0 - Ca) / Ca0 */
double cstr_get_conversion(const CSTREstimator *est);

/* --------------------------------------------------------------------------
 * L6: GPS/INS — navigation filter (Stanford AA272, Georgia Tech AE 6530)
 * -------------------------------------------------------------------------- */

/**
 * GPS/INS loosely-coupled integration using Kalman filter.
 *
 * State: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, ...] (9-state)
 *   x = [p_x, p_y, p_z, v_x, v_y, v_z, a_bias_x, a_bias_y, a_bias_z]
 *
 * IMU provides: acceleration + angular rate (high rate, drifts)
 * GPS provides: position (low rate, no drift)
 * KF fuses the two for high-rate, drift-free navigation.
 */
#define INS_STATE_DIM  9
#define INS_MEAS_DIM   3
#define INS_INPUT_DIM  3

typedef struct {
    /** Kalman filter for 9-state INS/GPS integration */
    KalmanFilterState kf;
    KalmanModel model;

    /** Navigation parameters */
    double dt;              /**< IMU sampling period (s) */
    double gravity;         /**< Local gravity (m/s^2), typically 9.81 */

    /** IMU noise parameters */
    double accel_noise_psd; /**< Accelerometer noise PSD (m^2/s^4/Hz) */
    double accel_bias_psd;  /**< Accelerometer bias random walk PSD */
    double gps_pos_std;     /**< GPS position standard deviation (m) */

    /** Estimated states (navigation solution) */
    double pos_x, pos_y, pos_z;         /**< Position (m) in NED frame */
    double vel_x, vel_y, vel_z;         /**< Velocity (m/s) */
    double accel_bias_x;                /**< Accel bias x (m/s^2) */
    double accel_bias_y;                /**< Accel bias y (m/s^2) */
    double accel_bias_z;                /**< Accel bias z (m/s^2) */

    /** Position uncertainty (CEP — Circular Error Probable) */
    double cep;

    uint32_t imu_count;
    uint32_t gps_count;
    uint8_t gps_valid : 1;
    uint8_t initialized : 1;
    uint8_t reserved : 6;
} INSGPSNavigator;

/** Initialize INS/GPS navigation filter */
void ins_gps_init(INSGPSNavigator *nav, double dt,
                  double accel_noise, double accel_bias_noise,
                  double gps_std, double init_pos_std, double init_vel_std);

/** IMU prediction step (high rate, e.g., 100 Hz) */
void ins_gps_predict_imu(INSGPSNavigator *nav,
                          double ax, double ay, double az);

/** GPS update step (low rate, e.g., 1 Hz) */
void ins_gps_update_gps(INSGPSNavigator *nav,
                         double gps_x, double gps_y, double gps_z);

/** Get navigation solution position (x, y, z) */
void ins_gps_get_position(const INSGPSNavigator *nav,
                           double *px, double *py, double *pz);

/** Get navigation solution velocity (vx, vy, vz) */
void ins_gps_get_velocity(const INSGPSNavigator *nav,
                           double *vx, double *vy, double *vz);

/* --------------------------------------------------------------------------
 * L7: Industrial Soft Sensor Interfaces — DCS/PLC/Historian Integration
 * -------------------------------------------------------------------------- */

/**
 * SoftSensorConfig — configuration for deploying KF as industrial soft sensor.
 *
 * Models how a Kalman filter-based soft sensor is deployed in
 * Honeywell Experion, Siemens PCS7, or Rockwell PlantPAx.
 *
 * Reference: Honeywell Profit SensorPro documentation
 */
typedef struct {
    /** Tag name in the DCS/historian */
    char tag_name[64];

    /** Engineering units string */
    char units[16];

    /** Description for operator display */
    char description[128];

    /** Measurement source tags */
    char meas_tags[KF_MAX_MEAS_DIM][64];
    uint8_t num_meas_sources;

    /** Update period (seconds) */
    double update_period;

    /** Bad-value detection threshold (absolute innovation) */
    double bad_value_threshold;

    /** Freeze threshold: if exceeded, hold last good value */
    double freeze_threshold;

    /** Minimum / maximum valid estimate range */
    double range_min;
    double range_max;

    /** Filter configuration */
    uint8_t filter_type;  /**< 0=KF, 1=EKF, 2=UKF, 3=AKF */
    uint8_t status;       /**< 0=inactive, 1=active, 2=fault, 3=bypass */
    uint8_t reserved[2];
} SoftSensorConfig;

/**
 * IndustrialQualityEstimator — quality variable prediction
 *
 * Typical application: infer product quality (e.g., octane number,
 * polymer melt index, distillation purity) from readily available
 * measurements (temperatures, pressures, flow rates).
 *
 * This is a primary industrial use case for Kalman filter soft sensors.
 */
typedef struct {
    /** Adaptive Kalman filter for quality estimation */
    AKFState akf;
    KalmanModel model;

    /** Model parameters from PLS/PCA preprocessing */
    double beta[KF_MAX_STATE_DIM];  /**< Regression coefficients */
    double intercept;               /**< Regression intercept */

    /** Quality prediction and confidence */
    double quality_pred;            /**< Predicted quality value */
    double quality_std;             /**< Prediction standard deviation */
    double quality_min;             /**< Lower confidence bound (95%) */
    double quality_max;             /**< Upper confidence bound (95%) */

    /** Lab sample for calibration update */
    double lab_value;               /**< Latest lab measurement */
    uint32_t lab_timestamp;         /**< Lab sample time */
    uint8_t lab_valid;              /**< Lab value available flag */

    /** Sensor validation */
    uint8_t sensor_fault[KF_MAX_MEAS_DIM];
    uint8_t num_sensors;

    /** Exponentially weighted moving average of prediction error */
    double ewma_error;
    double ewma_alpha;

    /** Configuration */
    SoftSensorConfig config;

    uint8_t initialized;
    uint32_t update_count;
} IndustrialQualityEstimator;

/**
 * Initialize industrial quality estimator.
 *
 * Sets up the adaptive Kalman filter for inferring product quality
 * from process measurements.
 *
 * @param est        Quality estimator
 * @param beta       Regression coefficients (from PLS model)
 * @param intercept  Regression intercept
 * @param n_coeffs   Number of regression coefficients (state dimension)
 * @param Q0         Initial process noise
 * @param R0         Initial measurement noise
 * @param update_period Update period in seconds
 *
 * Complexity: O(n^2)
 * Reference: Kano & Nakagawa (2008) "Data-based process monitoring..."
 */
void quality_estimator_init(IndustrialQualityEstimator *est,
                             const double *beta, double intercept,
                             uint8_t n_coeffs,
                             double Q0, double R0,
                             double update_period);

/**
 * Update quality estimator with new process measurements.
 *
 * @param est        Quality estimator
 * @param x_meas     Process measurement vector, size est->akf.m
 *
 * Complexity: O(n^3 + m^3)
 */
void quality_estimator_update(IndustrialQualityEstimator *est,
                               const double *x_meas);

/**
 * Calibrate quality estimator with lab sample.
 *
 * Updates the model to reduce bias between inferred and lab quality.
 *
 * @param est         Quality estimator
 * @param lab_quality Lab-measured quality value
 *
 * Complexity: O(n^3)
 */
void quality_estimator_calibrate(IndustrialQualityEstimator *est,
                                  double lab_quality);

/**
 * Get quality prediction with confidence bounds.
 *
 * @param est       Quality estimator
 * @param quality   Output: predicted quality
 * @param lower_ci  Output: lower 95% confidence bound
 * @param upper_ci  Output: upper 95% confidence bound
 */
void quality_estimator_get_prediction(const IndustrialQualityEstimator *est,
                                       double *quality,
                                       double *lower_ci,
                                       double *upper_ci);

/**
 * Check for sensor faults using innovation analysis.
 *
 * Tests each measurement channel independently for bias, drift, or
 * excessive noise using the innovation sequence.
 *
 * @param est  Quality estimator
 * @return Bitmask of faulty sensor indices (bit i set = sensor i fault)
 */
uint8_t quality_estimator_check_sensors(IndustrialQualityEstimator *est);

/* --------------------------------------------------------------------------
 * L7: OSIsoft PI / Honeywell PHD historian interface structures
 * -------------------------------------------------------------------------- */

/**
 * HistorianSnapshot — data snapshot for historian integration
 *
 * Represents a timestamped value suitable for writing to
 * OSIsoft PI, Honeywell PHD, AspenTech IP.21, or similar
 * industrial time-series historians.
 */
typedef struct {
    /** Timestamp (Unix epoch seconds) */
    uint32_t timestamp_sec;

    /** Sub-second fraction (milliseconds) */
    uint16_t timestamp_ms;

    /** Quality flag (OPC-style: 0=bad, 64=uncertain, 192=good) */
    uint8_t quality;

    /** Data type and flags */
    uint8_t flags;

    /** The value */
    double value;
} HistorianSnapshot;

/**
 * Create a historian snapshot from the current estimator state.
 *
 * @param est     Quality estimator
 * @param snap    Output snapshot
 * @param tag_idx Index into config.meas_tags (0 for main quality)
 */
void historian_create_snapshot(const IndustrialQualityEstimator *est,
                                HistorianSnapshot *snap, uint8_t tag_idx);

#ifdef __cplusplus
}
#endif
#endif /* KALMAN_APPLICATIONS_H */
