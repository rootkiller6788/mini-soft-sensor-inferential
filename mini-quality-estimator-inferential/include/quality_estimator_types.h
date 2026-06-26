/**
 * @file quality_estimator_types.h
 * @brief Core type definitions for inferential quality estimation — structs, enums, constants.
 *
 * Level: L1 Definitions + L2 Core Concepts
 * Reference: 
 *   Seborg, Edgar, Mellichamp (2016) "Process Dynamics and Control" Ch. 20 — Inferential Control
 *   Qin, S.J. (2014) "Process Data Analytics in the Era of Big Data" — AIChE J, 60(9)
 *   Kadlec, Gabrys, Strandt (2009) "Data-driven Soft Sensors in the Process Industry" — CACE, 33(4)
 *
 * Course mapping:
 *   Stanford ENGR205: Process Control — soft sensing, inferential measurement
 *   Purdue ME 575: Industrial Control — quality estimation in manufacturing
 *   Tsinghua: Process Control Engineering — 软测量与推断控制
 *   CMU 24-677: Advanced Control Systems — state estimation for quality
 */

#ifndef QUALITY_ESTIMATOR_TYPES_H
#define QUALITY_ESTIMATOR_TYPES_H

#include <stddef.h>   /* size_t */
#include <stdint.h>   /* int32_t, int64_t */

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
 * L1: Core Enumerations — Model Types, Update Strategies, Sensor Modes
 *===========================================================================*/

/**
 * @brief Type of inferential model used for quality estimation.
 *
 * Reference: Kadlec et al. (2009) — taxonomy of soft sensor models.
 */
typedef enum {
    QMODEL_FIRST_PRINCIPLES = 0,   /**< White-box: mass/energy balances, reaction kinetics */
    QMODEL_DATA_DRIVEN      = 1,   /**< Black-box: PLS, ANN, SVR trained from historical data */
    QMODEL_HYBRID_GREY      = 2,   /**< Grey-box: first-principles + data-driven correction term */
    QMODEL_KALMAN_FILTER    = 3,   /**< State-space observer: Kalman/EKF/UKF for dynamic quality */
    QMODEL_JIT_LEARNING     = 4,   /**< Just-in-time learning: query local database at runtime */
    QMODEL_GAUSSIAN_PROCESS = 5    /**< Gaussian Process regression with uncertainty bounds */
} quality_model_type_t;

/**
 * @brief Bias update strategy for inferential quality estimators.
 *
 * Lab measurements arrive infrequently (hours) but process variables arrive
 * fast (seconds). Bias update reconciles the two time scales.
 *
 * Reference: Seborg et al. (2016), Section 20.4 — Inferential Control with Bias Update.
 */
typedef enum {
    BIAS_NONE            = 0,    /**< No bias correction — raw model output */
    BIAS_ADDITIVE        = 1,    /**< y_corrected = y_model + bias */
    BIAS_MULTIPLICATIVE  = 2,    /**< y_corrected = y_model * (1 + bias) */
    BIAS_KALMAN          = 3,    /**< Kalman-based bias state estimation */
    BIAS_EWMA_FILTERED   = 4,    /**< Exponentially weighted moving average of lab-model residual */
    BIAS_CUSUM_TRIGGERED = 5     /**< CUSUM detection triggers bias recalc when drift detected */
} bias_strategy_t;

/**
 * @brief Estimator operating mode.
 */
typedef enum {
    QEST_MODE_STANDBY     = 0,   /**< Not active — awaiting configuration or data */
    QEST_MODE_ACTIVE      = 1,   /**< Active estimation — producing quality predictions */
    QEST_MODE_BIAS_UPDATE = 2,   /**< Lab sample received — recalibrating bias */
    QEST_MODE_FAULT       = 3,   /**< Detected fault — estimate may be unreliable */
    QEST_MODE_RECALIBRATE = 4    /**< Full model recalibration in progress */
} qest_mode_t;

/**
 * @brief Data-driven sub-model types for PLS/PCA-based soft sensors.
 */
typedef enum {
    DATA_MODEL_PLS       = 0,   /**< Partial Least Squares regression */
    DATA_MODEL_PCA       = 1,   /**< Principal Component Analysis + regression */
    DATA_MODEL_ANN       = 2,   /**< Artificial Neural Network */
    DATA_MODEL_SVR       = 3,   /**< Support Vector Regression */
    DATA_MODEL_RF        = 4,   /**< Random Forest regression */
    DATA_MODEL_LW_PLS    = 5,   /**< Locally Weighted PLS for time-varying processes */
    DATA_MODEL_ARX       = 6    /**< Auto-Regressive with eXogenous inputs (dynamic model) */
} data_model_subtype_t;

/**
 * @brief Lab measurement quality / reliability level.
 */
typedef enum {
    LAB_QUALITY_GOOD     = 0,    /**< Normal lab measurement, within expected uncertainty */
    LAB_QUALITY_SUSPECT  = 1,    /**< Flagged by outlier test, use with caution */
    LAB_QUALITY_BAD      = 2,    /**< Failed quality checks, should not be used for update */
    LAB_QUALITY_DELAYED  = 3     /**< Sample taken but lab result delayed beyond normal window */
} lab_quality_t;

/**
 * @brief Soft sensor maintenance state for aging detection.
 */
typedef enum {
    MAINT_OK              = 0,   /**< Performance within acceptable bounds */
    MAINT_DEGRADING       = 1,   /**< Prediction error trending upward */
    MAINT_NEEDS_ATTENTION = 2,   /**< Exceeded warning threshold */
    MAINT_CRITICAL        = 3    /**< Immediately needs recalibration or replacement */
} maintenance_state_t;

/*===========================================================================
 * L1: Core Configuration Constants
 *===========================================================================*/

#define QEST_MAX_INPUT_VARS      64   /**< Maximum number of input process variables */
#define QEST_MAX_LAGS            32   /**< Maximum number of time lags for dynamic models */
#define QEST_MAX_LAB_HISTORY     256  /**< Maximum stored lab measurement history for bias */
#define QEST_MAX_LATENT_FACTORS  16   /**< Maximum latent factors in PLS/PCA models */
#define QEST_MODEL_NAME_LEN      64   /**< Maximum length of model descriptive name string */
#define QEST_TAG_NAME_LEN        32   /**< Maximum HMI/SCADA tag name length */

/* Default confidence level for quality estimates (95% = 1.96 sigma) */
#define QEST_DEFAULT_CONFIDENCE  0.95
#define QEST_Z_SCORE_95          1.96

/*===========================================================================
 * L1: Time Stamp and Lab Sample Structures
 *===========================================================================*/

/**
 * @brief Time stamp for lab samples and quality estimates.
 *
 * Industrial quality estimation operates at multiple time scales:
 * process variables sampled every 1-60 seconds, lab samples every 2-8 hours.
 */
typedef struct {
    int32_t year;
    int32_t month;
    int32_t day;
    int32_t hour;
    int32_t minute;
    double  second;
} qest_timestamp_t;

/**
 * @brief A single lab measurement record.
 *
 * Each lab sample is expensive ($50-$500) and infrequent.
 * The inferential estimator must extract maximum value from each one.
 */
typedef struct {
    qest_timestamp_t sample_time;    /**< When the sample was drawn from the process */
    qest_timestamp_t result_time;    /**< When the lab result became available */
    double           measured_value; /**< Lab-measured quality value (the "ground truth") */
    double           lab_stddev;     /**< Lab measurement standard deviation (repeatability) */
    lab_quality_t    quality_flag;   /**< Reliability assessment of this measurement */
    char             sample_id[32];  /**< LIMS sample identifier for traceability */
    int              is_used;        /**< Flag: 1 = used in bias update, 0 = rejected */
} lab_sample_t;

/*===========================================================================
 * L1: Quality Estimate Output Structure
 *===========================================================================*/

/**
 * @brief Complete quality estimation output with uncertainty bounds.
 *
 * A quality estimator does not output a single number — it must also quantify
 * the uncertainty of that estimate so that operators and advanced controllers
 * can make risk-aware decisions.
 */
typedef struct {
    double              predicted_value;      /**< Predicted quality variable value */
    double              bias_corrected_value; /**< Bias-corrected quality estimate */
    double              prediction_variance;  /**< Variance of the prediction (sigma^2) */
    double              lower_bound_95;       /**< Lower 95% confidence bound */
    double              upper_bound_95;       /**< Upper 95% confidence bound */
    double              bias_current;         /**< Current bias term value */
    double              bias_ewma;            /**< EWMA-filtered bias for trending */
    qest_timestamp_t    estimate_time;        /**< When this estimate was computed */
    qest_mode_t         mode;                 /**< Estimator mode at computation time */
    maintenance_state_t health;               /**< Soft sensor health / aging indicator */
    int                 is_valid;             /**< 1 = valid estimate, 0 = unreliable */
} quality_estimate_t;

/*===========================================================================
 * L1: Process Variable Input Structure
 *===========================================================================*/

/**
 * @brief One process variable input to the quality estimator.
 *
 * Process variables are the "hard" sensors (temperature, pressure, flow, etc.)
 * that drive the "soft" (inferential) quality sensor.
 */
typedef struct {
    char     tag_name[QEST_TAG_NAME_LEN]; /**< SCADA/DCS tag identifier */
    double   raw_value;                   /**< Raw measured value (engineering units) */
    double   scaled_value;                /**< Scaled/normalized value [0,1] or [-1,1] */
    double   low_range;                   /**< Instrument range low (for scaling) */
    double   high_range;                  /**< Instrument range high (for scaling) */
    double   sensor_stddev;               /**< Sensor measurement noise std deviation */
    double   rate_of_change;              /**< First difference for trend detection */
    int      is_faulted;                  /**< Sensor fault flag (1 = unreliable) */
    int      freeze_count;                /**< Consecutive samples with zero change (stuck sensor) */
} process_variable_t;

/*===========================================================================
 * L2: Quality Estimator Configuration
 *===========================================================================*/

/**
 * @brief Master configuration structure for an inferential quality estimator.
 *
 * This captures all user-configurable parameters that define how the
 * soft sensor operates. Industrial deployments typically have hundreds
 * of these estimators monitoring different quality variables.
 */
typedef struct {
    /* === Identification === */
    char                estimator_name[QEST_MODEL_NAME_LEN]; /**< Unique name for this estimator */
    char                quality_tag[QEST_TAG_NAME_LEN];      /**< Quality variable tag name */
    char                units[16];                            /**< Engineering units (e.g., "wt%", "cP") */

    /* === Model configuration === */
    quality_model_type_t model_type;       /**< Type of inferential model */
    data_model_subtype_t data_subtype;     /**< Sub-type for data-driven models */
    bias_strategy_t      bias_strategy;    /**< Active bias correction strategy */

    /* === Dimensions === */
    int     n_input_vars;        /**< Number of input process variables actually used */
    int     n_lags;              /**< Number of time lags for dynamic modeling */
    int     n_latent;            /**< Number of latent variables (PLS/PCA) */

    /* === Sampling rates === */
    double  fast_sample_period;  /**< Fast loop: process variable sampling interval (seconds) */
    double  lab_sample_period;   /**< Slow loop: typical lab sampling interval (seconds) */
    double  bias_filter_gain;    /**< EWMA gain for bias filtering (0..1, small = slow update) */

    /* === Statistical thresholds === */
    double  bias_warning_limit;     /**< Absolute bias exceeding this triggers warning */
    double  bias_alarm_limit;       /**< Absolute bias exceeding this triggers alarm */
    double  variance_alarm_limit;   /**< Prediction variance exceeding this triggers alarm */
    double  cusum_threshold;        /**< CUSUM threshold for drift detection */
    double  outlier_sigma;          /**< Number of std deviations for lab outlier rejection */

    /* === Model validity === */
    int     require_input_validity;     /**< Require all inputs valid (1) or use last-good (0) */
    int     max_consecutive_bad_labs;   /**< Maximum consecutive bad lab samples before alarm */

    /* === Process context === */
    int     is_dynamic;          /**< 1 = dynamic model (state-space/ARX), 0 = static */
    int     is_multivariate;     /**< 1 = multi-output quality, 0 = single quality */
} qest_config_t;

/*===========================================================================
 * L1: Model Coefficient Structures
 *===========================================================================*/

/**
 * @brief First-principles model definition.
 *
 * Example: For a distillation column, the top composition y is inferred from
 * tray temperatures via the Antoine equation and mass balances.
 *
 *   y = f(T_tray[1..N], P_column, R_reflux, ...)
 *
 * The function pointer captures arbitrary nonlinear static/dynamic mappings.
 */
typedef struct {
    int     n_params;           /**< Number of physical parameters */
    double  params[64];         /**< Physical parameter values */
    int     n_states;           /**< For dynamic models: number of internal states */
    double  initial_states[16]; /**< Initial state values */
    /**
     * @brief Model evaluation function.
     * @param inputs   Array of n_inputs process variable values
     * @param n_inputs Number of inputs
     * @param params   Model parameters
     * @param n_params Number of parameters
     * @return         Predicted quality value
     */
    double (*evaluate)(const double *inputs, int n_inputs,
                       const double *params, int n_params);
} fpm_model_t;

/**
 * @brief Linear regression model for data-driven quality estimation.
 *
 * y_pred = beta_0 + sum_{i=1}^{n_inputs} beta_i * x_i
 *
 * This is the simplest data-driven model, serving as the baseline.
 */
typedef struct {
    int     n_inputs;           /**< Number of independent variables */
    double  intercept;          /**< beta_0 — bias term */
    double  coefficients[QEST_MAX_INPUT_VARS]; /**< beta_1..beta_n — regression coefficients */
    double  r_squared;          /**< R-squared of the fitted model */
    double  rmse;               /**< Root mean square error on training data */
} linear_model_t;

/**
 * @brief PLS (Partial Least Squares) model for collinear process data.
 *
 * PLS projects both X (process variables) and Y (quality) onto latent space,
 * handling highly correlated inputs that would break ordinary least squares.
 *
 * Reference: Wold, S. et al. (2001) "PLS-regression: a basic tool of chemometrics"
 *            Chemometrics and Intelligent Laboratory Systems, 58(2).
 */
typedef struct {
    int     n_inputs;           /**< Number of X variables (process measurements) */
    int     n_outputs;          /**< Number of Y variables (quality attributes) */
    int     n_latent;           /**< Number of latent variables A */
    double  x_means[QEST_MAX_INPUT_VARS];   /**< X variable means for centering */
    double  x_stds[QEST_MAX_INPUT_VARS];    /**< X variable std devs for scaling */
    double  y_means[8];                     /**< Y variable means */
    double  y_stds[8];                      /**< Y variable std devs */
    double  x_weights[QEST_MAX_INPUT_VARS][QEST_MAX_LATENT_FACTORS]; /**< W matrix (X-weights) */
    double  x_loadings[QEST_MAX_INPUT_VARS][QEST_MAX_LATENT_FACTORS]; /**< P matrix (X-loadings) */
    double  y_loadings[8][QEST_MAX_LATENT_FACTORS];                  /**< Q matrix (Y-loadings) */
    double  beta_coeffs[QEST_MAX_INPUT_VARS][8];                     /**< Regression coefficients B */
    double  intercept[8];                     /**< Y intercept */
    double  explained_variance_x[QEST_MAX_LATENT_FACTORS]; /**< Cumulative X variance explained */
    double  explained_variance_y[QEST_MAX_LATENT_FACTORS]; /**< Cumulative Y variance explained */
} pls_model_t;

/*===========================================================================
 * L2: Moving Horizon Estimator State
 *===========================================================================*/

/**
 * @brief Moving Horizon Estimation (MHE) buffer for quality estimation.
 *
 * MHE maintains a window of recent measurements and solves an optimization
 * problem to estimate states/parameters. For quality estimation, this handles
 * varying lab sample delays naturally.
 *
 * Reference: Rawlings, Mayne, Diehl (2017) "Model Predictive Control" Ch. 4 — State Estimation.
 */
typedef struct {
    int      horizon_length;    /**< Length of the estimation horizon N */
    int      current_index;     /**< Write position in circular buffer */
    int      n_samples_stored;  /**< Number of valid samples currently in buffer */
    double   x_buffer[QEST_MAX_INPUT_VARS * QEST_MAX_LAGS]; /**< Circular buffer for inputs */
    double   y_buffer[QEST_MAX_LAGS];   /**< Circular buffer for lab measurements */
    qest_timestamp_t t_buffer[QEST_MAX_LAGS]; /**< Time stamps for each horizon point */
    double   arrival_cost_weight;      /**< Weight on arrival cost term */
    double   measurement_noise_weight; /**< Weight on measurement term */
} mhe_buffer_t;

/*===========================================================================
 * L3: Multi-Rate Data Fusion Context
 *===========================================================================*/

/**
 * @brief Context for multi-rate sensor fusion in quality estimation.
 *
 * In industrial plants, different measurements arrive at different rates:
 *   - Fast: temperature, pressure, flow (every 1-10 sec)
 *   - Medium: on-line analyzers (every 5-15 min)
 *   - Slow: lab samples (every 2-8 hours)
 *
 * The multi-rate fusion context manages these time-scale differences.
 */
typedef struct {
    int      n_fast_streams;     /**< Number of fast-sampled process variables */
    int      n_medium_streams;   /**< Number of medium-rate analyzer streams */
    int      n_slow_streams;     /**< Number of slow lab quality streams */

    double   fast_period;        /**< Sampling period for fast streams (seconds) */
    double   medium_period;      /**< Sampling period for medium streams (seconds) */
    double   slow_period;        /**< Sampling period for slow streams (seconds) */

    int64_t  fast_tick;          /**< Counter for fast samples processed */
    int64_t  medium_tick;        /**< Counter for medium samples processed */
    int64_t  slow_tick;          /**< Counter for slow samples processed */

    /** Hold-out: when medium/slow values are missing, use last-known value or model */
    double   last_medium_values[16];
    double   last_slow_values[8];
    qest_timestamp_t last_medium_time[16];
    qest_timestamp_t last_slow_time[8];

    int      use_interpolation;  /**< 1 = interpolate between slow samples, 0 = hold last */
} multi_rate_context_t;

/*===========================================================================
 * L2: Estimator Performance Metrics
 *===========================================================================*/

/**
 * @brief Accumulated performance metrics for quality estimator monitoring.
 *
 * These metrics support the L8 maintenance-aging submodule and provide
 * real-time health assessment of the soft sensor.
 */
typedef struct {
    int64_t  n_predictions;      /**< Total number of quality predictions made */
    int64_t  n_bias_updates;     /**< Number of bias correction updates applied */
    int64_t  n_lab_rejections;   /**< Number of lab samples rejected as outliers */

    double   mse;                /**< Mean squared error (model vs lab) */
    double   mae;                /**< Mean absolute error */
    double   rmse;               /**< Root mean squared error */
    double   r_squared;          /**< Coefficient of determination */
    double   mape;               /**< Mean absolute percentage error */

    double   current_bias;        /**< Current bias value (latest) */
    double   bias_stddev;         /**< Standard deviation of recent bias values */
    double   bias_trend;          /**< Slope of bias over recent N samples (drift indicator) */

    double   prediction_variance_mean; /**< Average prediction variance */

    int      n_consecutive_bad_labs;   /**< Consecutive rejected lab samples counter */

    maintenance_state_t health;  /**< Current health state */
} qest_performance_t;

/*===========================================================================
 * L1-L2: Core API Function Declarations (implementations in quality_estimator_core.c)
 *===========================================================================*/

/* --- Timestamp Operations --- */
void qest_timestamp_set(qest_timestamp_t *ts, int year, int month, int day,
                        int hour, int minute, double second);
double qest_timestamp_diff_seconds(const qest_timestamp_t *t1, const qest_timestamp_t *t2);
int qest_timestamp_compare(const qest_timestamp_t *t1, const qest_timestamp_t *t2);

/* --- Process Variable Operations --- */
void pv_init(process_variable_t *pv, const char *tag_name,
             double low, double high, double stddev);
void pv_update(process_variable_t *pv, double raw_val);

/* --- Quality Estimate Operations --- */
void qest_init_estimate(quality_estimate_t *est);
void qest_compute_confidence_bounds(quality_estimate_t *est, double z_score);

/* --- Config Operations --- */
void qest_config_init(qest_config_t *config, const char *name, const char *tag,
                      const char *units, int n_inputs);

/* --- Quality Estimator Instance --- */
/* Forward declaration of the opaque estimator type */
typedef struct quality_estimator quality_estimator_t;

quality_estimator_t *qest_alloc(void);
void qest_free(quality_estimator_t *qest);
int  qest_configure(quality_estimator_t *qest, const qest_config_t *config);
int  qest_set_linear_model(quality_estimator_t *qest, double intercept,
                           const double *coefficients, int n_inputs,
                           double r_squared, double rmse);
int  qest_set_inputs(quality_estimator_t *qest, const double *inputs, int n);
const quality_estimate_t *qest_step(quality_estimator_t *qest);
lab_quality_t qest_process_lab(quality_estimator_t *qest, const lab_sample_t *lab_sample);
void qest_get_performance(const quality_estimator_t *qest, qest_performance_t *perf);
const quality_estimate_t *qest_get_estimate(const quality_estimator_t *qest);
void qest_set_mode(quality_estimator_t *qest, qest_mode_t mode);
qest_mode_t qest_get_mode(const quality_estimator_t *qest);

/* --- PLS Model Operations --- */
void pls_model_predict(const pls_model_t *model, const double *x, int n_inputs, double *y_pred);
double pls_mahalanobis_distance(const pls_model_t *model, const double *x, int n_inputs);
double pls_hotelling_t2(const pls_model_t *model, const double *t_scores,
                        const double *score_variance);

/* --- Utility Functions --- */
double ewma_smooth(double current_value, double previous_smoothed, double alpha);
double normal_cdf(double x);
double normal_quantile(double p);
int    mad_outlier_detect(const double *data, int n, double k, int *flags);

#ifdef __cplusplus
}
#endif

#endif /* QUALITY_ESTIMATOR_TYPES_H */
