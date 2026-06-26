/**
 * @file    soft_sensor_metrics.h
 * @brief   Soft sensor performance metrics - L1 Definitions & L2 Core Concepts
 *
 * Knowledge coverage:
 *   L1: RMSE, MAE, MAPE, R^2, Q-statistic (SPE), Hotelling T^2,
 *       Aging Index, Health Index, Drift Rate, Degradation Rate
 *   L2: Model performance degradation concept, concept drift vs data drift,
 *       condition-based maintenance triggers
 *   L3: Cumulative statistics (running mean/variance via Welford algorithm),
 *       sliding window statistics, confidence intervals
 *   L4: ISO 13374 condition monitoring, ASTM D6299 PAT, SPC Western Electric rules
 *
 * Reference: Qin, S.J. (1998) "Recursive PLS algorithms for adaptive data
 *            monitoring". Computers & Chemical Engineering, 22(4-5), 503-514.
 *
 * 9-school: MIT 6.302, Stanford ENGR205, CMU 24-677, RWTH Aachen
 */

#ifndef SOFT_SENSOR_METRICS_H
#define SOFT_SENSOR_METRICS_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

/* ============================================================================
 * L1: Core Metric Definitions - typedef struct
 * ===========================================================================*/

/** @brief Individual point prediction error */
typedef struct {
    double residual;
    double relative_error;
    double squared_error;
    double timestamp;
} PredictionError;

/** @brief Aggregate regression metrics (batch) */
typedef struct {
    double rmse;
    double mae;
    double mape;
    double r2;
    double mse;
    double max_error;
    double bias;
    size_t n_samples;
} RegressionMetrics;

/**
 * @brief Cumulative statistics via Welford online algorithm
 *
 * Theorem (Welford 1962): For sequence x_1,...,x_n,
 *   mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
 *   M2_n   = M2_{n-1} + (x_n - mean_{n-1}) * (x_n - mean_n)
 *   var_n  = M2_n / n   (population) or M2_n / (n-1)  (sample)
 */
typedef struct {
    double mean;
    double m2;
    double var_population;
    double var_sample;
    double min_val;
    double max_val;
    size_t count;
} RunningStatistics;

/** @brief Sliding window metrics (fixed-size ring buffer) */
typedef struct {
    double *buffer;
    size_t capacity;
    size_t head;
    size_t count;
    double sum;
    double sum_sq;
    double current_mean;
    double current_var;
} SlidingWindow;

/* ============================================================================
 * L1: Soft Sensor Lifecycle States (ISO 13374 aligned)
 * ===========================================================================*/

typedef enum {
    LIFECYCLE_COMMISSIONING   = 0,
    LIFECYCLE_NORMAL          = 1,
    LIFECYCLE_WARNING         = 2,
    LIFECYCLE_AGING           = 3,
    LIFECYCLE_MAINTENANCE     = 4,
    LIFECYCLE_FAILURE         = 5,
    LIFECYCLE_RETIRED         = 6
} SensorLifecycleStage;

/** @brief Health index combining multiple degradation indicators */
typedef struct {
    double accuracy_score;
    double drift_index;
    double noise_index;
    double reliability_score;
    SensorLifecycleStage stage;
    uint64_t hours_since_last_cal;
    double aging_rate;
} HealthIndex;

/* ============================================================================
 * L1: Multivariate Monitoring Statistics
 * ===========================================================================*/

/** @brief Q-statistic (Squared Prediction Error / SPE) */
typedef struct {
    double spe_value;
    double spe_limit_95;
    double spe_limit_99;
    int    spe_alarm;
} QStatistic;

/** @brief Hotelling T^2 statistic */
typedef struct {
    double t2_value;
    double t2_limit_95;
    double t2_limit_99;
    int    t2_alarm;
} T2Statistic;

/** @brief Combined multivariate monitoring result */
typedef struct {
    QStatistic  q_stat;
    T2Statistic t2_stat;
    double contributions[20];
    int    n_vars;
    int    overall_alarm;
} MultivariateMonitor;

/* ============================================================================
 * L2: Concept Drift & Data Drift Indicators
 * ===========================================================================*/

/**
 * @brief Concept drift type classification
 *
 * Gama et al. (2014), "A survey on concept drift adaptation"
 * ACM Computing Surveys, 46(4), 1-37.
 */
typedef enum {
    DRIFT_NONE               = 0,
    DRIFT_SUDDEN             = 1,
    DRIFT_INCREMENTAL        = 2,
    DRIFT_GRADUAL            = 3,
    DRIFT_RECURRING          = 4,
    DRIFT_BLIP               = 5
} DriftType;

/** @brief Comprehensive drift analysis result */
typedef struct {
    DriftType type;
    double   magnitude;
    double   rate_of_change;
    double   confidence;
    size_t   detection_lag;
    double   onset_timestamp;
} DriftAnalysis;

/* ============================================================================
 * L3: Confidence Interval & Hypothesis Testing Structures
 * ===========================================================================*/

/** @brief Confidence interval (Student t-distribution based) */
typedef struct {
    double point_estimate;
    double lower_bound;
    double upper_bound;
    double confidence_level;
    size_t degrees_freedom;
    double standard_error;
} ConfidenceInterval;

/** @brief Paired t-test result */
typedef struct {
    double t_statistic;
    double p_value;
    double critical_value;
    double alpha;
    int    reject_null;
    double mean_difference;
    double cohens_d;
} PairedTTest;

/* ============================================================================
 * API: Regression Metrics
 * ===========================================================================*/

RegressionMetrics compute_regression_metrics(const double *actual,
                                              const double *predicted,
                                              size_t n);

void update_regression_metrics_online(RegressionMetrics *state,
                                       double actual, double pred);

/* ============================================================================
 * API: Running Statistics (Welford Algorithm)
 * ===========================================================================*/

RunningStatistics running_stats_init(void);
void running_stats_push(RunningStatistics *stats, double x);
void running_stats_remove(RunningStatistics *stats, double x);
RunningStatistics running_stats_merge(const RunningStatistics *a,
                                       const RunningStatistics *b);

/* ============================================================================
 * API: Sliding Window
 * ===========================================================================*/

void sliding_window_init(SlidingWindow *window, size_t capacity);
void sliding_window_destroy(SlidingWindow *window);
void sliding_window_push(SlidingWindow *window, double value);
double sliding_window_mean(const SlidingWindow *window);
double sliding_window_variance(const SlidingWindow *window);
double sliding_window_stddev(const SlidingWindow *window);
double sliding_window_ewma(const SlidingWindow *window, double lambda);

/* ============================================================================
 * API: Health Index & Lifecycle
 * ===========================================================================*/

HealthIndex health_index_init(void);
void health_index_update(HealthIndex *health,
                          double current_rmse, double baseline_rmse,
                          double current_bias, double current_var,
                          double baseline_var,
                          double hours_elapsed);
const char *lifecycle_stage_name(SensorLifecycleStage stage);
int health_index_recommends_maintenance(const HealthIndex *health);

/* ============================================================================
 * API: Multivariate Monitoring
 * ===========================================================================*/

QStatistic compute_q_statistic(const double *residuals, size_t m,
                                const double *eigenvalues, size_t n_components);

T2Statistic compute_t2_statistic(const double *scores, const double *eigenvalues,
                                  size_t n_components, size_t n_samples,
                                  double alpha);

void compute_spe_contributions(const double *residuals, size_t m,
                                double *contributions);

/* ============================================================================
 * API: Drift Analysis
 * ===========================================================================*/

DriftAnalysis detect_drift(const double *metric_stream, size_t n,
                            double confidence);

double compute_cusum_statistic(const double *stream, size_t n,
                                double target, double sigma,
                                double k, double h,
                                int *alarm_idx);

double page_hinkley_test(const double *stream, size_t n,
                          double delta, double lambda,
                          int *alarm_idx);

/* ============================================================================
 * API: Confidence Intervals & Hypothesis Tests
 * ===========================================================================*/

ConfidenceInterval confidence_interval_mean(const double *data,
                                              size_t n, double conf_level);

PairedTTest paired_t_test(const double *errors_a, const double *errors_b,
                           size_t n, double alpha);

double t_distribution_critical_value(double p, double df);

double f_distribution_critical_value(double p, double df1, double df2);

double compute_p_value_from_t(double t_stat, double df);

#endif /* SOFT_SENSOR_METRICS_H */
