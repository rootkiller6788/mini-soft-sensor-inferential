/**
 * @file    model_health_monitor.h
 * @brief   Soft sensor model health monitoring - L2 Core Concepts & L5 Algorithms
 *
 * Tracks: performance trajectory, degradation rate estimation,
 *         residual analysis, auto-correlation monitoring,
 *         maintenance scheduling via RUL prediction.
 *
 * L2: Model health concept, performance degradation trajectory
 * L4: ISO 13379 data processing for diagnostics
 * L5: ARIMA-based residual auto-correlation, linear degradation model,
 *     RUL (Remaining Useful Life) estimation
 * L6: Refinery CDU soft sensor health monitoring
 *
 * Ref: Kadlec et al. (2009) "Soft sensors: review and recent trends"
 *      Comp. & Chem. Eng., 33(4), 795-814.
 */

#ifndef MODEL_HEALTH_MONITOR_H
#define MODEL_HEALTH_MONITOR_H

#include "soft_sensor_metrics.h"
#include <stddef.h>
#include <stdint.h>

/* ============================================================================
 * L2: Performance Trajectory
 * ===========================================================================*/

/** @brief Performance sample at one point in time */
typedef struct {
    double timestamp;
    double rmse;
    double r2;
    double bias;
    double variance;
} PerformanceSnapshot;

/** @brief Performance trajectory tracked over time */
typedef struct {
    PerformanceSnapshot *snapshots;
    size_t capacity;
    size_t count;
    double baseline_rmse;
    double baseline_r2;
    double degradation_threshold;
} PerformanceTrajectory;

/** @brief Degradation model parameters from linear fit */
typedef struct {
    double slope;
    double intercept;
    double r_squared;
    double degradation_rate_per_1000h;
} DegradationModel;

/* ============================================================================
 * L2: Residual Analysis
 * ===========================================================================*/

/** @brief Auto-correlation analysis of prediction residuals */
typedef struct {
    double *residuals;
    size_t n;
    double mean_residual;
    double var_residual;
    double lag1_autocorr;
    double lag2_autocorr;
    double durbin_watson;
    double ljung_box_q;
    double ljung_box_pvalue;
    int    is_white_noise;
} ResidualAnalysis;

/* ============================================================================
 * L5: RUL Estimation
 * ===========================================================================*/

/** @brief Remaining Useful Life (RUL) estimate */
typedef struct {
    double rul_hours;
    double rul_lower_ci;
    double rul_upper_ci;
    double confidence;
    double current_degradation_pct;
    int    maintenance_urgent;
} RULEstimate;

/* ============================================================================
 * L8: Bayesian Change Point
 * ===========================================================================*/

/** @brief Bayesian change point detection result */
typedef struct {
    size_t change_point;
    double posterior_prob;
    double bayes_factor;
    int    change_detected;
    double mean_before;
    double mean_after;
} BayesianChangePoint;

/* ============================================================================
 * API: Performance Trajectory
 * ===========================================================================*/

void performance_trajectory_init(PerformanceTrajectory *traj,
                                  size_t capacity,
                                  double baseline_rmse,
                                  double baseline_r2,
                                  double degradation_threshold);
void performance_trajectory_destroy(PerformanceTrajectory *traj);
void performance_trajectory_add(PerformanceTrajectory *traj,
                                 double timestamp, double rmse,
                                 double r2, double bias, double variance);

/**
 * @brief Fit linear degradation model: rmse = slope * hours + intercept.
 * Uses simple linear regression on (timestamp, rmse) pairs.
 */
DegradationModel fit_degradation_model(const PerformanceTrajectory *traj);

/**
 * @brief Check if degradation is statistically significant.
 * Uses t-test on slope coefficient.
 * Returns 1 if degradation is confirmed (p < 0.05 on slope).
 */
int is_degradation_significant(const PerformanceTrajectory *traj);

/* ============================================================================
 * API: Residual Analysis
 * ===========================================================================*/

/**
 * @brief Analyze prediction residuals for patterns.
 *
 * Computes:
 *   - Lag-1 and Lag-2 auto-correlation
 *   - Durbin-Watson statistic for serial correlation
 *   - Ljung-Box Q test for overall randomness
 *
 * Durbin-Watson ? 2: no autocorrelation
 *                 < 1: strong positive autocorrelation
 *                 > 3: strong negative autocorrelation
 *
 * Ref: Durbin & Watson (1950) Biometrika 37, 409-428.
 *      Ljung & Box (1978) Biometrika 65, 297-303.
 */
ResidualAnalysis analyze_residuals(const double *actual,
                                    const double *predicted,
                                    size_t n);

/**
 * @brief Compute Durbin-Watson statistic.
 * DW = sum(e_t - e_{t-1})^2 / sum(e_t^2)
 */
double compute_durbin_watson(const double *residuals, size_t n);

/**
 * @brief Ljung-Box portmanteau test for autocorrelation.
 * Q = n(n+2) * sum(r_k^2 / (n-k)) for k=1..m
 * p-value from chi-squared distribution with m df.
 */
double ljung_box_test(const double *residuals, size_t n, size_t m);

/* ============================================================================
 * API: RUL Estimation
 * ===========================================================================*/

/**
 * @brief Estimate Remaining Useful Life based on degradation model.
 *
 * RUL = (failure_threshold - current_rmse) / degradation_rate
 *
 * @param traj          Performance trajectory.
 * @param failure_rmse  RMSE threshold at which sensor is considered failed.
 * @return RULEstimate with RUL, confidence intervals.
 */
RULEstimate estimate_rul(const PerformanceTrajectory *traj,
                          double failure_rmse);

/**
 * @brief Schedule next maintenance based on RUL and policy.
 *
 * Returns recommended hours until next maintenance.
 * Policy: schedule at min(RUL * 0.7, max_interval) if RUL > 0.
 *
 * @param rul             Current RUL estimate.
 * @param max_interval    Maximum allowed interval between checks (hours).
 * @param safety_margin   Fraction of RUL to use as interval (e.g. 0.7).
 * @return Hours until next recommended maintenance check.
 */
double schedule_maintenance(const RULEstimate *rul,
                             double max_interval,
                             double safety_margin);

/* ============================================================================
 * L8: Bayesian Change Point Detection
 * ===========================================================================*/

/**
 * @brief Bayesian change point detection on performance metric stream.
 *
 * Assumes observations follow N(mu_before, sigma^2) before change,
 * and N(mu_after, sigma^2) after change. Uses conjugate Normal-Inverse-Gamma
 * prior and computes posterior probability of change at each time point.
 *
 * Ref: Barry & Hartigan (1993) JASA 88, 309-319.
 *
 * @param stream     Performance metric time series.
 * @param n          Number of observations.
 * @param threshold  Posterior probability threshold for detection (e.g. 0.95).
 * @return BayesianChangePoint result.
 */
BayesianChangePoint bayesian_change_point_detect(const double *stream,
                                                   size_t n,
                                                   double threshold);

#endif /* MODEL_HEALTH_MONITOR_H */
