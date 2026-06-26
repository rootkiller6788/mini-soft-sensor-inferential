/**
 * @file multi_rate_fusion.h
 * @brief Multi-rate sensor fusion for quality estimation — combining fast/medium/slow data streams.
 *
 * Level: L3 Engineering Structures + L5 Algorithms
 * Reference:
 *   Gustafsson, F. (2012) "Statistical Sensor Fusion" — Ch. 4-6
 *   Rawlings, Mayne, Diehl (2017) "Model Predictive Control" Ch. 4 — Multi-rate state estimation
 *   Smyth, Wu (2007) "Multi-rate Kalman filtering for the data fusion of... — IEEE TIE 54(1)
 *
 * Course mapping:
 *   CMU 24-677: Advanced Control — multi-rate systems, sensor fusion
 *   Stanford AA272: Multi-sensor GNSS fusion → industrial multi-rate analogy
 *   Purdue ME 575: Industrial data reconciliation across time scales
 */

#ifndef MULTI_RATE_FUSION_H
#define MULTI_RATE_FUSION_H

#include "quality_estimator_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
 * L3: Multi-Rate Kalman Fusion
 *===========================================================================*/

/**
 * @brief Multi-rate Kalman filter for fusing fast/medium/slow quality-relevant data.
 *
 * The core idea: run a fast-rate Kalman filter using fast process variables.
 * When slower measurements (analyzers, lab samples) arrive, perform a
 * measurement update using the stored predicted state at that past time.
 *
 * This avoids the "out-of-sequence measurement" problem by maintaining
 * a buffer of past state predictions.
 */
typedef struct {
    int      n_states;         /**< State vector dimension */
    int      n_fast_inputs;    /**< Fast-rate input dimension */
    int      n_medium_meas;    /**< Medium-rate measurement dimension */
    int      n_slow_meas;      /**< Slow-rate measurement dimension */

    /* State-space matrices for the fast-rate model */
    double  *A;                /**< Fast-rate state transition [n_states × n_states] */
    double  *B;                /**< Fast-rate input matrix [n_states × n_fast_inputs] */
    double  *C_medium;         /**< Medium-rate measurement matrix [n_medium_meas × n_states] */
    double  *C_slow;           /**< Slow-rate measurement matrix [n_slow_meas × n_states] */
    double  *Q;                /**< Fast-rate process noise [n_states × n_states] */
    double  *R_medium;         /**< Medium-rate measurement noise [n_medium_meas × n_medium_meas] */
    double  *R_slow;           /**< Slow-rate measurement noise [n_slow_meas × n_slow_meas] */

    /* Current filter state */
    double  *x;                /**< Current state estimate [n_states] */
    double  *P;                /**< Current error covariance [n_states × n_states] */

    /* Buffered state for delayed measurement processing (out-of-sequence) */
    int      buffer_size;      /**< Number of past states stored */
    double  *x_buffer;         /**< Stored past states [buffer_size × n_states] */
    double  *P_buffer;         /**< Stored past covariances [buffer_size × n_states × n_states] */
    double  *t_buffer;         /**< Time stamps for buffer entries [buffer_size] */
    int      buffer_head;      /**< Current write position */
    int      buffer_count;     /**< Valid entries in buffer */

    /* Rate tracking */
    double   t_current;        /**< Current time (seconds) */
    double   t_last_medium;    /**< Time of last medium-rate measurement */
    double   t_last_slow;      /**< Time of last slow-rate measurement */
    double   dt_fast;          /**< Fast-rate sampling interval */

    /* Workspace */
    double  *temp_nn;          /**< Scratch [n_states × n_states] */
    double  *temp_n;           /**< Scratch [n_states] */
    double  *temp_mm_m;        /**< Scratch for medium [n_medium_meas × n_medium_meas] */
    double  *temp_ss_s;        /**< Scratch for slow [n_slow_meas × n_slow_meas] */
} mrf_kalman_t;

/**
 * @brief Allocate and initialize a multi-rate Kalman fusion filter.
 *
 * @param mrf           Uninitialized multi-rate Kalman filter
 * @param n_states      State dimension
 * @param n_fast        Fast-rate input dimension
 * @param n_medium      Medium-rate measurement dimension
 * @param n_slow        Slow-rate measurement dimension
 * @param buffer_size   State history buffer size (typically slow_period / fast_period)
 */
void mrf_kalman_alloc(mrf_kalman_t *mrf, int n_states, int n_fast,
                      int n_medium, int n_slow, int buffer_size);

/**
 * @brief Free multi-rate Kalman filter memory.
 *
 * @param mrf  Filter to free
 */
void mrf_kalman_free(mrf_kalman_t *mrf);

/**
 * @brief Set system matrices for the multi-rate filter.
 *
 * @param mrf      Multi-rate Kalman filter
 * @param A        Fast-rate state transition
 * @param B        Fast-rate input matrix
 * @param C_medium Medium-rate measurement matrix
 * @param C_slow   Slow-rate measurement matrix
 */
void mrf_set_matrices(mrf_kalman_t *mrf, const double *A, const double *B,
                      const double *C_medium, const double *C_slow);

/**
 * @brief Set noise covariance matrices.
 *
 * @param mrf      Multi-rate Kalman filter
 * @param Q        Fast-rate process noise
 * @param R_medium Medium-rate measurement noise
 * @param R_slow   Slow-rate measurement noise
 */
void mrf_set_noise(mrf_kalman_t *mrf, const double *Q,
                   const double *R_medium, const double *R_slow);

/**
 * @brief Process one fast-rate sample with input u.
 *
 * This is called every fast sampling period (e.g., every 10 seconds).
 * Performs predict step only (no measurement update).
 * Stores the predicted state for later out-of-sequence updates.
 *
 * @param mrf  Multi-rate Kalman filter (updated in-place)
 * @param u    Fast-rate input vector [n_fast]
 * @param t    Current time stamp (seconds)
 */
void mrf_fast_step(mrf_kalman_t *mrf, const double *u, double t);

/**
 * @brief Process a medium-rate measurement update.
 *
 * Searches the state buffer for the state at the measurement time,
 * then applies a Kalman update using the medium-rate measurement model.
 *
 * @param mrf      Multi-rate Kalman filter (updated in-place)
 * @param y_medium Medium-rate measurement [n_medium]
 * @param t_meas   Measurement time (may be in the past due to analyzer delay)
 */
void mrf_medium_update(mrf_kalman_t *mrf, const double *y_medium, double t_meas);

/**
 * @brief Process a slow-rate (lab) measurement update.
 *
 * @param mrf    Multi-rate Kalman filter (updated in-place)
 * @param y_slow Slow-rate measurement [n_slow]
 * @param t_meas Measurement time (may be hours in the past)
 */
void mrf_slow_update(mrf_kalman_t *mrf, const double *y_slow, double t_meas);

/**
 * @brief Get the current state estimate.
 *
 * @param mrf  Multi-rate Kalman filter
 * @param x    [out] State estimate [n_states]
 */
void mrf_get_state(const mrf_kalman_t *mrf, double *x);

/**
 * @brief Get the quality estimate (first element of state by default).
 *
 * @param mrf  Multi-rate Kalman filter
 * @return     Current quality estimate
 */
double mrf_get_quality(const mrf_kalman_t *mrf);

/**
 * @brief Get the quality estimate variance.
 *
 * @param mrf  Multi-rate Kalman filter
 * @return     Quality estimation error variance
 */
double mrf_get_quality_variance(const mrf_kalman_t *mrf);

/*===========================================================================
 * L3: Multi-Rate Interpolation
 *===========================================================================*/

/**
 * @brief Linear interpolation between slow lab samples for continuous estimates.
 *
 * When lab samples are hours apart, we need to estimate quality continuously.
 * Linear interpolation between the most recent lab-corrected values provides
 * a simple, robust solution.
 *
 * y_est(t) = y_prev + (y_next - y_prev) * (t - t_prev) / (t_next - t_prev)
 *
 * This is also known as the "zero-order hold with linear trend" method.
 */
typedef struct {
    int      has_prev;          /**< Flag: at least one lab sample received */
    int      has_next;          /**< Flag: at least two lab samples for interpolation */
    double   y_prev;            /**< Previous lab-corrected quality value */
    double   y_next;            /**< Next lab-corrected quality value */
    double   t_prev;            /**< Time of previous lab sample */
    double   t_next;            /**< Time of next lab sample */
    double   max_gap;           /**< Maximum interpolation gap (seconds) — beyond this, hold last */
} mrf_interpolator_t;

/**
 * @brief Initialize a multi-rate interpolator.
 *
 * @param interp  Uninitialized interpolator
 * @param max_gap Maximum gap for interpolation (seconds, e.g., 4*lab_period)
 */
void mrf_interp_init(mrf_interpolator_t *interp, double max_gap);

/**
 * @brief Feed a new lab measurement to the interpolator.
 *
 * Shifts: prev ← next, next ← (y_new, t_new)
 *
 * @param interp  Interpolator (updated in-place)
 * @param y_lab   New lab-corrected quality value
 * @param t_lab   Time of lab sample
 */
void mrf_interp_feed(mrf_interpolator_t *interp, double y_lab, double t_lab);

/**
 * @brief Interpolate quality at a requested time.
 *
 * @param interp Interpolator
 * @param t      Requested time
 * @param y_est  [out] Interpolated quality estimate
 * @return       1 if interpolation successful, 0 if extrapolation (hold last)
 */
int mrf_interp_evaluate(const mrf_interpolator_t *interp, double t, double *y_est);

/*===========================================================================
 * L5: Multi-Rate Cubature Kalman Filter for strongly nonlinear quality models
 *===========================================================================*/

/**
 * @brief Cubature Kalman Filter (CKF) for multi-rate nonlinear quality estimation.
 *
 * The CKF uses 2*n spherical-radial cubature points to approximate the
 * nonlinear transformation of Gaussian random variables, avoiding the
 * linearization error of EKF and the tuning parameters of UKF.
 *
 * Reference: Arasaratnam, Haykin (2009) "Cubature Kalman Filters"
 *            IEEE Trans Automatic Control, 54(6), 1254-1269.
 *
 * For quality estimation with strongly nonlinear models (e.g., Arrhenius
 * kinetics), CKF provides superior accuracy over EKF at similar cost.
 */
typedef struct {
    int      n_states;          /**< State dimension */
    int      n_inputs;          /**< Input dimension */
    int      n_measurements;    /**< Measurement dimension */

    /* Cubature points (stored for reuse) */
    double  *cubature_points;   /**< Cubature points [2*n_states × n_states] */
    double  *propagated_points; /**< Propagated cubature points [2*n_states × n_states] */
    double  *meas_pred_points;  /**< Predicted measurements [2*n_states × n_measurements] */

    /* Weights */
    double   weight;            /**< Weight per cubature point (1 / (2*n_states)) */

    /* Current state */
    double  *x;                 /**< State estimate [n_states] */
    double  *P;                 /**< Error covariance [n_states × n_states] */
    double  *Q;                 /**< Process noise [n_states × n_states] */
    double  *R;                 /**< Measurement noise [n_measurements × n_measurements] */

    /* State transition and measurement functions */
    void (*f)(const double *x, const double *u, double dt, double *x_next);
    void (*h)(const double *x, const double *u, double *y_pred);

    /* Workspace */
    double  *x_prior;           /**< Predicted state [n_states] */
    double  *P_prior;           /**< Predicted covariance [n_states × n_states] */
    double  *y_prior;           /**< Predicted measurement [n_measurements] */
    double  *P_xy;              /**< Cross-covariance [n_states × n_measurements] */
    double  *P_yy;              /**< Innovation covariance [n_measurements × n_measurements] */
    double  *K;                 /**< Kalman gain [n_states × n_measurements] */
    double  *temp_n;            /**< Scratch [n_states] */
    double  *temp_nn;           /**< Scratch [n_states × n_states] */
    double  *temp_mm;           /**< Scratch [n_measurements × n_measurements] */
    double  *temp_nm;           /**< Scratch [n_states × n_measurements] */
} ckf_filter_t;

/**
 * @brief Allocate and initialize a Cubature Kalman Filter.
 *
 * @param ckf            Uninitialized CKF
 * @param n_states       State dimension
 * @param n_inputs       Input dimension  
 * @param n_measurements Measurement dimension
 * @param f              State transition function
 * @param h              Measurement function
 */
void ckf_alloc(ckf_filter_t *ckf, int n_states, int n_inputs, int n_measurements,
               void (*f)(const double*, const double*, double, double*),
               void (*h)(const double*, const double*, double*));

/**
 * @brief Free CKF memory.
 */
void ckf_free(ckf_filter_t *ckf);

/**
 * @brief Setup CKF with noise and initial conditions.
 */
void ckf_setup(ckf_filter_t *ckf, const double *Q, const double *R,
               const double *x0, const double *P0);

/**
 * @brief Run one CKF step (predict + optional update).
 */
void ckf_step(ckf_filter_t *ckf, const double *u, const double *y, double dt);

/**
 * @brief Get quality estimate from CKF.
 */
double ckf_get_quality(const ckf_filter_t *ckf);

/**
 * @brief Get quality variance from CKF.
 */
double ckf_get_quality_variance(const ckf_filter_t *ckf);

#ifdef __cplusplus
}
#endif

#endif /* MULTI_RATE_FUSION_H */
