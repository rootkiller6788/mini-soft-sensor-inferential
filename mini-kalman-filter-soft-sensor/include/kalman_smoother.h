/**
 * @file kalman_smoother.h
 * @brief Kalman Smoothing — Fixed-Interval, Fixed-Point, Fixed-Lag Smoothers
 *
 * L5 Algorithms: RTS smoother, Two-Filter smoother, Fixed-lag smoothing
 *
 * While the Kalman filter estimates x[k] given z[0..k] (filtering),
 * the smoother estimates x[k] given z[0..N] for N > k (smoothing).
 * Smoothing uses all available data and produces more accurate estimates
 * than filtering alone, at the cost of being non-causal (offline).
 *
 * Reference: Anderson & Moore (1979) "Optimal Filtering", Ch. 7
 * Course alignment: Stanford ENGR205, Berkeley ME233
 */
#ifndef KALMAN_SMOOTHER_H
#define KALMAN_SMOOTHER_H

#include "kalman_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum number of time steps for stored history */
#define KSM_MAX_HISTORY 4096

/**
 * L1: KFHistoryEntry — one time-step of forward-pass results
 *
 * Stores the results of a single Kalman filter predict+update cycle
 * for later use by the RTS smoother.
 */
typedef struct {
    /** Posterior state estimate x[k|k] */
    double x_post[KF_MAX_STATE_DIM];

    /** Posterior covariance P[k|k], n x n row-major */
    double P_post[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Prior state prediction x[k|k-1] */
    double x_prior[KF_MAX_STATE_DIM];

    /** Prior covariance P[k|k-1], n x n row-major */
    double P_prior[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Kalman gain K[k], n x m row-major */
    double K[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM];

    /** Innovation y[k], size m */
    double innovation[KF_MAX_MEAS_DIM];

    /** Innovation covariance S[k], m x m row-major */
    double S[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];

    /** Timestamp or index */
    uint32_t step;

    /** State dimension */
    uint8_t n;

    /** Measurement dimension */
    uint8_t m;

    /** Valid flag */
    uint8_t valid;
} KFHistoryEntry;

/**
 * L1: KFHistory — circular buffer of forward-pass results
 */
typedef struct {
    /** History entries */
    KFHistoryEntry entries[KSM_MAX_HISTORY];

    /** Number of stored entries */
    uint32_t count;

    /** Maximum capacity */
    uint32_t capacity;

    /** State dimension */
    uint8_t n;

    /** Measurement dimension */
    uint8_t m;

    /** Write index for circular buffer */
    uint32_t write_idx;
} KFHistory;

/**
 * L1: RTSSmoother — Rauch-Tung-Striebel smoother state
 */
typedef struct {
    /** Smoothed state at current step */
    double x_smooth[KF_MAX_STATE_DIM];

    /** Smoothed covariance at current step */
    double P_smooth[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Smoothing gain G[k] */
    double G[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Work arrays */
    double F_transpose[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    double temp_nn[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    double temp_nn2[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** State dimension */
    uint8_t n;
} RTSSmoother;

/**
 * L1: FixedLagSmoother — real-time capable smoother with delay L
 */
typedef struct {
    /** Stored filter states for lag window */
    KFHistoryEntry lag_buffer[KSM_MAX_HISTORY];

    /** Smoothed output (state at time k-L) */
    double x_smooth_lag[KF_MAX_STATE_DIM];

    /** Smoothed covariance */
    double P_smooth_lag[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Lag length */
    uint32_t lag;

    /** Current time index */
    uint32_t current_idx;

    /** State dimension */
    uint8_t n;

    /** Buffer write position */
    uint32_t buf_pos;
} FixedLagSmoother;

/* --------------------------------------------------------------------------
 * L5: Smoother API
 * -------------------------------------------------------------------------- */

/**
 * Initialize forward-pass history buffer.
 *
 * @param hist     History buffer to initialize
 * @param capacity Maximum number of steps to store
 * @param n        State dimension
 * @param m        Measurement dimension
 */
void kf_history_init(KFHistory *hist, uint32_t capacity, uint8_t n, uint8_t m);

/**
 * Store one forward-pass result in the history buffer.
 *
 * @param hist    History buffer
 * @param kf      Current KF state to copy
 * @param model   Model (for storing F matrix)
 * @param step    Time step index
 */
void kf_history_store(KFHistory *hist, const KalmanFilterState *kf,
                      const KalmanModel *model, uint32_t step);

/**
 * Rauch-Tung-Striebel fixed-interval smoother.
 *
 * Processes the stored history backward from N-1 to 0:
 *   G[k] = P[k|k] * F'' * inv(P[k+1|k])
 *   x[k|N] = x[k|k] + G[k] * (x[k+1|N] - x[k+1|k])
 *   P[k|N] = P[k|k] + G[k] * (P[k+1|N] - P[k+1|k]) * G[k]''
 *
 * @param hist        Stored forward-pass history (N entries)
 * @param model       System model (F matrix used)
 * @param rts         RTS smoother workspace
 * @param x_smoothed  Output: all smoothed states, (hist->count) x n, row-major
 * @param P_smoothed  Output: all smoothed covariances, (hist->count) x (n x n), row-major
 *
 * Complexity: O(N * n^3)
 * Reference: Rauch, Tung, Striebel (1965) AIAA Journal
 */
void kf_rts_smooth_full(const KFHistory *hist, const KalmanModel *model,
                         RTSSmoother *rts,
                         double *x_smoothed, double *P_smoothed);

/**
 * Two-Filter (Fraser-Potter) smoother.
 *
 * Runs a forward filter and a backward information filter independently,
 * then combines the estimates. More numerically stable for some problems.
 *
 * x[k|N] = P[k|N] * (inv(P_f[k|k]) * x_f[k|k] + inv(P_b[k|k+1]) * x_b[k|k+1])
 * inv(P[k|N]) = inv(P_f[k|k]) + inv(P_b[k|k+1])
 *
 * Complexity: O(N * n^3)
 * Reference: Fraser & Potter (1969) "The Optimum Linear Smoother..."
 */
void kf_two_filter_smooth(const KFHistory *hist, const KalmanModel *model,
                           double *x_smoothed, double *P_smoothed);

/**
 * Initialize a fixed-lag smoother.
 *
 * Provides smoothed estimates with a constant delay of L steps,
 * suitable for soft-sensor applications where a small delay is acceptable
 * in exchange for improved accuracy.
 *
 * @param fls  Fixed-lag smoother
 * @param lag  Number of steps to delay
 * @param n    State dimension
 */
void kf_fixed_lag_init(FixedLagSmoother *fls, uint32_t lag, uint8_t n);

/**
 * Update fixed-lag smoother with a new forward-pass result.
 *
 * Produces the smoothed estimate for time k-L.
 *
 * @param fls   Fixed-lag smoother
 * @param kf    Current KF state (just after update at time k)
 * @param model System model
 *
 * Complexity: O(L * n^3)
 * Reference: Moore (1973) "Discrete-Time Fixed-Lag Smoothing Algorithms"
 */
void kf_fixed_lag_update(FixedLagSmoother *fls,
                          const KalmanFilterState *kf,
                          const KalmanModel *model);

/**
 * Get the current fixed-lag smoothed state.
 *
 * @param fls  Fixed-lag smoother
 * @return Pointer to x_smooth_lag, size n
 */
const double *kf_fixed_lag_get_state(const FixedLagSmoother *fls);

/**
 * Fixed-point smoother: estimate state at a specific fixed time point t_fixed.
 *
 * As new measurements arrive, the estimate of x[t_fixed] improves.
 * Useful for initial condition estimation and batch process end-point quality.
 *
 * @param kf         Current KF state
 * @param model      System model
 * @param x_fixed    Smoothed estimate of x[t_fixed] (in-place updated)
 * @param P_fixed    Smoothed covariance of x[t_fixed] (in-place updated)
 * @param S_fixed    Smoothing gain (workspace, n x n)
 * @param n          State dimension
 *
 * Complexity: O(n^3)
 * Reference: Meditch (1969) "Stochastic Optimal Linear Estimation and Control"
 */
void kf_fixed_point_smooth(const KalmanFilterState *kf,
                            const KalmanModel *model,
                            double *x_fixed, double *P_fixed,
                            double *S_fixed, uint8_t n);

/**
 * Compute smoothing improvement: compare filtered vs smoothed covariance.
 *
 * Returns the trace ratio: trace(P_smooth) / trace(P_filter)
 * Values < 1 indicate improvement from smoothing.
 *
 * @param P_filter  Filtered covariance, n x n
 * @param P_smooth  Smoothed covariance, n x n
 * @param n         State dimension
 * @return Ratio of traces (<= 1.0 for valid smoothing, > 1.0 indicates issues)
 */
double kf_smoothing_improvement(const double *P_filter,
                                 const double *P_smooth, uint8_t n);

#ifdef __cplusplus
}
#endif
#endif /* KALMAN_SMOOTHER_H */
