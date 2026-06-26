/**
 * @file kalman_extended.h
 * @brief Extended Kalman Filter (EKF) for nonlinear systems
 *
 * L2 Core Concepts: Nonlinear state estimation, linearization
 * L5 Algorithms: First-order Taylor expansion, Jacobian computation
 *
 * The EKF linearizes the nonlinear system around the current estimate
 * using Jacobian matrices, then applies standard Kalman filter equations.
 *
 * System:  x[k] = f(x[k-1], u[k-1]) + w[k-1],  w ~ N(0, Q)
 * Measure: z[k] = h(x[k]) + v[k],                v ~ N(0, R)
 *
 * Linearized: F[k] = df/dx | x[k-1|k-1],  H[k] = dh/dx | x[k|k-1]
 *
 * Reference: Jazwinski, A.H. (1970) "Stochastic Processes and Filtering Theory"
 * Course alignment: Stanford AA272, Berkeley ME233, CMU 24-677
 */
#ifndef KALMAN_EXTENDED_H
#define KALMAN_EXTENDED_H

#include "kalman_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * L1: EKF-specific type definitions
 * -------------------------------------------------------------------------- */

/** Function pointer type for nonlinear state transition f(x, u) */
typedef void (*ekf_state_transition_fn)(const double *x, const double *u,
                                         uint8_t n, uint8_t p, double *x_next);

/** Function pointer type for nonlinear measurement function h(x) */
typedef void (*ekf_measurement_fn)(const double *x, uint8_t n,
                                    uint8_t m, double *z_pred);

/** Function pointer type for state Jacobian F = df/dx */
typedef void (*ekf_state_jacobian_fn)(const double *x, const double *u,
                                       uint8_t n, uint8_t p, double *F_out);

/** Function pointer type for measurement Jacobian H = dh/dx */
typedef void (*ekf_meas_jacobian_fn)(const double *x, uint8_t n,
                                      uint8_t m, double *H_out);

/**
 * L1: EKFModel — extended Kalman filter model
 *
 * Stores function pointers for nonlinear dynamics and measurement models.
 * Users provide their own f, h, and Jacobian implementations.
 */
typedef struct {
    /** Nonlinear state transition: x[k+1] = f(x[k], u[k]) */
    ekf_state_transition_fn f;

    /** Nonlinear measurement: z[k] = h(x[k]) */
    ekf_measurement_fn h;

    /** State transition Jacobian: F[k] = df/dx evaluated at x[k-1|k-1] */
    ekf_state_jacobian_fn F_jacobian;

    /** Measurement Jacobian: H[k] = dh/dx evaluated at x[k|k-1] */
    ekf_meas_jacobian_fn H_jacobian;

    /** Process noise covariance Q, n x n */
    double Q[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Measurement noise covariance R, m x m */
    double R[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];

    /** State dimension */
    uint8_t n;

    /** Measurement dimension */
    uint8_t m;

    /** Control input dimension */
    uint8_t p;

    /** Flags */
    uint8_t initialized : 1;
    uint8_t reserved    : 7;
} EKFModel;

/**
 * L1: EKFState — EKF state with function pointer model reference
 */
typedef struct {
    /** Kalman filter state (standard linear KF state structure) */
    KalmanFilterState kf;

    /** Pointer to the EKF model */
    const EKFModel *model;

    /** Workspace for temporary Jacobian F, n x n */
    double F_jac_workspace[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];

    /** Workspace for temporary Jacobian H, m x n */
    double H_jac_workspace[KF_MAX_MEAS_DIM * KF_MAX_STATE_DIM];

    /** Predicted measurement h(x[k|k-1]), size m */
    double z_pred[KF_MAX_MEAS_DIM];

    /** Iteration count for iterative EKF */
    uint32_t iter_count;

    /** Flag: use iterative EKF (re-linearize at updated estimate) */
    uint8_t use_iterative : 1;
    uint8_t use_second_order : 1;
    uint8_t reserved      : 6;
} EKFState;

/* --------------------------------------------------------------------------
 * L2-L5: EKF API
 * -------------------------------------------------------------------------- */

/**
 * Initialize an Extended Kalman Filter.
 *
 * @param ekf    EKF state to initialize
 * @param model  EKF model with function pointers
 * @param x0     Initial state estimate, size n
 * @param P0     Initial covariance matrix, size n x n, row-major
 * @param n      State dimension
 * @param m      Measurement dimension
 * @param p      Control input dimension
 *
 * Complexity: O(n^2)
 */
void ekf_init(EKFState *ekf, const EKFModel *model,
              const double *x0, const double *P0,
              uint8_t n, uint8_t m, uint8_t p);

/**
 * EKF prediction step using nonlinear state transition.
 *
 * 1. Compute Jacobian: F[k] = df/dx | x[k-1|k-1], u[k-1]
 * 2. Propagate state: x[k|k-1] = f(x[k-1|k-1], u[k-1])
 * 3. Propagate covariance: P[k|k-1] = F[k] * P[k-1|k-1] * F[k]'' + Q
 *
 * @param ekf  EKF state
 * @param u    Control input, size p (NULL if no control)
 *
 * Complexity: O(n^3) for covariance propagation + user''s f() cost
 */
void ekf_predict(EKFState *ekf, const double *u);

/**
 * EKF update step using nonlinear measurement function.
 *
 * 1. Predict measurement: z_pred = h(x[k|k-1])
 * 2. Compute Jacobian: H[k] = dh/dx | x[k|k-1]
 * 3. Compute innovation: y = z - z_pred
 * 4. Compute innovation covariance: S = H*P*H'' + R
 * 5. Compute Kalman gain: K = P*H''*inv(S)
 * 6. Update state: x = x + K*y
 * 7. Update covariance: P = (I - K*H)*P
 *
 * @param ekf  EKF state
 * @param z    Measurement vector, size m
 *
 * Complexity: O(n^3 + n^2*m + n*m^2 + m^3) + user''s h() cost
 */
void ekf_update(EKFState *ekf, const double *z);

/**
 * Single combined EKF predict+update step.
 *
 * @param ekf  EKF state
 * @param z    Measurement vector
 * @param u    Control input (NULL if none)
 * @return Pointer to posterior state estimate
 */
const double *ekf_step(EKFState *ekf, const double *z, const double *u);

/**
 * Iterated EKF update: re-linearize H at the updated state estimate.
 *
 * Repeats the update with H evaluated at x[k|k] instead of x[k|k-1],
 * reducing linearization error for highly nonlinear measurement functions.
 *
 * @param ekf       EKF state
 * @param z         Measurement vector
 * @param max_iter  Maximum iterations (typically 3-5)
 *
 * Complexity: O(iter * (n^3 + m^3))
 * Reference: Bell & Cathey (1993) "The Iterated Kalman Filter Update..."
 */
void ekf_update_iterated(EKFState *ekf, const double *z, uint8_t max_iter);

/**
 * Second-order EKF: include Hessian term in covariance propagation.
 *
 * Adds the contribution of the second derivative (curvature) of the
 * measurement function to reduce bias from neglected higher-order terms.
 *
 * @param ekf          EKF state
 * @param meas_hessian Array of m Hessian matrices, each n x n, row-major
 *
 * Complexity: O(m * n^3)
 * Reference: Athans, Wishner, Bertolini (1968) "Suboptimal state estimation..."
 */
void ekf_second_order_correction(EKFState *ekf,
                                  const double *meas_hessian);

/**
 * Compute the normalized innovation squared (NIS) for the EKF.
 *
 * NIS = y'' * inv(S) * y, distributed as chi^2(m) under correct model.
 * Used for filter consistency checking and fault detection.
 *
 * @param ekf  EKF state after update
 * @return NIS value
 *
 * Reference: Bar-Shalom, Li, Kirubarajan (2001) "Estimation with Applications..."
 */
double ekf_nis(const EKFState *ekf);

/**
 * EKF consistency check using NEES and NIS tests.
 *
 * @param ekf  EKF state
 * @return 1 if consistent, 0 otherwise
 */
int ekf_is_consistent(const EKFState *ekf);

#ifdef __cplusplus
}
#endif
#endif /* KALMAN_EXTENDED_H */
