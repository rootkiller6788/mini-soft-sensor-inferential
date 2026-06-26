/**
 * @file kalman_unscented.h
 * @brief Unscented Kalman Filter (UKF) — Sigma-Point Kalman Filter
 *
 * L5 Algorithms: Unscented transform, sigma point generation
 * L8 Advanced Topics: Derivative-free nonlinear filtering
 *
 * The UKF uses a deterministic sampling approach (the unscented transform)
 * to capture the mean and covariance of a nonlinear transformation without
 * requiring Jacobian computation. It achieves 3rd-order accuracy for
 * Gaussian inputs vs EKF''s 1st-order.
 *
 * Reference: Julier & Uhlmann (1997) "A New Extension of the Kalman Filter
 *   to Nonlinear Systems." Proc. SPIE 3068, 182-193.
 *
 * Course alignment: Stanford ENGR205, Georgia Tech AE 6530, Purdue ME 575
 */
#ifndef KALMAN_UNSCENTED_H
#define KALMAN_UNSCENTED_H

#include "kalman_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * L1: UKF-specific type definitions
 * -------------------------------------------------------------------------- */

/** Maximum sigma points = 2*n + 1 (symmetric set) */
#define UKF_MAX_SIGMA_PTS (2 * KF_MAX_STATE_DIM + 1)

/** Default UKF scaling parameter alpha (spread of sigma points) */
#define UKF_DEFAULT_ALPHA 1.0

/** Default UKF scaling parameter beta (prior knowledge of distribution, 2=Gaussian) */
#define UKF_DEFAULT_BETA  2.0

/** Default UKF scaling parameter kappa (secondary scaling, 0 for state est) */
#define UKF_DEFAULT_KAPPA 0.0

/**
 * L1: UKF function pointer types for process and measurement models.
 *
 * Unlike the EKF, the UKF only needs the nonlinear functions themselves,
 * not their Jacobians.
 */
typedef void (*ukf_process_fn)(const double *x, const double *u,
                                uint8_t n, uint8_t p, double *x_out);

typedef void (*ukf_measure_fn)(const double *x, uint8_t n,
                                uint8_t m, double *z_out);

/**
 * L1: UKFParams — UKF tuning/scaling parameters
 *
 * The unscented transform generates 2n+1 sigma points:
 *   X_0 = x_mean,  w_0^m = lambda/(n+lambda), w_0^c = w_0^m + (1-alpha^2+beta)
 *   X_i = x_mean + sqrt((n+lambda)*P)_i, for i = 1..n
 *   X_{i+n} = x_mean - sqrt((n+lambda)*P)_i, for i = 1..n
 *   w_i^m = w_i^c = 1/(2*(n+lambda)), for i = 1..2n
 *
 * where lambda = alpha^2 * (n + kappa) - n
 */
typedef struct {
    /** Primary scaling parameter (0.001 to 1.0) */
    double alpha;

    /** Distribution prior (2.0 for Gaussian) */
    double beta;

    /** Secondary scaling (0 or 3-n) */
    double kappa;

    /** Composite scaling: lambda = alpha^2 * (n + kappa) - n */
    double lambda;

    /** sqrt(n + lambda) — precomputed for efficiency */
    double sqrt_n_plus_lambda;

    /** Weight for mean of 0th sigma point: w_0^m = lambda/(n+lambda) */
    double w_mean_0;

    /** Weight for covariance of 0th sigma point: w_0^c = w_mean_0 + (1-alpha^2+beta) */
    double w_cov_0;

    /** Weight for mean and cov of i-th sigma point (i >= 1): 1/(2*(n+lambda)) */
    double w_i;

    /** State dimension */
    uint8_t n;
} UKFParams;

/**
 * L1: UKFState — Unscented Kalman Filter state
 */
typedef struct {
    /** Core linear KF state (holds x, P, etc.) */
    KalmanFilterState kf;

    /** UKF tuning parameters */
    UKFParams params;

    /** Sigma points: (2n+1) points, each of dimension n, row-major */
    double sigma_points[UKF_MAX_SIGMA_PTS * KF_MAX_STATE_DIM];

    /** Transformed sigma points (through process model), row-major */
    double sigma_points_f[UKF_MAX_SIGMA_PTS * KF_MAX_STATE_DIM];

    /** Predicted measurement sigma points (through measurement model), row-major */
    double sigma_points_h[UKF_MAX_SIGMA_PTS * KF_MAX_MEAS_DIM];

    /** Predicted measurement mean, size m */
    double z_pred[KF_MAX_MEAS_DIM];

    /** Cross-covariance P_xz, n x m, row-major */
    double P_xz[KF_MAX_STATE_DIM * KF_MAX_MEAS_DIM];

    /** Innovation covariance P_zz, m x m, row-major */
    double P_zz[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];

    /** Function pointers */
    ukf_process_fn process_fn;
    ukf_measure_fn measure_fn;

    /** Control input dimension */
    uint8_t p;

    /** Flags */
    uint8_t initialized : 1;
    uint8_t reserved    : 7;
} UKFState;

/* --------------------------------------------------------------------------
 * L5: UKF API — derivative-free nonlinear filtering
 * -------------------------------------------------------------------------- */

/**
 * Initialize UKF parameters.
 *
 * Sets alpha, beta, kappa and computes derived parameters lambda,
 * sqrt(n+lambda), and sigma point weights.
 *
 * @param params  UKF params to initialize
 * @param n       State dimension
 * @param alpha   Spread parameter (UKF_DEFAULT_ALPHA = 1.0)
 * @param beta    Distribution knowledge (UKF_DEFAULT_BETA = 2.0)
 * @param kappa   Secondary scaling (UKF_DEFAULT_KAPPA = 0.0)
 *
 * Complexity: O(1)
 * Reference: Wan & van der Merwe (2000) "The Unscented Kalman Filter..."
 */
void ukf_params_init(UKFParams *params, uint8_t n,
                     double alpha, double beta, double kappa);

/**
 * Initialize the UKF state.
 *
 * @param ukf      UKF state
 * @param x0       Initial state, size n
 * @param P0       Initial covariance, n x n, row-major
 * @param proc_fn  Process model function
 * @param meas_fn  Measurement function
 * @param n        State dimension
 * @param m        Measurement dimension
 * @param p        Control input dimension
 * @param params   UKF parameters
 *
 * Complexity: O(n^2)
 */
void ukf_init(UKFState *ukf,
              const double *x0, const double *P0,
              ukf_process_fn proc_fn, ukf_measure_fn meas_fn,
              uint8_t n, uint8_t m, uint8_t p,
              const UKFParams *params);

/**
 * Generate sigma points from current state mean and covariance.
 *
 * Uses Cholesky decomposition of P to compute sqrt(P):
 *   sigma_0 = x
 *   sigma_i = x + sqrt((n+lambda)*P)_i     for i = 1..n
 *   sigma_{i+n} = x - sqrt((n+lambda)*P)_i  for i = 1..n
 *
 * @param ukf  UKF state (reads ukf->kf.x, ukf->kf.P)
 *
 * Complexity: O(n^3) for Cholesky decomposition
 * Reference: Julier & Uhlmann (2004) "Unscented Filtering and Nonlinear Estimation"
 */
void ukf_generate_sigma_points(UKFState *ukf);

/**
 * UKF prediction step:
 * 1. Generate sigma points from current state
 * 2. Propagate each sigma point through the process model
 * 3. Recombine to get predicted mean and covariance
 *
 * x[k|k-1] = sum_i w_i * f(X_i)
 * P[k|k-1] = sum_i w_i * (f(X_i) - x[k|k-1]) * (f(X_i) - x[k|k-1])' + Q
 *
 * @param ukf  UKF state
 * @param u    Control input (NULL if none), size p
 * @param Q    Process noise covariance, n x n, row-major
 *
 * Complexity: O(n^3 + (2n+1) * cost_of_f)
 */
void ukf_predict(UKFState *ukf, const double *u, const double *Q);

/**
 * UKF update step:
 * 1. Propagate sigma points through measurement model
 * 2. Compute predicted measurement mean and innovation covariance
 * 3. Compute cross-covariance between state and measurement
 * 4. Compute Kalman gain and update
 *
 * K = P_xz * inv(P_zz)
 * x = x + K * (z - z_pred)
 * P = P - K * P_zz * K'
 *
 * @param ukf  UKF state
 * @param z    Measurement vector, size m
 * @param R    Measurement noise covariance, m x m, row-major
 *
 * Complexity: O(n^2*m + n*m^2 + m^3 + (2n+1) * cost_of_h)
 */
void ukf_update(UKFState *ukf, const double *z, const double *R);

/**
 * Single UKF predict+update step.
 *
 * @return Pointer to posterior state estimate
 */
const double *ukf_step(UKFState *ukf, const double *z, const double *u,
                       const double *Q, const double *R);

/**
 * Square-root UKF: propagate Cholesky factors instead of full covariance.
 *
 * Uses QR decomposition and Cholesky downdate to maintain sqrt(P) directly,
 * guaranteeing positive semi-definiteness and improving numerical stability.
 *
 * @param ukf    UKF state
 * @param u      Control input (NULL if none)
 * @param sqrt_Q Cholesky factor of Q: Q = sqrt_Q * sqrt_Q'
 *
 * Complexity: O(n^3)
 * Reference: van der Merwe & Wan (2001) "The Square-Root Unscented Kalman Filter..."
 */
void ukf_predict_sqrt(UKFState *ukf, const double *u, const double *sqrt_Q);

/**
 * Square-root UKF update using QR/Cholesky.
 *
 * @param ukf    UKF state
 * @param z      Measurement vector
 * @param sqrt_R Cholesky factor of R: R = sqrt_R * sqrt_R'
 */
void ukf_update_sqrt(UKFState *ukf, const double *z, const double *sqrt_R);

/**
 * Compute UKF NIS for filter consistency monitoring.
 *
 * @param ukf  UKF state after update
 * @return NIS value ~ chi^2(m)
 */
double ukf_nis(const UKFState *ukf);

#ifdef __cplusplus
}
#endif
#endif /* KALMAN_UNSCENTED_H */
