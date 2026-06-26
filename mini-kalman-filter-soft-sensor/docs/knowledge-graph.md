# Knowledge Graph — mini-kalman-filter-soft-sensor

## L1: Definitions (Complete)

| # | Definition | C Type | Lean Type | Status |
|---|-----------|--------|-----------|--------|
| 1 | State vector x (n x 1) | `double x[KF_MAX_STATE_DIM]` | `StateVector n` | DONE |
| 2 | Error covariance P (n x n) | `double P[...]` | `List (List Float)` | DONE |
| 3 | Kalman gain K (n x m) | `double K[...]` | — | DONE |
| 4 | Innovation y (m x 1) | `double innovation[...]` | — | DONE |
| 5 | Innovation covariance S (m x m) | `double S[...]` | — | DONE |
| 6 | State transition F (n x n) | `double F[...]` | `LinearSystemModel.F` | DONE |
| 7 | Observation matrix H (m x n) | `double H[...]` | `LinearSystemModel.H` | DONE |
| 8 | Process noise Q (n x n) | `double Q[...]` | `LinearSystemModel.Q` | DONE |
| 9 | Measurement noise R (m x m) | `double R[...]` | `LinearSystemModel.R` | DONE |
| 10 | Control input B (n x p) | `double B[...]` | `LinearSystemModel.B` | DONE |
| 11 | NIS (chi-square statistic) | `KalmanDiagnostics.nis` | `nis_chi_squared_distribution` theorem | DONE |
| 12 | Log-likelihood | `kf_log_likelihood()` | — | DONE |
| 13 | Sigma points (2n+1) | `UKFState.sigma_points` | — | DONE |
| 14 | UKF scaling params | `UKFParams` | — | DONE |
| 15 | Model probability mu | `AKFMultiModel.mu` | — | DONE |
| 16 | Smoothing gain G | `RTSSmoother.G` | — | DONE |

## L2: Core Concepts (Complete)

| # | Concept | Implementation | Status |
|---|---------|---------------|--------|
| 1 | Prediction-correction cycle | `kf_predict()` + `kf_update()` | DONE |
| 2 | Minimum mean square error (MMSE) | `kf_step()` returns optimal estimate | DONE |
| 3 | Bayesian filtering | Prior/posterior in `KalmanFilterState` | DONE |
| 4 | Gaussian assumption | N(0,Q), N(0,R) in model docs | DONE |
| 5 | Innovation sequence | `kf_get_innovation()`, `akf_update_innovation_stats()` | DONE |
| 6 | Filter convergence | `kf_has_converged()`, `KFStep.converged` | DONE |
| 7 | Observability | DARE solvability check in `kf_solve_dare()` | DONE |
| 8 | Information form | `kf_sequential_update()` avoids matrix inverse | DONE |
| 9 | Fading memory | `kf_fading_memory_predict()` | DONE |

## L3: Engineering Structures (Complete)

| # | Structure | Implementation | Status |
|---|-----------|---------------|--------|
| 1 | Matrix multiply (row-major) | `mat_mul()`, `mat_mul_AT_B()`, `mat_mul_A_BT()` | DONE |
| 2 | Cholesky decomposition | `mat_cholesky()`, `mat_solve_cholesky()` | DONE |
| 3 | QR decomposition (MGS + Householder) | `mat_qr_mgs()`, `mat_qr_householder()` | DONE |
| 4 | Matrix inverse (Cholesky-based) | `mat_inverse_cholesky()` | DONE |
| 5 | Log-determinant | `mat_logdet_cholesky()` | DONE |
| 6 | Forward/back substitution | `mat_forward_sub()`, `mat_back_sub()` | DONE |
| 7 | Eigenvalue computation (2x2, power iter) | `mat_eigen_2x2()`, `mat_power_iteration()` | DONE |
| 8 | Condition number estimation | `mat_cond_estimate_sym()` | DONE |
| 9 | Symmetry / PSD checks | `mat_is_symmetric()`, `mat_is_positive_definite()` | DONE |
| 10 | Rank-1 Cholesky update | `mat_cholesky_rank1_update()` | DONE |
| 11 | Sigma point generation | `ukf_generate_sigma_points()` | DONE |

## L4: Engineering Laws (Complete)

| # | Law/Theorem | C Implementation | Lean Theorem | Status |
|---|-----------|-----------------|-------------|--------|
| 1 | State prediction eq | `kf_predict()` | `state_prediction_eq` | DONE |
| 2 | Covariance prediction eq | `kf_predict()` | `covariance_prediction_eq` | DONE |
| 3 | Innovation eq | `kf_update()` | `innovation_eq` | DONE |
| 4 | Innovation covariance eq | `kf_update()` | `innovation_covariance_eq` | DONE |
| 5 | Kalman gain eq | `kf_update()` | `kalman_gain_eq` | DONE |
| 6 | State update eq | `kf_update()` | `state_update_eq` | DONE |
| 7 | Covariance update eq | `kf_update()` | `covariance_update_eq` | DONE |
| 8 | Joseph form | `kf_joseph_update()` | `joseph_form_preserves_psd` | DONE |
| 9 | Riccati convergence | `kf_solve_dare()` | `riccati_convergence` | DONE |
| 10 | Innovation whiteness | `kf_diagnostics()` | `innovation_whiteness` | DONE |
| 11 | NIS ~ chi-square(m) | `kf_diagnostics()` NIS fault flag | `nis_chi_squared_distribution` | DONE |
| 12 | Steady-state gain optimality | `kf_solve_dare()` | `steady_state_gain_optimal` | DONE |
| 13 | Cramer-Rao lower bound | `kf_get_covariance()` | — | DONE |

## L5: Algorithms/Methods (Complete)

| # | Algorithm | File | Complexity | Status |
|---|----------|------|-----------|--------|
| 1 | Standard Linear KF | `kalman_core.c` | O(n^3 + m^3) | DONE |
| 2 | Extended KF (EKF) | `kalman_extended.c` | O(n^3 + cost of f,h) | DONE |
| 3 | Iterated EKF | `kalman_extended.c:ekf_update_iterated()` | O(iter * n^3) | DONE |
| 4 | Second-order EKF | `kalman_extended.c:ekf_second_order_correction()` | O(m * n^3) | DONE |
| 5 | Unscented KF (UKF) | `kalman_unscented.c` | O((2n+1) * cost) | DONE |
| 6 | Square-root UKF | `kalman_unscented.c:ukf_predict_sqrt()` | O((2n+1) * cost) | DONE |
| 7 | Sequential update | `kalman_core.c:kf_sequential_update()` | O(m * n^2) | DONE |
| 8 | Joseph-form update | `kalman_core.c:kf_joseph_update()` | O(n^3) | DONE |
| 9 | Fading memory KF | `kalman_core.c:kf_fading_memory_predict()` | O(n^3) | DONE |
| 10 | DARE iterative solver | `kalman_core.c:kf_solve_dare()` | O(n^3 * iter) | DONE |
| 11 | RTS smoother | `kalman_smoother.c:kf_rts_smooth_full()` | O(N * n^3) | DONE |
| 12 | Two-filter smoother | `kalman_smoother.c:kf_two_filter_smooth()` | O(N * n^3) | DONE |
| 13 | Fixed-lag smoother | `kalman_smoother.c:kf_fixed_lag_update()` | O(L * n^3) | DONE |
| 14 | Fixed-point smoother | `kalman_smoother.c:kf_fixed_point_smooth()` | O(n^3) | DONE |
| 15 | Covariance matching (Q est) | `kalman_adaptive.c:akf_estimate_Q_covariance_matching()` | O(n^3) | DONE |
| 16 | Innovation correlation (R est) | `kalman_adaptive.c:akf_estimate_R_innovation()` | O(n^3+m^3) | DONE |
| 17 | Bayesian MAP (Q,R est) | `kalman_adaptive.c:akf_estimate_bayesian()` | O(n^3+m^3) | DONE |
| 18 | IMM estimation | `kalman_adaptive.c:imm_step()` | O(N * n^3) | DONE |
| 19 | Log-likelihood evaluation | `kalman_core.c:kf_log_likelihood()` | O(m^3) | DONE |
| 20 | Filter diagnostics | `kalman_core.c:kf_diagnostics()` | O(n^3+m^3) | DONE |

## L6: Canonical Problems (Complete)

| # | Problem | Implementation | Status |
|---|---------|---------------|--------|
| 1 | DC motor velocity estimation | `dc_motor_estimator_step()` | DONE |
| 2 | CSTR concentration inference | `cstr_estimator_step()` (EKF) | DONE |
| 3 | GPS/INS navigation (9-state) | `ins_gps_predict_imu()` + `ins_gps_update_gps()` | DONE |
| 4 | DC motor torque estimation | `dc_motor_get_torque()` | DONE |
| 5 | CSTR conversion estimation | `cstr_get_conversion()` | DONE |

## L7: Industrial Applications (Complete - 3 applications)

| # | Application | Implementation | Keywords | Status |
|---|------------|---------------|----------|--------|
| 1 | Quality soft sensor (polymer melt index) | `quality_estimator_update()` | Honeywell, OSIsoft PI | DONE |
| 2 | DCS/historian data snapshot | `historian_create_snapshot()` | OPC quality, PI | DONE |
| 3 | Sensor fault detection | `quality_estimator_check_sensors()` | Innovation analysis | DONE |

## L8: Advanced Topics (Partial+ - 5 topics, 3 implemented)

| # | Topic | Implementation | Status |
|---|-------|---------------|--------|
| 1 | Adaptive Kalman filtering | `kalman_adaptive.c` | DONE |
| 2 | Multiple model estimation (IMM) | `imm_step()` | DONE |
| 3 | Square-root filtering | `ukf_predict_sqrt()`, `ukf_update_sqrt()` | DONE |
| 4 | Particle filtering | — (documented in L9) | PLANNED |
| 5 | Distributed Kalman filtering | — (documented in L9) | PLANNED |

## L9: Research Frontiers (Partial - documented)

| # | Topic | Reference | Status |
|---|-------|----------|--------|
| 1 | Deep Kalman Filter (NN + KF) | Krishnan et al. (2015) | DOCUMENTED |
| 2 | Ensemble KF (Monte Carlo) | Evensen (2003) | DOCUMENTED |
| 3 | Information-geometric KF | Amari (1998) | DOCUMENTED |
| 4 | Distributed consensus KF | Olfati-Saber (2007) | DOCUMENTED |
| 5 | Edge AI with KF | — | DOCUMENTED |
