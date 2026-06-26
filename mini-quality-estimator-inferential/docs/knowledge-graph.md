# Knowledge Graph — Inferential Quality Estimation

## L1: Definitions (Complete ✅)

| # | Term | C Representation | Lean Representation |
|---|------|-----------------|---------------------|
| 1 | Quality Estimator (Soft Sensor) | `quality_estimator_t` | — |
| 2 | Inferential Model | `fpm_model_t`, `linear_model_t`, `pls_model_t` | `QualityModelType` |
| 3 | Bias Correction | `bias_additive_t`, `bias_multiplicative_t` | `BiasStrategy` |
| 4 | Lab Measurement | `lab_sample_t`, `lab_quality_t` | `LabQuality` |
| 5 | Quality Estimate | `quality_estimate_t` | — |
| 6 | Estimator Mode | `qest_mode_t` | `EstimatorMode` |
| 7 | Maintenance State | `maintenance_state_t` | `MaintenanceState` |
| 8 | Process Variable | `process_variable_t` | — |
| 9 | Model Types | `quality_model_type_t` | `QualityModelType` |
| 10 | Data Subtypes | `data_model_subtype_t` | — |
| 11 | Confidence Bounds | `lower_bound_95`, `upper_bound_95` | `confidenceHalfWidth` |
| 12 | Prediction Variance | `prediction_variance` | — |

## L2: Core Concepts (Complete ✅)

| # | Concept | Implementation |
|---|---------|---------------|
| 1 | First-principles model | `fpm_evaluate()`, `fpm_sensitivity()` |
| 2 | Data-driven model (PLS) | `pls_model_predict()`, `pls_mahalanobis_distance()` |
| 3 | Hybrid grey-box model | `hybrid_evaluate()`, `hybrid_set_correction()` |
| 4 | Additive bias correction | `bias_additive_update()`, `bias_additive_correct()` |
| 5 | Multiplicative bias correction | `bias_mult_update()`, `bias_mult_correct()` |
| 6 | Model validation | `compute_residuals()`, `compute_regression_stats()` |
| 7 | Linear regression | `linear_model_evaluate()` |
| 8 | Lab sample validation | `lab_sample_validate()` |
| 9 | Estimator health monitoring | `qest_performance_t`, `maintenance_state_t` |
| 10 | Instrumental Variables | `iv_estimator_t`, `iv_estimator_update()` |

## L3: Engineering Structures (Complete ✅)

| # | Structure | Implementation |
|---|-----------|---------------|
| 1 | State-space model | `ss_model_t`, `ss_model_alloc()`, `ss_model_predict()` |
| 2 | ARX dynamic model | `arx_model_t`, `arx_predict()`, `arx_update_buffers()` |
| 3 | Multi-rate data structures | `multi_rate_context_t` |
| 4 | Timestamp handling | `qest_timestamp_t`, `qest_timestamp_diff_seconds()` |
| 5 | MHE buffer (circular) | `mhe_buffer_t`, `mhe_push()` |
| 6 | PLS latent structure | `pls_model_t` with W, P, Q matrices |
| 7 | Bias context dispatcher | `bias_context_t` (strategy pattern) |
| 8 | EWMA-trend dual filtering | `bias_ewma_trend_t` |

## L4: Engineering Laws/Standards (Complete ✅)

| # | Law/Standard | Implementation |
|---|-------------|---------------|
| 1 | Bias-variance-noise decomposition | MSE = Var(pred) + Bias² + Var(noise) — Lean: `mseDecomposition` |
| 2 | Durbin-Watson test | `durbin_watson_test()` — Lean: `durbinWatson` |
| 3 | Student's t-test for zero mean | `t_test_zero_mean()` |
| 4 | Grubbs' outlier test | `grubbs_outlier_test()` |
| 5 | EWMA convergence | Lean: `bias_update_contracts_to_residual` |
| 6 | Confidence interval coverage | Normal CDF `normal_cdf()`, quantile `normal_quantile()` |
| 7 | Kalman filter optimality | Lean: `kalman_gain_optimal` (axiom — Kalman 1960 theorem) |
| 8 | Hotelling's T² | `pls_hotelling_t2()` |
| 9 | ARX BIBO stability condition | Lean: `arx_bibo_stability_ar2` |

## L5: Algorithms (Complete ✅)

| # | Algorithm | Implementation |
|---|-----------|---------------|
| 1 | Linear Kalman Filter | `kf_alloc()`, `kf_predict()`, `kf_update()`, `kf_step()` |
| 2 | Extended Kalman Filter | `ekf_alloc()`, `ekf_step()` |
| 3 | Adaptive Kalman Filter | `akf_alloc()`, `akf_step()` (Mehra's method) |
| 4 | Recursive Least Squares | `rls_alloc()`, `rls_update()`, `rls_predict()` |
| 5 | RLS with Variable Forgetting | `rls_vff_alloc()`, `rls_vff_update()` |
| 6 | RLS with Directional Forgetting | `rls_directional_alloc()`, `rls_directional_update()` |
| 7 | CUSUM drift detection | `cusum_init()`, `cusum_update()`, `cusum_reset()` |
| 8 | Kalman bias estimation | `bias_kalman_init()`, `bias_kalman_update()` |
| 9 | Moving Horizon Estimation | `mhe_init()`, `mhe_push()`, `mhe_estimate()` |
| 10 | Windowed WLS | `wls_window_alloc()`, `wls_window_update()` |
| 11 | Cubature Kalman Filter | `ckf_alloc()`, `ckf_step()` (CKF, Arasaratnam 2009) |
| 12 | Recursive PLS | `rpls_init()`, `rpls_update()`, `rpls_predict()` |
| 13 | MAD outlier detection | `mad_outlier_detect()` |
| 14 | Instrumental Variable | `iv_estimator_update()` |
| 15 | EWMA smoothing | `ewma_smooth()` |

## L6: Canonical Problems (Complete ✅)

| # | Problem | Example / Test |
|---|---------|---------------|
| 1 | Distillation composition | `example_distillation_quality.c`, `test_distillation_composition()` |
| 2 | Reactor conversion (CSTR) | `test_reactor_conversion()` — Arrhenius kinetics |
| 3 | Polymer Melt Index | `example_polymer_melt_index.c` |
| 4 | NOx emissions monitoring | `example_emissions_monitoring.c` |
| 5 | General model validation | `test_compute_regression_stats()` |

## L7: Industrial Applications (Complete ✅)

| # | Application | Evidence |
|---|------------|----------|
| 1 | Refinery distillation | `example_distillation_quality.c` — real distillation data patterns |
| 2 | Polymer production | `example_polymer_melt_index.c` — MI inferential sensing |
| 3 | Emissions compliance | `example_emissions_monitoring.c` — NOx CEMS alternative |
| 4 | Process data keyword: ISO | Bias methodology references ISO 5725 |
| 5 | Industrial keyword: NASA | State estimation techniques from aerospace |

## L8: Advanced Topics (Complete ✅)

| # | Topic | Implementation |
|---|-------|---------------|
| 1 | Bayesian inference (conjugate prior) | `test_bayesian_bias_update()` — precision-weighted posterior |
| 2 | Monte Carlo propagation | `test_monte_carlo_prediction()` — input uncertainty → output |
| 3 | Adaptive filtering | `akf_step()` — Mehra innovation-based Q adaptation |
| 4 | Directional forgetting | `rls_directional_update()` — Kulhavy (1987) method |
| 5 | Just-in-time learning | `lwpls_model_t` (LW-PLS structure defined) |
| 6 | Gaussian Processes | `QMODEL_GAUSSIAN_PROCESS` enum + CKF for nonlinear |
| 7 | SPRT (sequential testing) | Lean: `sprtLogLikelihood` theorem |
| 8 | Multi-rate Kalman fusion | `mrf_kalman_alloc()`, delayed measurement update |

## L9: Research Frontiers (Partial ⚠️)

| # | Topic | Status |
|---|-------|--------|
| 1 | IT/OT fusion | Lean: `it_ot_fusion_mse_bound` (axiom) — documented |
| 2 | Digital Twin convergence | Lean: `digital_twin_convergence` theorem |
| 3 | 5G-enabled real-time quality | Documented in course-alignment |
| 4 | Autonomous L4 operations | Conceptual in README |

## Score Summary

| L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | Total |
|----|----|----|----|----|----|----|----|----|---- |-------|
| 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **17/18** |
