# Knowledge Graph — PCA-based Data-Driven Soft Sensor

## L1: Definitions ✅ Complete

| # | Concept | Implementation |
|---|---------|---------------|
| 1 | PCA Model (loadings, scores, eigenvalues) | `pca_model` struct, `pca_core.h` |
| 2 | Data Matrix (N x M) | `pca_matrix` struct, row-major storage |
| 3 | Variance Explained Ratio | `pca_compute_variance_explained()` |
| 4 | Cumulative Variance | `pca_model::cum_var` |
| 5 | Hotelling T² Statistic | `pca_t2_statistic()` |
| 6 | SPE (Q) Statistic | `pca_spe_statistic()` |
| 7 | Contribution Plots | `pca_t2_contributions()`, `pca_spe_contributions()` |
| 8 | Soft Sensor / Inferential Sensor | `pca_soft_sensor` struct |
| 9 | Primary vs Secondary Variables | PCR training/prediction API |
| 10 | Kernel Function (RBF, Polynomial) | `kpca_kernel_compute()` |

## L2: Core Concepts ✅ Complete

| # | Concept | Implementation |
|---|---------|---------------|
| 1 | Centering and Auto-scaling | `pca_center_scale()` |
| 2 | Covariance vs Correlation PCA | `pca_compute_covariance()`, `pca_compute_correlation()` |
| 3 | Dimensionality Reduction | PC selection via Kaiser/cumvar rules |
| 4 | Fault Detection & Diagnosis | `pca_monitor_observation()` |
| 5 | Model Plane vs Residual Space | T² (model) + SPE (residual) decomposition |
| 6 | Latent Variable Regression | PCR: `pca_pcr_train()` |
| 7 | Process Monitoring (MSPC) | Combined index, alarm generation |
| 8 | Kernel Trick | Implicit feature mapping in KPCA |

## L3: Engineering Structures ✅ Complete

| # | Concept | Implementation |
|---|---------|---------------|
| 1 | Row-major 2D Matrix | `pca_matrix` storage layout |
| 2 | Symmetric Matrix | Exploited in covariance, Jacobi |
| 3 | Tridiagonal Form | `pca_tridiagonalize()` |
| 4 | Householder Reflection | `householder_vector()` helper |
| 5 | Givens Rotation | `pca_givens_rotation()` |
| 6 | Circular Buffer | MWPCA `window_data` |

## L4: Engineering Laws ✅ Complete

| # | Theorem/Law | Implementation |
|---|------------|---------------|
| 1 | Spectral Theorem (Symmetric) | Jacobi: real eigenvalues, orthogonal eigenvectors |
| 2 | T² ~ F-distribution | `pca_t2_threshold_f()` |
| 3 | Jackson-Mudholkar SPE approx | `pca_spe_threshold()` |
| 4 | Kaiser Criterion (λ > avg) | `pca_kaiser_rule()` |
| 5 | Chi-squared T² approximation | `pca_t2_threshold_chi2()` |
| 6 | Mercer's Theorem (Kernel PSD) | Kernel matrix construction |

## L5: Algorithms ✅ Complete

| # | Algorithm | Implementation |
|---|-----------|---------------|
| 1 | Jacobi Eigenvalue Algorithm | `pca_jacobi_eigen()` |
| 2 | NIPALS (Sequential PC) | `pca_nipals()` |
| 3 | Power Iteration | `pca_power_iteration()` |
| 4 | Deflation | `pca_deflate_matrix()` |
| 5 | Householder QR | `pca_householder_qr()` |
| 6 | SVD (Golub-Reinsch) | `pca_svd_decomposition()` |
| 7 | PCR Training | `pca_pcr_train()` |
| 8 | LOOCV for PC selection | `pca_pcr_cross_validate()` |
| 9 | RPCA (Recursive PCA) | `rpca_update_full()` |
| 10 | MWPCA (Moving Window) | `mwpca_add_observation()` |
| 11 | KPCA Training | `kpca_train()` |
| 12 | Contribution Plots | `pca_t2_contributions()`, `pca_spe_contributions()` |
| 13 | VIP Scores | `pca_vip_scores()` |
| 14 | Welford's Variance Algorithm | `rpca_update_std()` |

## L6: Canonical Problems ✅ Complete

| # | Problem | Implementation |
|---|---------|---------------|
| 1 | Distillation Composition Estimation | `examples/example_distillation.c` |
| 2 | Chemical Reactor Concentration | `examples/example_reactor.c` |
| 3 | Tennessee Eastman Fault Detection | `examples/example_tep.c` |

## L7: Industrial Applications ⚠️ Partial

| # | Application | Status |
|---|------------|--------|
| 1 | Oil Refinery Inferential Measurement | Documented (PCR framework applicable) |
| 2 | Pharmaceutical Batch Monitoring | Documented (MWPCA for batch evolution) |
| 3 | Chemical Plant MSPC | Implemented (TEP example) |

## L8: Advanced Topics ⚠️ Partial

| # | Topic | Status |
|---|-------|--------|
| 1 | Kernel PCA (Nonlinear) | Implemented (`pca_kernel.c`) |
| 2 | Recursive/Adaptive PCA | Implemented (`pca_adaptive.c`) |
| 3 | Moving Window PCA | Implemented (`pca_adaptive.c`) |
| 4 | Bayesian PCA | Documented only |
| 5 | Probabilistic PCA | Documented only |

## L9: Research Frontiers ⚠️ Partial

| # | Topic | Status |
|---|-------|--------|
| 1 | Deep Learning Soft Sensors | Documented only |
| 2 | Edge Computing PCA | Documented only |
| 3 | Transfer Learning for Soft Sensors | Documented only |
