# Knowledge Graph — PLS (Partial Least Squares)

## L1: Definitions

| ID | Concept | C Implementation | Status |
|----|---------|-----------------|--------|
| L1.1 | PLSModel struct | `include/pls_model.h` struct `PLSModel` | Complete |
| L1.2 | Matrix/Vector types | `include/matrix_ops.h` struct `Matrix`, `Vector` | Complete |
| L1.3 | PLS decomposition (X=TP'+E, Y=UQ'+F) | `pls_model_alloc()` in `src/pls_model.c` | Complete |
| L1.4 | Latent variables (scores T, loadings P, weights W) | `PLSModel.W/P/T/U/Q/B_inner` fields | Complete |
| L1.5 | Regression coefficients Beta | `pls_model_compute_beta()` | Complete |
| L1.6 | R2X, R2Y (goodness of fit) | `compute_R2X()`, `compute_R2Y()` in `pls_statistics.c` | Complete |
| L1.7 | Q2 (cross-validated predictive ability) | `compute_Q2()` in `pls_statistics.c` | Complete |
| L1.8 | Hotelling T2 | `compute_T2_single()`, `compute_T2_limit()` | Complete |
| L1.9 | SPE (Q-statistic) | `compute_SPE_single()`, `compute_SPE_limit()` | Complete |
| L1.10 | VIP (Variable Importance in Projection) | `compute_VIP()` in `pls_statistics.c` | Complete |
| L1.11 | DModX (Distance to Model) | `compute_DModX_single()`, `compute_DModX_critical()` | Complete |
| L1.12 | PRESS | `compute_PRESS()` in `pls_statistics.c` | Complete |
| L1.13 | Preprocessing methods enum | `PreprocessingMethod` in `pls_preprocessing.h` | Complete |
| L1.14 | KernelPLSModel struct | `include/pls_kernel.h` struct `KernelPLSModel` | Complete |
| L1.15 | OnlinePrediction struct | `include/pls_online.h` struct `OnlinePrediction` | Complete |
| L1.16 | RecursivePLS struct | `include/pls_recursive.h` struct `RecursivePLS` | Complete |

## L2: Core Concepts

| ID | Concept | C Implementation | Status |
|----|---------|-----------------|--------|
| L2.1 | Dimensionality reduction via latent projection | `nipals_fit()` extracts T from X | Complete |
| L2.2 | Inner relation (U = B*T) | `pls_model_inner_predict()` | Complete |
| L2.3 | Outer relations (X and Y decompositions) | NIPALS decomposition loop | Complete |
| L2.4 | Deflation process | `nipals_deflate()` | Complete |
| L2.5 | Kernel trick (Kernel PLS) | `kernel_pls_fit()` | Complete |
| L2.6 | Online prediction pipeline | `online_predict_sample()` | Complete |
| L2.7 | Cross-validation for model selection | `crossval_kfold()` | Complete |
| L2.8 | Adaptive/recursive model updating | `recursive_pls_update()` | Complete |

## L3: Engineering Structures

| ID | Concept | C Implementation | Status |
|----|---------|-----------------|--------|
| L3.1 | Row-major matrix storage | `Matrix` struct, `MATRIX_AT` macro | Complete |
| L3.2 | Matrix lifecycle (alloc/copy/free) | `matrix_alloc/copy/free()` | Complete |
| L3.3 | GEMM (BLAS Level 3) | `matrix_gemm()` | Complete |
| L3.4 | Cholesky decomposition | `matrix_cholesky_decomp()` | Complete |
| L3.5 | QR decomposition (Householder) | `matrix_qr_decomp()` | Complete |
| L3.6 | Pseudo-inverse | `matrix_pinv()` | Complete |
| L3.7 | Preprocessing pipeline | `preprocess_fit/transform/inverse()` | Complete |
| L3.8 | Online monitoring state | `OnlineMonitor` struct | Complete |
| L3.9 | Component statistics | `PLSComponentStats` struct | Complete |

## L4: Engineering Laws

| ID | Concept | C Implementation | Status |
|----|---------|-----------------|--------|
| L4.1 | Cholesky-Banachiewicz theorem | `matrix_cholesky_decomp()` implementation | Complete |
| L4.2 | NIPALS convergence to dominant eigenvector | `nipals_extract_one_component()` with convergence check | Complete |
| L4.3 | Orthogonality of PLS scores | Implicit in NIPALS deflation | Complete |
| L4.4 | SIMPLS equivalence to NIPALS (univariate Y) | `simpls_fit()` | Complete |
| L4.5 | Jackson-Mudholkar SPE limit | `compute_SPE_limit()` | Complete |
| L4.6 | Hotelling T2 F-distribution limit | `compute_T2_limit()` with Wilson-Hilferty approx | Complete |
| L4.7 | Welford's online variance algorithm | `welford_update/finalize()` | Complete |
| L4.8 | Eckart-Young-Mirsky SVD theorem | `matrix_svd_values()` documentation | Complete |

## L5: Algorithms/Methods

| ID | Concept | C Implementation | Status |
|----|---------|-----------------|--------|
| L5.1 | NIPALS algorithm | `nipals_fit()`, `nipals_extract_one_component()` | Complete |
| L5.2 | SIMPLS algorithm | `simpls_fit()`, `simpls_compute_weights()` | Complete |
| L5.3 | Kernel PLS (RBF, polynomial, sigmoid) | `kernel_pls_fit()` | Complete |
| L5.4 | K-fold cross-validation | `crossval_kfold()` | Complete |
| L5.5 | Leave-One-Out CV | `crossval_leave_one_out()` | Complete |
| L5.6 | Monte Carlo CV | `crossval_monte_carlo()` | Complete |
| L5.7 | VIP computation | `compute_VIP()` | Complete |
| L5.8 | Power iteration (eigenvalue) | `matrix_power_iteration()` | Complete |
| L5.9 | SVD via A^T*A + deflation | `matrix_svd_values()` | Complete |
| L5.10 | Model selection (min PRESS, 1-SE, Q2) | `crossval_select_optimal()` | Complete |
| L5.11 | Recursive PLS with forgetting factor | `recursive_pls_update()` | Complete |
| L5.12 | EWMA update | `ewma_update()` | Complete |
| L5.13 | Range scaling | `range_scale_matrix()` | Complete |
| L5.14 | Robust scaling (median/MAD) | `robust_scale_matrix()`, `compute_mad()` | Complete |
| L5.15 | Pareto scaling | `pareto_scale_matrix()` | Complete |

## L6: Canonical Problems

| ID | Concept | Implementation | Status |
|----|---------|---------------|--------|
| L6.1 | Soft sensor for quality prediction | `examples/example_quality_prediction.c` | Complete |
| L6.2 | Multivariate process monitoring | `examples/example_process_monitoring.c` | Complete |
| L6.3 | Model selection via cross-validation | `examples/example_cross_validation.c` | Complete |

## L7: Industrial Applications

| ID | Concept | Implementation | Status |
|----|---------|---------------|--------|
| L7.1 | Online soft sensor deployment | `pls_online.c`, `online_predict_sample()` | Complete |
| L7.2 | MSPC with T2/SPE monitoring | `online_monitor_init()`, alarm generation | Complete |
| L7.3 | Fault diagnosis via variable contributions | `online_T2_contributions()`, `online_SPE_contributions()` | Complete |

## L8: Advanced Topics

| ID | Concept | Implementation | Status |
|----|---------|---------------|--------|
| L8.1 | Kernel PLS (nonlinear regression) | `pls_kernel.c` with RBF/polynomial/sigmoid kernels | Complete |
| L8.2 | Recursive PLS with forgetting factor | `pls_recursive.c` | Complete |
| L8.3 | Moving window PLS | `pls_recursive.c` ring buffer | Complete |
| L8.4 | Robust preprocessing (median/MAD) | `robust_scale_matrix()` | Complete |

## L9: Research Frontiers

| ID | Concept | Status |
|----|---------|--------|
| L9.1 | Online adaptive PLS for Industry 4.0 | Partial (documented, recursive PLS implementation exists) |
| L9.2 | Deep PLS (neural network + PLS hybrid) | Partial (documented only) |
| L9.3 | Multi-block PLS for multi-unit processes | Partial (documented only) |
