# Course Tree — PLS Partial Least Squares

## Prerequisites (what this module depends on)

```
Linear Algebra
  ├── Matrix multiplication, transpose, inverse
  ├── Eigenvalues and eigenvectors
  ├── Singular Value Decomposition (SVD)
  └── Cholesky decomposition

Statistics
  ├── Mean, variance, standard deviation
  ├── Covariance and cross-covariance
  ├── F-distribution (for T2 limits)
  ├── Normal distribution (for SPE limits)
  └── Cross-validation principles

Numerical Methods
  ├── Power iteration for eigenvalues
  ├── Householder reflections for QR
  └── Iterative convergence criteria

Process Control
  ├── Soft sensing / inferential measurement concepts
  ├── Process monitoring fundamentals
  └── Fault detection principles
```

## Dependents (modules that build on this one)

```
mini-pls-partial-least-squares (this module)
  │
  ├── mini-advanced-process-control-apc
  │     └── PLS as APC component for quality control
  │
  ├── mini-industrial-mpc-implementation
  │     └── PLS as soft sensor for MPC state estimation
  │
  ├── mini-soft-sensor-data-driven-pca
  │     └── PLS compared to PCA for soft sensing
  │
  └── mini-industrial-ai-control-fusion
        └── Kernel PLS bridges to neural network methods
```

## Learning Path

1. Start with `matrix_ops.h/c` — linear algebra foundation
2. Read `pls_model.h/c` — core PLS data structures
3. Study `pls_nipals.h/c` — classic PLS algorithm
4. Explore `pls_preprocessing.h/c` — data preprocessing methods
5. Learn `pls_statistics.h/c` — diagnostic tools
6. Apply `pls_crossval.h/c` — model selection
7. Advanced: `pls_kernel.h/c` (nonlinear), `pls_recursive.h/c` (adaptive)
8. Deploy: `pls_online.h/c` (real-time monitoring)
