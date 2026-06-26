# Course Tree — Data Reconciliation

## Prerequisites
```
Linear Algebra
  ├── Matrix operations (multiply, transpose, inverse)
  ├── Vector spaces (null space, range space, rank)
  ├── Matrix factorizations (QR, Cholesky, SVD)
  └── Positive definite matrices (covariance matrices)

Probability & Statistics
  ├── Random variables, expectation, variance
  ├── Normal (Gaussian) distribution
  ├── Chi-squared distribution
  ├── Hypothesis testing (significance level, p-value)
  ├── Maximum likelihood estimation
  └── Bayesian inference

Optimization
  ├── Constrained optimization (Lagrange multipliers)
  ├── KKT conditions
  ├── Least squares (ordinary and weighted)
  └── Robust optimization (M-estimators)

Control Theory
  ├── State-space models (F, B, H matrices)
  ├── Observability (Gramian, rank condition)
  ├── Kalman filter (prediction and update)
  └── Optimal estimation (MMSE, BLUE)

Numerical Methods
  ├── Floating-point arithmetic (condition numbers)
  ├── Direct solvers (Cholesky, QR, LU)
  ├── Iterative methods (power iteration, IRLS)
  └── Stability analysis (backward error)
```

## Module Dependency Tree
```
This module: Data Reconciliation
  ├── dr_core.h/c — WLS solvers, problem/result lifecycle
  │   └── dr_matrix.h/c — Linear algebra (QR, Cholesky, solve)
  ├── dr_measurement.h/c — Measurement statistics, GUM, outlier detection
  │   └── dr_core.h — Measurement types
  ├── dr_redundancy.h/c — Variable classification, observability
  │   └── dr_core.h + dr_matrix.h — Constraint analysis via QR
  ├── dr_gross_error.h/c — Statistical tests, robust estimation
  │   └── dr_core.h + dr_measurement.h + dr_matrix.h
  └── dr_dynamic.h/c — Kalman filter, MHE
      └── dr_core.h + dr_matrix.h + dr_measurement.h
```

## Downstream Modules (in this project)
```
mini-soft-sensor-inferential (parent module)
  └── mini-soft-sensor-data-reconciliation (this module)
      ├── Feeds into: process monitoring dashboards
      ├── Feeds into: APC (Advanced Process Control) data preprocessing
      └── Feeds into: real-time optimization (RTO) reconciled data
```

## Research Frontiers (L9)
- **Digital Twin**: continuous DR feeds virtual plant models
- **Real-time large-scale DR**: 10^5+ variable systems with sparse solvers
- **Nonlinear DR**: first-principles (thermodynamic) model constraints
- **ML-augmented DR**: neural network surrogate models replacing explicit constraints
- **Federated DR**: multi-plant reconciliation across organizational boundaries
