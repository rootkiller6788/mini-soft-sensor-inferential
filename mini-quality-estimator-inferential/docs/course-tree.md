# Course Dependency Tree — Inferential Quality Estimation

## Prerequisites

```
mini-quality-estimator-inferential
├── mini-industrial-measurement-actuator (L1: process variables, sensors)
├── mini-pid-control-engineering (L2: feedback concepts)
├── mini-feedforward-cascade-ratio (L2: inferential control architecture)
├── mini-advanced-process-control-apc (L5: state estimation, Kalman)
└── mini-soft-sensor-data-driven-pca (L5: PLS, PCA for data-driven models)
```

## Knowledge Dependencies by Layer

### L1 → L2 → L3 (Linear Dependencies)
```
L1: Type definitions
 ├── quality_model_type_t, bias_strategy_t, qest_mode_t
 ├── quality_estimate_t, lab_sample_t, process_variable_t
 └── qest_config_t, qest_performance_t
     ↓
L2: Core concepts
 ├── First-principles models (fpm_evaluate)
 ├── Data-driven models (linear_model_evaluate, pls_model_predict)
 ├── Bias correction (additive, multiplicative, Kalman)
 └── Model validation (residual analysis)
     ↓
L3: Engineering structures
 ├── State-space (ss_model_t, ARX)
 ├── Multi-rate fusion (mrf_kalman_t)
 └── Bias context (dispatcher pattern)
```

### L4 → L5 (Theory → Algorithm)
```
L4: Laws and theorems
 ├── Bias-variance decomposition
 ├── Durbin-Watson autocorrelation test
 ├── Confidence intervals (normal CDF)
 └── ARX stability conditions
     ↓
L5: Algorithms
 ├── Kalman filter (KF, EKF, AKF, CKF)
 ├── Recursive Least Squares (RLS, VFF, directional)
 ├── CUSUM drift detection
 └── Moving Horizon Estimation (MHE)
```

### L6 → L7 → L8 → L9 (Application Layers)
```
L6: Canonical problems
 ├── Distillation composition
 ├── Reactor conversion
 ├── Polymer melt index
 └── NOx emissions
     ↓
L7: Industrial applications
 ├── Refinery/petrochemical
 ├── Polymer manufacturing
 └── Environmental compliance
     ↓
L8: Advanced topics
 ├── Bayesian inference
 ├── Monte Carlo methods
 ├── Adaptive filtering
 └── Multi-rate fusion
     ↓
L9: Research frontiers
 ├── IT/OT fusion
 ├── Digital twin
 └── Autonomous operation
```
