# Course Tree — mini-kalman-filter-soft-sensor

## Prerequisite Dependency Tree

```
┌─────────────────────────────────────────────────────────────┐
│                mini-kalman-filter-soft-sensor                │
│           Kalman Filter-Based Soft Sensor Module             │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Linear KF    │   │  Nonlinear KF    │   │  Adaptive KF     │
│  (kalman_core)│   │  (EKF + UKF)     │   │  (AKF + IMM)     │
└───────────────┘   └──────────────────┘   └──────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────────────────────────────────────────────────────┐
│              kalman_matrix_ops (Linear Algebra)               │
│  mat_mul, Cholesky, QR, inverse, eigen, condition, sqrt      │
└───────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Probability  │   │  Linear Algebra  │   │  Optimization    │
│  (Gaussian,   │   │  (vector, matrix │   │  (MMSE, least    │
│   chi-square) │   │   decomposition) │   │   squares)       │
└───────────────┘   └──────────────────┘   └──────────────────┘
```

## Intra-Module Dependencies

```
kalman_core.h/c
├── depends on: kalman_matrix_ops.h
└── provides: KalmanFilterState, KalmanModel, kf_*() API

kalman_extended.h/c
├── depends on: kalman_core.h, kalman_matrix_ops.h
└── provides: EKFModel, EKFState, ekf_*() API

kalman_unscented.h/c
├── depends on: kalman_core.h, kalman_matrix_ops.h
└── provides: UKFParams, UKFState, ukf_*() API

kalman_adaptive.h/c
├── depends on: kalman_core.h, kalman_matrix_ops.h
└── provides: AKFState, AKFMultiModel, akf_*(), imm_*() API

kalman_smoother.h/c
├── depends on: kalman_core.h, kalman_matrix_ops.h
└── provides: KFHistory, RTSSmoother, FixedLagSmoother, smoothing API

kalman_applications.h/c
├── depends on: kalman_core.h, kalman_extended.h, kalman_unscented.h, kalman_adaptive.h
└── provides: DCMotorEstimator, CSTREstimator, INSGPSNavigator, IndustrialQualityEstimator

kalman_matrix_ops.h/c
├── depends on: <math.h> (sqrt, fabs, exp, log)
└── provides: All linear algebra primitives (no external BLAS/LAPACK)
```

## Cross-Module Dependencies

```
mini-kalman-filter-soft-sensor
├── depends on: (none — self-contained)
├── used by:
│   ├── mini-soft-sensor-data-driven-pca (Kalman for PCA residual monitoring)
│   ├── mini-quality-estimator-inferential (Kalman for quality tracking)
│   ├── mini-virtual-flow-meter (Kalman for flow estimation)
│   └── mini-industrial-mpc-implementation (Kalman for state estimation in MPC)
└── related to:
    ├── mini-pls-partial-least-squares (PLS preprocessing for quality model)
    └── mini-neural-network-soft-sensor (alternative to KF for nonlinear soft sensing)
```

## Learning Path

### Beginner Track (L1-L3)
1. Start with `kalman_matrix_ops.h/c` — learn linear algebra primitives
2. Read `kalman_core.h` — understand struct definitions and API
3. Run `example_dc_motor.c` — see KF in action on a simple 2-state problem

### Intermediate Track (L4-L6)
4. Study `kalman_core.c: kf_predict()` and `kf_update()` — understand the 5 equations
5. Read `kalman_extended.c` — learn linearization for nonlinear systems
6. Run `example_cstr_reactor.c` — EKF for chemical process
7. Run `example_ins_gps.c` — multi-rate sensor fusion

### Advanced Track (L7-L9)
8. Study `kalman_unscented.c` — derivative-free nonlinear filtering
9. Read `kalman_adaptive.c` — online noise estimation and IMM
10. Study `kalman_smoother.c` — backward estimation for offline analysis
11. Run `example_quality_estimator.c` — industrial deployment with lab calibration
12. Read `kalman_model.lean` — formalized theory and research frontiers
