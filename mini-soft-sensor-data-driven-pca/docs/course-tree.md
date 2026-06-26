# Course Tree — PCA Data-Driven Soft Sensor

## Prerequisite Knowledge Dependencies

```
Linear Algebra (Matrix ops, eigendecomposition)
├── Statistics (mean, variance, covariance, normal distribution)
│   └── PCA Theory (Jolliffe Ch.3)
│       ├── PCA for Process Monitoring (MacGregor & Kourti)
│       │   ├── T² Statistic (Hotelling)
│       │   ├── SPE / Q Statistic (Jackson & Mudholkar)
│       │   └── Contribution Plots
│       ├── PCA Regression / PCR (Jolliffe Ch.8)
│       │   ├── Soft Sensor Design
│       │   ├── Cross-Validation
│       │   └── VIP Scores
│       ├── Adaptive PCA
│       │   ├── Recursive PCA (Li et al. 2000)
│       │   └── Moving Window PCA (Jeng 2010)
│       └── Kernel PCA (Scholkopf et al. 1998)
│           └── Nonlinear Process Monitoring (Lee et al. 2004)
└── Numerical Methods (Golub & Van Loan)
    ├── Jacobi Eigenvalue Algorithm
    ├── Householder Reflections
    ├── QR Decomposition
    ├── SVD (Golub-Reinsch)
    └── Tridiagonalization
```

## Dependency Graph (This Module)

```
pca_core.h/c (L1-L3: data structures, basic ops)
├── pca_decomposition.h/c (L4-L5: eigen, SVD, NIPALS)
│   ├── pca_monitoring.h/c (L1-L5: T², SPE, fault detection)
│   │   └── examples/example_tep.c
│   ├── pca_inferential.h/c (L1-L5: PCR, soft sensor)
│   │   ├── examples/example_distillation.c
│   │   └── examples/example_reactor.c
│   ├── pca_adaptive.h/c (L5-L8: RPCA, MWPCA)
│   └── pca_kernel.h/c (L5-L8: nonlinear KPCA)
└── tests/test_pca_*.c
```

## Knowledge Flow

1. Start with `pca_core.c` — understand matrix operations and PCA model structure
2. Move to `pca_decomposition.c` — learn how PCA actually works (eigen decomposition)
3. Study `pca_monitoring.c` — apply PCA to detect process faults
4. Study `pca_inferential.c` — use PCA for soft sensor regression
5. Explore `pca_adaptive.c` — handle time-varying industrial processes
6. Dive into `pca_kernel.c` — nonlinear extension for complex processes
