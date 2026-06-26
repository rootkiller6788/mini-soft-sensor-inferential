# mini-soft-sensor-data-reconciliation

**Data Reconciliation: WLS-based adjustment of process measurements to satisfy mass/energy balance constraints.**

## Module Status: COMPLETE (symbol checkmark)

- **L1-L6**: Complete
- **L7**: Partial+ (3 industrial applications: steam metering, flow network, tank dynamics)
- **L8**: Partial+ (robust M-estimators, PCA detection, test power, Bayesian DR documented)
- **L9**: Partial (digital twin, ML-augmented DR documented)

## Quick Start

```bash
make          # build library
make test     # run tests (7 pass)
make examples # run 3 end-to-end examples
```

## Line Counts

| Category | Lines |
|----------|-------|
| include/ (6 headers) | 1,708 |
| src/ (6 C files) | 4,071 |
| src/ (1 Lean file) | 143 |
| tests/ (1 test file) | 141 |
| examples/ (3 files) | 415 |
| **Total** | **~6,478** |
| **include/ + src/ (.h and .c)** | **5,779** |

## Core Definitions (L1)

| Definition | C Type |
|-----------|--------|
| Measurement | `dr_measurement_t` |
| Constraint | `dr_constraint_type_t` |
| Covariance matrix | `double[n*n]` row-major |
| Reconciled value | `dr_result_t.x_reconciled` |
| Lagrange multiplier | `dr_result_t.lagrange_mult` |
| Variable class | `dr_var_class_t` |
| Gross error test | `dr_ge_test_result_t` |
| Kalman gain | `dr_kf_state_t.K` |
| Innovation | `dr_kf_state_t.innovation` |
| Redundancy degree | `int` |

## Core Theorems (L4)

1. **Kuehn-Davidson Uniqueness** (1961): Under full rank A and Sigma > 0, the WLS solution x_hat is unique.
2. **Precision Improvement**: Var(x_hat_i) <= Var(y_i) for all i. Reconciliation never increases uncertainty.
3. **Constraint Satisfaction**: A * x_hat = b, the reconciled values exactly satisfy physical laws.
4. **Global Test Distribution** (Mah & Tamhane, 1982): z_GT ~ chi^2(m) under H0.
5. **Kalman Optimality** (Kalman, 1960): The filter provides the MMSE estimate for linear Gaussian systems.
6. **Huber Breakdown Point**: The Huber M-estimator has positive breakdown point, unlike WLS.
7. **Redundancy Monotonicity**: Adding measurements never decreases system redundancy.
8. **Observability Sufficiency**: A variable is observable if it is the only unmeasured variable in a constraint.

## Core Algorithms (L5)

| Algorithm | Implementation | Complexity |
|-----------|---------------|------------|
| WLS (Lagrange) | `dr_solve(..., DR_SOLVER_LAGRANGE)` | O(m^3 + n*m^2) |
| WLS (QR) | `dr_solve(..., DR_SOLVER_QR_ORTHOG)` | O(m^2*n + m^3) |
| Global Test | `dr_ge_global_test` | O(m^3 + n*m^2) |
| Nodal Test | `dr_ge_nodal_test` | O(m^3 + n*m^2) |
| Measurement Test | `dr_ge_measurement_test` | O(m^3 + n*m^2 + n^2*m) |
| Serial Elimination | `dr_ge_serial_elimination` | O(k * (m^3 + n*m^2)) |
| Robust Huber | `dr_ge_robust_huber` | O(k * (m^3 + n*m^2)) |
| Kalman Filter | `dr_kf_predict` + `dr_kf_update` | O(nx^3 + ny^3) |
| RTS Smoother | `dr_kf_smooth_rts` | O(N * nx^3) |
| PCA Detection | `dr_ge_pca_detection` | O(m^3 + n*m^2) |
| Outlier Detection (IQR) | `dr_meas_detect_outliers_iqr` | O(n log n) |
| Cholesky | `dr_mat_cholesky` | O(n^3/3) |
| QR (Householder) | `dr_mat_qr` | O(2*m*n^2 - 2*n^3/3) |

## Classic Problems (L6)

1. **Flow Network Reconciliation** (Example 1): 5-stream, 2-node pipeline mass balance
2. **Steam Metering Gross Error Detection** (Example 2): Boiler-turbine-bypass system, 3 scenarios
3. **Dynamic Tank Level Kalman Filtering** (Example 3): 20-step Kalman filter with RTS smoothing

## Nine-School Curriculum Mapping

| School | Key Courses | Mapping |
|--------|------------|---------|
| **MIT** | 6.302, 2.171 | Kalman filter, observability analysis |
| **Stanford** | ENGR205, EE392 | Process monitoring, gross error detection |
| **Berkeley** | ME233, EE C128 | Optimal estimation, sensor fusion |
| **CMU** | 24-677, 18-771 | Linear systems, estimation theory |
| **Georgia Tech** | ECE 6550, AE 6530 | Robust estimation |
| **Purdue** | ECE 602, ME 575 | Industrial process control |
| **RWTH Aachen** | Industrial Control | VDI 2048 standard |
| **Tsinghua** | 过程控制工程 | Process control, digital twin |
| **ISA/IEC** | ISA-88, IEC 61508 | Measurement standards |

## References

1. Kuehn & Davidson (1961). Chem. Eng. Progress, 57(6), 44-47.
2. Crowe et al. (1983). AIChE Journal, 29(6), 881-888.
3. VDI 2048 Part 1 (2008). Control and quality improvement of process data.
4. Narasimhan & Jordache (2000). Data Reconciliation & Gross Error Detection. Gulf Publishing.
5. Romagnoli & Sanchez (2000). Data Processing and Reconciliation. Academic Press.
6. Mah & Tamhane (1982). AIChE Journal, 28(5), 828-835.
7. Kalman (1960). J. Basic Eng., 82(1), 35-45.
8. Huber & Ronchetti (2009). Robust Statistics, 2nd ed. Wiley.
9. Golub & Van Loan (2013). Matrix Computations, 4th ed. Johns Hopkins.
10. Tong & Crowe (1995). AIChE Journal, 41(7), 1712-1722.
