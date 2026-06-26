# mini-soft-sensor-maintenance-aging

## Soft Sensor Maintenance & Aging Management

Soft sensor lifecycle management: monitoring performance degradation,
detecting model aging, triggering adaptive retraining, and scheduling
predictive maintenance for inferential sensors.

---

## Module Status: COMPLETE

- **L1 Definitions**: Complete (11 typedef structs/enums)
- **L2 Core Concepts**: Complete (performance trajectory, health index, drift)
- **L3 Engineering Structures**: Complete (Welford, sliding window, matrix ops, PLS)
- **L4 Fundamental Laws**: Complete (Jackson-Mudholkar, Hotelling T2, t/F dist, Durbin-Watson)
- **L5 Algorithms**: Complete (CUSUM, SPRT, Page-Hinkley, Mann-Kendall, RPLS, MW-PLS, JIT-PLS)
- **L6 Canonical Problems**: Complete (distillation drift, CDU dashboard, bioreactor retraining)
- **L7 Applications**: Partial+ (refinery CDU, pharmaceutical bioreactor, chemical distillation)
- **L8 Advanced Topics**: Partial+ (Bayesian change point, JIT learning, ensemble pruning)
- **L9 Frontiers**: Partial (predictive maintenance policy framework)

---

## Knowledge Coverage Summary

| Level | Topic | Status | Artifacts |
|-------|-------|--------|-----------|
| L1 | RMSE, MAE, MAPE, R2, HealthIndex, LifecycleStage, QStatistic, T2Statistic | Complete | soft_sensor_metrics.h |
| L1 | PredictionError, RunningStatistics, SlidingWindow, DriftType, CI | Complete | soft_sensor_metrics.h |
| L2 | Performance trajectory, model health, drift classification | Complete | model_health_monitor.h |
| L2 | Ensemble concept, weighting strategies, diversity measures | Complete | ensemble_maintenance.h |
| L3 | Welford online, West removal, Chan parallel merge | Complete | soft_sensor_metrics.c |
| L3 | Circular buffer sliding window, O(1) stats | Complete | soft_sensor_metrics.c |
| L3 | Matrix dense ops, PLS model representation | Complete | adaptive_model_updater.c |
| L4 | Jackson-Mudholkar SPE limits (Technometrics 1979) | Complete | soft_sensor_metrics.c |
| L4 | Hotelling T2 with F-distribution limits | Complete | soft_sensor_metrics.c |
| L4 | Student t critical value (Hill 1970, Algorithm 396) | Complete | soft_sensor_metrics.c |
| L4 | F-distribution (Wilson-Hilferty), Durbin-Watson, Ljung-Box | Complete | soft_sensor_metrics.c, model_health_monitor.c |
| L4 | Welch t-test, paired t-test, bootstrap CI | Complete | statistical_validation.c |
| L5 | CUSUM (Page 1954), SPRT (Wald 1945), Page-Hinkley (1971) | Complete | aging_detection.c |
| L5 | Mann-Kendall trend + Sen slope (1945/1975) | Complete | aging_detection.c |
| L5 | RPLS (Qin 1998), MW-PLS, JIT locally weighted | Complete | adaptive_model_updater.c |
| L5 | Ensemble: uniform, inverse-RMSE, recent, Bayesian, diversity-aware | Complete | ensemble_maintenance.c |
| L5 | Permutation test, Diebold-Mariano, bootstrap CI | Complete | statistical_validation.c |
| L6 | Distillation column sensor aging (2000h simulation) | Complete | example_drift_monitor.c |
| L6 | Refinery CDU health dashboard (18-month) | Complete | example_health_dashboard.c |
| L6 | Bioreactor adaptive retraining (50 batches) | Complete | example_adaptive_retraining.c |
| L7 | Refinery CDU diesel boiling point | Partial+ | example_health_dashboard.c |
| L7 | Pharmaceutical bioreactor biomass | Partial+ | example_adaptive_retraining.c |
| L7 | Chemical distillation composition | Partial+ | example_drift_monitor.c |
| L8 | Bayesian change point (Barry & Hartigan 1993) | Partial+ | model_health_monitor.c |
| L8 | Greedy ensemble pruning (Martinez-Munoz 2006) | Partial+ | ensemble_maintenance.c |
| L8 | Bayesian predictive check (Gelman et al. 1996) | Partial+ | statistical_validation.c |
| L9 | Predictive maintenance policy framework | Partial | maintenance_formal.lean |

---

## Core Definitions (L1)

| Definition | Type | Description |
|------------|------|-------------|
| PredictionError | struct | Individual residual, relative error, squared error |
| RegressionMetrics | struct | RMSE, MAE, MAPE, R2, MSE, max_error, bias |
| RunningStatistics | struct | Welford online mean/variance accumulator |
| SlidingWindow | struct | Fixed-size circular buffer with O(1) stats |
| SensorLifecycleStage | enum | Commissioning->Normal->Warning->Aging->Maintenance->Failure->Retired |
| HealthIndex | struct | Composite health: accuracy, drift, noise, reliability |
| QStatistic | struct | SPE with Jackson-Mudholkar 95%/99% limits |
| T2Statistic | struct | Hotelling T2 with F-distribution limits |
| DriftType | enum | None, Sudden, Incremental, Gradual, Recurring, Blip |
| ConfidenceInterval | struct | Point estimate with t-distribution bounds |
| PairedTTest | struct | Student paired t-test with Cohen d |

---

## Core Theorems (L4)

| Theorem | Formula | Reference |
|---------|---------|-----------|
| Welford Online Variance | mean_n = mean_{n-1} + (x_n-mean_{n-1})/n; M2_n = M2_{n-1} + delta*delta2 | Welford (1962) |
| Chan Parallel Merge | M2_AB = M2_A + M2_B + delta^2*n_A*n_B/(n_A+n_B) | Chan et al. (1979) |
| Jackson-Mudholkar SPE | Q_alpha = theta1 * [c_alpha*sqrt(2*theta2*h0^2)/theta1 + ...]^(1/h0) | Jackson & Mudholkar (1979) |
| Hotelling T2 Limit | T2_alpha = A*(N-1)*(N+1)/[N*(N-A)] * F_{A,N-A,alpha} | Hotelling (1931) |
| EWMA Control Limits | UCL/LCL = mu0 +/- L*sigma*sqrt(lambda/(2-lambda)) | Hunter (1986) |
| CUSUM Decision | S_t = max(0, S_{t-1} + x_t - mu0 - k); Alarm if S_t > h | Page (1954) |
| SPRT LLR | LLR = (mu1-mu0)/sigma^2 * sum(x_i - (mu0+mu1)/2) | Wald (1945) |
| Mann-Kendall Z | Z = (S - sign(S))/sqrt(Var[S]) | Mann (1945) |
| Durbin-Watson | DW = sum(e_t - e_{t-1})^2 / sum(e_t^2) | Durbin & Watson (1950) |
| Ljung-Box Q | Q = n*(n+2)*sum(r_k^2/(n-k)) | Ljung & Box (1978) |

---

## Core Algorithms (L5)

| Algorithm | Complexity | Reference |
|-----------|-----------|-----------|
| RPLS Update | O(n_vars * n_comp) | Qin (1998) Comp. Chem. Eng. |
| MW-PLS | O(window * n_vars) | Dayal & MacGregor (1997) |
| JIT-LW-PLS | O(db_size * n_vars) | Bontempi et al. (1999) |
| CUSUM | O(1) per sample | Page (1954) Biometrika |
| SPRT | O(1) per sample | Wald (1945) Ann. Math. Stat. |
| Mann-Kendall | O(n^2) | Mann (1945) Econometrica |
| Page-Hinkley | O(n) | Hinkley (1971) Biometrika |
| Bootstrap CI | O(B * n) | Efron & Tibshirani (1993) |
| Permutation Test | O(P * n) | Good (2013) Springer |
| Diebold-Mariano | O(n * max_lag) | Diebold & Mariano (1995) |
| Ensemble Pruning | O(K^2 * n) | Martinez-Munoz & Suarez (2006) |

---

## Classic Problems (L6)

1. **Distillation Column Composition Sensor Aging** (example_drift_monitor.c)
   - Benzene overhead composition over 2000 hours, fouling-induced drift, RUL estimation

2. **Refinery CDU Health Dashboard** (example_health_dashboard.c)
   - Diesel 95% boiling point over 18 months, multi-metric health tracking, stage transitions

3. **Bioreactor Adaptive Retraining** (example_adaptive_retraining.c)
   - Biomass concentration over 50 batches, RPLS+JIT ensemble, performance-triggered retraining

---

## 9-School Curriculum Mapping

| School | Course | Relevance |
|--------|--------|-----------|
| MIT | 6.302 Feedback Systems | Estimation theory, sensor fusion |
| Stanford | ENGR205 Process Control | Process monitoring, SPC |
| Berkeley | ME233 Advanced Control | State estimation primitives |
| CMU | 24-677 Adv Ctrl Systems | System health, degradation |
| Georgia Tech | AE 6530 Optimal Estimation | Sequential testing, change detection |
| Purdue | ME 575 Industrial Control | Soft sensors, inferential measurement |
| RWTH Aachen | Industrial Control Systems | Condition monitoring, maintenance |
| Tsinghua | Process Control Engineering | APC, lifecycle management |
| ISA/IEC | ISA-88/95, IEC 61508/61511 | Functional safety standards |

---

## Compilation

```
make all      # Build tests and examples
make test     # Run all tests
make examples # Run all examples
make check    # Safety review (filler detection, line counts)
make clean    # Remove build artifacts
```

## Module Status: COMPLETE

- L1-L6: Complete
- L7: Partial+ (3 industrial applications)
- L8: Partial+ (3 advanced topics implemented)
- L9: Partial (documented, formal policy framework)

include/ + src/ line count: >4800 lines (exceeds 3000 minimum)
