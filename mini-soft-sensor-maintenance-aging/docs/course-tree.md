# Course Tree - Prerequisite Dependencies

## Direct Prerequisites
- Statistics: mean, variance, distributions (t, F, chi2)
- Linear algebra: matrices, eigenvalues, PLS decomposition
- Control theory: feedback, estimation
- Optimization: least squares, gradient methods

## Internal Dependencies
soft_sensor_metrics.h -> model_health_monitor.h -> aging_detection.h
soft_sensor_metrics.h -> adaptive_model_updater.h
soft_sensor_metrics.h -> ensemble_maintenance.h
statistical_validation.c uses metrics + health monitor

## External Dependencies
- mini-kalman-filter-soft-sensor (state estimation)
- mini-pls-partial-least-squares (PLS theory)
- mini-soft-sensor-data-driven-pca (PCA monitoring)

## Learning Path
1. Statistics -> Regression metrics
2. Online algorithms -> Running statistics, sliding window
3. Control charts -> CUSUM, EWMA, SPRT
4. Multivariate monitoring -> SPE, T2, contributions
5. Adaptive modeling -> RPLS, MW-PLS, JIT
6. Ensemble methods -> Weighting, diversity, pruning
7. Maintenance -> RUL, scheduling, Bayesian CP
