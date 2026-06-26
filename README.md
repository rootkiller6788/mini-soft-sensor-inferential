# Mini Soft Sensor & Inferential Measurement

A collection of **from-scratch, zero-dependency C implementations** of soft sensor and inferential measurement algorithms for process control engineering. Each module maps to MIT, Stanford, Berkeley, and CMU courses, bridging theory and practice by translating textbook equations into runnable C code.

## Sub-Modules

| Sub-Module | Topics | Key Courses |
|------------|--------|-------------|
| [mini-kalman-filter-soft-sensor](mini-kalman-filter-soft-sensor/) | Linear KF, EKF, UKF, adaptive noise estimation, Kalman smoothing, industrial soft sensor applications | MIT 6.302, Stanford ENGR205, CMU 24-677 |
| [mini-neural-network-soft-sensor](mini-neural-network-soft-sensor/) | Feedforward NN for quality estimation, backprop/SGD training, model validation, performance monitoring | MIT 6.390, Stanford CS229, CMU 10-701 |
| [mini-pls-partial-least-squares](mini-pls-partial-least-squares/) | NIPALS, SIMPLS, kernel PLS, recursive/online PLS, cross-validation, latent-variable soft sensor modeling | MIT 2.171, Stanford ENGR205, UMetrics |
| [mini-quality-estimator-inferential](mini-quality-estimator-inferential/) | First-principles & hybrid inferential models, Kalman quality estimation, bias correction, multi-rate sensor fusion | MIT 6.302, Stanford ENGR205, Berkeley ME233 |
| [mini-soft-sensor-data-driven-pca](mini-soft-sensor-data-driven-pca/) | PCA decomposition (SVD/EVD), kernel PCA, adaptive/recursive PCA, T²/SPE monitoring, inferential sensing | MIT 2.171, Stanford ENGR205, CMU 24-677 |
| [mini-soft-sensor-data-reconciliation](mini-soft-sensor-data-reconciliation/) | Steady-state/dynamic data reconciliation, WLS solver, gross error detection, measurement redundancy analysis | MIT 10.490, Stanford ENGR205, CMU 24-677 |
| [mini-soft-sensor-maintenance-aging](mini-soft-sensor-maintenance-aging/) | Performance metrics (RMSE/R²/SPE), aging drift detection, adaptive model updating, ensemble maintenance strategies | MIT 2.171, Stanford ENGR205, CMU 24-677 |
| [mini-virtual-flow-meter](mini-virtual-flow-meter/) | Virtual flow metering via process measurements, fluid properties, pipeline hydraulics, state estimation, uncertainty quantification | MIT 10.490, Stanford ENGR205, TU Delft P&E |

## Design Philosophy

- **Zero external dependencies** — pure C (C99/C11), only `libc` and `libm`
- **Self-contained modules** — each directory has its own `Makefile`, `include/`, `src/`, `examples/`, `demos/`, `tests/`
- **Theory-to-code mapping** — every module includes `docs/` with course-alignment notes
- **Industrial relevance** — all implementations target real process control scenarios: quality estimation, sensor validation, fault detection, and virtual instrumentation

## Building

Each module is standalone. Navigate to a module directory and run:

```bash
cd mini-kalman-filter-soft-sensor
make all    # build everything
make test   # run tests
```

Requires **GCC** and **GNU Make**.

## Project Structure

```
mini-soft-sensor-inferential/
├── mini-kalman-filter-soft-sensor/        # Kalman filtering for soft sensor applications
├── mini-neural-network-soft-sensor/       # Neural network-based inferential sensors
├── mini-pls-partial-least-squares/        # Partial Least Squares regression
├── mini-quality-estimator-inferential/    # Inferential quality estimation & sensor fusion
├── mini-soft-sensor-data-driven-pca/      # PCA-driven soft sensors & process monitoring
├── mini-soft-sensor-data-reconciliation/  # Data reconciliation & gross error detection
├── mini-soft-sensor-maintenance-aging/    # Soft sensor maintenance & aging management
└── mini-virtual-flow-meter/               # Virtual flow metering & state estimation
```

## License

MIT
