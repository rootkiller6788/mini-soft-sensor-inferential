# Course Alignment — PCA Data-Driven Soft Sensor

## Nine-School Curriculum Mapping

### MIT
- **6.302 Feedback Systems**: PCA as state estimation → soft sensor as output estimator
- **2.171 Digital Control**: Data-driven modeling for control, system identification
- Map: `pca_pcr_train()` for inferential model identification

### Stanford
- **ENGR205 Process Control**: Multivariate SPC, PCA for fault detection
- **EE392 Industrial AI**: Kernel methods, nonlinear process monitoring
- Map: `pca_monitoring.c` for MSPC, `pca_kernel.c` for nonlinear extension

### Berkeley
- **ME233 Advanced Control**: Kalman filtering vs PCA for state estimation
- **EE C128 Mechatronics**: Sensor fusion, soft sensing
- Map: PCA as static counterpart to dynamic Kalman filter

### CMU
- **24-677 Advanced Control Systems**: Model reduction, system identification
- Map: PCA as dimensionality reduction for control-relevant modeling

### Georgia Tech
- **ECE 6550 Nonlinear Control**: Kernel PCA for nonlinear systems
- **AE 6530 Optimal Estimation**: SVD-based methods
- Map: `pca_svd_decomposition()`, `kpca_train()`

### Purdue
- **ME 575 Industrial Control**: Practical soft sensors, process monitoring
- Map: `examples/example_distillation.c`, `examples/example_reactor.c`

### RWTH Aachen
- **Industrial Control Systems**: Adaptive process monitoring (RPCA, MWPCA)
- Map: `pca_adaptive.c` for time-varying industrial processes

### Tsinghua (清华)
- **Process Control Engineering**: PCA for large-scale process industry
- Map: PCR regression for quality estimation, MSPC for fault detection

### ISA/IEC Standards
- **ISA-88**: Batch process monitoring with MWPCA
- **IEC 61511**: Functional safety — PCA for sensor validation
- Map: T² and SPE for sensor fault detection

## Textbook References

| Textbook | Chapters | Coverage |
|----------|----------|----------|
| Jolliffe, "Principal Component Analysis" (2002) | Ch.3, Ch.6, Ch.8 | PCA theory, PC selection, regression |
| Qin, "Statistical Process Monitoring" (2003) | All | T², SPE, fault detection |
| Golub & Van Loan, "Matrix Computations" (2013) | Ch.8 | Jacobi, QR, SVD |
| MacGregor & Kourti (1995) | - | MSPC for industrial processes |
| Scholkopf, Smola & Muller (1998) | - | Kernel PCA |
