# Course Alignment — Data Reconciliation

## Nine-School Curriculum Mapping

### MIT — 6.302 Feedback Systems / 2.171 Digital Control
- **State estimation**: Chapter 4, Kalman filter theory
- **Observability**: Gramian analysis, structural observability
- **Mapping**: dr_dynamic.c (Kalman filter, RTS smoother)
- **Mapping**: dr_redundancy.c (observability analysis)

### Stanford — ENGR205 Process Control / EE392 Industrial AI
- **Process monitoring**: Chapter 8, data reconciliation fundamentals
- **Gross error detection**: Chapter 9, statistical process control
- **Mapping**: dr_gross_error.c (GT, NT, MT, PCA-based detection)
- **Mapping**: dr_core.c (WLS reconciliation with multiple solvers)

### Berkeley — ME233 Advanced Control / EE C128 Mechatronics
- **Optimal estimation**: Chapters 5-7, Kalman filtering and smoothing
- **Sensor fusion**: combining redundant measurements for improved accuracy
- **Mapping**: dr_dynamic.c (Kalman predict/update, RTS smoothing)
- **Mapping**: dr_core.c (multi-sensor reconciliation)

### CMU — 24-677 Advanced Control Systems
- **Linear systems theory**: observability, controllability, Gramians
- **Estimation theory**: MMSE, BLUE, information filter
- **Mapping**: dr_redundancy.c (observability, structural rank)
- **Mapping**: dr_core.c (WLS as BLUE estimator)

### Georgia Tech — ECE 6550 Nonlinear Control / AE 6530 Optimal Estimation
- **Robust estimation**: M-estimators, influence functions, breakdown point
- **Nonlinear filtering**: EKF, UKF, particle filters for dynamic DR
- **Mapping**: dr_gross_error.c (Huber, Biweight, Cauchy, Fair M-estimators)
- **Mapping**: dr_dynamic.c (dynamic DR framework)

### Purdue — ECE 602 Lumped Systems / ME 575 Industrial Control
- **Industrial process control**: data reconciliation in plant operations
- **Sensor validation**: gross error detection and identification
- **Mapping**: dr_gross_error.c (serial elimination, simultaneous estimation)
- **Mapping**: examples/example2_gross_error.c (steam metering case)

### RWTH Aachen — Industrial Control Systems
- **VDI 2048**: German standard for data reconciliation
- **PLC/SCADA integration**: real-time data quality management
- **Mapping**: All files reference VDI 2048
- **Mapping**: dr_core.c (Lagrange multiplier method from German literature)

### Tsinghua University — 过程控制工程 / 工业互联网
- **Process control engineering**: mass/energy balance reconciliation
- **Industrial Internet**: real-time data quality for digital twins
- **Mapping**: examples/example1_flow_network.c (flow network)
- **Mapping**: dr_dynamic.c (real-time dynamic DR)

### ISA/IEC Standards
- **ISA-88**: batch control data handling
- **IEC 61508/61511**: functional safety — sensor validation requirements
- **ISO 5167**: orifice flow measurement
- **Mapping**: dr_measurement.c (GUM uncertainty propagation)
- **Mapping**: dr_gross_error.c (statistical tests for sensor validation)

## Reference Textbooks
| Textbook | Author(s) | Year | Topics Covered |
|----------|-----------|------|----------------|
| Data Reconciliation & Gross Error Detection | Narasimhan, Jordache | 2000 | L1-L6 (complete DR theory) |
| Data Processing & Reconciliation | Romagnoli, Sanchez | 2000 | L1-L7 (applications) |
| Applied Optimal Estimation | Gelb | 1974 | L4-L5 (Kalman filtering) |
| Optimal State Estimation | Simon | 2006 | L5 (Kalman, H-infinity, particle) |
| Robust Statistics | Huber, Ronchetti | 2009 | L5, L8 (M-estimation) |
| Matrix Computations | Golub, Van Loan | 2013 | L3 (QR, Cholesky, SVD) |
| VDI 2048 Part 1 | VDI | 2008 | L4, L7 (German DR standard) |
