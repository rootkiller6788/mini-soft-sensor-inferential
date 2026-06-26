# Course Alignment — Inferential Quality Estimation

## Nine-School Curriculum Mapping

### MIT (6.302 Feedback Systems / 2.171 Digital Control)
- State estimation and observer design → Kalman filter implementation
- Discrete-time Kalman filter → `kf_predict()`, `kf_update()`
- Sampling and discretization → ARX models, multi-rate fusion

### Stanford (ENGR205 Process Control / AA272 GNSS)
- Soft sensing and inferential measurement → Full quality estimator framework
- Multi-sensor fusion → Multi-rate Kalman fusion
- GPS state estimation → Quality state estimation analogy

### Berkeley (ME233 Advanced Control / EE C128 Mechatronics)
- Stochastic estimation → Kalman filter, EKF, CKF
- Sensor fusion → Multi-rate fusion algorithms
- Mechatronics sensing → Process variable validation

### CMU (24-677 Advanced Control Systems)
- Multi-rate systems → `mrf_kalman_t`, delayed measurement updates
- State estimation for control → Kalman-based quality prediction
- Advanced estimation algorithms → RLS, VFF-RLS, directional forgetting

### Georgia Tech (ECE 6550 Nonlinear Control / AE 6530 Optimal Estimation)
- Nonlinear estimation → EKF for Arrhenius kinetics, CKF
- Cubature Kalman Filter → `ckf_alloc()`, `ckf_step()`
- Aerospace estimation → Heritage in Kalman filter design

### Purdue (ME 575 Industrial Control)
- Industrial soft sensors → Inferential quality estimation
- Bias update procedures → Additive/multiplicative/Kalman bias
- Operator handoff → Lab sample validation and rejection

### RWTH Aachen (Industrial Control Systems)
- PLC-based soft sensors → Real-time estimator cycle
- IEC 61131-3 structured text patterns → C struct-based API
- German Industry 4.0 → IT/OT fusion concepts

### Tsinghua (过程控制工程 / 工业互联网)
- 软测量与推断控制 → Core inferential estimation
- 在线系统辨识 → RLS and recursive PLS
- 中国制造2025 → Industrial IoT readiness

### ISA/IEC Standards
- ISA-88 batch quality → Quality tracking across batches
- IEC 61511 functional safety → Estimator health and fault modes
- ISO 5725 measurement uncertainty → Confidence bounds, lab validation

## Key Textbook References

1. Seborg, Edgar, Mellichamp (2016) *Process Dynamics and Control*, Ch. 20 — Inferential Control
2. Simon (2006) *Optimal State Estimation* — Kalman filtering
3. Ljung (1999) *System Identification* — RLS, ARX models
4. Fortuna et al. (2007) *Soft Sensors for Monitoring and Control*
5. Kadlec et al. (2009) *Data-driven Soft Sensors in the Process Industry*
6. Qin (2014) *Process Data Analytics in the Era of Big Data*
