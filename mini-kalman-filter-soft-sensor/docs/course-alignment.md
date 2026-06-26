# Course Alignment — mini-kalman-filter-soft-sensor

## Nine-School Curriculum Mapping

### MIT — 6.302 Feedback Systems / 2.171 Digital Control

| Chapter | Topic | Module Coverage |
|---------|-------|----------------|
| Kalman Filter Theory | Optimal state estimation | `kalman_core.c` — full KF implementation |
| Riccati Equation | DARE solution, convergence | `kf_solve_dare()` — iterative DARE solver |
| Stochastic Systems | Process/measurement noise modeling | `KalmanModel` with Q, R covariance matrices |
| Observer Design | State estimation from measurements | All KF variants (KF, EKF, UKF) |

### Stanford — ENGR205 Process Control / AA272 GNSS

| Chapter | Topic | Module Coverage |
|---------|-------|----------------|
| Soft Sensors | Inferential quality estimation | `IndustrialQualityEstimator` — quality soft sensor |
| Process Monitoring | NIS/NEES diagnostics | `kf_diagnostics()` with chi-square thresholds |
| GPS Navigation | INS/GPS integration | `INSGPSNavigator` — 9-state navigation filter |
| Kalman Filter Applications | Process + navigation | Two L6 examples covering both domains |

### Berkeley — ME233 Advanced Control / EE C128 Mechatronics

| Chapter | Topic | Module Coverage |
|---------|-------|----------------|
| State Estimation | KF, EKF, UKF comparison | All three variants implemented |
| DC Motor Control | Velocity/current estimation | `DCMotorEstimator` — 2-state augmented model |
| Nonlinear Filtering | EKF, UKF, particle filter | EKF + UKF (particle filter documented) |
| Optimal Filtering | MMSE optimality proofs | `kalman_model.lean` — formalized theory |

### CMU — 24-677 Advanced Control Systems / 18-771 Linear Systems

| Chapter | Topic | Module Coverage |
|---------|-------|----------------|
| Linear Systems Theory | State-space models | `KalmanModel` with F, B, H matrices |
| Matrix Computations | Cholesky, QR, condition numbers | `kalman_matrix_ops.c` — full LA library |
| Optimal Estimation | KF derivations, Joseph form | `kf_joseph_update()` — numerically stable |
| Formal Verification | Lean proofs of KF properties | `kalman_model.lean` — 15+ theorems |

### Georgia Tech — AE 6530 Optimal Estimation / ECE 6550 Nonlinear Control

| Chapter | Topic | Module Coverage |
|---------|-------|----------------|
| Optimal Filtering | Wiener/Kalman theory | Full KF implementation |
| Nonlinear Estimation | EKF, UKF | Both implemented with iterative variants |
| Aerospace Applications | INS/GPS, target tracking | GPS/INS navigator example |
| Adaptive Filtering | Noise covariance estimation | `AKFState` with cov matching + Bayesian |

### Purdue — ME 575 Industrial Control / ECE 602 Lumped Systems

| Chapter | Topic | Module Coverage |
|---------|-------|----------------|
| Industrial Estimation | Soft sensors for quality | Quality estimator with lab calibration |
| Process Control | CSTR, distillation | CSTR estimator with conversion inference |
| Multi-Model Estimation | IMM for process monitoring | `AKFMultiModel` — full IMM implementation |

### RWTH Aachen — Industrial Control Systems / PLC-SCADA Engineering

| Chapter | Topic | Module Coverage |
|---------|-------|----------------|
| PLC-Based Estimation | KF on industrial controllers | Fixed-dimension KF (KF_MAX_STATE_DIM=12) |
| DCS Integration | Honeywell Experion, Siemens PCS7 | `SoftSensorConfig` with DCS tag names |
| Historian Interface | OSIsoft PI, PHD | `HistorianSnapshot` with OPC quality flags |
| SCADA Soft Sensors | Operator-facing quality displays | Quality estimator with confidence bounds |

### Tsinghua — 过程控制工程 / 运动控制

| Chapter | Topic | Module Coverage |
|---------|-------|----------------|
| Process Control Engineering | Soft sensing methodology | Complete KF soft sensor framework |
| Motion Control | DC motor velocity estimation | `DCMotorEstimator` with torque output |
| Industrial Internet | IT/OT integration | Historian snapshot interface |

### ISA/IEC Standards

| Standard | Relevance | Module Coverage |
|----------|----------|----------------|
| ISA-88 | Batch process quality estimation | Quality estimator lab calibration cycle |
| ISA-101 | HMI soft sensor displays | Quality with confidence bounds for operators |
| IEC 61131-3 | PLC implementation constraints | Fixed-dimension design (KF_MAX_STATE_DIM=12) |
| IEC 61511 | Safety instrumented functions | Sensor fault detection, innovation monitoring |
| OPC UA | Data quality flags | `HistorianSnapshot.quality` (OPC-compliant) |
