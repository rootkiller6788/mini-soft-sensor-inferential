# Course Alignment - mini-virtual-flow-meter

## MIT 6.302 / 2.171
- **State estimation (Kalman filter)**: `vfm_kalman_t`, `src/vfm_state_estimation.c`
- **System identification (RLS)**: `vfm_rls_t`, online parameter estimation

## Stanford ENGR205
- **Soft sensors**: Core VFM concept, entire module
- **Inferential control**: `vfm_estimate()` replaces physical sensor

## Berkeley ME233 / EE C128
- **Sensor fusion**: `vfm_sensor_fusion_wls()`, inverse-variance weighting
- **Kalman filtering**: 3-state VFM Kalman filter implementation

## CMU 24-677
- **Optimal estimation**: MHE, Kalman gain derivation
- **Constrained optimization**: Lagrange multiplier reconciliation

## Georgia Tech AE 6530
- **Monte Carlo methods**: `vfm_monte_carlo_uncertainty()` (GUM Supplement 1)
- **Uncertainty propagation**: Taylor series, Welch-Satterthwaite

## Purdue ME 575
- **Industrial flow measurement**: ISO 5167 orifice/venturi implementation
- **Pump curves**: Pump affinity laws for VFM

## RWTH Aachen
- **Process automation**: VFM as soft sensor in DCS
- **Field instrumentation**: Sensor health monitoring, drift detection

## Tsinghua University
- **Process control engineering**: Virtual flow metering for energy optimization
- **Oil & gas applications**: Multi-phase flow VFM

## ISA/IEC Standards
- **ISO 5167-1:2022**: Orifice flow measurement
- **ISO 5167-4:2022**: Venturi flow measurement
- **IEC 60534-2-1**: Control valve flow
- **JCGM 100:2008 (GUM)**: Uncertainty in measurement
- **AGA Report No. 8**: Gas compressibility
- **Crane TP-410**: Fitting loss coefficients