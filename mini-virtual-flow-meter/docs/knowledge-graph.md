# Knowledge Graph - mini-virtual-flow-meter

## L1: Definitions — Complete

| # | Definition | Struct / Enum | Location |
|---|-----------|---------------|----------|
| 1 | VFM Model Types (orifice, venturi, Bernoulli, pump, choke, hybrid) | `vfm_model_type_t` | `include/virtual_flow_meter.h` |
| 2 | Flow Regime (laminar, transitional, turbulent, critical) | `vfm_regime_t` | `include/virtual_flow_meter.h` |
| 3 | VFM Operational Status | `vfm_status_t` | `include/virtual_flow_meter.h` |
| 4 | Process Sensor with Uncertainty | `vfm_sensor_t` | `include/virtual_flow_meter.h` |
| 5 | Single-Phase Fluid Properties | `vfm_fluid_t` | `include/virtual_flow_meter.h` |
| 6 | Multi-Phase Flow Properties | `vfm_multiphase_t` | `include/virtual_flow_meter.h` |
| 7 | VFM Configuration | `vfm_config_t` | `include/virtual_flow_meter.h` |
| 8 | VFM State Vector (Kalman) | `vfm_state_t` | `include/virtual_flow_meter.h` |
| 9 | VFM Result with Uncertainty | `vfm_result_t` | `include/virtual_flow_meter.h` |
| 10 | Orifice Plate Parameters (ISO 5167) | `orifice_params_t` | `include/flow_models.h` |
| 11 | Venturi Tube Parameters | `venturi_params_t` | `include/flow_models.h` |
| 12 | Bernoulli Equation Parameters | `bernoulli_params_t` | `include/flow_models.h` |
| 13 | Pump Curve Parameters | `pump_curve_params_t` | `include/flow_models.h` |
| 14 | Choke Valve Parameters | `choke_params_t` | `include/flow_models.h` |
| 15 | Pipe Segment Geometry | `pipe_segment_t` | `include/pipeline_geometry.h` |
| 16 | Pipe Fitting Loss Coefficient | `pipe_fitting_t` | `include/pipeline_geometry.h` |
| 17 | Uncertainty Component (GUM) | `vfm_uncertainty_component_t` | `include/vfm_uncertainty.h` |
| 18 | Uncertainty Budget | `vfm_uncertainty_budget_t` | `include/vfm_uncertainty.h` |
| 19 | Kalman Filter State | `vfm_kalman_t` | `include/vfm_state_estimation.h` |
| 20 | RLS Parameter Estimator | `vfm_rls_t` | `include/vfm_state_estimation.h` |
| 21 | MHE Configuration | `vfm_mhe_t` | `include/vfm_state_estimation.h` |

## L2: Core Concepts — Complete

| # | Concept | Implementation |
|---|---------|---------------|
| 1 | Soft Sensing / Inferential Measurement | `vfm_estimate()` |
| 2 | Sensor Fusion (Inverse-Variance Weighting) | `vfm_sensor_fusion_wls()` |
| 3 | Sensor Health Monitoring | `vfm_sensor_health_check()` |
| 4 | Flow Regime Classification | `flow_regime_classify()` |
| 5 | Reynolds Number | `flow_reynolds_number()` |
| 6 | Confidence Interval | `vfm_coverage_factor()`, `vfm_expanded_uncertainty()` |
| 7 | Mass Balance Reconciliation | `dr_node_mass_balance()` |
| 8 | Gross Error Detection | `dr_global_test()` |
| 9 | Drift Detection (CUSUM) | `vfm_cusum_drift_detect()` |

## L3: Engineering Structures — Complete

| # | Structure | Implementation |
|---|-----------|---------------|
| 1 | Hierarchical VFM Config | `vfm_config_t` with primary/fallback model |
| 2 | Multi-Sensor Array Management | `vfm_sensor_register()` with max_sensors |
| 3 | Pipeline Network Representation | `pipeline_config_t` with segments+fittings |
| 4 | Uncertainty Budget with Components | `vfm_uncertainty_budget_t` (up to 16 components) |
| 5 | Combined State Estimator Architecture | KF + RLS + MHE + CUSUM |

## L4: Engineering Laws — Complete

| # | Law / Standard | Implementation |
|---|---------------|---------------|
| 1 | ISO 5167-1 Orifice Flow | `orifice_discharge_coeff_iso5167()`, `orifice_mass_flow()` |
| 2 | ISO 5167-4 Venturi Flow | `venturi_mass_flow()` |
| 3 | Bernoulli's Principle (1738) | `bernoulli_flow_rate()` |
| 4 | Darcy-Weisbach Equation (1857) | `darcy_weisbach_head_loss()` |
| 5 | Colebrook-White Friction Factor (1939) | `darcy_friction_colebrook()` |
| 6 | Hagen-Poiseuille Laminar Flow | `darcy_friction_laminar()` |
| 7 | Pump Affinity Laws | `pump_curve_flow_estimate()` |
| 8 | IEC 60534-2-1 Control Valve Sizing | `choke_valve_liquid_flow()`, `choke_valve_gas_flow()` |
| 9 | Ideal Gas Law | `gas_density_ideal()` |
| 10 | Sutherland Viscosity (1893) | `gas_viscosity_sutherland()` |
| 11 | Andrade Viscosity Correlation | `liquid_viscosity_andrade()` |
| 12 | Standing Oil Correlation (1947) | `oil_bubble_point_standing()`, `oil_live_density()` |
| 13 | AGA-8 Compressibility (DAK) | `gas_compressibility_dak()` |
| 14 | GUM (JCGM 100:2008) | `vfm_uncertainty_budget_init/combine/expand()` |
| 15 | Welch-Satterthwaite DOF | `vfm_welch_satterthwaite_dof()` |

## L5: Algorithms/Methods — Complete

| # | Algorithm | Implementation |
|---|-----------|---------------|
| 1 | Linear Kalman Filter | `vfm_kalman_init/predict/update()` |
| 2 | Recursive Least Squares (RLS) | `vfm_rls_init/update()` |
| 3 | Moving Horizon Estimation (MHE) | `vfm_mhe_init/step()` |
| 4 | Monte Carlo Uncertainty (GUM S1) | `vfm_monte_carlo_uncertainty()` |
| 5 | Newton-Raphson Colebrook Solver | within `darcy_friction_colebrook()` |
| 6 | Darcy-Weisbach Inverse Flow Solver | `darcy_inverse_flow()` |
| 7 | CUSUM Drift Detection | `vfm_cusum_drift_detect()` |
| 8 | Chisholm Two-Phase Multiplier | `chisholm_phi_liquid_sq()` |
| 9 | Beggs & Brill Flow Pattern | `multiphase_pattern_beggs_brill()` |
| 10 | Lagrange Multiplier Reconciliation | `dr_linear_reconciliation()` |

## L6: Canonical Problems — Complete

| # | Problem | Location |
|---|---------|----------|
| 1 | Orifice Plate Flow Metering | `example_orifice_flow.c` |
| 2 | Pump Station Flow Estimation | `example_pump_vfm.c` |
| 3 | Kalman Filter Flow Tracking | `example_kalman_vfm.c` |
| 4 | Oil Production Allocation Reconciliation | `example_data_recon.c` |
| 5 | Multi-Phase Well Flow VFM | `example_multiphase.c` |
| 6 | Darcy Head Loss Inverse Problem | `darcy_inverse_flow()` |
| 7 | Orifice Uncertainty Budget | `vfm_orifice_relative_uncertainty()` |

## L7: Industrial Applications — Partial (3/5)

| # | Application | Implementation |
|---|------------|---------------|
| 1 | Oil & Gas Production Allocation | `dr_node_mass_balance()`, `example_data_recon.c` |
| 2 | Subsea VFM (Multi-phase) | `multiphase_flow_from_dp()`, `example_multiphase.c` |
| 3 | Pump Station Monitoring | `pump_curve_flow_estimate()`, `example_pump_vfm.c` |
| 4 | Natural Gas Custody Transfer | `gas_compressibility_dak()`, `orifice_mass_flow()` |
| 5 | Control Valve Flow Estimation | `choke_valve_liquid_flow()`, `choke_valve_gas_flow()` |

## L8: Advanced Topics — Partial (3/5)

| # | Topic | Implementation |
|---|-------|---------------|
| 1 | Adaptive Kalman Filtering | `vfm_kalman_adapt_noise()` |
| 2 | Monte Carlo Uncertainty Propagation | `vfm_monte_carlo_uncertainty()` |
| 3 | Online Parameter Estimation (RLS) | `vfm_rls_update()` |
| 4 | Moving Horizon Estimation | `vfm_mhe_step()` (simplified) |
| 5 | Bayesian Sensor Fusion | Documented, partial implementation |

## L9: Research Frontiers — Partial

| # | Topic | Status |
|---|-------|--------|
| 1 | Digital Twin VFM | Documented |
| 2 | Edge AI VFM | Documented |
| 3 | Autonomous Recalibration | `enable_auto_calibration` flag in config |