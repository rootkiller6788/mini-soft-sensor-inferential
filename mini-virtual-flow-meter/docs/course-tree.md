# Course Dependency Tree - mini-virtual-flow-meter

## Prerequisites (modules this depends on)
```
0. mini-industrial-measurement-actuator
   -> Sensor models, uncertainty basics
   
1. mini-pid-control-engineering
   -> Feedback concepts (used in drift compensation)

4. mini-plc-iec61131-fundamentals
   -> Discrete-time implementation context

10. mini-safety-instrumented-system
   -> Uncertainty quantification for safety
   
14. mini-soft-sensor-inferential (parent)
   -> Soft sensor / inferential measurement paradigm
```

## Knowledge Dependency Graph
```
Fluid Mechanics  ─┬─ Bernoulli's Principle
 (Bernoulli 1738) │
                  ├─ Darcy-Weisbach (1857)
                  │   └─ Colebrook (1939)
                  │       └─ VFM from head loss
                  │
                  ├─ Reynolds Number (1883)
                  │   └─ Flow regime classification
                  │       └─ Model selection
                  │
                  └─ ISO 5167 (orifice/venturi)
                      └─ Discharge coefficient
                          └─ Flow-rate -> dP mapping

Thermodynamics  ──┬─ Ideal Gas Law
                   │   └─ Real gas (AGA-8/DAK)
                   │
                   ├─ Sutherland viscosity
                   └─ Standing oil correlation

Estimation Theory  ─┬─ Kalman Filter (1960)
 (Kalman 1960)      │   ├─ Linear KF for VFM state
                    │   └─ Adaptive noise tuning
                    │
                    ├─ RLS (Recursive Least Squares)
                    │   └─ Online Cd calibration
                    │
                    ├─ MHE (Moving Horizon)
                    │   └─ Constrained estimation
                    │
                    └─ CUSUM (Page 1954)
                        └─ Drift detection

Metrology (GUM)  ───┬─ Uncertainty budget
                     ├─ Taylor propagation
                     ├─ Monte Carlo (GUM S1)
                     └─ Welch-Satterthwaite DOF

Multi-Phase Flow  ───┬─ Beggs & Brill (1973)
 (oil & gas)         ├─ Lockhart-Martinelli (1949)
                     ├─ Chisholm correlation (1967)
                     └─ Flow pattern maps
```

## Build Order
1. Fluid properties (no dependencies)
2. Flow models (depends on fluid properties)
3. Pipeline geometry (depends on flow models)
4. State estimation (depends on flow models)
5. Uncertainty quantification (depends on all above)
6. Data reconciliation (depends on flow models)
7. Multi-phase flow (depends on fluid + flow models)