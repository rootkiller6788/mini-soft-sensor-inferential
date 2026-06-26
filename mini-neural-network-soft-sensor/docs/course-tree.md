# Course Tree — Dependency Structure

## Prerequisites (what this module depends on)

```
mini-neural-network-soft-sensor
├── Linear Algebra (matrix ops, vector spaces)
├── Calculus (chain rule for backpropagation)
├── Probability & Statistics (R², AIC/BIC, distributions)
├── Optimization Theory (gradient descent, convexity)
├── Process Control Fundamentals (PID, feedback)
├── Numerical Methods (floating point, conditioning)
├── mini-pid-control-engineering (basic process control)
└── mini-soft-sensor-inferential (parent module: soft sensor concepts)
```

## Dependents (what depends on this module)

```
mini-neural-network-soft-sensor provides →
├── mini-industrial-ai-control-fusion (NN for control)
│   └── Neural network models for AI-based control
├── mini-soft-sensor-maintenance-aging (model maintenance)
│   └── Performance monitoring, drift detection
├── mini-quality-estimator-inferential (quality estimation)
│   └── Neural network for quality variable inference
└── mini-virtual-flow-meter (virtual instrumentation)
    └── Data-driven virtual sensor methodology
```

## Internal Dependency Tree

```
Neural Network Soft Sensor
│
├── L1 Definitions
│   ├── Activation types → L2 Forward pass
│   ├── Loss types → L5 Training
│   ├── Optimizer types → L5 Weight updates
│   └── Sensor types → L7 Deployment
│
├── L2 Core Concepts
│   ├── Forward propagation → L2 Inference
│   ├── Backward propagation → L5 Gradient computation
│   └── Weight initialization → L1 Network creation
│
├── L3 Engineering Structures
│   ├── Matrix ops → L2 Forward/backward passes
│   ├── Normalization → L6 Data preprocessing
│   └── Dataset splits → L5 Training, L4 Validation
│
├── L4 Engineering Laws
│   ├── AIC/BIC → L6 Model selection
│   ├── Universal approximation → L2 Architecture design
│   └── Backpropagation theorem → L5 Gradient computation
│
├── L5 Algorithms
│   ├── Optimizers → L5 Training loop
│   ├── Loss functions → L5 Backpropagation
│   ├── Regularization → L5 Generalization
│   └── LR schedules → L5 Convergence
│
├── L6 Canonical Problems
│   ├── Distillation → L7 Industrial deployment
│   ├── Fermentation → L7 Bioprocess monitoring
│   └── Polymerization → L7 Quality control
│
├── L7 Applications
│   ├── DCS integration → L9 IT/OT fusion
│   └── Performance monitoring → L8 Adaptive sensors
│
├── L8 Advanced Topics
│   ├── Ensemble methods → L6 Robust prediction
│   ├── MC Dropout → L6 Uncertainty quantification
│   └── Digital twin → L9 Industry 4.0
│
└── L9 Frontiers
    ├── IT/OT convergence (documented)
    ├── Edge AI (conceptual)
    └── Autonomous operations (conceptual)
```
