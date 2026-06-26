# Knowledge Graph — Neural Network Soft Sensor

## L1: Definitions

| # | Definition | C Type | Lean Definition |
|---|-----------|--------|-----------------|
| 1 | Activation Function Types (9 types) | `nn_activation_type_t` | `ActivationType` |
| 2 | Loss Function Types (6 types) | `nn_loss_type_t` | `LossType` |
| 3 | Optimizer Types (8 types) | `nn_optimizer_type_t` | `OptimizerType` |
| 4 | Regularization Types (7 types) | `nn_regularization_type_t` | `RegularizationType` |
| 5 | Soft Sensor Types (5 types) | `soft_sensor_type_t` | `SoftSensorType` |
| 6 | Normalization Types (4 types) | `nn_normalization_type_t` | — |
| 7 | Neural Network Layer | `nn_layer_t` | `LayerSpec` |
| 8 | Neural Network Architecture | `nn_architecture_t` | `NetworkArchitecture` |
| 9 | Complete Neural Network | `nn_network_t` | — |
| 10 | Soft Sensor | `soft_sensor_t` | `SoftSensor` |
| 11 | Dataset | `nn_dataset_t` | — |
| 12 | Performance Metrics | `nn_regression_metrics_t` | — |
| 13 | Ensemble Network | `nn_ensemble_t` | `Ensemble` |
| 14 | MC Dropout (Bayesian) | `nn_mc_dropout_t` | — |
| 15 | Industrial Deployment State | `nn_industrial_soft_sensor_t` | — |
| 16 | Learning Rate Schedules (6 types) | `nn_lr_schedule_type_t` | — |
| 17 | Sensor Operational Modes | — | `SensorMode` |

## L2: Core Concepts

| # | Concept | Implementation |
|---|---------|---------------|
| 1 | Forward Propagation | `nn_layer_forward()`, `nn_network_forward()` |
| 2 | Backward Propagation | `nn_layer_backward()`, `nn_network_backward()` |
| 3 | Gradient Descent | `nn_optimizer_sgd_step()` |
| 4 | Mini-batch Training | `nn_network_train_epoch()` |
| 5 | Weight Initialization (He/Xavier) | `nn_he_initialize()`, `nn_xavier_initialize()` |
| 6 | Activation Functions (9 types) | `activation_forward()` |
| 7 | Loss Minimization | `nn_network_train()` |
| 8 | Model Persistence | `nn_network_save()`, `nn_network_load()` |

## L3: Engineering Structures

| # | Structure | Implementation |
|---|-----------|---------------|
| 1 | Matrix Operations | `matrix_vector_mul()` |
| 2 | Data Normalization Pipeline | `nn_dataset_normalize()` |
| 3 | Train/Val/Test Split | `nn_dataset_create()` |
| 4 | Optimizer State Management | Adam state, momentum buffers |
| 5 | Dropout Implementation | `nn_dropout_apply()` |
| 6 | Batch Normalization | `nn_batch_normalize()` |
| 7 | Loss Gradient Computation | All loss functions with gradients |
| 8 | Layer Architecture Management | `nn_layer_t` with prev/next links |

## L4: Engineering Laws/Theorems

| # | Theorem/Law | Source | Proof |
|---|------------|--------|-------|
| 1 | Universal Approximation Theorem | Cybenko (1989), Hornik (1991) | `has_hidden_layer` |
| 2 | Backpropagation (chain rule) | Rumelhart et al. (1986) | Code in `nn_network_backward()` |
| 3 | AIC Model Selection | Akaike (1974) | `nn_aic_criterion()`, Lean `aic_penalty` |
| 4 | BIC Model Selection | Schwarz (1978) | `nn_bic_criterion()`, Lean `bic_penalty_bound` |
| 5 | Bias-Variance Tradeoff | Geman et al. (1992) | L1/L2 regularization |
| 6 | Dropout as Regularization | Srivastava et al. (2014) | `nn_dropout_apply()` |
| 7 | Adam Convergence | Kingma & Ba (2014) | `nn_optimizer_adam_step()` |
| 8 | He Initialization | He et al. (2015) | `nn_he_initialize()` |
| 9 | Xavier Initialization | Glorot & Bengio (2010) | `nn_xavier_initialize()` |
| 10 | Sensor Valid Transition | — | Lean `valid_transition` theorems |
| 11 | BIC > AIC penalty (n ≥ 3) | — | `bic_penalty_gt_aic_penalty` |
| 12 | R² Identity (SS_res = 0) | — | `r2_perfect` |

## L5: Algorithms/Methods

| # | Algorithm | Complexity | Source |
|---|-----------|-----------|--------|
| 1 | SGD | O(n_params) | Robbins & Monro (1951) |
| 2 | SGD + Momentum | O(n_params) | Polyak (1964) |
| 3 | Nesterov Accelerated Gradient | O(n_params) | Nesterov (1983) |
| 4 | AdaGrad | O(n_params) | Duchi et al. (2011) |
| 5 | RMSprop | O(n_params) | Tieleman & Hinton (2012) |
| 6 | Adam | O(n_params) | Kingma & Ba (2014) |
| 7 | AdaMax | O(n_params) | Kingma & Ba (2014) |
| 8 | Nadam | O(n_params) | Dozat (2016) |
| 9 | MSE Loss + Gradient | O(n) | Standard |
| 10 | MAE Loss + Gradient | O(n) | Standard |
| 11 | Huber Loss + Gradient | O(n) | Huber (1964) |
| 12 | MAPE Loss + Gradient | O(n) | Standard |
| 13 | L1 Regularization | O(n_params) | Tibshirani (1996) |
| 14 | L2 Regularization | O(n_params) | Hoerl & Kennard (1970) |
| 15 | Elastic Net | O(n_params) | Zou & Hastie (2005) |
| 16 | Dropout | O(n_neurons) | Srivastava et al. (2014) |
| 17 | Batch Normalization | O(m*n_features) | Ioffe & Szegedy (2015) |
| 18 | Learning Rate Schedules (6 types) | O(1) | Various |
| 19 | Xorshift32 RNG | O(1) | Marsaglia (2003) |
| 20 | Box-Muller Normal | O(1) | Box & Muller (1958) |
| 21 | K-Fold Cross Validation | O(k*e*n*n_params) | Standard ML |
| 22 | CUSUM Drift Detection | O(1) | Page (1954) |
| 23 | Paired t-test Model Comparison | O(n) | Standard stats |

## L6: Canonical Problems

| # | Problem | Example/Generator |
|---|---------|------------------|
| 1 | Distillation Column Composition Estimation | `nn_generate_distillation_data()` |
| 2 | Fermentation Biomass Estimation | `nn_generate_fermentation_data()` |
| 3 | Polymerization Melt Index Soft Sensor | `nn_generate_polymerization_data()` |
| 4 | Cement Kiln Free Lime Prediction | `nn_generate_cement_kiln_data()` |
| 5 | FCC Unit Gasoline Yield | `nn_generate_fcc_data()` |
| 6 | Blast Furnace Hot Metal Quality | `nn_generate_blast_furnace_data()` |

## L7: Industrial Applications

| # | Application | Implementation |
|---|------------|---------------|
| 1 | DCS-Integrated Soft Sensor (Honeywell Experion) | `nn_industrial_sensor_init()` |
| 2 | Historian Data Interface (OSIsoft PI) | `nn_industrial_soft_sensor_t` |
| 3 | Online Performance Monitoring | `soft_sensor_monitor_performance()` |
| 4 | Concept Drift Detection (CUSUM) | `soft_sensor_detect_drift()` |
| 5 | Model Comparison for Deployment | `soft_sensor_compare_models()` |
| 6 | OPC Server Connectivity | `opc_server` field |

## L8: Advanced Topics

| # | Topic | Implementation |
|---|-------|---------------|
| 1 | Ensemble Neural Networks (Bagging) | `nn_ensemble_t`, `nn_ensemble_predict()` |
| 2 | Bayesian NN via MC Dropout | `nn_mc_dropout_t`, `nn_mc_dropout_predict()` |
| 3 | Epistemic vs Aleatoric Uncertainty | MC Dropout variance decomposition |
| 4 | Adaptive Soft Sensors | `SENSOR_TYPE_ADAPTIVE`, drift detection |
| 5 | Digital Twin Integration | `nn_digital_twin_interface_t` |
| 6 | Elastic Net Regularization | `nn_regularization_elastic_net()` |

## L9: Industry Frontiers

| # | Topic | Status |
|---|-------|--------|
| 1 | IT/OT Convergence for Soft Sensors | Documented (DCS + Historian integration) |
| 2 | Edge AI Deployment | `SENSOR_TYPE_ADAPTIVE` enables edge updates |
| 3 | Explainable AI for Process Industry | Residual analysis, Durbin-Watson |
| 4 | Autonomous Operations (L4) | Adaptive retraining triggers |
| 5 | Industrial 5G for Soft Sensors | Conceptual (update_interval_ms field) |
