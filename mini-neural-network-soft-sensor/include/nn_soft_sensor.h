/**
 * @file nn_soft_sensor.h
 * @brief Neural Network Soft Sensor API - core creation, training, and inference.
 *
 * Level: L1-L5: All core API functions for neural network soft sensor.
 * Reference: Bishop (1995), Goodfellow et al. (2016), Kadlec et al. (2009)
 *            Fortuna et al., "Soft Sensors for Monitoring and Control..." (2007)
 *            Qin, "Neural networks for intelligent sensors..." CEP (1997)
 *
 * A neural network soft sensor uses a trained feedforward neural network
 * to estimate quality variables (e.g., product composition, melt index,
 * octane number) from routinely measured process variables (temperature,
 * pressure, flow rates, etc.).
 */

#ifndef NN_SOFT_SENSOR_H
#define NN_SOFT_SENSOR_H

#include "nn_sensor_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
 * L1-L2: Memory Management
 *===========================================================================*/

/**
 * @brief Allocate and initialize a neural network.
 *
 * @param arch  Network architecture descriptor.
 * @return      Pointer to initialized network, or NULL on failure.
 *
 * Complexity: O(n_layers * max_layer_size^2) for weight initialization.
 * Weights initialized using He initialization (He et al. 2015):
 *   W ~ N(0, sqrt(2/n_in)) for ReLU
 *   W ~ N(0, sqrt(1/n_in)) for tanh/sigmoid (Xavier/Glorot)
 * Biases initialized to zero.
 */
nn_network_t *nn_network_create(const nn_architecture_t *arch);

/**
 * @brief Free a neural network and all associated memory.
 * @param net  Network to free.
 *
 * Complexity: O(total_parameters).
 */
void nn_network_free(nn_network_t *net);

/**
 * @brief Allocate and configure a soft sensor.
 *
 * @param name           Sensor name.
 * @param process_unit   Process unit description.
 * @param quality_var    Quality variable being estimated.
 * @param unit           Physical unit of measurement.
 * @param sensor_type    Static, dynamic, hybrid, ensemble, or adaptive.
 * @param net            Associated neural network.
 * @return               Pointer to initialized soft sensor, or NULL.
 *
 * Complexity: O(1) for allocation + configuration.
 */
soft_sensor_t *soft_sensor_create(const char *name, const char *process_unit,
                                   const char *quality_var, const char *unit,
                                   soft_sensor_type_t sensor_type,
                                   nn_network_t *net);

/**
 * @brief Free a soft sensor and its associated resources.
 * @param sensor  Soft sensor to free.
 */
void soft_sensor_free(soft_sensor_t *sensor);

/*===========================================================================
 * L2: Forward Propagation (Inference)
 *===========================================================================*/

/**
 * @brief Forward pass through a single layer.
 *
 * Computation: z = W * x + b, then a = activation(z)
 *
 * @param layer  Layer to compute forward pass on.
 * @param input  Input vector (length = layer->input_size).
 *
 * Complexity: O(output_size * input_size).
 *
 * Theorem: Single hidden layer with sigmoid activation can approximate
 *          any continuous function on compact set (Cybenko 1989, Hornik 1991).
 */
void nn_layer_forward(nn_layer_t *layer, const double *input);

/**
 * @brief Full forward pass through entire network.
 *
 * Computation sequence: input -> hidden_1 -> ... -> hidden_n -> output
 * Each layer: z_i = W_i * a_{i-1} + b_i, a_i = f_i(z_i)
 *
 * @param net    Network.
 * @param input  Input feature vector (length = net->layers[0]->output_size).
 * @param output Output buffer for predictions (length = last layer output_size).
 *
 * Complexity: O(total_parameters) per forward pass.
 *
 * Universal Approximation Theorem (Cybenko 1989, Hornik 1991):
 * Let sigma be a nonconstant, bounded, monotonically increasing continuous
 * function. For any eps > 0 and any continuous function f on [0,1]^n,
 * there exists N and weights W_i, b_i such that:
 * |F(x) - f(x)| < eps for all x in [0,1]^n,
 * where F(x) = sum(v_i * sigma(w_i'x + b_i)) is a single-hidden-layer network.
 */
void nn_network_forward(const nn_network_t *net, const double *input, double *output);

/**
 * @brief Soft sensor prediction (forward pass + denormalization if needed).
 *
 * @param sensor  Trained soft sensor.
 * @param input   Raw (possibly un-normalized) input features.
 * @param output  Raw prediction in original units.
 *
 * Complexity: O(forward_pass) + O(n_features) for normalization.
 */
void soft_sensor_predict(soft_sensor_t *sensor, const double *input, double *output);

/*===========================================================================
 * L2-L5: Training Configuration
 *===========================================================================*/

/**
 * @brief Configure network training parameters.
 *
 * @param net    Network to configure.
 * @param config Training configuration.
 *
 * Complexity: O(1).
 */
void nn_network_configure_training(nn_network_t *net, const nn_training_config_t *config);

/**
 * @brief Get default training configuration.
 *
 * Defaults: Adam optimizer, MSE loss, no regularization,
 *           learning_rate=0.001, batch_size=32, max_epochs=100.
 *
 * @param config  Output: populated with defaults.
 */
void nn_training_config_default(nn_training_config_t *config);

/*===========================================================================
 * L5: Training
 *===========================================================================*/

/**
 * @brief Train the neural network using mini-batch gradient descent.
 *
 * Implements the specified optimizer (SGD, Adam, RMSprop, etc.) with
 * backpropagation to compute gradients.
 *
 * @param net     Network to train.
 * @param dataset Dataset with training, validation splits.
 *
 * Complexity: O(epochs * n_samples * total_parameters).
 *
 * Backpropagation Theorem (Rumelhart, Hinton, Williams 1986):
 * The gradient of loss L w.r.t. any weight w_ij can be computed via:
 * dL/dw_ij = a_j * delta_i
 * where a_j is the activation of neuron j (input to synapse),
 * and delta_i is the backpropagated error at neuron i:
 * delta_i = activation'(z_i) * sum(delta_k * w_ki) for hidden layers,
 * delta_i = dL/da_i * activation'(z_i) for output layer.
 */
void nn_network_train(nn_network_t *net, nn_dataset_t *dataset);

/**
 * @brief Single training epoch (one pass through all mini-batches).
 *
 * @param net     Network.
 * @param dataset Dataset.
 * @param epoch   Current epoch number (for learning rate scheduling).
 *
 * @return Average training loss for this epoch.
 *
 * Complexity: O(n_samples * total_parameters).
 */
double nn_network_train_epoch(nn_network_t *net, nn_dataset_t *dataset, int epoch);

/**
 * @brief Compute loss for a single prediction.
 *
 * @param net        Network (for loss type).
 * @param prediction Model output.
 * @param target     Ground truth.
 * @param n          Output dimension.
 *
 * @return Loss value.
 */
double nn_compute_loss(const nn_network_t *net, const double *prediction,
                       const double *target, int n);

/*===========================================================================
 * L5: Backpropagation
 *===========================================================================*/

/**
 * @brief Backpropagate error through a single layer.
 *
 * Given the error signal from the next layer (delta_next), compute:
 * - Error signal for this layer: delta = activation'(z) * (W_next^T * delta_next)
 * - Gradient w.r.t. weights: dW = delta * a_prev^T
 * - Gradient w.r.t. biases: db = delta
 *
 * @param layer       Layer to backpropagate through.
 * @param delta_next  Error signal from next layer (or loss gradient for output).
 * @param input       Input to this layer (activations of previous layer).
 *
 * Reference: Rumelhart, Hinton, Williams, Nature (1986)
 */
void nn_layer_backward(nn_layer_t *layer, const double *delta_next,
                       const double *input);

/**
 * @brief Full backpropagation through entire network.
 *
 * Computes gradients dW, db for all layers from output back to first hidden.
 *
 * @param net     Network.
 * @param input   Original input (needed for input layer).
 * @param target  Ground truth value.
 *
 * Complexity: O(total_parameters).
 */
void nn_network_backward(nn_network_t *net, const double *input, const double *target);

/*===========================================================================
 * L5: Optimizer Steps
 *===========================================================================*/

/**
 * @brief Apply gradient update using configured optimizer.
 *
 * Implements: SGD, Momentum, Nesterov, AdaGrad, RMSprop, Adam, AdaMax, Nadam.
 *
 * @param net     Network with computed gradients.
 * @param epoch   Current epoch (for schedule).
 *
 * Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
 *            Ruder, "An overview of gradient descent..." (2016)
 */
void nn_network_update_weights(nn_network_t *net, int epoch);

/*===========================================================================
 * L1-L3: Dataset Operations
 *===========================================================================*/

/**
 * @brief Create a dataset from raw arrays.
 *
 * @param X            Input features (row-major: n_samples x n_features).
 * @param Y            Target values (row-major: n_samples x n_outputs).
 * @param n_samples    Number of data samples.
 * @param n_features   Number of input features.
 * @param n_outputs    Number of output targets.
 * @param train_ratio  Fraction for training (e.g., 0.7).
 * @param val_ratio    Fraction for validation (e.g., 0.15, test gets rest).
 *
 * @return Allocated dataset, or NULL on failure.
 *
 * Split: [0, train_end] train, (train_end, val_end] val, (val_end, n) test.
 */
nn_dataset_t *nn_dataset_create(const double *X, const double *Y,
                                 int n_samples, int n_features, int n_outputs,
                                 double train_ratio, double val_ratio);

/**
 * @brief Free a dataset.
 * @param dataset Dataset to free.
 */
void nn_dataset_free(nn_dataset_t *dataset);

/**
 * @brief Normalize dataset features using specified method.
 *
 * @param dataset  Dataset to normalize.
 * @param x_type   Normalization method for inputs.
 * @param y_type   Normalization method for outputs.
 */
void nn_dataset_normalize(nn_dataset_t *dataset,
                          nn_normalization_type_t x_type,
                          nn_normalization_type_t y_type);

/*===========================================================================
 * L4-L6: Model Validation & Performance
 *===========================================================================*/

/**
 * @brief Evaluate the trained network on a data subset.
 *
 * @param net      Trained network.
 * @param dataset  Dataset with defined splits.
 * @param start    Sample index (inclusive).
 * @param end      Sample index (exclusive).
 * @param metrics  Output: computed regression metrics.
 *
 * Complexity: O((end-start) * total_parameters).
 */
void nn_network_evaluate(const nn_network_t *net, const nn_dataset_t *dataset,
                          int start, int end, nn_regression_metrics_t *metrics);

/**
 * @brief Compute coefficient of determination R².
 *
 * R² = 1 - SS_res / SS_tot
 * where SS_res = sum((y_i - y_pred_i)^2), SS_tot = sum((y_i - mean(y))^2)
 *
 * @param y_true  Ground truth values.
 * @param y_pred  Predicted values.
 * @param n       Number of samples.
 * @return R² in range (-inf, 1].
 */
double nn_compute_r_squared(const double *y_true, const double *y_pred, int n);

/*===========================================================================
 * L5: Learning Rate Scheduling
 *===========================================================================*/

/**
 * @brief Compute learning rate for a given epoch using schedule.
 *
 * Supports: constant, step decay, exponential decay, inverse time,
 *           cosine annealing (Loshchilov & Hutter 2017).
 *
 * @param base_lr      Initial learning rate.
 * @param epoch        Current epoch.
 * @param schedule     Schedule type.
 * @param decay_rate   Decay parameter.
 * @param decay_steps  Steps for step decay.
 * @return             Current learning rate.
 */
double nn_learning_rate_schedule(double base_lr, int epoch,
                                  nn_lr_schedule_type_t schedule,
                                  double decay_rate, int decay_steps);

/*===========================================================================
 * L8: Advanced Features
 *===========================================================================*/

/**
 * @brief Create an ensemble of neural networks.
 *
 * Each network is trained on a bootstrap sample of the data (bagging).
 *
 * @param num_members  Number of ensemble members.
 * @param arch         Architecture for each member.
 * @return             Allocated ensemble, or NULL.
 *
 * Reference: Breiman, "Bagging Predictors" (1996)
 */
nn_ensemble_t *nn_ensemble_create(int num_members, const nn_architecture_t *arch);

/**
 * @brief Make prediction using ensemble (weighted average).
 *
 * @param ensemble  Trained ensemble.
 * @param input     Input features.
 * @param output    Ensemble prediction.
 */
void nn_ensemble_predict(nn_ensemble_t *ensemble, const double *input, double *output);

/**
 * @brief Free an ensemble.
 * @param ensemble  Ensemble to free.
 */
void nn_ensemble_free(nn_ensemble_t *ensemble);

/*===========================================================================
 * L7-L8: Industrial Deployment
 *===========================================================================*/

/**
 * @brief Initialize an industrial soft sensor deployment.
 *
 * @param app           Application state to initialize.
 * @param sensor        Trained soft sensor.
 * @param plant         Plant location.
 * @param dcs_system    DCS system name.
 * @param historian     Historian database name.
 */
void nn_industrial_sensor_init(nn_industrial_soft_sensor_t *app,
                                soft_sensor_t *sensor,
                                const char *plant,
                                const char *dcs_system,
                                const char *historian);

/**
 * @brief Monitor soft sensor performance drift.
 *
 * Computes moving average of prediction errors and raises flag
 * if performance degrades below threshold.
 *
 * @param sensor   Deployed soft sensor.
 * @param actual   Actual measured value (lab sample for calibration).
 * @param predicted Predicted value from soft sensor.
 *
 * @return 1 if retraining is needed, 0 otherwise.
 *
 * Reference: Kaneko & Funatsu, "Adaptive Soft Sensor..." AIChE J. (2014)
 */
int soft_sensor_monitor_performance(soft_sensor_t *sensor, double actual,
                                     double predicted);

/*===========================================================================
 * L5: Model Persistence
 *===========================================================================*/

/**
 * @brief Serialize network weights to file (ASCII format).
 *
 * Format: num_layers layer_sizes... weights..., biases...
 *
 * @param net       Network to save.
 * @param filename  Output file path.
 * @return 0 on success, -1 on failure.
 */
int nn_network_save(const nn_network_t *net, const char *filename);

/**
 * @brief Load network from serialized file.
 *
 * @param filename  Input file path.
 * @return          Loaded network, or NULL on failure.
 */
nn_network_t *nn_network_load(const char *filename);

/*===========================================================================
 * L8: Bayesian Approximation (MC Dropout)
 *===========================================================================*/

/**
 * @brief Create Monte Carlo dropout uncertainty estimator.
 *
 * @param net         Trained network.
 * @param num_samples Number of stochastic forward passes.
 * @return            MC dropout state, or NULL.
 *
 * Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016)
 */
nn_mc_dropout_t *nn_mc_dropout_create(nn_network_t *net, int num_samples);

/**
 * @brief Estimate prediction with uncertainty using MC Dropout.
 *
 * Performs multiple forward passes with dropout active to estimate
 * mean and variance of predictions.
 *
 * @param mc      MC dropout state.
 * @param input   Input features.
 * @param output  Mean prediction.
 */
void nn_mc_dropout_predict(nn_mc_dropout_t *mc, const double *input, double *output);

/**
 * @brief Free MC dropout state.
 * @param mc  State to free.
 */
void nn_mc_dropout_free(nn_mc_dropout_t *mc);

/*===========================================================================
 * L3: Helper Functions
 *===========================================================================*/

/**
 * @brief He (Kaiming) weight initialization for ReLU.
 * W ~ N(0, sqrt(2/n_in))
 *
 * @param data   Weight matrix data.
 * @param rows   Output size.
 * @param cols   Input size.
 *
 * Reference: He, Zhang, Ren, Sun, "Delving Deep into Rectifiers" (2015)
 */
void nn_he_initialize(double *data, int rows, int cols);

/**
 * @brief Xavier/Glorot weight initialization for tanh/sigmoid.
 * W ~ U(-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out)))
 *
 * @param data   Weight matrix data.
 * @param rows   Output size.
 * @param cols   Input size.
 *
 * Reference: Glorot & Bengio, "Understanding the difficulty..." (2010)
 */
void nn_xavier_initialize(double *data, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif /* NN_SOFT_SENSOR_H */
