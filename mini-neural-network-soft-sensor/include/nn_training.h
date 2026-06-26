/**
 * @file nn_training.h
 * @brief Neural network training algorithms and optimization methods.
 *
 * Level: L5 Algorithms/Methods
 * Reference: Rumelhart, Hinton, Williams, "Learning representations by
 *            back-propagating errors," Nature (1986)
 *            Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
 *            Ruder, "An overview of gradient descent optimization algorithms" (2016)
 *            Duchi et al., "Adaptive Subgradient Methods..." JMLR (2011)
 *            Tieleman & Hinton, "Lecture 6.5 RMSprop" COURSERA (2012)
 *            Nesterov, "A method of solving a convex programming problem..." (1983)
 */

#ifndef NN_TRAINING_H
#define NN_TRAINING_H

#include "nn_sensor_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
 * L5: Gradient Descent Optimizers
 *
 * General form: theta_{t+1} = theta_t - lr * m_t / (sqrt(v_t) + eps)
 * where m_t is the first moment estimate and v_t is the second moment.
 *===========================================================================*/

/**
 * @brief SGD (Stochastic Gradient Descent) update step.
 *
 * The simplest first-order optimization method.
 * theta_{t+1} = theta_t - learning_rate * gradient_t
 *
 * Complexity: O(n_parameters) per step.
 *
 * Theorem (Robbins-Monro 1951): Under certain conditions, SGD converges
 * to a local minimum for convex optimization problems.
 * Convergence rate: O(1/sqrt(t)) for non-convex, O(1/t) for strongly convex.
 */
void nn_optimizer_sgd_step(nn_matrix_t *W, nn_matrix_t *dW,
                            nn_vector_t *b, nn_vector_t *db,
                            double learning_rate);

/**
 * @brief SGD with Momentum update step (Polyak 1964).
 *
 * v_t = momentum * v_{t-1} - learning_rate * gradient_t
 * theta_{t+1} = theta_t + v_t
 *
 * Momentum helps accelerate convergence and dampen oscillations
 * in ravines of the loss surface.
 *
 * @param W,dW      Weight matrix and its gradient.
 * @param b,db      Bias vector and its gradient.
 * @param lr        Learning rate.
 * @param momentum  Momentum coefficient (typically 0.9).
 * @param v_W       Velocity for weights (updated in-place).
 * @param v_b       Velocity for biases (updated in-place).
 *
 * Complexity: O(n_parameters).
 */
void nn_optimizer_momentum_step(nn_matrix_t *W, nn_matrix_t *dW,
                                 nn_vector_t *b, nn_vector_t *db,
                                 double learning_rate, double momentum,
                                 nn_matrix_t *v_W, nn_vector_t *v_b);

/**
 * @brief Nesterov Accelerated Gradient step (Nesterov 1983).
 *
 * Look-ahead version of momentum:
 * theta_lookahead = theta_t + momentum * v_t
 * v_{t+1} = momentum * v_t - lr * grad(lookahead)
 * theta_{t+1} = theta_t + v_{t+1}
 *
 * Nesterov provides better convergence guarantees:
 * O(1/t^2) for convex problems vs O(1/t) for standard momentum.
 *
 * @param W,dW,v_W    Weights, gradient, and velocity.
 * @param b,db,v_b    Biases, gradient, and velocity.
 * @param lr          Learning rate.
 * @param momentum    Momentum coefficient.
 */
void nn_optimizer_nesterov_step(nn_matrix_t *W, nn_matrix_t *dW,
                                 nn_vector_t *b, nn_vector_t *db,
                                 double learning_rate, double momentum,
                                 nn_matrix_t *v_W, nn_vector_t *v_b);

/**
 * @brief AdaGrad optimizer step (Duchi et al. 2011).
 *
 * Accumulates squared gradients and adapts learning rate per-parameter:
 * G_t = G_{t-1} + grad_t^2
 * theta_{t+1} = theta_t - lr * grad_t / sqrt(G_t + eps)
 *
 * Well-suited for sparse data, but learning rate decays to zero
 * with accumulation of squared gradients.
 *
 * Complexity: O(n_parameters).
 */
void nn_optimizer_adagrad_step(nn_matrix_t *W, nn_matrix_t *dW,
                                nn_vector_t *b, nn_vector_t *db,
                                double learning_rate, double epsilon,
                                nn_matrix_t *G_W, nn_vector_t *G_b);

/**
 * @brief RMSprop optimizer step (Tieleman & Hinton 2012).
 *
 * Uses moving average of squared gradients instead of sum:
 * E[g^2]_t = rho * E[g^2]_{t-1} + (1-rho) * g_t^2
 * theta_{t+1} = theta_t - lr * g_t / sqrt(E[g^2]_t + eps)
 *
 * Fixes AdaGrad's diminishing learning rate problem.
 * Default rho = 0.9, epsilon = 1e-8.
 */
void nn_optimizer_rmsprop_step(nn_matrix_t *W, nn_matrix_t *dW,
                                nn_vector_t *b, nn_vector_t *db,
                                double learning_rate, double rho, double epsilon,
                                nn_matrix_t *E_W, nn_vector_t *E_b);

/**
 * @brief Adam optimizer step (Kingma & Ba 2014).
 *
 * Combines momentum (first moment) and RMSprop (second moment)
 * with bias correction:
 *
 * m_t = beta1 * m_{t-1} + (1-beta1) * g_t
 * v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
 * m_hat = m_t / (1 - beta1^t)
 * v_hat = v_t / (1 - beta2^t)
 * theta_{t+1} = theta_t - lr * m_hat / (sqrt(v_hat) + eps)
 *
 * Currently the most widely used optimizer for deep learning.
 * Default: beta1=0.9, beta2=0.999, epsilon=1e-8.
 *
 * @return 0 on success, -1 on parameter error.
 *
 * Complexity: O(n_parameters).
 */
int nn_optimizer_adam_step(nn_matrix_t *W, nn_matrix_t *dW,
                            nn_vector_t *b, nn_vector_t *db,
                            double learning_rate, double beta1, double beta2,
                            double epsilon, int t,
                            nn_matrix_t *m_W, nn_matrix_t *v_W,
                            nn_vector_t *m_b, nn_vector_t *v_b);

/**
 * @brief AdaMax optimizer step (Kingma & Ba 2014).
 *
 * Variant of Adam using infinity norm:
 * u_t = max(beta2 * u_{t-1}, |g_t|)
 * theta_{t+1} = theta_t - lr * m_hat / u_t
 *
 * More stable than Adam for certain parameter ranges.
 */
void nn_optimizer_adamax_step(nn_matrix_t *W, nn_matrix_t *dW,
                               nn_vector_t *b, nn_vector_t *db,
                               double learning_rate, double beta1, double beta2,
                               int t,
                               nn_matrix_t *m_W, nn_matrix_t *u_W,
                               nn_vector_t *m_b, nn_vector_t *u_b);

/**
 * @brief Nadam (Nesterov-accelerated Adam) step (Dozat 2016).
 *
 * Incorporates Nesterov momentum into Adam:
 * m_nesterov = beta1 * m_hat + (1-beta1) * g_t / (1 - beta1^t)
 *
 * Often converges slightly faster than standard Adam.
 */
int nn_optimizer_nadam_step(nn_matrix_t *W, nn_matrix_t *dW,
                             nn_vector_t *b, nn_vector_t *db,
                             double learning_rate, double beta1, double beta2,
                             double epsilon, int t,
                             nn_matrix_t *m_W, nn_matrix_t *v_W,
                             nn_vector_t *m_b, nn_vector_t *v_b);

/*===========================================================================
 * L5: Loss Functions with Gradients
 *===========================================================================*/

/**
 * @brief MSE loss and gradient.
 *
 * L = (1/n) * sum((y_pred_i - y_true_i)^2)
 * dL/d(y_pred_i) = (2/n) * (y_pred_i - y_true_i)
 *
 * @param y_pred    Predictions (length n).
 * @param y_true    Targets (length n).
 * @param n         Output dimension.
 * @param gradient  Output: gradient of loss w.r.t. predictions.
 * @return          Loss value.
 */
double nn_loss_mse(const double *y_pred, const double *y_true, int n, double *gradient);

/**
 * @brief MAE loss and gradient.
 *
 * L = (1/n) * sum(|y_pred_i - y_true_i|)
 * dL/d(y_pred_i) = sign(y_pred_i - y_true_i) / n
 *
 * More robust to outliers than MSE.
 */
double nn_loss_mae(const double *y_pred, const double *y_true, int n, double *gradient);

/**
 * @brief Huber loss and gradient (Huber 1964).
 *
 * For each element i:
 * e_i = y_pred_i - y_true_i
 * if |e_i| <= delta:
 *   L_i = 0.5 * e_i^2, dL_i = e_i
 * else:
 *   L_i = delta * (|e_i| - 0.5*delta), dL_i = delta * sign(e_i)
 *
 * Smooth L1 - differentiable everywhere, robust to outliers.
 *
 * @param delta  Threshold parameter.
 */
double nn_loss_huber(const double *y_pred, const double *y_true, int n,
                     double delta, double *gradient);

/**
 * @brief MAPE loss and gradient.
 *
 * L = (100/n) * sum(|(y_true_i - y_pred_i) / y_true_i|)
 *
 * Interpretable as percentage error. Handle y_true_i near zero carefully.
 */
double nn_loss_mape(const double *y_pred, const double *y_true, int n, double *gradient);

/*===========================================================================
 * L5: Regularization
 *===========================================================================*/

/**
 * @brief Compute L1 regularization loss and gradient for weights.
 *
 * L = lambda * sum(|w_ij|)
 * dL/dw_ij = lambda * sign(w_ij)
 *
 * Induces sparsity - drives many weights to exactly zero.
 * Based on LASSO (Tibshirani 1996).
 */
double nn_regularization_l1(const nn_matrix_t *W, double lambda, nn_matrix_t *dW_reg);

/**
 * @brief Compute L2 regularization loss and gradient for weights.
 *
 * L = (lambda/2) * sum(w_ij^2)
 * dL/dw_ij = lambda * w_ij
 *
 * Weight decay - penalizes large weights, smooths the model.
 * Based on Ridge regression (Hoerl & Kennard 1970).
 */
double nn_regularization_l2(const nn_matrix_t *W, double lambda, nn_matrix_t *dW_reg);

/**
 * @brief Compute Elastic Net regularization (Zou & Hastie 2005).
 *
 * L = l1_lambda * sum(|w|) + (l2_lambda/2) * sum(w^2)
 *
 * Combines L1 sparsity with L2 stability.
 */
double nn_regularization_elastic_net(const nn_matrix_t *W, double l1_lambda,
                                      double l2_lambda, nn_matrix_t *dW_reg);

/**
 * @brief Apply dropout mask to layer activations during training.
 *
 * Randomly sets a fraction (`rate`) of activations to zero,
 * and scales the rest by 1/(1-rate) to maintain expected sum.
 *
 * @param activations  Layer activations (modified in-place).
 * @param size         Number of neurons.
 * @param rate         Dropout probability (e.g., 0.5).
 * @param seed         Random seed for reproducibility.
 *
 * Reference: Srivastava et al., "Dropout..." JMLR (2014)
 */
void nn_dropout_apply(double *activations, int size, double rate, unsigned int *seed);

/*===========================================================================
 * L5: Batch Normalization
 *===========================================================================*/

/**
 * @brief Apply batch normalization to a mini-batch.
 *
 * mu = (1/m) * sum(x_i)
 * sigma^2 = (1/m) * sum((x_i - mu)^2)
 * x_hat_i = (x_i - mu) / sqrt(sigma^2 + eps)
 * y_i = gamma * x_hat_i + beta
 *
 * @param batch      Input/output batch data (row-major: m x n_features).
 * @param m          Batch size.
 * @param n_features Number of features per sample.
 * @param gamma      Scale parameter (length n_features).
 * @param beta       Shift parameter (length n_features).
 * @param eps        Small constant for numerical stability.
 *
 * Reference: Ioffe & Szegedy, "Batch Normalization..." ICML (2015)
 */
void nn_batch_normalize(double *batch, int m, int n_features,
                         double *gamma, double *beta, double eps);

/*===========================================================================
 * L5: Mini-batch Sampling
 *===========================================================================*/

/**
 * @brief Sample a random mini-batch from dataset.
 *
 * @param dataset      Source dataset.
 * @param batch_indices Output: indices of samples in this batch.
 * @param batch_size   Number of samples to draw.
 * @param seed         Random seed for reproducibility.
 */
void nn_sample_mini_batch(const nn_dataset_t *dataset, int *batch_indices,
                           int batch_size, unsigned int *seed);

/**
 * @brief Simple pseudo-random number generator (xorshift32).
 *
 * @param state  Current state (modified in-place).
 * @return       Pseudo-random 32-bit integer.
 *
 * Reference: Marsaglia, "Xorshift RNGs" (2003)
 */
unsigned int nn_random_xorshift32(unsigned int *state);

/**
 * @brief Draw sample from standard normal distribution N(0,1).
 * Uses Box-Muller transform.
 *
 * @param state  RNG state.
 * @return       N(0,1) sample.
 *
 * Reference: Box & Muller, "A Note on the Generation of Random Normal Deviates" (1958)
 */
double nn_random_normal(unsigned int *state);

/**
 * @brief Draw sample from uniform distribution U(0,1).
 * @param state  RNG state.
 * @return       U(0,1) sample.
 */
double nn_random_uniform(unsigned int *state);

#ifdef __cplusplus
}
#endif

#endif /* NN_TRAINING_H */
