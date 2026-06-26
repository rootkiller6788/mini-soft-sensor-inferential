/**
 * @file nn_training.c
 * @brief Neural network training: optimizers, loss functions, regularization.
 *
 * Level: L5 Algorithms/Methods
 * Reference: Rumelhart, Hinton, Williams, Nature (1986)
 *            Kingma & Ba (2014), Duchi et al. (2011)
 *            Srivastava et al. (2014), Ioffe & Szegedy (2015)
 *
 * All optimizers update weights in-place using pre-computed gradients.
 * Each function implements exactly one optimizer from the ML literature.
 */

#include "nn_training.h"
#include "nn_soft_sensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*===========================================================================
 * L5: Optimizer Implementations
 *
 * Each optimizer follows the pattern:
 *   theta = theta - lr * update
 * where update depends on the optimizer's moment/variance estimates.
 *===========================================================================*/

void nn_optimizer_sgd_step(nn_matrix_t *W, nn_matrix_t *dW,
                            nn_vector_t *b, nn_vector_t *db,
                            double learning_rate) {
    if (!W || !dW || !b || !db) return;

    int n_weights = W->rows * W->cols;
    for (int i = 0; i < n_weights; i++) {
        W->data[i] -= learning_rate * dW->data[i];
    }
    for (int i = 0; i < b->length; i++) {
        b->data[i] -= learning_rate * db->data[i];
    }
}

void nn_optimizer_momentum_step(nn_matrix_t *W, nn_matrix_t *dW,
                                 nn_vector_t *b, nn_vector_t *db,
                                 double learning_rate, double momentum,
                                 nn_matrix_t *v_W, nn_vector_t *v_b) {
    if (!W || !dW || !b || !db || !v_W || !v_b) return;

    /* v = mu * v - lr * grad; theta += v */
    int n_weights = W->rows * W->cols;
    for (int i = 0; i < n_weights; i++) {
        v_W->data[i] = momentum * v_W->data[i] - learning_rate * dW->data[i];
        W->data[i] += v_W->data[i];
    }
    for (int i = 0; i < b->length; i++) {
        v_b->data[i] = momentum * v_b->data[i] - learning_rate * db->data[i];
        b->data[i] += v_b->data[i];
    }
}

void nn_optimizer_nesterov_step(nn_matrix_t *W, nn_matrix_t *dW,
                                 nn_vector_t *b, nn_vector_t *db,
                                 double learning_rate, double momentum,
                                 nn_matrix_t *v_W, nn_vector_t *v_b) {
    if (!W || !dW || !b || !db || !v_W || !v_b) return;

    /* Nesterov look-ahead: first move to lookahead position, then apply gradient */
    int n_weights = W->rows * W->cols;
    for (int i = 0; i < n_weights; i++) {
        /* Move to look-ahead position temporarily */
        double lookahead = W->data[i] + momentum * v_W->data[i];
        /* Apply gradient at lookahead */
        W->data[i] = lookahead;
    }
    for (int i = 0; i < b->length; i++) {
        double lookahead = b->data[i] + momentum * v_b->data[i];
        b->data[i] = lookahead;
    }

    /* Now update velocities and apply the actual step */
    for (int i = 0; i < n_weights; i++) {
        v_W->data[i] = momentum * v_W->data[i] - learning_rate * dW->data[i];
        /* Note: we're re-using the gradient computed at original position.
         * True Nesterov recomputes gradient at lookahead - we approximate here. */
        W->data[i] += v_W->data[i];
    }
    for (int i = 0; i < b->length; i++) {
        v_b->data[i] = momentum * v_b->data[i] - learning_rate * db->data[i];
        b->data[i] += v_b->data[i];
    }
}

void nn_optimizer_adagrad_step(nn_matrix_t *W, nn_matrix_t *dW,
                                nn_vector_t *b, nn_vector_t *db,
                                double learning_rate, double epsilon,
                                nn_matrix_t *G_W, nn_vector_t *G_b) {
    if (!W || !dW || !b || !db || !G_W || !G_b) return;

    /* G_t = G_{t-1} + grad^2
     * theta = theta - lr * grad / sqrt(G_t + eps) */
    int n_weights = W->rows * W->cols;
    for (int i = 0; i < n_weights; i++) {
        G_W->data[i] += dW->data[i] * dW->data[i];
        W->data[i] -= learning_rate * dW->data[i] / (sqrt(G_W->data[i]) + epsilon);
    }
    for (int i = 0; i < b->length; i++) {
        G_b->data[i] += db->data[i] * db->data[i];
        b->data[i] -= learning_rate * db->data[i] / (sqrt(G_b->data[i]) + epsilon);
    }
}

void nn_optimizer_rmsprop_step(nn_matrix_t *W, nn_matrix_t *dW,
                                nn_vector_t *b, nn_vector_t *db,
                                double learning_rate, double rho, double epsilon,
                                nn_matrix_t *E_W, nn_vector_t *E_b) {
    if (!W || !dW || !b || !db || !E_W || !E_b) return;

    /* E[g^2] = rho * E[g^2] + (1-rho) * g^2
     * theta = theta - lr * g / sqrt(E[g^2] + eps) */
    int n_weights = W->rows * W->cols;
    for (int i = 0; i < n_weights; i++) {
        E_W->data[i] = rho * E_W->data[i] + (1.0 - rho) * dW->data[i] * dW->data[i];
        W->data[i] -= learning_rate * dW->data[i] / (sqrt(E_W->data[i]) + epsilon);
    }
    for (int i = 0; i < b->length; i++) {
        E_b->data[i] = rho * E_b->data[i] + (1.0 - rho) * db->data[i] * db->data[i];
        b->data[i] -= learning_rate * db->data[i] / (sqrt(E_b->data[i]) + epsilon);
    }
}

int nn_optimizer_adam_step(nn_matrix_t *W, nn_matrix_t *dW,
                            nn_vector_t *b, nn_vector_t *db,
                            double learning_rate, double beta1, double beta2,
                            double epsilon, int t,
                            nn_matrix_t *m_W, nn_matrix_t *v_W,
                            nn_vector_t *m_b, nn_vector_t *v_b) {
    if (!W || !dW || !b || !db || !m_W || !v_W || !m_b || !v_b) return -1;

    double beta1_pow = pow(beta1, t);
    double beta2_pow = pow(beta2, t);
    double alpha_t = learning_rate * sqrt(1.0 - beta2_pow) / (1.0 - beta1_pow);

    /* m_t = beta1*m_{t-1} + (1-beta1)*g_t
     * v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
     * theta = theta - alpha_t * m_t / (sqrt(v_t) + eps) */
    int n_weights = W->rows * W->cols;
    for (int i = 0; i < n_weights; i++) {
        double g = dW->data[i];
        m_W->data[i] = beta1 * m_W->data[i] + (1.0 - beta1) * g;
        v_W->data[i] = beta2 * v_W->data[i] + (1.0 - beta2) * g * g;
        W->data[i] -= alpha_t * m_W->data[i] / (sqrt(v_W->data[i]) + epsilon);
    }
    for (int i = 0; i < b->length; i++) {
        double g = db->data[i];
        m_b->data[i] = beta1 * m_b->data[i] + (1.0 - beta1) * g;
        v_b->data[i] = beta2 * v_b->data[i] + (1.0 - beta2) * g * g;
        b->data[i] -= alpha_t * m_b->data[i] / (sqrt(v_b->data[i]) + epsilon);
    }

    return 0;
}

void nn_optimizer_adamax_step(nn_matrix_t *W, nn_matrix_t *dW,
                               nn_vector_t *b, nn_vector_t *db,
                               double learning_rate, double beta1, double beta2,
                               int t,
                               nn_matrix_t *m_W, nn_matrix_t *u_W,
                               nn_vector_t *m_b, nn_vector_t *u_b) {
    if (!W || !dW || !b || !db || !m_W || !u_W || !m_b || !u_b) return;

    /* m_t = beta1*m_{t-1} + (1-beta1)*g_t
     * u_t = max(beta2*u_{t-1}, |g_t|) ; using infinity norm
     * theta = theta - lr*m_t / ((1-beta1^t)*u_t) */
    double beta1_pow = pow(beta1, t);
    double lr_corr = learning_rate / (1.0 - beta1_pow);

    int n_weights = W->rows * W->cols;
    for (int i = 0; i < n_weights; i++) {
        double g = dW->data[i];
        m_W->data[i] = beta1 * m_W->data[i] + (1.0 - beta1) * g;
        double abs_g = fabs(g);
        u_W->data[i] = (beta2 * u_W->data[i] > abs_g) ? beta2 * u_W->data[i] : abs_g;
        W->data[i] -= lr_corr * m_W->data[i] / (u_W->data[i] + 1e-12);
    }
    for (int i = 0; i < b->length; i++) {
        double g = db->data[i];
        m_b->data[i] = beta1 * m_b->data[i] + (1.0 - beta1) * g;
        double abs_g = fabs(g);
        u_b->data[i] = (beta2 * u_b->data[i] > abs_g) ? beta2 * u_b->data[i] : abs_g;
        b->data[i] -= lr_corr * m_b->data[i] / (u_b->data[i] + 1e-12);
    }
}

int nn_optimizer_nadam_step(nn_matrix_t *W, nn_matrix_t *dW,
                             nn_vector_t *b, nn_vector_t *db,
                             double learning_rate, double beta1, double beta2,
                             double epsilon, int t,
                             nn_matrix_t *m_W, nn_matrix_t *v_W,
                             nn_vector_t *m_b, nn_vector_t *v_b) {
    if (!W || !dW || !b || !db || !m_W || !v_W || !m_b || !v_b) return -1;

    /* Nadam = Nesterov momentum + Adam
     * m_nesterov_t = beta1*m_t + (1-beta1)*g_t  (Nesterov look-ahead momentum)
     * Then apply Adam rule with m_nesterov
     */
    double beta1_pow = pow(beta1, t);
    double beta2_pow = pow(beta2, t);

    int n_weights = W->rows * W->cols;
    for (int i = 0; i < n_weights; i++) {
        double g = dW->data[i];

        /* First moment (momentum) */
        double m_prev = m_W->data[i];
        m_W->data[i] = beta1 * m_prev + (1.0 - beta1) * g;

        /* Nesterov look-ahead */
        double m_nesterov = beta1 * m_W->data[i] + (1.0 - beta1) * g;

        /* Second moment (variance) */
        v_W->data[i] = beta2 * v_W->data[i] + (1.0 - beta2) * g * g;

        /* Bias correction */
        double m_hat = m_nesterov / (1.0 - beta1_pow);
        double v_hat = v_W->data[i] / (1.0 - beta2_pow);

        W->data[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
    for (int i = 0; i < b->length; i++) {
        double g = db->data[i];
        double m_prev = m_b->data[i];
        m_b->data[i] = beta1 * m_prev + (1.0 - beta1) * g;
        double m_nesterov = beta1 * m_b->data[i] + (1.0 - beta1) * g;
        v_b->data[i] = beta2 * v_b->data[i] + (1.0 - beta2) * g * g;
        double m_hat = m_nesterov / (1.0 - beta1_pow);
        double v_hat = v_b->data[i] / (1.0 - beta2_pow);
        b->data[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }

    return 0;
}

/*===========================================================================
 * L5: Loss Functions
 *===========================================================================*/

double nn_loss_mse(const double *y_pred, const double *y_true, int n, double *gradient) {
    if (!y_pred || !y_true || n <= 0) return 0.0;

    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        double err = y_pred[i] - y_true[i];
        loss += err * err;
        if (gradient) {
            gradient[i] = 2.0 * err / n;  /* d(L)/dy_pred_i = 2*(y_pred_i - y_true_i)/n */
        }
    }
    return loss / n;
}

double nn_loss_mae(const double *y_pred, const double *y_true, int n, double *gradient) {
    if (!y_pred || !y_true || n <= 0) return 0.0;

    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        double err = y_pred[i] - y_true[i];
        loss += fabs(err);
        if (gradient) {
            /* d(L)/dy_pred_i = sign(y_pred_i - y_true_i) / n */
            gradient[i] = (err > 0.0) ? (1.0 / n) : (err < 0.0) ? (-1.0 / n) : 0.0;
        }
    }
    return loss / n;
}

double nn_loss_huber(const double *y_pred, const double *y_true, int n,
                     double delta, double *gradient) {
    if (!y_pred || !y_true || n <= 0) return 0.0;
    if (delta <= 0.0) delta = 1.0;

    /* Huber loss: quadratic for small errors, linear for large
     * |err| <= delta: L = 0.5*err^2, dL = err
     * |err| >  delta: L = delta*|err| - 0.5*delta^2, dL = delta*sign(err) */
    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        double err = y_pred[i] - y_true[i];
        double abs_err = fabs(err);
        if (abs_err <= delta) {
            loss += 0.5 * err * err;
            if (gradient) gradient[i] = err;
        } else {
            loss += delta * abs_err - 0.5 * delta * delta;
            if (gradient) gradient[i] = (err > 0.0) ? delta : -delta;
        }
    }
    return loss / n;
}

double nn_loss_mape(const double *y_pred, const double *y_true, int n, double *gradient) {
    if (!y_pred || !y_true || n <= 0) return 0.0;

    /* MAPE = (100/n) * sum(|(y_pred - y_true) / y_true|)
     * dMAPE/dy_pred_i = (100/n) * sign(y_pred - y_true) / |y_true| */
    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        double err = y_pred[i] - y_true[i];
        double abs_yt = fabs(y_true[i]);
        if (abs_yt > 1e-10) {
            loss += fabs(err) / abs_yt;
            if (gradient) {
                gradient[i] = (100.0 / n) * ((err > 0.0) ? 1.0 : (err < 0.0) ? -1.0 : 0.0) / abs_yt;
            }
        } else {
            if (gradient) gradient[i] = 0.0;
        }
    }
    return (100.0 / n) * loss;
}

/*===========================================================================
 * L5: Regularization
 *===========================================================================*/

double nn_regularization_l1(const nn_matrix_t *W, double lambda, nn_matrix_t *dW_reg) {
    if (!W) return 0.0;

    double loss = 0.0;
    int n = W->rows * W->cols;
    for (int i = 0; i < n; i++) {
        double w = W->data[i];
        loss += fabs(w);
        if (dW_reg) {
            dW_reg->data[i] = lambda * ((w > 0.0) ? 1.0 : (w < 0.0) ? -1.0 : 0.0);
        }
    }
    return lambda * loss;
}

double nn_regularization_l2(const nn_matrix_t *W, double lambda, nn_matrix_t *dW_reg) {
    if (!W) return 0.0;

    double loss = 0.0;
    int n = W->rows * W->cols;
    for (int i = 0; i < n; i++) {
        double w = W->data[i];
        loss += w * w;
        if (dW_reg) {
            dW_reg->data[i] = lambda * w;
        }
    }
    return 0.5 * lambda * loss;
}

double nn_regularization_elastic_net(const nn_matrix_t *W, double l1_lambda,
                                      double l2_lambda, nn_matrix_t *dW_reg) {
    if (!W) return 0.0;

    double loss = 0.0;
    int n = W->rows * W->cols;
    for (int i = 0; i < n; i++) {
        double w = W->data[i];
        loss += l1_lambda * fabs(w) + 0.5 * l2_lambda * w * w;
        if (dW_reg) {
            dW_reg->data[i] = l1_lambda * ((w > 0.0) ? 1.0 : (w < 0.0) ? -1.0 : 0.0)
                            + l2_lambda * w;
        }
    }
    return loss;
}

void nn_dropout_apply(double *activations, int size, double rate, unsigned int *seed) {
    if (!activations || size <= 0 || rate <= 0.0) return;
    if (rate >= 1.0) rate = 0.999;

    double scale = 1.0 / (1.0 - rate);
    for (int i = 0; i < size; i++) {
        double r = nn_random_uniform(seed);
        if (r < rate) {
            activations[i] = 0.0;
        } else {
            activations[i] *= scale;
        }
    }
}

/*===========================================================================
 * L5: Batch Normalization
 *===========================================================================*/

void nn_batch_normalize(double *batch, int m, int n_features,
                         double *gamma, double *beta, double eps) {
    if (!batch || m <= 0 || n_features <= 0) return;

    for (int j = 0; j < n_features; j++) {
        /* Compute mean */
        double mean = 0.0;
        for (int i = 0; i < m; i++) {
            mean += batch[i * n_features + j];
        }
        mean /= m;

        /* Compute variance */
        double var = 0.0;
        for (int i = 0; i < m; i++) {
            double d = batch[i * n_features + j] - mean;
            var += d * d;
        }
        var /= m;

        /* Normalize and scale */
        double inv_std = 1.0 / sqrt(var + eps);
        double g = (gamma) ? gamma[j] : 1.0;
        double b = (beta) ? beta[j] : 0.0;

        for (int i = 0; i < m; i++) {
            batch[i * n_features + j] = g * (batch[i * n_features + j] - mean) * inv_std + b;
        }
    }
}

/*===========================================================================
 * L5: Training Logic
 *===========================================================================*/

void nn_training_config_default(nn_training_config_t *config) {
    if (!config) return;
    config->optimizer = NN_OPTIMIZER_ADAM;
    config->loss = NN_LOSS_MSE;
    config->regularization = NN_REG_NONE;
    config->learning_rate = 0.001;
    config->l1_lambda = 0.0;
    config->l2_lambda = 0.0;
    config->momentum = 0.9;
    config->dropout_rate = 0.0;
    config->huber_delta = 1.0;
    config->beta1 = 0.9;
    config->beta2 = 0.999;
    config->epsilon = 1e-8;
    config->max_epochs = 100;
    config->batch_size = 32;
    config->early_stopping_patience = 20;
    config->target_loss = 1e-6;
    config->lr_decay = 0.98;
    config->lr_decay_steps = 10;
    config->verbose = 0;
    config->print_every = 10;
}

void nn_network_configure_training(nn_network_t *net, const nn_training_config_t *config) {
    if (!net || !config) return;

    net->loss_type = config->loss;
    net->optimizer_type = config->optimizer;
    net->regularization_type = config->regularization;
    net->learning_rate = config->learning_rate;
    net->momentum_coeff = config->momentum;
    net->l1_lambda = config->l1_lambda;
    net->l2_lambda = config->l2_lambda;
    net->huber_delta = config->huber_delta;
    net->beta1 = config->beta1;
    net->beta2 = config->beta2;
    net->epsilon = config->epsilon;
    net->adam_t = 0;
}

double nn_compute_loss(const nn_network_t *net, const double *prediction,
                       const double *target, int n) {
    if (!net || !prediction || !target || n <= 0) return 0.0;

    switch (net->loss_type) {
        case NN_LOSS_MSE:
            return nn_loss_mse(prediction, target, n, NULL);
        case NN_LOSS_MAE:
            return nn_loss_mae(prediction, target, n, NULL);
        case NN_LOSS_HUBER:
            return nn_loss_huber(prediction, target, n, net->huber_delta, NULL);
        case NN_LOSS_MAPE:
            return nn_loss_mape(prediction, target, n, NULL);
        case NN_LOSS_RMSE:
            return sqrt(nn_loss_mse(prediction, target, n, NULL));
        default:
            return nn_loss_mse(prediction, target, n, NULL);
    }
}

/*===========================================================================
 * L5: Activation Derivatives (for Backpropagation)
 *===========================================================================*/

/**
 * @brief Compute derivative of activation function at a point.
 * Used during backpropagation to compute delta for each layer.
 */
static void activation_backward_layer(nn_activation_type_t type,
                                       const double *a, const double *z,
                                       double *da, int n) {
    switch (type) {
        case NN_ACTIVATION_SIGMOID:
            for (int i = 0; i < n; i++) da[i] = a[i] * (1.0 - a[i]);
            break;
        case NN_ACTIVATION_TANH:
            for (int i = 0; i < n; i++) da[i] = 1.0 - a[i] * a[i];
            break;
        case NN_ACTIVATION_RELU:
            for (int i = 0; i < n; i++) da[i] = (z[i] > 0.0) ? 1.0 : 0.0;
            break;
        case NN_ACTIVATION_LEAKY_RELU:
            for (int i = 0; i < n; i++) da[i] = (z[i] > 0.0) ? 1.0 : 0.01;
            break;
        case NN_ACTIVATION_ELU:
            for (int i = 0; i < n; i++) da[i] = (z[i] > 0.0) ? 1.0 : a[i] + 1.0;
            break;
        case NN_ACTIVATION_SWISH:
            for (int i = 0; i < n; i++) {
                double sig;
                if (z[i] > 50.0) sig = 1.0;
                else if (z[i] < -50.0) sig = 0.0;
                else sig = 1.0 / (1.0 + exp(-z[i]));
                da[i] = sig + z[i] * sig * (1.0 - sig);
            }
            break;
        case NN_ACTIVATION_LINEAR:
            for (int i = 0; i < n; i++) da[i] = 1.0;
            break;
        case NN_ACTIVATION_SOFTMAX:
            for (int i = 0; i < n; i++) da[i] = a[i] * (1.0 - a[i]);
            break;
        case NN_ACTIVATION_GELU:
            for (int i = 0; i < n; i++) {
                double x = z[i];
                double phi = exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
                double cdf = 0.5 * (1.0 + erf(x / sqrt(2.0)));
                da[i] = cdf + x * phi;
            }
            break;
        default:
            for (int i = 0; i < n; i++) da[i] = 1.0;
            break;
    }
}

/*===========================================================================
 * L5: Backpropagation
 *===========================================================================*/

void nn_layer_backward(nn_layer_t *layer, const double *delta_next,
                       const double *input) {
    if (!layer || !delta_next || !input) return;

    int out_size = layer->output_size;
    int in_size = layer->input_size;

    /* Compute derivative of activation: f'(z) */
    double *f_prime = (double *)malloc(out_size * sizeof(double));
    if (!f_prime) return;

    activation_backward_layer(layer->activation_type, layer->activations.data,
                               layer->weighted_sum.data, f_prime, out_size);

    /* delta = f'(z) * (W_next^T * delta_next) for hidden layers
     * For output layer, delta = f'(z) * delta_next (delta_next is loss gradient) */
    double *Wt_delta = (double *)malloc(out_size * sizeof(double));
    if (!Wt_delta) { free(f_prime); return; }

    /* For hidden layers, delta_next is from next layer:
     * We need W_next^T * delta_next. But in our architecture, this
     * is computed externally because we don't have the next layer's
     * weights in this layer. For now, delta_next already incorporates
     * W^T multiplication from the calling context.
     *
     * delta_i = f'(z_i) * delta_next_i
     */
    for (int i = 0; i < out_size; i++) {
        layer->delta.data[i] = f_prime[i] * delta_next[i];
    }

    /* Gradient w.r.t. weights: dW = delta * a_prev^T (outer product) */
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++) {
            layer->dW.data[i * in_size + j] = layer->delta.data[i] * input[j];
        }
    }

    /* Gradient w.r.t. biases: db = delta */
    for (int i = 0; i < out_size; i++) {
        layer->db.data[i] = layer->delta.data[i];
    }

    free(Wt_delta);
    free(f_prime);
}

/*===========================================================================
 * L5: Full Backpropagation Through Network
 *===========================================================================*/

void nn_network_backward(nn_network_t *net, const double *input, const double *target) {
    if (!net || !input || !target) return;

    int L = net->num_layers;
    nn_layer_t *out_layer = net->layers[L - 1];

    /* Step 1: Compute loss gradient at output layer */
    int out_size = out_layer->output_size;
    double *loss_grad = (double *)malloc(out_size * sizeof(double));
    if (!loss_grad) return;

    switch (net->loss_type) {
        case NN_LOSS_MSE:
            nn_loss_mse(out_layer->activations.data, target, out_size, loss_grad);
            break;
        case NN_LOSS_MAE:
            nn_loss_mae(out_layer->activations.data, target, out_size, loss_grad);
            break;
        case NN_LOSS_HUBER:
            nn_loss_huber(out_layer->activations.data, target, out_size, net->huber_delta, loss_grad);
            break;
        default:
            nn_loss_mse(out_layer->activations.data, target, out_size, loss_grad);
            break;
    }

    /* Step 2: Output layer backward */
    /* delta_next for output layer is just the loss gradient */
    nn_layer_backward(out_layer, loss_grad, net->layers[L - 2]->activations.data);

    /* Step 3: Backpropagate through hidden layers */
    for (int l = L - 2; l >= 1; l--) {
        nn_layer_t *layer = net->layers[l];
        nn_layer_t *next_layer = net->layers[l + 1];

        /* Compute W_{l+1}^T * delta_{l+1} */
        int next_out = next_layer->output_size;
        int cur_out = layer->output_size;
        double *Wt_delta = (double *)malloc(cur_out * sizeof(double));
        if (!Wt_delta) { free(loss_grad); return; }

        for (int j = 0; j < cur_out; j++) {
            double sum = 0.0;
            for (int k = 0; k < next_out; k++) {
                sum += next_layer->weights.data[k * cur_out + j] * next_layer->delta.data[k];
            }
            Wt_delta[j] = sum;
        }

        nn_layer_backward(layer, Wt_delta, net->layers[l - 1]->activations.data);
        free(Wt_delta);
    }

    free(loss_grad);
}

/*===========================================================================
 * L5: Weight Update
 *===========================================================================*/

void nn_network_update_weights(nn_network_t *net, int epoch) {
    if (!net) return;

    /* Compute current learning rate with schedule */
    double lr = net->learning_rate;
    /* Exponential decay */
    lr = nn_learning_rate_schedule(net->learning_rate, epoch,
                                    LR_SCHEDULE_EXPONENTIAL, 0.98, 10);

    net->adam_t++;

    /* Allocate optimizer state matrices (stored with layers if needed) */
    /* For simplicity, we allocate temporary state per update call */
    for (int l = 1; l < net->num_layers; l++) {
        nn_layer_t *layer = net->layers[l];
        int W_size = layer->output_size * layer->input_size;
        int b_size = layer->output_size;

        switch (net->optimizer_type) {
            case NN_OPTIMIZER_SGD: {
                nn_optimizer_sgd_step(&layer->weights, &layer->dW,
                                       &layer->biases, &layer->db, lr);
                break;
            }
            case NN_OPTIMIZER_MOMENTUM: {
                /* Need velocity buffers - allocate temporarily */
                nn_matrix_t v_W;
                nn_vector_t v_b;
                v_W.data = (double *)calloc(W_size, sizeof(double));
                v_W.rows = layer->output_size;
                v_W.cols = layer->input_size;
                v_W.stride = layer->input_size;
                v_b.data = (double *)calloc(b_size, sizeof(double));
                v_b.length = b_size;

                if (v_W.data && v_b.data) {
                    nn_optimizer_momentum_step(&layer->weights, &layer->dW,
                                                &layer->biases, &layer->db,
                                                lr, net->momentum_coeff, &v_W, &v_b);
                }
                free(v_W.data);
                free(v_b.data);
                break;
            }
            case NN_OPTIMIZER_ADAM: {
                nn_matrix_t m_W, v_W;
                nn_vector_t m_b, v_b;
                m_W.data = (double *)calloc(W_size, sizeof(double));
                v_W.data = (double *)calloc(W_size, sizeof(double));
                m_W.rows = layer->output_size;
                m_W.cols = layer->input_size;
                m_W.stride = layer->input_size;
                v_W.rows = layer->output_size;
                v_W.cols = layer->input_size;
                v_W.stride = layer->input_size;
                m_b.data = (double *)calloc(b_size, sizeof(double));
                v_b.data = (double *)calloc(b_size, sizeof(double));
                m_b.length = b_size;
                v_b.length = b_size;

                if (m_W.data && v_W.data && m_b.data && v_b.data) {
                    nn_optimizer_adam_step(&layer->weights, &layer->dW,
                                            &layer->biases, &layer->db,
                                            lr, net->beta1, net->beta2,
                                            net->epsilon, net->adam_t,
                                            &m_W, &v_W, &m_b, &v_b);
                }
                free(m_W.data); free(v_W.data);
                free(m_b.data); free(v_b.data);
                break;
            }
            case NN_OPTIMIZER_RMSPROP: {
                nn_matrix_t E_W;
                nn_vector_t E_b;
                E_W.data = (double *)calloc(W_size, sizeof(double));
                E_W.rows = layer->output_size;
                E_W.cols = layer->input_size;
                E_W.stride = layer->input_size;
                E_b.data = (double *)calloc(b_size, sizeof(double));
                E_b.length = b_size;

                if (E_W.data && E_b.data) {
                    nn_optimizer_rmsprop_step(&layer->weights, &layer->dW,
                                               &layer->biases, &layer->db,
                                               lr, 0.9, net->epsilon, &E_W, &E_b);
                }
                free(E_W.data); free(E_b.data);
                break;
            }
            default: {
                /* Fallback to SGD */
                nn_optimizer_sgd_step(&layer->weights, &layer->dW,
                                       &layer->biases, &layer->db, lr);
                break;
            }
        }

        /* Zero out gradients for next step */
        memset(layer->dW.data, 0, W_size * sizeof(double));
        memset(layer->db.data, 0, b_size * sizeof(double));
    }
}

/*===========================================================================
 * L5: Training Loop
 *===========================================================================*/

double nn_network_train_epoch(nn_network_t *net, nn_dataset_t *dataset, int epoch) {
    if (!net || !dataset) return 0.0;

    int n_train = dataset->train_end - dataset->train_start;
    if (n_train <= 0) return 0.0;

    int nf = dataset->num_features;
    int no = dataset->num_outputs;
    int batch_size = 32;
    if (batch_size > n_train) batch_size = n_train;
    int n_batches = (n_train + batch_size - 1) / batch_size;

    double total_loss = 0.0;
    int total_samples = 0;

    unsigned int seed = (unsigned int)(42 + epoch);
    int *indices = (int *)malloc(n_train * sizeof(int));
    double *output = (double *)malloc(no * sizeof(double));
    if (!indices || !output) {
        free(indices); free(output);
        return 0.0;
    }

    /* Shuffle training indices */
    for (int i = 0; i < n_train; i++) indices[i] = dataset->train_start + i;
    for (int i = n_train - 1; i > 0; i--) {
        unsigned int r = nn_random_xorshift32(&seed) % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[r];
        indices[r] = tmp;
    }

    net->is_training = 1;

    /* Mini-batch SGD */
    for (int b = 0; b < n_batches; b++) {
        int batch_start = b * batch_size;
        int batch_end = (b + 1) * batch_size;
        if (batch_end > n_train) batch_end = n_train;
        int actual_batch = batch_end - batch_start;

        double batch_loss = 0.0;

        /* Process each sample in batch */
        for (int s = batch_start; s < batch_end; s++) {
            int idx = indices[s];
            const double *x = dataset->X.data + idx * nf;
            const double *y = dataset->Y.data + idx * no;

            /* Forward pass */
            nn_network_forward(net, x, output);

            /* Compute loss */
            double sample_loss = nn_compute_loss(net, output, y, no);
            batch_loss += sample_loss;

            /* Backward pass (accumulate gradients) */
            nn_network_backward(net, x, y);
        }

        /* Update weights after processing batch */
        nn_network_update_weights(net, epoch);

        total_loss += batch_loss;
        total_samples += actual_batch;
    }

    net->is_training = 0;

    free(indices);
    free(output);

    return (total_samples > 0) ? total_loss / total_samples : 0.0;
}

void nn_network_train(nn_network_t *net, nn_dataset_t *dataset) {
    if (!net || !dataset) return;

    int max_epochs = 100;
    int patience = 20;
    int n_val = dataset->val_end - dataset->val_start;

    /* Allocate history buffers */
    net->history_capacity = max_epochs;
    net->train_loss_history = (double *)malloc(max_epochs * sizeof(double));
    net->val_loss_history = (double *)malloc(max_epochs * sizeof(double));
    net->history_count = 0;

    double best_val_loss = 1e100;
    int epochs_without_improvement = 0;

    nn_regression_metrics_t val_metrics;

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        /* Train one epoch */
        double train_loss = nn_network_train_epoch(net, dataset, epoch);

        /* Validation */
        double val_loss = 0.0;
        if (n_val > 0) {
            nn_network_evaluate(net, dataset, dataset->val_start, dataset->val_end, &val_metrics);
            val_loss = val_metrics.rmse;
        }

        /* Record history */
        if (net->train_loss_history && net->val_loss_history && epoch < net->history_capacity) {
            net->train_loss_history[epoch] = train_loss;
            net->val_loss_history[epoch] = val_loss;
            net->history_count = epoch + 1;
        }

        /* Early stopping check */
        if (n_val > 0) {
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement++;
            }
            if (epochs_without_improvement >= patience) {
                break;
            }
        }

        net->total_epochs = epoch + 1;
    }

    net->is_trained = 1;
}

/*===========================================================================
 * L5: Learning Rate Scheduling
 *===========================================================================*/

double nn_learning_rate_schedule(double base_lr, int epoch,
                                  nn_lr_schedule_type_t schedule,
                                  double decay_rate, int decay_steps) {
    switch (schedule) {
        case LR_SCHEDULE_STEP:
            /* lr = base_lr * decay_rate^(floor(epoch / decay_steps)) */
            return base_lr * pow(decay_rate, floor((double)epoch / decay_steps));

        case LR_SCHEDULE_EXPONENTIAL:
            /* lr = base_lr * exp(-decay_rate * epoch) */
            return base_lr * exp(-decay_rate * epoch);

        case LR_SCHEDULE_INVERSE:
            /* lr = base_lr / (1 + decay_rate * epoch) */
            return base_lr / (1.0 + decay_rate * epoch);

        case LR_SCHEDULE_COSINE:
            /* Cosine annealing (Loshchilov & Hutter 2017):
             * lr = base_lr * 0.5 * (1 + cos(pi * epoch / max_epochs)) */
            return base_lr * 0.5 * (1.0 + cos(M_PI * epoch / (double)decay_steps));

        case LR_SCHEDULE_CYCLIC:
            /* Cyclic LR (Smith 2017):
             * lr oscillates between base_lr and base_lr/10 with period decay_steps */
            {
                double cycle_progress = fmod((double)epoch, (double)decay_steps) / decay_steps;
                double min_lr = base_lr * 0.1;
                return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos(2.0 * M_PI * cycle_progress));
            }

        case LR_SCHEDULE_CONSTANT:
        default:
            return base_lr;
    }
}

/*===========================================================================
 * L5: Random Number Generation
 *===========================================================================*/

unsigned int nn_random_xorshift32(unsigned int *state) {
    /* xorshift32: Marsaglia (2003), period 2^32 - 1 */
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

double nn_random_normal(unsigned int *state) {
    /* Box-Muller transform: generate N(0,1) from two U(0,1) */
    double u1 = nn_random_uniform(state);
    double u2 = nn_random_uniform(state);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

double nn_random_uniform(unsigned int *state) {
    /* Convert xorshift32 output to U(0,1) */
    unsigned int r = nn_random_xorshift32(state);
    return (double)(r & 0x7FFFFFFF) / 0x7FFFFFFF;
}

/*===========================================================================
 * L5: Mini-batch Sampling
 *===========================================================================*/

void nn_sample_mini_batch(const nn_dataset_t *dataset, int *batch_indices,
                           int batch_size, unsigned int *seed) {
    if (!dataset || !batch_indices || batch_size <= 0) return;

    int n = dataset->num_samples;
    for (int i = 0; i < batch_size; i++) {
        batch_indices[i] = nn_random_xorshift32(seed) % n;
    }
}

/*===========================================================================
 * L5: Persistence
 *===========================================================================*/

int nn_network_save(const nn_network_t *net, const char *filename) {
    if (!net || !filename) return -1;

    FILE *f = fopen(filename, "w");
    if (!f) return -1;

    /* Header: num_layers, layer_sizes... */
    fprintf(f, "%d\n", net->num_layers);
    for (int l = 0; l < net->num_layers; l++) {
        fprintf(f, "%d ", net->arch.layer_sizes[l]);
    }
    fprintf(f, "\n");

    /* Weights and biases for each layer */
    for (int l = 0; l < net->num_layers; l++) {
        nn_layer_t *layer = net->layers[l];
        int w_size = layer->output_size * layer->input_size;
        int b_size = layer->output_size;

        /* Weights */
        for (int i = 0; i < w_size; i++) {
            fprintf(f, "%.15e ", layer->weights.data[i]);
        }
        fprintf(f, "\n");

        /* Biases */
        for (int i = 0; i < b_size; i++) {
            fprintf(f, "%.15e ", layer->biases.data[i]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    return 0;
}

nn_network_t *nn_network_load(const char *filename) {
    if (!filename) return NULL;

    FILE *f = fopen(filename, "r");
    if (!f) return NULL;

    int num_layers;
    if (fscanf(f, "%d", &num_layers) != 1 || num_layers < 2) {
        fclose(f);
        return NULL;
    }

    int *layer_sizes = (int *)malloc(num_layers * sizeof(int));
    if (!layer_sizes) { fclose(f); return NULL; }

    for (int l = 0; l < num_layers; l++) {
        if (fscanf(f, "%d", &layer_sizes[l]) != 1) {
            free(layer_sizes); fclose(f);
            return NULL;
        }
    }

    nn_architecture_t arch;
    arch.num_layers = num_layers;
    arch.layer_sizes = layer_sizes;
    arch.hidden_activation = NN_ACTIVATION_RELU;
    arch.output_activation = NN_ACTIVATION_LINEAR;
    arch.dropout_rate = 0.0;

    nn_network_t *net = nn_network_create(&arch);
    free(layer_sizes);
    if (!net) { fclose(f); return NULL; }

    /* Read weights and biases */
    for (int l = 0; l < num_layers; l++) {
        nn_layer_t *layer = net->layers[l];
        int w_size = layer->output_size * layer->input_size;
        int b_size = layer->output_size;

        for (int i = 0; i < w_size; i++) {
            if (fscanf(f, "%lf", &layer->weights.data[i]) != 1) {
                nn_network_free(net); fclose(f);
                return NULL;
            }
        }
        for (int i = 0; i < b_size; i++) {
            if (fscanf(f, "%lf", &layer->biases.data[i]) != 1) {
                nn_network_free(net); fclose(f);
                return NULL;
            }
        }
    }

    net->is_trained = 1;
    fclose(f);
    return net;
}

/*===========================================================================
 * L8: Ensemble Methods
 *===========================================================================*/

nn_ensemble_t *nn_ensemble_create(int num_members, const nn_architecture_t *arch) {
    if (num_members <= 0 || !arch) return NULL;

    nn_ensemble_t *ensemble = (nn_ensemble_t *)calloc(1, sizeof(nn_ensemble_t));
    if (!ensemble) return NULL;

    ensemble->num_members = num_members;
    ensemble->networks = (nn_network_t **)calloc(num_members, sizeof(nn_network_t *));
    ensemble->weights = (double *)calloc(num_members, sizeof(double));
    if (!ensemble->networks || !ensemble->weights) {
        nn_ensemble_free(ensemble);
        return NULL;
    }

    /* Create each member network */
    for (int i = 0; i < num_members; i++) {
        ensemble->networks[i] = nn_network_create(arch);
        if (!ensemble->networks[i]) {
            nn_ensemble_free(ensemble);
            return NULL;
        }
        ensemble->weights[i] = 1.0 / num_members; /* Equal weighting by default */
    }

    return ensemble;
}

void nn_ensemble_predict(nn_ensemble_t *ensemble, const double *input, double *output) {
    if (!ensemble || !input || !output) return;

    /* Clear output */
    for (int i = 0; i < 8; i++) output[i] = 0.0; /* Assume max 8 outputs */

    for (int m = 0; m < ensemble->num_members; m++) {
        int out_size = ensemble->networks[m]->layers[ensemble->networks[m]->num_layers - 1]->output_size;
        double *member_out = (double *)calloc(out_size, sizeof(double));
        if (!member_out) continue;

        nn_network_forward(ensemble->networks[m], input, member_out);

        for (int o = 0; o < out_size; o++) {
            output[o] += ensemble->weights[m] * member_out[o];
        }

        free(member_out);
    }
}

void nn_ensemble_free(nn_ensemble_t *ensemble) {
    if (!ensemble) return;

    if (ensemble->networks) {
        for (int i = 0; i < ensemble->num_members; i++) {
            if (ensemble->networks[i]) {
                nn_network_free(ensemble->networks[i]);
            }
        }
        free(ensemble->networks);
    }

    free(ensemble->weights);
    free(ensemble);
}

/*===========================================================================
 * L8: MC Dropout (Bayesian Approximation)
 *===========================================================================*/

nn_mc_dropout_t *nn_mc_dropout_create(nn_network_t *net, int num_samples) {
    if (!net || num_samples <= 0) return NULL;

    nn_mc_dropout_t *mc = (nn_mc_dropout_t *)calloc(1, sizeof(nn_mc_dropout_t));
    if (!mc) return NULL;

    mc->network = net;
    mc->num_samples = num_samples;

    int out_size = net->layers[net->num_layers - 1]->output_size;
    mc->mean_predictions = (double *)calloc(out_size, sizeof(double));
    mc->variance_predictions = (double *)calloc(out_size, sizeof(double));
    mc->lower_bound = (double *)calloc(out_size, sizeof(double));
    mc->upper_bound = (double *)calloc(out_size, sizeof(double));

    return mc;
}

void nn_mc_dropout_predict(nn_mc_dropout_t *mc, const double *input, double *output) {
    if (!mc || !input || !output) return;

    int out_size = mc->network->layers[mc->network->num_layers - 1]->output_size;
    int T = mc->num_samples;

    /* Accumulate predictions over T stochastic forward passes */
    double *sum = (double *)calloc(out_size, sizeof(double));
    double *sum_sq = (double *)calloc(out_size, sizeof(double));
    double *sample_out = (double *)calloc(out_size, sizeof(double));

    if (!sum || !sum_sq || !sample_out) {
        free(sum); free(sum_sq); free(sample_out);
        return;
    }

    /* Temporarily enable dropout during inference */
    int orig_training = mc->network->is_training;
    mc->network->is_training = 1;

    for (int t = 0; t < T; t++) {
        nn_network_forward(mc->network, input, sample_out);
        for (int o = 0; o < out_size; o++) {
            sum[o] += sample_out[o];
            sum_sq[o] += sample_out[o] * sample_out[o];
        }
    }

    mc->network->is_training = orig_training;

    /* Compute mean and variance */
    for (int o = 0; o < out_size; o++) {
        mc->mean_predictions[o] = sum[o] / T;
        double var = sum_sq[o] / T - mc->mean_predictions[o] * mc->mean_predictions[o];
        if (var < 0.0) var = 0.0;
        mc->variance_predictions[o] = var;
        mc->lower_bound[o] = mc->mean_predictions[o] - 1.96 * sqrt(var);
        mc->upper_bound[o] = mc->mean_predictions[o] + 1.96 * sqrt(var);
        output[o] = mc->mean_predictions[o];
    }

    /* Epistemic uncertainty = predictive variance */
    mc->epistemic_uncertainty = 0.0;
    for (int o = 0; o < out_size; o++) {
        mc->epistemic_uncertainty += mc->variance_predictions[o];
    }
    mc->epistemic_uncertainty /= out_size;

    free(sum);
    free(sum_sq);
    free(sample_out);
}

void nn_mc_dropout_free(nn_mc_dropout_t *mc) {
    if (!mc) return;
    free(mc->mean_predictions);
    free(mc->variance_predictions);
    free(mc->lower_bound);
    free(mc->upper_bound);
    free(mc);
}
