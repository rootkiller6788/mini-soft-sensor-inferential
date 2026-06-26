/**
 * @file nn_soft_sensor_core.c
 * @brief Core implementation: network creation, forward pass, soft sensor lifecycle.
 *
 * Level: L1-L3 (Definitions, Core Concepts, Engineering Structures)
 * Reference: Bishop (1995), Goodfellow et al. (2016)
 *            He et al. (2015), Glorot & Bengio (2010)
 */

#include "nn_soft_sensor.h"
#include "nn_training.h"
#include "nn_sensor_validation.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*===========================================================================
 * L3: Internal Matrix/Vector Helpers
 *===========================================================================*/

/**
 * @brief Matrix-vector multiply: y = A * x.
 * A is (m x n), x is (n), y is (m). No bias added.
 */
static void matrix_vector_mul(const nn_matrix_t *A, const double *x, double *y) {
    int m = A->rows;
    int n = A->cols;
    int stride = A->stride;
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        const double *row = A->data + i * stride;
        for (int j = 0; j < n; j++) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

/*===========================================================================
 * L1-L2: Weight Initialization
 *===========================================================================*/

void nn_he_initialize(double *data, int rows, int cols) {
    /* He initialization: W ~ N(0, sqrt(2/n_in))
     * Using Box-Muller with deterministic seed for reproducibility.
     * Reference: He, Zhang, Ren, Sun (2015) - ICCV */
    unsigned int seed = 42;
    double std_dev = sqrt(2.0 / cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            /* Box-Muller: sqrt(-2*ln(U1)) * cos(2*pi*U2) */
            seed = seed * 1103515245 + 12345;
            double u1 = (double)(seed & 0x7FFFFFFF) / 0x7FFFFFFF;
            seed = seed * 1103515245 + 12345;
            double u2 = (double)(seed & 0x7FFFFFFF) / 0x7FFFFFFF;
            if (u1 < 1e-10) u1 = 1e-10;
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            /* Next iteration uses sin for second independent normal */
            data[i * cols + j] = z * std_dev;
            j++;
            if (j >= cols) break;
            z = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
            data[i * cols + j] = z * std_dev;
        }
    }
}

void nn_xavier_initialize(double *data, int rows, int cols) {
    /* Xavier initialization: W ~ U(-limit, limit)
     * limit = sqrt(6 / (n_in + n_out))
     * Reference: Glorot & Bengio (2010) - AISTATS */
    double limit = sqrt(6.0 / (cols + rows));
    unsigned int seed = 42;
    for (int i = 0; i < rows * cols; i++) {
        seed = seed * 1103515245 + 12345;
        double r = (double)(seed & 0x7FFFFFFF) / 0x7FFFFFFF;
        data[i] = -limit + 2.0 * limit * r;
    }
}

/*===========================================================================
 * L1-L3: Network Creation and Destruction
 *===========================================================================*/

nn_network_t *nn_network_create(const nn_architecture_t *arch) {
    if (!arch || arch->num_layers < 2 || !arch->layer_sizes) {
        return NULL;
    }

    nn_network_t *net = (nn_network_t *)calloc(1, sizeof(nn_network_t));
    if (!net) return NULL;

    memcpy(&net->arch, arch, sizeof(nn_architecture_t));
    net->num_layers = arch->num_layers;

    /* Deep copy layer sizes */
    net->arch.layer_sizes = (int *)malloc(arch->num_layers * sizeof(int));
    if (!net->arch.layer_sizes) { free(net); return NULL; }
    memcpy(net->arch.layer_sizes, arch->layer_sizes, arch->num_layers * sizeof(int));

    /* Allocate layer pointers */
    net->layers = (nn_layer_t **)calloc(arch->num_layers, sizeof(nn_layer_t *));
    if (!net->layers) {
        free(net->arch.layer_sizes); free(net);
        return NULL;
    }

    /* Create each layer */
    for (int l = 0; l < arch->num_layers; l++) {
        nn_layer_t *layer = (nn_layer_t *)calloc(1, sizeof(nn_layer_t));
        if (!layer) {
            /* Partial cleanup handled by nn_network_free */
            nn_network_free(net);
            return NULL;
        }

        int input_size = (l == 0) ? arch->layer_sizes[0] : arch->layer_sizes[l - 1];
        int output_size = arch->layer_sizes[l];

        layer->input_size = input_size;
        layer->output_size = output_size;
        layer->dropout_rate = arch->dropout_rate;
        layer->prev = (l > 0) ? net->layers[l - 1] : NULL;
        if (l > 0) net->layers[l - 1]->next = layer;

        /* Determine activation type */
        if (l == 0) {
            layer->activation_type = NN_ACTIVATION_LINEAR; /* Input layer: pass-through */
        } else if (l == arch->num_layers - 1) {
            layer->activation_type = arch->output_activation;
        } else {
            layer->activation_type = arch->hidden_activation;
        }

        /* Allocate and initialize weights, biases, activations, etc. */
        /* Weights: output_size x input_size */
        layer->weights.data = (double *)calloc((size_t)output_size * input_size, sizeof(double));
        layer->weights.rows = output_size;
        layer->weights.cols = input_size;
        layer->weights.stride = input_size;

        /* Biases: output_size */
        layer->biases.data = (double *)calloc(output_size, sizeof(double));
        layer->biases.length = output_size;

        /* Activations */
        layer->activations.data = (double *)calloc(output_size, sizeof(double));
        layer->activations.length = output_size;

        /* Weighted sum z = Wx + b */
        layer->weighted_sum.data = (double *)calloc(output_size, sizeof(double));
        layer->weighted_sum.length = output_size;

        /* Gradients for training */
        layer->dW.data = (double *)calloc((size_t)output_size * input_size, sizeof(double));
        layer->dW.rows = output_size;
        layer->dW.cols = input_size;
        layer->dW.stride = input_size;

        layer->db.data = (double *)calloc(output_size, sizeof(double));
        layer->db.length = output_size;

        layer->delta.data = (double *)calloc(output_size, sizeof(double));
        layer->delta.length = output_size;

        /* Initialize weights if not input layer */
        if (l > 0) {
            if (layer->activation_type == NN_ACTIVATION_RELU ||
                layer->activation_type == NN_ACTIVATION_LEAKY_RELU ||
                layer->activation_type == NN_ACTIVATION_ELU) {
                nn_he_initialize(layer->weights.data, output_size, input_size);
            } else {
                nn_xavier_initialize(layer->weights.data, output_size, input_size);
            }
            /* Biases already zero-initialized */
        }

        net->layers[l] = layer;
    }

    /* Default configuration */
    nn_training_config_t config;
    nn_training_config_default(&config);
    nn_network_configure_training(net, &config);

    return net;
}

void nn_network_free(nn_network_t *net) {
    if (!net) return;

    if (net->layers) {
        for (int l = 0; l < net->num_layers; l++) {
            nn_layer_t *layer = net->layers[l];
            if (layer) {
                free(layer->weights.data);
                free(layer->biases.data);
                free(layer->activations.data);
                free(layer->weighted_sum.data);
                free(layer->dW.data);
                free(layer->db.data);
                free(layer->delta.data);
                free(layer);
            }
        }
        free(net->layers);
    }

    free(net->arch.layer_sizes);
    free(net->train_loss_history);
    free(net->val_loss_history);
    free(net);
}

/*===========================================================================
 * L2: Activation Functions
 *
 * Each activation has the form: a = f(z), where z = Wx + b
 *===========================================================================*/

/**
 * @brief Apply activation function to weighted sum z.
 * @param type  Activation type.
 * @param z     Weighted sum (input, length n).
 * @param a     Activation output (length n).
 * @param n     Vector length.
 */
static void activation_forward(nn_activation_type_t type, const double *z, double *a, int n) {
    switch (type) {
        case NN_ACTIVATION_SIGMOID:
            /* sigma(x) = 1 / (1 + exp(-x))
             * Range: (0, 1). Used in binary classification output.
             * Derivative: sigma(x)*(1-sigma(x)) */
            for (int i = 0; i < n; i++) {
                /* Clamp for numerical stability */
                if (z[i] > 50.0) a[i] = 1.0;
                else if (z[i] < -50.0) a[i] = 0.0;
                else a[i] = 1.0 / (1.0 + exp(-z[i]));
            }
            break;
        case NN_ACTIVATION_TANH:
            /* tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
             * Range: (-1, 1). Zero-centered, preferred over sigmoid for hidden layers.
             * Derivative: 1 - tanh(x)^2 */
            for (int i = 0; i < n; i++) {
                a[i] = tanh(z[i]);
            }
            break;
        case NN_ACTIVATION_RELU:
            /* ReLU(x) = max(0, x)
             * Most widely used activation for hidden layers.
             * Sparse activation, avoids vanishing gradient for x > 0.
             * Derivative: 1 if x > 0 else 0.
             * Dying ReLU problem: neurons with x <= 0 get zero gradient. */
            for (int i = 0; i < n; i++) {
                a[i] = (z[i] > 0.0) ? z[i] : 0.0;
            }
            break;
        case NN_ACTIVATION_LEAKY_RELU:
            /* LeakyReLU(x) = x if x > 0 else alpha * x
             * Fixes dying ReLU by allowing small gradient for x <= 0.
             * Typical alpha = 0.01 (Maas, Hannun, Ng 2013). */
            for (int i = 0; i < n; i++) {
                a[i] = (z[i] > 0.0) ? z[i] : 0.01 * z[i];
            }
            break;
        case NN_ACTIVATION_ELU:
            /* ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
             * Exponential Linear Unit (Clevert, Unterthiner, Hochreiter 2015).
             * Smooth for x < 0, pushes mean activation towards zero.
             * Derivative: 1 if x > 0 else ELU(x) + alpha */
            for (int i = 0; i < n; i++) {
                if (z[i] > 0.0) a[i] = z[i];
                else {
                    if (z[i] < -50.0) a[i] = -1.0; /* e^{-50} ~ 0 */
                    else a[i] = exp(z[i]) - 1.0;
                }
            }
            break;
        case NN_ACTIVATION_SWISH:
            /* Swish(x) = x * sigmoid(x)
             * Self-gated activation (Ramachandran, Zoph, Le 2017).
             * Discovered by neural architecture search.
             * Non-monotonic, smooth, unbounded above, bounded below. */
            for (int i = 0; i < n; i++) {
                double sig;
                if (z[i] > 50.0) sig = 1.0;
                else if (z[i] < -50.0) sig = 0.0;
                else sig = 1.0 / (1.0 + exp(-z[i]));
                a[i] = z[i] * sig;
            }
            break;
        case NN_ACTIVATION_LINEAR:
            /* f(x) = x. Used for regression output layer.
             * Derivative: 1. */
            for (int i = 0; i < n; i++) {
                a[i] = z[i];
            }
            break;
        case NN_ACTIVATION_SOFTMAX:
            /* softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
             * Outputs form a probability distribution (sum = 1).
             * Used for multi-class classification output.
             * For numerical stability, subtract max first:
             * exp(x_i - max) / sum(exp(x_j - max)) */
            {
                double max_val = z[0];
                for (int i = 1; i < n; i++) {
                    if (z[i] > max_val) max_val = z[i];
                }
                double sum_exp = 0.0;
                for (int i = 0; i < n; i++) {
                    a[i] = exp(z[i] - max_val);
                    sum_exp += a[i];
                }
                if (sum_exp > 0.0) {
                    for (int i = 0; i < n; i++) {
                        a[i] /= sum_exp;
                    }
                }
            }
            break;
        case NN_ACTIVATION_GELU:
            /* GELU(x) = x * Phi(x) where Phi is the standard normal CDF.
             * Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
             * Gaussian Error Linear Unit (Hendrycks & Gimpel 2016).
             * Used in BERT, GPT, and many transformer architectures. */
            for (int i = 0; i < n; i++) {
                double x = z[i];
                double cdf_approx = 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
                a[i] = x * cdf_approx;
            }
            break;
        default:
            break;
    }
}

/*===========================================================================
 * L2: Forward Propagation
 *===========================================================================*/

void nn_layer_forward(nn_layer_t *layer, const double *input) {
    if (!layer || !input) return;

    int out_size = layer->output_size;

    /* z = W * x + b */
    /* First, compute W * x */
    double *z = layer->weighted_sum.data;
    matrix_vector_mul(&layer->weights, input, layer->weighted_sum.data);

    /* Add bias */
    for (int i = 0; i < out_size; i++) {
        z[i] += layer->biases.data[i];
    }

    /* Apply dropout during training (handled externally) */

    /* Apply activation */
    activation_forward(layer->activation_type, z, layer->activations.data, out_size);
}

void nn_network_forward(const nn_network_t *net, const double *input, double *output) {
    if (!net || !input || !output) return;

    /* Set input layer activations directly (no computation for input layer) */
    nn_layer_t *first_layer = net->layers[0];
    for (int i = 0; i < first_layer->output_size; i++) {
        first_layer->activations.data[i] = input[i];
        first_layer->weighted_sum.data[i] = input[i];
    }

    /* Forward through all subsequent layers */
    for (int l = 1; l < net->num_layers; l++) {
        nn_layer_t *layer = net->layers[l];
        const double *prev_activations = net->layers[l - 1]->activations.data;

        nn_layer_forward(layer, prev_activations);

        /* Apply dropout during training */
        if (net->is_training && net->regularization_type == NN_REG_DROPOUT
            && layer->dropout_rate > 0.0 && l < net->num_layers - 1) {
            unsigned int seed = (unsigned int)(42 + l);
            nn_dropout_apply(layer->activations.data, layer->output_size,
                             layer->dropout_rate, &seed);
        }
    }

    /* Copy output layer activations to output buffer */
    nn_layer_t *output_layer = net->layers[net->num_layers - 1];
    int out_size = output_layer->output_size;
    for (int i = 0; i < out_size; i++) {
        output[i] = output_layer->activations.data[i];
    }
}

/*===========================================================================
 * L2-L5: Soft Sensor Lifecycle
 *===========================================================================*/

soft_sensor_t *soft_sensor_create(const char *name, const char *process_unit,
                                   const char *quality_var, const char *unit,
                                   soft_sensor_type_t sensor_type,
                                   nn_network_t *net) {
    soft_sensor_t *sensor = (soft_sensor_t *)calloc(1, sizeof(soft_sensor_t));
    if (!sensor) return NULL;

    if (name) strncpy(sensor->name, name, 63);
    if (process_unit) strncpy(sensor->process_unit, process_unit, 63);
    if (quality_var) strncpy(sensor->quality_variable, quality_var, 63);
    if (unit) strncpy(sensor->unit, unit, 15);

    sensor->network = net;
    sensor->sensor_type = sensor_type;
    sensor->sampling_period_sec = 1.0;

    return sensor;
}

void soft_sensor_free(soft_sensor_t *sensor) {
    if (!sensor) return;

    /* Note: network is owned separately and freed by caller if desired.
     * Dataset is also owned separately. Here we free sensor-owned buffers. */
    if (sensor->input_buffer) {
        free(sensor->input_buffer);
    }

    free(sensor);
}

void soft_sensor_predict(soft_sensor_t *sensor, const double *input, double *output) {
    if (!sensor || !sensor->network || !input || !output) return;

    /* If dataset has normalization parameters, normalize input first */
    double *norm_input = NULL;
    const double *effective_input = input;

    if (sensor->dataset && sensor->dataset->is_normalized
        && sensor->dataset->X_norm) {
        int nf = sensor->dataset->X_norm->num_features;
        norm_input = (double *)malloc(nf * sizeof(double));
        if (norm_input) {
            nn_normalization_t *norm = sensor->dataset->X_norm;
            if (norm->type == NN_NORM_ZSCORE && norm->mean && norm->std) {
                for (int i = 0; i < nf; i++) {
                    double s = norm->std[i];
                    norm_input[i] = (s > 1e-10) ? (input[i] - norm->mean[i]) / s : input[i];
                }
            } else if (norm->type == NN_NORM_MINMAX && norm->min && norm->max) {
                for (int i = 0; i < nf; i++) {
                    double range = norm->max[i] - norm->min[i];
                    norm_input[i] = (range > 1e-10) ? (input[i] - norm->min[i]) / range : input[i];
                }
            }
            effective_input = norm_input;
        }
    }

    /* Forward pass */
    nn_network_forward(sensor->network, effective_input, output);

    /* Denormalize output if needed */
    if (sensor->dataset && sensor->dataset->is_normalized
        && sensor->dataset->Y_norm) {
        nn_normalization_t *norm = sensor->dataset->Y_norm;
        int no = norm->num_features;
        if (norm->type == NN_NORM_ZSCORE && norm->mean && norm->std) {
            for (int i = 0; i < no; i++) {
                output[i] = output[i] * norm->std[i] + norm->mean[i];
            }
        } else if (norm->type == NN_NORM_MINMAX && norm->min && norm->max) {
            for (int i = 0; i < no; i++) {
                output[i] = output[i] * (norm->max[i] - norm->min[i]) + norm->min[i];
            }
        }
    }

    free(norm_input);

    sensor->total_predictions++;
}

/*===========================================================================
 * L5: Dataset Operations
 *===========================================================================*/

nn_dataset_t *nn_dataset_create(const double *X, const double *Y,
                                 int n_samples, int n_features, int n_outputs,
                                 double train_ratio, double val_ratio) {
    if (!X || !Y || n_samples <= 0 || n_features <= 0 || n_outputs <= 0) {
        return NULL;
    }
    if (train_ratio <= 0.0 || train_ratio + val_ratio >= 1.0) {
        return NULL;
    }

    nn_dataset_t *ds = (nn_dataset_t *)calloc(1, sizeof(nn_dataset_t));
    if (!ds) return NULL;

    ds->num_samples = n_samples;
    ds->num_features = n_features;
    ds->num_outputs = n_outputs;

    /* Allocate and copy data */
    size_t x_size = (size_t)n_samples * n_features;
    size_t y_size = (size_t)n_samples * n_outputs;

    ds->X.data = (double *)malloc(x_size * sizeof(double));
    ds->Y.data = (double *)malloc(y_size * sizeof(double));
    if (!ds->X.data || !ds->Y.data) {
        nn_dataset_free(ds);
        return NULL;
    }
    memcpy(ds->X.data, X, x_size * sizeof(double));
    memcpy(ds->Y.data, Y, y_size * sizeof(double));
    ds->X.rows = n_samples;
    ds->X.cols = n_features;
    ds->X.stride = n_features;
    ds->Y.rows = n_samples;
    ds->Y.cols = n_outputs;
    ds->Y.stride = n_outputs;

    /* Compute data splits */
    ds->train_start = 0;
    ds->train_end = (int)(n_samples * train_ratio);
    ds->val_start = ds->train_end;
    ds->val_end = ds->val_start + (int)(n_samples * val_ratio);
    ds->test_start = ds->val_end;
    ds->test_end = n_samples;

    return ds;
}

void nn_dataset_free(nn_dataset_t *dataset) {
    if (!dataset) return;

    free(dataset->X.data);
    free(dataset->Y.data);

    if (dataset->X_norm) {
        free(dataset->X_norm->mean);
        free(dataset->X_norm->std);
        free(dataset->X_norm->min);
        free(dataset->X_norm->max);
        free(dataset->X_norm->median);
        free(dataset->X_norm->iqr);
        free(dataset->X_norm);
    }
    if (dataset->Y_norm) {
        free(dataset->Y_norm->mean);
        free(dataset->Y_norm->std);
        free(dataset->Y_norm->min);
        free(dataset->Y_norm->max);
        free(dataset->Y_norm->median);
        free(dataset->Y_norm->iqr);
        free(dataset->Y_norm);
    }

    free(dataset);
}

void nn_dataset_normalize(nn_dataset_t *dataset,
                          nn_normalization_type_t x_type,
                          nn_normalization_type_t y_type) {
    if (!dataset || dataset->num_samples <= 0) return;

    int nf = dataset->num_features;
    int no = dataset->num_outputs;
    int ns = dataset->num_samples;

    /* X normalization */
    dataset->X_norm = (nn_normalization_t *)calloc(1, sizeof(nn_normalization_t));
    dataset->X_norm->type = x_type;
    dataset->X_norm->num_features = nf;

    /* Allocate parameter arrays */
    dataset->X_norm->mean = (double *)calloc(nf, sizeof(double));
    dataset->X_norm->std = (double *)calloc(nf, sizeof(double));
    dataset->X_norm->min = (double *)calloc(nf, sizeof(double));
    dataset->X_norm->max = (double *)calloc(nf, sizeof(double));
    dataset->X_norm->median = (double *)calloc(nf, sizeof(double));
    dataset->X_norm->iqr = (double *)calloc(nf, sizeof(double));

    /* Compute statistics and normalize X */
    for (int j = 0; j < nf; j++) {
        /* Compute mean */
        double sum = 0.0;
        for (int i = 0; i < ns; i++) sum += dataset->X.data[i * nf + j];
        dataset->X_norm->mean[j] = sum / ns;

        /* Compute min/max */
        double min_val = dataset->X.data[j];
        double max_val = dataset->X.data[j];
        for (int i = 1; i < ns; i++) {
            double v = dataset->X.data[i * nf + j];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        dataset->X_norm->min[j] = min_val;
        dataset->X_norm->max[j] = max_val;

        /* Compute std */
        double sum_sq = 0.0;
        double m = dataset->X_norm->mean[j];
        for (int i = 0; i < ns; i++) {
            double d = dataset->X.data[i * nf + j] - m;
            sum_sq += d * d;
        }
        dataset->X_norm->std[j] = sqrt(sum_sq / ns);

        /* Apply normalization */
        if (x_type == NN_NORM_ZSCORE) {
            double s = dataset->X_norm->std[j];
            if (s < 1e-10) s = 1e-10;
            for (int i = 0; i < ns; i++) {
                dataset->X.data[i * nf + j] = (dataset->X.data[i * nf + j] - m) / s;
            }
        } else if (x_type == NN_NORM_MINMAX) {
            double range = max_val - min_val;
            if (range < 1e-10) range = 1e-10;
            for (int i = 0; i < ns; i++) {
                dataset->X.data[i * nf + j] = (dataset->X.data[i * nf + j] - min_val) / range;
            }
        }
    }

    /* Y normalization */
    dataset->Y_norm = (nn_normalization_t *)calloc(1, sizeof(nn_normalization_t));
    dataset->Y_norm->type = y_type;
    dataset->Y_norm->num_features = no;
    dataset->Y_norm->mean = (double *)calloc(no, sizeof(double));
    dataset->Y_norm->std = (double *)calloc(no, sizeof(double));
    dataset->Y_norm->min = (double *)calloc(no, sizeof(double));
    dataset->Y_norm->max = (double *)calloc(no, sizeof(double));

    for (int j = 0; j < no; j++) {
        double sum = 0.0;
        for (int i = 0; i < ns; i++) sum += dataset->Y.data[i * no + j];
        dataset->Y_norm->mean[j] = sum / ns;

        double min_val = dataset->Y.data[j];
        double max_val = dataset->Y.data[j];
        for (int i = 1; i < ns; i++) {
            double v = dataset->Y.data[i * no + j];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        dataset->Y_norm->min[j] = min_val;
        dataset->Y_norm->max[j] = max_val;

        double sum_sq = 0.0;
        double m = dataset->Y_norm->mean[j];
        for (int i = 0; i < ns; i++) {
            double d = dataset->Y.data[i * no + j] - m;
            sum_sq += d * d;
        }
        double std_val = sqrt(sum_sq / ns);
        dataset->Y_norm->std[j] = (std_val > 1e-10) ? std_val : 1.0;

        /* Apply normalization */
        if (y_type == NN_NORM_ZSCORE) {
            double s = dataset->Y_norm->std[j];
            if (s < 1e-10) s = 1e-10;
            for (int i = 0; i < ns; i++) {
                dataset->Y.data[i * no + j] = (dataset->Y.data[i * no + j] - m) / s;
            }
        } else if (y_type == NN_NORM_MINMAX) {
            double range = max_val - min_val;
            if (range < 1e-10) range = 1e-10;
            for (int i = 0; i < ns; i++) {
                dataset->Y.data[i * no + j] = (dataset->Y.data[i * no + j] - min_val) / range;
            }
        }
    }

    dataset->is_normalized = 1;
}

/*===========================================================================
 * L4-L6: Model Evaluation
 *===========================================================================*/

void nn_network_evaluate(const nn_network_t *net, const nn_dataset_t *dataset,
                          int start, int end, nn_regression_metrics_t *metrics) {
    if (!net || !dataset || !metrics) return;
    if (start < 0 || end > dataset->num_samples || start >= end) return;

    memset(metrics, 0, sizeof(nn_regression_metrics_t));

    int nf = dataset->num_features;
    int no = dataset->num_outputs;
    int n = end - start;

    double *pred = (double *)malloc(no * sizeof(double));
    if (!pred) return;

    double sse = 0.0;
    double sae = 0.0;
    double smape = 0.0;
    double max_err = 0.0;
    double sum_err = 0.0;
    double sum_sq_err = 0.0;
    double y_mean = 0.0, y_tot = 0.0;

    /* Compute mean of true values */
    for (int i = start; i < end; i++) {
        y_mean += dataset->Y.data[i * no]; /* First output dimension */
    }
    y_mean /= n;

    for (int i = start; i < end; i++) {
        const double *x_row = dataset->X.data + i * nf;
        const double *y_row = dataset->Y.data + i * no;

        nn_network_forward(net, x_row, pred);

        double err = pred[0] - y_row[0]; /* For first output dimension */
        double abs_err = fabs(err);
        double sq_err = err * err;

        sum_err += err;
        sse += sq_err;
        sae += abs_err;
        sum_sq_err += sq_err;
        y_tot += (y_row[0] - y_mean) * (y_row[0] - y_mean);

        if (abs_err > max_err) max_err = abs_err;

        if (fabs(y_row[0]) > 1e-10) {
            smape += abs_err / fabs(y_row[0]);
        }
    }

    double mse = (n > 0) ? sse / n : 0.0;
    double rmse = sqrt(mse);
    double mae = (n > 0) ? sae / n : 0.0;
    double mape = (n > 0) ? (smape * 100.0 / n) : 0.0;
    double r2 = (y_tot > 1e-10) ? (1.0 - sse / y_tot) : 0.0;
    double bias = sum_err / n;

    /* Standard deviation of errors */
    double var_err = 0.0;
    for (int i = start; i < end; i++) {
        const double *x_row = dataset->X.data + i * nf;
        const double *y_row = dataset->Y.data + i * no;
        nn_network_forward(net, x_row, pred);
        double err = pred[0] - y_row[0];
        double d = err - bias;
        var_err += d * d;
    }
    double std_err = sqrt(var_err / n);

    /* Count parameters for AIC/BIC */
    int n_params = nn_count_parameters((nn_network_t *)net);

    metrics->rmse = rmse;
    metrics->mae = mae;
    metrics->mape = mape;
    metrics->r_squared = r2;
    metrics->mse = mse;
    metrics->max_error = max_err;
    metrics->std_error = std_err;
    metrics->bias = bias;
    metrics->adj_r_squared = 1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1);
    metrics->aic = nn_aic_criterion(n, sse, n_params);
    metrics->bic = nn_bic_criterion(n, sse, n_params);

    free(pred);
}

double nn_compute_r_squared(const double *y_true, const double *y_pred, int n) {
    if (!y_true || !y_pred || n <= 0) return 0.0;

    double y_bar = 0.0;
    for (int i = 0; i < n; i++) y_bar += y_true[i];
    y_bar /= n;

    double ss_res = 0.0, ss_tot = 0.0;
    for (int i = 0; i < n; i++) {
        double r = y_true[i] - y_pred[i];
        ss_res += r * r;
        double t = y_true[i] - y_bar;
        ss_tot += t * t;
    }

    if (ss_tot < 1e-15) return 1.0; /* Perfect fit when all y are equal */
    return 1.0 - ss_res / ss_tot;
}

/*===========================================================================
 * L4: Model Selection Criteria
 *===========================================================================*/

int nn_count_parameters(const nn_network_t *net) {
    if (!net) return 0;

    int total = 0;
    for (int l = 0; l < net->num_layers; l++) {
        total += net->layers[l]->input_size * net->layers[l]->output_size;
        total += net->layers[l]->output_size; /* biases */
    }
    return total;
}
