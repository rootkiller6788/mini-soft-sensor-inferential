/**
 * @file nn_sensor_types.h
 * @brief Core type definitions for Neural Network Soft Sensor (Inferential Sensor).
 *
 * Level: L1 Definitions + L3 Engineering Structures
 * Reference: Bishop, "Neural Networks for Pattern Recognition" (1995)
 *            Goodfellow, Bengio, Courville, "Deep Learning" (2016), Ch.6
 *            Kadlec, Gabrys, Strandt, "Data-driven Soft Sensors..." Comp.Chem.Eng. (2009)
 *            Fortuna, Graziani, Rizzo, Xibilia, "Soft Sensors for Monitoring..." (2007)
 *
 * A soft sensor (inferential sensor) estimates hard-to-measure process variables
 * (quality variables) from easily-measured ones using a mathematical model.
 * Neural networks are universal function approximators ideal for this task.
 */

#ifndef NN_SENSOR_TYPES_H
#define NN_SENSOR_TYPES_H

#include <stdint.h>
#include <stddef.h>

/*===========================================================================
 * L1: Core Definitions - Activation Functions
 *===========================================================================*/

/**
 * @brief Activation function types for neural network layers.
 *
 * SIGMOID:  sigma(x) = 1/(1+exp(-x)), range (0,1)
 * TANH:     tanh(x), range (-1,1)
 * RELU:     max(0, x), range [0, inf)
 * LEAKY_RELU: max(alpha*x, x), alpha=0.01 typical
 * ELU:      x if x>0 else alpha*(exp(x)-1)
 * SWISH:    x * sigmoid(beta*x), self-gated (Ramachandran et al. 2017)
 * LINEAR:   f(x) = x, for regression output layer
 * SOFTMAX:  exp(x_i) / sum(exp(x_j)), for classification
 * GELU:     x * Phi(x), Gaussian Error Linear Unit (Hendrycks & Gimpel 2016)
 */
typedef enum {
    NN_ACTIVATION_SIGMOID     = 0,
    NN_ACTIVATION_TANH        = 1,
    NN_ACTIVATION_RELU        = 2,
    NN_ACTIVATION_LEAKY_RELU  = 3,
    NN_ACTIVATION_ELU         = 4,
    NN_ACTIVATION_SWISH       = 5,
    NN_ACTIVATION_LINEAR      = 6,
    NN_ACTIVATION_SOFTMAX     = 7,
    NN_ACTIVATION_GELU        = 8
} nn_activation_type_t;

/*===========================================================================
 * L1: Core Definitions - Loss Functions
 *===========================================================================*/

/**
 * @brief Loss (cost) function types for neural network training.
 *
 * MSE:  Mean Squared Error, L = (1/N)*sum((y_pred - y_true)^2)
 * MAE:  Mean Absolute Error, L = (1/N)*sum(|y_pred - y_true|)
 * RMSE: Root Mean Squared Error (sqrt of MSE)
 * HUBER: Huber loss, smooth L1, robust to outliers
 *        L = 0.5*(y_pred-y_true)^2 for |y_pred-y_true|<=delta
 *            delta*(|y_pred-y_true| - 0.5*delta) otherwise
 * MAPE: Mean Absolute Percentage Error, for interpretability
 * CROSS_ENTROPY: For classification tasks
 */
typedef enum {
    NN_LOSS_MSE           = 0,
    NN_LOSS_MAE           = 1,
    NN_LOSS_RMSE          = 2,
    NN_LOSS_HUBER         = 3,
    NN_LOSS_MAPE          = 4,
    NN_LOSS_CROSS_ENTROPY = 5
} nn_loss_type_t;

/*===========================================================================
 * L1: Core Definitions - Optimizer Types
 *===========================================================================*/

/**
 * @brief Gradient-based optimizer types.
 *
 * SGD:        Stochastic Gradient Descent (Robbins & Monro 1951)
 * MOMENTUM:   SGD with momentum (Polyak 1964)
 * NESTEROV:   Nesterov accelerated gradient (Nesterov 1983)
 * ADAGRAD:    Adaptive gradient (Duchi et al. 2011)
 * RMSPROP:    RMSprop (Tieleman & Hinton 2012)
 * ADAM:       Adam - Adaptive Moment Estimation (Kingma & Ba 2014)
 * ADAMAX:     Adam with infinity norm
 * NADAM:      Adam with Nesterov momentum (Dozat 2016)
 */
typedef enum {
    NN_OPTIMIZER_SGD       = 0,
    NN_OPTIMIZER_MOMENTUM  = 1,
    NN_OPTIMIZER_NESTEROV  = 2,
    NN_OPTIMIZER_ADAGRAD   = 3,
    NN_OPTIMIZER_RMSPROP   = 4,
    NN_OPTIMIZER_ADAM      = 5,
    NN_OPTIMIZER_ADAMAX    = 6,
    NN_OPTIMIZER_NADAM     = 7
} nn_optimizer_type_t;

/*===========================================================================
 * L1: Core Definitions - Regularization Types
 *===========================================================================*/

/**
 * @brief Regularization methods to prevent overfitting.
 *
 * NONE:          No regularization
 * L1:            Lasso - sum of absolute weights, induces sparsity
 * L2:            Ridge - sum of squared weights, weight decay
 * ELASTIC_NET:   Combination of L1 and L2
 * DROPOUT:       Randomly zero out neurons during training (Srivastava et al. 2014)
 * EARLY_STOPPING: Stop training when validation loss increases
 * BATCH_NORM:    Normalize layer inputs (Ioffe & Szegedy 2015)
 */
typedef enum {
    NN_REG_NONE          = 0,
    NN_REG_L1            = 1,
    NN_REG_L2            = 2,
    NN_REG_ELASTIC_NET   = 3,
    NN_REG_DROPOUT       = 4,
    NN_REG_EARLY_STOPPING = 5,
    NN_REG_BATCH_NORM    = 6
} nn_regularization_type_t;

/*===========================================================================
 * L1: Core Definitions - Normalization Types
 *===========================================================================*/

/**
 * @brief Data normalization methods for input preprocessing.
 *
 * MINMAX:     x_norm = (x - min) / (max - min), range [0,1]
 * ZSCORE:     x_norm = (x - mean) / std, zero mean unit variance
 * ROBUST:     x_norm = (x - median) / IQR, robust to outliers
 * DECIMAL:    x_norm = x / 10^j where j = ceil(log10(max(|x|)))
 */
typedef enum {
    NN_NORM_MINMAX  = 0,
    NN_NORM_ZSCORE  = 1,
    NN_NORM_ROBUST  = 2,
    NN_NORM_DECIMAL = 3
} nn_normalization_type_t;

/*===========================================================================
 * L1: Core Definitions - Inferential Sensor Types
 *===========================================================================*/

/**
 * @brief Soft sensor (inferential sensor) model type.
 *
 * STATIC:    y = f(x), no time dependence; steady-state estimation
 * DYNAMIC:   y(t) = f(x(t), x(t-1), ..., y(t-1), ...), includes history
 * HYBRID:    Combines first-principles model with neural network correction
 * ENSEMBLE:  Weighted combination of multiple models for robustness
 * ADAPTIVE:  Online learning, model updates with new data
 */
typedef enum {
    SENSOR_TYPE_STATIC   = 0,
    SENSOR_TYPE_DYNAMIC  = 1,
    SENSOR_TYPE_HYBRID   = 2,
    SENSOR_TYPE_ENSEMBLE = 3,
    SENSOR_TYPE_ADAPTIVE = 4
} soft_sensor_type_t;

/*===========================================================================
 * L1: Core Definitions - Layer Type
 *===========================================================================*/

/**
 * @brief Neural network layer type.
 */
typedef enum {
    LAYER_DENSE      = 0,
    LAYER_INPUT      = 1,
    LAYER_OUTPUT     = 2,
    LAYER_DROPOUT    = 3
} nn_layer_type_t;

/*===========================================================================
 * L3: Engineering Structures - Matrix and Vector Types
 *===========================================================================*/

/**
 * @brief 2D matrix structure for neural network weights and data.
 */
typedef struct {
    double *data;
    int     rows;
    int     cols;
    int     stride;    /* For sub-matrix views: real stride in data array */
} nn_matrix_t;

/**
 * @brief 1D vector structure for biases and activations.
 */
typedef struct {
    double *data;
    int     length;
} nn_vector_t;

/*===========================================================================
 * L1-L3: Neural Network Layer
 *===========================================================================*/

/**
 * @brief Single layer in a feedforward neural network.
 *
 * Computation: a = activation(W * input + b)
 * where W is weights matrix (output_size x input_size) and b is bias vector.
 */
typedef struct nn_layer {
    nn_matrix_t          weights;          /* W: output_size x input_size */
    nn_vector_t          biases;           /* b: output_size */
    nn_vector_t          activations;      /* a: output of this layer */
    nn_vector_t          weighted_sum;     /* z = Wx + b before activation */
    nn_activation_type_t activation_type;
    int                  input_size;
    int                  output_size;
    double               dropout_rate;     /* For dropout regularization */
    struct nn_layer     *prev;             /* Previous layer in network */
    struct nn_layer     *next;             /* Next layer in network */

    /* Training state - gradients */
    nn_matrix_t dW;                       /* Gradient w.r.t. weights */
    nn_vector_t db;                       /* Gradient w.r.t. biases */
    nn_vector_t delta;                    /* Error signal for backprop */
} nn_layer_t;

/*===========================================================================
 * L1-L3: Neural Network Architecture
 *===========================================================================*/

/**
 * @brief Neural network architecture descriptor.
 */
typedef struct {
    int                  num_layers;       /* Including input and output */
    int                 *layer_sizes;      /* Array of neurons per layer */
    nn_activation_type_t hidden_activation;/* Default hidden layer activation */
    nn_activation_type_t output_activation;/* Output layer activation */
    double               dropout_rate;     /* Default dropout probability */
} nn_architecture_t;

/*===========================================================================
 * L1-L3: Complete Neural Network
 *===========================================================================*/

/**
 * @brief Complete feedforward neural network.
 */
typedef struct {
    /* Architecture */
    nn_architecture_t arch;

    /* Layers */
    nn_layer_t **layers;       /* Array of layer pointers, layers[0] = input */
    int          num_layers;

    /* Training configuration */
    nn_loss_type_t          loss_type;
    nn_optimizer_type_t     optimizer_type;
    nn_regularization_type_t regularization_type;
    double                   learning_rate;
    double                   momentum_coeff;   /* For momentum optimizer */
    double                   l1_lambda;        /* L1 regularization strength */
    double                   l2_lambda;        /* L2 regularization strength */
    double                   huber_delta;      /* Delta for Huber loss */
    double                   leaky_relu_alpha; /* For Leaky ReLU */

    /* Adam optimizer state (Kingma & Ba 2014) */
    double beta1;            /* First moment decay (default 0.9) */
    double beta2;            /* Second moment decay (default 0.999) */
    double epsilon;          /* Numerical stability (default 1e-8) */
    int    adam_t;           /* Time step counter */

    /* Training history */
    double  *train_loss_history;
    double  *val_loss_history;
    int      history_capacity;
    int      history_count;
    int      total_epochs;

    /* State */
    int   is_trained;
    int   is_training;       /* Boolean: in training mode (affects dropout) */
} nn_network_t;

/*===========================================================================
 * L1-L3: Soft Sensor Control Data
 *===========================================================================*/

/**
 * @brief Data normalization parameters.
 */
typedef struct {
    nn_normalization_type_t type;
    double *mean;
    double *std;
    double *min;
    double *max;
    double *median;
    double *iqr;
    int     num_features;
} nn_normalization_t;

/**
 * @brief Dataset for supervised learning of soft sensor.
 */
typedef struct {
    nn_matrix_t X;           /* Input features: n_samples x n_features */
    nn_matrix_t Y;           /* Target values: n_samples x n_outputs */
    int         num_samples;
    int         num_features;
    int         num_outputs;

    /* Data splits */
    int         train_start;
    int         train_end;
    int         val_start;
    int         val_end;
    int         test_start;
    int         test_end;

    nn_normalization_t *X_norm;  /* Input normalization parameters */
    nn_normalization_t *Y_norm;  /* Output normalization parameters */
    int                  is_normalized;
} nn_dataset_t;

/*===========================================================================
 * L1-L3: Soft Sensor Complete
 *===========================================================================*/

/**
 * @brief Complete soft sensor (inferential sensor) based on neural network.
 *
 * This is the top-level structure for a neural network soft sensor
 * deployed in an industrial process environment.
 */
typedef struct {
    /* Identification */
    char               name[64];
    char               process_unit[64];
    char               quality_variable[64];
    char               unit[16];

    /* Core model */
    nn_network_t      *network;
    nn_dataset_t      *dataset;
    soft_sensor_type_t sensor_type;

    /* Dynamic sensor - lag/delay parameters */
    int                time_lags;          /* Number of historical samples */
    double             sampling_period_sec;
    double            *input_buffer;       /* Ring buffer for dynamic inputs */

    /* Training results */
    double             train_rmse;
    double             val_rmse;
    double             test_rmse;
    double             train_mae;
    double             val_mae;
    double             test_mae;
    double             r_squared;

    /* Online performance monitoring */
    double             prediction_variance;
    double             drift_indicator;
    double             mse_moving_average;
    double             mse_moving_history[100];
    int                mse_history_idx;
    int                total_predictions;

    /* Status */
    int                is_trained;
    int                is_deployed;
    int                needs_retraining;
} soft_sensor_t;

/*===========================================================================
 * L3: Engineering Structures - Training Configuration
 *===========================================================================*/

/**
 * @brief Complete training configuration.
 */
typedef struct {
    nn_optimizer_type_t     optimizer;
    nn_loss_type_t          loss;
    nn_regularization_type_t regularization;
    double                   learning_rate;
    double                   l1_lambda;
    double                   l2_lambda;
    double                   momentum;
    double                   dropout_rate;
    double                   huber_delta;

    /* Adam-specific */
    double beta1;
    double beta2;
    double epsilon;

    /* Training parameters */
    int    max_epochs;
    int    batch_size;
    int    early_stopping_patience;
    double target_loss;

    /* Learning rate schedule */
    double lr_decay;
    int    lr_decay_steps;

    /* Verbosity */
    int    verbose;
    int    print_every;
} nn_training_config_t;

/*===========================================================================
 * L4-L6: Model Validation Metrics
 *===========================================================================*/

/**
 * @brief Comprehensive regression metrics for soft sensor validation.
 */
typedef struct {
    double rmse;          /* Root Mean Squared Error */
    double mae;           /* Mean Absolute Error */
    double mape;          /* Mean Absolute Percentage Error */
    double r_squared;     /* Coefficient of determination R^2 */
    double adj_r_squared; /* Adjusted R^2 (accounting for model complexity) */
    double mse;           /* Mean Squared Error */
    double max_error;     /* Maximum absolute error */
    double std_error;     /* Standard deviation of errors */
    double bias;          /* Mean error (systematic bias) */
    double aic;           /* Akaike Information Criterion */
    double bic;           /* Bayesian Information Criterion */
} nn_regression_metrics_t;

/*===========================================================================
 * L5: Learning Rate Schedules
 *===========================================================================*/

/**
 * @brief Learning rate schedule type.
 */
typedef enum {
    LR_SCHEDULE_CONSTANT    = 0,
    LR_SCHEDULE_STEP        = 1,
    LR_SCHEDULE_EXPONENTIAL = 2,
    LR_SCHEDULE_INVERSE     = 3,
    LR_SCHEDULE_COSINE      = 4,
    LR_SCHEDULE_CYCLIC      = 5
} nn_lr_schedule_type_t;

/*===========================================================================
 * L8: Ensemble Neural Network
 *===========================================================================*/

/**
 * @brief Ensemble of neural networks for robust prediction.
 *
 * Reference: Hansen & Salamon, "Neural Network Ensembles," IEEE TPAMI (1990)
 *            Krogh & Vedelsby, "Neural Network Ensembles..." NIPS (1995)
 */
typedef struct {
    nn_network_t  **networks;
    double         *weights;       /* Ensemble member weights */
    int             num_members;
    double          ensemble_bias;
    double          ensemble_variance;
    int             bagging_samples; /* Bootstrap sample size for bagging */
} nn_ensemble_t;

/*===========================================================================
 * L8: Bayesian Neural Network Approximation (Monte Carlo Dropout)
 *===========================================================================*/

/**
 * @brief Bayesian approximation using MC Dropout (Gal & Ghahramani 2016).
 *
 * Performs multiple stochastic forward passes with dropout active
 * to estimate prediction uncertainty.
 */
typedef struct {
    nn_network_t  *network;
    int            num_samples;     /* Number of MC samples */
    double        *mean_predictions;
    double        *variance_predictions;
    double        *lower_bound;     /* 95% confidence lower bound */
    double        *upper_bound;     /* 95% confidence upper bound */
    double         epistemic_uncertainty;
    double         aleatoric_uncertainty;
} nn_mc_dropout_t;

/*===========================================================================
 * L7-L8: Industrial Soft Sensor Application State
 *===========================================================================*/

/**
 * @brief State of an industrial soft sensor application.
 * Reference: Fortuna et al. "Soft Sensors for Monitoring and Control..." (2007)
 */
typedef struct {
    char     application_name[128];
    char     plant_location[64];
    char     dcs_system[32];        /* e.g., "Honeywell Experion", "Siemens PCS7" */
    char     historian_system[32];   /* e.g., "OSIsoft PI", "AspenTech IP.21" */

    /* Process data interface */
    int      num_input_tags;
    char   **input_tag_names;
    int      num_output_tags;
    char   **output_tag_names;

    /* OPC connectivity */
    char     opc_server[128];
    int      update_interval_ms;

    /* Model management */
    soft_sensor_t *sensor;
    double         performance_threshold;  /* Below this RMSE, model is valid */
    double         retraining_threshold;   /* Above this, trigger retraining */
    int            auto_retraining;

    /* Deployment status */
    int            is_online;
    int            is_validated;
    int            validation_samples_needed;
    double         uptime_hours;
    int            fault_count;
} nn_industrial_soft_sensor_t;

/*===========================================================================
 * L9: Digital Twin Interface
 *===========================================================================*/

/**
 * @brief Soft sensor as part of a digital twin.
 */
typedef struct {
    soft_sensor_t    *physical_twin_sensor;
    nn_network_t     *process_simulator;
    double            synchronization_error;
    int               sync_frequency_hz;
    double            last_sync_timestamp;
    int               is_synced;
} nn_digital_twin_interface_t;

#endif /* NN_SENSOR_TYPES_H */
