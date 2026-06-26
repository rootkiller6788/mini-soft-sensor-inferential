/**
 * @file nn_sensor_validation.c
 * @brief Soft sensor validation, performance monitoring, and industrial process models.
 *
 * Level: L4-L8 (Engineering Laws, Algorithms, Canonical Problems, Applications, Advanced)
 * Reference: Fortuna et al. (2007), Kadlec et al. (2009), Kaneko & Funatsu (2014)
 *            Skogestad (1997), Bailey & Ollis (1986), McAuley & MacGregor (1991)
 *
 * Each process data generator implements a distinct industrial unit operation
 * based on simplified first-principles models.
 */

#include "nn_sensor_validation.h"
#include "nn_training.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*===========================================================================
 * L4: Model Selection Criteria
 *===========================================================================*/

double nn_aic_criterion(int n_samples, double sse, int n_parameters) {
    /* AIC = n * ln(SSE/n) + 2*k
     * Valid when n >> k. For small n, use AICc. */
    if (n_samples <= 0 || n_parameters <= 0) return 1e100;

    double sigma2 = sse / n_samples;
    if (sigma2 < 1e-15) sigma2 = 1e-15;
    return n_samples * log(sigma2) + 2.0 * n_parameters;
}

double nn_bic_criterion(int n_samples, double sse, int n_parameters) {
    /* BIC = n * ln(SSE/n) + k * ln(n)
     * Schwarz criterion - heavier penalty for complexity. */
    if (n_samples <= 0 || n_parameters <= 0) return 1e100;

    double sigma2 = sse / n_samples;
    if (sigma2 < 1e-15) sigma2 = 1e-15;
    return n_samples * log(sigma2) + n_parameters * log((double)n_samples);
}

/*===========================================================================
 * L4-L6: Cross-Validation
 *===========================================================================*/

int nn_kfold_cross_validate(nn_network_t *net,
                             const double *X, const double *Y,
                             int n_samples, int n_features, int n_outputs,
                             int k, int max_epochs,
                             double *mean_val_rmse, double *std_val_rmse) {
    if (!net || !X || !Y || n_samples <= 0 || k < 2 || k > n_samples) {
        return -1;
    }

    int fold_size = n_samples / k;
    double *rmse_folds = (double *)malloc(k * sizeof(double));
    if (!rmse_folds) return -1;

    for (int fold = 0; fold < k; fold++) {
        /* Compute fold boundaries */
        int test_start = fold * fold_size;
        int test_end = (fold == k - 1) ? n_samples : (fold + 1) * fold_size;
        int n_test = test_end - test_start;
        int n_train = n_samples - n_test;

        /* Allocate fold training data */
        double *X_train = (double *)malloc(n_train * n_features * sizeof(double));
        double *Y_train = (double *)malloc(n_train * n_outputs * sizeof(double));
        double *X_test = (double *)malloc(n_test * n_features * sizeof(double));
        double *Y_test = (double *)malloc(n_test * n_outputs * sizeof(double));

        if (!X_train || !Y_train || !X_test || !Y_test) {
            free(X_train); free(Y_train); free(X_test); free(Y_test);
            free(rmse_folds);
            return -1;
        }

        /* Split data: training = all except fold, test = fold */
        int train_idx = 0, test_idx = 0;
        for (int i = 0; i < n_samples; i++) {
            if (i >= test_start && i < test_end) {
                memcpy(X_test + test_idx * n_features, X + i * n_features,
                       n_features * sizeof(double));
                memcpy(Y_test + test_idx * n_outputs, Y + i * n_outputs,
                       n_outputs * sizeof(double));
                test_idx++;
            } else {
                memcpy(X_train + train_idx * n_features, X + i * n_features,
                       n_features * sizeof(double));
                memcpy(Y_train + train_idx * n_outputs, Y + i * n_outputs,
                       n_outputs * sizeof(double));
                train_idx++;
            }
        }

        /* Create dataset from fold split */
        nn_dataset_t *ds = nn_dataset_create(X_train, Y_train, n_train,
                                               n_features, n_outputs, 0.9, 0.0);
        if (!ds) {
            free(X_train); free(Y_train); free(X_test); free(Y_test);
            free(rmse_folds);
            return -1;
        }
        /* Adjust validation split: use last 10% of training for validation */
        ds->val_start = (int)(n_train * 0.9);
        ds->val_end = n_train;
        ds->train_end = ds->val_start;

        /* Create fresh network with same architecture */
        nn_network_t *fold_net = nn_network_create(&net->arch);
        if (!fold_net) {
            nn_dataset_free(ds);
            free(X_train); free(Y_train); free(X_test); free(Y_test);
            free(rmse_folds);
            return -1;
        }

        /* Train on this fold */
        for (int e = 0; e < max_epochs; e++) {
            nn_network_train_epoch(fold_net, ds, e);
        }

        /* Evaluate on hold-out fold */
        nn_regression_metrics_t metrics;
        nn_dataset_t *test_ds = nn_dataset_create(X_test, Y_test, n_test,
                                                     n_features, n_outputs, 1.0, 0.0);
        if (test_ds) {
            nn_network_evaluate(fold_net, test_ds, 0, n_test, &metrics);
            rmse_folds[fold] = metrics.rmse;
            nn_dataset_free(test_ds);
        } else {
            rmse_folds[fold] = 0.0;
        }

        nn_network_free(fold_net);
        nn_dataset_free(ds);
        free(X_train); free(Y_train); free(X_test); free(Y_test);
    }

    /* Compute mean and std of fold RMSEs */
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < k; i++) {
        sum += rmse_folds[i];
        sum_sq += rmse_folds[i] * rmse_folds[i];
    }
    double mean = sum / k;
    double var = sum_sq / k - mean * mean;
    if (var < 0.0) var = 0.0;

    if (mean_val_rmse) *mean_val_rmse = mean;
    if (std_val_rmse) *std_val_rmse = sqrt(var);

    free(rmse_folds);
    return 0;
}

/*===========================================================================
 * L4-L6: Overfitting Detection
 *===========================================================================*/

int nn_detect_overfitting(const double *train_loss_history,
                           const double *val_loss_history,
                           int n_epochs, int patience) {
    if (!val_loss_history || n_epochs < patience + 1 || patience <= 0) {
        return 0;
    }

    /* Check if validation loss has been increasing for `patience` consecutive epochs */
    int increasing_streak = 0;
    for (int i = n_epochs - patience; i < n_epochs - 1; i++) {
        if (val_loss_history[i + 1] > val_loss_history[i]) {
            increasing_streak++;
        }
    }

    /* Also check that training loss is still decreasing */
    int train_decreasing = 0;
    if (train_loss_history) {
        train_decreasing = (train_loss_history[n_epochs - 1] < train_loss_history[n_epochs - patience]);
    }

    return (increasing_streak >= patience - 1) && (train_decreasing);
}

int nn_early_stopping_epoch(const double *val_loss_history, int n_epochs) {
    if (!val_loss_history || n_epochs <= 0) return 0;

    int best_epoch = 0;
    double best_loss = val_loss_history[0];

    for (int i = 1; i < n_epochs; i++) {
        if (val_loss_history[i] < best_loss) {
            best_loss = val_loss_history[i];
            best_epoch = i;
        }
    }

    return best_epoch;
}

/*===========================================================================
 * L4-L6: Residual Analysis
 *===========================================================================*/

void nn_residual_analysis(const double *y_true, const double *y_pred, int n,
                           double *residuals, double *bias, double *std_dev,
                           double *skewness, double *kurtosis) {
    if (!y_true || !y_pred || n <= 0) return;

    /* Compute residuals and basic stats */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double r = y_true[i] - y_pred[i];
        if (residuals) residuals[i] = r;
        sum += r;
    }
    double mean_r = sum / n;

    double sum_sq = 0.0, sum_cu = 0.0, sum_qu = 0.0;
    for (int i = 0; i < n; i++) {
        double r = y_true[i] - y_pred[i];
        double d = r - mean_r;
        double d2 = d * d;
        sum_sq += d2;
        sum_cu += d2 * d;
        sum_qu += d2 * d2;
    }

    double var = sum_sq / n;
    double sd = sqrt(var);

    /* Skewness: E[(r - mu)^3] / sigma^3 */
    double m3 = sum_cu / n;
    double skew = (sd > 1e-10) ? m3 / (sd * sd * sd) : 0.0;

    /* Excess kurtosis: E[(r - mu)^4] / sigma^4 - 3 */
    double m4 = sum_qu / n;
    double kurt = (sd > 1e-10) ? (m4 / (var * var)) - 3.0 : 0.0;

    if (bias) *bias = mean_r;
    if (std_dev) *std_dev = sd;
    if (skewness) *skewness = skew;
    if (kurtosis) *kurtosis = kurt;
}

double nn_durbin_watson(const double *residuals, int n) {
    if (!residuals || n < 2) return 2.0;

    double num = 0.0, den = 0.0;
    for (int i = 1; i < n; i++) {
        double d = residuals[i] - residuals[i - 1];
        num += d * d;
    }
    for (int i = 0; i < n; i++) {
        den += residuals[i] * residuals[i];
    }

    if (den < 1e-15) return 2.0;
    return num / den;
}

/*===========================================================================
 * L7: Performance Monitoring
 *===========================================================================*/

void soft_sensor_update_metrics(soft_sensor_t *sensor, double actual, double predicted) {
    if (!sensor) return;

    double err = actual - predicted;
    double sq_err = err * err;

    /* Update moving average MSE (window size = 100) */
    sensor->mse_moving_history[sensor->mse_history_idx] = sq_err;
    sensor->mse_history_idx = (sensor->mse_history_idx + 1) % 100;

    if (sensor->total_predictions < 100) {
        /* Cumulative average */
        sensor->mse_moving_average = (sensor->mse_moving_average * sensor->total_predictions + sq_err)
                                     / (sensor->total_predictions + 1);
    } else {
        /* Rolling window average */
        double sum = 0.0;
        for (int i = 0; i < 100; i++) {
            sum += sensor->mse_moving_history[i];
        }
        sensor->mse_moving_average = sum / 100.0;
    }

    /* Drift indicator: exponential weighted moving average */
    double alpha = 0.1;
    sensor->drift_indicator = alpha * fabs(err) + (1.0 - alpha) * sensor->drift_indicator;

    sensor->prediction_variance = (sensor->total_predictions > 1)
        ? (sensor->prediction_variance * (sensor->total_predictions - 1) + sq_err) / sensor->total_predictions
        : sq_err;
}

int soft_sensor_monitor_performance(soft_sensor_t *sensor, double actual,
                                     double predicted) {
    if (!sensor) return 0;

    soft_sensor_update_metrics(sensor, actual, predicted);

    /* Check if performance has degraded */
    double mse_threshold = 4.0 * sensor->val_rmse * sensor->val_rmse;
    if (sensor->val_rmse > 0.0 && sensor->mse_moving_average > mse_threshold) {
        sensor->needs_retraining = 1;
        return 1;
    }

    return 0;
}

int soft_sensor_detect_drift(soft_sensor_t *sensor, double threshold) {
    if (!sensor) return 0;

    /* CUSUM: S_t = max(0, S_{t-1} + e_t - delta)
     * With delta = RMSE_validation / 2 as reference */
    double delta = (sensor->val_rmse > 0.0) ? sensor->val_rmse / 2.0 : 0.01;

    /* We approximate CUSUM by comparing drift indicator to threshold */
    if (sensor->drift_indicator > threshold * delta) {
        return 1;
    }

    return 0;
}

int soft_sensor_compare_models(soft_sensor_t *sensor_a, soft_sensor_t *sensor_b,
                                const double *X_test, const double *Y_test,
                                int n_test, double *p_value) {
    if (!sensor_a || !sensor_b || !X_test || !Y_test || n_test <= 0) return 0;

    /* Compute squared errors for both models */
    double *err_a = (double *)malloc(n_test * sizeof(double));
    double *err_b = (double *)malloc(n_test * sizeof(double));
    double *output_a = (double *)malloc(sizeof(double));
    double *output_b = (double *)malloc(sizeof(double));
    if (!err_a || !err_b || !output_a || !output_b) {
        free(err_a); free(err_b); free(output_a); free(output_b);
        return 0;
    }

    int nf = sensor_a->dataset ? sensor_a->dataset->num_features : 5;

    for (int i = 0; i < n_test; i++) {
        soft_sensor_predict(sensor_a, X_test + i * nf, output_a);
        soft_sensor_predict(sensor_b, X_test + i * nf, output_b);
        err_a[i] = (output_a[0] - Y_test[i]) * (output_a[0] - Y_test[i]);
        err_b[i] = (output_b[0] - Y_test[i]) * (output_b[0] - Y_test[i]);
    }

    /* Paired differences */
    double sum_diff = 0.0, sum_sq_diff = 0.0;
    for (int i = 0; i < n_test; i++) {
        double d = err_a[i] - err_b[i];
        sum_diff += d;
        sum_sq_diff += d * d;
    }

    double mean_diff = sum_diff / n_test;
    double var_diff = sum_sq_diff / n_test - mean_diff * mean_diff;
    if (var_diff < 0.0) var_diff = 0.0;
    double se = sqrt(var_diff / n_test);
    if (se < 1e-15) se = 1e-15;
    double t_stat = mean_diff / se;

    /* Approximate p-value using standard normal (for large n)
     * Two-tailed test: p = 2 * (1 - Phi(|t|))
     * Use asymptotic approximation */
    double abs_t = fabs(t_stat);
    double p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs_t / sqrt(2.0))));
    if (p > 1.0) p = 1.0;
    if (p < 0.0) p = 0.0;

    if (p_value) *p_value = p;

    /* Model A better if mean_diff < 0 (smaller squared errors) and significant */
    int result = (mean_diff < -1e-10 && p < 0.05) ? 1 : 0;

    free(err_a); free(err_b); free(output_a); free(output_b);
    return result;
}

/*===========================================================================
 * L6-L7: Industrial Process Data Generators
 *
 * Each generator creates realistic simulation data for a specific
 * industrial process unit operation. These are simplified models
 * that capture key nonlinear relationships for soft sensor validation.
 *===========================================================================*/

/* Local RNG helpers for data generation (defined before use) */
static double nn_random_uniform_manual(unsigned int *seed) {
    unsigned int x = *seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *seed = x;
    return (double)(x & 0x7FFFFFFF) / 0x7FFFFFFF;
}

static double gaussian_noise_manual(unsigned int *seed) {
    double u1 = nn_random_uniform_manual(seed);
    double u2 = nn_random_uniform_manual(seed);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void nn_generate_distillation_data(double *X, double *Y, int n_samples, double noise_std) {
    /* Binary distillation column - simplified McCabe-Thiele model.
     * Inputs (5): reflux ratio, feed rate, feed composition, reboiler duty, feed temperature
     * Outputs (3): top composition, bottom composition, column pressure drop
     *
     * Key nonlinearities:
     * - log-mean of tray compositions
     * - temperature-composition relationship (Antoine equation)
     * - interaction between reflux and boil-up
     */
    unsigned int seed = 12345;

    for (int i = 0; i < n_samples; i++) {
        /* Generate realistic input ranges */
        double RR = 1.5 + 1.0 * nn_random_uniform_manual(&seed);      /* Reflux ratio: 1.5-2.5 */
        double F  = 90.0 + 20.0 * nn_random_uniform_manual(&seed);    /* Feed rate: 90-110 kg/h */
        double zF = 0.4 + 0.2 * nn_random_uniform_manual(&seed);      /* Feed composition: 0.4-0.6 */
        double QB = 800.0 + 200.0 * nn_random_uniform_manual(&seed);  /* Reboiler duty: 800-1000 kW */
        double TF = 80.0 + 40.0 * nn_random_uniform_manual(&seed);    /* Feed temp: 80-120 C */

        /* Fenske-Underwood-Gilliland shortcut model */
        double relative_volatility = 2.5 + 0.3 * (zF - 0.5);
        /* Minimum stages: N_min ~ 15.0 + 5*(RR-1.5), implicitly used in composition model */

        /* Top composition: increases with reflux ratio, relative volatility */
        double xD = 0.9 + 0.05 * (RR - 1.5) + 0.03 * (relative_volatility - 2.5)
                     - 0.02 * (F - 100.0) / 10.0;
        xD = (xD > 0.995) ? 0.995 : (xD < 0.5) ? 0.5 : xD;

        /* Bottom composition: decreases with reboiler duty, increases with feed composition */
        double xB = 0.05 + 0.1 * zF - 0.02 * (QB - 800.0) / 200.0;
        xB = (xB > 0.5) ? 0.5 : (xB < 0.005) ? 0.005 : xB;

        /* Pressure drop: depends on vapor load (~reboiler duty) */
        double dP = 0.5 + 0.3 * (QB - 800.0) / 200.0 + 0.1 * RR;

        /* Add measurement noise */
        X[i * 5 + 0] = RR + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 1] = F  + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 2] = zF + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 3] = QB + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 4] = TF + noise_std * gaussian_noise_manual(&seed);

        Y[i * 3 + 0] = xD + noise_std * 0.01 * gaussian_noise_manual(&seed);
        Y[i * 3 + 1] = xB + noise_std * 0.01 * gaussian_noise_manual(&seed);
        Y[i * 3 + 2] = dP + noise_std * 0.05 * gaussian_noise_manual(&seed);
    }
}

void nn_generate_fermentation_data(double *X, double *Y, int n_samples, double noise_std) {
    /* Fed-batch fermentation process - Monod kinetics.
     * Inputs (5): temperature, pH, dissolved oxygen, agitation rate, substrate feed rate
     * Outputs (3): biomass concentration, product concentration, substrate concentration
     *
     * Monod kinetics: mu = mu_max * S / (K_s + S)
     * Growth: dX/dt = mu * X
     * Product: dP/dt = Y_{p/x} * mu * X
     * Substrate: dS/dt = -mu * X / Y_{x/s} + F/V * (S_f - S)
     */
    unsigned int seed = 54321;

    for (int i = 0; i < n_samples; i++) {
        double T  = 30.0 + 7.0 * nn_random_uniform_manual(&seed);   /* Temp: 30-37 C */
        double pH = 6.5 + 1.0 * nn_random_uniform_manual(&seed);    /* pH: 6.5-7.5 */
        double DO = 20.0 + 60.0 * nn_random_uniform_manual(&seed);  /* DO: 20-80% */
        double N  = 200.0 + 600.0 * nn_random_uniform_manual(&seed);/* Agitation: 200-800 rpm */
        double F  = 0.5 + 1.5 * nn_random_uniform_manual(&seed);    /* Feed: 0.5-2.0 L/h */

        /* Temperature-dependent mu_max (Arrhenius-like) */
        double T_opt = 35.0;
        double mu_max = 0.4 * exp(-0.05 * (T - T_opt) * (T - T_opt));

        /* pH and DO inhibition */
        double pH_factor = exp(-5.0 * (pH - 7.0) * (pH - 7.0));
        double DO_factor = (DO > 10.0) ? 1.0 : DO / 10.0;
        double effective_mu = mu_max * pH_factor * DO_factor;

        /* Substrate inhibition (Andrews model) */
        double K_s = 0.5;
        double K_i = 50.0;
        double S_input = 10.0 + 30.0 * F; /* Substrate available */
        double mu = effective_mu * S_input / (K_s + S_input + S_input * S_input / K_i);

        /* Biomass */
        double X_bio = 5.0 + 20.0 * mu / mu_max;
        X_bio *= (0.9 + 0.1 * N / 800.0);

        /* Product (partially growth-associated - Luedeking-Piret model) */
        double Y_px = 0.5;
        double product = Y_px * X_bio * (0.7 + 0.3 * mu / mu_max);

        /* Residual substrate */
        double substrate = S_input - X_bio / 0.5 - product / Y_px;
        if (substrate < 0.1) substrate = 0.1;

        /* Add noise */
        X[i * 5 + 0] = T + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 1] = pH + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 2] = DO + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 3] = N + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 4] = F + noise_std * gaussian_noise_manual(&seed);

        Y[i * 3 + 0] = X_bio + noise_std * 0.5 * gaussian_noise_manual(&seed);
        Y[i * 3 + 1] = product + noise_std * 0.3 * gaussian_noise_manual(&seed);
        Y[i * 3 + 2] = substrate + noise_std * 0.2 * gaussian_noise_manual(&seed);
    }
}

void nn_generate_polymerization_data(double *X, double *Y, int n_samples, double noise_std) {
    /* Gas-phase ethylene polymerization in fluidized bed reactor.
     * Based on simplified Ziegler-Natta kinetics.
     *
     * Inputs (5): reactor temperature, partial pressure, catalyst feed,
     *             H2/C2 ratio, residence time
     * Outputs (3): melt index (MI), density, production rate
     *
     * Melt index: log(MI) = a0 + a1*log(H2/C2) + a2/T + a3*catalyst
     * Density: rho = f(comonomer incorporation, temperature)
     */
    unsigned int seed = 98765;

    for (int i = 0; i < n_samples; i++) {
        double T  = 75.0 + 15.0 * nn_random_uniform_manual(&seed);   /* Temp: 75-90 C */
        double P  = 18.0 + 4.0 * nn_random_uniform_manual(&seed);    /* Pressure: 18-22 bar */
        double cat = 0.5 + 1.5 * nn_random_uniform_manual(&seed);    /* Catalyst: 0.5-2.0 g/h */
        double H2C2 = 0.1 + 0.3 * nn_random_uniform_manual(&seed);   /* H2/C2: 0.1-0.4 */
        double tau = 2.0 + 2.0 * nn_random_uniform_manual(&seed);    /* Residence: 2-4 h */

        /* Melt index model (Arrhenius + H2 response) */
        double logMI = 2.0 - 800.0 / (T + 273.15) + 1.5 * log(H2C2) - 0.5 * tau;
        double MI = exp(logMI);
        MI = (MI < 0.1) ? 0.1 : (MI > 50.0) ? 50.0 : MI;

        /* Density model (depends on temperature and H2 concentration) */
        double density = 0.94 - 0.01 * (T - 80.0) / 15.0 + 0.005 * log(H2C2);
        density = (density < 0.91) ? 0.91 : (density > 0.97) ? 0.97 : density;

        /* Production rate: catalyst activity * residence time */
        double prod_rate = cat * 500.0 * exp(-0.1 * (T - 80.0) * (T - 80.0)) * (1.0 - exp(-tau));
        prod_rate *= P / 20.0;

        X[i * 5 + 0] = T + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 1] = P + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 2] = cat + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 3] = H2C2 + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 4] = tau + noise_std * gaussian_noise_manual(&seed);

        Y[i * 3 + 0] = MI + noise_std * 0.5 * gaussian_noise_manual(&seed);
        Y[i * 3 + 1] = density + noise_std * 0.002 * gaussian_noise_manual(&seed);
        Y[i * 3 + 2] = prod_rate + noise_std * 10.0 * gaussian_noise_manual(&seed);
    }
}

void nn_generate_cement_kiln_data(double *X, double *Y, int n_samples, double noise_std) {
    /* Rotary cement kiln - clinker quality soft sensor.
     * Based on simplified heat and mass balance.
     *
     * Inputs (5): kiln temperature (burning zone), kiln torque,
     *             O2% at preheater exit, NOx at kiln inlet, raw meal feed rate
     * Outputs (3): free lime (fCaO), liter weight, clinker quality index
     */
    unsigned int seed = 24680;

    for (int i = 0; i < n_samples; i++) {
        double T_kiln = 1400.0 + 100.0 * nn_random_uniform_manual(&seed);  /* 1400-1500 C */
        double torque = 40.0 + 20.0 * nn_random_uniform_manual(&seed);     /* 40-60% */
        double O2 = 1.5 + 3.5 * nn_random_uniform_manual(&seed);           /* 1.5-5% */
        double NOx = 800.0 + 400.0 * nn_random_uniform_manual(&seed);      /* 800-1200 ppm */
        double feed = 200.0 + 50.0 * nn_random_uniform_manual(&seed);      /* 200-250 t/h */

        /* Free lime: lower at higher temperature and longer residence */
        double fCaO = 3.0 - 2.5 * (T_kiln - 1400.0) / 100.0
                      + 0.5 * (feed - 225.0) / 25.0
                      - 0.3 * (O2 - 3.0);
        fCaO = (fCaO < 0.5) ? 0.5 : (fCaO > 3.0) ? 3.0 : fCaO;

        /* Liter weight: related to burning zone temperature and kiln speed */
        double liter_wt = 1200.0 + 100.0 * (T_kiln - 1400.0) / 100.0
                          + 20.0 * (torque - 50.0) / 10.0;

        /* Quality index (C3S content proxy): free lime inversely related */
        double quality = 60.0 - 5.0 * fCaO + 5.0 * (T_kiln - 1400.0) / 100.0
                         + 0.01 * NOx;

        X[i * 5 + 0] = T_kiln + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 1] = torque + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 2] = O2 + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 3] = NOx + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 4] = feed + noise_std * gaussian_noise_manual(&seed);

        Y[i * 3 + 0] = fCaO + noise_std * 0.1 * gaussian_noise_manual(&seed);
        Y[i * 3 + 1] = liter_wt + noise_std * 10.0 * gaussian_noise_manual(&seed);
        Y[i * 3 + 2] = quality + noise_std * 1.0 * gaussian_noise_manual(&seed);
    }
}

void nn_generate_fcc_data(double *X, double *Y, int n_samples, double noise_std) {
    /* Fluid Catalytic Cracking (FCC) unit - gasoline yield soft sensor.
     *
     * Inputs (5): riser outlet temperature, regenerator dense phase temperature,
     *             catalyst circulation rate, feed preheat temperature, cat/oil ratio
     * Outputs (3): gasoline yield, light cycle oil yield, conversion
     */
    unsigned int seed = 13579;

    for (int i = 0; i < n_samples; i++) {
        double T_riser = 520.0 + 30.0 * nn_random_uniform_manual(&seed);      /* 520-550 C */
        double T_regen = 680.0 + 40.0 * nn_random_uniform_manual(&seed);      /* 680-720 C */
        double cat_circ = 30.0 + 10.0 * nn_random_uniform_manual(&seed);      /* 30-40 t/min */
        double T_feed = 200.0 + 50.0 * nn_random_uniform_manual(&seed);       /* 200-250 C */
        double cor = 6.0 + 2.0 * nn_random_uniform_manual(&seed);             /* 6-8 */

        /* Conversion: function of temperature and C/O ratio */
        double conversion = 70.0 + 15.0 * (T_riser - 520.0) / 30.0
                            + 5.0 * (cor - 7.0)
                            - 2.0 * (T_feed - 225.0) / 25.0;
        conversion = (conversion < 60.0) ? 60.0 : (conversion > 85.0) ? 85.0 : conversion;

        /* Gasoline yield: peaks around 75% conversion then drops (overcracking) */
        double conv_opt = 75.0;
        double gasoline = 50.0 - 0.5 * (conversion - conv_opt) * (conversion - conv_opt) / 100.0
                          + 0.2 * (cat_circ - 35.0);

        /* LCO yield: decreases with conversion */
        double lco = 20.0 - 0.5 * (conversion - 70.0);

        X[i * 5 + 0] = T_riser + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 1] = T_regen + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 2] = cat_circ + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 3] = T_feed + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 4] = cor + noise_std * gaussian_noise_manual(&seed);

        Y[i * 3 + 0] = gasoline + noise_std * 1.0 * gaussian_noise_manual(&seed);
        Y[i * 3 + 1] = lco + noise_std * 0.5 * gaussian_noise_manual(&seed);
        Y[i * 3 + 2] = conversion + noise_std * 0.5 * gaussian_noise_manual(&seed);
    }
}

void nn_generate_blast_furnace_data(double *X, double *Y, int n_samples, double noise_std) {
    /* Blast furnace hot metal quality prediction.
     *
     * Inputs (5): blast temperature, blast volume, oxygen enrichment,
     *             pulverized coal injection rate, burden descent speed
     * Outputs (3): hot metal temperature, silicon content, sulfur content
     */
    unsigned int seed = 75319;

    for (int i = 0; i < n_samples; i++) {
        double T_blast = 1100.0 + 100.0 * nn_random_uniform_manual(&seed);    /* 1100-1200 C */
        double V_blast = 4000.0 + 1000.0 * nn_random_uniform_manual(&seed);   /* 4000-5000 Nm3/min */
        double O2 = 2.0 + 3.0 * nn_random_uniform_manual(&seed);              /* 2-5% enrichment */
        double PCI = 100.0 + 50.0 * nn_random_uniform_manual(&seed);          /* 100-150 kg/tHM */
        double burden = 80.0 + 20.0 * nn_random_uniform_manual(&seed);        /* 80-100 mm/min */

        /* Hot metal temperature */
        double HMT = 1480.0 + 30.0 * (T_blast - 1150.0) / 50.0
                     + 10.0 * (O2 - 3.5)
                     - 15.0 * (PCI - 125.0) / 25.0;  /* PCI cools the hearth */

        /* Silicon content: indicator of thermal state, increases with temperature */
        double Si = 0.5 + 0.3 * (HMT - 1480.0) / 30.0
                    + 0.1 * (T_blast - 1150.0) / 50.0
                    - 0.05 * (burden - 90.0) / 10.0;
        Si = (Si < 0.2) ? 0.2 : (Si > 1.2) ? 1.2 : Si;

        /* Sulfur content: decreases with higher temperature and slag basicity */
        double S = 0.03 - 0.01 * (HMT - 1480.0) / 30.0
                   + 0.005 * (PCI - 125.0) / 25.0;
        S = (S < 0.01) ? 0.01 : (S > 0.06) ? 0.06 : S;

        X[i * 5 + 0] = T_blast + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 1] = V_blast + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 2] = O2 + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 3] = PCI + noise_std * gaussian_noise_manual(&seed);
        X[i * 5 + 4] = burden + noise_std * gaussian_noise_manual(&seed);

        Y[i * 3 + 0] = HMT + noise_std * 5.0 * gaussian_noise_manual(&seed);
        Y[i * 3 + 1] = Si + noise_std * 0.05 * gaussian_noise_manual(&seed);
        Y[i * 3 + 2] = S + noise_std * 0.005 * gaussian_noise_manual(&seed);
    }
}

/*===========================================================================
 * L7: Industrial Deployment
 *===========================================================================*/

void nn_industrial_sensor_init(nn_industrial_soft_sensor_t *app,
                                soft_sensor_t *sensor,
                                const char *plant,
                                const char *dcs_system,
                                const char *historian) {
    if (!app) return;

    memset(app, 0, sizeof(nn_industrial_soft_sensor_t));

    app->sensor = sensor;
    if (plant) strncpy(app->plant_location, plant, 63);
    if (dcs_system) strncpy(app->dcs_system, dcs_system, 31);
    if (historian) strncpy(app->historian_system, historian, 31);

    app->update_interval_ms = 1000;
    app->performance_threshold = 1.0;
    app->retraining_threshold = 2.0;

    if (sensor && sensor->is_trained) {
        snprintf(app->application_name, 127, "NN Soft Sensor: %s at %s",
                 sensor->quality_variable, plant);
    }
}
