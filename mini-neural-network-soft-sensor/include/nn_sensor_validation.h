/**
 * @file nn_sensor_validation.h
 * @brief Soft sensor model validation, performance monitoring, and maintenance.
 *
 * Level: L4 Engineering Laws, L6 Canonical Problems, L7-L8 Applications
 * Reference: Fortuna et al., "Soft Sensors for Monitoring and Control..." (2007)
 *            Kadlec, Gabrys, Strandt, "Data-driven Soft Sensors..." Comp.Chem.Eng. (2009)
 *            Gonzaga et al., "ANN-based soft sensor for real-time process monitoring..."
 *            Kaneko & Funatsu, "Adaptive Soft Sensor..." AIChE J. (2014)
 *            Souza et al., "Soft sensor for industrial polyethylene process..." (2016)
 */

#ifndef NN_SENSOR_VALIDATION_H
#define NN_SENSOR_VALIDATION_H

#include "nn_sensor_types.h"
#include "nn_soft_sensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
 * L4: Model Selection Criteria
 *===========================================================================*/

/**
 * @brief Compute Akaike Information Criterion (AIC).
 *
 * AIC = n * ln(SSE/n) + 2 * k
 * where n = number of samples, k = number of parameters, SSE = sum of squared errors.
 *
 * Lower AIC indicates better model fit with parsimony.
 * Penalizes model complexity to avoid overfitting selection.
 *
 * Reference: Akaike, "A new look at the statistical model identification," IEEE TAC (1974)
 *
 * @param n_samples    Number of data points.
 * @param sse          Sum of squared errors.
 * @param n_parameters Number of trainable parameters.
 * @return AIC value.
 */
double nn_aic_criterion(int n_samples, double sse, int n_parameters);

/**
 * @brief Compute Bayesian Information Criterion (BIC).
 *
 * BIC = n * ln(SSE/n) + k * ln(n)
 *
 * Heavier penalty for complexity than AIC. As n -> inf, BIC is
 * asymptotically consistent (selects true model with probability -> 1).
 *
 * Reference: Schwarz, "Estimating the dimension of a model," Annals of Stats (1978)
 *
 * @param n_samples    Number of data points.
 * @param sse          Sum of squared errors.
 * @param n_parameters Number of trainable parameters.
 * @return BIC value.
 */
double nn_bic_criterion(int n_samples, double sse, int n_parameters);

/**
 * @brief Count trainable parameters in a neural network.
 *
 * Parameters = sum over layers of (W + b):
 * For layer i with n_in inputs and n_out outputs:
 *   n_params_i = n_in * n_out + n_out
 *
 * @param net  Neural network.
 * @return     Total number of trainable parameters.
 *
 * Complexity: O(n_layers).
 */
int nn_count_parameters(const nn_network_t *net);

/*===========================================================================
 * L4-L6: Cross-Validation
 *===========================================================================*/

/**
 * @brief Perform k-fold cross-validation.
 *
 * Splits training data into k folds. For each fold i:
 * - Train on k-1 folds
 * - Evaluate on fold i
 * Report mean and std of validation metrics across all k folds.
 *
 * @param net            Network (architecture only, weights will be reset each fold).
 * @param X              Input data (n_samples x n_features, row-major).
 * @param Y              Target data (n_samples x n_outputs).
 * @param n_samples      Total number of samples.
 * @param n_features     Number of input features.
 * @param n_outputs      Number of outputs.
 * @param k              Number of folds (typically 5 or 10).
 * @param max_epochs     Training epochs per fold.
 * @param mean_val_rmse  Output: mean validation RMSE across folds.
 * @param std_val_rmse   Output: standard deviation of validation RMSE.
 *
 * @return 0 on success, -1 on failure.
 *
 * Complexity: O(k * epochs * (k-1)/k * n_samples * n_params).
 */
int nn_kfold_cross_validate(nn_network_t *net,
                             const double *X, const double *Y,
                             int n_samples, int n_features, int n_outputs,
                             int k, int max_epochs,
                             double *mean_val_rmse, double *std_val_rmse);

/*===========================================================================
 * L4-L6: Overfitting Detection and Early Stopping
 *===========================================================================*/

/**
 * @brief Check for overfitting using validation loss.
 *
 * Overfitting is detected when validation loss increases for
 * `patience` consecutive epochs while training loss continues to decrease.
 *
 * @param train_loss_history  Training loss per epoch.
 * @param val_loss_history    Validation loss per epoch.
 * @param n_epochs            Number of epochs so far.
 * @param patience            Consecutive increases to trigger detection.
 * @return 1 if overfitting detected, 0 otherwise.
 */
int nn_detect_overfitting(const double *train_loss_history,
                           const double *val_loss_history,
                           int n_epochs, int patience);

/**
 * @brief Determine optimal early stopping epoch.
 *
 * Returns the epoch with minimum validation loss.
 *
 * @param val_loss_history  Validation loss per epoch.
 * @param n_epochs          Number of epochs recorded.
 * @return                  Index (epoch) of minimum validation loss.
 */
int nn_early_stopping_epoch(const double *val_loss_history, int n_epochs);

/*===========================================================================
 * L6: Residual Analysis (Statistical Validation)
 *===========================================================================*/

/**
 * @brief Analyze prediction residuals for statistical properties.
 *
 * Good model should have:
 * - Mean residual ~ 0 (unbiased)
 * - Residuals uncorrelated with inputs (no remaining structure)
 * - Residuals approximately normal (Q-Q, though lightweight check)
 *
 * @param y_true    Ground truth.
 * @param y_pred    Predictions.
 * @param n         Number of samples.
 * @param residuals Output: prediction residuals (y_true - y_pred), length n.
 * @param bias      Output: mean of residuals (systematic bias).
 * @param std_dev   Output: standard deviation of residuals.
 * @param skewness  Output: skewness of residuals.
 * @param kurtosis  Output: excess kurtosis of residuals.
 */
void nn_residual_analysis(const double *y_true, const double *y_pred, int n,
                           double *residuals, double *bias, double *std_dev,
                           double *skewness, double *kurtosis);

/**
 * @brief Compute Durbin-Watson statistic for residual autocorrelation.
 *
 * DW = sum_{t=2}^n (e_t - e_{t-1})^2 / sum_{t=1}^n e_t^2
 *
 * Tests for first-order autocorrelation. Values near 2 indicate
 * no autocorrelation. < 1 indicates positive autocorrelation.
 *
 * Reference: Durbin & Watson, Biometrika (1950, 1951)
 *
 * @param residuals  Residuals in time order.
 * @param n          Number of residuals.
 * @return           Durbin-Watson statistic.
 */
double nn_durbin_watson(const double *residuals, int n);

/*===========================================================================
 * L7: Soft Sensor Performance Monitoring
 *===========================================================================*/

/**
 * @brief Update soft sensor performance tracking with new observation.
 *
 * Computes moving-window MSE for drift detection.
 *
 * @param sensor    Soft sensor.
 * @param actual    Actual (lab) measurement.
 * @param predicted Soft sensor prediction.
 *
 * Reference: Kadlec et al. (2009) - Section on "Maintenance of soft sensors"
 */
void soft_sensor_update_metrics(soft_sensor_t *sensor, double actual, double predicted);

/**
 * @brief Detect concept drift in soft sensor using CUSUM.
 *
 * CUSUM: S_t = max(0, S_{t-1} + e_t - delta)
 * Alarm when S_t > threshold.
 *
 * @param sensor    Soft sensor with accumulated errors.
 * @param threshold CUSUM alarm threshold.
 * @return          1 if drift detected, 0 otherwise.
 *
 * Reference: Page, "Continuous Inspection Schemes," Biometrika (1954)
 *            Gama et al., "A survey on concept drift adaptation," ACM CSUR (2014)
 */
int soft_sensor_detect_drift(soft_sensor_t *sensor, double threshold);

/**
 * @brief Compare two soft sensor models on a test set for model selection.
 *
 * Performs paired t-test on squared errors to determine if
 * model A is significantly better than model B.
 *
 * @param sensor_a  First soft sensor.
 * @param sensor_b  Second soft sensor.
 * @param X_test    Test input data.
 * @param Y_test    Test target data.
 * @param n_test    Number of test samples.
 * @param p_value   Output: p-value of paired t-test.
 * @return          1 if model A significantly better (p < 0.05), 0 otherwise.
 */
int soft_sensor_compare_models(soft_sensor_t *sensor_a, soft_sensor_t *sensor_b,
                                const double *X_test, const double *Y_test,
                                int n_test, double *p_value);

/*===========================================================================
 * L7-L8: Industrial Process Models (Simulated Data Generators)
 *===========================================================================*/

/**
 * @brief Simulate distillation column data for soft sensor training.
 *
 * Generates input (tray temperatures, pressures, reflux ratio, feed rate)
 * and output (top product composition, bottom product composition).
 *
 * Based on simplified binary distillation model (McCabe-Thiele).
 *
 * @param X          Output: input features (n_samples x n_features).
 * @param Y          Output: output targets (n_samples x n_outputs).
 * @param n_samples  Number of samples to generate.
 * @param noise_std  Standard deviation of measurement noise.
 *
 * Reference: Skogestad, "Dynamics and control of distillation columns" (1997)
 *            Luyben, "Practical Distillation Control" (1992)
 */
void nn_generate_distillation_data(double *X, double *Y, int n_samples, double noise_std);

/**
 * @brief Simulate fermentation process data for soft sensor.
 *
 * Generates inputs (temperature, pH, DO, agitation, feed rate)
 * and outputs (biomass concentration, product concentration, substrate).
 *
 * Based on Monod kinetics for microbial growth.
 *
 * @param X          Output: input features (n x 5).
 * @param Y          Output: output targets (n x 3).
 * @param n_samples  Number of samples.
 * @param noise_std  Measurement noise std.
 *
 * Reference: Bailey & Ollis, "Biochemical Engineering Fundamentals" (1986)
 */
void nn_generate_fermentation_data(double *X, double *Y, int n_samples, double noise_std);

/**
 * @brief Simulate polymerization process data for soft sensor.
 *
 * Generates inputs (temperature, pressure, catalyst flow, H2/C2 ratio, residence time)
 * and outputs (melt index, density, production rate).
 *
 * Based on Ziegler-Natta ethylene polymerization model.
 *
 * @param X          Output: input features (n x 5).
 * @param Y          Output: output targets (n x 3).
 * @param n_samples  Number of samples.
 * @param noise_std  Measurement noise std.
 *
 * Reference: McAuley & MacGregor, "On-line inference of polymer properties..." AIChE J. (1991)
 *            Richards & Congalidis, "Measurement and control of polymerization reactors" (2006)
 */
void nn_generate_polymerization_data(double *X, double *Y, int n_samples, double noise_std);

/**
 * @brief Simulate cement kiln data for soft sensor.
 *
 * Generates inputs (kiln temperature, kiln torque, O2%, NOx, feed rate)
 * and outputs (free lime content, liter weight, clinker quality index).
 *
 * @param X          Output: input features (n x 5).
 * @param Y          Output: output targets (n x 3).
 * @param n_samples  Number of samples.
 * @param noise_std  Measurement noise std.
 *
 * Reference: Pani et al., "Soft sensing of cement..." JPC (2013)
 */
void nn_generate_cement_kiln_data(double *X, double *Y, int n_samples, double noise_std);

/**
 * @brief Simulate fluid catalytic cracking (FCC) unit data.
 *
 * Generates inputs (riser temperature, regenerator temp, catalyst circulation,
 * feed preheat, C/O ratio) and outputs (gasoline yield, LCO yield, conversion).
 *
 * @param X          Output: input features (n x 5).
 * @param Y          Output: output targets (n x 3).
 * @param n_samples  Number of samples.
 * @param noise_std  Measurement noise std.
 *
 * Reference: Avidan & Shinnar, "Development of catalytic cracking..." IECR (1990)
 */
void nn_generate_fcc_data(double *X, double *Y, int n_samples, double noise_std);

/**
 * @brief Simulate blast furnace hot metal quality data.
 *
 * Generates inputs (blast temperature, blast volume, oxygen enrichment,
 * PCI rate, burden distribution) and outputs (hot metal temperature,
 * Si content, S content).
 *
 * @param X          Output: input features (n x 5).
 * @param Y          Output: output targets (n x 3).
 * @param n_samples  Number of samples.
 * @param noise_std  Measurement noise std.
 *
 * Reference: Saxen et al., "Evolving fuzzy logic models for blast furnace..." (2016)
 */
void nn_generate_blast_furnace_data(double *X, double *Y, int n_samples, double noise_std);

#ifdef __cplusplus
}
#endif

#endif /* NN_SENSOR_VALIDATION_H */
