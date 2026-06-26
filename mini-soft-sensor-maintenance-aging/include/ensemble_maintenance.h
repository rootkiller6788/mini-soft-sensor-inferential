/**
 * @file    ensemble_maintenance.h
 * @brief   Ensemble soft sensor maintenance strategies - L5 & L8
 *
 * Multi-model ensemble for robust soft sensor maintenance:
 *   - Weighted average ensemble with dynamic weighting
 *   - Stacked generalization (meta-learner)
 *   - Model diversity measurement
 *   - Automatic model retirement and replacement
 *   - Ensemble pruning for computational efficiency
 *
 * L5: Ensemble methods, weighted averaging, stacking
 * L8: Dynamic ensemble pruning, diversity-driven maintenance
 *
 * Ref: Kadlec et al. (2011) "Data-driven soft sensors in the process
 *      industry". Comp. & Chem. Eng., 35(1), 1-24.
 */

#ifndef ENSEMBLE_MAINTENANCE_H
#define ENSEMBLE_MAINTENANCE_H

#include <stddef.h>
#include <stdint.h>

/* ============================================================================
 * Ensemble Member
 * ===========================================================================*/

/** @brief Single ensemble member metadata */
typedef struct {
    uint64_t member_id;
    char     name[64];
    int      is_active;
    double   current_weight;
    double   recent_rmse;
    double   recent_r2;
    double   age_hours;
    double   diversity_score;
    double   reliability;
    uint64_t last_update_time;
    size_t   training_samples;
    int      retired;
} EnsembleMember;

/** @brief Aggregate ensemble prediction */
typedef struct {
    double weighted_mean;
    double weighted_variance;
    double simple_mean;
    double best_member_pred;
    double worst_member_pred;
    double prediction_spread;
    double consensus_index;
    size_t active_members;
    size_t total_members;
} EnsemblePrediction;

/* ============================================================================
 * Weighting Strategy
 * ===========================================================================*/

/** @brief Ensemble weighting strategy types */
typedef enum {
    WEIGHT_UNIFORM           = 0,
    WEIGHT_INVERSE_RMSE      = 1,
    WEIGHT_RECENT_PERFORMANCE = 2,
    WEIGHT_BAYESIAN          = 3,
    WEIGHT_DIVERSITY_AWARE   = 4
} WeightingStrategy;

/* ============================================================================
 * Maintenance Actions
 * ===========================================================================*/

/** @brief Maintenance action types for ensemble */
typedef enum {
    MAINT_NONE               = 0,
    MAINT_REWEIGHT           = 1,
    MAINT_RETRAIN_MEMBER     = 2,
    MAINT_REPLACE_MEMBER     = 3,
    MAINT_ADD_MEMBER         = 4,
    MAINT_REMOVE_MEMBER      = 5,
    MAINT_REBUILD_ENSEMBLE   = 6
} MaintenanceAction;

/** @brief Maintenance recommendation for ensemble */
typedef struct {
    MaintenanceAction action;
    uint64_t target_member_id;
    double urgency;
    char description[256];
    double expected_improvement;
} MaintenanceRecommendation;

/* ============================================================================
 * Ensemble State
 * ===========================================================================*/

/** @brief Full ensemble state */
typedef struct {
    EnsembleMember *members;
    size_t capacity;
    size_t count;
    WeightingStrategy weighting;
    double *member_weights;
    double total_weight;
    uint64_t ensemble_age_hours;
    size_t total_predictions;
    double cumulative_rmse;
    double recent_rmse;
    int needs_maintenance;
} EnsembleState;

/* ============================================================================
 * L5: Diversity Analysis
 * ===========================================================================*/

/** @brief Pairwise model diversity metrics */
typedef struct {
    double correlation;
    double disagreement_rate;
    double q_statistic;
    double kappa_statistic;
    double double_fault;
} PairwiseDiversity;

/** @brief Ensemble-wide diversity summary */
typedef struct {
    double mean_correlation;
    double mean_disagreement;
    double mean_q_statistic;
    double entropy_diversity;
    double kohavi_wolpert_variance;
    int is_diverse;
} EnsembleDiversity;

/* ============================================================================
 * L8: Optimal Ensemble Pruning
 * ===========================================================================*/

/** @brief Pruning result - identifies members to keep/remove */
typedef struct {
    int *keep_mask;
    size_t n_keep;
    size_t n_prune;
    double pruned_rmse;    /**< Expected RMSE after pruning */
    double improvement;     /**< RMSE improvement from pruning */
} PruningResult;

/* ============================================================================
 * API: Ensemble Management
 * ===========================================================================*/

EnsembleState ensemble_create(size_t capacity);
void ensemble_destroy(EnsembleState *ens);

/**
 * @brief Add a new member to the ensemble.
 * Returns member_id (>0) or 0 on failure (ensemble full).
 */
uint64_t ensemble_add_member(EnsembleState *ens, const char *name);

/**
 * @brief Remove a member from the ensemble.
 * Returns 1 on success, 0 if member not found.
 */
int ensemble_remove_member(EnsembleState *ens, uint64_t member_id);

/**
 * @brief Update a member's performance statistics.
 * This drives weight re-computation.
 */
void ensemble_update_member_stats(EnsembleState *ens, uint64_t member_id,
                                   double rmse, double r2, uint64_t timestamp);

/**
 * @brief Get weighted ensemble prediction.
 * Combines all active members using current weighting strategy.
 * Also computes prediction spread and consensus index.
 *
 * Consensus = 1.0 - (spread / (mean + epsilon))
 * High consensus means members agree; low suggests conflicting models.
 */
EnsemblePrediction ensemble_predict(EnsembleState *ens,
                                     const double *member_predictions);

/* ============================================================================
 * API: Weighting Strategies
 * ===========================================================================*/

/**
 * @brief Recompute all member weights based on strategy.
 *
 * Uniform:       w_i = 1/N
 * Inverse RMSE:  w_i = (1/rmse_i) / sum(1/rmse_j)
 * Recent:        w_i = exp(-alpha * age_i) * (1/rmse_i) / normalizer
 * Bayesian:      w_i ? P(data | model_i) * P(model_i)
 * Diversity-aware: w_i combined from accuracy and diversity contribution
 */
void ensemble_recompute_weights(EnsembleState *ens);

/**
 * @brief Change weighting strategy.
 * Triggers automatic weight recomputation.
 */
void ensemble_set_weighting(EnsembleState *ens, WeightingStrategy strategy);

/* ============================================================================
 * API: Maintenance
 * ===========================================================================*/

/**
 * @brief Analyze ensemble health and generate maintenance recommendation.
 *
 * Checks:
 *   1. Any member with reliability below threshold
 *   2. Ensemble average RMSE trend
 *   3. Diversity level (too low = redundant members)
 *   4. Age of members relative to ensemble age
 *
 * @return MaintenanceRecommendation with action and urgency.
 */
MaintenanceRecommendation ensemble_diagnose(const EnsembleState *ens,
                                              double reliability_threshold);

/**
 * @brief Execute the recommended maintenance action.
 * Handles reweighting, retiring, replacing members.
 */
void ensemble_execute_maintenance(EnsembleState *ens,
                                   const MaintenanceRecommendation *rec);

/* ============================================================================
 * L5: Diversity Analysis API
 * ===========================================================================*/

/**
 * @brief Compute pairwise diversity between two member's prediction histories.
 *
 * Metrics (Kuncheva & Whitaker 2003):
 *   - Q-statistic: Q_{i,j} = (ad-bc)/(ad+bc)
 *   - Correlation: Pearson correlation of errors
 *   - Disagreement: fraction of samples where members disagree
 *   - Double fault: fraction where both are wrong
 */
PairwiseDiversity compute_pairwise_diversity(const double *pred1,
                                              const double *pred2,
                                              const double *actual,
                                              size_t n);

/**
 * @brief Compute ensemble-wide diversity from all pairwise metrics.
 */
EnsembleDiversity compute_ensemble_diversity(const EnsembleState *ens,
                                              const double **member_preds,
                                              const double *actual,
                                              size_t n_samples);

/* ============================================================================
 * L8: Ensemble Pruning API
 * ===========================================================================*/

/**
 * @brief Greedy ensemble pruning via ordered aggregation.
 *
 * Algorithm (Martinez-Munoz & Suarez 2006):
 *   1. Order members by individual accuracy
 *   2. Iteratively add best remaining member
 *   3. Stop when adding more members does not improve ensemble accuracy
 *
 * Returns pruning mask (1 = keep, 0 = prune).
 *
 * @param ens          Ensemble to prune.
 * @param member_preds Matrix [n_members * n_samples] of predictions.
 * @param actual       Actual values, length n_samples.
 * @param n_samples    Number of validation samples.
 * @return PruningResult with keep_mask (caller must free keep_mask).
 */
PruningResult ensemble_prune_greedy(EnsembleState *ens,
                                      const double **member_preds,
                                      const double *actual,
                                      size_t n_samples);

/* ============================================================================
 * API: Ensemble Metrics
 * ===========================================================================*/

/**
 * @brief Compute ensemble age-weighted performance.
 * Recent performance weighted more heavily.
 */
double ensemble_recent_performance(const EnsembleState *ens);

/**
 * @brief Determine if ensemble is degrading.
 * Returns 1 if ensemble RMSE trend is significantly increasing.
 */
int ensemble_is_degrading(const EnsembleState *ens);

/**
 * @brief Get ensemble health summary as a single 0-1 score.
 */
double ensemble_health_score(const EnsembleState *ens);

#endif /* ENSEMBLE_MAINTENANCE_H */
