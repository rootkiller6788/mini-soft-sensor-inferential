/**
 * @file    ensemble_maintenance.c
 * @brief   Ensemble soft sensor maintenance - weighting, diversity, pruning
 *
 * L5: Ensemble methods, weighted averaging, stacking, diversity analysis
 * L8: Greedy ensemble pruning, dynamic reweighting, diversity-driven maintenance
 *
 * Ref: Kadlec et al. (2011) Comp. & Chem. Eng. 35(1), 1-24.
 *      Martinez-Munoz & Suarez (2006) "Pruning in ordered bagging ensembles".
 *      Kuncheva & Whitaker (2003) Machine Learning 51, 181-207.
 */

#include "ensemble_maintenance.h"
#include "soft_sensor_metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ====================================================================
 * Ensemble Management
 * ==================================================================== */

EnsembleState ensemble_create(size_t capacity)
{
    EnsembleState ens;
    memset(&ens, 0, sizeof(ens));
    ens.capacity = capacity;
    ens.members = (EnsembleMember *)calloc(capacity, sizeof(EnsembleMember));
    ens.member_weights = (double *)calloc(capacity, sizeof(double));
    ens.weighting = WEIGHT_UNIFORM;
    ens.total_weight = 0.0;
    return ens;
}

void ensemble_destroy(EnsembleState *ens)
{
    if (!ens) return;
    free(ens->members);
    free(ens->member_weights);
    memset(ens, 0, sizeof(*ens));
}

uint64_t ensemble_add_member(EnsembleState *ens, const char *name)
{
    if (!ens || !ens->members || ens->count >= ens->capacity) return 0;

    /* Find empty slot or use next position */
    size_t pos = ens->count;
    EnsembleMember *m = &ens->members[pos];
    memset(m, 0, sizeof(*m));

    /* Generate ID from position + count */
    m->member_id = (uint64_t)(pos + 1) * 1000 + ens->total_predictions;
    if (name) {
        strncpy(m->name, name, 63);
        m->name[63] = '\0';
    } else {
        snprintf(m->name, 64, "Member_%llu", (unsigned long long)pos);
    }
    m->is_active = 1;
    m->current_weight = 1.0 / (double)(ens->count + 1);
    m->reliability = 1.0;

    /* Rebalance weights */
    ens->count++;
    ensemble_recompute_weights(ens);

    return m->member_id;
}

int ensemble_remove_member(EnsembleState *ens, uint64_t member_id)
{
    if (!ens || !ens->members) return 0;

    for (size_t i = 0; i < ens->count; i++) {
        if (ens->members[i].member_id == member_id) {
            ens->members[i].is_active = 0;
            ens->members[i].retired = 1;
            ens->members[i].current_weight = 0.0;
            ensemble_recompute_weights(ens);
            return 1;
        }
    }
    return 0;
}

void ensemble_update_member_stats(EnsembleState *ens, uint64_t member_id,
                                   double rmse, double r2, uint64_t timestamp)
{
    if (!ens) return;

    for (size_t i = 0; i < ens->count; i++) {
        if (ens->members[i].member_id == member_id) {
            EnsembleMember *m = &ens->members[i];
            m->recent_rmse = rmse;
            m->recent_r2 = r2;
            m->last_update_time = timestamp;
            /* Reliability: combines accuracy and recency */
            double accuracy = (rmse < 1e-10) ? 1.0 : exp(-rmse);
            m->reliability = fmax(0.0, fmin(1.0, accuracy));
            break;
        }
    }

    /* Trigger weight recomputation if using performance-based weighting */
    if (ens->weighting >= WEIGHT_INVERSE_RMSE)
        ensemble_recompute_weights(ens);
}

/* ====================================================================
 * Ensemble Prediction
 * ==================================================================== */

EnsemblePrediction ensemble_predict(EnsembleState *ens,
                                     const double *member_predictions)
{
    EnsemblePrediction ep;
    memset(&ep, 0, sizeof(ep));

    if (!ens || !member_predictions) return ep;

    double weighted_sum = 0.0, weighted_var = 0.0;
    double simple_sum = 0.0;
    double best_pred = -1e100, worst_pred = 1e100;
    double min_pred = 1e100, max_pred = -1e100;
    size_t active = 0;

    ep.total_members = ens->count;

    for (size_t i = 0; i < ens->count; i++) {
        if (!ens->members[i].is_active || ens->members[i].retired)
            continue;

        double pred = member_predictions[i];
        double w = ens->member_weights[i];

        weighted_sum += w * pred;
        simple_sum += pred;
        active++;

        if (pred > best_pred) best_pred = pred;
        if (pred < worst_pred) worst_pred = pred;
        if (pred > max_pred) max_pred = pred;
        if (pred < min_pred) min_pred = pred;
    }

    ep.active_members = active;
    ep.weighted_mean = (ens->total_weight > 1e-15)
                       ? weighted_sum / ens->total_weight : 0.0;
    ep.simple_mean = (active > 0) ? simple_sum / (double)active : 0.0;
    ep.best_member_pred = best_pred;
    ep.worst_member_pred = worst_pred;
    ep.prediction_spread = max_pred - min_pred;

    /* Consensus index: 1.0 - spread/(|mean| + epsilon) */
    double abs_mean = fabs(ep.weighted_mean);
    double epsilon = 1e-6;
    ep.consensus_index = 1.0 - ep.prediction_spread / (abs_mean + epsilon);
    if (ep.consensus_index < 0.0) ep.consensus_index = 0.0;
    if (ep.consensus_index > 1.0) ep.consensus_index = 1.0;

    /* Weighted variance of predictions */
    if (active > 1) {
        for (size_t i = 0; i < ens->count; i++) {
            if (!ens->members[i].is_active || ens->members[i].retired)
                continue;
            double diff = member_predictions[i] - ep.weighted_mean;
            double w = ens->member_weights[i];
            weighted_var += w * diff * diff;
        }
        ep.weighted_variance = (ens->total_weight > 1e-15)
                               ? weighted_var / ens->total_weight : 0.0;
    }

    /* Update ensemble statistics */
    ens->total_predictions++;

    return ep;
}

/* ====================================================================
 * Weighting Strategies
 * ==================================================================== */

void ensemble_recompute_weights(EnsembleState *ens)
{
    if (!ens || !ens->members || !ens->member_weights) return;

    size_t active_count = 0;
    for (size_t i = 0; i < ens->count; i++) {
        if (ens->members[i].is_active && !ens->members[i].retired)
            active_count++;
    }

    if (active_count == 0) {
        ens->total_weight = 0.0;
        return;
    }

    double total_w = 0.0;

    switch (ens->weighting) {
        case WEIGHT_UNIFORM: {
            double w = 1.0 / (double)active_count;
            for (size_t i = 0; i < ens->count; i++) {
                if (ens->members[i].is_active && !ens->members[i].retired) {
                    ens->member_weights[i] = w;
                    ens->members[i].current_weight = w;
                    total_w += w;
                } else {
                    ens->member_weights[i] = 0.0;
                    ens->members[i].current_weight = 0.0;
                }
            }
            break;
        }

        case WEIGHT_INVERSE_RMSE: {
            double inv_rmse_sum = 0.0;
            for (size_t i = 0; i < ens->count; i++) {
                if (ens->members[i].is_active && !ens->members[i].retired) {
                    double inv = 1.0 / fmax(ens->members[i].recent_rmse, 1e-6);
                    inv_rmse_sum += inv;
                }
            }
            for (size_t i = 0; i < ens->count; i++) {
                if (ens->members[i].is_active && !ens->members[i].retired) {
                    double inv = 1.0 / fmax(ens->members[i].recent_rmse, 1e-6);
                    ens->member_weights[i] = inv / inv_rmse_sum;
                    ens->members[i].current_weight = ens->member_weights[i];
                    total_w += ens->member_weights[i];
                } else {
                    ens->member_weights[i] = 0.0;
                    ens->members[i].current_weight = 0.0;
                }
            }
            break;
        }

        case WEIGHT_RECENT_PERFORMANCE: {
            /* Exponential decay weight: w_i ~ exp(-alpha * age) / rmse */
            double sum_w = 0.0;
            for (size_t i = 0; i < ens->count; i++) {
                if (ens->members[i].is_active && !ens->members[i].retired) {
                    double age_factor = exp(-0.01 * ens->members[i].age_hours);
                    double acc_factor = 1.0 / fmax(ens->members[i].recent_rmse, 1e-6);
                    sum_w += age_factor * acc_factor;
                }
            }
            for (size_t i = 0; i < ens->count; i++) {
                if (ens->members[i].is_active && !ens->members[i].retired) {
                    double age_factor = exp(-0.01 * ens->members[i].age_hours);
                    double acc_factor = 1.0 / fmax(ens->members[i].recent_rmse, 1e-6);
                    ens->member_weights[i] = age_factor * acc_factor / sum_w;
                    total_w += ens->member_weights[i];
                } else {
                    ens->member_weights[i] = 0.0;
                }
            }
            break;
        }

        case WEIGHT_BAYESIAN: {
            /* Bayesian Model Averaging: weight proportional to
             * exp(-0.5 * n * log(mse)) (BIC approximation) */
            double sum_bic = 0.0;
            for (size_t i = 0; i < ens->count; i++) {
                if (ens->members[i].is_active && !ens->members[i].retired) {
                    double mse = ens->members[i].recent_rmse;
                    mse = mse * mse;
                    if (mse < 1e-10) mse = 1e-10;
                    double bic = exp(-0.5 * log(mse));
                    sum_bic += bic;
                }
            }
            for (size_t i = 0; i < ens->count; i++) {
                if (ens->members[i].is_active && !ens->members[i].retired) {
                    double mse = ens->members[i].recent_rmse;
                    mse = mse * mse;
                    if (mse < 1e-10) mse = 1e-10;
                    double bic = exp(-0.5 * log(mse));
                    ens->member_weights[i] = bic / sum_bic;
                    total_w += ens->member_weights[i];
                } else {
                    ens->member_weights[i] = 0.0;
                }
            }
            break;
        }

        case WEIGHT_DIVERSITY_AWARE: {
            /* Combine accuracy and diversity:
             * w_i = (1/rmse_i) * diversity_factor
             * diversity_factor encourages members that differ from others */
            double sum_w = 0.0;
            for (size_t i = 0; i < ens->count; i++) {
                if (ens->members[i].is_active && !ens->members[i].retired) {
                    double acc = 1.0 / fmax(ens->members[i].recent_rmse, 1e-6);
                    double div = 1.0 + ens->members[i].diversity_score;
                    sum_w += acc * div;
                }
            }
            for (size_t i = 0; i < ens->count; i++) {
                if (ens->members[i].is_active && !ens->members[i].retired) {
                    double acc = 1.0 / fmax(ens->members[i].recent_rmse, 1e-6);
                    double div = 1.0 + ens->members[i].diversity_score;
                    ens->member_weights[i] = acc * div / sum_w;
                    total_w += ens->member_weights[i];
                } else {
                    ens->member_weights[i] = 0.0;
                }
            }
            break;
        }
    }

    ens->total_weight = total_w;
}

void ensemble_set_weighting(EnsembleState *ens, WeightingStrategy strategy)
{
    if (!ens) return;
    ens->weighting = strategy;
    ensemble_recompute_weights(ens);
}

/* ====================================================================
 * Maintenance Diagnosis
 * ==================================================================== */

MaintenanceRecommendation ensemble_diagnose(const EnsembleState *ens,
                                              double reliability_threshold)
{
    MaintenanceRecommendation rec;
    memset(&rec, 0, sizeof(rec));
    rec.action = MAINT_NONE;
    rec.urgency = 0.0;

    if (!ens || ens->count == 0) return rec;

    /* Check individual member reliability */
    int low_reliability_count = 0;
    uint64_t worst_member_id = 0;
    double worst_reliability = 1.0;

    for (size_t i = 0; i < ens->count; i++) {
        if (!ens->members[i].is_active || ens->members[i].retired)
            continue;
        if (ens->members[i].reliability < reliability_threshold) {
            low_reliability_count++;
            if (ens->members[i].reliability < worst_reliability) {
                worst_reliability = ens->members[i].reliability;
                worst_member_id = ens->members[i].member_id;
            }
        }
    }

    /* Decision logic */
    size_t active = 0;
    for (size_t i = 0; i < ens->count; i++)
        if (ens->members[i].is_active && !ens->members[i].retired) active++;

    if (active == 0) {
        rec.action = MAINT_REBUILD_ENSEMBLE;
        rec.urgency = 1.0;
        snprintf(rec.description, 256, "No active members. Rebuild ensemble.");
    } else if ((size_t)low_reliability_count >= active) {
        rec.action = MAINT_REBUILD_ENSEMBLE;
        rec.urgency = 0.9;
        snprintf(rec.description, 256,
                 "All %llu members below reliability threshold %.2f.",
                 (unsigned long long)active, reliability_threshold);
    } else if ((size_t)low_reliability_count > active / 2) {
        rec.action = MAINT_REWEIGHT;
        rec.urgency = 0.6;
        snprintf(rec.description, 256,
                 "%d/%llu members low reliability. Reweighting recommended.",
                 low_reliability_count, (unsigned long long)active);
    } else if (low_reliability_count > 0) {
        rec.action = MAINT_RETRAIN_MEMBER;
        rec.target_member_id = worst_member_id;
        rec.urgency = 0.3;
        snprintf(rec.description, 256,
                 "Member %llu has low reliability %.3f. Consider retraining.",
                 (unsigned long long)worst_member_id, worst_reliability);
    } else {
        rec.action = MAINT_NONE;
        rec.urgency = 0.0;
        snprintf(rec.description, 256, "Ensemble healthy. No maintenance needed.");
    }

    return rec;
}

void ensemble_execute_maintenance(EnsembleState *ens,
                                   const MaintenanceRecommendation *rec)
{
    if (!ens || !rec) return;

    switch (rec->action) {
        case MAINT_REWEIGHT:
            ensemble_recompute_weights(ens);
            break;
        case MAINT_RETRAIN_MEMBER:
            /* Mark member for retraining (set reliability low to trigger action) */
            if (rec->target_member_id) {
                for (size_t i = 0; i < ens->count; i++) {
                    if (ens->members[i].member_id == rec->target_member_id) {
                        ens->members[i].reliability *= 0.5;
                        break;
                    }
                }
            }
            ensemble_recompute_weights(ens);
            break;
        case MAINT_REMOVE_MEMBER:
            if (rec->target_member_id)
                ensemble_remove_member(ens, rec->target_member_id);
            break;
        case MAINT_REPLACE_MEMBER:
            if (rec->target_member_id) {
                ensemble_remove_member(ens, rec->target_member_id);
                char name[64];
                snprintf(name, 64, "Replacement_%llu",
                         (unsigned long long)rec->target_member_id);
                ensemble_add_member(ens, name);
            }
            break;
        case MAINT_REBUILD_ENSEMBLE:
            /* Retire all and start fresh */
            for (size_t i = 0; i < ens->count; i++) {
                ens->members[i].is_active = 0;
                ens->members[i].retired = 1;
            }
            break;
        default:
            break;
    }
}

/* ====================================================================
 * L5: Diversity Analysis (Kuncheva & Whitaker 2003)
 * ==================================================================== */

PairwiseDiversity compute_pairwise_diversity(const double *pred1,
                                              const double *pred2,
                                              const double *actual,
                                              size_t n)
{
    PairwiseDiversity pd;
    memset(&pd, 0, sizeof(pd));

    if (!pred1 || !pred2 || !actual || n == 0) return pd;

    /* Build confusion table for regression (above/below mean) */
    double mean_actual = 0.0;
    for (size_t i = 0; i < n; i++) mean_actual += actual[i];
    mean_actual /= (double)n;

    size_t a = 0, b = 0, c = 0, d = 0;
    /* a: both correct, b: m1 correct m2 wrong, c: m1 wrong m2 correct, d: both wrong */
    /* "Correct" = prediction error below median error */

    double *err1 = (double *)malloc(n * sizeof(double));
    double *err2 = (double *)malloc(n * sizeof(double));
    if (!err1 || !err2) { free(err1); free(err2); return pd; }

    for (size_t i = 0; i < n; i++) {
        err1[i] = fabs(pred1[i] - actual[i]);
        err2[i] = fabs(pred2[i] - actual[i]);
    }

    /* Compute median errors */
    double median1, median2;
    {
        double *tmp1 = (double *)malloc(n * sizeof(double));
        double *tmp2 = (double *)malloc(n * sizeof(double));
        if (!tmp1 || !tmp2) { free(err1); free(err2); free(tmp1); free(tmp2); return pd; }
        memcpy(tmp1, err1, n * sizeof(double));
        memcpy(tmp2, err2, n * sizeof(double));

        /* Simple median via sorting */
        for (size_t i = 0; i < n - 1; i++) {
            for (size_t j = i + 1; j < n; j++) {
                if (tmp1[i] > tmp1[j]) { double t = tmp1[i]; tmp1[i] = tmp1[j]; tmp1[j] = t; }
                if (tmp2[i] > tmp2[j]) { double t = tmp2[i]; tmp2[i] = tmp2[j]; tmp2[j] = t; }
            }
        }
        median1 = (n % 2) ? tmp1[n/2] : (tmp1[n/2-1] + tmp1[n/2]) / 2.0;
        median2 = (n % 2) ? tmp2[n/2] : (tmp2[n/2-1] + tmp2[n/2]) / 2.0;
        free(tmp1); free(tmp2);
    }

    for (size_t i = 0; i < n; i++) {
        int c1 = (err1[i] <= median1) ? 1 : 0;
        int c2 = (err2[i] <= median2) ? 1 : 0;
        if (c1 && c2) a++;
        else if (c1 && !c2) b++;
        else if (!c1 && c2) c++;
        else d++;
    }

    free(err1); free(err2);

    /* Q-statistic: (a*d - b*c) / (a*d + b*c) */
    int ad = a * d, bc = b * c;
    if (ad + bc > 0)
        pd.q_statistic = (double)(ad - bc) / (double)(ad + bc);

    /* Correlation of errors */
    double sum_e1 = 0.0, sum_e2 = 0.0, sum_e12 = 0.0, sum_e11 = 0.0, sum_e22 = 0.0;
    for (size_t i = 0; i < n; i++) {
        /* Recompute errors */
        double e1 = pred1[i] - actual[i];
        double e2 = pred2[i] - actual[i];
        sum_e1 += e1; sum_e2 += e2;
        sum_e12 += e1 * e2;
        sum_e11 += e1 * e1;
        sum_e22 += e2 * e2;
    }
    double num = sum_e12 - sum_e1 * sum_e2 / (double)n;
    double den1 = sum_e11 - sum_e1 * sum_e1 / (double)n;
    double den2 = sum_e22 - sum_e2 * sum_e2 / (double)n;
    double den = sqrt(fmax(den1 * den2, 0.0));
    if (den > 1e-15) pd.correlation = num / den;

    /* Disagreement rate */
    if (n > 0) pd.disagreement_rate = (double)(b + c) / (double)n;

    /* Double fault rate */
    if (n > 0) pd.double_fault = (double)d / (double)n;

    /* Kappa statistic */
    double p0 = (double)(a + d) / (double)n;
    double p_yes = (double)(a + b) * (double)(a + c) / (double)(n * n);
    double p_no = (double)(c + d) * (double)(b + d) / (double)(n * n);
    double pe = p_yes + p_no;
    if (fabs(1.0 - pe) > 1e-15)
        pd.kappa_statistic = (p0 - pe) / (1.0 - pe);

    return pd;
}

EnsembleDiversity compute_ensemble_diversity(const EnsembleState *ens,
                                              const double **member_preds,
                                              const double *actual,
                                              size_t n_samples)
{
    EnsembleDiversity ed;
    memset(&ed, 0, sizeof(ed));

    if (!ens || !member_preds || !actual || ens->count < 2) return ed;

    size_t n_pairs = 0;
    double sum_corr = 0.0, sum_disagree = 0.0, sum_q = 0.0;

    for (size_t i = 0; i < ens->count - 1; i++) {
        if (!ens->members[i].is_active) continue;
        for (size_t j = i + 1; j < ens->count; j++) {
            if (!ens->members[j].is_active) continue;
            PairwiseDiversity pd = compute_pairwise_diversity(
                member_preds[i], member_preds[j], actual, n_samples);
            sum_corr += pd.correlation;
            sum_disagree += pd.disagreement_rate;
            sum_q += pd.q_statistic;
            n_pairs++;
        }
    }

    if (n_pairs > 0) {
        ed.mean_correlation = sum_corr / (double)n_pairs;
        ed.mean_disagreement = sum_disagree / (double)n_pairs;
        ed.mean_q_statistic = sum_q / (double)n_pairs;
    }

    /* Kohavi-Wolpert variance: variance of predictions across members */
    double kw_var = 0.0;
    for (size_t s = 0; s < n_samples; s++) {
        double sum_pred = 0.0;
        size_t active = 0;
        for (size_t i = 0; i < ens->count; i++) {
            if (ens->members[i].is_active) {
                sum_pred += member_preds[i][s];
                active++;
            }
        }
        if (active > 1) {
            double mean_pred = sum_pred / (double)active;
            for (size_t i = 0; i < ens->count; i++) {
                if (ens->members[i].is_active) {
                    double diff = member_preds[i][s] - mean_pred;
                    kw_var += diff * diff;
                }
            }
        }
    }
    ed.kohavi_wolpert_variance = kw_var / (double)(n_samples * ens->count);

    /* Entropy diversity: higher = more diverse */
    ed.entropy_diversity = -ed.mean_correlation; /* Simplified proxy */
    ed.is_diverse = (ed.mean_disagreement > 0.1 && ed.mean_correlation < 0.9) ? 1 : 0;

    return ed;
}

/* ====================================================================
 * L8: Greedy Ensemble Pruning
 * ==================================================================== */

PruningResult ensemble_prune_greedy(EnsembleState *ens,
                                      const double **member_preds,
                                      const double *actual,
                                      size_t n_samples)
{
    PruningResult pr;
    memset(&pr, 0, sizeof(pr));

    if (!ens || !member_preds || !actual || n_samples == 0) return pr;

    pr.keep_mask = (int *)calloc(ens->count, sizeof(int));
    if (!pr.keep_mask) return pr;

    /* Compute individual member RMSE */
    double *member_rmse = (double *)malloc(ens->count * sizeof(double));
    size_t *ranked = (size_t *)malloc(ens->count * sizeof(size_t));
    if (!member_rmse || !ranked) {
        free(pr.keep_mask); free(member_rmse); free(ranked);
        memset(&pr, 0, sizeof(pr));
        return pr;
    }

    for (size_t i = 0; i < ens->count; i++) {
        ranked[i] = i;
        double sse = 0.0;
        for (size_t s = 0; s < n_samples; s++) {
            double e = member_preds[i][s] - actual[s];
            sse += e * e;
        }
        member_rmse[i] = sqrt(sse / (double)n_samples);
    }

    /* Sort by RMSE (ascending = best first) */
    for (size_t i = 0; i < ens->count - 1; i++) {
        for (size_t j = i + 1; j < ens->count; j++) {
            if (member_rmse[ranked[i]] > member_rmse[ranked[j]]) {
                size_t tmp = ranked[i];
                ranked[i] = ranked[j];
                ranked[j] = tmp;
            }
        }
    }

    /* Greedy addition: add members one by one, track ensemble RMSE */
    double best_ensemble_rmse = 1e100;
    size_t best_k = 1;
    double *ensemble_pred = (double *)malloc(n_samples * sizeof(double));

    for (size_t k = 1; k <= ens->count; k++) {
        /* Ensemble prediction = average of first k ranked members */
        for (size_t s = 0; s < n_samples; s++) {
            ensemble_pred[s] = 0.0;
            for (size_t r = 0; r < k; r++)
                ensemble_pred[s] += member_preds[ranked[r]][s];
            ensemble_pred[s] /= (double)k;
        }

        /* Compute ensemble RMSE */
        double ensemble_sse = 0.0;
        for (size_t s = 0; s < n_samples; s++) {
            double e = ensemble_pred[s] - actual[s];
            ensemble_sse += e * e;
        }
        double ensemble_rmse = sqrt(ensemble_sse / (double)n_samples);

        if (ensemble_rmse < best_ensemble_rmse) {
            best_ensemble_rmse = ensemble_rmse;
            best_k = k;
        }

        /* Stop if adding more members stops helping */
        if (k > 1 && ensemble_rmse > best_ensemble_rmse * 1.05)
            break;
    }

    free(ensemble_pred);

    /* Mark best_k members to keep */
    for (size_t r = 0; r < best_k; r++) {
        pr.keep_mask[ranked[r]] = 1;
        pr.n_keep++;
    }
    pr.n_prune = ens->count - pr.n_keep;
    pr.pruned_rmse = best_ensemble_rmse;
    pr.improvement = member_rmse[ranked[0]] - best_ensemble_rmse;

    free(member_rmse);
    free(ranked);

    return pr;
}

/* ====================================================================
 * Ensemble Metrics
 * ==================================================================== */

double ensemble_recent_performance(const EnsembleState *ens)
{
    if (!ens || ens->count == 0) return 1e10;

    double weighted_rmse = 0.0, total_w = 0.0;
    for (size_t i = 0; i < ens->count; i++) {
        if (ens->members[i].is_active && !ens->members[i].retired) {
            double w = ens->member_weights[i];
            weighted_rmse += w * ens->members[i].recent_rmse;
            total_w += w;
        }
    }
    return (total_w > 1e-15) ? weighted_rmse / total_w : 1e10;
}

int ensemble_is_degrading(const EnsembleState *ens)
{
    if (!ens) return 0;
    /* Degrading if recent RMSE > cumulative RMSE by factor > 1.2 */
    if (ens->cumulative_rmse < 1e-10) return 0;
    return (ens->recent_rmse > ens->cumulative_rmse * 1.2) ? 1 : 0;
}

double ensemble_health_score(const EnsembleState *ens)
{
    if (!ens || ens->count == 0) return 0.0;

    double total_reliability = 0.0;
    size_t active = 0;
    for (size_t i = 0; i < ens->count; i++) {
        if (ens->members[i].is_active && !ens->members[i].retired) {
            total_reliability += ens->members[i].reliability;
            active++;
        }
    }
    return (active > 0) ? total_reliability / (double)active : 0.0;
}
