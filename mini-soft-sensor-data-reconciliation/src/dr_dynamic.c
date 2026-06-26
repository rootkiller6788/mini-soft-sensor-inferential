/**
 * @file dr_dynamic.c
 * @brief Dynamic data reconciliation via Kalman filter and MHE.
 */
#include "dr_dynamic.h"
#include "dr_core.h"
#include "dr_matrix.h"
#include "dr_measurement.h"
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Model lifecycle ------------------------------------------------- */

dr_dyn_model_t *dr_dyn_model_create(int nx, int nu, int ny, int nw) {
    dr_dyn_model_t *m;
    if (nx <= 0 || ny < 0 || nu < 0 || nw < 0) return NULL;
    m = (dr_dyn_model_t *)calloc(1, sizeof(dr_dyn_model_t));
    if (!m) return NULL;
    m->n_states = nx;
    m->n_inputs = nu;
    m->n_measurements = ny;
    m->n_noise = nw;
    m->F = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
    m->B = (double *)calloc((size_t)nx * (size_t)nu, sizeof(double));
    m->H = (double *)calloc((size_t)ny * (size_t)nx, sizeof(double));
    m->Q = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
    m->R = (double *)calloc((size_t)ny * (size_t)ny, sizeof(double));
    m->G = (double *)calloc((size_t)nx * (size_t)nw, sizeof(double));
    if (!m->F || !m->B || !m->H || !m->Q || !m->R || !m->G) {
        dr_dyn_model_free(m); return NULL;
    }
    for (int i = 0; i < nx; i++) { m->F[i * nx + i] = 1.0; m->Q[i * nx + i] = 1.0; }
    for (int i = 0; i < ny; i++) m->R[i * ny + i] = 1.0;
    return m;
}

void dr_dyn_model_free(dr_dyn_model_t *model) {
    if (!model) return;
    free(model->F); free(model->B); free(model->H);
    free(model->Q); free(model->R); free(model->G);
    free(model);
}

int dr_dyn_model_set_F(dr_dyn_model_t *model, const double *F) {
    if (!model || !F) return DR_ERR_NULL_POINTER;
    memcpy(model->F, F, (size_t)model->n_states * (size_t)model->n_states * sizeof(double));
    return DR_OK;
}

int dr_dyn_model_set_H(dr_dyn_model_t *model, const double *H) {
    if (!model || !H) return DR_ERR_NULL_POINTER;
    memcpy(model->H, H, (size_t)model->n_measurements * (size_t)model->n_states * sizeof(double));
    return DR_OK;
}

int dr_dyn_model_set_noise(dr_dyn_model_t *model, const double *Q, const double *R) {
    if (!model) return DR_ERR_NULL_POINTER;
    if (Q) memcpy(model->Q, Q, (size_t)model->n_states * (size_t)model->n_states * sizeof(double));
    if (R) memcpy(model->R, R, (size_t)model->n_measurements * (size_t)model->n_measurements * sizeof(double));
    return DR_OK;
}

/* ---- Kalman filter lifecycle ----------------------------------------- */

int dr_kf_init(dr_kf_state_t *kf, int nx, int ny, const double *x0, const double *P0) {
    if (!kf || !x0 || !P0 || nx <= 0 || ny < 0) return DR_ERR_NULL_POINTER;
    kf->n_states = nx;
    kf->x_hat = (double *)malloc((size_t)nx * sizeof(double));
    kf->P = (double *)malloc((size_t)nx * (size_t)nx * sizeof(double));
    kf->K = (double *)calloc((size_t)nx * (size_t)ny, sizeof(double));
    kf->innovation = (double *)calloc((size_t)ny, sizeof(double));
    kf->S = (double *)calloc((size_t)ny * (size_t)ny, sizeof(double));
    if (!kf->x_hat || !kf->P || !kf->K || !kf->innovation || !kf->S) {
        dr_kf_free(kf); return DR_ERR_NULL_POINTER;
    }
    memcpy(kf->x_hat, x0, (size_t)nx * sizeof(double));
    memcpy(kf->P, P0, (size_t)nx * (size_t)nx * sizeof(double));
    kf->log_likelihood = 0.0;
    kf->k = 0;
    kf->initialized = 1;
    return DR_OK;
}

void dr_kf_free(dr_kf_state_t *kf) {
    if (!kf) return;
    free(kf->x_hat); free(kf->P); free(kf->K);
    free(kf->innovation); free(kf->S);
    kf->initialized = 0;
}

int dr_kf_predict(dr_kf_state_t *kf, const dr_dyn_model_t *model, const double *u) {
    int i, j, k, nx;
    if (!kf || !model || !kf->initialized) return DR_ERR_NULL_POINTER;
    nx = model->n_states;
    double *Fx = (double *)calloc((size_t)nx, sizeof(double));
    if (!Fx) return DR_ERR_NULL_POINTER;
    for (i = 0; i < nx; i++) {
        double sum = 0.0;
        for (j = 0; j < nx; j++) sum += model->F[i * nx + j] * kf->x_hat[j];
        Fx[i] = sum;
    }
    if (u && model->n_inputs > 0) {
        for (i = 0; i < nx; i++) {
            double sum = 0.0;
            for (j = 0; j < model->n_inputs; j++)
                sum += model->B[i * model->n_inputs + j] * u[j];
            Fx[i] += sum;
        }
    }
    for (i = 0; i < nx; i++) kf->x_hat[i] = Fx[i];
    double *FP_temp = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
    if (!FP_temp) { free(Fx); return DR_ERR_NULL_POINTER; }
    for (i = 0; i < nx; i++)
        for (j = 0; j < nx; j++) {
            double sum = 0.0;
            for (k = 0; k < nx; k++)
                sum += model->F[i * nx + k] * kf->P[k * nx + j];
            FP_temp[i * nx + j] = sum;
        }
    double *P_new = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
    if (!P_new) { free(FP_temp); free(Fx); return DR_ERR_NULL_POINTER; }
    for (i = 0; i < nx; i++)
        for (j = 0; j < nx; j++) {
            double sum = 0.0;
            for (k = 0; k < nx; k++)
                sum += FP_temp[i * nx + k] * model->F[j * nx + k];
            P_new[i * nx + j] = sum;
        }
    if (model->n_noise > 0) {
        double *GQ = (double *)calloc((size_t)nx * (size_t)model->n_noise, sizeof(double));
        if (GQ) {
            for (i = 0; i < nx; i++)
                for (j = 0; j < model->n_noise; j++) {
                    double sum = 0.0;
                    for (k = 0; k < nx; k++)
                        sum += model->G[i * nx + k] * model->Q[k * nx + j];
                    GQ[i * model->n_noise + j] = sum;
                }
            for (i = 0; i < nx; i++)
                for (j = 0; j < nx; j++) {
                    double sum = 0.0;
                    for (k = 0; k < model->n_noise; k++)
                        sum += GQ[i * model->n_noise + k] * model->G[j * nx + k];
                    P_new[i * nx + j] += sum;
                }
            free(GQ);
        }
    } else {
        for (i = 0; i < nx; i++)
            for (j = 0; j < nx; j++)
                P_new[i * nx + j] += model->Q[i * nx + j];
    }
    memcpy(kf->P, P_new, (size_t)nx * (size_t)nx * sizeof(double));
    kf->k++;
    free(P_new); free(FP_temp); free(Fx);
    return DR_OK;
}

int dr_kf_update(dr_kf_state_t *kf, const dr_dyn_model_t *model, const double *y) {
    int i, j, k, nx, ny;
    if (!kf || !model || !y || !kf->initialized) return DR_ERR_NULL_POINTER;
    nx = model->n_states; ny = model->n_measurements;
    if (ny == 0) return DR_OK;
    for (i = 0; i < ny; i++) {
        double sum = 0.0;
        for (j = 0; j < nx; j++) sum += model->H[i * nx + j] * kf->x_hat[j];
        kf->innovation[i] = y[i] - sum;
    }
    double *HP = (double *)calloc((size_t)ny * (size_t)nx, sizeof(double));
    if (!HP) return DR_ERR_NULL_POINTER;
    for (i = 0; i < ny; i++)
        for (j = 0; j < nx; j++) {
            double sum = 0.0;
            for (k = 0; k < nx; k++)
                sum += model->H[i * nx + k] * kf->P[k * nx + j];
            HP[i * nx + j] = sum;
        }
    for (i = 0; i < ny; i++)
        for (j = 0; j < ny; j++) {
            double sum = 0.0;
            for (k = 0; k < nx; k++)
                sum += HP[i * nx + k] * model->H[j * nx + k];
            kf->S[i * ny + j] = sum + model->R[i * ny + j];
        }
    double *S_copy = (double *)malloc((size_t)ny * (size_t)ny * sizeof(double));
    if (!S_copy) { free(HP); return DR_ERR_NULL_POINTER; }
    memcpy(S_copy, kf->S, (size_t)ny * (size_t)ny * sizeof(double));
    if (dr_mat_cholesky_raw(S_copy, ny) != 0) { free(S_copy); free(HP); return DR_ERR_SINGULAR; }

    double *PHt = (double *)calloc((size_t)nx * (size_t)ny, sizeof(double));
    if (!PHt) { free(S_copy); free(HP); return DR_ERR_NULL_POINTER; }
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
            double sum = 0.0;
            for (k = 0; k < nx; k++)
                sum += kf->P[i * nx + k] * model->H[j * nx + k];
            PHt[i * ny + j] = sum;
        }
    for (j = 0; j < nx; j++) {
        double *b = (double *)malloc((size_t)ny * sizeof(double));
        double *x = (double *)malloc((size_t)ny * sizeof(double));
        if (!b || !x) { free(x); free(b); free(PHt); free(S_copy); free(HP); return DR_ERR_NULL_POINTER; }
        for (i = 0; i < ny; i++) b[i] = PHt[j * ny + i];
        dr_mat_cholesky_solve_raw(S_copy, b, x, ny);
        for (i = 0; i < ny; i++) kf->K[j * ny + i] = x[i];
        free(x); free(b);
    }
    for (i = 0; i < nx; i++) {
        double sum = 0.0;
        for (j = 0; j < ny; j++) sum += kf->K[i * ny + j] * kf->innovation[j];
        kf->x_hat[i] += sum;
    }
    /* Joseph form: P = (I-KH)*P*(I-KH)^T + K*R*K^T */
    double *KH = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
    if (KH) {
        for (i = 0; i < nx; i++)
            for (j = 0; j < nx; j++) {
                double sum = 0.0;
                for (k = 0; k < ny; k++)
                    sum += kf->K[i * ny + k] * model->H[k * nx + j];
                KH[i * nx + j] = sum;
            }
        double *IKH = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
        if (IKH) {
            for (i = 0; i < nx; i++)
                for (j = 0; j < nx; j++)
                    IKH[i * nx + j] = (i == j) ? (1.0 - KH[i * nx + j]) : (-KH[i * nx + j]);
            double *temp = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
            if (temp) {
                for (i = 0; i < nx; i++)
                    for (j = 0; j < nx; j++) {
                        double sum = 0.0;
                        for (k = 0; k < nx; k++)
                            sum += IKH[i * nx + k] * kf->P[k * nx + j];
                        temp[i * nx + j] = sum;
                    }
                for (i = 0; i < nx; i++)
                    for (j = 0; j < nx; j++) {
                        double sum = 0.0;
                        for (k = 0; k < nx; k++)
                            sum += temp[i * nx + k] * IKH[j * nx + k];
                        kf->P[i * nx + j] = sum;
                    }
                double *KR = (double *)calloc((size_t)nx * (size_t)ny, sizeof(double));
                if (KR) {
                    for (i = 0; i < nx; i++)
                        for (j = 0; j < ny; j++) {
                            double sum = 0.0;
                            for (k = 0; k < ny; k++)
                                sum += kf->K[i * ny + k] * model->R[k * ny + j];
                            KR[i * ny + j] = sum;
                        }
                    for (i = 0; i < nx; i++)
                        for (j = 0; j < nx; j++) {
                            double sum = 0.0;
                            for (k = 0; k < ny; k++)
                                sum += KR[i * ny + k] * kf->K[j * ny + k];
                            kf->P[i * nx + j] += sum;
                        }
                    free(KR);
                }
                free(temp);
            }
            free(IKH);
        }
        free(KH);
    }
    /* Update log-likelihood */
    double *innov_scaled = (double *)malloc((size_t)ny * sizeof(double));
    if (innov_scaled) {
        dr_mat_cholesky_solve_raw(S_copy, kf->innovation, innov_scaled, ny);
        double quad_form = 0.0;
        for (i = 0; i < ny; i++) quad_form += kf->innovation[i] * innov_scaled[i];
        double log_det_S = 0.0;
        for (i = 0; i < ny; i++) log_det_S += log(fabs(S_copy[i * ny + i]));
        log_det_S *= 2.0;
        kf->log_likelihood += -0.5 * ((double)ny * log(2.0 * M_PI) + log_det_S + quad_form);
        free(innov_scaled);
    }
    free(PHt); free(S_copy); free(HP);
    return DR_OK;
}

int dr_kf_smooth_rts(const dr_kf_state_t *kf_hist, const dr_dyn_model_t *model,
                     int N, double *x_smooth, double *P_smooth) {
    int i, j, k, nx, t;
    if (!kf_hist || !model || !x_smooth || !P_smooth || N <= 0)
        return DR_ERR_NULL_POINTER;
    nx = model->n_states;
    /* Copy last filtered estimate */
    for (i = 0; i < nx; i++) x_smooth[(N-1) * nx + i] = kf_hist[N-1].x_hat[i];
    for (i = 0; i < nx; i++)
        for (j = 0; j < nx; j++)
            P_smooth[(N-1) * nx * nx + i * nx + j] = kf_hist[N-1].P[i * nx + j];
    /* Backward pass */
    for (t = N - 2; t >= 0; t--) {
        /* Compute predicted P: P_pred = F * P(t) * F^T + Q */
        double *P_pred = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
        if (!P_pred) return DR_ERR_NULL_POINTER;
        double *FP = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
        if (FP) {
            for (i = 0; i < nx; i++)
                for (j = 0; j < nx; j++) {
                    double sum = 0.0;
                    for (k = 0; k < nx; k++)
                        sum += model->F[i * nx + k] * kf_hist[t].P[k * nx + j];
                    FP[i * nx + j] = sum;
                }
            for (i = 0; i < nx; i++)
                for (j = 0; j < nx; j++) {
                    double sum = 0.0;
                    for (k = 0; k < nx; k++)
                        sum += FP[i * nx + k] * model->F[j * nx + k];
                    P_pred[i * nx + j] = sum + model->Q[i * nx + j];
                }
            free(FP);
        }
        /* C = P(t) * F^T * P_pred^{-1} */
        double *PFt = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
        if (PFt) {
            for (i = 0; i < nx; i++)
                for (j = 0; j < nx; j++) {
                    double sum = 0.0;
                    for (k = 0; k < nx; k++)
                        sum += kf_hist[t].P[i * nx + k] * model->F[j * nx + k];
                    PFt[i * nx + j] = sum;
                }
            double *P_pred_copy = (double *)malloc((size_t)nx * (size_t)nx * sizeof(double));
            if (P_pred_copy) {
                memcpy(P_pred_copy, P_pred, (size_t)nx * (size_t)nx * sizeof(double));
                if (dr_mat_cholesky_raw(P_pred_copy, nx) == 0) {
                    double *C_col = (double *)malloc((size_t)nx * sizeof(double));
                    if (C_col) {
                        for (j = 0; j < nx; j++) {
                            for (i = 0; i < nx; i++) C_col[i] = PFt[i * nx + j];
                            dr_mat_cholesky_solve_raw(P_pred_copy, C_col, C_col, nx);
                            for (i = 0; i < nx; i++) PFt[i * nx + j] = C_col[i];
                        }
                        free(C_col);
                    }
                }
                free(P_pred_copy);
            }
            /* x_smooth = x_filtered + C * (x_smooth_next - x_predicted_next) */
            double *x_pred_next = (double *)calloc((size_t)nx, sizeof(double));
            if (x_pred_next) {
                for (i = 0; i < nx; i++) {
                    double sum = 0.0;
                    for (j = 0; j < nx; j++)
                        sum += model->F[i * nx + j] * kf_hist[t].x_hat[j];
                    x_pred_next[i] = sum;
                }
                double *diff = (double *)malloc((size_t)nx * sizeof(double));
                if (diff) {
                    for (i = 0; i < nx; i++)
                        diff[i] = x_smooth[(t+1) * nx + i] - x_pred_next[i];
                    for (i = 0; i < nx; i++) {
                        double sum = 0.0;
                        for (k = 0; k < nx; k++) sum += PFt[i * nx + k] * diff[k];
                        x_smooth[t * nx + i] = kf_hist[t].x_hat[i] + sum;
                    }
                    for (i = 0; i < nx; i++)
                        for (j = 0; j < nx; j++) {
                            double sum = 0.0;
                            for (k = 0; k < nx; k++)
                                sum += PFt[i * nx + k] *
                                       (P_smooth[(t+1)*nx*nx + k*nx + j] - P_pred[k*nx + j]);
                            P_smooth[t*nx*nx + i*nx + j] = kf_hist[t].P[i*nx + j] + sum;
                        }
                    free(diff);
                }
                free(x_pred_next);
            }
            free(PFt);
        }
        free(P_pred);
    }
    return DR_OK;
}

int dr_kf_innovation_whiteness(const double *innovations, int ny, int N,
                               int max_lag, double *Q_stat_out, int *is_white_out) {
    int lag, t, d;
    double Q_stat = 0.0;
    if (!innovations || !Q_stat_out || !is_white_out || N < 10 || max_lag < 1)
        return DR_ERR_NULL_POINTER;
    for (lag = 1; lag <= max_lag && lag < N; lag++) {
        double r_lag = 0.0;
        double denom = 0.0;
        for (t = 0; t < N - lag; t++) {
            double dot = 0.0;
            for (d = 0; d < ny; d++)
                dot += innovations[t * ny + d] * innovations[(t + lag) * ny + d];
            r_lag += dot;
        }
        for (t = 0; t < N; t++) {
            double dot = 0.0;
            for (d = 0; d < ny; d++)
                dot += innovations[t * ny + d] * innovations[t * ny + d];
            denom += dot;
        }
        if (denom > 1e-15) {
            double rho = r_lag / denom;
            Q_stat += (double)N * (double)(N + 2) * rho * rho / (double)(N - lag);
        }
    }
    *Q_stat_out = Q_stat;
    double chi2_crit = dr_meas_chi2_critical(max_lag, 0.05);
    *is_white_out = (Q_stat > chi2_crit) ? 0 : 1;
    return DR_OK;
}

int dr_kf_steady_gain(const dr_dyn_model_t *model, double *K_ss,
                      double *P_ss, int max_iter, double tol) {
    int i, j, k, iter, nx, ny;
    if (!model || !K_ss || !P_ss) return DR_ERR_NULL_POINTER;
    nx = model->n_states; ny = model->n_measurements;
    if (ny == 0) { for (i = 0; i < nx*nx; i++) P_ss[i] = 0.0; return DR_OK; }
    /* Initialize P with Q */
    for (i = 0; i < nx; i++)
        for (j = 0; j < nx; j++)
            P_ss[i * nx + j] = model->Q[i * nx + j];
    double *P_prev = (double *)malloc((size_t)nx * (size_t)nx * sizeof(double));
    if (!P_prev) return DR_ERR_NULL_POINTER;
    for (iter = 0; iter < max_iter; iter++) {
        memcpy(P_prev, P_ss, (size_t)nx * (size_t)nx * sizeof(double));
        /* S = H*P*H^T + R */
        double *S = (double *)calloc((size_t)ny * (size_t)ny, sizeof(double));
        if (!S) { free(P_prev); return DR_ERR_NULL_POINTER; }
        for (i = 0; i < ny; i++)
            for (j = 0; j < ny; j++) {
                double sum = 0.0;
                for (k = 0; k < nx; k++)
                    sum += model->H[i * nx + k] *
                           (P_ss[k * nx + 0] * 0 + 0);
                S[i * ny + j] = model->R[i * ny + j];
            }
        /* Recompute S properly */
        double *HP = (double *)calloc((size_t)ny * (size_t)nx, sizeof(double));
        if (!HP) { free(S); free(P_prev); return DR_ERR_NULL_POINTER; }
        for (i = 0; i < ny; i++)
            for (j = 0; j < nx; j++) {
                double sum = 0.0;
                for (k = 0; k < nx; k++)
                    sum += model->H[i * nx + k] * P_ss[k * nx + j];
                HP[i * nx + j] = sum;
            }
        for (i = 0; i < ny; i++)
            for (j = 0; j < ny; j++) {
                S[i * ny + j] = model->R[i * ny + j];
                for (k = 0; k < nx; k++)
                    S[i * ny + j] += HP[i * nx + k] * model->H[j * nx + k];
            }
        /* Cholesky on S */
        double *S_copy = (double *)malloc((size_t)ny * (size_t)ny * sizeof(double));
        if (!S_copy) { free(HP); free(S); free(P_prev); return DR_ERR_NULL_POINTER; }
        memcpy(S_copy, S, (size_t)ny * (size_t)ny * sizeof(double));
        if (dr_mat_cholesky_raw(S_copy, ny) != 0) { free(S_copy); free(HP); free(S); free(P_prev); return DR_ERR_SINGULAR; }
        /* K = P*H^T * S^{-1} */
        double *PHt = (double *)calloc((size_t)nx * (size_t)ny, sizeof(double));
        if (!PHt) { free(S_copy); free(HP); free(S); free(P_prev); return DR_ERR_NULL_POINTER; }
        for (i = 0; i < nx; i++)
            for (j = 0; j < ny; j++) {
                double sum = 0.0;
                for (k = 0; k < nx; k++)
                    sum += P_ss[i * nx + k] * model->H[j * nx + k];
                PHt[i * ny + j] = sum;
            }
        for (j = 0; j < nx; j++) {
            double *b = (double *)malloc((size_t)ny * sizeof(double));
            double *x = (double *)malloc((size_t)ny * sizeof(double));
            if (!b || !x) { free(x); free(b); break; }
            for (i = 0; i < ny; i++) b[i] = PHt[j * ny + i];
            dr_mat_cholesky_solve_raw(S_copy, b, x, ny);
            for (i = 0; i < ny; i++) K_ss[j * ny + i] = x[i];
            free(x); free(b);
        }
        /* Riccati: P = F*P*F^T - K*H*P*F^T + Q */
        double *FP = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
        if (FP) {
            for (i = 0; i < nx; i++)
                for (j = 0; j < nx; j++) {
                    double sum = 0.0;
                    for (k = 0; k < nx; k++)
                        sum += model->F[i * nx + k] * P_ss[k * nx + j];
                    FP[i * nx + j] = sum;
                }
            for (i = 0; i < nx; i++)
                for (j = 0; j < nx; j++) {
                    double sum = 0.0;
                    for (k = 0; k < nx; k++)
                        sum += FP[i * nx + k] * model->F[j * nx + k];
                    P_ss[i * nx + j] = sum + model->Q[i * nx + j];
                }
            /* Subtract K*S*K^T */
            for (i = 0; i < nx; i++)
                for (j = 0; j < nx; j++) {
                    double sum = 0.0;
                    for (k = 0; k < ny; k++) {
                        double KS_ik = 0.0;
                        for (int p = 0; p < ny; p++)
                            KS_ik += K_ss[i * ny + p] * S[p * ny + k];
                        sum += KS_ik * K_ss[j * ny + k];
                    }
                    P_ss[i * nx + j] -= sum;
                }
            free(FP);
        }
        /* Check convergence */
        double dmax = 0.0;
        for (i = 0; i < nx * nx; i++) {
            double d = fabs(P_ss[i] - P_prev[i]);
            if (d > dmax) dmax = d;
        }
        free(PHt); free(S_copy); free(HP); free(S);
        if (dmax < tol) break;
    }
    free(P_prev);
    return DR_OK;
}

int dr_mhe_init(dr_mhe_config_t *config, int nx, int ny, int horizon) {
    if (!config || nx <= 0 || ny < 0 || horizon <= 0) return DR_ERR_NULL_POINTER;
    config->n_states = nx;
    config->n_measurements = ny;
    config->horizon_length = horizon;
    config->use_constraints = 0;
    config->arrival_cost_P = (double *)calloc((size_t)nx * (size_t)nx, sizeof(double));
    config->arrival_cost_x = (double *)calloc((size_t)nx, sizeof(double));
    if (!config->arrival_cost_P || !config->arrival_cost_x)
        return DR_ERR_NULL_POINTER;
    for (int i = 0; i < nx; i++) config->arrival_cost_P[i * nx + i] = 1e6;
    return DR_OK;
}

void dr_mhe_free(dr_mhe_config_t *config) {
    if (!config) return;
    free(config->arrival_cost_P);
    free(config->arrival_cost_x);
}

int dr_mhe_solve(const dr_dyn_model_t *model, const dr_mhe_config_t *config,
                 const double *y_hist, const double *u_hist, double *x_out) {
    int i, j, k, p, N, nx, ny;
    if (!model || !config || !y_hist || !x_out) return DR_ERR_NULL_POINTER;
    N = config->horizon_length;
    nx = config->n_states;
    ny = config->n_measurements;
    if (N <= 0 || nx <= 0) return DR_ERR_DIM_MISMATCH;

    /* Build stacked least-squares problem:
       min ||x_0 - x_bar||^2_{P_bar^{-1}} + sum ||y_k - H*x_k||^2_{R^{-1}}
                                          + sum ||x_{k+1} - F*x_k||^2_{Q^{-1}}
       This is a large sparse least-squares problem.
       For small horizons, solve via dense stacking. */

    /* n_total = nx * (N + 1) would be all states in horizon */
    /* m_total = nx + ny * (N + 1) + nx * N for full MHE formulation */

    /* Simplified: use sequential Kalman-like approach as approximation */
    /* Initialize state estimate */
    for (i = 0; i < nx; i++) x_out[i] = config->arrival_cost_x[i];

    /* Forward pass */
    for (k = 0; k <= N; k++) {
        /* Measurement update */
        if (ny > 0) {
            double *innov = (double *)calloc((size_t)ny, sizeof(double));
            if (innov) {
                for (i = 0; i < ny; i++) {
                    double sum = 0.0;
                    for (j = 0; j < nx; j++)
                        sum += model->H[i * nx + j] * x_out[j];
                    innov[i] = y_hist[k * ny + i] - sum;
                }
                double *HPHt = (double *)calloc((size_t)ny * (size_t)ny, sizeof(double));
                if (HPHt) {
                    for (i = 0; i < ny; i++)
                        for (j = 0; j < ny; j++) {
                            HPHt[i * ny + j] = model->R[i * ny + j];
                            for (p = 0; p < nx; p++) {
                                double hp = model->H[i * nx + p];
                                double hq = model->H[j * nx + p];
                                HPHt[i * ny + j] += hp * config->arrival_cost_P[p * nx + p] * hq;
                            }
                        }
                    double *HPHt_copy = (double *)malloc((size_t)ny * (size_t)ny * sizeof(double));
                    if (HPHt_copy) {
                        memcpy(HPHt_copy, HPHt, (size_t)ny * (size_t)ny * sizeof(double));
                        if (dr_mat_cholesky_raw(HPHt_copy, ny) == 0) {
                            double *weights = (double *)malloc((size_t)ny * sizeof(double));
                            if (weights) {
                                dr_mat_cholesky_solve_raw(HPHt_copy, innov, weights, ny);
                                for (i = 0; i < nx; i++) {
                                    double corr = 0.0;
                                    for (j = 0; j < ny; j++)
                                        corr += model->H[j * nx + i] * weights[j];
                                    x_out[i] += config->arrival_cost_P[i * nx + i] * corr;
                                }
                                free(weights);
                            }
                        }
                        free(HPHt_copy);
                    }
                    free(HPHt);
                }
                free(innov);
            }
        }
        /* Prediction step */
        if (k < N) {
            double *Fx = (double *)calloc((size_t)nx, sizeof(double));
            if (Fx) {
                for (i = 0; i < nx; i++) {
                    double sum = 0.0;
                    for (j = 0; j < nx; j++)
                        sum += model->F[i * nx + j] * x_out[j];
                    Fx[i] = sum;
                }
                if (u_hist) {
                    for (i = 0; i < nx; i++) {
                        double sum = 0.0;
                        for (j = 0; j < model->n_inputs; j++)
                            sum += model->B[i * model->n_inputs + j] *
                                   u_hist[k * model->n_inputs + j];
                        Fx[i] += sum;
                    }
                }
                for (i = 0; i < nx; i++) x_out[i] = Fx[i];
                free(Fx);
            }
        }
    }
    return DR_OK;
}

int dr_mhe_update_arrival_cost(dr_mhe_config_t *config,
                               const dr_dyn_model_t *model,
                               const dr_kf_state_t *kf) {
    int i, j, nx;
    if (!config || !model || !kf) return DR_ERR_NULL_POINTER;
    nx = model->n_states;
    for (i = 0; i < nx; i++) config->arrival_cost_x[i] = kf->x_hat[i];
    for (i = 0; i < nx; i++)
        for (j = 0; j < nx; j++)
            config->arrival_cost_P[i * nx + j] = kf->P[i * nx + j];
    return DR_OK;
}
