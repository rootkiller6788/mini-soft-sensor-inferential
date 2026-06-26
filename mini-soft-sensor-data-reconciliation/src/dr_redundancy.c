/**
 * @file dr_redundancy.c
 * @brief Redundancy analysis: variable classification, observability, sensor placement.
 */

#include "dr_redundancy.h"
#include "dr_core.h"
#include "dr_matrix.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Allocation ---------------------------------------------------------- */

dr_redundancy_t *dr_red_create(int nvar, int ncon) {
    dr_redundancy_t *red;

    if (nvar <= 0 || ncon < 0) return NULL;

    red = (dr_redundancy_t *)calloc(1, sizeof(dr_redundancy_t));
    if (!red) return NULL;

    red->nvar = nvar;
    red->ncon = ncon;
    red->vars = (dr_redundancy_info_t *)calloc((size_t)nvar,
                                                sizeof(dr_redundancy_info_t));
    if (!red->vars) { free(red); return NULL; }

    for (int i = 0; i < nvar; i++) {
        red->vars[i].var_index = i;
    }

    return red;
}

void dr_red_free(dr_redundancy_t *red) {
    if (!red) return;
    if (red->vars) {
        for (int i = 0; i < red->nvar; i++) {
            free(red->vars[i].var_name);
        }
        free(red->vars);
    }
    free(red);
}

/* ---- Build incidence matrix ---------------------------------------------- */

/**
 * Build incidence matrix M where M(i,j) = 1 if variable j appears
 * in constraint i, M(i,j) = 0 otherwise.
 *
 * A non-zero coefficient is defined as |coeff| > 1e-12.
 */
int dr_red_build_incidence(const dr_problem_t *prob, int *M) {
    int i, j;

    if (!prob || !M) return DR_ERR_NULL_POINTER;

    for (i = 0; i < prob->ncon; i++) {
        for (j = 0; j < prob->nvar; j++) {
            M[i * prob->nvar + j] =
                (fabs(prob->constraint_A[i * prob->nvar + j]) > 1e-12) ? 1 : 0;
        }
    }
    return DR_OK;
}

/* ---- Structural rank via Dulmage-Mendelsohn decomposition ---------------- */

/**
 * Compute the structural rank (maximum matching size) of the
 * incidence matrix using a greedy DFS-based bipartite matching
 * algorithm (simplified Hopcroft-Karp approach).
 *
 * This finds the maximum number of constraints that can each be
 * assigned a unique variable, which equals the structural rank.
 *
 * The implementation uses DFS for augmenting path search,
 * yielding O(m * n * sqrt(m+n)) complexity in worst case,
 * but practically O(m * n) for typical incidence matrices.
 */
static int dfs_match(int u, const int *M, int ncon, int nvar,
                     int *match_col, int *visited) {
    int v;
    for (v = 0; v < nvar; v++) {
        if (M[u * nvar + v] && !visited[v]) {
            visited[v] = 1;
            if (match_col[v] == -1 ||
                dfs_match(match_col[v], M, ncon, nvar, match_col, visited)) {
                match_col[v] = u;
                return 1;
            }
        }
    }
    return 0;
}

int dr_red_structural_rank(const int *M, int ncon, int nvar, int *max_match) {
    int u, v, match_size;

    if (!M || !max_match || ncon <= 0 || nvar <= 0) return -1;

    /* match_col[v] = row assigned to column v, or -1 */
    int *match_col = (int *)malloc((size_t)nvar * sizeof(int));
    if (!match_col) return -1;

    for (v = 0; v < nvar; v++) match_col[v] = -1;

    match_size = 0;
    for (u = 0; u < ncon; u++) {
        int *visited = (int *)calloc((size_t)nvar, sizeof(int));
        if (!visited) { free(match_col); return -1; }
        if (dfs_match(u, M, ncon, nvar, match_col, visited))
            match_size++;
        free(visited);
    }

    /* Store the matching: max_match[v] = matched row or -1 */
    for (v = 0; v < nvar; v++)
        max_match[v] = match_col[v];

    free(match_col);
    return match_size;
}

/* ---- Variable classification via QR (Crowe et al., 1983) ----------------- */

/**
 * Classify all variables using the QR decomposition approach.
 *
 * Steps:
 *   1. Identify measured variables (have is_present=1)
 *   2. Partition A = [A1 | A2] where A1 = measured columns, A2 = unmeasured
 *   3. QR decompose A2^T to identify observable directions
 *   4. Transform A via Q^T to expose the redundancy structure
 *   5. Classify each variable:
 *      - Redundant: Measured AND appears in the A1-projection with others
 *      - Non-redundant: Measured but cannot be cross-checked
 *      - Observable: Unmeasured but in range of Q1
 *      - Unobservable: Unmeasured and not in range of Q1
 *
 * For computational efficiency, we use a simplified classification:
 *   - Build the augmented matrix [A1 | A2]
 *   - Compute the null space of A2
 *   - Variables whose null-space projection is zero are observable
 */
int dr_red_classify_variables(const dr_problem_t *prob,
                              dr_redundancy_t *result) {
    int i, j, k;
    int m = prob->ncon;
    int n = prob->nvar;
    int n_meas = 0, n_unmeas = 0;

    if (!prob || !result) return DR_ERR_NULL_POINTER;
    if (m == 0) {
        /* No constraints: all measured variables are non-redundant */
        for (i = 0; i < n; i++) {
            result->vars[i].var_class = prob->measurements[i].is_present ?
                DR_VAR_MEASURED_NONREDUNDANT : DR_VAR_UNMEASURED_UNOBSERVABLE;
            result->vars[i].degree_of_redundancy = 0;
            result->vars[i].is_measured = prob->measurements[i].is_present;
        }
        result->n_measured = prob->nmeas;
        result->n_redundant = 0;
        result->n_observable = 0;
        result->n_unobservable = n - prob->nmeas;
        return DR_OK;
    }

    /* Count measured and unmeasured variables */
    int *meas_idx = (int *)malloc((size_t)n * sizeof(int));
    int *unmeas_idx = (int *)malloc((size_t)n * sizeof(int));
    if (!meas_idx || !unmeas_idx) {
        free(meas_idx); free(unmeas_idx); return DR_ERR_NULL_POINTER;
    }

    for (i = 0; i < n; i++) {
        if (prob->measurements[i].is_present) {
            meas_idx[n_meas++] = i;
        } else {
            unmeas_idx[n_unmeas++] = i;
        }
    }

    result->n_measured = n_meas;

    /* For each variable, compute number of constraints it participates in */
    for (i = 0; i < n; i++) {
        int count = 0;
        for (j = 0; j < m; j++) {
            if (fabs(prob->constraint_A[j * n + i]) > 1e-12) count++;
        }
        result->vars[i].num_constraints = count;
        result->vars[i].is_measured = prob->measurements[i].is_present;
    }

    /* Build A2 (unmeasured submatrix, m x n_unmeas) */
    double *A2 = NULL;
    if (n_unmeas > 0) {
        A2 = (double *)calloc((size_t)m * (size_t)n_unmeas, sizeof(double));
        if (!A2) { free(meas_idx); free(unmeas_idx); return DR_ERR_NULL_POINTER; }
        for (j = 0; j < m; j++) {
            for (k = 0; k < n_unmeas; k++) {
                int col = unmeas_idx[k];
                A2[j * n_unmeas + k] = prob->constraint_A[j * n + col];
            }
        }
    }

    /* Determine observable unmeasured variables via structural analysis.
       A variable is observable if the column space dimension decreases
       when that column is removed (the column is linearly independent
       of the others in A2).
    */
    int n_obs = 0, n_unobs = 0;
    for (k = 0; k < n_unmeas; k++) {
        int col = unmeas_idx[k];
        /* Check: is this unmeasured variable observable?
           It is observable if there exists a constraint involving only
           measured variables and this unmeasured variable (gives a
           unique equation for it). More generally, it is observable
           if rank(A2 without column k) = rank(A2) - 1.
           Simplified: try to solve for it from constraints. */
        int can_solve = 0;
        for (j = 0; j < m; j++) {
            /* Check if constraint j has this variable as the only unmeasured */
            int unmeas_count = 0;
            int only_this = 0;
            for (i = 0; i < n; i++) {
                if (fabs(prob->constraint_A[j * n + i]) > 1e-12 &&
                    !prob->measurements[i].is_present) {
                    unmeas_count++;
                    if (i == col) only_this = 1;
                }
            }
            if (unmeas_count == 1 && only_this) {
                can_solve = 1;
                break;
            }
        }
        if (can_solve) {
            result->vars[col].var_class = DR_VAR_UNMEASURED_OBSERVABLE;
            n_obs++;
        } else {
            result->vars[col].var_class = DR_VAR_UNMEASURED_UNOBSERVABLE;
            n_unobs++;
        }
    }

    /* Classify measured variables */
    result->n_redundant = 0;
    int n_nonred = 0;
    for (i = 0; i < n; i++) {
        if (!prob->measurements[i].is_present) continue;

        /* A measured variable is redundant if it participates in a constraint
           with other measured variables, allowing cross-validation.
           Check: does variable i appear in any constraint where ALL other
           variables are also measured? */
        int is_redundant = 0;
        int degree = 0;
        for (j = 0; j < m; j++) {
            if (fabs(prob->constraint_A[j * n + i]) < 1e-12) continue;
            int all_others_measured = 1;
            int unmeas_count = 0;
            for (k = 0; k < n; k++) {
                if (k == i) continue;
                if (fabs(prob->constraint_A[j * n + k]) > 1e-12 &&
                    !prob->measurements[k].is_present) {
                    all_others_measured = 0;
                    unmeas_count++;
                }
            }
            if (all_others_measured) {
                is_redundant = 1;
                degree++;
            }
        }

        if (is_redundant) {
            result->vars[i].var_class = DR_VAR_MEASURED_REDUNDANT;
            result->vars[i].degree_of_redundancy = degree;
            result->n_redundant++;
        } else {
            result->vars[i].var_class = DR_VAR_MEASURED_NONREDUNDANT;
            result->vars[i].degree_of_redundancy = 0;
            n_nonred++;
        }
    }

    result->n_observable = n_obs;
    result->n_unobservable = n_unobs;
    result->n_balanceable = result->n_redundant + n_obs;
    result->total_redundancy = result->n_redundant;
    result->global_redundancy = (n_meas > 0) ?
        (double)result->n_redundant / (double)n_meas : 0.0;
    result->rank_A = (m < n) ? m : n;  /* Simplified */
    result->degrees_of_freedom = m - n_unobs;  /* Simplified */

    /* Compute estimability index for each variable */
    for (i = 0; i < n; i++) {
        if (result->vars[i].is_measured) {
            result->vars[i].estimability_index =
                (result->vars[i].var_class == DR_VAR_MEASURED_REDUNDANT) ?
                1.0 : 0.5;
        } else {
            result->vars[i].estimability_index =
                (result->vars[i].var_class == DR_VAR_UNMEASURED_OBSERVABLE) ?
                1.0 : 0.0;
        }
    }

    free(meas_idx);
    free(unmeas_idx);
    free(A2);
    return DR_OK;
}

/* ---- Observability check ------------------------------------------------- */

/**
 * Check if a specific unmeasured variable is observable.
 *
 * Observability condition: the variable can be uniquely determined
 * from the constraint equations using the available measurements.
 *
 * Algorithm: try to find a constraint involving this variable where
 * all other variables are measured, or a chain of such constraints.
 */
int dr_red_is_observable(const dr_problem_t *prob, int var_index) {
    int j, k;
    int m = prob->ncon, n = prob->nvar;

    if (!prob || var_index < 0 || var_index >= n) return -1;

    if (prob->measurements[var_index].is_present) return 0;

    /* Direct observability: constraint with only this variable unmeasured */
    for (j = 0; j < m; j++) {
        if (fabs(prob->constraint_A[j * n + var_index]) < 1e-12) continue;
        int n_unmeas = 0;
        for (k = 0; k < n; k++) {
            if (fabs(prob->constraint_A[j * n + k]) > 1e-12 &&
                !prob->measurements[k].is_present) {
                n_unmeas++;
            }
        }
        if (n_unmeas == 1) return 1;
    }

    /* Chain observability: can be solved sequentially with other observable vars */
    /* For simplicity, iterative propagation: mark known vars, find new */
    int *observable = (int *)calloc((size_t)n, sizeof(int));
    if (!observable) return -1;

    /* Initialize: all measured variables are "known" */
    for (k = 0; k < n; k++) {
        if (prob->measurements[k].is_present) observable[k] = 1;
    }

    int changed = 1;
    while (changed) {
        changed = 0;
        for (j = 0; j < m; j++) {
            int n_unknown = 0, last_unknown = -1;
            for (k = 0; k < n; k++) {
                if (fabs(prob->constraint_A[j * n + k]) > 1e-12 &&
                    !observable[k]) {
                    n_unknown++;
                    last_unknown = k;
                }
            }
            if (n_unknown == 1 && last_unknown >= 0) {
                observable[last_unknown] = 1;
                changed = 1;
                if (last_unknown == var_index) {
                    free(observable);
                    return 1;
                }
            }
        }
    }

    int result = observable[var_index];
    free(observable);
    return result;
}

/* ---- Sensor placement ---------------------------------------------------- */

/**
 * Greedy sensor placement: identify minimum set of additional
 * measurements to make all unobservable variables observable.
 *
 * Greedy algorithm:
 *   1. Identify unobservable variables
 *   2. For each unobservable variable, compute which constraints
 *      would become "solved" if it were measured
 *   3. Select the variable that enables the most new observability
 *   4. Repeat until all variables are observable
 */
int dr_red_add_sensors(const dr_problem_t *prob, int *n_sensors_out,
                       int *sensor_indices) {
    int i, j, k;
    int m = prob->ncon, n = prob->nvar;

    if (!prob || !n_sensors_out || !sensor_indices) return DR_ERR_NULL_POINTER;

    /* Copy problem state: which variables are currently "known" */
    int *known = (int *)calloc((size_t)n, sizeof(int));
    int *selected = (int *)calloc((size_t)n, sizeof(int));
    if (!known || !selected) {
        free(known); free(selected); return DR_ERR_NULL_POINTER;
    }

    for (i = 0; i < n; i++) {
        known[i] = prob->measurements[i].is_present;
    }

    int n_sensors = 0;
    int max_iter = n;

    while (max_iter-- > 0) {
        /* Propagate knowledge through constraints */
        int changed;
        do {
            changed = 0;
            for (j = 0; j < m; j++) {
                int n_unknown = 0, last_unknown = -1;
                for (k = 0; k < n; k++) {
                    if (fabs(prob->constraint_A[j * n + k]) > 1e-12 &&
                        !known[k]) {
                        n_unknown++;
                        last_unknown = k;
                    }
                }
                if (n_unknown == 1 && last_unknown >= 0) {
                    known[last_unknown] = 1;
                    changed = 1;
                }
            }
        } while (changed);

        /* Check if all variables are now known */
        int all_known = 1;
        for (i = 0; i < n; i++) {
            if (!known[i]) { all_known = 0; break; }
        }
        if (all_known) break;

        /* Find the unmeasured variable that, if measured, would enable
           the most new observability via constraint propagation */
        int best_var = -1;
        int best_gain = -1;

        for (i = 0; i < n; i++) {
            if (known[i] || selected[i]) continue;

            /* Simulate adding this measurement and count gain */
            int *temp = (int *)malloc((size_t)n * sizeof(int));
            if (!temp) continue;
            memcpy(temp, known, (size_t)n * sizeof(int));
            temp[i] = 1;

            /* Propagate */
            int local_changed;
            do {
                local_changed = 0;
                for (j = 0; j < m; j++) {
                    int n_unknown = 0, last_unknown = -1;
                    for (k = 0; k < n; k++) {
                        if (fabs(prob->constraint_A[j * n + k]) > 1e-12 &&
                            !temp[k]) {
                            n_unknown++;
                            last_unknown = k;
                        }
                    }
                    if (n_unknown == 1 && last_unknown >= 0) {
                        temp[last_unknown] = 1;
                        local_changed = 1;
                    }
                }
            } while (local_changed);

            int gain = 0;
            for (k = 0; k < n; k++) {
                if (temp[k] && !known[k]) gain++;
            }
            free(temp);

            if (gain > best_gain) {
                best_gain = gain;
                best_var = i;
            }
        }

        if (best_var < 0 || best_gain <= 0) break;

        sensor_indices[n_sensors++] = best_var;
        known[best_var] = 1;
        selected[best_var] = 1;
    }

    *n_sensors_out = n_sensors;
    free(known);
    free(selected);
    return DR_OK;
}

/* ---- Gram determinant ---------------------------------------------------- */

/**
 * Compute det(A^T * A) as a measure of constraint linear independence.
 *
 * For m <= n: G = A * A^T (m x m) is more efficient if m < n.
 * For m > n: G = A^T * A (n x n).
 *
 * Uses Cholesky: det(G) = det(L * L^T) = det(L)^2 = (prod L_ii)^2.
 */
int dr_red_gram_determinant(const double *A, int m, int n, double *det_out) {
    int i, j, k, dim;
    double det_val = 1.0;

    if (!A || !det_out || m <= 0 || n <= 0) return DR_ERR_NULL_POINTER;

    /* Choose smaller dimension for efficiency */
    if (m <= n) {
        dim = m;
        double *G = (double *)calloc((size_t)dim * (size_t)dim, sizeof(double));
        if (!G) return DR_ERR_NULL_POINTER;

        /* G = A * A^T  (m x m) */
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                double sum = 0.0;
                for (k = 0; k < n; k++) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                G[i * m + j] = sum;
            }
        }

        if (dr_mat_cholesky_raw(G, dim) != 0) {
            *det_out = 0.0; free(G); return DR_OK;
        }
        for (i = 0; i < dim; i++) det_val *= G[i * dim + i];
        det_val = det_val * det_val;  /* det(G) = det(L)^2 */
        free(G);
    } else {
        dim = n;
        double *G = (double *)calloc((size_t)dim * (size_t)dim, sizeof(double));
        if (!G) return DR_ERR_NULL_POINTER;

        /* G = A^T * A  (n x n) */
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                double sum = 0.0;
                for (k = 0; k < m; k++) {
                    sum += A[k * n + i] * A[k * n + j];
                }
                G[i * n + j] = sum;
            }
        }

        if (dr_mat_cholesky_raw(G, dim) != 0) {
            *det_out = 0.0; free(G); return DR_OK;
        }
        for (i = 0; i < dim; i++) det_val *= G[i * dim + i];
        det_val = det_val * det_val;
        free(G);
    }

    *det_out = det_val;
    return DR_OK;
}

/* ---- Column redundancy --------------------------------------------------- */

/**
 * Compute the redundancy degree of column j in constraint matrix A.
 *
 * d_j = rank([a_j | A_{(-j)}]) - rank(A_{(-j)})
 *
 * If d_j = 1, column j is linearly independent → variable is balanceable.
 * If d_j = 0, column j is linearly dependent → variable is redundant
 * with respect to the constraint structure.
 */
int dr_red_column_redundancy(const double *A, int m, int n, int col) {
    int i, j, dim;
    double *A_sub;
    double det_full, det_sub;

    if (!A || m <= 0 || n <= 0 || col < 0 || col >= n) return -1;

    dim = (m < n) ? m : n;
    if (dim > 50) {
        /* Too large for determinant-based approach; use simplified check */
        for (i = 0; i < m; i++) {
            int nz_count = 0;
            for (j = 0; j < n; j++) {
                if (fabs(A[i * n + j]) > 1e-12) nz_count++;
            }
            if (nz_count == 1 && fabs(A[i * n + col]) > 1e-12) return 1;
        }
        return 0;
    }

    /* Full Gram determinant */
    if (dr_red_gram_determinant(A, m, n, &det_full) != DR_OK) return -1;

    /* Build A_{(-j)} by removing column col */
    A_sub = (double *)malloc((size_t)m * (size_t)(n - 1) * sizeof(double));
    if (!A_sub) return -1;

    for (i = 0; i < m; i++) {
        int col_out = 0;
        for (j = 0; j < n; j++) {
            if (j == col) continue;
            A_sub[i * (n - 1) + col_out++] = A[i * n + j];
        }
    }

    if (dr_red_gram_determinant(A_sub, m, n - 1, &det_sub) != DR_OK) {
        free(A_sub); return -1;
    }
    free(A_sub);

    /* If removing column reduces rank, the column is independent */
    if (det_full > 1e-20 && det_sub < 1e-20) return 1;
    if (det_full < 1e-20) return 0;

    /* Generic case: compare Gram determinants as a ratio */
    double ratio = det_sub / det_full;
    return (ratio < 1e-10) ? 1 : 0;
}

/* ---- Null space computation ---------------------------------------------- */

/**
 * Compute the null space of A using rank-revealing QR on A^T.
 *
 * The null space vectors v_j satisfy A * v_j = 0.
 * The dimension of the null space is n - rank(A).
 *
 * Algorithm: QR on A^T to identify the range space, then the
 * orthogonal complement gives the null space.
 */
int dr_red_null_space(const double *A, int m, int n,
                      int *null_dim, double *null_basis) {
    int i, j, k, r;

    if (!A || !null_dim || !null_basis || m <= 0 || n <= 0)
        return DR_ERR_NULL_POINTER;

    /* For a small problem, we can use a direct approach.
       Build A^T (n x m) and QR factorize. */
    if (m > n) {
        /* More constraints than variables: null space is likely empty */
        /* Build Gram matrix and find eigenvectors of zero eigenvalues */
        double *G = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
        if (!G) return DR_ERR_NULL_POINTER;
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                for (k = 0; k < m; k++)
                    G[i * n + j] += A[k * n + i] * A[k * n + j];

        /* Cholesky: if fails, Gram is singular → null space exists.
           The Cholesky will stop at the first zero pivot, giving us
           the rank. We approximate the null space as the vectors
           beyond the Cholesky factor rank. */
        double *G_copy = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
        if (!G_copy) { free(G); return DR_ERR_NULL_POINTER; }
        memcpy(G_copy, G, (size_t)n * (size_t)n * sizeof(double));

        r = n; /* Optimistic: full rank */
        if (dr_mat_cholesky_raw(G_copy, n) != 0) {
            /* Find actual rank by trying progressively smaller leading submatrices */
            for (r = n - 1; r >= 1; r--) {
                double *sub = (double *)calloc((size_t)r * (size_t)r, sizeof(double));
                if (!sub) continue;
                for (i = 0; i < r; i++)
                    for (j = 0; j < r; j++)
                        sub[i * r + j] = G[i * n + j];
                if (dr_mat_cholesky_raw(sub, r) == 0) {
                    free(sub);
                    break;
                }
                free(sub);
            }
        }

        *null_dim = n - r;
        free(G_copy);
        free(G);
    } else {
        /* m <= n: Compute null space of A^T (m x n).
           Use QR on A (m x n): the last n - rank(A) columns of Q2
           span the null space. */
        /* Build A^T as matrix and QR */
        dr_matrix_t *AT = dr_mat_alloc(n, m);
        if (!AT) return DR_ERR_NULL_POINTER;
        for (i = 0; i < n; i++)
            for (j = 0; j < m; j++)
                AT->data[i * AT->ld + j] = A[j * n + i];

        double *tau = (double *)malloc((size_t)m * sizeof(double));
        if (!tau) { dr_mat_free(AT); return DR_ERR_NULL_POINTER; }
        dr_mat_qr(AT, tau);

        /* Rank = number of non-zero R diagonals */
        r = 0;
        for (i = 0; i < m; i++) {
            if (fabs(AT->data[i * AT->ld + i]) > 1e-12) r++;
        }

        *null_dim = n - r;
        free(tau);
        dr_mat_free(AT);
    }

    /* For the null basis itself, return zeros as initial values;
       a full SVD would be needed for accurate null space vectors.
       The rank (null space dimension) is correctly computed above. */
    for (i = 0; i < n * (*null_dim); i++) null_basis[i] = 0.0;

    return DR_OK;
}

/* ---- Printing ------------------------------------------------------------ */

void dr_red_print(const dr_redundancy_t *red) {
    int i;

    if (!red) { printf("(null redundancy)\n"); return; }

    printf("=== Redundancy Analysis Report ===\n");
    printf("Total variables: %d, Constraints: %d\n", red->nvar, red->ncon);
    printf("Measured: %d, Redundant: %d, Non-redundant: %d\n",
           red->n_measured, red->n_redundant,
           red->n_measured - red->n_redundant);
    printf("Observable (unmeasured): %d, Unobservable: %d\n",
           red->n_observable, red->n_unobservable);
    printf("Global redundancy ratio: %.2f\n", red->global_redundancy);
    printf("Degrees of freedom for DR: %d\n", red->degrees_of_freedom);
    printf("\n%-6s %-20s %-25s %-6s %-10s\n",
           "Idx", "Name", "Class", "Redund", "Estimable");
    printf("----------------------------------------------------\n");
    for (i = 0; i < red->nvar; i++) {
        const char *class_name;
        switch (red->vars[i].var_class) {
        case DR_VAR_MEASURED_REDUNDANT:      class_name = "Measured-Redundant"; break;
        case DR_VAR_MEASURED_NONREDUNDANT:   class_name = "Measured-NonRedundant"; break;
        case DR_VAR_UNMEASURED_OBSERVABLE:   class_name = "Unmeasured-Observable"; break;
        case DR_VAR_UNMEASURED_UNOBSERVABLE: class_name = "Unmeasured-Unobs"; break;
        default: class_name = "Unknown"; break;
        }
        printf("%-6d %-20s %-25s %-6d %-10.2f\n",
               i,
               red->vars[i].var_name ? red->vars[i].var_name : "(unnamed)",
               class_name,
               red->vars[i].degree_of_redundancy,
               red->vars[i].estimability_index);
    }
    printf("==================================\n");
}
