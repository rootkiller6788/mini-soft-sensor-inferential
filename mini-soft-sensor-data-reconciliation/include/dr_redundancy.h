/**
 * @file dr_redundancy.h
 * @brief Redundancy analysis and observability for data reconciliation.
 *
 * Redundancy analysis determines which process variables can be estimated
 * from measurements and which require additional instrumentation.
 *
 * Key concepts:
 *   - Spatial redundancy: A variable is measured more than once (directly or
 *     via constraints with other measured variables).
 *   - Temporal redundancy: Multiple measurements over time provide redundancy.
 *   - Observability: An unmeasured variable is observable if it can be
 *     uniquely determined from the constraint equations and measurements.
 *   - Degree of redundancy: The maximum number of measurements that can be
 *     eliminated and still allow estimation.
 *
 * Variable classification (Madron, 1992; VDI 2048):
 *   1. Measurable variables: Can be directly measured
 *   2. Redundant: Measured AND can also be estimated from other measurements
 *   3. Non-redundant: Measured but cannot be cross-checked
 *   4. Observable: Unmeasured but calculable from constraints
 *   5. Unobservable: Neither measured nor calculable
 *
 * Algorithm overview:
 *   1. Partition constraint matrix A = [A1 | A2] where A1 corresponds to
 *      measured variables and A2 to unmeasured variables.
 *   2. Perform QR factorization on A2^T to identify observable unmeasured
 *      variables via the null space structure.
 *   3. Classify each variable based on the factorization result.
 *
 * References:
 *   [1] Madron, F. (1992). "Process Plant Performance: Measurement and
 *       Data Processing for Optimization and Retrofits." Ellis Horwood.
 *   [2] VDI 2048 Part 1 (2008). "Control and quality improvement of
 *       process data - Control of measurement uncertainties."
 *   [3] Crowe, C.M. (1986). "Reconciliation of Process Flow Rates by
 *       Matrix Projection. Part II: The Nonlinear Case." AIChE Journal,
 *       32(4), 616-623.
 *   [4] van der Heijden, W.P.A.M., et al. (1994). "Classification of
 *       Variables in Mass Balance Reconciliation." Comp. Chem. Eng.,
 *       18(10), 967-974.
 *   [5] Stanley, G.M., Mah, R.S.H. (1981). "Observability and
 *       Redundancy in Process Data Estimation." Chem. Eng. Sci.,
 *       36(2), 259-272.
 */

#ifndef DR_REDUNDANCY_H
#define DR_REDUNDANCY_H

#include "dr_core.h"

/* ============================================================================
 * L1: Redundancy Classification Structures
 * ============================================================================ */

/**
 * @brief Detailed redundancy analysis result for a single variable.
 */
typedef struct {
    int     var_index;           /**< Variable index (0-based) */
    char   *var_name;            /**< Variable name */
    dr_var_class_t var_class;    /**< Classification */
    int     degree_of_redundancy; /**< 0 = non-redundant, >0 = number of extra estimates */
    int     is_balanceable;      /**< 1 if variable participates in closed balance */
    int     num_constraints;     /**< Number of constraints involving this variable */
    int     is_measured;         /**< 1 if measurement exists */
    double  estimability_index;  /**< 0-1: how well the variable can be estimated */
} dr_redundancy_info_t;

/**
 * @brief Overall redundancy analysis for the entire process network.
 */
typedef struct {
    int     nvar;                /**< Total number of variables */
    int     ncon;                /**< Total number of constraints */
    int     n_measured;          /**< Number of measured variables */
    int     n_redundant;         /**< Number of redundant measured variables */
    int     n_observable;        /**< Number of observable unmeasured variables */
    int     n_unobservable;      /**< Number of unobservable variables */
    int     n_balanceable;       /**< Number of balanceable variables */
    int     total_redundancy;    /**< Sum of redundancy degrees across all variables */
    double  global_redundancy;   /**< Overall redundancy ratio (0-1) */
    int     rank_A;              /**< Rank of constraint matrix A */
    int     rank_A2;             /**< Rank of unmeasured submatrix A2 */
    int     degrees_of_freedom;  /**< Statistical degrees of freedom for DR */
    dr_redundancy_info_t *vars;  /**< Per-variable analysis (length nvar) */
} dr_redundancy_t;

/* ============================================================================
 * L2: Incidence Matrix Analysis
 * ============================================================================ */

/**
 * @brief Build the incidence matrix from a set of constraints.
 *
 * The incidence matrix M indicates which variables appear in which
 * constraints: M(i,j) = 1 if variable j appears in constraint i,
 * M(i,j) = 0 otherwise.
 *
 * @param prob  Problem with constraints defined.
 * @param M     Output incidence matrix (row-major, ncon x nvar).
 * @return      DR_OK or error code.
 *
 * Complexity: O(ncon * nvar).
 */
int dr_red_build_incidence(const dr_problem_t *prob, int *M);

/**
 * @brief Compute the structural rank (maximum matching) of the
 * incidence matrix using the Dulmage-Mendelsohn decomposition.
 *
 * The structural rank is an upper bound on the algebraic rank of a
 * matrix. It equals the algebraic rank generically (for almost all
 * numerical values of the non-zero entries).
 *
 * @param M        Incidence matrix (row-major, ncon x nvar).
 * @param ncon     Number of constraints (rows).
 * @param nvar     Number of variables (columns).
 * @param max_match Output: column assignment for maximum matching.
 * @return         Structural rank.
 *
 * Complexity: O(ncon * nvar * sqrt(ncon + nvar)) (Hopcroft-Karp).
 *
 * Reference: Dulmage, A.L., Mendelsohn, N.S. (1958). "Coverings of
 * Bipartite Graphs." Canadian J. Math., 10, 517-534.
 */
int dr_red_structural_rank(const int *M, int ncon, int nvar, int *max_match);

/* ============================================================================
 * L3: Variable Classification
 * ============================================================================ */

/**
 * @brief Perform full variable classification via QR decomposition
 * of the partitioned constraint matrix.
 *
 * Algorithm (Crowe et al., 1983; Madron, 1992):
 *   1. Identify measured variable columns → submatrix A1
 *   2. Identify unmeasured variable columns → submatrix A2
 *   3. QR factorization of A2:
 *        A2^T = Q * [R1; 0] (full QR)
 *   4. Q^T * A = [A11 A12; A21 0]
 *   5. Redundant if A21 has rows → constraint residuals independent
 *      of unmeasured variables
 *
 * @param prob      Problem with measurements and constraints set.
 * @param result    Output redundancy analysis.
 * @return          DR_OK or error code.
 *
 * Complexity: O(m * n^2 + n^3) for QR.
 *
 * Reference: Crowe, C.M., et al. (1983). "Reconciliation of Process
 * Flow Rates by Matrix Projection." AIChE Journal, 29(6), 881-888.
 */
int dr_red_classify_variables(const dr_problem_t *prob, dr_redundancy_t *result);

/**
 * @brief Allocate and initialize a redundancy analysis structure.
 *
 * @param nvar  Number of variables.
 * @param ncon  Number of constraints.
 * @return      Pointer to allocated structure, or NULL.
 *
 * Complexity: O(nvar).
 */
dr_redundancy_t *dr_red_create(int nvar, int ncon);

/**
 * @brief Free a redundancy analysis structure.
 *
 * @param red  Structure to free. Safe to pass NULL.
 */
void dr_red_free(dr_redundancy_t *red);

/**
 * @brief Print a formatted redundancy analysis report.
 *
 * @param red  Redundancy analysis results.
 *
 * Complexity: O(nvar).
 */
void dr_red_print(const dr_redundancy_t *red);

/* ============================================================================
 * L4: Observability Analysis
 * ============================================================================ */

/**
 * @brief Check if a specific unmeasured variable is observable.
 *
 * An unmeasured variable is observable if it can be uniquely determined
 * from the constraint equations given the measured variables.
 *
 * @param prob       Problem description.
 * @param var_index  Variable index to check.
 * @return           1 if observable, 0 if not, -1 on error.
 *
 * Complexity: O(n^3) via QR analysis.
 */
int dr_red_is_observable(const dr_problem_t *prob, int var_index);

/**
 * @brief Identify the minimal set of additional measurements needed
 * to make all unobservable variables observable.
 *
 * This is the sensor placement problem for data reconciliation.
 * Uses a greedy algorithm based on the structural incidence matrix.
 *
 * @param prob            Problem description.
 * @param n_sensors_out   Output: number of new sensors needed.
 * @param sensor_indices  Output: variable indices needing measurement.
 *                        Must be pre-allocated (size >= nvar).
 * @return                DR_OK or error code.
 *
 * Complexity: O(ncon * nvar^2) greedy search.
 *
 * Reference: Madron, F., Veverka, V. (1992). "Optimal Selection of
 * Measuring Points in Complex Plants." Chem. Eng. Sci., 47(5),
 * 1207-1216.
 */
int dr_red_add_sensors(const dr_problem_t *prob, int *n_sensors_out,
                       int *sensor_indices);

/**
 * @brief Compute the Gram matrix G = A^T * A and its determinant
 * as a measure of linear independence of constraints.
 *
 * det(G) = 0 if and only if the constraint rows are linearly dependent.
 * This is useful for detecting redundant constraint equations.
 *
 * @param A       Constraint matrix (m x n, row-major).
 * @param m       Number of constraints.
 * @param n       Number of variables.
 * @param det_out Output: determinant of the Gram matrix.
 * @return        DR_OK or error code.
 *
 * Complexity: O(m * n * min(m,n) + min(m,n)^3).
 */
int dr_red_gram_determinant(const double *A, int m, int n, double *det_out);

/**
 * @brief Compute the null space basis of the constraint matrix
 * using SVD-like rank-revealing QR factorization.
 *
 * The null space vectors represent unobservable subspaces:
 * if a variable can be expressed as a linear combination of null
 * space vectors, it is unobservable.
 *
 * @param A        Constraint matrix (m x n, row-major).
 * @param m        Number of constraints.
 * @param n        Number of variables.
 * @param null_dim Output: dimension of null space (n - rank(A)).
 * @param null_basis Output: null space basis vectors (n x null_dim, row-major).
 *                  Must be pre-allocated.
 * @return         DR_OK or error code.
 *
 * Complexity: O(m * n^2).
 */
int dr_red_null_space(const double *A, int m, int n,
                      int *null_dim, double *null_basis);

/**
 * @brief Compute the redundancy degree for a single variable.
 *
 * The redundancy degree of variable j is:
 *   d_j = rank([a_j | A_{(-j)}]) - rank(A_{(-j)})
 *
 * where a_j is column j and A_{(-j)} is A without column j.
 * d_j = 1 means the column is linearly independent of the others
 * (variable is identifiable/balanceable).
 *
 * @param A      Constraint matrix (m x n, row-major).
 * @param m      Number of constraints.
 * @param n      Number of variables.
 * @param col    Column index to evaluate.
 * @return       Degree of redundancy (0 or 1), -1 on error.
 *
 * Complexity: O(m * n * min(m,n)).
 */
int dr_red_column_redundancy(const double *A, int m, int n, int col);

#endif /* DR_REDUNDANCY_H */
