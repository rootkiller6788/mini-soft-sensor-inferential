/**
 * @file dr_core.h
 * @brief Core data structures and WLS solver for steady-state data reconciliation.
 *
 * Data Reconciliation (DR) adjusts measured process variables to satisfy
 * known physical constraints (mass/energy balances) while minimizing
 * the weighted sum of squared adjustments.
 *
 * Mathematical formulation:
 *   min  (x̂ - y)' Σ⁻¹ (x̂ - y)
 *   s.t. A·x̂ = b
 *
 * where:
 *   y  : vector of measured values (n×1)
 *   Σ  : measurement covariance matrix (n×n, symmetric positive definite)
 *   A  : constraint matrix (m×n)
 *   b  : constraint right-hand side (m×1)
 *
 * Solution via Lagrange multipliers (Kuehn & Davidson, 1961):
 *   x̂ = y − Σ·A'·(A·Σ·A')⁻¹·(A·y − b)
 *   λ = (A·Σ·A')⁻¹·(A·y − b)
 *
 * References:
 *   [1] Kuehn, D.R., Davidson, H. (1961). "Computer Control II:
 *       Mathematics of Control." Chem. Eng. Progress, 57(6), 44-47.
 *   [2] Crowe, C.M., et al. (1983). "Reconciliation of Process Flow
 *       Rates by Matrix Projection." AIChE Journal, 29(6), 881-888.
 *   [3] VDI 2048 Part 1 (2008). "Control and quality improvement of
 *       process data."
 *   [4] Narasimhan, S., Jordache, C. (2000). "Data Reconciliation &
 *       Gross Error Detection." Gulf Publishing.
 *   [5] Romagnoli, J.A., Sanchez, M.C. (2000). "Data Processing and
 *       Reconciliation for Chemical Process Operations." Academic Press.
 *
 * @author Mini-Control-Engineering-Practice
 * @date 2026
 */

#ifndef DR_CORE_H
#define DR_CORE_H

#include <stddef.h>
#include <stdint.h>

/* Maximum supported problem dimension (adjustable at compile time). */
#ifndef DR_MAX_DIM
#define DR_MAX_DIM 512
#endif

/**
 * @brief Classification of process variables in data reconciliation.
 *
 * In the Crowe projection framework (Crowe et al., 1983), each variable
 * is classified into one of four categories based on the constraint matrix
 * structure and measurement availability.
 */
typedef enum {
    DR_VAR_MEASURED_REDUNDANT      = 0,
    DR_VAR_MEASURED_NONREDUNDANT   = 1,
    DR_VAR_UNMEASURED_OBSERVABLE   = 2,
    DR_VAR_UNMEASURED_UNOBSERVABLE = 3
} dr_var_class_t;

/**
 * @brief Status codes for data reconciliation operations.
 */
typedef enum {
    DR_OK                    =  0,
    DR_ERR_NULL_POINTER      = -1,
    DR_ERR_DIM_MISMATCH      = -2,
    DR_ERR_SINGULAR          = -3,
    DR_ERR_NOT_SPD           = -4,
    DR_ERR_RANK_DEFICIENT    = -5,
    DR_ERR_DIM_EXCEEDED      = -6,
    DR_ERR_NOT_CONVERGED     = -7,
    DR_ERR_NON_REDUNDANT     = -8,
    DR_ERR_GROSS_ERROR_DETECTED = -9
} dr_status_t;

/**
 * @brief Solver method selection for data reconciliation.
 */
typedef enum {
    DR_SOLVER_LAGRANGE   = 0,
    DR_SOLVER_QR_ORTHOG  = 1,
    DR_SOLVER_CHOLESKY   = 2,
    DR_SOLVER_SVD        = 3,
    DR_SOLVER_IRLS       = 4
} dr_solver_t;

/**
 * @brief A single process measurement with uncertainty.
 *
 * According to GUM (JCGM 100:2008), the standard uncertainty u(y)
 * is the estimated standard deviation of the measurement.
 */
typedef struct {
    double value;        /**< Measured value (engineering units) */
    double stddev;       /**< Standard deviation / standard uncertainty σ */
    int    tag_id;       /**< Sensor tag identifier for traceability */
    int    is_present;   /**< 1 if measurement is available, 0 if missing */
    int    is_validated; /**< 1 if passed gross error check */
} dr_measurement_t;

/**
 * @brief Constraint type classification.
 */
typedef enum {
    DR_CONSTRAINT_MASS        = 0,
    DR_CONSTRAINT_COMPONENT   = 1,
    DR_CONSTRAINT_ENERGY      = 2,
    DR_CONSTRAINT_NORMALIZE   = 3,
    DR_CONSTRAINT_EQUIPMENT   = 4
} dr_constraint_type_t;

/**
 * @brief Sparsity pattern entry for a constraint row.
 */
typedef struct {
    int    col;
    double coeff;
} dr_sparse_entry_t;

/**
 * @brief Problem description for steady-state data reconciliation.
 *
 * Degrees of freedom for redundancy:
 *   ν = m − rank(A_u)  (where A_u is submatrix of unmeasured columns)
 *   If ν > 0, there is spatial redundancy.
 */
typedef struct {
    int                nvar;
    int                ncon;
    int                nmeas;
    dr_measurement_t  *measurements;
    double            *constraint_A;
    double            *constraint_b;
    dr_constraint_type_t *constraint_types;
    char             **var_names;
    char             **con_names;
} dr_problem_t;

/**
 * @brief Result of data reconciliation.
 *
 * Contains reconciled values, Lagrange multipliers, constraint residuals,
 * and diagnostic information.
 */
typedef struct {
    int     nvar;
    int     ncon;
    double *x_reconciled;
    double *x_adjustments;
    double *lagrange_mult;
    double *constraint_resid;
    double  objective;
    double  chi2_threshold;
    int     iterations;
    int     converged;
    dr_var_class_t *var_class;
} dr_result_t;

/* ---- Core API ---------------------------------------------------------- */

dr_problem_t *dr_problem_create(int nvar, int ncon);
void dr_problem_free(dr_problem_t *prob);

int dr_set_measurement(dr_problem_t *prob, int index,
                       double value, double stddev, int tag_id);

int dr_set_constraint(dr_problem_t *prob, int con_index,
                      const double *coeffs, double rhs,
                      dr_constraint_type_t ctype);

int dr_build_diag_covariance(const dr_problem_t *prob, double *cov_out);

int dr_solve(const dr_problem_t *prob, dr_result_t *result, dr_solver_t solver);

dr_result_t *dr_result_create(int nvar, int ncon);
void dr_result_free(dr_result_t *result);

int dr_global_test(const dr_problem_t *prob, const double *x_recon,
                   double *stat_out, int *df_out);

void dr_result_print(const dr_result_t *result);

int dr_reconciled_covariance(const dr_problem_t *prob, double *cov_out);

int dr_compute_redundancy(const dr_problem_t *prob, int *redundancy);

int dr_compute_V_matrix(const dr_problem_t *prob, double *V_out);

#endif /* DR_CORE_H */
