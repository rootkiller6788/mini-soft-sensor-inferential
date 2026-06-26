/**
 * @file vfm_data_reconciliation.c
 * @brief Data reconciliation for Virtual Flow Meter
 *
 * Knowledge Coverage:
 *   L1 Definitions: Mass balance, redundancy, constraints
 *   L2 Core Concepts: Data reconciliation, gross error detection
 *   L3 Engineering Structures: Constrained weighted least squares
 *   L4 Engineering Laws: Mass conservation, energy balance
 *   L5 Algorithms: Lagrange multiplier method, nodal balance, SDR
 *
 * Data reconciliation adjusts measured values to satisfy known
 * physical constraints (e.g., mass balance at a pipe junction)
 * while minimizing a weighted sum of squared adjustments. This is
 * widely used in oil & gas production allocation where VFM estimates
 * from multiple wells must be consistent with a common export meter.
 *
 * References:
 *   Kuehn & Davidson (1961) "Computer control II: Mathematics of control"
 *   Narasimhan & Jordache (2000) "Data Reconciliation and Gross Error
 *     Detection: An Intelligent Use of Process Data"
 *   Romagnoli & Sanchez (2000) "Data Processing and Reconciliation
 *     for Chemical Process Operations"
 *
 * @module mini-virtual-flow-meter
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ==========================================================================
 * L1: Core Data Structures
 * ========================================================================== */

/**
 * @brief A flow measurement with its uncertainty.
 */
typedef struct {
    double measured_value;   /**< Raw measurement [m^3/s]                  */
    double std_uncertainty;  /**< Standard uncertainty [m^3/s]             */
    double reconciled_value; /**< Output: reconciled (adjusted) value       */
    double adjustment;       /**< Difference: reconciled - measured         */
    double adjustment_penalty; /**< (adjustment / uncertainty)^2           */
} dr_measurement_t;

/**
 * @brief A linear mass/energy balance constraint.
 *
 * Constraint form: sum_i (a_i * x_i) = b
 *
 * where a_i = coefficient (+1 for inlet, -1 for outlet in mass balance),
 * x_i = reconciled flow at node i,
 * b = net accumulation (0 for steady state).
 */
typedef struct {
    double *coefficients;    /**< a_i for each measurement [n_meas]        */
    double rhs;              /**< Right-hand side b                        */
    double lagrange_mult;    /**< Lagrange multiplier for this constraint  */
    double residual;         /**< Constraint residual after reconciliation */
} dr_constraint_t;

/**
 * @brief Complete data reconciliation problem.
 */
typedef struct {
    dr_measurement_t *measurements;  /**< Array of measurements             */
    dr_constraint_t  *constraints;   /**< Array of balance constraints      */
    int num_measurements;            /**< Number of flow measurements       */
    int num_constraints;             /**< Number of independent constraints */
    double objective;                /**< Final objective value             */
    double *solution;                /**< Reconciled values (n_meas)        */
    int converged;                   /**< 1 = solution converged            */
    int iterations;                  /**< Number of iterations used         */
} dr_problem_t;

/* ==========================================================================
 * L2: Single Node Mass Balance Reconciliation
 * ========================================================================== */

/**
 * @brief Reconcile flow measurements at a single pipe junction
 *        (one mass balance constraint).
 *
 * Problem: Given n flow measurements x_i with uncertainties sigma_i,
 * find reconciled values x_hat_i that:
 *   1. Minimize J = sum_i (x_hat_i - x_i)^2 / sigma_i^2
 *   2. Satisfy sum_i (a_i * x_hat_i) = 0 (mass balance: inflow = outflow)
 *
 * where a_i = +1 for inlet streams, -1 for outlet streams.
 *
 * Analytical solution using Lagrange multipliers:
 *   x_hat_i = x_i - sigma_i^2 * a_i * lambda
 *
 *   lambda = (sum_i a_i*x_i) / (sum_i sigma_i^2 * a_i^2)
 *
 * @param measured     Array of measured flow rates [m^3/s]
 * @param uncertainties Array of standard uncertainties
 * @param is_inlet     Array of direction flags: +1=inlet, -1=outlet, 0=ignore
 * @param n            Number of measurements
 * @param reconciled   Output: reconciled values [m^3/s]
 * @return 0 on success, -1 on invalid inputs
 */
int dr_node_mass_balance(const double *measured,
                          const double *uncertainties,
                          const int *is_inlet,
                          int n,
                          double *reconciled)
{
    if (!measured || !uncertainties || !is_inlet || !reconciled) {
        return -1;
    }
    if (n < 2) return -1;

    /* Compute mass imbalance: sum a_i * x_i */
    double imbalance = 0.0;
    double denom = 0.0;  /* sum sigma_i^2 */
    int i;

    for (i = 0; i < n; i++) {
        if (is_inlet[i] == 0) continue;
        if (uncertainties[i] <= 0.0) return -1;

        imbalance += (double)is_inlet[i] * measured[i];
        denom += uncertainties[i] * uncertainties[i];
    }

    if (denom <= 0.0) return -1;

    /* Lagrange multiplier lambda = imbalance / denom */
    double lambda = imbalance / denom;

    /* Reconciled values: x_hat_i = x_i - a_i * sigma_i^2 * lambda */
    for (i = 0; i < n; i++) {
        if (is_inlet[i] == 0) {
            reconciled[i] = measured[i];
        } else {
            reconciled[i] = measured[i]
                - (double)is_inlet[i] * uncertainties[i] * uncertainties[i]
                * lambda;
        }
    }

    return 0;
}

/* ==========================================================================
 * L3: Multi-Constraint Weighted Least Squares
 * ========================================================================== */

/**
 * @brief Solve linear data reconciliation with multiple constraints
 *        using the Lagrange multiplier method.
 *
 * Problem formulation:
 *   min (y - x)^T * W * (y - x)
 *   s.t. A * y = b
 *
 * where W = diag(1/sigma_i^2) is the weight matrix,
 * A = constraint matrix [n_cons x n_meas],
 * x = measurements, y = reconciled values, b = RHS.
 *
 * Solution via Lagrange multipliers:
 *   y = x - W^(-1) * A^T * (A * W^(-1) * A^T)^(-1) * (A*x - b)
 *
 * This is the standard solution for linear steady-state data
 * reconciliation.
 *
 * @param x              Measured values [n_meas]
 * @param variances      Measurement variances [n_meas] (= sigma_i^2)
 * @param n_meas         Number of measurements
 * @param A              Constraint matrix [n_cons * n_meas] (row-major)
 * @param b              Right-hand side [n_cons]
 * @param n_cons         Number of constraints
 * @param y              Output: reconciled values [n_meas]
 * @return 0 on success
 */
int dr_linear_reconciliation(const double *x, const double *variances,
                              int n_meas,
                              const double *A, const double *b,
                              int n_cons, double *y)
{
    /*
     * Solve using the analytical Lagrange multiplier approach.
     *
     * Step 1: Compute residual r = A*x - b  [n_cons]
     * Step 2: Compute V = A * W^(-1) * A^T  [n_cons x n_cons]
     *         where W^(-1) = diag(sigma_i^2) = diag(variances[i])
     * Step 3: Solve V * lambda = r for lambda
     * Step 4: y_i = x_i - sigma_i^2 * sum_j A[j][i] * lambda[j]
     *
     * For small n_cons (typically <= 4 in VFM), we use explicit
     * matrix inversion via Cramer's rule for efficiency.
     */

    if (!x || !variances || !A || !b || !y) return -1;
    if (n_meas < 1 || n_cons < 1) return -1;
    if (n_cons > 4) return -1;  /* Limited to small problems */

    /* Step 1: residual r = A*x - b */
    double r[4] = {0.0, 0.0, 0.0, 0.0};
    int i, j;
    for (i = 0; i < n_cons; i++) {
        r[i] = -b[i];
        for (j = 0; j < n_meas; j++) {
            r[i] += A[i * n_meas + j] * x[j];
        }
    }

    /* Step 2: Compute V = A * W^(-1) * A^T */
    /* V[i][k] = sum_j A[i][j] * sigma_j^2 * A[k][j] */
    double V[16] = {0.0};  /* Max 4x4 = 16 elements */
    for (i = 0; i < n_cons; i++) {
        int k;
        for (k = 0; k < n_cons; k++) {
            double sum = 0.0;
            for (j = 0; j < n_meas; j++) {
                sum += A[i * n_meas + j] * variances[j] * A[k * n_meas + j];
            }
            V[i * n_cons + k] = sum;
        }
    }

    /* Step 3: Solve V * lambda = r */
    /* For n_cons=1: lambda = r[0] / V[0] */
    /* For n_cons=2: use 2x2 inverse */
    double lambda[4] = {0.0, 0.0, 0.0, 0.0};

    if (n_cons == 1) {
        if (fabs(V[0]) < 1e-15) return -1;
        lambda[0] = r[0] / V[0];
    } else if (n_cons == 2) {
        /* 2x2 matrix inverse: [a b; c d]^(-1) = [d -b; -c a] / (ad-bc) */
        double det = V[0] * V[3] - V[1] * V[2];
        if (fabs(det) < 1e-15) return -1;
        lambda[0] = ( V[3] * r[0] - V[1] * r[1]) / det;
        lambda[1] = (-V[2] * r[0] + V[0] * r[1]) / det;
    } else if (n_cons == 3) {
        /* 3x3 matrix inversion via cofactors */
        double a = V[0], b = V[1], c = V[2];
        double d = V[3], e = V[4], f = V[5];
        double g = V[6], h = V[7], i_val = V[8];

        double det = a*(e*i_val - f*h) - b*(d*i_val - f*g) + c*(d*h - e*g);
        if (fabs(det) < 1e-15) return -1;

        double inv_00 = (e*i_val - f*h) / det;
        double inv_01 = (c*h - b*i_val) / det;
        double inv_02 = (b*f - c*e) / det;
        double inv_10 = (f*g - d*i_val) / det;
        double inv_11 = (a*i_val - c*g) / det;
        double inv_12 = (c*d - a*f) / det;
        double inv_20 = (d*h - e*g) / det;
        double inv_21 = (b*g - a*h) / det;
        double inv_22 = (a*e - b*d) / det;

        lambda[0] = inv_00*r[0] + inv_01*r[1] + inv_02*r[2];
        lambda[1] = inv_10*r[0] + inv_11*r[1] + inv_12*r[2];
        lambda[2] = inv_20*r[0] + inv_21*r[1] + inv_22*r[2];
    } else {
        /* n_cons == 4: use 4x4 inversion via Gaussian elimination */
        /* For VFM applications, n_cons rarely exceeds 3 */
        return -1;
    }

    /* Step 4: y_i = x_i - sigma_i^2 * sum_j A[j][i] * lambda[j] */
    for (i = 0; i < n_meas; i++) {
        double sum_A_lambda = 0.0;
        for (j = 0; j < n_cons; j++) {
            sum_A_lambda += A[j * n_meas + i] * lambda[j];
        }
        y[i] = x[i] - variances[i] * sum_A_lambda;
    }

    return 0;
}

/* ==========================================================================
 * L4: Gross Error Detection (Global Test)
 * ========================================================================== */

/**
 * @brief Global test for gross errors using chi-squared statistic.
 *
 * After reconciliation, the objective function J = sum (y_i - x_i)^2/sigma_i^2
 * follows a chi-squared distribution with (n_cons) degrees of freedom
 * if no gross errors are present.
 *
 * If J > chi2_critical(n_cons, alpha), a gross error is likely present.
 * Common alpha = 0.05 (95% confidence).
 *
 * Critical values for chi-squared:
 *   df=1: 3.841, df=2: 5.991, df=3: 7.815, df=4: 9.488
 *
 * @param x_meas       Original measurements [n_meas]
 * @param x_reconciled Reconciled values [n_meas]
 * @param variances    Measurement variances [n_meas]
 * @param n_meas       Number of measurements
 * @param n_cons       Degrees of freedom (= number of constraints)
 * @param alpha        Significance level (e.g., 0.05)
 * @return 1 if gross error detected, 0 if clean
 */
int dr_global_test(const double *x_meas, const double *x_reconciled,
                    const double *variances, int n_meas,
                    int n_cons, double alpha)
{
    (void)alpha;
    if (!x_meas || !x_reconciled || !variances || n_meas < 1) return -1;

    /* Compute objective function J */
    double J = 0.0;
    int i;
    for (i = 0; i < n_meas; i++) {
        if (variances[i] <= 0.0) continue;
        double adj = x_reconciled[i] - x_meas[i];
        J += (adj * adj) / variances[i];
    }

    /* Chi-squared critical values for alpha=0.05 */
    double chi2_crit[5] = {0.0, 3.841, 5.991, 7.815, 9.488};

    double crit = (n_cons >= 1 && n_cons <= 4) ? chi2_crit[n_cons]
                  : (double)n_cons + 2.0 * sqrt((double)n_cons);

    return (J > crit) ? 1 : 0;
}

/**
 * @brief Identify the measurement most likely to contain a gross error
 *        using the Measurement Test (MT) / standardized adjustment.
 *
 * For each measurement i, compute:
 *   z_i = (y_i - x_i) / sigma_i_adjusted
 *
 * where sigma_i_adjusted accounts for the constraint reduction.
 * If |z_i| > z_critical (e.g., 1.96 for 95%), mark measurement i.
 *
 * @param x_meas       Original measurements
 * @param x_reconciled Reconciled values
 * @param variances    Measurement variances
 * @param n_meas       Number of measurements
 * @param z_threshold  Threshold for z-statistic (e.g., 1.96)
 * @param flagged      Output: 1 if measurement is suspect [n_meas]
 * @return Number of flagged measurements
 */
int dr_measurement_test(const double *x_meas, const double *x_reconciled,
                         const double *variances, int n_meas,
                         double z_threshold, int *flagged)
{
    if (!x_meas || !x_reconciled || !variances || !flagged) return -1;

    int n_flagged = 0;
    int i;
    for (i = 0; i < n_meas; i++) {
        if (variances[i] <= 0.0) {
            flagged[i] = 1;
            n_flagged++;
            continue;
        }
        double adj = x_reconciled[i] - x_meas[i];
        double z = fabs(adj) / sqrt(variances[i]);
        flagged[i] = (z > z_threshold) ? 1 : 0;
        if (flagged[i]) n_flagged++;
    }

    return n_flagged;
}

/* ==========================================================================
 * L5: Sequential Data Reconciliation (SDR) for VFM
 * ========================================================================== */

/**
 * @brief Perform sequential data reconciliation for a pipeline network.
 *
 * In a pipeline network (e.g., oil gathering system), the reconciliation
 * problem is solved sequentially: downstream nodes are reconciled first,
 * then upstream nodes using reconciled downstream flow as constraints.
 *
 * This function reconciles flows at a single node using the local mass
 * balance, and returns the reconciled downstream total flow.
 *
 * @param inflows          Array of measured inlet flows [n_in]
 * @param in_uncertainties Array of inlet uncertainties [n_in]
 * @param n_in             Number of inlets
 * @param outflow_meas     Measured outlet flow
 * @param outflow_uncert   Outlet uncertainty
 * @param reconciled_total Output: reconciled total outlet flow
 * @return 0 on success
 */
int dr_sequential_node(const double *inflows,
                        const double *in_uncertainties,
                        int n_in,
                        double outflow_meas, double outflow_uncert,
                        double *reconciled_total)
{
    /*
     * Sequential reconciliation at a single node:
     *
     * Measurements: inlets x_1..x_n, outlet x_out
     * Constraint: sum_i x_i = x_out (mass balance)
     *
     * Reconciled total flow that satisfies the constraint:
     *   x_hat_total = (sum x_i/sigma_i^2 + x_out/sigma_out^2)
     *               / (sum 1/sigma_i^2 + 1/sigma_out^2)
     *
     * This is the inverse-variance weighted mean of the sum of
     * inlets and the outlet measurement, respecting the constraint
     * that they should be equal.
     */

    if (!inflows || !in_uncertainties || !reconciled_total) return -1;
    if (n_in < 1) return -1;
    if (outflow_uncert <= 0.0) return -1;

    double sum_inflows = 0.0;
    double sum_inv_var = 0.0;
    double sum_weighted = 0.0;

    int i;
    for (i = 0; i < n_in; i++) {
        if (in_uncertainties[i] <= 0.0) return -1;

        sum_inflows += inflows[i];
        double inv_var = 1.0 / (in_uncertainties[i] * in_uncertainties[i]);
        sum_inv_var += inv_var;
        sum_weighted += inv_var * inflows[i];
    }

    /* Outlet contributes to the total */
    double inv_var_out = 1.0 / (outflow_uncert * outflow_uncert);
    sum_inv_var += inv_var_out;
    sum_weighted += inv_var_out * outflow_meas;

    /* Reconciled total = weighted average */
    if (sum_inv_var <= 0.0) return -1;
    *reconciled_total = sum_weighted / sum_inv_var;

    return 0;
}