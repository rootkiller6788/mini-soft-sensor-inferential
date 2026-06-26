# Knowledge Graph — Data Reconciliation

## L1: Definitions
- **Measurement**: value, standard deviation (uncertainty), tag ID, presence flag
- **Constraint**: linear equation a^T x = b (mass, energy, component, normalization)
- **Covariance matrix Sigma**: symmetric positive definite, variance of measurements
- **Reconciled value x_hat**: adjusted measurement satisfying constraints
- **Adjustment delta**: x_hat - y, the correction applied to each measurement
- **Lagrange multiplier lambda**: sensitivity of objective to constraint violations
- **Constraint residual**: A x_hat - b, should be ~0 at solution
- **WLS objective**: (x_hat - y)^T Sigma^{-1} (x_hat - y), minimized by reconciliation
- **Redundancy degree**: number of independent ways a variable can be estimated
- **Observability**: ability to determine an unmeasured variable from constraints + measurements
- **Gross error**: non-random measurement error (bias, sensor fault, leak)
- **Global Test statistic**: z = r^T V^{-1} r ~ chi^2(m), tests for any gross error
- **Nodal Test**: z_NT(j) = r_j / sqrt(V_jj) ~ N(0,1), per-constraint test
- **Measurement Test**: z_MT(i) = |d_i| / sqrt(W_ii) ~ N(0,1), per-measurement test
- **Kalman gain K**: optimal blending of prediction and measurement in dynamic DR
- **Innovation**: y_k - H x_{k|k-1}, the measurement residual in Kalman filtering

## L2: Core Concepts
- **Mass/Energy balance constraints**: F_in - F_out = 0 for each node
- **Weighted Least Squares (WLS)**: minimize weighted sum of squared adjustments
- **KKT conditions**: optimality conditions for constrained optimization
- **Constraint-based reconciliation**: enforcing physical laws on measured data
- **Observability analysis**: determining which variables can be estimated
- **Redundancy classification**: measured/non-redundant/unmeasured-observable/unobservable
- **Gross error detection**: identifying faulty sensors via statistical tests
- **Serial elimination**: iterative removal of suspect measurements
- **Robust M-estimation**: using robust loss functions to handle outliers
- **Dynamic reconciliation**: Kalman filter for time-varying processes

## L3: Engineering Structures
- **Incidence matrix**: binary matrix showing which variables appear in which constraints
- **Variance-covariance matrix**: encodes measurement uncertainty structure
- **Lagrange multiplier method**: direct solution of KKT system
- **QR decomposition**: orthogonal transformation for numerical stability
- **Cholesky factorization**: efficient SPD system solver
- **Projection matrix**: P = I - Sigma A^T (A Sigma A^T)^{-1} A
- **Joseph form**: numerically stable Kalman covariance update
- **Power iteration**: dominant eigenvector estimation for PCA
- **Gauss-Jordan elimination**: matrix inversion with partial pivoting
- **Householder reflections**: orthogonal transformations for QR

## L4: Engineering Laws / Standards
- **GUM (JCGM 100:2008)**: Guide to the expression of uncertainty in measurement
- **VDI 2048**: German standard for data reconciliation and uncertainty control
- **ISO 5167**: Orifice flow measurement standard
- **ISO 5725-2**: Accuracy of measurement methods
- **Conservation of mass**: total mass in = total mass out (steady state)
- **Conservation of energy**: energy in = energy out + accumulation
- **Cochran's theorem**: quadratic forms of multivariate normals follow chi-squared
- **Kuehn-Davidson theorem**: unique WLS solution under full rank conditions
- **Gauss-Markov theorem**: WLS is BLUE (Best Linear Unbiased Estimator)
- **Kalman optimality**: Kalman filter is MMSE for linear Gaussian systems

## L5: Algorithms / Methods
- **WLS with Lagrange multipliers**: dense direct solver for DR
- **WLS with QR decomposition**: numerically stable orthogonal method
- **WLS with Cholesky**: fastest for well-conditioned SPD systems
- **Global Test (GT)**: chi-squared test for gross error presence
- **Nodal Test (NT)**: per-constraint gross error detection
- **Measurement Test (MT)**: per-measurement gross error identification
- **Serial elimination**: iterative gross error removal
- **Simultaneous estimation**: joint reconciliation and gross error estimation
- **Huber M-estimator**: robust DR with bounded influence function
- **Biweight (Tukey) M-estimator**: robust DR with complete outlier rejection
- **Cauchy M-estimator**: robust DR with smooth heavy-tailed weighting
- **Fair M-estimator**: robust DR with linear asymptotic behavior
- **IRLS**: Iteratively Reweighted Least Squares for robust estimation
- **Kalman filter**: optimal linear dynamic state estimation
- **RTS smoother**: fixed-interval optimal smoothing
- **Moving Horizon Estimation**: constrained dynamic DR over finite window
- **Steady-state Kalman gain**: DARE solution for time-invariant systems
- **Power iteration**: dominant eigenvector for PCA-based detection
- **Gauss-Jordan elimination**: matrix inversion algorithm
- **Hyndman-Fan Method 7**: quantile estimation for outlier detection

## L6: Canonical Problems
1. **Flow network reconciliation**: 5-stream, 2-node pipeline mass balance (Example 1)
2. **Steam metering gross error detection**: boiler-turbine-bypass system (Example 2)
3. **Dynamic tank level estimation**: Kalman filter for time-varying level (Example 3)

## L7: Industrial Applications
- **Steam metering in power plants**: VDI 2048 application (in gross_error.h/c)
- **Flow network reconciliation**: oil & gas production allocation (in examples)
- **Tank farm inventory reconciliation**: mass balance with tank level measurements
- **Distillation column mass balance**: component-wise reconciliation

## L8: Advanced Topics
- **Robust M-estimators**: Huber, Biweight, Cauchy, Fair functions
- **PCA-based gross error detection**: Tong & Crowe (1995) method
- **Test power analysis**: non-central chi-squared distribution
- **Bayesian data reconciliation**: prior information on parameter distributions

## L9: Research Frontiers
- **Digital Twin integration**: continuous DR feeding virtual plant models
- **Real-time large-scale DR**: 10^5+ variable systems
- **Nonlinear DR**: first-principles model constraints
- **ML-augmented DR**: neural network surrogate models for complex processes
