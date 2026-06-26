/**
 * @file vfm_multi_phase.c
 * @brief Multi-phase flow models for Virtual Flow Meter
 *
 * Knowledge Coverage:
 *   L1 Definitions: Gas void fraction, liquid holdup, slip velocity
 *   L2 Core Concepts: Two-phase flow regimes, phase slip
 *   L3 Engineering Structures: Empirical correlation pipeline
 *   L4 Engineering Laws: Beggs & Brill, Lockhart-Martinelli, Chisholm
 *   L5 Algorithms: Flow pattern prediction, pressure drop calculation
 *   L6 Canonical Problems: Oil-gas-water well flow estimation
 *
 * Multi-phase flow is the rule rather than the exception in oil & gas
 * production. VFM for multi-phase wells must handle gas-liquid slip,
 * flow regime transitions, and complex mixture properties.
 *
 * References:
 *   Beggs, H.D. & Brill, J.P. (1973) "A Study of Two-Phase Flow in
 *     Inclined Pipes", Journal of Petroleum Technology
 *   Lockhart, R.W. & Martinelli, R.C. (1949) "Proposed Correlation of
 *     Data for Isothermal Two-Phase, Two-Component Flow in Pipes"
 *   Chisholm, D. (1967) "A Theoretical Basis for the Lockhart-Martinelli
 *     Correlation for Two-Phase Flow"
 *
 * @module mini-virtual-flow-meter
 */

#include <math.h>
#include <stdlib.h>

/* ==========================================================================
 * L1: Core Data Structures and Constants
 * ========================================================================== */

#define MULTI_PHASE_G 9.80665
#define MULTI_PHASE_PI 3.14159265358979323846

/** Multi-phase mixture properties (matches header definition) */
typedef struct {
    double gas_fraction;
    double oil_fraction;
    double water_fraction;
    double liquid_holdup;
    double mixture_density;
    double mixture_viscosity;
    double slip_velocity;
    int    flow_pattern;
} multiphase_mix_t;

/** Lockhart-Martinelli parameter structure */
typedef struct {
    double X;             /**< Lockhart-Martinelli parameter                */
    double phi_l_sq;      /**< Two-phase multiplier for liquid             */
    double phi_g_sq;      /**< Two-phase multiplier for gas               */
    double dpf_liq;       /**< Frictional pressure drop (liquid-only) [Pa/m] */
    double dpf_gas;       /**< Frictional pressure drop (gas-only) [Pa/m]  */
    double dpf_tp;        /**< Two-phase frictional pressure drop [Pa/m]    */
} lockhart_martinelli_t;

/* ==========================================================================
 * L2: Flow Pattern Prediction (Beggs & Brill)
 * ========================================================================== */

/**
 * Beggs & Brill flow pattern transitions are based on:
 *  1. No-slip liquid holdup: lambda_L = Q_L / (Q_L + Q_G)
 *  2. Froude number: Fr = v_m^2 / (g * D)
 *  3. Transition boundaries L1, L2, L3, L4
 *
 * Flow patterns:
 *   1 = Segregated (stratified, wavy, annular)
 *   2 = Intermittent (slug, plug)
 *   3 = Distributed (bubble, mist)
 *   4 = Transition
 */

/**
 * @brief Compute no-slip liquid holdup.
 *
 * lambda_L = Q_liq / (Q_liq + Q_gas)
 *
 * This is the inlet liquid volume fraction assuming no slip
 * between phases (both travel at same average velocity).
 *
 * @param Q_liq  Liquid volumetric flow rate [m^3/s]
 * @param Q_gas  Gas volumetric flow rate [m^3/s]
 * @return No-slip liquid holdup [0..1]
 */
double multiphase_noslip_holdup(double Q_liq, double Q_gas)
{
    double Q_total = Q_liq + Q_gas;
    if (Q_total <= 0.0) return 1.0;  /* No flow => assume liquid-filled */
    return Q_liq / Q_total;
}

/**
 * @brief Predict two-phase flow pattern using Beggs & Brill (1973).
 *
 * The original Beggs & Brill correlation was developed from air-water
 * experiments in 1-inch and 1.5-inch pipes at various inclination angles.
 * It is the most widely used flow pattern map in the petroleum industry.
 *
 * @param lambda_L    No-slip liquid holdup [0..1]
 * @param Froude_mix  Mixture Froude number v_m^2/(g*D)
 * @return Flow pattern code: 1=Segregated, 2=Intermittent,
 *         3=Distributed, 0=Transition
 */
int multiphase_pattern_beggs_brill(double lambda_L, double Froude_mix)
{
    if (lambda_L < 0.0) lambda_L = 0.0;
    if (lambda_L > 1.0) lambda_L = 1.0;

    /* Transition boundaries L1-L4 from Beggs & Brill Table 1 */
    double L1 = 316.0 * pow(lambda_L, 0.302);
    double L2 = 0.0009252 * pow(lambda_L, -2.4684);
    double L3 = 0.10 * pow(lambda_L, -1.4516);
    double L4 = 0.5 * pow(lambda_L, -6.738);

    if (lambda_L < 0.01) {
        return 3;  /* Distributed (mist flow at very low liquid fraction) */
    }

    /* Segregated flow condition */
    if ((lambda_L < 0.01 && Froude_mix < L1) ||
        (lambda_L >= 0.01 && Froude_mix < L2)) {
        return 1;  /* Segregated */
    }

    /* Transition zone */
    if (lambda_L >= 0.01 && L2 <= Froude_mix && Froude_mix <= L3) {
        return 0;  /* Transition */
    }

    /* Intermittent flow condition */
    if ((lambda_L >= 0.01 && lambda_L < 0.4 && L3 < Froude_mix && Froude_mix <= L1) ||
        (lambda_L >= 0.4 && L3 < Froude_mix && Froude_mix <= L4)) {
        return 2;  /* Intermittent */
    }

    /* Distributed flow condition */
    if ((lambda_L < 0.4 && Froude_mix >= L1) ||
        (lambda_L >= 0.4 && Froude_mix > L4)) {
        return 3;  /* Distributed */
    }

    return 2;  /* Default: intermittent */
}

/* ==========================================================================
 * L3: Liquid Holdup (Beggs & Brill)
 * ========================================================================== */

/**
 * @brief Compute actual liquid holdup using Beggs & Brill correlation.
 *
 * The actual (in-situ) liquid holdup H_L differs from the no-slip
 * holdup lambda_L because gas travels faster than liquid (slip).
 *
 * Step 1: Compute horizontal holdup H_L(0) from:
 *   H_L(0) = a * lambda_L^b / Fr^c
 *   where a, b, c depend on flow pattern.
 *
 * Step 2: Correct for inclination angle theta:
 *   H_L(theta) = H_L(0) * psi
 *   where psi = 1 + C * (sin(1.8*theta) - (1/3)*sin^3(1.8*theta))
 *   C = (1 - lambda_L) * ln(d * lambda_L^e * N_LV^f * Fr^g)
 *
 * Here we implement the simplified horizontal holdup only.
 *
 * @param lambda_L      No-slip liquid holdup [0..1]
 * @param Froude_mix    Mixture Froude number
 * @param flow_pattern  Flow pattern (from multiphase_pattern_beggs_brill)
 * @return Actual liquid holdup [0..1]
 */
double multiphase_holdup_beggs_brill(double lambda_L, double Froude_mix,
                                      int flow_pattern)
{
    if (lambda_L < 0.001) return 0.0;
    if (lambda_L > 0.999) return 1.0;
    if (Froude_mix < 0.001) Froude_mix = 0.001;

    double a, b, c;

    /* Constants from Beggs & Brill Table 2 */
    switch (flow_pattern) {
    case 1: /* Segregated */
        a = 0.98;  b = 0.4846;  c = 0.0868;
        break;
    case 2: /* Intermittent */
        a = 0.845; b = 0.5351;  c = 0.0173;
        break;
    case 3: /* Distributed */
        a = 1.065; b = 0.5824;  c = 0.0609;
        break;
    default: /* Transition */
        /* Interpolate between segregated and intermittent */
        a = 0.91;  b = 0.51;    c = 0.05;
        break;
    }

    /* Horizontal holdup */
    double HL_0 = a * pow(lambda_L, b) / pow(Froude_mix, c);

    /* Constrain holdup: H_L >= lambda_L (slip makes liquid hold up more) */
    if (HL_0 < lambda_L) HL_0 = lambda_L;
    if (HL_0 > 1.0) HL_0 = 1.0;

    return HL_0;
}

/* ==========================================================================
 * L4: Lockhart-Martinelli Two-Phase Pressure Drop
 * ========================================================================== */

/**
 * @brief Compute Lockhart-Martinelli parameter X^2.
 *
 * X^2 = (dP/dz)_f,liq / (dP/dz)_f,gas
 *
 * where (dP/dz)_f,liq and (dP/dz)_f,gas are the frictional pressure
 * gradients if each phase flowed alone in the pipe.
 *
 * Using the Blasius friction factor f = 0.079 * Re^(-0.25) for
 * turbulent flow, the ratio simplifies to:
 *
 * X^2 = (rho_gas/rho_liq) * (mu_liq/mu_gas)^0.25
 *     * ((1-x)/x)^1.75
 *
 * where x = mass quality (gas mass fraction).
 *
 * @param quality     Gas mass quality [0..1]
 * @param rho_liq     Liquid density [kg/m^3]
 * @param rho_gas     Gas density [kg/m^3]
 * @param mu_liq      Liquid dynamic viscosity [Pa*s]
 * @param mu_gas      Gas dynamic viscosity [Pa*s]
 * @return Lockhart-Martinelli parameter X (dimensionless)
 */
double lockhart_martinelli_X(double quality,
                              double rho_liq, double rho_gas,
                              double mu_liq, double mu_gas)
{
    if (quality <= 0.001) return 100.0;  /* Almost all liquid */
    if (quality >= 0.999) return 0.001;  /* Almost all gas */
    if (rho_liq <= 0.0 || rho_gas <= 0.0) return 1.0;
    if (mu_liq <= 0.0 || mu_gas <= 0.0) return 1.0;

    /* Density ratio */
    double ratio_rho = rho_gas / rho_liq;

    /* Viscosity ratio (mu_L/mu_G)^0.25 */
    double ratio_mu = pow(mu_liq / mu_gas, 0.25);

    /* Quality ratio: ((1-x)/x)^0.9 for Chisholm correlation
     * or ((1-x)/x)^0.875 = ((1-x)/x)^1.75 * 0.5 actually
     * Standard LM: X = ((1-x)/x)^0.9 * sqrt(rho_gas/rho_liq) * ... */
    double quality_ratio = pow((1.0 - quality) / quality, 0.9);

    double X = quality_ratio * sqrt(ratio_rho) * ratio_mu;

    if (X < 0.001) X = 0.001;
    if (X > 1000.0) X = 1000.0;

    return X;
}

/**
 * @brief Compute two-phase multiplier for liquid using Chisholm correlation.
 *
 * phi_L^2 = 1 + C/X + 1/X^2
 *
 * where C is the Chisholm parameter:
 *   C = 20 for turbulent-turbulent (tt) flow
 *   C = 12 for viscous-turbulent (vt) flow
 *   C = 10 for turbulent-viscous (tv) flow
 *   C =  5 for viscous-viscous (vv) flow
 *
 * @param X  Lockhart-Martinelli parameter
 * @param C  Chisholm constant (typically 20 for tt)
 * @return Two-phase friction multiplier for liquid phi_L^2
 */
double chisholm_phi_liquid_sq(double X, double C)
{
    if (X < 1e-10) return 1e10;
    double X_inv = 1.0 / X;
    return 1.0 + C * X_inv + X_inv * X_inv;
}

/**
 * @brief Compute two-phase frictional pressure gradient.
 *
 * (dP/dz)_tp = phi_L^2 * (dP/dz)_liq
 *
 * where phi_L^2 is the two-phase multiplier and (dP/dz)_liq
 * is the single-phase liquid pressure gradient.
 *
 * @param phi_l_sq     Two-phase multiplier phi_L^2
 * @param dpdz_liq     Single-phase liquid pressure gradient [Pa/m]
 * @return Two-phase frictional pressure gradient [Pa/m]
 */
double tp_frictional_pressure_gradient(double phi_l_sq, double dpdz_liq)
{
    return phi_l_sq * dpdz_liq;
}

/**
 * @brief Compute single-phase frictional pressure gradient
 *        using Darcy-Weisbach with Blasius friction factor.
 *
 * (dP/dz)_f = f * (1/D) * (rho * v^2 / 2)
 *
 * where f = 0.079 * Re^(-0.25) for turbulent flow (Blasius).
 *
 * @param diameter    Pipe diameter [m]
 * @param velocity    Superficial velocity of phase [m/s]
 * @param rho         Density [kg/m^3]
 * @param mu          Dynamic viscosity [Pa*s]
 * @return Frictional pressure gradient [Pa/m]
 */
double sp_frictional_pressure_gradient(double diameter, double velocity,
                                        double rho, double mu)
{
    if (diameter <= 0.0) return 0.0;

    /* Reynolds number */
    double Re = rho * fabs(velocity) * diameter / mu;
    if (Re <= 0.0) return 0.0;

    /* Blasius friction factor (turbulent) */
    double f = 0.079 * pow(Re, -0.25);

    /* For laminar flow, use 64/Re instead */
    if (Re < 2300.0) {
        f = 64.0 / Re;
    }

    return f * (1.0 / diameter) * (rho * velocity * velocity / 2.0);
}

/* ==========================================================================
 * L5: Multi-Phase Flow Rate Estimation (VFM)
 * ========================================================================== */

/**
 * @brief Estimate total mixture flow rate from measured pressure drop
 *        in two-phase flow using the Lockhart-Martinelli method.
 *
 * This is the core VFM problem for multi-phase wells: given a measured
 * pressure drop and phase fractions, estimate the total flow rate.
 *
 * Algorithm:
 *   1. Guess total mass flux G
 *   2. Compute phase velocities and Reynolds numbers
 *   3. Compute single-phase pressure gradients
 *   4. Compute Lockhart-Martinelli parameter X
 *   5. Compute two-phase multiplier phi_L^2
 *   6. Compute predicted pressure gradient
 *   7. Adjust G using Newton step: G_new = G * sqrt(dP_meas / dP_pred)
 *   8. Repeat until convergence
 *
 * @param diameter        Pipe diameter [m]
 * @param dp_meas         Measured pressure drop [Pa]
 * @param length          Pipe length [m]
 * @param quality         Gas mass quality [0..1]
 * @param rho_liq         Liquid density [kg/m^3]
 * @param rho_gas         Gas density [kg/m^3]
 * @param mu_liq          Liquid viscosity [Pa*s]
 * @param mu_gas          Gas viscosity [Pa*s]
 * @param max_iter        Maximum Newton iterations
 * @param tol             Convergence tolerance
 * @return Total mass flux G [kg/(m^2*s)], -1 on failure
 */
double multiphase_flow_from_dp(double diameter, double dp_meas,
                                double length, double quality,
                                double rho_liq, double rho_gas,
                                double mu_liq, double mu_gas,
                                int max_iter, double tol)
{
    if (diameter <= 0.0 || length <= 0.0 || dp_meas < 0.0) return -1.0;
    if (quality < 0.0) quality = 0.0;
    if (quality > 1.0) quality = 1.0;

    double area = MULTI_PHASE_PI * diameter * diameter / 4.0;
    (void)area;

    /* Mixture density (homogeneous) */
    double rho_mix = quality * rho_gas + (1.0 - quality) * rho_liq;
    if (rho_mix <= 0.0) return -1.0;

    /* Initial guess: assume single-phase liquid flow */
    /* Estimate G from dP = f*L/D * (G^2/(2*rho)) */
    double f_guess = 0.02;  /* Assume turbulent */
    double G = sqrt(2.0 * rho_liq * dp_meas * diameter
                    / (f_guess * length));

    if (G < 1e-6) return 0.0;

    double dp_meas_per_m = dp_meas / length;

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        /* Superficial velocities */
        double vel_liq = G * (1.0 - quality) / rho_liq;
        double vel_gas = G * quality / rho_gas;

        /* Single-phase pressure gradients */
        double dpdz_liq = sp_frictional_pressure_gradient(
            diameter, vel_liq, rho_liq, mu_liq);
        double dpdz_gas = sp_frictional_pressure_gradient(
            diameter, vel_gas, rho_gas, mu_gas);

        if (dpdz_liq < 1e-15 && dpdz_gas < 1e-15) break;

        /* Lockhart-Martinelli parameter */
        double X = lockhart_martinelli_X(quality, rho_liq, rho_gas,
                                          mu_liq, mu_gas);

        /* Chisholm two-phase multiplier (turbulent-turbulent, C=20) */
        double phi_l_sq = chisholm_phi_liquid_sq(X, 20.0);

        /* Predicted two-phase pressure gradient */
        double dpdz_pred = tp_frictional_pressure_gradient(phi_l_sq, dpdz_liq);

        if (dpdz_pred < 1e-15) break;

        /* Newton-like correction: G ~ sqrt(dP/dz) */
        double G_new = G * sqrt(dp_meas_per_m / dpdz_pred);

        if (fabs(G_new - G) / (G + 1e-6) < tol) {
            return G_new;  /* Converged */
        }

        G = G_new;
    }

    return G;  /* Return best estimate */
}

/**
 * @brief Compute mixture volumetric flow rate from mass flux.
 *
 * Q_mix = G * A / rho_mix
 *
 * @param G        Mass flux [kg/(m^2*s)]
 * @param diameter Pipe diameter [m]
 * @param rho_mix  Mixture density [kg/m^3]
 * @return Mixture volumetric flow rate [m^3/s]
 */
double multiphase_mass_flux_to_volumetric(double G, double diameter,
                                           double rho_mix)
{
    if (diameter <= 0.0 || rho_mix <= 0.0) return 0.0;
    double area = MULTI_PHASE_PI * diameter * diameter / 4.0;
    (void)area;
    return G * area / rho_mix;
}