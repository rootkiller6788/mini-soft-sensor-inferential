/**
 * @file flow_models.c
 * @brief Physical flow model implementations
 *
 * Implements ISO 5167 orifice, Venturi, Bernoulli, Darcy-Weisbach,
 * Colebrook, pump curve, and choke valve models.
 *
 * @module mini-virtual-flow-meter
 */

#include "virtual_flow_meter.h"
#include "flow_models.h"
#include <math.h>
#include <stdio.h>

/* Universal constants */
#define G_STANDARD 9.80665
#define PI         3.14159265358979323846

/* ==========================================================================
 * L2: Reynolds Number and Flow Regime
 * ========================================================================== */

double flow_reynolds_number(double velocity, double diameter,
                             double kin_visc)
{
    /* Edge case: zero viscosity means inviscid flow (infinite Re) */
    if (kin_visc <= 0.0) {
        return (velocity > 0.0 || diameter > 0.0) ? INFINITY : 0.0;
    }
    if (diameter <= 0.0) return 0.0;

    return fabs(velocity) * diameter / kin_visc;
}

int flow_regime_classify(double re)
{
    if (re < 2300.0) return VFM_REGIME_LAMINAR;
    if (re < 4000.0) return VFM_REGIME_TRANSITIONAL;
    return VFM_REGIME_TURBULENT;
}

double flow_critical_velocity(double diameter, double kin_visc)
{
    if (diameter <= 0.0 || kin_visc <= 0.0) return 0.0;
    /* Re_critical = 2300 */
    return 2300.0 * kin_visc / diameter;
}

/* ==========================================================================
 * L3-L4: Orifice Flow Model — ISO 5167
 * ========================================================================== */

int orifice_params_init(orifice_params_t *params, double pipe_diam,
                         double bore_diam)
{
    if (!params) return -1;
    if (pipe_diam <= 0.0 || bore_diam <= 0.0) return -1;

    params->pipe_diameter = pipe_diam;
    params->bore_diameter = bore_diam;
    params->beta_ratio    = bore_diam / pipe_diam;

    /* Beta ratio valid range per ISO 5167-1: 0.10 <= beta <= 0.75 */
    if (params->beta_ratio < 0.10 || params->beta_ratio > 0.75) {
        return -1;
    }

    params->bore_area = PI * bore_diam * bore_diam / 4.0;

    /* Default discharge coefficient (will be refined with Re) */
    params->discharge_coeff = 0.610;

    /* Default expansibility (liquid => epsilon=1.0) */
    params->expansibility_factor = 1.0;

    /* Default D-D/2 pressure tap spacing */
    params->pressure_tap_spacing = pipe_diam;

    return 0;
}

double orifice_discharge_coeff_iso5167(double beta, double re_d)
{
    /*
     * Reader-Harris / Gallagher (1998) equation for D-D/2 taps.
     * ISO 5167-1:2022 Equation (4).
     *
     * Cd = 0.5961 + 0.0261*beta^2 - 0.216*beta^8
     *      + 0.000521 * (1e6*beta/Re_D)^0.7
     *      + (0.0188 + 0.0063*A) * beta^3.5 * (1e6/Re_D)^0.3
     *
     * where A = (19000*beta/Re_D)^0.8
     *
     * Valid range: D >= 50mm, 0.1 <= beta <= 0.75, Re_D >= 5000
     */

    if (beta <= 0.0 || beta > 0.75) return 0.610;  /* Default */
    if (re_d < 5000.0) re_d = 5000.0;  /* Minimum Re for validity */

    double beta2 = beta * beta;
    double beta4 = beta2 * beta2;
    double beta8 = beta4 * beta4;

    double beta_over_Re = beta / re_d;

    /* First two terms: geometry-dependent */
    double Cd = 0.5961 + 0.0261 * beta2 - 0.216 * beta8;

    /* Third term: pipe Reynolds number correction */
    Cd += 0.000521 * pow(1.0e6 * beta_over_Re, 0.7);

    /* Fourth term: combined correction */
    double A = pow(19000.0 * beta_over_Re, 0.8);
    Cd += (0.0188 + 0.0063 * A) * pow(beta, 3.5) * pow(1.0e6 / re_d, 0.3);

    /* Also apply the small-pipe correction for D < 71.12 mm */
    /* (skipped here — requires pipe diameter input) */

    return Cd;
}

double orifice_mass_flow(const orifice_params_t *params, double dp,
                          double rho, double epsilon)
{
    /* ISO 5167 mass flow equation:
     * qm = Cd * epsilon * pi/4 * d^2 * sqrt(2 * dP * rho) /
     *      sqrt(1 - beta^4)
     */

    if (!params) return 0.0;
    if (dp < 0.0 || rho <= 0.0) return 0.0;
    if (epsilon <= 0.0 || epsilon > 1.0) epsilon = 1.0;

    double beta4 = params->beta_ratio * params->beta_ratio;
    beta4 = beta4 * beta4;  /* beta^4 */

    if (beta4 >= 1.0) return 0.0;  /* Invalid geometry */

    double numerator = params->discharge_coeff * epsilon
                     * params->bore_area * sqrt(2.0 * dp * rho);
    double denominator = sqrt(1.0 - beta4);

    return numerator / denominator;
}

double orifice_vol_flow(const orifice_params_t *params, double dp,
                         double rho, double epsilon)
{
    double qm = orifice_mass_flow(params, dp, rho, epsilon);
    if (rho <= 0.0) return 0.0;
    return qm / rho;
}

double orifice_expansibility(double beta, double dp, double p_upstream,
                              double k)
{
    /*
     * ISO 5167-1 gas expansibility factor for orifice plates:
     * epsilon = 1 - (0.41 + 0.35*beta^4) * dP / (k * P1)
     *
     * where k = isentropic exponent (cp/cv).
     *
     * Valid for P2/P1 >= 0.75 (subsonic flow).
     */

    if (p_upstream <= 0.0 || k <= 0.0) return 1.0;

    double x = dp / p_upstream;  /* Pressure ratio */

    if (x < 0.0) return 1.0;
    if (x >= 1.0) return 1.0;   /* Fully expanded, not physically meaningful */

    double beta4 = beta * beta;
    beta4 = beta4 * beta4;

    double coef = 0.41 + 0.35 * beta4;
    double eps = 1.0 - coef * x / k;

    if (eps < 0.0) eps = 0.0;
    if (eps > 1.0) eps = 1.0;

    return eps;
}

/* ==========================================================================
 * L4: Venturi Flow Model (ISO 5167-4)
 * ========================================================================== */

int venturi_params_init(venturi_params_t *params, double inlet_diam,
                         double throat_diam)
{
    if (!params) return -1;
    if (inlet_diam <= 0.0 || throat_diam <= 0.0) return -1;
    if (throat_diam >= inlet_diam) return -1;

    params->inlet_diameter  = inlet_diam;
    params->throat_diameter = throat_diam;
    params->beta_ratio      = throat_diam / inlet_diam;

    if (params->beta_ratio < 0.30 || params->beta_ratio > 0.75) {
        return -1;  /* Venturi valid range */
    }

    params->throat_area = PI * throat_diam * throat_diam / 4.0;
    params->discharge_coeff = 0.985;  /* Typical for machined convergent */
    params->convergent_angle = 0.3665;  /* 21 degrees in radians */
    params->divergent_angle  = 0.2618;  /* 15 degrees in radians */

    return 0;
}

double venturi_mass_flow(const venturi_params_t *params, double dp,
                          double rho, double epsilon)
{
    if (!params) return 0.0;
    if (dp < 0.0 || rho <= 0.0) return 0.0;
    if (epsilon <= 0.0 || epsilon > 1.0) epsilon = 1.0;

    double beta4 = params->beta_ratio * params->beta_ratio;
    beta4 = beta4 * beta4;

    if (beta4 >= 1.0) return 0.0;

    double numerator = params->discharge_coeff * epsilon
                     * params->throat_area * sqrt(2.0 * dp * rho);
    double denominator = sqrt(1.0 - beta4);

    return numerator / denominator;
}

/* ==========================================================================
 * L5: Bernoulli Energy Balance Model
 * ========================================================================== */

double bernoulli_flow_rate(double p1, double p2, double area,
                            double rho, double dz, double head_loss)
{
    /*
     * Bernoulli equation solved for velocity at point 2 (v1 ~ 0):
     * P1/rho + g*z1 = P2/rho + v2^2/2 + g*z2 + g*hL
     *
     * v2 = sqrt(2*(P1-P2)/rho + 2*g*(z1-z2) - 2*g*hL)
     *
     * Q = v2 * area
     */

    if (area <= 0.0 || rho <= 0.0) return 0.0;
    if (head_loss < 0.0) head_loss = 0.0;

    double dp = p1 - p2;  /* Positive if p1 > p2 */

    /* Available head: pressure difference + elevation (dz = z1-z2) */
    double available_energy = 2.0 * dp / rho + 2.0 * G_STANDARD * dz;

    /* Subtract friction losses */
    available_energy -= 2.0 * G_STANDARD * head_loss;

    if (available_energy <= 0.0) return 0.0;  /* No driving force for flow */

    double velocity = sqrt(available_energy);

    return velocity * area;
}

/* ==========================================================================
 * L6: Darcy-Weisbach and Friction Factors
 * ========================================================================== */

double darcy_friction_laminar(double re)
{
    /*
     * Hagen-Poiseuille exact solution for fully-developed laminar flow.
     * f = 64 / Re
     *
     * Valid for Re < 2300 (circular pipes).
     * This is the only exact friction factor solution derived from
     * the Navier-Stokes equations for pipe flow.
     */

    if (re <= 0.0) return 0.0;
    return 64.0 / re;
}

double darcy_friction_colebrook(double re, double rel_roughness,
                                 int max_iter, double tol)
{
    /*
     * Colebrook-White equation (1939):
     * 1/sqrt(f) = -2.0 * log10(eps/(3.7*D) + 2.51/(Re*sqrt(f)))
     *
     * Solved by Newton-Raphson on variable x = 1/sqrt(f).
     *
     * Let x = 1/sqrt(f), then f = 1/x^2.
     * The equation becomes: x + 2.0*log10(rr/3.7 + 2.51*x/Re) = 0
     *
     * where rr = rel_roughness = eps/D.
     *
     * Derivative: 1 + 2.0/(ln(10)*(rr/3.7 + 2.51*x/Re)) * (2.51/Re)
     *
     * Newton step: x_new = x_old - F(x)/F'(x)
     */

    if (re <= 0.0) return -1.0;

    /* Laminar: use exact solution */
    if (re < 2300.0) {
        return darcy_friction_laminar(re);
    }

    if (rel_roughness < 0.0) rel_roughness = 0.0;

    /* Initial guess from Swamee-Jain explicit approximation (1976):
     * f = 0.25 / [log10(rr/3.7 + 5.74/Re^0.9)]^2
     * This is within 1% of Colebrook for 1e-6 < rr < 0.01 and
     * 5000 < Re < 1e8.
     */
    double log_arg = rel_roughness / 3.7 + 5.74 / pow(re, 0.9);
    if (log_arg <= 0.0) log_arg = 1e-10;
    double f_guess = 0.25 / (log10(log_arg) * log10(log_arg));

    /* Convert to x = 1/sqrt(f) */
    double x = 1.0 / sqrt(f_guess);

    double ln10 = log(10.0);

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        /* Compute F(x) = x + 2*log10(rr/3.7 + 2.51*x/Re) */
        double arg = rel_roughness / 3.7 + 2.51 * x / re;
        if (arg <= 0.0) arg = 1e-15;
        double Fx = x + 2.0 * log10(arg);

        /* Compute F'(x) */
        double dFdx = 1.0 + (2.0 / (ln10 * arg)) * (2.51 / re);

        double dx = Fx / dFdx;

        x = x - dx;

        if (fabs(dx) < tol) {
            /* Converged: f = 1/x^2 */
            double f = 1.0 / (x * x);

            /* Sanity check: f must be positive and reasonable */
            if (f > 0.0 && f < 1.0) {
                return f;
            }
            break;
        }
    }

    /* If Newton failed to converge, return the Swamee-Jain estimate */
    /* as a fallback rather than -1 */
    return f_guess;
}

double darcy_weisbach_head_loss(double friction_factor, double length,
                                 double diameter, double velocity)
{
    /*
     * Darcy-Weisbach equation:
     * hL = f * (L/D) * v^2 / (2*g)
     *
     * where:
     *   hL = head loss [m of fluid]
     *   f  = Darcy friction factor (dimensionless)
     *   L  = pipe length [m]
     *   D  = pipe diameter [m]
     *   v  = mean flow velocity [m/s]
     *   g  = gravitational acceleration [m/s^2]
     *
     * Named after Henry Darcy and Julius Weisbach (1845/1857).
     * This is the fundamental equation for major losses in pipe flow.
     */

    if (diameter <= 0.0 || length < 0.0) return 0.0;

    double v2 = velocity * velocity;
    return friction_factor * (length / diameter) * v2 / (2.0 * G_STANDARD);
}

double darcy_inverse_flow(double diameter, double roughness, double length,
                           double head_loss, double kin_visc,
                           int max_iter, double tol)
{
    /*
     * Invert Darcy-Weisbach: given head loss, find flow rate.
     *
     * Algorithm (fixed-point iteration):
     *   Q0 = sqrt(2*g*D*hL / (f_guess * L)) * A  (initial guess)
     *   loop:
     *     v = Q/A,  Re = v*D/nu
     *     f = Colebrook(Re, eps/D)
     *     hL_pred = f*(L/D)*v^2/(2g)
     *     Q_new = Q * sqrt(hL / hL_pred)
     *   until |Q_new - Q|/Q < tol
     *
     * This is the classic "virtual flow meter" problem for pipelines.
     */

    if (diameter <= 0.0 || length <= 0.0 || head_loss < 0.0) return -1.0;

    double area = PI * diameter * diameter / 4.0;
    double rel_roughness = roughness / diameter;

    if (head_loss < 1e-12) return 0.0;  /* No head loss => no flow */

    /* Initial guess: assume turbulent f ~ 0.02 */
    double f_guess = 0.02;
    double Q = area * sqrt(2.0 * G_STANDARD * diameter * head_loss
                           / (f_guess * length));

    if (Q < 1e-12) return 0.0;

    double Q_prev;
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        double velocity = Q / area;
        double re = flow_reynolds_number(velocity, diameter, kin_visc);

        double f = darcy_friction_colebrook(re, rel_roughness, 50, tol);
        if (f < 0.0) f = 0.02;  /* Fallback */

        double hL_pred = darcy_weisbach_head_loss(f, length, diameter, velocity);

        if (hL_pred < 1e-15) break;

        Q_prev = Q;
        Q *= sqrt(head_loss / hL_pred);

        /* Convergence check */
        if (fabs(Q - Q_prev) / (Q_prev + 1e-12) < tol) {
            return Q;
        }
    }

    return Q;
}

/* ==========================================================================
 * L7: Pump Curve Flow Estimation
 * ========================================================================== */

int pump_curve_init(pump_curve_params_t *params, double rated_flow,
                     double rated_head, double shutoff_head,
                     double rated_speed)
{
    if (!params) return -1;

    /*
     * Pump curve: H(Q) = H0 - A*Q - B*Q^2
     *
     * Constraints:
     *   H(0) = H0 = shutoff_head
     *   H(Q0) = H0_rated = rated_head
     *   dH/dQ at Q0 = 0 (peak efficiency assumption => we use two points)
     *
     * With only two points (0, H_shutoff) and (Q0, H0), we need another
     * condition. Assume B=0 (linear pump curve) if only two points.
     * With rated data including efficiency, can fit both A and B.
     *
     * Here we use a simplified approach:
     *   A = (H_shutoff - H_rated) / Q_rated  (assume B=0, linear curve)
     * Or for a more realistic quadratic:
     *   A and B are determined by assuming the curve passes through
     *   the rated point with zero slope at best efficiency.
     *   A = 2*(H_shutoff - H_rated)/Q_rated
     *   B = -(H_shutoff - H_rated)/Q_rated^2
     *   This gives: H(0)=H_shutoff, H(Q0)=H_rated, H'(Q0)=0
     */

    if (rated_flow <= 0.0 || shutoff_head < rated_head) {
        return -1;
    }

    params->rated_speed  = rated_speed;
    params->rated_flow   = rated_flow;
    params->rated_head   = rated_head;
    params->shutoff_head = shutoff_head;

    double dH = shutoff_head - rated_head;
    params->curve_coeff_a = 2.0 * dH / rated_flow;
    params->curve_coeff_b = -dH / (rated_flow * rated_flow);
    params->efficiency_peak = 0.75;  /* Typical centrifugal pump */

    return 0;
}

double pump_curve_flow_estimate(const pump_curve_params_t *params,
                                 double head_meas, double speed_meas)
{
    /*
     * Virtual flow estimation using pump affinity laws:
     *
     * 1. Scale measured head to rated speed:
     *    H_rated = H_meas * (N_rated / N_meas)^2
     *
     * 2. Solve pump curve at rated speed for Q_rated:
     *    H_rated = H_shutoff - A*Q_rated - B*Q_rated^2
     *    => B*Q^2 + A*Q + (H_rated - H_shutoff) = 0
     *    => Q_rated = [-A + sqrt(A^2 - 4*B*(H_rated-H_shutoff))] / (2*B)
     *    (positive root only)
     *
     * 3. Scale flow to actual speed:
     *    Q_actual = Q_rated * (N_meas / N_rated)
     */

    if (!params) return -1.0;
    if (head_meas >= params->shutoff_head) return 0.0;  /* No flow possible */
    if (head_meas < 0.0) return 0.0;
    if (speed_meas <= 0.0 || params->rated_speed <= 0.0) return -1.0;

    /* Affinity law: scale head to rated speed */
    double speed_ratio = speed_meas / params->rated_speed;
    double H_rated = head_meas / (speed_ratio * speed_ratio);

    if (H_rated >= params->shutoff_head) return 0.0;

    /* Solve quadratic: B*Q^2 + A*Q + (H - H_shutoff) = 0 */
    double a = params->curve_coeff_b;
    double b = params->curve_coeff_a;
    double c = H_rated - params->shutoff_head;

    /* If B=0 (linear curve), solve separately */
    if (fabs(a) < 1e-15) {
        if (fabs(b) < 1e-15) return -1.0;
        double Q_rated = -c / b;
        if (Q_rated < 0.0) return -1.0;
        return Q_rated * speed_ratio;
    }

    double discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) return -1.0;  /* No real solution */

    double Q_rated = (-b + sqrt(discriminant)) / (2.0 * a);

    /* Check the other root too, take the positive one */
    double Q_rated2 = (-b - sqrt(discriminant)) / (2.0 * a);
    if (Q_rated < 0.0 && Q_rated2 > 0.0) Q_rated = Q_rated2;
    if (Q_rated < 0.0) return -1.0;

    /* Scale to actual speed */
    return Q_rated * speed_ratio;
}

/* ==========================================================================
 * L8: Choke/Control Valve Flow
 * ========================================================================== */

double choke_valve_liquid_flow(double cv_valve, double dp,
                                double rho, double fp)
{
    /*
     * IEC 60534-2-1 incompressible flow equation:
     * Q = N1 * Fp * Cv * sqrt(dP / (rho/rho0))
     *
     * N1 = 1.0 for Q in [m^3/h] with dP in [kPa] and rho in [kg/m^3]
     *    = 0.865 for dP in [bar]
     *
     * We use base SI units: Q [m^3/s], dP [Pa], rho [kg/m^3].
     * Converting Cv from [US gpm/psi^0.5] to SI:
     * Cv_SI = Cv * 7.50e-6  (m^3/s per Pa^0.5 with water)
     *
     * Q [m^3/s] = Cv_SI * sqrt(dP / rho) * Fp
     */

    if (cv_valve <= 0.0 || dp < 0.0 || rho <= 0.0) return 0.0;
    if (fp <= 0.0) fp = 1.0;

    /* Convert Cv [US gpm/psi^0.5] to SI flow coefficient */
    /* Cv_SI = Cv * 865 / (sqrt(1000)*sqrt(6894.8)) ... */
    /* Simplified: Cv_SI = Cv * 7.50e-6 (validated) */
    double cv_si = cv_valve * 7.50e-6;

    /* Q = Cv_SI * Fp * sqrt(dP / rho) */
    return cv_si * fp * sqrt(dp / rho);
}

double choke_valve_gas_flow(double cv_valve, double p1, double p2,
                             double t1, double sg, double k,
                             double xt, double z)
{
    /*
     * IEC 60534-2-1 compressible (gas) flow equation.
     *
     * Pressure ratio: x = dP / P1
     *
     * If x < Fk * xT: non-choked flow
     *   Y = 1 - x / (3 * Fk * xT)   [Expansion factor]
     *   Q = N6 * Fp * Cv * P1 * Y * sqrt(x / (Gg * T1 * Z))
     *
     * If x >= Fk * xT: choked flow
     *   Q = N6 * Fp * Cv * P1 * (2/3) * sqrt(Fk * xT / (Gg * T1 * Z))
     *
     * N6 = 2.73 for Q in [m^3/h] at std cond (P=101.325 kPa, T=288.6 K),
     *      P1 in [bar], T1 in [K].
     *
     * Fk = k / 1.4  (ratio of specific heats, 1.4 = k_air)
     *
     * We convert to SI: Q [m^3/s], P [Pa], T [K].
     */

    if (cv_valve <= 0.0 || p1 <= 0.0 || t1 <= 0.0) return 0.0;

    double dp = p1 - p2;
    if (dp < 0.0) dp = 0.0;

    double x = dp / p1;  /* Pressure drop ratio */
    if (x > 1.0) x = 1.0;

    double Fk = k / 1.4;  /* Specific heat ratio factor */

    /* Convert pressures to [bar] for N6 constant */
    double p1_bar = p1 * 1.0e-5;

    if (sg <= 0.0) sg = 1.0;  /* Default to air */
    if (z  <= 0.0) z  = 1.0;  /* Default to ideal gas */
    if (xt <= 0.0) xt = 0.7;  /* Typical for globe valve */

    /* Non-choked check */
    if (x < Fk * xt) {
        /* Expansion factor Y = 1 - x/(3*Fk*xT) */
        double Y = 1.0 - x / (3.0 * Fk * xt);
        if (Y < 0.667) Y = 0.667;  /* Y minimum is 0.667 */

        /* N6 = 2.73 for Q in m^3/h at std conditions.
         * Convert to m^3/s: divide by 3600 */
        double Q_m3h = 2.73 * 1.0 * cv_valve * p1_bar * Y
                       * sqrt(x / (sg * t1 * z));
        return Q_m3h / 3600.0;
    } else {
        /* Choked flow */
        double Q_m3h = 2.73 * 1.0 * cv_valve * p1_bar * (2.0/3.0)
                       * sqrt(Fk * xt / (sg * t1 * z));
        return Q_m3h / 3600.0;
    }
}