/**
 * @file flow_models.h
 * @brief Physical flow models for virtual flow metering
 *
 * Knowledge Coverage:
 *   L1 Definitions: Orifice model, Bernoulli equation, Darcy-Weisbach
 *   L2 Core Concepts: Conservation of energy, mass, momentum in pipe flow
 *   L3 Engineering Structures: Parameterized flow model API
 *   L4 Engineering Laws: ISO 5167 orifice, Bernoulli principle,
 *       Darcy-Weisbach, Colebrook friction factor, pump affinity laws
 *
 * This module implements the fundamental physics-based flow models that
 * form the inferential core of a Virtual Flow Meter. Each model maps a
 * set of available measurements (differential pressure, pump speed,
 * valve position, etc.) to an estimated flow rate.
 *
 * References:
 *   ISO 5167-1:2022 Measurement of fluid flow by pressure differential devices
 *   Bernoulli, D. (1738) Hydrodynamica
 *   Darcy, H. (1857) Recherches experimentales relatives au mouvement
 *
 * @module mini-virtual-flow-meter
 */

#ifndef FLOW_MODELS_H
#define FLOW_MODELS_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * L1: Core Definitions -- Flow Model Parameters
 * ========================================================================== */

/**
 * @brief Orifice plate parameters per ISO 5167-1.
 *
 * Governing equation:
 *   Q = Cd * A_bore * sqrt(2*dP / (rho*(1-beta^4)))
 *
 * where Cd = discharge coefficient, A_bore = bore area, dP = differential
 * pressure, rho = fluid density, beta = d/D (bore/pipe diameter ratio).
 */
typedef struct {
    double pipe_diameter;
    double bore_diameter;
    double beta_ratio;
    double bore_area;
    double discharge_coeff;
    double expansibility_factor;
    double pressure_tap_spacing;
} orifice_params_t;

/**
 * @brief Venturi tube parameters (ISO 5167-4).
 */
typedef struct {
    double inlet_diameter;
    double throat_diameter;
    double beta_ratio;
    double throat_area;
    double discharge_coeff;
    double convergent_angle;
    double divergent_angle;
} venturi_params_t;

/**
 * @brief Bernoulli equation parameters.
 *
 * For steady, incompressible, inviscid flow:
 *   P1/rho + v1^2/2 + g*z1 = P2/rho + v2^2/2 + g*z2
 */
typedef struct {
    double pressure_upstream;
    double pressure_downstream;
    double velocity_upstream;
    double velocity_downstream;
    double elevation_upstream;
    double elevation_downstream;
    double energy_loss;
} bernoulli_params_t;

/**
 * @brief Pump curve parameters.
 *
 * Pump affinity laws (centrifugal pumps):
 *   Q1/Q2 = N1/N2
 *   H1/H2 = (N1/N2)^2
 *   P1/P2 = (N1/N2)^3
 *
 * Pump curve H(Q) at rated speed: H = H0 - A*Q - B*Q^2
 */
typedef struct {
    double rated_speed;
    double rated_flow;
    double rated_head;
    double shutoff_head;
    double curve_coeff_a;
    double curve_coeff_b;
    double efficiency_peak;
} pump_curve_params_t;

/**
 * @brief Choke/control valve flow parameters (IEC 60534-2-1).
 *
 * For incompressible flow:  Q = N1 * Cv * sqrt(dP / (rho/rho0))
 * For choked gas flow: P2/P1 < xT -> flow is sonic.
 */
typedef struct {
    double cv_coefficient;
    double xt_critical_ratio;
    double fl_liquid_recovery;
    double pipe_geometry_factor;
    double expansion_factor_y;
    double specific_heat_ratio;
} choke_params_t;

/* ==========================================================================
 * L2: Reynolds Number and Flow Regime
 * ========================================================================== */

/**
 * @brief Compute Reynolds number.
 *
 * Re = rho * v * D / mu = v * D / nu
 *
 * Re < 2300: laminar, Re >= 4000: turbulent.
 * Named after Osborne Reynolds (1883).
 *
 * @param velocity  Mean flow velocity [m/s]
 * @param diameter  Hydraulic diameter [m]
 * @param kin_visc  Kinematic viscosity [m^2/s]
 * @return Reynolds number (dimensionless)
 */
double flow_reynolds_number(double velocity, double diameter,
                             double kin_visc);

/**
 * @brief Determine flow regime from Reynolds number.
 *
 * @param re  Reynolds number
 * @return 0=laminar, 1=transitional, 2=turbulent
 */
int flow_regime_classify(double re);

/**
 * @brief Compute critical velocity for laminar-turbulent transition.
 *
 * v_crit = Re_crit * nu / D  (Re_crit = 2300)
 *
 * @param diameter  Pipe diameter [m]
 * @param kin_visc  Kinematic viscosity [m^2/s]
 * @return Critical velocity [m/s]
 */
double flow_critical_velocity(double diameter, double kin_visc);

/* ==========================================================================
 * L3-L4: Orifice Flow Model -- ISO 5167
 * ========================================================================== */

int orifice_params_init(orifice_params_t *params, double pipe_diam,
                         double bore_diam);

double orifice_discharge_coeff_iso5167(double beta, double re_d);

double orifice_mass_flow(const orifice_params_t *params, double dp,
                          double rho, double epsilon);

double orifice_vol_flow(const orifice_params_t *params, double dp,
                         double rho, double epsilon);

double orifice_expansibility(double beta, double dp, double p_upstream,
                              double k);

/* ==========================================================================
 * L4: Venturi Flow Model
 * ========================================================================== */

int venturi_params_init(venturi_params_t *params, double inlet_diam,
                         double throat_diam);

double venturi_mass_flow(const venturi_params_t *params, double dp,
                          double rho, double epsilon);

/* ==========================================================================
 * L5: Bernoulli Energy Balance
 * ========================================================================== */

double bernoulli_flow_rate(double p1, double p2, double area,
                            double rho, double dz, double head_loss);

/* ==========================================================================
 * L6: Darcy-Weisbach and Friction Factors
 * ========================================================================== */

double darcy_friction_laminar(double re);

double darcy_friction_colebrook(double re, double rel_roughness,
                                 int max_iter, double tol);

double darcy_weisbach_head_loss(double friction_factor, double length,
                                 double diameter, double velocity);

double darcy_inverse_flow(double diameter, double roughness, double length,
                           double head_loss, double kin_visc,
                           int max_iter, double tol);

/* ==========================================================================
 * L7: Pump Curve Flow Estimation
 * ========================================================================== */

int pump_curve_init(pump_curve_params_t *params, double rated_flow,
                     double rated_head, double shutoff_head,
                     double rated_speed);

double pump_curve_flow_estimate(const pump_curve_params_t *params,
                                 double head_meas, double speed_meas);

/* ==========================================================================
 * L8: Choke/Control Valve Flow
 * ========================================================================== */

double choke_valve_liquid_flow(double cv_valve, double dp,
                                double rho, double fp);

double choke_valve_gas_flow(double cv_valve, double p1, double p2,
                             double t1, double sg, double k,
                             double xt, double z);

#ifdef __cplusplus
}
#endif

#endif /* FLOW_MODELS_H */