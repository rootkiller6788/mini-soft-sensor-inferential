/**
 * @file pipeline_geometry.h
 * @brief Pipeline geometry and hydraulics for VFM
 *
 * Knowledge Coverage:
 *   L1 Definitions: Pipe diameter, roughness, fittings, minor losses
 *   L2 Core Concepts: Hydraulic diameter, equivalent length, K-factors
 *   L3 Engineering Structures: Pipeline network representation
 *   L4 Engineering Laws: Darcy-Weisbach, minor loss coefficients (Crane TP-410)
 *
 * Pipeline geometry determines how pressure measurements relate to
 * flow rate. Accurate geometric parameters are essential for VFM accuracy.
 *
 * References:
 *   Crane Co. (1988) "Flow of Fluids Through Valves, Fittings, and Pipe"
 *     Technical Paper No. 410
 *   Idelchik, I.E. (1994) "Handbook of Hydraulic Resistance"
 *
 * @module mini-virtual-flow-meter
 */

#ifndef PIPELINE_GEOMETRY_H
#define PIPELINE_GEOMETRY_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * L1: Core Types
 * ========================================================================== */

/**
 * @brief Pipe segment geometry parameters.
 */
typedef struct {
    double internal_diameter; /**< Inner diameter [m]                      */
    double wall_thickness;    /**< Wall thickness [m]                      */
    double absolute_roughness;/**< Surface roughness epsilon [m]           */
    double length;            /**< Pipe segment length [m]                 */
    double elevation_change;  /**< Elevation change dz = z_out - z_in [m] */
    double cross_section_area;/**< Flow area = pi*D^2/4 [m^2]             */
} pipe_segment_t;

/**
 * @brief Pipe fitting for minor loss calculation.
 *
 * Minor losses (also called local losses) are caused by fittings,
 * valves, bends, expansions, contractions, etc. They are expressed
 * as a loss coefficient K or equivalent length Le/D.
 *
 * hL_minor = K * v^2 / (2*g)
 * or equivalently: hL_minor = f * (Le/D) * v^2 / (2*g)
 */
typedef struct {
    int    type;           /**< Fitting type code (see enum below)          */
    double k_factor;       /**< Resistance coefficient K (dimensionless)    */
    double equiv_length_ratio; /**< Equivalent length ratio Le/D           */
    char   description[64];/**< Fitting description                         */
} pipe_fitting_t;

/** Fitting type codes (Crane TP-410 standard) */
typedef enum {
    FITTING_ELBOW_90      = 0,   /**< 90-degree elbow (standard radius)    */
    FITTING_ELBOW_45      = 1,   /**< 45-degree elbow                      */
    FITTING_ELBOW_90_LR   = 2,   /**< 90-degree long-radius elbow          */
    FITTING_TEE_THROUGH   = 3,   /**< Tee -- flow through run              */
    FITTING_TEE_BRANCH    = 4,   /**< Tee -- flow through branch           */
    FITTING_RETURN_BEND   = 5,   /**< 180-degree return bend               */
    FITTING_GATE_VALVE    = 6,   /**< Gate valve (fully open)              */
    FITTING_GLOBE_VALVE   = 7,   /**< Globe valve (fully open)             */
    FITTING_BALL_VALVE    = 8,   /**< Ball valve (fully open)              */
    FITTING_BUTTERFLY     = 9,   /**< Butterfly valve (fully open)         */
    FITTING_CHECK_SWING   = 10,  /**< Swing check valve                    */
    FITTING_CHECK_LIFT    = 11,  /**< Lift check valve                     */
    FITTING_SUDDEN_EXP    = 12,  /**< Sudden expansion                     */
    FITTING_SUDDEN_CONT   = 13,  /**< Sudden contraction                   */
    FITTING_PIPE_ENTRANCE = 14,  /**< Pipe entrance (from reservoir)       */
    FITTING_PIPE_EXIT     = 15   /**< Pipe exit (to reservoir)             */
} pipe_fitting_type_t;

/**
 * @brief Complete pipeline configuration for VFM.
 *
 * A pipeline is composed of multiple segments and fittings.
 * Total head loss = sum of major losses (Darcy-Weisbach in each segment)
 *                 + sum of minor losses (K-factors for each fitting)
 */
typedef struct {
    pipe_segment_t *segments;     /**< Array of pipe segments              */
    pipe_fitting_t *fittings;     /**< Array of pipe fittings              */
    int    num_segments;          /**< Number of pipe segments             */
    int    num_fittings;          /**< Number of pipe fittings             */
    double total_length;          /**< Total pipe length [m]               */
    double total_elevation;       /**< Total elevation change [m]          */
    double min_diameter;          /**< Minimum diameter in pipeline [m]    */
    double avg_roughness;         /**< Length-weighted avg roughness [m]   */
} pipeline_config_t;

/* ==========================================================================
 * L2: Pipe Geometry API
 * ========================================================================== */

/**
 * @brief Initialize a pipe segment from basic dimensions.
 *
 * @param seg       Pipe segment to initialize
 * @param id        Internal diameter [m]
 * @param length    Segment length [m]
 * @param roughness Absolute roughness [m]
 * @param dz        Elevation change [m] (positive = uphill)
 * @return 0 on success
 */
int pipe_segment_init(pipe_segment_t *seg, double id, double length,
                       double roughness, double dz);

/**
 * @brief Compute cross-sectional area of a circular pipe.
 *
 * A = pi * D^2 / 4
 *
 * @param diameter  Internal diameter [m]
 * @return Cross-sectional area [m^2]
 */
double pipe_cross_section_area(double diameter);

/**
 * @brief Compute hydraulic diameter for non-circular cross-sections.
 *
 * Dh = 4 * A / P_wetted
 *
 * where A = flow area, P_wetted = wetted perimeter.
 * For a circular pipe running full: Dh = D.
 *
 * @param area           Cross-sectional flow area [m^2]
 * @param wetted_perim   Wetted perimeter [m]
 * @return Hydraulic diameter [m]
 */
double pipe_hydraulic_diameter(double area, double wetted_perim);

/* ==========================================================================
 * L3: Fitting Loss Coefficients
 * ========================================================================== */

/**
 * @brief Initialize a fitting with standard K-factor.
 *
 * Uses Crane TP-410 tabulated K-factors for fully turbulent flow.
 * For laminar/transitional flow, K-factors are Reynolds-number dependent
 * and typically higher (not implemented here).
 *
 * @param fitting  Fitting to initialize
 * @param type     Fitting type (from enum)
 * @param size     Nominal pipe size [m] (for size-dependent K-factors)
 * @return 0 on success, -1 if fitting type unknown
 */
int pipe_fitting_init(pipe_fitting_t *fitting, pipe_fitting_type_t type,
                       double size);

/**
 * @brief Compute minor head loss for a fitting.
 *
 * hL_minor = K * v^2 / (2*g)
 *
 * @param fitting   Fitting with K-factor
 * @param velocity  Flow velocity [m/s]
 * @return Minor head loss [m of fluid]
 */
double pipe_minor_head_loss(const pipe_fitting_t *fitting, double velocity);

/**
 * @brief Compute total minor losses for an array of fittings.
 *
 * Sum of individual K * v^2/(2g) for all fittings.
 *
 * @param fittings    Array of fittings
 * @param n_fittings  Number of fittings
 * @param velocity    Flow velocity [m/s]
 * @return Total minor head loss [m]
 */
double pipe_total_minor_loss(const pipe_fitting_t *fittings,
                              int n_fittings, double velocity);

/* ==========================================================================
 * L4: Pipeline Total Head Loss
 * ========================================================================== */

/**
 * @brief Initialize a pipeline configuration from segment array.
 *
 * @param pipeline    Pipeline config to initialize
 * @param segments    Array of pipe segments
 * @param n_segments  Number of segments
 * @param fittings    Array of fittings (can be NULL if none)
 * @param n_fittings  Number of fittings
 * @return 0 on success
 */
int pipeline_config_init(pipeline_config_t *pipeline,
                          const pipe_segment_t *segments, int n_segments,
                          const pipe_fitting_t *fittings, int n_fittings);

/**
 * @brief Compute total head loss across a pipeline.
 *
 * Total head loss:
 *   hL_total = sum_i f_i*(L_i/D_i)*v_i^2/(2g)   [major losses]
 *            + sum_j K_j * v_j^2/(2g)            [minor losses]
 *            + dz_total                          [elevation]
 *
 * @param pipeline     Pipeline configuration
 * @param flow_rate    Volumetric flow rate [m^3/s]
 * @param friction     Darcy friction factor (assumed uniform in pipeline)
 * @param major_loss   Output: major loss component [m]
 * @param minor_loss   Output: minor loss component [m]
 * @return Total head loss [m]
 */
double pipeline_total_head_loss(const pipeline_config_t *pipeline,
                                 double flow_rate, double friction,
                                 double *major_loss, double *minor_loss);

/**
 * @brief Compute the equivalent length of a pipeline.
 *
 * The equivalent length is the length of straight pipe that would
 * produce the same head loss as the actual pipeline including fittings.
 *
 * Le_total = L_pipe + sum (K * D / f)
 *
 * @param pipeline  Pipeline configuration
 * @param friction  Darcy friction factor
 * @return Equivalent length [m]
 */
double pipeline_equivalent_length(const pipeline_config_t *pipeline,
                                   double friction);

/**
 * @brief Free pipeline configuration memory.
 *
 * @param pipeline  Pipeline to free
 */
void pipeline_config_free(pipeline_config_t *pipeline);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINE_GEOMETRY_H */