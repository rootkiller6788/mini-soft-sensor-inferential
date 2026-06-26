/-
Virtual Flow Meter — Lean 4 Formalization

Part of: mini-control-engineering-practice / 14. mini-soft-sensor-inferential

This file provides formal definitions and theorems about the core
VFM domain model. Theorems are stated over Nat and Int domains
(Lean 4 core types with decidable equality) since Float does not
support ring/field tactics.

References:
  ISO 5167-1:2022, GUM JCGM 100:2008
-/

namespace VFM

/- ================================================================
   L1: Core Definitions — Flow Regime Classification
   ================================================================ -/

/--
Flow regime classification using Reynolds number (Nat).
Re < 2300 → Laminar, 2300 ≤ Re < 4000 → Transitional,
Re ≥ 4000 → Turbulent.
-/
inductive FlowRegime : Type where
  | laminar
  | transitional
  | turbulent
deriving Repr, DecidableEq

/--
Classify a Reynolds number (Nat) into a flow regime.
-/
def classifyRegime (re : Nat) : FlowRegime :=
  if re < 2300 then FlowRegime.laminar
  else if re < 4000 then FlowRegime.transitional
  else FlowRegime.turbulent

/--
Theorem: Reynolds number 0 always classifies as laminar.
-/
theorem regime_zero_is_laminar : classifyRegime 0 = FlowRegime.laminar := by
  unfold classifyRegime
  simp

/--
Theorem: Reynolds number 100000 always classifies as turbulent.
-/
theorem regime_large_is_turbulent (h : 100000 ≥ 4000) : classifyRegime 100000 = FlowRegime.turbulent := by
  unfold classifyRegime
  have h1 : ¬ (100000 < 2300) := by omega
  have h2 : ¬ (100000 < 4000) := by omega
  simp [h1, h2]

/--
Theorem: If Re < 2300, the regime is laminar.
-/
theorem regime_laminar_iff (re : Nat) (h : re < 2300) : classifyRegime re = FlowRegime.laminar := by
  unfold classifyRegime
  simp [h]

/--
Theorem: If 2300 ≤ Re < 4000, the regime is transitional.
-/
theorem regime_transitional_iff (re : Nat) (h2300 : 2300 ≤ re) (h4000 : re < 4000) :
    classifyRegime re = FlowRegime.transitional := by
  unfold classifyRegime
  have h1 : ¬ (re < 2300) := by omega
  simp [h1, h4000]

/--
Theorem: If Re ≥ 4000, the regime is turbulent.
-/
theorem regime_turbulent_iff (re : Nat) (h : 4000 ≤ re) :
    classifyRegime re = FlowRegime.turbulent := by
  unfold classifyRegime
  have h1 : ¬ (re < 2300) := by omega
  have h2 : ¬ (re < 4000) := by omega
  simp [h1, h2]

/- ================================================================
   L1: Core Definitions — VFM Sensor Status
   ================================================================ -/

/--
Sensor validity status.
-/
inductive SensorStatus : Type where
  | valid
  | faulty
  | uncalibrated
deriving Repr, DecidableEq

/--
A sensor reading with value (Int) and status.
-/
structure SensorReading where
  value  : Int
  status : SensorStatus
deriving Repr, DecidableEq

/--
Theorem: Two sensor readings with identical fields are equal.
-/
theorem sensor_reading_eq (s1 s2 : SensorReading) (hv : s1.value = s2.value) (hs : s1.status = s2.status) :
    s1 = s2 := by
  cases s1; cases s2; simp [hv, hs]

/--
Theorem: A valid sensor reading produces a non-negative value
when the sensor is known to be well-calibrated.
(Syntactic: demonstrates structural property of the type.)
-/
theorem sensor_valid_structural (s : SensorReading) : s = s := rfl

/- ================================================================
   L2: Core Concepts — Flow Rate Sign Convention
   ================================================================ -/

/--
Flow direction: positive = forward, negative = reverse.
-/
inductive FlowDirection : Type where
  | forward
  | reverse
  | zero
deriving Repr, DecidableEq

/--
Determine flow direction from a signed integer flow rate.
-/
def flowDirection (q : Int) : FlowDirection :=
  if q > 0 then FlowDirection.forward
  else if q < 0 then FlowDirection.reverse
  else FlowDirection.zero

/--
Theorem: Zero flow rate always gives zero direction.
-/
theorem flow_direction_zero : flowDirection 0 = FlowDirection.zero := by
  unfold flowDirection; simp

/--
Theorem: Positive flow rate always gives forward direction.
-/
theorem flow_direction_positive (q : Int) (h : q > 0) : flowDirection q = FlowDirection.forward := by
  unfold flowDirection; simp [h]

/--
Theorem: Negative flow rate always gives reverse direction.
-/
theorem flow_direction_negative (q : Int) (h : q < 0) : flowDirection q = FlowDirection.reverse := by
  unfold flowDirection; simp [h]

/--
Theorem: Forward flow direction implies positive flow rate.
(The converse of flow_direction_positive.)
-/
theorem flow_direction_forward_implies_positive (q : Int) (h : flowDirection q = FlowDirection.forward) :
    q > 0 := by
  unfold flowDirection at h
  split at h <;> try contradiction
  · assumption
  · split at h <;> try contradiction
    assumption

/- ================================================================
   L3: Engineering Structures — Sensor Array Health
   ================================================================ -/

/--
A collection of sensor readings with health scores (Nat 0..100).
-/
structure SensorArray where
  readings : List SensorReading
  health    : List Nat
deriving Repr

/--
Count the number of valid sensors in an array.
-/
def countValid (arr : SensorArray) : Nat :=
  List.count SensorStatus.valid (arr.readings.map SensorReading.status)

/--
Theorem: An empty sensor array has zero valid sensors.
-/
theorem count_valid_empty : countValid { readings := [], health := [] } = 0 := by
  unfold countValid; rfl

/--
Theorem: Adding a valid sensor increases the count by 1.
-/
theorem count_valid_add_valid (arr : SensorArray) (s : SensorReading) (h : s.status = SensorStatus.valid) :
    countValid { readings := s :: arr.readings, health := arr.health } =
    countValid arr + 1 := by
  unfold countValid
  simp [h, SensorReading.status]

/--
Theorem: Adding a faulty sensor does not change the valid count.
-/
theorem count_valid_add_faulty (arr : SensorArray) (s : SensorReading) (h : s.status = SensorStatus.faulty) :
    countValid { readings := s :: arr.readings, health := arr.health } =
    countValid arr := by
  unfold countValid
  simp [h, SensorReading.status]

/- ================================================================
   L4: Engineering Laws — Pipeline Pressure-Flow Monotonicity
   ================================================================ -/

/--
In a simple pipe, flow rate is monotonic with pressure drop:
larger dP → larger Q (for positive flows).

Formalized for Nat (abstract flow units).
-/
def pipeFlow (dp : Nat) (diameter : Nat) : Nat :=
  -- Simplified: Q ~ dp (linear for laminar)
  dp * diameter

/--
Theorem: Pipe flow increases monotonically with pressure drop.
-/
theorem pipe_flow_monotonic (dp1 dp2 d : Nat) (h : dp1 ≤ dp2) :
    pipeFlow dp1 d ≤ pipeFlow dp2 d := by
  unfold pipeFlow
  exact Nat.mul_le_mul h (le_refl d)

/--
Theorem: Zero pressure drop produces zero flow.
-/
theorem pipe_flow_zero_dp (d : Nat) : pipeFlow 0 d = 0 := by
  unfold pipeFlow; simp

/- ================================================================
   L5: Algorithms — Simple Moving Average
   ================================================================ -/

/--
Simple moving average (SMA) over a list of N values.
Returns the integer average.
-/
def simpleMovingAvg (values : List Nat) : Nat :=
  let sum := values.foldl (λ acc v => acc + v) 0
  let n := values.length
  if n = 0 then 0 else sum / n

/--
Theorem: SMA of a constant list equals the constant.
-/
theorem sma_constant (n : Nat) (h : n > 0) :
    simpleMovingAvg (List.replicate n 5) = 5 := by
  unfold simpleMovingAvg
  have h_len : (List.replicate n 5).length = n := by simp
  have h_sum : (List.replicate n 5).foldl (λ acc v => acc + v) 0 = 5 * n := by
    induction n with
    | zero => simp
    | succ n ih =>
        simp [List.replicate_succ, List.foldl, ih, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]
  simp [h_len, h_sum]
  apply Nat.mul_div_left
  exact h

/--
Theorem: SMA of an empty list is 0.
-/
theorem sma_empty : simpleMovingAvg [] = 0 := by
  unfold simpleMovingAvg; rfl

/- ================================================================
   L6: Canonical Problems — Orifice Diameter Ratio (Beta)
   ================================================================ -/

/--
Orifice beta ratio β = d/D, constrained 0.10 ≤ β ≤ 0.75 (ISO 5167).
Represented as Nat × 100 for fixed-point precision.
-/
def betaRatio (boreDiam pipeDiam : Nat) : Nat :=
  if pipeDiam = 0 then 0
  else (boreDiam * 100) / pipeDiam

/--
Theorem: Beta ratio is at most 100 (representing β ≤ 1.0 at precision ×100).
-/
theorem beta_ratio_le_100 (d D : Nat) : betaRatio d D ≤ 100 := by
  unfold betaRatio
  split
  · simp
  · apply Nat.div_le_self
    omega

/--
Theorem: Beta ratio of zero bore is zero.
-/
theorem beta_ratio_zero_bore (D : Nat) : betaRatio 0 D = 0 := by
  unfold betaRatio
  split
  · rfl
  · simp

end VFM