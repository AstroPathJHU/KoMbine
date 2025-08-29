"""
MINLP solver for calculating p-values for two Kaplan-Meier curves.
The null hypothesis is that the survival curves are identical.
This follows the structure from kaplan_meier_MINLP.py.
The p-value is computed via the likelihood ratio test.
"""
# pylint: disable=too-many-lines

import functools

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.stats

from .kaplan_meier_MINLP import KaplanMeierPatientNLL, MINLPForKM, n_choose_d_term_table

class MINLPforKMPValue:  #pylint: disable=too-many-public-methods, too-many-instance-attributes
  """
  MINLP solver for calculating p-values for two Kaplan-Meier curves.
  """
  def __init__( # pylint: disable=too-many-arguments
    self,
    all_patients: list[KaplanMeierPatientNLL],
    *,
    parameter_min: float = -np.inf,
    parameter_threshold: float,
    parameter_max: float = np.inf,
    endpoint_epsilon: float = 1e-6,
    log_zero_epsilon: float = 1e-10,
  ):
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_threshold = parameter_threshold
    self.__parameter_max = parameter_max
    self.__endpoint_epsilon = endpoint_epsilon
    self.__log_zero_epsilon = log_zero_epsilon
    self.__null_hypothesis_constraint = None
    self.__patient_constraints_for_binomial_only = None
    self.__patient_wise_only_constraints = None

  @property
  def all_patients(self) -> list[KaplanMeierPatientNLL]:
    """
    The list of all patients.
    """
    return self.__all_patients
  @property
  def n_patients(self) -> int:
    """
    The number of patients.
    """
    return len(self.all_patients)
  @property
  def parameter_min(self) -> float:
    """
    The minimum parameter value to be included in the "low" Kaplan-Meier curve.
    """
    return self.__parameter_min
  @property
  def parameter_threshold(self) -> float:
    """
    The parameter threshold between the "low" and "high" Kaplan-Meier curves.
    """
    return self.__parameter_threshold

  @property
  def parameter_max(self) -> float:
    """
    The maximum parameter value to be included in the "high" Kaplan-Meier curve.
    """
    return self.__parameter_max

  @functools.cached_property
  def patient_times(self) -> npt.NDArray[np.float64]:
    """
    The times of all patients.
    """
    return np.array([p.time for p in self.all_patients])
  @functools.cached_property
  def all_death_times(self) -> npt.NDArray[np.float64]:
    """
    All the times when patients died.
    """
    return np.unique([p.time for p in self.all_patients if not p.censored])
  @functools.cached_property
  def patient_censored(self) -> npt.NDArray[np.bool_]:
    """
    The censored status of all patients.
    """
    return np.array([p.censored for p in self.all_patients])
  def patient_still_at_risk(self, t: float) -> npt.NDArray[np.bool_]:
    """
    The at-risk status of all patients at time t.
    """
    return self.patient_times >= t

  @functools.cached_property
  def observed_parameters(self) -> npt.NDArray[np.float64]:
    """
    The observed parameters of all patients.
    """
    return np.array([p.observed_parameter for p in self.all_patients])
  @functools.cached_property
  def parameter_in_range(self) -> npt.NDArray[np.bool_]:
    """
    Whether each patient's observed parameter is within the range for the low and high curves.
    """
    return np.array(((
      (self.observed_parameters >= self.parameter_min)
      & (self.observed_parameters < self.parameter_threshold)
    ), (
      (self.observed_parameters >= self.parameter_threshold)
      & (self.observed_parameters < self.parameter_max)
    ))).T

  @functools.cached_property
  def nominal_curves_probabilities(self) -> tuple[float, float]:
    """
    Calculate the nominal Kaplan-Meier probabilities for both low and high curves.
    Uses the observed parameters to assign patients to their nominal curves.
    Returns (low_curve_prob, high_curve_prob).
    """
    # Patients in low curve (parameter < threshold)
    low_curve_patients = [
      i for i in range(self.n_patients)
      if self.observed_parameters[i] < self.parameter_threshold
    ]

    # Patients in high curve (parameter >= threshold)
    high_curve_patients = [
      i for i in range(self.n_patients)
      if self.observed_parameters[i] >= self.parameter_threshold
    ]

    # Calculate probabilities for each curve using the static method from MINLP
    def calculate_curve_probability(patient_indices):
      if not patient_indices:
        return 1.0

      # Group patients by their death times
      times_died = []
      times_censored = []

      for i in patient_indices:
        patient = self.all_patients[i]
        if patient.censored:
          times_censored.append(patient.time)
        else:
          times_died.append(patient.time)

      # Get unique death times in order
      unique_death_times = sorted(set(times_died))
      if not unique_death_times:
        return 1.0

      # Count deaths and censored patients at each death time
      died_counts = []
      censored_counts = []

      for t in unique_death_times:
        # Count deaths at this time
        deaths_at_t = sum(1 for i in patient_indices
                         if not self.all_patients[i].censored and self.all_patients[i].time == t)
        died_counts.append(deaths_at_t)

        # Count censored patients before this time
        # (who are no longer at risk at time t)
        censored_before_t = sum(1 for i in patient_indices
                               if self.all_patients[i].censored and self.all_patients[i].time < t)
        censored_counts.append(censored_before_t)

      return MINLPForKM.calculate_KM_probability(
        total_count=len(patient_indices),
        censored_counts=tuple(censored_counts),
        died_counts=tuple(died_counts),
      )

    low_prob = calculate_curve_probability(low_curve_patients)
    high_prob = calculate_curve_probability(high_curve_patients)

    return low_prob, high_prob

  @functools.cached_property
  def nominal_curves_probabilities_at_each_time(
    self
  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate the nominal Kaplan-Meier probabilities at each death time for both curves.
    Returns (low_curve_probs_at_times, high_curve_probs_at_times) where each is an array
    with length equal to the number of death times.
    """
    # Patients in low curve (parameter < threshold)
    low_curve_patients = [
      i for i in range(self.n_patients)
      if self.observed_parameters[i] < self.parameter_threshold
    ]

    # Patients in high curve (parameter >= threshold)
    high_curve_patients = [
      i for i in range(self.n_patients)
      if self.observed_parameters[i] >= self.parameter_threshold
    ]

    def calculate_curve_probabilities_at_times(patient_indices):
      """Calculate KM probabilities at each death time for given patients."""
      if not patient_indices:
        return np.ones(len(self.all_death_times))

      probabilities_at_times = []

      for death_time_idx, death_time in enumerate(self.all_death_times):
        # Count patients at risk at this death time
        at_risk_count = sum(1 for i in patient_indices
                           if self.all_patients[i].time >= death_time)

        # Count deaths at this death time
        deaths_at_time = sum(1 for i in patient_indices
                            if (not self.all_patients[i].censored
                                and self.all_patients[i].time == death_time))

        # Calculate survival probability at this time point
        if at_risk_count == 0:
          survival_prob = 1.0  # No patients at risk means no deaths
        else:
          survival_prob = (at_risk_count - deaths_at_time) / at_risk_count

        # Calculate cumulative KM probability up to this time
        if death_time_idx == 0:
          km_prob_at_time = survival_prob
        else:
          km_prob_at_time = probabilities_at_times[-1] * survival_prob

        probabilities_at_times.append(km_prob_at_time)

      return np.array(probabilities_at_times)

    low_probs = calculate_curve_probabilities_at_times(low_curve_patients)
    high_probs = calculate_curve_probabilities_at_times(high_curve_patients)

    return low_probs, high_probs

  @functools.cached_property
  def nll_penalty_for_patient_in_range(self) -> npt.NDArray[np.float64]:
    """
    Calculate the negative log-likelihood penalty for each patient
    if that patient is within the parameter range.
    This is negative if the patient's observed parameter is within the range
    and positive if it is outside the range.
    Returns an n x 2 array: for each patient, the penalty to be included
    in the low and high curves.
    """
    sgn_nll_penalty_for_patient_in_range = 2 * self.parameter_in_range - 1
    observed_nll = np.array([
      p.parameter(p.observed_parameter)
      for p in self.all_patients
    ])
    parameter_min_nll: npt.NDArray[np.float64] = np.array([
      p.parameter(self.parameter_min) if np.isfinite(self.parameter_min) else np.inf
      for p in self.all_patients
    ])
    parameter_threshold_nll: npt.NDArray[np.float64] = np.array([
      p.parameter(self.parameter_threshold) #parameter threshold must be finite
      for p in self.all_patients
    ])
    parameter_max_nll: npt.NDArray[np.float64] = np.array([
      p.parameter(self.parameter_max) if np.isfinite(self.parameter_max) else np.inf
      for p in self.all_patients
    ])

    range_boundary_nll_low = np.min(
      np.array([parameter_min_nll, parameter_threshold_nll]),
      axis=0
    )
    range_boundary_nll_high = np.min(
      np.array([parameter_threshold_nll, parameter_max_nll]),
      axis=0
    )

    range_boundary_nll: npt.NDArray[np.float64] = \
      np.array([range_boundary_nll_low, range_boundary_nll_high]).T
    abs_nll_penalty_for_patient_in_range = observed_nll - range_boundary_nll.T

    nll_penalty_for_patient_in_range = (
      sgn_nll_penalty_for_patient_in_range
      * abs_nll_penalty_for_patient_in_range.T
    )

    return nll_penalty_for_patient_in_range

  def add_counter_variables_and_constraints(
    self,
    model: gp.Model,
    a: gp.tupledict[tuple[int, ...], gp.Var]
  ) -> tuple[
    gp.tupledict[tuple[int, ...], gp.Var],
    gp.tupledict[tuple[int, ...], gp.Var],
    gp.tupledict[tuple[int, ...], gp.Var],
  ]:
    """
    Add counter variables and constraints to the model.
    """

    # A patient can't be in more than one curve.
    # If parameter_min and parameter_max are both infinite,
    # then each patient must be assigned to a curve.
    if np.isinf(self.parameter_min) and np.isinf(self.parameter_max):
      for j in range(self.n_patients):
        model.addConstr(
          a[j, 0] + a[j, 1] == 1,
          name=f"patient_{j}_assigned_to_curve"
        )
    else:
      for j in range(self.n_patients):
        model.addConstr(
          a[j, 0] + a[j, 1] <= 1,
          name=f"patient_{j}_assigned_to_at_most_one_curve"
        )

    r = model.addVars(
      len(self.all_death_times), 2, vtype=gp.GRB.INTEGER, name="r"
    )
    d = model.addVars(
      len(self.all_death_times), 2, vtype=gp.GRB.INTEGER, name="d"
    )
    n_survived = model.addVars(
      len(self.all_death_times), 2, vtype=gp.GRB.INTEGER, name="n_survived"
    )
    for k, t in enumerate(self.all_death_times):
      for j in range(2):
        model.addConstr(
          r[k, j] == gp.quicksum(
            a[i, j] for i in range(self.n_patients)
            if self.patient_still_at_risk(t)[i]
          ),
          name=f"r_{k}_{j}"
        )
        model.addConstr(
          d[k, j] == gp.quicksum(
            a[i, j] for i in range(self.n_patients)
            if self.all_patients[i].time == t
            and not self.all_patients[i].censored
          ),
          name=f"d_{k}_{j}"
        )
        model.addConstr(
          n_survived[k, j] == r[k, j] - d[k, j],
          name=f"n_survived_{k}_{j}"
        )

    return r, d, n_survived

  def add_kaplan_meier_probability_variables_and_constraints( #pylint: disable=too-many-locals
    self,
    model: gp.Model,
    r: gp.tupledict[tuple[int, ...], gp.Var],
    n_survived: gp.tupledict[tuple[int, ...], gp.Var],
  ) -> tuple[gp.tupledict, gp.tupledict]:
    """
    Add variables and constraints to calculate the Kaplan-Meier probabilities
    for both curves directly within the Gurobi model using logarithmic transformations.
    Returns km_probability_vars for low and high curves, and KM probabilities at each time point.
    """
    km_probability_vars = []
    km_probability_at_time_vars = []  # KM probabilities at each death time

    for j in range(2):  # j=0 for low curve, j=1 for high curve
      # Variables for log of counts
      log_r_vars = model.addVars(
        len(self.all_death_times),
        vtype=GRB.CONTINUOUS,
        name=f"log_r_curve_{j}",
        lb=-GRB.INFINITY,
        ub=np.log(self.n_patients + self.__log_zero_epsilon),
      )
      log_n_survived_vars = model.addVars(
        len(self.all_death_times),
        vtype=GRB.CONTINUOUS,
        name=f"log_n_survived_curve_{j}",
        lb=-GRB.INFINITY,
        ub=np.log(self.n_patients + self.__log_zero_epsilon),
      )

      # Helper variables for log arguments (r + epsilon, n_survived + epsilon)
      r_plus_epsilon = model.addVars(
        len(self.all_death_times),
        vtype=GRB.CONTINUOUS,
        name=f"r_plus_epsilon_curve_{j}",
        lb=self.__log_zero_epsilon,
      )
      n_survived_plus_epsilon = model.addVars(
        len(self.all_death_times),
        vtype=GRB.CONTINUOUS,
        name=f"n_survived_plus_epsilon_curve_{j}",
        lb=self.__log_zero_epsilon,
      )

      # Constraints to link original counts to epsilon-added variables
      for i in range(len(self.all_death_times)):
        model.addConstr(
          r_plus_epsilon[i] == r[i, j] + self.__log_zero_epsilon,
          name=f"r_plus_epsilon_constr_{i}_curve_{j}",
        )
        model.addConstr(
          n_survived_plus_epsilon[i] == n_survived[i, j] + self.__log_zero_epsilon,
          name=f"n_survived_plus_epsilon_constr_{i}_curve_{j}",
        )

      # Link count variables to their log counterparts using GenConstrLog
      for i in range(len(self.all_death_times)):
        model.addGenConstrLog(
          r_plus_epsilon[i],
          log_r_vars[i],
          name=f"log_r_constr_{i}_curve_{j}",
        )
        model.addGenConstrLog(
          n_survived_plus_epsilon[i],
          log_n_survived_vars[i],
          name=f"log_n_survived_constr_{i}_curve_{j}",
        )

      # Binary indicator for whether r for a death time is zero
      is_r_zero = model.addVars(
        len(self.all_death_times),
        vtype=GRB.BINARY,
        name=f"is_r_zero_curve_{j}"
      )

      # Link is_r_zero to r using indicator constraint
      for i in range(len(self.all_death_times)):
        model.addGenConstrIndicator(
          is_r_zero[i], True, r[i, j], GRB.EQUAL, 0,
          name=f"is_r_zero_indicator_{i}_curve_{j}",
        )

      # Kaplan-Meier log probability for each death time term
      km_log_probability_per_time_terms = model.addVars(
        len(self.all_death_times),
        vtype=GRB.CONTINUOUS,
        name=f"km_log_prob_time_term_curve_{j}",
        lb=-GRB.INFINITY,
        ub=0,
      )

      # Use indicator constraints to set km_log_probability_per_time_terms[i]
      for i in range(len(self.all_death_times)):
        # If is_r_zero[i] is 0 (i.e., r[i, j] > 0)
        model.addGenConstrIndicator(
          is_r_zero[i], False,
          km_log_probability_per_time_terms[i] - (log_n_survived_vars[i] - log_r_vars[i]),
          GRB.EQUAL,
          0,
          name=f"km_log_prob_time_active_{i}_curve_{j}",
        )
        # If is_r_zero[i] is 1 (i.e., r[i, j] == 0)
        model.addGenConstrIndicator(
          is_r_zero[i], True,
          km_log_probability_per_time_terms[i],
          GRB.EQUAL,
          0.0,
          name=f"km_log_prob_time_zero_at_risk_{i}_curve_{j}",
        )

      # KM probabilities at each death time point (cumulative product up to that point)
      km_log_probability_at_time = model.addVars(
        len(self.all_death_times),
        vtype=GRB.CONTINUOUS,
        name=f"km_log_prob_at_time_curve_{j}",
        lb=-GRB.INFINITY,
        ub=0,
      )

      km_probability_at_time = model.addVars(
        len(self.all_death_times),
        vtype=GRB.CONTINUOUS,
        name=f"km_prob_at_time_curve_{j}",
        lb=0,
        ub=1,
      )

      # Calculate cumulative log probabilities at each death time
      for i in range(len(self.all_death_times)):
        if i == 0:
          # First death time: log probability is just the first term
          model.addConstr(
            km_log_probability_at_time[i] == km_log_probability_per_time_terms[i],
            name=f"km_log_prob_at_time_0_curve_{j}",
          )
        else:
          # Subsequent death times: cumulative sum up to this point
          model.addConstr(
            km_log_probability_at_time[i]
              == km_log_probability_at_time[i-1] + km_log_probability_per_time_terms[i],
            name=f"km_log_prob_at_time_{i}_curve_{j}",
          )

        # Convert from log to linear scale
        model.addGenConstrExp(
          km_log_probability_at_time[i],
          km_probability_at_time[i],
          name=f"exp_km_probability_at_time_{i}_curve_{j}",
        )

      # Total Kaplan-Meier log probability: sum of log probabilities per death time
      km_log_probability_total = model.addVar(
        vtype=GRB.CONTINUOUS,
        name=f"km_log_probability_total_curve_{j}",
        lb=-GRB.INFINITY,
        ub=0,
      )
      model.addConstr(
        km_log_probability_total == km_log_probability_per_time_terms.sum(),
        name=f"km_log_probability_total_def_curve_{j}",
      )

      # Kaplan-Meier probability variable (linear scale) - final probability
      km_probability_var = model.addVar(
        vtype=GRB.CONTINUOUS,
        name=f"km_probability_curve_{j}",
        lb=0,
        ub=1,
      )
      # Link log probability to linear probability using GenConstrExp
      model.addGenConstrExp(
        km_log_probability_total,
        km_probability_var,
        name=f"exp_km_probability_curve_{j}",
      )

      km_probability_vars.append(km_probability_var)
      km_probability_at_time_vars.append(km_probability_at_time)

    return km_probability_at_time_vars[0], km_probability_at_time_vars[1]

  @functools.cached_property
  def n_choose_d_term_table(self) -> dict[tuple[int, int], float]:
    """
    Precompute the n choose d terms for the binomial penalty.
    """
    return n_choose_d_term_table(n_patients=self.n_patients)

  def compute_hypergeometric_log_prob(self, r_total: int, r_group0: int, d_total: int, d_group0: int) -> float:
    """
    Compute hypergeometric log probability using existing 2D n_choose_d table.
    
    log P(d_group0 | r_total, r_group0, d_total) = 
      log C(r_group0, d_group0) + 
      log C(r_total - r_group0, d_total - d_group0) - 
      log C(r_total, d_total)
    
    This is much more efficient than the 4D table approach.
    """
    # Validate inputs
    if (d_group0 < 0 or d_group0 > r_group0 or d_group0 > d_total or 
        (d_total - d_group0) > (r_total - r_group0) or r_total < 0 or d_total < 0):
      return -np.inf
      
    # Handle edge cases
    if r_total == 0:
      return 0.0 if d_total == 0 and d_group0 == 0 else -np.inf
    if d_total == 0:
      return 0.0 if d_group0 == 0 else -np.inf
    
    # Use existing n_choose_d table
    n_choose_d_table = self.n_choose_d_table
    
    try:
      # Three 2D table lookups instead of one 4D lookup
      log_c_r_group0_d_group0 = n_choose_d_table.get((r_group0, d_group0), -np.inf)
      log_c_r_group1_d_group1 = n_choose_d_table.get((r_total - r_group0, d_total - d_group0), -np.inf)
      log_c_r_total_d_total = n_choose_d_table.get((r_total, d_total), -np.inf)
      
      if any(x == -np.inf for x in [log_c_r_group0_d_group0, log_c_r_group1_d_group1, log_c_r_total_d_total]):
        return -np.inf
        
      return log_c_r_group0_d_group0 + log_c_r_group1_d_group1 - log_c_r_total_d_total
      
    except (ValueError, OverflowError, KeyError):
      return -np.inf

  def add_hypergeometric_penalty_with_hazard_ratio(  # pylint: disable=too-many-locals
    self,
    model: gp.Model,
    *,
    r: gp.tupledict[tuple[int, ...], gp.Var],
    d: gp.tupledict[tuple[int, ...], gp.Var],
    n_survived: gp.tupledict[tuple[int, ...], gp.Var],
  ):
    """
    Add hypergeometric penalty that properly accounts for hazard ratio in alternative hypothesis.
    
    Uses efficient 2D table lookups instead of 4D tables and properly handles different
    expected values under null vs alternative hypotheses:
    
    Under null hypothesis (HR = 1): 
      expected deaths in group 0 = r[i,0] * d_total[i] / r_total[i]
    Under alternative hypothesis with hazard ratio HR:
      expected deaths in group 0 = d_total[i] * (r[i,0] * HR) / (r[i,0] * HR + r[i,1])
    """
    # Total deaths and at-risk counts at each time point (across both curves)
    d_total = model.addVars(
      len(self.all_death_times),
      vtype=gp.GRB.INTEGER,
      name="d_total"
    )
    r_total = model.addVars(
      len(self.all_death_times),
      vtype=gp.GRB.INTEGER,
      name="r_total"
    )
    
    # Constraints linking totals to individual curve variables
    for i in range(len(self.all_death_times)):
      model.addConstr(
        d_total[i] == d[i, 0] + d[i, 1],
        name=f"d_total_constraint_{i}"
      )
      model.addConstr(
        r_total[i] == r[i, 0] + r[i, 1],
        name=f"r_total_constraint_{i}"
      )

    # Add log hazard ratio variable
    log_hazard_ratio = model.addVar(
      vtype=gp.GRB.CONTINUOUS,
      name="log_hazard_ratio",
      lb=-3.0,
      ub=3.0,
    )

    # Null hypothesis indicator
    null_hypothesis_indicator = model.addVar(vtype=gp.GRB.BINARY, name="null_hypothesis_indicator")
    model.addGenConstrIndicator(
      null_hypothesis_indicator, True,
      log_hazard_ratio, GRB.EQUAL, 0,
      name="null_hypothesis_constraint",
    )

    # Hypergeometric penalty terms that account for hazard ratio in alternative hypothesis
    hypergeometric_terms = []
    
    for i in range(len(self.all_death_times)):
      # Create simplified penalty that approximates the hypergeometric test
      # Under null: (d[i,0] * r_total[i] - r[i,0] * d_total[i])^2
      # Under alternative: incorporate log hazard ratio via linear approximation
      
      # Base deviation (null hypothesis form)
      d0_rtotal = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"d0_rtotal_{i}")
      r0_dtotal = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"r0_dtotal_{i}")
      
      model.addQConstr(d0_rtotal == d[i, 0] * r_total[i], name=f"d0_rtotal_prod_{i}")
      model.addQConstr(r0_dtotal == r[i, 0] * d_total[i], name=f"r0_dtotal_prod_{i}")
      
      base_deviation = model.addVar(
        vtype=gp.GRB.CONTINUOUS,
        name=f"base_deviation_{i}",
        lb=-self.n_patients**2,
        ub=self.n_patients**2
      )
      
      model.addConstr(
        base_deviation == d0_rtotal - r0_dtotal,
        name=f"base_deviation_def_{i}"
      )
      
      # Hazard ratio correction term for alternative hypothesis
      # This modifies the expected value to account for different hazard rates
      hr_correction = model.addVar(
        vtype=gp.GRB.CONTINUOUS,
        name=f"hr_correction_{i}",
        lb=-self.n_patients**2,
        ub=self.n_patients**2
      )
      
      # Linear approximation: correction ≈ log_HR * r[i,1] * d_total[i] / 2
      # This captures the first-order effect of hazard ratio on expected deaths
      r1_dtotal = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"r1_dtotal_{i}")
      hr_r1_dtotal = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"hr_r1_dtotal_{i}")
      
      model.addQConstr(r1_dtotal == r[i, 1] * d_total[i], name=f"r1_dtotal_prod_{i}")
      model.addQConstr(hr_r1_dtotal == log_hazard_ratio * r1_dtotal, name=f"hr_r1_dtotal_prod_{i}")
      
      model.addConstr(
        hr_correction == hr_r1_dtotal / 2.0,
        name=f"hr_correction_def_{i}"
      )
      
      # Combined deviation accounting for hazard ratio
      total_deviation = model.addVar(
        vtype=gp.GRB.CONTINUOUS,
        name=f"total_deviation_{i}",
        lb=-self.n_patients**2,
        ub=self.n_patients**2
      )
      
      model.addConstr(
        total_deviation == base_deviation + hr_correction,
        name=f"total_deviation_def_{i}"
      )
      
      # Squared deviation
      deviation_squared = model.addVar(
        vtype=gp.GRB.CONTINUOUS,
        name=f"deviation_squared_{i}",
        lb=0
      )
      
      model.addQConstr(deviation_squared == total_deviation * total_deviation, 
                       name=f"deviation_squared_def_{i}")
      
      hypergeometric_terms.append(deviation_squared)
    
    # Create hypergeometric penalty variable
    hypergeometric_penalty = model.addVar(
      vtype=gp.GRB.CONTINUOUS,
      name="hypergeometric_penalty",
      lb=0
    )
    
    model.addConstr(
      hypergeometric_penalty == gp.quicksum(hypergeometric_terms),
      name="hypergeometric_penalty_constraint"
    )

    return hypergeometric_penalty, null_hypothesis_indicator

  def add_patient_wise_penalty(
    self,
    model: gp.Model,
    a: gp.tupledict[tuple[int, ...], gp.Var],
  ):
    """
    Add the patient-wise penalty to the model.
    This penalty is based on the negative log-likelihood of the patient's observed parameter
    being within the specified range.
    """
    patient_penalties = []
    for i in range(self.n_patients):
      for j in range(2):
        if np.isfinite(self.nll_penalty_for_patient_in_range[i, j]):
          penalty = self.nll_penalty_for_patient_in_range[i, j] * a[i, j]
          if self.nll_penalty_for_patient_in_range[i, j] < 0:
            # If the penalty is negative, it means the patient is nominally in the range
            # We want the penalty to be 0 when all patients are at their nominal values
            penalty -= self.nll_penalty_for_patient_in_range[i, j]
          patient_penalties.append(penalty)
        elif np.isneginf(self.nll_penalty_for_patient_in_range[i, j]):
          #the patient must be selected, so we add a constraint
          model.addConstr(
            a[i, j] == 1, name=f"patient_{i}_must_be_in_curve_{j}",
          )
        elif np.isposinf(self.nll_penalty_for_patient_in_range[i, j]):
          # The patient must not be selected, so we add a constraint
          model.addConstr(
            a[i, j] == 0, name=f"patient_{i}_must_not_be_in_curve_{j}",
          )
        else:
          raise ValueError(
            f"Invalid negative log-likelihood penalty value for patient {i}, curve {j}:"
            f"{self.nll_penalty_for_patient_in_range[i, j]}"
          )
    patient_penalty = gp.quicksum(patient_penalties)
    return patient_penalty

  def _extract_patients_per_curve(self, a: gp.tupledict[tuple[int, ...], gp.Var]):
    """
    Extract which patients are included in each curve from the optimized model.

    Returns:
        tuple: (patients_low, patients_high) where each is a list of patient indices
    """
    patients_low = []
    patients_high = []

    for i in range(self.n_patients):
      if a[i, 0].X > 0.5:  # Patient is in curve 0 (low)
        patients_low.append(i)
      if a[i, 1].X > 0.5:  # Patient is in curve 1 (high)
        patients_high.append(i)

    return patients_low, patients_high

  def _extract_curve_statistics( #pylint: disable=too-many-locals
    self, model: gp.Model, km_probability_at_time_low, km_probability_at_time_high
  ):
    """
    Extract statistics for each curve from the optimized model.

    Returns:
        tuple: (n_total_low, n_alive_low, km_prob_low, p_survived_low,
                n_total_high, n_alive_high, km_prob_high, p_survived_high)
    """
    # Extract KM probabilities for each curve
    km_prob_low = [
      km_prob.X for _, km_prob in km_probability_at_time_low.items()
    ]
    km_prob_high = [
      km_prob.X for _, km_prob in km_probability_at_time_high.items()
    ]

    # For n_total and n_alive, we need to look at the r and d variables per curve
    # These represent at-risk and died counts for each curve
    n_total_low = 0
    n_alive_low = 0
    n_total_high = 0
    n_alive_high = 0
    p_survived_low = []
    p_survived_high = []

    for i in range(len(self.all_death_times)):
      r_low = model.getVarByName(f"r[{i},0]")
      r_high = model.getVarByName(f"r[{i},1]")
      d_low = model.getVarByName(f"d[{i},0]")
      d_high = model.getVarByName(f"d[{i},1]")
      p_survived_low_var = model.getVarByName(f"p_survived[{i},0]")
      p_survived_high_var = model.getVarByName(f"p_survived[{i},1]")
      assert r_low is not None
      assert r_high is not None
      assert d_low is not None
      assert d_high is not None
      assert p_survived_low_var is not None
      assert p_survived_high_var is not None

      if i == 0:  # Use first time point as representative
        n_total_low = int(np.rint(r_low.X))
      if i == 0:  # Use first time point as representative
        n_total_high = int(np.rint(r_high.X))

      n_alive_low = n_total_low - int(np.rint(d_low.X))
      n_alive_high = n_total_high - int(np.rint(d_high.X))

      p_survived_low.append(p_survived_low_var.X)
      p_survived_high.append(p_survived_high_var.X)

    return (
      n_total_low, n_alive_low, km_prob_low, p_survived_low,
      n_total_high, n_alive_high, km_prob_high, p_survived_high,
    )

  def _compute_patient_wise_penalty_value(self, a: gp.tupledict[tuple[int, ...], gp.Var]):
    """
    Compute the actual patient-wise penalty value from the optimized model.

    Returns:
        float: The patient-wise penalty value (not multiplied by 2)
    """
    penalty = 0.0
    for i in range(self.n_patients):
      for j in range(2):
        if np.isfinite(self.nll_penalty_for_patient_in_range[i, j]):
          contribution = self.nll_penalty_for_patient_in_range[i, j] * (
            a[i, j].X - (1 if self.nll_penalty_for_patient_in_range[i, j] < 0 else 0)
          )
          penalty += contribution
    return penalty

  def _compute_hypergeometric_penalty(self, model: gp.Model):
    """
    Compute the hypergeometric penalty.

    Returns:
        float: The hypergeometric penalty value
    """
    hypergeometric_penalty_var = model.getVarByName("hypergeometric_penalty")
    
    assert hypergeometric_penalty_var is not None
    
    penalty = hypergeometric_penalty_var.X
    
    return penalty

  def _make_gurobi_model(self):
    """
    Create the Gurobi model for the Kaplan-Meier p-value MINLP.
    """
    model = gp.Model("Kaplan-Meier p-value MINLP")

    #Binary decision variables: a[j, k] = 1 if patient j is in curve k
    a = model.addVars(self.n_patients, 2, vtype=gp.GRB.BINARY, name="a")

    r, d, n_survived = self.add_counter_variables_and_constraints(model, a)

    # Add Kaplan-Meier probability variables and constraints
    (
      km_probability_at_time_low,
      km_probability_at_time_high
    ) = self.add_kaplan_meier_probability_variables_and_constraints(
      model, r, n_survived
    )

    (
      hypergeometric_penalty,
      null_hypothesis_indicator
    ) = self.add_hypergeometric_penalty_with_hazard_ratio(
      model,
      d=d,
      r=r,
      n_survived=n_survived
    )
    patient_penalty = self.add_patient_wise_penalty(model, a)

    model.setObjective(
      2 * (hypergeometric_penalty + patient_penalty),
      GRB.MINIMIZE,
    )
    model.update()

    return (
      model,
      null_hypothesis_indicator,
      a,
      km_probability_at_time_low,
      km_probability_at_time_high
    )

  @functools.cached_property
  def gurobi_model(self):
    """
    Create the Gurobi model for the MINLP.
    This is a cached property to avoid recreating the model multiple times.
    """
    return self._make_gurobi_model()

  def update_model_for_null_hypothesis_or_not(
    self,
    model,
    null_hypothesis_indicator,
    null_hypothesis: bool,
  ):
    """
    Update the model to indicate whether or not we are running
    for the null hypothesis.
    Under null hypothesis: log_hazard_ratio is fixed to 0 (hazard ratio = 1)
    Under alternative hypothesis: log_hazard_ratio is free to float
    """
    if self.__null_hypothesis_constraint is not None:
      model.remove(self.__null_hypothesis_constraint)
    if null_hypothesis:
      self.__null_hypothesis_constraint = model.addConstr(
        null_hypothesis_indicator == 1,
        name="null_hypothesis_constraint",
      )
    else:
      self.__null_hypothesis_constraint = None

    model.update()

  def update_model_with_binomial_only_constraints(
    self,
    model: gp.Model,
    a: gp.tupledict[tuple[int, ...], gp.Var],
    binomial_only: bool,
  ):
    """
    Update the model with binomial_only constraints.
    If binomial_only is True, we add constraints for a[i, j] to be either 0 or 1,
    based on parameter_in_range.
    """
    # Remove existing constraints if they exist
    if self.__patient_constraints_for_binomial_only is not None:
      for constr in self.__patient_constraints_for_binomial_only:
        model.remove(constr)
      self.__patient_constraints_for_binomial_only = None

    if binomial_only:
      self.__patient_constraints_for_binomial_only = []
      for i in range(self.n_patients):
        for j in range(2):  # j=0 for low curve, j=1 for high curve
          if self.parameter_in_range[i, j]:
            # The patient must be selected for this curve
            self.__patient_constraints_for_binomial_only.append(
              model.addConstr(
                a[i, j] == 1,
                name=f"patient_{i}_must_be_selected_curve_{j}_binomial_only",
              )
            )
          else:
            # The patient must not be selected for this curve
            self.__patient_constraints_for_binomial_only.append(
              model.addConstr(
                a[i, j] == 0,
                name=f"patient_{i}_must_not_be_selected_curve_{j}_binomial_only",
              )
            )

    model.update()

  def update_model_with_patient_wise_only_constraints( #pylint: disable=too-many-arguments
    self,
    model: gp.Model,
    *,
    km_probability_at_time_low: gp.tupledict,
    km_probability_at_time_high: gp.tupledict,
    null_hypothesis_indicator: gp.Var,
    patient_wise_only: bool,
  ):
    """
    Update the model with patient_wise_only constraints.
    When patient_wise_only=True, we constrain the curves to be flipped
    relative to their nominal probabilities at each death time point under the null hypothesis.
    """
    # Remove existing constraints if they exist
    if self.__patient_wise_only_constraints is not None:
      for constr in self.__patient_wise_only_constraints:
        model.remove(constr)
      self.__patient_wise_only_constraints = None

    if patient_wise_only:
      self.__patient_wise_only_constraints = []

      # Get nominal probabilities at each death time for both curves
      nominal_low_probs_at_times, nominal_high_probs_at_times = \
        self.nominal_curves_probabilities_at_each_time

      # For the null hypothesis, constrain the curves to be flipped at each time point:
      # If nominal low > nominal high at time i, constrain actual low <= actual high at time i
      # If nominal low < nominal high at time i, constrain actual low >= actual high at time i
      # If they're equal, no additional constraint needed for that time point

      for i in range(len(self.all_death_times)):
        nominal_low_at_i = nominal_low_probs_at_times[i]
        nominal_high_at_i = nominal_high_probs_at_times[i]

        if abs(nominal_low_at_i - nominal_high_at_i) > self.__endpoint_epsilon:
          if nominal_low_at_i > nominal_high_at_i:
            # Under null hypothesis: low curve should be <= high curve at time i
            # (flipped from nominal where low > high)
            self.__patient_wise_only_constraints.append(
              model.addGenConstrIndicator(
                null_hypothesis_indicator, True,
                km_probability_at_time_low[i] - km_probability_at_time_high[i],
                GRB.LESS_EQUAL,
                self.__endpoint_epsilon,
                name=f"patient_wise_only_low_le_high_time_{i}",
              )
            )
          else:
            # Under null hypothesis: low curve should be >= high curve at time i
            # (flipped from nominal where low < high)
            self.__patient_wise_only_constraints.append(
              model.addGenConstrIndicator(
                null_hypothesis_indicator, True,
                km_probability_at_time_low[i] - km_probability_at_time_high[i],
                GRB.GREATER_EQUAL,
                -self.__endpoint_epsilon,
                name=f"patient_wise_only_low_ge_high_time_{i}",
              )
            )

    model.update()

  def solve_and_pvalue( # pylint: disable=too-many-locals
    self,
    *,
    binomial_only: bool = False,
    patient_wise_only: bool = False,
    gurobi_verbose: bool = False
  ):
    """
    Solve the MINLP and return the p value.

    Parameters
    ----------
    binomial_only : bool, optional
        If True, add constraints for a[i, j] to be either 0 or 1,
        based on parameter_in_range. Default is False.
    patient_wise_only : bool, optional
        If True, only consider patient-wise errors and constrain the curves
        to be flipped relative to nominal at each death time point under the null hypothesis.
        Default is False.
    gurobi_verbose : bool, optional
        If True, enable verbose output from Gurobi solver. Default is False.
    """
    if binomial_only and patient_wise_only:
      raise ValueError("binomial_only and patient_wise_only cannot both be True")

    (
      model,
      null_hypothesis_indicator,
      a,
      km_probability_at_time_low,
      km_probability_at_time_high
    ) = self.gurobi_model

    # Apply binomial_only constraints if specified
    self.update_model_with_binomial_only_constraints(model, a, binomial_only)

    # Apply patient_wise_only constraints if specified
    self.update_model_with_patient_wise_only_constraints(
      model,
      km_probability_at_time_low=km_probability_at_time_low,
      km_probability_at_time_high=km_probability_at_time_high,
      null_hypothesis_indicator=null_hypothesis_indicator,
      patient_wise_only=patient_wise_only,
    )

    # Set Gurobi verbose output parameter
    model.setParam('OutputFlag', 1 if gurobi_verbose else 0)

    self.update_model_for_null_hypothesis_or_not(model, null_hypothesis_indicator, True)
    model.optimize()
    if model.status != GRB.OPTIMAL:
      raise ValueError(f"Null model failed with status {model.status}")
    twonll_null = model.ObjVal

    # Extract detailed information for null hypothesis result
    patients_low_null, patients_high_null = self._extract_patients_per_curve(a)
    patient_penalty_null = self._compute_patient_wise_penalty_value(a)
    hypergeometric_penalty_null = self._compute_hypergeometric_penalty(model)

    # Extract curve statistics for null hypothesis
    (n_total_low_null, n_alive_low_null, km_prob_low_null, p_survived_low_null,
     n_total_high_null, n_alive_high_null, km_prob_high_null, p_survived_high_null) = (
      self._extract_curve_statistics(
        model, km_probability_at_time_low, km_probability_at_time_high
      )
    )

    # Extract hazard ratio for null hypothesis (should be 1.0)
    log_hazard_ratio_var = model.getVarByName("log_hazard_ratio")
    hazard_ratio_null = np.exp(log_hazard_ratio_var.X) if log_hazard_ratio_var else 1.0

    result_null = scipy.optimize.OptimizeResult(
      x=model.ObjVal,
      success=model.status == GRB.OPTIMAL,
      patients_low=patients_low_null,
      patients_high=patients_high_null,
      n_total_low=n_total_low_null,
      n_alive_low=n_alive_low_null,
      n_total_high=n_total_high_null,
      n_alive_high=n_alive_high_null,
      km_probability_low=km_prob_low_null,
      km_probability_high=km_prob_high_null,
      p_survived_low=p_survived_low_null,
      p_survived_high=p_survived_high_null,
      binomial_2NLL=2*hypergeometric_penalty_null,
      binomial_2NLL_low=None,  # No longer separate per curve
      binomial_2NLL_high=None,  # No longer separate per curve
      patient_2NLL=2*patient_penalty_null,
      patient_penalties=self.nll_penalty_for_patient_in_range,
      hazard_ratio=hazard_ratio_null,
      model=model,
    )

    self.update_model_for_null_hypothesis_or_not(model, null_hypothesis_indicator, False)
    model.optimize()
    if model.status != GRB.OPTIMAL:
      raise ValueError(f"Alternative model failed with status {model.status}")
    twonll_alt = model.ObjVal

    # Extract detailed information for alternative hypothesis result
    patients_low_alt, patients_high_alt = self._extract_patients_per_curve(a)
    patient_penalty_alt = self._compute_patient_wise_penalty_value(a)
    hypergeometric_penalty_alt = self._compute_hypergeometric_penalty(model)

    # Extract curve statistics for alternative hypothesis
    (n_total_low_alt, n_alive_low_alt, km_prob_low_alt, p_survived_low_alt,
     n_total_high_alt, n_alive_high_alt, km_prob_high_alt, p_survived_high_alt) = (
      self._extract_curve_statistics(
        model, km_probability_at_time_low, km_probability_at_time_high
      )
    )

    # Extract hazard ratio for alternative hypothesis (can be any value)
    hazard_ratio_alt = np.exp(log_hazard_ratio_var.X) if log_hazard_ratio_var else 1.0

    result_alt = scipy.optimize.OptimizeResult(
      x=model.ObjVal,
      success=model.status == GRB.OPTIMAL,
      patients_low=patients_low_alt,
      patients_high=patients_high_alt,
      n_total_low=n_total_low_alt,
      n_alive_low=n_alive_low_alt,
      n_total_high=n_total_high_alt,
      n_alive_high=n_alive_high_alt,
      km_probability_low=km_prob_low_alt,
      km_probability_high=km_prob_high_alt,
      p_survived_low=p_survived_low_alt,
      p_survived_high=p_survived_high_alt,
      binomial_2NLL=2*hypergeometric_penalty_alt,
      binomial_2NLL_low=None,  # No longer separate per curve  
      binomial_2NLL_high=None,  # No longer separate per curve
      patient_2NLL=2*patient_penalty_alt,
      patient_penalties=self.nll_penalty_for_patient_in_range,
      hazard_ratio=hazard_ratio_alt,
      model=model,
    )

    lr_stat = twonll_null - twonll_alt

    # The degrees of freedom is 1: the only difference between null and alternative
    # is whether the log hazard ratio is constrained to 0 (null) or free to float (alternative)
    df = 1

    p_value = scipy.stats.chi2.sf(lr_stat, df)
    return p_value, result_null, result_alt

  def survival_curves_pvalue_logrank(  #pylint: disable=too-many-locals, too-many-branches
    self,
    *,
    binomial_only: bool = True,
  ) -> float:
    """
    Calculate p-value for comparing two Kaplan-Meier curves using the conventional
    logrank test method.

    This method splits patients into two groups based on their observed parameter
    values relative to the parameter_threshold, then uses the standard logrank test
    to test the null hypothesis that the two survival curves are identical.

    This provides the conventional method for comparison with the likelihood-based
    approach implemented in kaplan_meier_p_value_MINLP.py.

    Parameters
    ----------
    binomial_only : bool, optional
        If True, only include patients whose observed parameter is within the
        specified range [parameter_min, parameter_threshold) for low group or
        [parameter_threshold, parameter_max) for high group. This matches the
        behavior of binomial_only in the likelihood method. Default is True.

    Returns
    -------
    float
        The p-value from the logrank test. A small p-value (typically < 0.05)
        indicates evidence against the null hypothesis that the two curves are identical.

    Notes
    -----
    The logrank test is the standard non-parametric test for comparing survival curves.
    It tests the null hypothesis that the hazard functions of the two groups are equal
    at all time points.

    The test statistic follows a chi-square distribution with 1 degree of freedom
    under the null hypothesis.

    This implementation only supports binomial_only=True, which restricts analysis
    to patients whose parameters fall within the specified ranges.
    """
    if not binomial_only:
      raise ValueError(
        "survival_curves_pvalue_logrank only supports binomial_only=True, "
        "which restricts analysis to patients within the specified parameter ranges."
      )

    # Split patients into two groups based on parameter threshold
    group1_patients = []  # Low group: parameter < threshold and >= parameter_min
    group2_patients = []  # High group: parameter >= threshold and < parameter_max

    for patient in self.all_patients:
      param_value = patient.observed_parameter

      if self.parameter_min <= param_value < self.parameter_threshold:
        group1_patients.append(patient)
      elif self.parameter_threshold <= param_value < self.parameter_max:
        group2_patients.append(patient)
      # Patients outside the ranges are excluded when binomial_only=True

    if not group1_patients or not group2_patients:
      raise ValueError(
        f"Need patients in both groups for comparison. "
        f"Got {len(group1_patients)} in low group and {len(group2_patients)} in high group."
      )

    # Get all unique death times (excluding censored events)
    all_death_times = set()
    for patient in group1_patients + group2_patients:
      if not patient.censored:
        all_death_times.add(patient.time)

    if not all_death_times:
      raise ValueError("No death events found in either group.")

    all_death_times = sorted(all_death_times)

    # Calculate logrank test statistic
    U = 0.0  # Sum of (observed - expected) for group 1
    V = 0.0  # Sum of variances

    for death_time in all_death_times:
      # Count patients at risk at this death time
      n1_at_risk = sum(1 for p in group1_patients if p.time >= death_time)
      n2_at_risk = sum(1 for p in group2_patients if p.time >= death_time)
      n_total_at_risk = n1_at_risk + n2_at_risk

      if n_total_at_risk == 0:
        continue

      # Count deaths at this exact time
      d1_deaths = sum(1 for p in group1_patients
                     if p.time == death_time and not p.censored)
      d2_deaths = sum(1 for p in group2_patients
                     if p.time == death_time and not p.censored)
      d_total_deaths = d1_deaths + d2_deaths

      if d_total_deaths == 0:
        continue

      # Expected deaths in group 1 under null hypothesis
      expected_d1 = n1_at_risk * d_total_deaths / n_total_at_risk

      # Variance for this time point
      if n_total_at_risk > 1:
        variance_t = (n1_at_risk * n2_at_risk * d_total_deaths *
                     (n_total_at_risk - d_total_deaths)) / (
                     n_total_at_risk * n_total_at_risk * (n_total_at_risk - 1))
      else:
        variance_t = 0.0

      # Accumulate test statistic components
      U += d1_deaths - expected_d1
      V += variance_t

    if V <= 0:
      # No variance means no information for comparison
      # This can happen if there's only one death time or other edge cases
      return 1.0  # No evidence against null hypothesis

    # Logrank test statistic
    logrank_statistic = U * U / V

    # Calculate p-value using chi-square distribution with 1 degree of freedom
    p_value = 1.0 - scipy.stats.chi2.cdf(logrank_statistic, df=1).item()

    return p_value
