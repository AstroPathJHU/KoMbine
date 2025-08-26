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
  ):
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_threshold = parameter_threshold
    self.__parameter_max = parameter_max
    self.__endpoint_epsilon = endpoint_epsilon
    self.__log_zero_epsilon = 1e-10
    self.__null_hypothesis_constraint = None
    self.__hazard_ratio_null_constraint = None
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
    x: gp.tupledict[tuple[int, ...], gp.Var]
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
      for i in range(self.n_patients):
        model.addConstr(x[i, 0] + x[i, 1] == 1)
    else:
      for i in range(self.n_patients):
        model.addConstr(x[i, 0] + x[i, 1] <= 1)

    n_at_risk = model.addVars(
      len(self.all_death_times), 2, vtype=gp.GRB.INTEGER, name="n_at_risk"
    )
    n_died = model.addVars(
      len(self.all_death_times), 2, vtype=gp.GRB.INTEGER, name="n_died"
    )
    n_survived = model.addVars(
      len(self.all_death_times), 2, vtype=gp.GRB.INTEGER, name="n_survived"
    )
    for k, t in enumerate(self.all_death_times):
      for j in range(2):
        model.addConstr(
          n_at_risk[k, j] == gp.quicksum(
            x[i, j] for i in range(self.n_patients)
            if self.patient_still_at_risk(t)[i]
          )
        )
        model.addConstr(
          n_died[k, j] == gp.quicksum(
            x[i, j] for i in range(self.n_patients)
            if self.all_patients[i].time == t
            and not self.all_patients[i].censored
          )
        )
        model.addConstr(n_survived[k, j] == n_at_risk[k, j] - n_died[k, j])

    return n_at_risk, n_died, n_survived

  def add_kaplan_meier_probability_variables_and_constraints( #pylint: disable=too-many-locals
    self,
    model: gp.Model,
    n_at_risk: gp.tupledict[tuple[int, ...], gp.Var],
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
      log_n_at_risk_vars = model.addVars(
        len(self.all_death_times),
        vtype=GRB.CONTINUOUS,
        name=f"log_n_at_risk_curve_{j}",
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

      # Helper variables for log arguments (n_at_risk + epsilon, n_survived + epsilon)
      n_at_risk_plus_epsilon = model.addVars(
        len(self.all_death_times),
        vtype=GRB.CONTINUOUS,
        name=f"n_at_risk_plus_epsilon_curve_{j}",
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
          n_at_risk_plus_epsilon[i] == n_at_risk[i, j] + self.__log_zero_epsilon,
          name=f"n_at_risk_plus_epsilon_constr_{i}_curve_{j}"
        )
        model.addConstr(
          n_survived_plus_epsilon[i] == n_survived[i, j] + self.__log_zero_epsilon,
          name=f"n_survived_plus_epsilon_constr_{i}_curve_{j}"
        )

      # Link count variables to their log counterparts using GenConstrLog
      for i in range(len(self.all_death_times)):
        model.addGenConstrLog(
          n_at_risk_plus_epsilon[i],
          log_n_at_risk_vars[i],
          name=f"log_n_at_risk_constr_{i}_curve_{j}"
        )
        model.addGenConstrLog(
          n_survived_plus_epsilon[i],
          log_n_survived_vars[i],
          name=f"log_n_survived_constr_{i}_curve_{j}"
        )

      # Binary indicator for whether n_at_risk for a death time is zero
      is_n_at_risk_zero = model.addVars(
        len(self.all_death_times),
        vtype=GRB.BINARY,
        name=f"is_n_at_risk_zero_curve_{j}"
      )

      # Link is_n_at_risk_zero to n_at_risk using indicator constraint
      for i in range(len(self.all_death_times)):
        model.addGenConstrIndicator(
          is_n_at_risk_zero[i], True, n_at_risk[i, j], GRB.EQUAL, 0,
          name=f"is_n_at_risk_zero_indicator_{i}_curve_{j}"
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
        # If is_n_at_risk_zero[i] is 0 (i.e., n_at_risk[i, j] > 0)
        model.addGenConstrIndicator(
          is_n_at_risk_zero[i], False,
          km_log_probability_per_time_terms[i] - (log_n_survived_vars[i] - log_n_at_risk_vars[i]),
          GRB.EQUAL,
          0,
          name=f"km_log_prob_time_active_{i}_curve_{j}"
        )
        # If is_n_at_risk_zero[i] is 1 (i.e., n_at_risk[i, j] == 0)
        model.addGenConstrIndicator(
          is_n_at_risk_zero[i], True,
          km_log_probability_per_time_terms[i],
          GRB.EQUAL,
          0.0,
          name=f"km_log_prob_time_zero_at_risk_{i}_curve_{j}"
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
            name=f"km_log_prob_at_time_0_curve_{j}"
          )
        else:
          # Subsequent death times: cumulative sum up to this point
          model.addConstr(
            km_log_probability_at_time[i]
              == km_log_probability_at_time[i-1] + km_log_probability_per_time_terms[i],
            name=f"km_log_prob_at_time_{i}_curve_{j}"
          )

        # Convert from log to linear scale
        model.addGenConstrExp(
          km_log_probability_at_time[i],
          km_probability_at_time[i],
          name=f"exp_km_probability_at_time_{i}_curve_{j}"
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
        name=f"km_log_probability_total_def_curve_{j}"
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
        name=f"exp_km_probability_curve_{j}"
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

  def add_binomial_penalty(  # pylint: disable=too-many-locals
    self,
    model: gp.Model,
    *,
    n_at_risk: gp.tupledict[tuple[int, ...], gp.Var],
    n_died: gp.tupledict[tuple[int, ...], gp.Var],
    n_survived: gp.tupledict[tuple[int, ...], gp.Var],
  ):
    """
    Add the binomial penalty to the model.
    """
    p_survived = model.addVars(
      len(self.all_death_times), 2,
      vtype=gp.GRB.CONTINUOUS, name="p_survived", lb=0, ub=1
    )
    p_died = model.addVars(
      len(self.all_death_times), 2,
      vtype=gp.GRB.CONTINUOUS, name="p_died", lb=0, ub=1
    )
    log_p_bounds = np.array([
      np.log(self.__endpoint_epsilon / len(self.all_death_times) / 2),
      np.log(1 - self.__endpoint_epsilon / len(self.all_death_times) / 2),
    ])
    log_p_survived = model.addVars(
      len(self.all_death_times), 2,
      vtype=gp.GRB.CONTINUOUS, name="log_p_survived",
      lb=log_p_bounds[0], ub=log_p_bounds[1]
    )
    # Expand bounds for log_p_died to accommodate hazard ratio constraint
    # log_p_died[i, 0] = log_hazard_ratio + log_p_died[i, 1] where log_hazard_ratio in [-3, 3]
    log_p_died_bounds = np.array([
      log_p_bounds[0] - 3.0,  # Allow for negative hazard ratio  
      log_p_bounds[1] + 3.0,  # Allow for positive hazard ratio
    ])
    log_p_died = model.addVars(
      len(self.all_death_times), 2,
      vtype=gp.GRB.CONTINUOUS, name="log_p_died",
      lb=log_p_died_bounds[0], ub=log_p_died_bounds[1]
    )
    for i in range(len(self.all_death_times)):
      for j in range(2):
        model.addGenConstrExp(
          log_p_survived[i, j], p_survived[i, j], name=f"log_p_survived_constr_{i}_{j}"
        )
        model.addGenConstrExp(
          log_p_died[i, j], p_died[i, j], name=f"log_p_died_constr_{i}_{j}"
        )
        model.addConstr(
          p_survived[i, j] + p_died[i, j] == 1,
          name=f"survived_died_constraint_{i}_{j}"
        )

    n_choose_d_table = self.n_choose_d_term_table
    n_choose_d_indicator_vars = model.addVars(
      len(self.all_death_times), 2, len(n_choose_d_table),
      vtype=gp.GRB.BINARY, name="n_choose_d_indicator",
    )
    n_died_indicator_vars = model.addVars(
      len(self.all_death_times), 2, self.n_patients + 1,
      vtype=gp.GRB.BINARY, name="n_died_indicator"
    )
    n_survived_indicator_vars = model.addVars(
      len(self.all_death_times), 2, self.n_patients + 1,
      vtype=gp.GRB.BINARY, name="n_survived_indicator"
    )
    binomial_terms = []
    for i in range(len(self.all_death_times)):
      for j in range(2):
        #make sure that exactly one of each indicator is selected
        model.addConstr(
          gp.quicksum(
            n_choose_d_indicator_vars[i, j, k]
            for k in range(len(n_choose_d_table))
          ) == 1,
          name=f"n_choose_d_indicator_unique_{i}_{j}",
        )
        model.addConstr(
          gp.quicksum(n_died_indicator_vars[i, j, k] for k in range(self.n_patients + 1)) == 1,
          name=f"n_died_indicator_unique_{i}_{j}",
        )
        model.addConstr(
          gp.quicksum(n_survived_indicator_vars[i, j, k] for k in range(self.n_patients + 1)) == 1,
          name=f"n_survived_indicator_unique_{i}_{j}",
        )

        #Add the n choose d term
        for k, ((n, d), penalty) in enumerate(n_choose_d_table.items()):
          indicator = n_choose_d_indicator_vars[i, j, k]
          binomial_terms.append(-penalty * indicator)
          model.addGenConstrIndicator(
            indicator,
            True,
            n_at_risk[i,j],
            GRB.EQUAL,
            n,
            name=f"n_choose_d_indicator_n_{i}_{j}_{n}"
          )
          model.addGenConstrIndicator(
            indicator,
            True,
            n_died[i,j],
            GRB.EQUAL,
            d,
            name=f"n_choose_d_indicator_d_{i}_{j}_{d}"
          )

        #Add the p_died term
        for d in range(self.n_patients + 1):
          binomial_terms.append(
            -d * log_p_died[i, j] * n_died_indicator_vars[i, j, d]
          )
          model.addGenConstrIndicator(
            n_died_indicator_vars[i, j, d],
            True,
            n_died[i, j],
            GRB.EQUAL,
            d,
            name=f"n_died_indicator_{i}_{j}_{d}"
          )

        #Add the p_survived term
        for s in range(self.n_patients + 1):
          binomial_terms.append(
            -s * log_p_survived[i, j] * n_survived_indicator_vars[i, j, s]
          )
          model.addGenConstrIndicator(
            n_survived_indicator_vars[i, j, s],
            True,
            n_survived[i, j],
            GRB.EQUAL,
            s,
            name=f"n_survived_indicator_{i}_{j}_{s}"
          )
    binomial_penalty = gp.quicksum(binomial_terms)

    # Add log hazard ratio variable for constant hazard ratio constraint
    # Range should allow for reasonable hazard ratios (e.g., 0.05 to 20)
    # log(0.05) ≈ -3.0, log(20) ≈ 3.0
    log_hazard_ratio = model.addVar(
      vtype=gp.GRB.CONTINUOUS,
      name="log_hazard_ratio",
      lb=-3.0,
      ub=3.0,
    )

    # Constant hazard ratio constraint: applies unconditionally 
    # (both under null and alternative hypothesis)
    # This constrains: p_died[i, 0] = hazard_ratio * p_died[i, 1]  
    # In log scale: log_p_died[i, 0] = log_hazard_ratio + log_p_died[i, 1]
    for i in range(len(self.all_death_times)):
      model.addConstr(
        log_p_died[i, 0] - log_p_died[i, 1] - log_hazard_ratio == 0,
        name=f"constant_hazard_ratio_{i}"
      )

    # Null hypothesis indicator for constraining log_hazard_ratio = 0
    null_hypothesis_indicator = model.addVar(vtype=gp.GRB.BINARY, name="null_hypothesis_indicator")

    return binomial_penalty, null_hypothesis_indicator

  def add_patient_wise_penalty(
    self,
    model: gp.Model,
    x: gp.tupledict[tuple[int, ...], gp.Var],
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
          penalty = self.nll_penalty_for_patient_in_range[i, j] * x[i, j]
          if self.nll_penalty_for_patient_in_range[i, j] < 0:
            # If the penalty is negative, it means the patient is nominally in the range
            # We want the penalty to be 0 when all patients are at their nominal values
            penalty -= self.nll_penalty_for_patient_in_range[i, j]
          patient_penalties.append(penalty)
        elif np.isneginf(self.nll_penalty_for_patient_in_range[i, j]):
          #the patient must be selected, so we add a constraint
          model.addConstr(
            x[i, j] == 1, name=f"patient_{i}_must_be_in_curve_{j}"
          )
        elif np.isposinf(self.nll_penalty_for_patient_in_range[i, j]):
          # The patient must not be selected, so we add a constraint
          model.addConstr(
            x[i, j] == 0, name=f"patient_{i}_must_not_be_in_curve_{j}"
          )
        else:
          raise ValueError(
            f"Invalid negative log-likelihood penalty value for patient {i}, curve {j}:"
            f"{self.nll_penalty_for_patient_in_range[i, j]}"
          )
    patient_penalty = gp.quicksum(patient_penalties)
    return patient_penalty

  def _make_gurobi_model(self):
    """
    Create the Gurobi model for the Kaplan-Meier p-value MINLP.
    """
    model = gp.Model("Kaplan-Meier p-value MINLP")

    #Binary decision variables: x[i, j] = 1 if patient i is in curve j
    x = model.addVars(self.n_patients, 2, vtype=gp.GRB.BINARY, name="x")

    n_at_risk, n_died, n_survived = self.add_counter_variables_and_constraints(model, x)

    # Add Kaplan-Meier probability variables and constraints
    (
      km_probability_at_time_low,
      km_probability_at_time_high
    ) = self.add_kaplan_meier_probability_variables_and_constraints(
      model, n_at_risk, n_survived
    )

    binomial_penalty, null_hypothesis_indicator = self.add_binomial_penalty(
      model,
      n_died=n_died,
      n_at_risk=n_at_risk,
      n_survived=n_survived
    )
    patient_penalty = self.add_patient_wise_penalty(model, x)

    model.setObjective(
      2 * (binomial_penalty + patient_penalty),
      GRB.MINIMIZE,
    )
    model.update()

    return (
      model,
      null_hypothesis_indicator,
      log_hazard_ratio,
      x,
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
    log_hazard_ratio,
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
    if self.__hazard_ratio_null_constraint is not None:
      model.remove(self.__hazard_ratio_null_constraint)

    if null_hypothesis:
      self.__null_hypothesis_constraint = model.addConstr(
        null_hypothesis_indicator == 1,
        name="null_hypothesis_constraint"
      )
      # Under null hypothesis: fix log_hazard_ratio = 0 (hazard ratio = 1)
      self.__hazard_ratio_null_constraint = model.addConstr(
        log_hazard_ratio == 0,
        name="hazard_ratio_null_constraint"
      )
    else:
      self.__null_hypothesis_constraint = None
      self.__hazard_ratio_null_constraint = None

    model.update()

  def update_model_with_binomial_only_constraints(
    self,
    model: gp.Model,
    x: gp.tupledict[tuple[int, ...], gp.Var],
    binomial_only: bool,
  ):
    """
    Update the model with binomial_only constraints.
    If binomial_only is True, we add constraints for x[i, j] to be either 0 or 1,
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
                x[i, j] == 1,
                name=f"patient_{i}_must_be_selected_curve_{j}_binomial_only",
              )
            )
          else:
            # The patient must not be selected for this curve
            self.__patient_constraints_for_binomial_only.append(
              model.addConstr(
                x[i, j] == 0,
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
                name=f"patient_wise_only_low_le_high_time_{i}"
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
                name=f"patient_wise_only_low_ge_high_time_{i}"
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
        If True, add constraints for x[i, j] to be either 0 or 1,
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
      log_hazard_ratio,
      x,
      km_probability_at_time_low,
      km_probability_at_time_high
    ) = self.gurobi_model

    # Apply binomial_only constraints if specified
    self.update_model_with_binomial_only_constraints(model, x, binomial_only)

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

    self.update_model_for_null_hypothesis_or_not(model, null_hypothesis_indicator, log_hazard_ratio, True)
    model.optimize()
    if model.status != GRB.OPTIMAL:
      raise ValueError("Null model did not converge")
    twonll_null = model.ObjVal
    result_null = scipy.optimize.OptimizeResult(
      x=model.ObjVal,
      success=model.status == GRB.OPTIMAL,
    )

    self.update_model_for_null_hypothesis_or_not(model, null_hypothesis_indicator, log_hazard_ratio, False)
    model.optimize()
    if model.status != GRB.OPTIMAL:
      raise ValueError("Alternative model did not converge")
    twonll_alt = model.ObjVal
    result_alt = scipy.optimize.OptimizeResult(
      x=model.ObjVal,
      success=model.status == GRB.OPTIMAL,
    )

    lr_stat = twonll_null - twonll_alt

    # The degrees of freedom is 1: the only difference between null and alternative
    # is whether the log hazard ratio is constrained to 0 (null) or free to float (alternative)
    if patient_wise_only:
      # For patient_wise_only, we add one constraint per death time where
      # the nominal probabilities differ between curves, plus the hazard ratio constraint
      nominal_low_probs, nominal_high_probs = self.nominal_curves_probabilities_at_each_time
      df_patient_wise = sum(1 for i in range(len(self.all_death_times))
                           if abs(nominal_low_probs[i] - nominal_high_probs[i]) > self.__endpoint_epsilon)
      # However, with the constant hazard ratio assumption, the effective degrees of freedom is still 1
      # because all the death time constraints are linked through the common hazard ratio
      df = 1
    else:
      # With constant hazard ratio assumption, degrees of freedom is 1:
      # the single parameter difference (log_hazard_ratio = 0 vs free)
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
