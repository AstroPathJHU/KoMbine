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

from .kaplan_meier_MINLP import KaplanMeierPatientNLL, n_choose_d_term_table
from .utilities import LOG_ZERO_EPSILON_DEFAULT

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
    log_zero_epsilon: float = LOG_ZERO_EPSILON_DEFAULT,
    tie_handling: str = "noties",
  ):
    if tie_handling not in ["noties", "breslow"]:
      raise ValueError(f"tie_handling must be 'noties' or 'breslow', got '{tie_handling}'")

    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_threshold = parameter_threshold
    self.__parameter_max = parameter_max
    self.__log_zero_epsilon = log_zero_epsilon
    self.__tie_handling = tie_handling
    self.__null_hypothesis_constraint = None
    self.__patient_constraints_for_binomial_only = None
    self.__patient_wise_only_constraint = None
    self.__hypergeometric_penalty_constraint = None

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

  @property
  def tie_handling(self) -> str:
    """
    The method used for handling ties in the hypergeometric penalty.
    Can be "noties" (default, assumes no ties) or "breslow" (Breslow approximation).
    """
    return self.__tie_handling

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

  def _add_breslow_nll_terms(  #pylint: disable=too-many-locals,too-many-statements
    self,
    model,
    *,
    omega,
    beta,
    r,
    d,
    d_total,
  ):
    """
    Add NLL terms for Breslow approximation.
    Returns a list of NLL terms.
    """
    nll_terms = []
    max_d_total = self.n_patients

    for j in range(len(self.all_death_times)):
      # affine risk set base: s_j = r0 + omega*r1
      omega_r1 = model.addVar(vtype=gp.GRB.CONTINUOUS,
                              name=f"omega_r1_{j}")
      model.addQConstr(omega_r1 == omega * r[j,1],
                      name=f"omega_r1_prod_{j}")

      s_j_base = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=1e-6,
                             name=f"s_base_{j}")
      model.addConstr(s_j_base == r[j,0] + omega_r1,
                      name=f"s_base_def_{j}")

      # Binary indicators for each possible value of d_total[j]
      d_total_indicators = model.addVars(
        max_d_total + 1,
        vtype=gp.GRB.BINARY,
        name=f"d_total_indicator_{j}"
      )

      # Exactly one d_total indicator must be active
      model.addConstr(
        gp.quicksum(d_total_indicators[k] for k in range(max_d_total + 1)) == 1,
        name=f"d_total_indicator_sum_{j}"
      )

      # Link d_total[j] to the indicators
      model.addConstr(
        d_total[j] == gp.quicksum(k * d_total_indicators[k] for k in range(max_d_total + 1)),
        name=f"d_total_value_{j}"
      )

      # For each possible value of d_total, create the corresponding sum
      breslow_sum_terms = []
      for k in range(max_d_total + 1):
        if k == 0:
          # If d_total = 0, the sum is empty, so contribution is 0
          breslow_sum_terms.append(0)
        else:
          # Sum over m from 0 to k-1
          sum_terms_for_k = []
          for m in range(k):
            # s_j_m = s_j_base - m (with bounds checking)
            s_j_m = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=1e-6,
                                name=f"s_{j}_m{m}_k{k}")
            model.addConstr(s_j_m == s_j_base - m,
                           name=f"s_m_def_{j}_{m}_{k}")

            # Helper variable for log argument (s_j_m + epsilon)
            s_j_m_plus_epsilon = model.addVar(vtype=gp.GRB.CONTINUOUS,
                                             lb=self.__log_zero_epsilon,
                                             name=f"s_plus_epsilon_{j}_{m}_{k}")
            model.addConstr(s_j_m_plus_epsilon == s_j_m + self.__log_zero_epsilon,
                           name=f"s_plus_epsilon_constr_{j}_{m}_{k}")

            # log(s_j_m)
            log_s_j_m = model.addVar(vtype=gp.GRB.CONTINUOUS,
                                    name=f"log_s_{j}_{m}_{k}",
                                    lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
            model.addGenConstrLog(s_j_m_plus_epsilon, log_s_j_m,
                                 name=f"log_s_def_{j}_{m}_{k}")

            sum_terms_for_k.append(log_s_j_m)

          if sum_terms_for_k:
            breslow_sum_terms.append(gp.quicksum(sum_terms_for_k))
          else:
            breslow_sum_terms.append(0)

      # Create the conditional sum based on d_total indicators
      breslow_sum_var = model.addVar(vtype=gp.GRB.CONTINUOUS,
                                    name=f"breslow_sum_{j}",
                                    lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)

      # Use big-M constraints to implement the conditional sum
      big_M = 1000.0  # Large enough constant
      for k in range(max_d_total + 1):
        if isinstance(breslow_sum_terms[k], (int, float)) and breslow_sum_terms[k] == 0:
          # If the term is 0, constrain breslow_sum_var to be 0 when this indicator is active
          model.addGenConstrIndicator(
            d_total_indicators[k], True,
            breslow_sum_var, gp.GRB.EQUAL, 0.0,
            name=f"breslow_sum_zero_{j}_{k}"
          )
        else:
          # Use big-M constraints for non-zero terms
          model.addConstr(
            breslow_sum_var <= breslow_sum_terms[k] + big_M * (1 - d_total_indicators[k]),
            name=f"breslow_sum_upper_{j}_{k}"
          )
          model.addConstr(
            breslow_sum_var >= breslow_sum_terms[k] - big_M * (1 - d_total_indicators[k]),
            name=f"breslow_sum_lower_{j}_{k}"
          )

      # NLL contribution: -d1*beta + breslow_sum
      term = model.addVar(vtype=gp.GRB.CONTINUOUS,
                          name=f"nll_term_{j}")
      model.addConstr(term == -d[j,1]*beta + breslow_sum_var,
                      name=f"nll_term_def_{j}")

      nll_terms.append(term)

    return nll_terms

  def _add_noties_nll_terms(self, model, *, omega, beta, r, d, d_total):
    """
    Add NLL terms for no-ties approximation.
    Returns a list of NLL terms.
    """
    nll_terms = []

    for j in range(len(self.all_death_times)):
      # affine risk set: s_j = r0 + omega*r1
      omega_r1 = model.addVar(vtype=gp.GRB.CONTINUOUS,
                              name=f"omega_r1_{j}")
      model.addQConstr(omega_r1 == omega * r[j,1],
                      name=f"omega_r1_prod_{j}")

      s_j = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=1e-6,
                        name=f"s_{j}")
      model.addConstr(s_j == r[j,0] + omega_r1,
                      name=f"s_def_{j}")

      # Helper variable for log argument (s_j + epsilon)
      s_j_plus_epsilon = model.addVar(vtype=gp.GRB.CONTINUOUS,
                                     lb=self.__log_zero_epsilon,
                                     name=f"s_plus_epsilon_{j}")
      model.addConstr(s_j_plus_epsilon == s_j + self.__log_zero_epsilon,
                      name=f"s_plus_epsilon_constr_{j}")

      # log(s_j)
      log_s_j = model.addVar(vtype=gp.GRB.CONTINUOUS,
                            name=f"log_s_{j}",
                            lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
      model.addGenConstrLog(s_j_plus_epsilon, log_s_j, name=f"log_s_def_{j}")

      # NLL contribution: -d1*beta + d_total*log(s_j)
      term = model.addVar(vtype=gp.GRB.CONTINUOUS,
                          name=f"nll_term_{j}")
      model.addConstr(term == -d[j,1]*beta + d_total[j]*log_s_j,
                      name=f"nll_term_def_{j}")

      nll_terms.append(term)

    return nll_terms

  def add_hypergeometric_penalty_with_hazard_ratio(  #pylint: disable=too-many-locals
    self,
    model: gp.Model,
    *,
    r: gp.tupledict[tuple[int, ...], gp.Var],
    d: gp.tupledict[tuple[int, ...], gp.Var],
  ):
    """
    Add Cox partial likelihood–style penalty using exact exponential form.

    Under null (HR = 1): beta = 0, omega = exp(beta) = 1
    Under alternative: beta free in bounds, omega = exp(beta)

    For "noties" tie handling (current implementation):
      NLL_j = -d[j,1] * beta + d_total[j] * log(r[j,0] + omega * r[j,1])

    For "breslow" tie handling (Breslow approximation):
      NLL_j = -d[j,1] * beta + sum_{m=0}^{d_total[j]-1} log(r[j,0] + omega * r[j,1] - m)
    """
    # total deaths and risk at each time
    d_total = model.addVars(len(self.all_death_times),
                            vtype=gp.GRB.INTEGER,
                            name="d_total")
    r_total = model.addVars(len(self.all_death_times),
                            vtype=gp.GRB.INTEGER,
                            name="r_total")

    for j in range(len(self.all_death_times)):
      model.addConstr(d_total[j] == d[j, 0] + d[j, 1],
                      name=f"d_total_constraint_{j}")
      model.addConstr(r_total[j] == r[j, 0] + r[j, 1],
                      name=f"r_total_constraint_{j}")

    # log hazard ratio (beta) and its exp (omega)
    beta = model.addVar(vtype=gp.GRB.CONTINUOUS,
                        lb=-6.0, ub=6.0,
                        name="log_hazard_ratio")
    omega = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=1e-6,
                        name="hazard_ratio")
    model.addGenConstrExp(beta, omega, name="omega_def")

    # null indicator: if 1 then force beta=0
    null_hypothesis_indicator = model.addVar(vtype=gp.GRB.BINARY,
                                            name="null_hypothesis_indicator")
    model.addGenConstrIndicator(null_hypothesis_indicator, True,
                                beta, gp.GRB.EQUAL, 0.0,
                                name="hazard_ratio_constraint")

    nll_terms = []

    if self.tie_handling == "noties":
      nll_terms = self._add_noties_nll_terms(
        model, omega=omega, beta=beta, r=r, d=d, d_total=d_total
      )
    elif self.tie_handling == "breslow":
      nll_terms = self._add_breslow_nll_terms(
        model, omega=omega, beta=beta, r=r, d=d, d_total=d_total
      )
    else:
      raise ValueError(f"Unknown tie_handling: {self.tie_handling}")

    # Indicator variable to control whether the hypergeometric penalty is used
    use_hypergeometric_penalty_indicator = model.addVar(
      vtype=GRB.BINARY,
      name="use_hypergeometric_penalty_indicator",
    )

    # overall negative log-likelihood penalty
    hypergeometric_penalty_expr = gp.quicksum(nll_terms)
    hypergeometric_penalty = model.addVar(vtype=gp.GRB.CONTINUOUS,
                                          lb=0,
                                          name="hypergeometric_penalty")

    # When use_hypergeometric_penalty_indicator is False, set penalty to 0
    model.addGenConstrIndicator(
      use_hypergeometric_penalty_indicator,
      False,
      hypergeometric_penalty,
      GRB.EQUAL,
      0.0
    )

    # Big M constraint to ensure hypergeometric penalty is only used when indicator is set
    # Calculate a reasonable big M value based on the problem size
    big_M = 1000.0 * len(self.all_death_times)  # Conservative upper bound
    model.addConstr(
      hypergeometric_penalty <= (
        hypergeometric_penalty_expr + big_M * (1 - use_hypergeometric_penalty_indicator)
      ),
      name="hypergeometric_penalty_expr_upper_bound"
    )
    model.addConstr(
      hypergeometric_penalty >= (
        hypergeometric_penalty_expr - big_M * (1 - use_hypergeometric_penalty_indicator)
      ),
      name="hypergeometric_penalty_expr_lower_bound"
    )

    return (
      hypergeometric_penalty,
      null_hypothesis_indicator,
      beta,
      use_hypergeometric_penalty_indicator,
    )

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
        tuple: (n_total_low, n_alive_low, km_prob_low,
                n_total_high, n_alive_high, km_prob_high)
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

    for i in range(len(self.all_death_times)):
      r_low = model.getVarByName(f"r[{i},0]")
      r_high = model.getVarByName(f"r[{i},1]")
      d_low = model.getVarByName(f"d[{i},0]")
      d_high = model.getVarByName(f"d[{i},1]")
      assert r_low is not None
      assert r_high is not None
      assert d_low is not None
      assert d_high is not None

      if i == 0:  # Use first time point as representative
        n_total_low = int(np.rint(r_low.X))
      if i == 0:  # Use first time point as representative
        n_total_high = int(np.rint(r_high.X))

      n_alive_low = n_total_low - int(np.rint(d_low.X))
      n_alive_high = n_total_high - int(np.rint(d_high.X))

    return (
      n_total_low, n_alive_low, km_prob_low,
      n_total_high, n_alive_high, km_prob_high,
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
      null_hypothesis_indicator,
      beta,
      use_hypergeometric_penalty_indicator,
    ) = self.add_hypergeometric_penalty_with_hazard_ratio(
      model,
      d=d,
      r=r,
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
      km_probability_at_time_high,
      beta,
      use_hypergeometric_penalty_indicator,
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

  @functools.cached_property
  def nominal_hazard_ratio(self):
    """
    Returns the nominal hazard ratio.
    This is calculated by running the binomial only model under the alternate hypothesis.
    """
    _, _, result_alt = self.solve_and_pvalue(
      binomial_only=True,
      patient_wise_only=False
    )
    return result_alt.hazard_ratio

  def update_model_with_patient_wise_only_constraint( #pylint: disable=too-many-arguments
    self,
    model: gp.Model,
    *,
    beta: gp.Var,
    null_hypothesis_indicator: gp.Var,
    patient_wise_only: bool,
    use_hypergeometric_penalty_indicator: gp.Var,
  ):
    """
    Update the model with patient_wise_only constraint.
    When patient_wise_only=True, we constrain the hazard to be flipped
    relative to the nominal under the null hypothesis, and disable the
    hypergeometric penalty.
    """
    # Remove existing constraints if they exist
    if self.__patient_wise_only_constraint is not None:
      model.remove(self.__patient_wise_only_constraint)
      self.__patient_wise_only_constraint = None

    if self.__hypergeometric_penalty_constraint is not None:
      model.remove(self.__hypergeometric_penalty_constraint)
      self.__hypergeometric_penalty_constraint = None

    if patient_wise_only:
      # Disable hypergeometric penalty when patient_wise_only is True
      self.__hypergeometric_penalty_constraint = model.addConstr(
        use_hypergeometric_penalty_indicator == 0,
        name="disable_hypergeometric_penalty_patient_wise_only"
      )

      self.__patient_wise_only_constraint = []

      # Get nominal hazard ratio
      nominal_log_hazard_ratio = np.log(self.nominal_hazard_ratio)

      # For the null hypothesis, constrain the hazard ratio to be flipped:
      # If nominal < 1, constraint actual >= 1
      # If nominal > 1, constraint actual <= 1
      # If nominal = 1, no additional constraint needed

      if nominal_log_hazard_ratio > 0:
        self.__patient_wise_only_constraint = model.addGenConstrIndicator(
          null_hypothesis_indicator, True,
          beta,
          GRB.LESS_EQUAL,
          0,
          name="patient_wise_only_hazard_ratio_le_1",
        )
      elif nominal_log_hazard_ratio < 0:
        self.__patient_wise_only_constraint = model.addGenConstrIndicator(
          null_hypothesis_indicator, True,
          beta,
          GRB.GREATER_EQUAL,
          0,
          name="patient_wise_only_hazard_ratio_ge_1",
        )

        raise NotImplementedError(
          "p value for patient-wise only is not implemented. "
          "The challenge is to compute the hazard ratio as a function "
          "of the patients included and excluded. "
          "Naively, this would require a sub-problem with binomial_only=True for each combination "
          "of included and excluded patients (similar to self.nominal_hazard_ratio). "
          "There may be a more clever way to do this within a single Gurobi model, but I will "
          "leave that to a later release."
        )
    else:
      # Enable hypergeometric penalty when patient_wise_only is False
      self.__hypergeometric_penalty_constraint = model.addConstr(
        use_hypergeometric_penalty_indicator == 1,
        name="enable_hypergeometric_penalty"
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

    if patient_wise_only:
      #make sure the nominal hazard ratio is cached before doing anything with the Gurobi model
      #because this causes the model to be updated.
      self.nominal_hazard_ratio # pylint: disable=pointless-statement

    (
      model,
      null_hypothesis_indicator,
      a,
      km_probability_at_time_low,
      km_probability_at_time_high,
      beta,
      use_hypergeometric_penalty_indicator,
    ) = self.gurobi_model

    # Apply binomial_only constraints if specified
    self.update_model_with_binomial_only_constraints(model, a, binomial_only)

    # Apply patient_wise_only constraints if specified
    self.update_model_with_patient_wise_only_constraint(
      model,
      beta=beta,
      null_hypothesis_indicator=null_hypothesis_indicator,
      patient_wise_only=patient_wise_only,
      use_hypergeometric_penalty_indicator=use_hypergeometric_penalty_indicator,
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
    (n_total_low_null, n_alive_low_null, km_prob_low_null,
     n_total_high_null, n_alive_high_null, km_prob_high_null) = (
      self._extract_curve_statistics(
        model, km_probability_at_time_low, km_probability_at_time_high
      )
    )

    # Extract hazard ratio for null hypothesis (should be 1.0)
    log_hazard_ratio_var = model.getVarByName("log_hazard_ratio")
    assert log_hazard_ratio_var is not None
    hazard_ratio_null = np.exp(log_hazard_ratio_var.X)

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
      hypergeometric_2NLL=2*hypergeometric_penalty_null,
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
    (n_total_low_alt, n_alive_low_alt, km_prob_low_alt,
     n_total_high_alt, n_alive_high_alt, km_prob_high_alt) = (
      self._extract_curve_statistics(
        model, km_probability_at_time_low, km_probability_at_time_high
      )
    )

    # Extract hazard ratio for alternative hypothesis (can be any value)
    hazard_ratio_alt = np.exp(log_hazard_ratio_var.X)

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
      hypergeometric_2NLL=2*hypergeometric_penalty_alt,
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
