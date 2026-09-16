"""
MINLP solver for calculating p-values for two Kaplan-Meier curves.
The null hypothesis is that the survival curves are identical.
This follows the structure from kaplan_meier_MINLP.py.
The p-value is computed via the likelihood ratio test.
"""
# pylint: disable=too-many-lines

import functools
import os
import datetime

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.stats

from .kaplan_meier_MINLP import (
  KaplanMeierPatientNLL,
  n_choose_d_term_table,
  km_survival_from_risk_counts,
)
from .utilities import (
  LOG_ZERO_EPSILON_DEFAULT,
  GurobiOptimizerMixin,
)


class MINLPforKMPValue(GurobiOptimizerMixin):  #pylint: disable=too-many-public-methods, too-many-instance-attributes
  """
  MINLP solver for calculating p-values for two Kaplan-Meier curves.
  """

  __default_MIPGap = 1e-6
  __default_MIPGapAbs = 1e-8

  def __init__( # pylint: disable=too-many-arguments
    self,
    all_patients: list[KaplanMeierPatientNLL],
    *,
    parameter_min: float = -np.inf,
    parameter_threshold: float,
    parameter_max: float = np.inf,
    log_zero_epsilon: float = LOG_ZERO_EPSILON_DEFAULT,
    tie_handling: str = "breslow",
    log_hazard_ratio_bounds: tuple[float, float] = (-10.0, 10.0),
  ):
    if tie_handling not in ["breslow"]:
      raise ValueError(f"tie_handling must be 'breslow', got '{tie_handling}'")

    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_threshold = parameter_threshold
    self.__parameter_max = parameter_max
    self.__log_zero_epsilon = log_zero_epsilon
    self.__tie_handling = tie_handling
    self.__log_hazard_ratio_bounds = log_hazard_ratio_bounds
    self.__null_hypothesis_constraint = None
    self.__patient_wise_only_constraint = None
    self.__cox_penalty_constraint = None
    self.__risk_set_r_vars = None
    self.__risk_set_d_vars = None

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
  def log_zero_epsilon(self) -> float:
    """
    Small epsilon used to avoid log(0) in the Gurobi model.
    """
    return self.__log_zero_epsilon

  @property
  def log_hazard_ratio_bounds(self) -> tuple[float, float]:
    """
    The bounds on log(hazard ratio) for the Gurobi model.
    These correspond to hazard ratio bounds of (exp(lb), exp(ub)).
    Default is (-10.0, 10.0), allowing HR in [0.000045, 22026].
    """
    return self.__log_hazard_ratio_bounds

  @property
  def tie_handling(self) -> str:
    """
    The method used for handling ties in the Cox penalty.
    Currently the only option is "breslow" (Breslow approximation).
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
  def assignment_nlls(self) -> npt.NDArray[np.float64]:
    """
    Absolute min NLL for each patient on low, high, and neither.

    Returns an n x 3 array. An infinite bound makes that tail empty
    (min NLL +inf). When both bounds are infinite, column 2 is +inf
    and every patient must be assigned to a group.
    """
    nlls = np.empty((self.n_patients, 3), dtype=float)
    for i, patient in enumerate(self.all_patients):
      nlls[i, 0] = patient.min_nll_on_interval(
        self.parameter_min, self.parameter_threshold,
      )
      nlls[i, 1] = patient.min_nll_on_interval(
        self.parameter_threshold, self.parameter_max,
      )
      nlls[i, 2] = min(
        patient.min_nll_on_interval(-np.inf, self.parameter_min),
        patient.min_nll_on_interval(self.parameter_max, np.inf),
      )
    return nlls

  @functools.cached_property
  def nll_penalty_for_patient_in_range(self) -> npt.NDArray[np.float64]:
    """
    Relative NLL for assigning each patient to the low and high ranges.

    Each column is min NLL on that range minus the best of low, high,
    and neither. Returns an n x 2 array of non-negative values.
    """
    best = np.min(self.assignment_nlls, axis=1, keepdims=True)
    return self.assignment_nlls[:, :2] - best

  @functools.cached_property
  def nll_penalty_for_unassigned(self) -> npt.NDArray[np.float64]:
    """
    Relative NLL for leaving each patient in neither group.

    This is min NLL on the tails minus the best of low, high, and neither.
    """
    best = np.min(self.assignment_nlls, axis=1)
    return self.assignment_nlls[:, 2] - best

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

  def _refresh_death_incidence_constraints(
    self,
    model: gp.Model,
    a: gp.tupledict[tuple[int, ...], gp.Var],
    r: gp.tupledict[tuple[int, ...], gp.Var],
    d: gp.tupledict[tuple[int, ...], gp.Var],
  ) -> None:
    """
    Rebuild r/d incidence constraints after (time, censored) change.

    Death-time grid size is invariant under outcome permutation; only which
    patients contribute to each death/risk-set row changes.
    """
    death_times = self.all_death_times
    for k, t in enumerate(death_times):
      at_risk = self.patient_still_at_risk(t)
      for j in range(2):
        for name in (f"r_{k}_{j}", f"d_{k}_{j}"):
          constr = model.getConstrByName(name)
          if constr is not None:
            model.remove(constr)
        model.addConstr(
          r[k, j] == gp.quicksum(
            a[i, j] for i in range(self.n_patients) if at_risk[i]
          ),
          name=f"r_{k}_{j}",
        )
        model.addConstr(
          d[k, j] == gp.quicksum(
            a[i, j] for i in range(self.n_patients)
            if self.all_patients[i].time == t
            and not self.all_patients[i].censored
          ),
          name=f"d_{k}_{j}",
        )
    model.update()

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

  def _add_breslow_nll_terms(  #pylint: disable=too-many-arguments
    self,
    model: gp.Model,
    *,
    omega: gp.Var,
    beta: gp.Var,
    r: gp.tupledict[tuple[int, ...], gp.Var],
    d: gp.tupledict[tuple[int, ...], gp.Var],
    d_total: gp.tupledict[int, gp.Var],
  ):
    """
    Add NLL terms for Breslow approximation.
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

      # NLL contribution: -d_{i1}*beta + d_total[i]*log(s_i)
      term = model.addVar(vtype=gp.GRB.CONTINUOUS,
                          name=f"nll_term_{j}")
      model.addConstr(term == -d[j,1]*beta + d_total[j]*log_s_j,
                      name=f"nll_term_def_{j}")

      nll_terms.append(term)

    return nll_terms

  def add_cox_penalty_with_hazard_ratio(  #pylint: disable=too-many-locals
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

    For "breslow" tie handling (Breslow approximation):
      NLL_i = -d_{i1} * beta + d_total[i] * log(r_{i0} + omega * r_{i1})

    Note: In the implementation, time index i corresponds to loop variable j,
    and r[j,0] corresponds to r_{i0}, r[j,1] to r_{i1}, etc.
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
    # Use configurable bounds (default -10.0 to 10.0)
    log_hr_lb, log_hr_ub = self.log_hazard_ratio_bounds
    beta = model.addVar(vtype=gp.GRB.CONTINUOUS,
                        lb=log_hr_lb, ub=log_hr_ub,
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

    if self.tie_handling == "breslow":
      nll_terms = self._add_breslow_nll_terms(
        model, omega=omega, beta=beta, r=r, d=d, d_total=d_total
      )
    else:
      raise ValueError(f"Unknown tie_handling: {self.tie_handling}")

    # Indicator variable to control whether the Cox penalty is used
    use_cox_penalty_indicator = model.addVar(
      vtype=GRB.BINARY,
      name="use_cox_penalty_indicator",
    )

    # overall negative log-likelihood penalty
    cox_penalty_expr = gp.quicksum(nll_terms)
    cox_penalty = model.addVar(vtype=gp.GRB.CONTINUOUS,
                                          lb=0,
                                          name="cox_penalty")

    # When use_cox_penalty_indicator is False, set penalty to 0
    model.addGenConstrIndicator(
      use_cox_penalty_indicator,
      False,
      cox_penalty,
      GRB.EQUAL,
      0.0
    )

    model.addGenConstrIndicator(
      use_cox_penalty_indicator,
      True,
      cox_penalty - cox_penalty_expr,
      GRB.EQUAL,
      0.0
    )

    return (
      cox_penalty,
      null_hypothesis_indicator,
      beta,
      use_cox_penalty_indicator,
    )

  def add_patient_wise_penalty(
    self,
    model: gp.Model,
    a: gp.tupledict[tuple[int, ...], gp.Var],
  ):
    """
    Add the patient-wise measurement NLL to the model.

    Each patient pays the relative min NLL of the chosen bin (low, high,
    or neither). The best bin is 0. A group-to-group flip costs one
    measurement NLL difference, not two. Exclusion uses the tail NLL,
    not the cost of flipping into the other group.
    """
    patient_penalties = []
    rel_range = self.nll_penalty_for_patient_in_range
    rel_neither = self.nll_penalty_for_unassigned
    for i in range(self.n_patients):
      for j in range(2):
        penalty_ij = rel_range[i, j]
        if np.isfinite(penalty_ij):
          patient_penalties.append(penalty_ij * a[i, j])
        elif np.isposinf(penalty_ij):
          model.addConstr(
            a[i, j] == 0, name=f"patient_{i}_must_not_be_in_curve_{j}",
          )
        else:
          raise ValueError(
            f"Invalid negative log-likelihood penalty value for patient {i}, curve {j}:"
            f"{penalty_ij}"
          )
      neither_i = rel_neither[i]
      if np.isfinite(neither_i):
        patient_penalties.append(neither_i * (1 - a[i, 0] - a[i, 1]))
      elif np.isposinf(neither_i):
        model.addConstr(
          a[i, 0] + a[i, 1] == 1,
          name=f"patient_{i}_must_be_assigned",
        )
      else:
        raise ValueError(
          f"Invalid unassigned NLL penalty for patient {i}: {neither_i}"
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

  def _extract_curve_statistics(self, model: gp.Model):  # pylint: disable=too-many-locals
    """
    Extract statistics for each curve from the optimized model.

    Returns:
        tuple: (n_total_low, n_alive_low, km_prob_low,
                n_total_high, n_alive_high, km_prob_high)
    """
    n_times = len(self.all_death_times)
    r_low = []
    r_high = []
    s_low = []
    s_high = []
    n_total_low = 0
    n_alive_low = 0
    n_total_high = 0
    n_alive_high = 0
    for i in range(n_times):
      r0 = model.getVarByName(f"r[{i},0]")
      r1 = model.getVarByName(f"r[{i},1]")
      d0 = model.getVarByName(f"d[{i},0]")
      d1 = model.getVarByName(f"d[{i},1]")
      s0 = model.getVarByName(f"n_survived[{i},0]")
      s1 = model.getVarByName(f"n_survived[{i},1]")
      assert r0 is not None and r1 is not None
      assert d0 is not None and d1 is not None
      assert s0 is not None and s1 is not None
      r_low.append(r0.X)
      r_high.append(r1.X)
      s_low.append(s0.X)
      s_high.append(s1.X)
      if i == 0:
        n_total_low = int(np.rint(r0.X))
        n_total_high = int(np.rint(r1.X))
      n_alive_low = n_total_low - int(np.rint(d0.X))
      n_alive_high = n_total_high - int(np.rint(d1.X))

    km_prob_low = km_survival_from_risk_counts(
      r_low, s_low,
      log_zero_epsilon=self.log_zero_epsilon,
      cumulative=True,
    )
    km_prob_high = km_survival_from_risk_counts(
      r_high, s_high,
      log_zero_epsilon=self.log_zero_epsilon,
      cumulative=True,
    )
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
    rel_range = self.nll_penalty_for_patient_in_range
    rel_neither = self.nll_penalty_for_unassigned
    for i in range(self.n_patients):
      assigned_low = a[i, 0].X
      assigned_high = a[i, 1].X
      if np.isfinite(rel_range[i, 0]):
        penalty += rel_range[i, 0] * assigned_low
      if np.isfinite(rel_range[i, 1]):
        penalty += rel_range[i, 1] * assigned_high
      if np.isfinite(rel_neither[i]):
        penalty += rel_neither[i] * (1.0 - assigned_low - assigned_high)
    return penalty

  def _compute_cox_penalty(self, model: gp.Model):
    """
    Compute the Cox penalty.

    Returns:
        float: The Cox penalty value
    """
    cox_penalty_var = model.getVarByName("cox_penalty")

    assert cox_penalty_var is not None

    penalty = cox_penalty_var.X

    return penalty

  def _make_gurobi_model(self):
    """
    Create the Gurobi model for the Kaplan-Meier p-value MINLP.
    """
    model = gp.Model("Kaplan-Meier p-value MINLP")

    #Binary decision variables: a[j, k] = 1 if patient j is in curve k
    a = model.addVars(self.n_patients, 2, vtype=gp.GRB.BINARY, name="a")

    r, d, _n_survived = self.add_counter_variables_and_constraints(model, a)

    (
      cox_penalty,
      null_hypothesis_indicator,
      beta,
      use_cox_penalty_indicator,
    ) = self.add_cox_penalty_with_hazard_ratio(
      model,
      d=d,
      r=r,
    )
    patient_penalty = self.add_patient_wise_penalty(model, a)

    model.setObjective(
      2 * (cox_penalty + patient_penalty),
      GRB.MINIMIZE,
    )
    model.update()
    self.__risk_set_r_vars = r
    self.__risk_set_d_vars = d

    return (  # pylint: disable=duplicate-code
      model,
      null_hypothesis_indicator,
      a,
      beta,
      use_cox_penalty_indicator,
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

  def update_model_with_cox_only_constraints(
    self,
    model: gp.Model,
    a: gp.tupledict[tuple[int, ...], gp.Var],
    cox_only: bool,
  ):
    """
    Fix or unfix assignment variables for cox_only mode.

    When cox_only is True, assignments are locked to the nominal in-range
    groups via bounds so presolve can drop the assignment MIP.
    """
    for i in range(self.n_patients):
      for j in range(2):
        if cox_only:
          value = 1.0 if self.parameter_in_range[i, j] else 0.0
          a[i, j].LB = value
          a[i, j].UB = value
        else:
          a[i, j].LB = 0.0
          a[i, j].UB = 1.0

    model.update()

  @functools.cached_property
  def nominal_hazard_ratio(self):
    """
    Returns the nominal hazard ratio.
    This is calculated by running the cox only model under the alternate hypothesis.
    """
    _, _, result_alt = self.solve_and_pvalue(
      cox_only=True,
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
    use_cox_penalty_indicator: gp.Var,
  ):
    """
    Update the model with patient_wise_only constraint.
    When patient_wise_only=True, we constrain the hazard to be flipped
    relative to the nominal under the null hypothesis, and disable the
    Cox penalty.
    """
    # Remove existing constraints if they exist
    if self.__patient_wise_only_constraint is not None:
      model.remove(self.__patient_wise_only_constraint)
      self.__patient_wise_only_constraint = None

    if self.__cox_penalty_constraint is not None:
      model.remove(self.__cox_penalty_constraint)
      self.__cox_penalty_constraint = None

    if patient_wise_only:
      # Disable Cox penalty when patient_wise_only is True
      self.__cox_penalty_constraint = model.addConstr(
        use_cox_penalty_indicator == 0,
        name="disable_cox_penalty_patient_wise_only"
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
          "Naively, this would require a sub-problem with cox_only=True for each combination "
          "of included and excluded patients (similar to self.nominal_hazard_ratio). "
          "There may be a more clever way to do this within a single Gurobi model, but I will "
          "leave that to a later release."
        )
    else:
      # Enable Cox penalty when patient_wise_only is False
      self.__cox_penalty_constraint = model.addConstr(
        use_cox_penalty_indicator == 1,
        name="enable_cox_penalty"
      )

    model.update()

  def _get_risk_set_vars(self):
    """Return cached r/d tupledicts from the current Gurobi model, if any."""
    return self.__risk_set_r_vars, self.__risk_set_d_vars

  def _dispose_gurobi_model(self) -> None:
    """Dispose a cached Gurobi model and drop outcome-tied constraint handles."""
    cached = self.__dict__.get("gurobi_model")
    if cached is not None:
      try:
        cached[0].dispose()
      except gp.GurobiError:
        pass
      self.__dict__.pop("gurobi_model", None)
    self.__null_hypothesis_constraint = None
    self.__patient_wise_only_constraint = None
    self.__cox_penalty_constraint = None
    self.__risk_set_r_vars = None
    self.__risk_set_d_vars = None

  def _invalidate_outcome_dependent_state(self, *, keep_model: bool = False) -> None:
    """
    Clear caches that depend on (time, censored).

    Measurement / assignment-NLL caches are kept: permutations shuffle
    outcomes only. When keep_model is True, the Gurobi model is retained so
    death-incidence constraints can be refreshed in place.
    """
    if keep_model:
      for name in (
        "patient_times",
        "all_death_times",
        "patient_censored",
        "nominal_hazard_ratio",
      ):
        self.__dict__.pop(name, None)
      return
    self._dispose_gurobi_model()
    for name in (
      "patient_times",
      "all_death_times",
      "patient_censored",
      "nominal_hazard_ratio",
    ):
      self.__dict__.pop(name, None)

  def _set_patient_outcomes(
    self,
    times: list[float],
    censored: list[bool],
    *,
    keep_model: bool = False,
  ) -> None:
    """
    Replace each patient's (time, censored), keeping measurement NLLs fixed.
    """
    if len(times) != self.n_patients or len(censored) != self.n_patients:
      raise ValueError(
        "times and censored must have length n_patients "
        f"({self.n_patients}), got {len(times)} and {len(censored)}"
      )
    self.__all_patients = [
      KaplanMeierPatientNLL(
        time=float(times[i]),
        censored=bool(censored[i]),
        parameter_nll=patient.parameter,
        observed_parameter=patient.observed_parameter,
      )
      for i, patient in enumerate(self.all_patients)
    ]
    self._invalidate_outcome_dependent_state(keep_model=keep_model)

  def _extract_optimize_result(
    self,
    model: gp.Model,
    a: gp.tupledict[tuple[int, ...], gp.Var],
  ) -> scipy.optimize.OptimizeResult:
    """
    Pack the current Gurobi solution into an OptimizeResult.
    """
    patients_low, patients_high = self._extract_patients_per_curve(a)
    (
      n_total_low, n_alive_low, km_prob_low,
      n_total_high, n_alive_high, km_prob_high,
    ) = self._extract_curve_statistics(model)
    log_hazard_ratio_var = model.getVarByName("log_hazard_ratio")
    assert log_hazard_ratio_var is not None
    return scipy.optimize.OptimizeResult(
      x=model.ObjVal,
      success=model.status == GRB.OPTIMAL,
      patients_low=patients_low,
      patients_high=patients_high,
      n_total_low=n_total_low,
      n_alive_low=n_alive_low,
      n_total_high=n_total_high,
      n_alive_high=n_alive_high,
      km_probability_low=km_prob_low,
      km_probability_high=km_prob_high,
      cox_2NLL=2 * self._compute_cox_penalty(model),
      patient_2NLL=2 * self._compute_patient_wise_penalty_value(a),
      patient_penalties=self.nll_penalty_for_patient_in_range,
      hazard_ratio=np.exp(log_hazard_ratio_var.X),
      model=model,
    )

  def _fit_null_and_alternative(  # pylint: disable=too-many-arguments, too-many-locals
    self,
    *,
    cox_only: bool,
    patient_wise_only: bool,
    verbose: bool,
    print_progress: bool,
    solve_null: bool,
    MIPGap: float,
    MIPGapAbs: float,
    TimeLimit: float | None,
    Threads: int | None,
    MIPFocus: int | None,
    LogFile: os.PathLike | None,
  ) -> tuple[
    scipy.optimize.OptimizeResult | None,
    scipy.optimize.OptimizeResult,
  ]:
    """
    Fit H0 and/or H1 on this calculator's observed (or permuted) data.
    """
    (
      model,
      null_hypothesis_indicator,
      a,
      beta,
      use_cox_penalty_indicator,
    ) = self.gurobi_model

    self.update_model_with_cox_only_constraints(model, a, cox_only)
    self.update_model_with_patient_wise_only_constraint(
      model,
      beta=beta,
      null_hypothesis_indicator=null_hypothesis_indicator,
      patient_wise_only=patient_wise_only,
      use_cox_penalty_indicator=use_cox_penalty_indicator,
    )

    solver_kwargs = {
      "verbose": verbose,
      "MIPGap": MIPGap,
      "MIPGapAbs": MIPGapAbs,
      "TimeLimit": TimeLimit,
      "Threads": Threads,
      "MIPFocus": MIPFocus,
      "LogFile": LogFile,
    }

    result_null = None
    start_a = None
    start_beta = None
    if solve_null:
      if print_progress or verbose:
        print(
          f"[{datetime.datetime.now()}] Solving for null hypothesis...",
          flush=True,
        )
      self.update_model_for_null_hypothesis_or_not(
        model, null_hypothesis_indicator, True,
      )
      model = self._setup_and_optimize(model, **solver_kwargs)
      if model.status != GRB.OPTIMAL:
        raise ValueError(f"Null model failed with status {model.status}")
      result_null = self._extract_optimize_result(model, a)
      if not cox_only:
        start_a = {
          (i, j): float(a[i, j].X)
          for i in range(self.n_patients)
          for j in range(2)
        }
        if beta is not None:
          start_beta = float(beta.X)

    if print_progress or verbose:
      print(
        f"[{datetime.datetime.now()}] Solving for alternative hypothesis...",
        flush=True,
      )
    self.update_model_for_null_hypothesis_or_not(
      model, null_hypothesis_indicator, False,
    )
    if start_a is not None:
      for (i, j), value in start_a.items():
        a[i, j].Start = value
      if start_beta is not None:
        beta.Start = start_beta
    model = self._setup_and_optimize(model, **solver_kwargs)
    if model.status != GRB.OPTIMAL:
      raise ValueError(f"Alternative model failed with status {model.status}")
    result_alt = self._extract_optimize_result(model, a)
    return result_null, result_alt

  def solve_and_pvalue( # pylint: disable=too-many-locals, too-many-arguments, too-many-branches
    self,
    *,
    cox_only: bool = False,
    patient_wise_only: bool = False,
    n_permutations: int = 199,
    rng: int | np.random.Generator | None = None,
    verbose: bool = False,
    print_progress: bool = False,
    MIPGap: float | None = None,
    MIPGapAbs: float | None = None,
    TimeLimit: float | None = None,
    Threads: int | None = None,
    MIPFocus: int | None = None,
    LogFile: os.PathLike | None = None,
  ):
    """
    Solve the MINLP and return the p value.

    When cox_only is False, the p-value is a permutation LRT: (time,
    censored) are shuffled across patients while measurements stay fixed,
    so the null has the same assignment freedom as the alternative.
    When cox_only is True, assignments are locked and the reference is
    chi-squared with 1 degree of freedom.

    Parameters
    ----------
    cox_only : bool, optional
        If True, add constraints for a[i, j] to be either 0 or 1,
        based on parameter_in_range. Default is False.
    patient_wise_only : bool, optional
        If True, only consider patient-wise errors and constrain the curves
        to be flipped relative to nominal at each death time point under the null hypothesis.
        Default is False.
    n_permutations : int, optional
        Number of outcome permutations when cox_only is False. Ignored
        when cox_only is True. Default is 199.
    rng : int or numpy.random.Generator or None, optional
        Seed or generator for the permutations. Default is None.
    verbose : bool, optional
        If True, enable verbose output from Gurobi solver. Default is False.
    """
    if print_progress or verbose:
      print(
        f"[{datetime.datetime.now()}] Running p-value MINLP",
        flush=True,
      )
    if cox_only and patient_wise_only:
      raise ValueError("cox_only and patient_wise_only cannot both be True")
    if n_permutations < 0:
      raise ValueError(
        f"n_permutations must be >= 0, got {n_permutations}"
      )

    if patient_wise_only:
      #make sure the nominal hazard ratio is cached before doing anything with the Gurobi model
      #because this causes the model to be updated.
      self.nominal_hazard_ratio # pylint: disable=pointless-statement

    if MIPGap is None:
      MIPGap = self.__default_MIPGap
    if MIPGapAbs is None:
      MIPGapAbs = self.__default_MIPGapAbs

    fit_kwargs = {
      "cox_only": cox_only,
      "patient_wise_only": patient_wise_only,
      "verbose": verbose,
      "print_progress": print_progress,
      "MIPGap": MIPGap,
      "MIPGapAbs": MIPGapAbs,
      "TimeLimit": TimeLimit,
      "Threads": Threads,
      "MIPFocus": MIPFocus,
      "LogFile": LogFile,
    }
    result_null, result_alt = self._fit_null_and_alternative(
      solve_null=True,
      **fit_kwargs,
    )
    assert result_null is not None

    if cox_only:
      lr_stat = result_null.x - result_alt.x
      p_value = scipy.stats.chi2.sf(lr_stat, 1)
      result_alt.n_permutations = None
      result_alt.n_extreme = None
      return p_value, result_null, result_alt

    base_times = [patient.time for patient in self.all_patients]
    base_censored = [patient.censored for patient in self.all_patients]
    perm_calc = type(self)(
      list(self.all_patients),
      parameter_min=self.parameter_min,
      parameter_threshold=self.parameter_threshold,
      parameter_max=self.parameter_max,
      log_zero_epsilon=self.log_zero_epsilon,
      tie_handling=self.tie_handling,
      log_hazard_ratio_bounds=self.log_hazard_ratio_bounds,
    )

    generator = np.random.default_rng(rng)
    n_extreme = 0
    try:
      for i_perm in range(n_permutations):
        if print_progress or verbose:
          print(
            f"[{datetime.datetime.now()}] Permutation "
            f"{i_perm + 1}/{n_permutations}...",
            flush=True,
          )
        order = generator.permutation(self.n_patients)
        reshuffled_times = [base_times[order[i]] for i in range(self.n_patients)]
        reshuffled_censored = [
          bool(base_censored[order[i]]) for i in range(self.n_patients)
        ]
        if i_perm == 0:
          perm_calc._set_patient_outcomes(  # pylint: disable=protected-access
            reshuffled_times,
            reshuffled_censored,
            keep_model=False,
          )
        else:
          perm_calc._set_patient_outcomes(  # pylint: disable=protected-access
            reshuffled_times,
            reshuffled_censored,
            keep_model=True,
          )
          (
            model,
            _null_hypothesis_indicator,
            a,
            _beta,
            _use_cox_penalty_indicator,
          ) = perm_calc.gurobi_model
          r_vars, d_vars = perm_calc._get_risk_set_vars()  # pylint: disable=protected-access
          assert r_vars is not None and d_vars is not None
          perm_calc._refresh_death_incidence_constraints(  # pylint: disable=protected-access
            model,
            a,
            r_vars,
            d_vars,
          )
        _, perm_alt = perm_calc._fit_null_and_alternative(  # pylint: disable=protected-access
          solve_null=False,
          **fit_kwargs,
        )
        # 2NLL_0 is invariant to pairing, so T_perm >= T_obs iff alt is as good.
        if perm_alt.x <= result_alt.x:
          n_extreme += 1
    finally:
      perm_calc._dispose_gurobi_model()  # pylint: disable=protected-access

    p_value = (1.0 + n_extreme) / (1.0 + n_permutations)
    result_alt.n_permutations = n_permutations
    result_alt.n_extreme = n_extreme
    return p_value, result_null, result_alt


  def survival_curves_pvalue_logrank(  #pylint: disable=too-many-locals, too-many-branches
    self,
    *,
    cox_only: bool = True,
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
    cox_only : bool, optional
        If True, only include patients whose observed parameter is within the
        specified range [parameter_min, parameter_threshold) for low group or
        [parameter_threshold, parameter_max) for high group. This matches the
        behavior of cox_only in the likelihood method. Default is True.

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

    This implementation only supports cox_only=True, which restricts analysis
    to patients whose parameters fall within the specified ranges.
    """
    if not cox_only:
      raise ValueError(
        "survival_curves_pvalue_logrank only supports cox_only=True, "
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
      # Patients outside the ranges are excluded when cox_only=True

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
