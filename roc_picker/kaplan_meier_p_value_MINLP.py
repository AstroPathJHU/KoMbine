"""
kaplan_meier_MINLP_p_value.py

MINLP solver for calculating p-values for two Kaplan–Meier curves using a shared-survival-probability null model.
This follows the structure from kaplan_meier_MINLP.py but with x[i] representing group membership (1=group1, 0=group2),
shared p_survived[j] under H0, and separate p_survived under H1. The p-value is computed via the likelihood ratio test.
"""

import functools
import math

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.stats

from .kaplan_meier_MINLP import KaplanMeierPatientNLL

class MINLPforKMPValue:
  def __init__(
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
    self.__null_hypothesis_constraint = None

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

    range_boundary_nll: npt.NDArray[np.float64] = np.array([range_boundary_nll_low, range_boundary_nll_high]).T
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
    # Add counter variables and constraints to the model

    # A patient can't be in more than one curve.
    # If parameter_min and parameter_max are both infinite,
    # then each patient must be assigned to a curve.
    if np.isinf(self.parameter_min) and np.isinf(self.parameter_max):
      for i in range(self.n_patients):
        model.addConstr(x[i, 0] + x[i, 1] == 1)
    else:
      for i in range(self.n_patients):
        model.addConstr(x[i, 0] + x[i, 1] <= 1)

    n_at_risk = model.addVars(len(self.all_death_times), 2, vtype=gp.GRB.INTEGER, name="n_at_risk")
    n_died = model.addVars(len(self.all_death_times), 2, vtype=gp.GRB.INTEGER, name="n_died")
    n_survived = model.addVars(len(self.all_death_times), 2, vtype=gp.GRB.INTEGER, name="n_survived")
    for k, t in enumerate(self.all_death_times):
      for j in range(2):
        model.addConstr(n_at_risk[k, j] == gp.quicksum(x[i, j] for i in range(self.n_patients) if self.patient_still_at_risk(t)[i]))
        model.addConstr(n_died[k, j] == gp.quicksum(x[i, j] for i in range(self.n_patients) if self.all_patients[i].time == t and not self.all_patients[i].censored))
        model.addConstr(n_survived[k, j] == n_at_risk[k, j] - n_died[k, j])

    return n_at_risk, n_died, n_survived

  @functools.cached_property
  def n_choose_d_term_table(self) -> dict[tuple[int, int], float]:
    """
    Precompute the n choose d terms for the binomial penalty.
    """
    n_choose_d_term_table = {}
    for n in range(self.n_patients + 1):
      for d in range(n + 1):
        n_choose_d_term_table[(n, d)] = (
          math.lgamma(n + 1)
          - math.lgamma(d + 1)
          - math.lgamma(n - d + 1)
        )
    return n_choose_d_term_table

  def add_binomial_penalty(
    self,
    model: gp.Model,
    *,
    n_at_risk: gp.tupledict[tuple[int, ...], gp.Var],
    n_died: gp.tupledict[tuple[int, ...], gp.Var],
    n_survived: gp.tupledict[tuple[int, ...], gp.Var],
  ):
    p_survived = model.addVars(len(self.all_death_times), 2, vtype=gp.GRB.CONTINUOUS, name="p_survived", lb=0, ub=1)
    p_died = model.addVars(len(self.all_death_times), 2, vtype=gp.GRB.CONTINUOUS, name="p_died", lb=0, ub=1)
    log_p_bounds = np.array([
      np.log(self.__endpoint_epsilon / len(self.all_death_times) / 2),
      np.log(1 - self.__endpoint_epsilon / len(self.all_death_times) / 2),
    ])
    log_p_survived = model.addVars(len(self.all_death_times), 2, vtype=gp.GRB.CONTINUOUS, name="log_p_survived", lb=log_p_bounds[0], ub=log_p_bounds[1])
    log_p_died = model.addVars(len(self.all_death_times), 2, vtype=gp.GRB.CONTINUOUS, name="log_p_died", lb=log_p_bounds[0], ub=log_p_bounds[1])
    for i in range(len(self.all_death_times)):
      for j in range(2):
        model.addGenConstrExp(log_p_survived[i, j], p_survived[i, j], name=f"log_p_survived_constr_{i}_{j}")
        model.addGenConstrExp(log_p_died[i, j], p_died[i, j], name=f"log_p_died_constr_{i}_{j}")
        model.addConstr(
          p_survived[i, j] + p_died[i, j] == 1,
          name=f"survived_died_constraint_{i}_{j}"
        )

    n_choose_d_term_table = self.n_choose_d_term_table
    n_choose_d_indicator_vars = model.addVars(
      len(self.all_death_times), 2, len(n_choose_d_term_table),
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
          gp.quicksum(n_choose_d_indicator_vars[i, j, k] for k in range(len(n_choose_d_term_table))) == 1,
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
        for k, ((n, d), penalty) in enumerate(n_choose_d_term_table.items()):
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

    #under the null hypothesis, both curves have the same survival probability
    null_hypothesis_indicator = model.addVar(vtype=gp.GRB.BINARY, name="null_hypothesis_indicator")
    for i in range(len(self.all_death_times)):
      model.addGenConstrIndicator(
        null_hypothesis_indicator,
        True,
        n_survived[i, 0] - n_survived[i, 1],
        GRB.EQUAL,
        0,
        name=f"null_hypothesis_{i}"
      )

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
    binomial_penalty, null_hypothesis_indicator = self.add_binomial_penalty(model, n_died=n_died, n_at_risk=n_at_risk, n_survived=n_survived)
    patient_penalty = self.add_patient_wise_penalty(model, x)

    model.setObjective(
      2 * (binomial_penalty + patient_penalty),
      GRB.MINIMIZE,
    )
    model.update()

    return model, null_hypothesis_indicator

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
    if self.__null_hypothesis_constraint is not None:
      model.remove(self.__null_hypothesis_constraint)

    if null_hypothesis:
      self.__null_hypothesis_constraint = model.addConstr(
        null_hypothesis_indicator == 1,
        name="null_hypothesis_constraint"
      )
    else:
      self.__null_hypothesis_constraint = None

  def solve_and_pvalue(self):
    model, null_hypothesis_indicator = self.gurobi_model
    self.update_model_for_null_hypothesis_or_not(model, null_hypothesis_indicator, True)
    model.optimize()
    if model.status != GRB.OPTIMAL:
      raise ValueError("Null model did not converge")
    twonll_null = model.ObjVal
    result_null = scipy.optimize.OptimizeResult(
      x=model.ObjVal,
      success=model.status == GRB.OPTIMAL,
    )

    self.update_model_for_null_hypothesis_or_not(model, null_hypothesis_indicator, False)
    model.optimize()
    if model.status != GRB.OPTIMAL:
      raise ValueError("Alternative model did not converge")
    twonll_alt = model.ObjVal
    result_alt = scipy.optimize.OptimizeResult(
      x=model.ObjVal,
      success=model.status == GRB.OPTIMAL,
    )

    lr_stat = twonll_null - twonll_alt
    # The number of degrees of freedom is the number of constraints added
    # under the null hypothesis.  This is one constraint per death time:
    # that the survival probabilities of the two curves are equal.
    df = len(self.all_death_times)
    p_value = scipy.stats.chi2.sf(lr_stat, df)
    return p_value, result_null, result_alt