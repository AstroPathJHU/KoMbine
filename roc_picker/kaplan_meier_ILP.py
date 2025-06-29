#pylint: disable=too-many-lines
"""
Integer Linear Programming implementation for the Kaplan-Meier likelihood method.
"""

import collections.abc
import datetime
import functools
import itertools
import math

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.stats

from .kaplan_meier import (
  KaplanMeierPatientBase,
  KaplanMeierPatient,
)

class KaplanMeierPatientNLL(KaplanMeierPatientBase):
  """
  A patient with a time and a parameter.
  The parameter is a log-likelihood function.
  """
  def __init__(
    self,
    time: float,
    censored: bool,
    parameter_nll: collections.abc.Callable[[float], float],
    observed_parameter: float,
  ):
    super().__init__(
      time=time,
      censored=censored,
      parameter=parameter_nll,
    )
    self.__observed_parameter = observed_parameter

  @property
  def parameter(self) -> collections.abc.Callable[[float], float]:
    """
    The parameter is a log-likelihood function.
    """
    return super().parameter

  @property
  def observed_parameter(self) -> float:
    """
    The observed value of the parameter.
    """
    return self.__observed_parameter

  @classmethod
  def from_count(
    cls,
    time: float,
    censored: bool,
    count: int,
  ):
    """
    Create a KaplanMeierPatientNLL from a count.
    The parameter NLL gives the negative log-likelihood to observe the count
    given the parameter, which is the mean of the Poisson distribution.
    """
    def parameter_nll(x: float) -> float:
      """
      The parameter is a log-likelihood function.
      """
      return -scipy.stats.poisson.logpmf(count, x).item()
    return cls(
      time=time,
      censored=censored,
      parameter_nll=parameter_nll,
      observed_parameter=count,
    )

  @classmethod
  def from_poisson_density(
    cls,
    time: float,
    censored: bool,
    numerator_count: int,
    denominator_area: float,
  ):
    """
    Create a KaplanMeierPatientNLL from a Poisson count
    divided by an area that is known precisely.
    """
    def parameter_nll(density: float) -> float:
      """
      The parameter is a log-likelihood function.
      """
      return -scipy.stats.poisson.logpmf(
        numerator_count,
        density * denominator_area,
      ).item()
    return cls(
      time=time,
      censored=censored,
      parameter_nll=parameter_nll,
      observed_parameter=numerator_count / denominator_area,
    )


  @classmethod
  def from_poisson_ratio(
    cls,
    time: float,
    censored: bool,
    numerator_count: int,
    denominator_count: int,
  ):
    """
    Create a KaplanMeierPatientNLL from a ratio of two counts.
    The parameter NLL gives the negative log-likelihood to observe the
    numberator and denominator counts given the parameter, which is the
    ratio of the two Poisson distribution means.  We do this by floating
    the denominator mean and fixing the numerator mean to the ratio
    times the denominator mean.  We then minimize the NLL to observe the
    numerator and denominator counts given the denominator mean.
    """
    def parameter_nll(ratio: float) -> float:
      if ratio < 0:
        return float('inf')  # Ratio must be positive

      # Define the NLL as a function of the (latent) denominator mean
      def nll_given_lambda_d(lambda_d: float) -> float:
        lambda_n = ratio * lambda_d
        # Poisson negative log-likelihoods
        nll_numerator = -scipy.stats.poisson.logpmf(numerator_count, lambda_n)
        nll_denominator = -scipy.stats.poisson.logpmf(denominator_count, lambda_d)
        return (nll_numerator + nll_denominator).item()

      # Optimize over lambda_d > 0
      result = scipy.optimize.minimize_scalar(
        nll_given_lambda_d,
        bounds=(1e-8, 1e6),
        method='bounded'
      )
      assert isinstance(result, scipy.optimize.OptimizeResult)
      return result.fun if result.success else float('inf')

    # Use the MLE ratio as the observed value
    observed_ratio = numerator_count / denominator_count if denominator_count > 0 else float('inf')

    return cls(
      time=time,
      censored=censored,
      parameter_nll=parameter_nll,
      observed_parameter=observed_ratio,
    )

  @property
  def nominal(self) -> KaplanMeierPatient:
    """
    The nominal value of the parameter.
    """
    return KaplanMeierPatient(
      time=self.time,
      censored=self.censored,
      parameter=self.observed_parameter,
    )

class ILPForKM:  # pylint: disable=too-many-public-methods
  """
  Integer Linear Programming for a point on the Kaplan-Meier curve.
  """
  def __init__(
    self,
    all_patients: list[KaplanMeierPatientNLL],
    parameter_min: float,
    parameter_max: float,
    time_point: float,
  ):
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_max = parameter_max
    self.__time_point = time_point
    self.__expected_probability_constraint = None
    self.__binomial_penalty_constraint = None
    if not np.isfinite(self.__parameter_min and self.__parameter_min != -np.inf):
      raise ValueError("parameter_min must be finite or -inf")
    if not np.isfinite(self.__parameter_max and self.__parameter_max != np.inf):
      raise ValueError("parameter_max must be finite or inf")

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
    The minimum parameter value.
    """
    return self.__parameter_min
  @property
  def parameter_max(self) -> float:
    """
    The maximum parameter value.
    """
    return self.__parameter_max
  @property
  def time_point(self) -> float:
    """
    The time point for the Kaplan-Meier curve.
    """
    return self.__time_point
  @functools.cached_property
  def patient_times(self) -> npt.NDArray[np.float64]:
    """
    The times of all patients.
    """
    return np.array([p.time for p in self.all_patients])
  @functools.cached_property
  def patient_censored(self) -> npt.NDArray[np.bool_]:
    """
    The censored status of all patients.
    """
    return np.array([p.censored for p in self.all_patients])
  @functools.cached_property
  def patient_still_at_risk(self) -> npt.NDArray[np.bool_]:
    """
    The status of all patients at risk at the time point.
    A patient is still at risk if their time is greater than the time point
    or if they are censored at the time point.
    """
    return (
      (self.patient_times > self.time_point)
      | ((self.patient_times == self.time_point) & self.patient_censored)
    )
  @functools.cached_property
  def patient_alive(self) -> npt.NDArray[np.bool_]:
    """
    The status of all patients who are alive at the time point.
    A patient is alive if their time is greater than the time point.
    """
    return self.patient_times > self.time_point
  @functools.cached_property
  def observed_parameters(self) -> npt.NDArray[np.float64]:
    """
    The observed parameters of all patients.
    """
    return np.array([p.observed_parameter for p in self.all_patients])
  @functools.cached_property
  def parameter_in_range(self) -> npt.NDArray[np.bool_]:
    """
    Whether each patient's observed parameter is within the specified range.
    """
    return (
      (self.observed_parameters >= self.parameter_min)
      & (self.observed_parameters < self.parameter_max)
    )

  @staticmethod
  def group_patients(  # pylint: disable=too-many-locals
    patient_times: npt.NDArray[np.float64],
    patient_censored: npt.NDArray[np.bool_],
    patient_still_at_risk: npt.NDArray[np.bool_],
  ) -> tuple[list[npt.NDArray[np.bool_]], list[npt.NDArray[np.bool_]]]:
    """
    divide the patients into groups:
    first the ones who were censored before anyone died
    then the ones who died
    then the next ones who were censored
    then the next ones who died
    etc.
    """
    censored_in_group = []
    died_in_group = []

    # Restrict to events that have occurred
    valid_mask = ~patient_still_at_risk
    event_times = patient_times[valid_mask]
    event_censored = patient_censored[valid_mask]

    # Order by time, deaths before censors
    event_order = np.lexsort((event_censored, event_times))
    sorted_indices = np.flatnonzero(valid_mask)[event_order]

    censored_in_group = []
    died_in_group = []

    i = 0
    n = len(sorted_indices)

    # Track grouped event types to adjust start and end if needed
    group_event_types = []

    while i < n:
      idx = sorted_indices[i]
      current_type = patient_censored[idx]  # True if censored

      group_indices = [idx]
      i += 1

      # Group consecutive events of the same type
      while i < n:
        next_idx = sorted_indices[i]
        next_type = patient_censored[next_idx]
        if next_type != current_type:
          break
        group_indices.append(next_idx)
        i += 1

      group_mask = np.zeros_like(patient_times, dtype=bool)
      group_mask[group_indices] = True

      group_event_types.append(current_type)
      if current_type:
        censored_in_group.append(group_mask)
      else:
        died_in_group.append(group_mask)

    # Adjust for leading death group (needs dummy censored group first)
    if group_event_types and not group_event_types[0]:
      censored_in_group.insert(0, np.zeros_like(patient_times, dtype=bool))

    # Adjust for trailing censor group (needs dummy death group last)
    if group_event_types and group_event_types[-1]:
      del censored_in_group[-1]

    # If there are no groups, we need to ensure at least one group exists
    if not censored_in_group and not died_in_group:
      # Create a dummy group with no patients
      dummy_group = np.zeros_like(patient_times, dtype=bool)
      censored_in_group.append(dummy_group)
      died_in_group.append(dummy_group)

    # Final sanity check
    if len(censored_in_group) != len(died_in_group):
      raise ValueError("Mismatched lengths between censored and died groups.")

    return censored_in_group, died_in_group

  @functools.cached_property
  def _patient_groups(self):
    """
    Group patients by their status at the time point.
    Returns a tuple of (censored_in_group, died_in_group, n_groups).
    """
    return self.group_patients(
      patient_times=self.patient_times,
      patient_censored=self.patient_censored,
      patient_still_at_risk=self.patient_still_at_risk,
    )
  @functools.cached_property
  def censored_in_group(self) -> list[npt.NDArray[np.bool_]]:
    """
    The groups of patients who were censored before the time point.
    """
    return self._patient_groups[0]
  @functools.cached_property
  def died_in_group(self) -> list[npt.NDArray[np.bool_]]:
    """
    The groups of patients who died before the time point.
    """
    return self._patient_groups[1]
  @functools.cached_property
  def n_groups(self) -> int:
    """
    The number of groups of patients.
    """
    result = len(self.censored_in_group)
    if result != len(self.died_in_group):
      raise ValueError(
        "The number of censored groups does not match the number of died groups."
      )
    return result

  @functools.cached_property
  def n_censored_in_group_total(self) -> npt.NDArray[np.int_]:
    """
    The total number of patients who were censored in each group.
    """
    return np.array([
      np.count_nonzero(self.censored_in_group[i])
      for i in range(self.n_groups)
    ])
  @functools.cached_property
  def n_died_in_group_total(self) -> npt.NDArray[np.int_]:
    """
    The total number of patients who died in each group.
    """
    return np.array([
      np.count_nonzero(self.died_in_group[i])
      for i in range(self.n_groups)
    ])

  @functools.cached_property
  def n_censored_in_group_obs(self) -> npt.NDArray[np.int_]:
    """
    The number of patients who were censored in each group
    using the observed parameters.
    """
    return np.array([
      np.count_nonzero(self.censored_in_group[i] & self.parameter_in_range)
      for i in range(self.n_groups)
    ])
  @functools.cached_property
  def n_died_in_group_obs(self) -> npt.NDArray[np.int_]:
    """
    The number of patients who died in each group
    using the observed parameters.
    """
    return np.array([
      np.count_nonzero(self.died_in_group[i] & self.parameter_in_range)
      for i in range(self.n_groups)
    ])
  @functools.cached_property
  def n_total_obs(self) -> int:
    """
    The total number of patients who are alive at the time point
    using the observed parameters.
    """
    return int(np.count_nonzero(self.parameter_in_range))
  @functools.cached_property
  def n_alive_obs(self) -> int:
    """
    The number of patients who are alive at the time point
    using the observed parameters.
    """
    return int(np.count_nonzero(self.patient_alive & self.parameter_in_range))

  @staticmethod
  @functools.cache
  def create_binomial_penalty_table(
    n_patients: int,
    expected_probability: float,
  ) -> dict[tuple[int, int], float]:
    """
    Create a table of binomial penalties for each (n_alive, n_total) pair.
    The penalty is the negative log-likelihood of observing n_alive out of n_total
    patients alive at the time point.
    This needs to be fixed to account for censoring.
    """
    binomial_penalty_table = {}
    for n_total in range(n_patients + 1):
      for n_alive in range(n_total + 1):
        penalty = -scipy.stats.binom.logpmf(n_alive, n_total, expected_probability)
        binomial_penalty_table[(n_alive, n_total)] = penalty.item()
    return binomial_penalty_table

  @classmethod
  def calculate_KM_probability(
    cls,
    total_count: int,
    censored_counts: tuple[int, ...],
    died_counts: tuple[int, ...],
  ) -> float:
    """
    Calculate the Kaplan-Meier probability at the time point.
    """
    if len(censored_counts) != len(died_counts):
      raise ValueError("Censored and died counts must have the same length")
    n_groups = len(censored_counts)

    if not n_groups:
      # If there are no groups, nobody died, so the probability is 1
      probability = 1.0
    else:
      n_at_risk = [total_count - censored_counts[0]]
      for i in range(1, n_groups):
        n_at_risk.append(
          n_at_risk[i - 1] - died_counts[i - 1] - censored_counts[i]
        )
      probability = 1.0
      for i in range(n_groups):
        if n_at_risk[i] > 0:
          probability *= (
            n_at_risk[i] - died_counts[i]
          ) / n_at_risk[i]
        else:
          probability = 0

    return probability

  @functools.cached_property
  def observed_KM_probability(self) -> float:
    """
    The observed Kaplan-Meier probability at the time point.
    This is calculated using the observed counts of patients who were censored or died.
    """
    return self.calculate_KM_probability(
      total_count=self.n_total_obs,
      censored_counts=tuple(self.n_censored_in_group_obs),
      died_counts=tuple(self.n_died_in_group_obs),
    )

  @classmethod
  @functools.cache
  def calculate_valid_trajectories(
    cls,
    n_total_patients: int,
    n_censored_in_group: tuple[int, ...],
    n_died_in_group: tuple[int, ...],
    verbose=False,
  ) -> list[tuple[int, tuple[int], tuple[int], float]]:
    """
    Generate valid trajectories - the total number of included patients and the numbers of patients
    who were censored or died in each group - based on the total number of patients and the
    total number who were censored or died in each group.

    The minimum total count is 1 - we don't allow all patients to be excluded.
    """
    if len(n_censored_in_group) != len(n_died_in_group):
      raise ValueError("n_censored_in_group and n_died_in_group must have the same length")
    n_groups = len(n_censored_in_group)
    if sum(n_censored_in_group) + sum(n_died_in_group) > n_total_patients:
      raise ValueError(
        "The total number of patients who were censored or died "
        "exceeds the total number of patients"
      )
    result = []
    # For each group, possible number of censored and died patients included: 0..n_censored/died
    censored_ranges = [range(nc + 1) for nc in n_censored_in_group]
    died_ranges = [range(nd + 1) for nd in n_died_in_group]
    n_trajectories = (
      n_total_patients
        * np.prod(np.array(n_censored_in_group) + 1)
        * np.prod(np.array(n_died_in_group) + 1)
    )
    if verbose:
      print(
        f"Generating {n_trajectories} trajectories for "
        f"{n_total_patients} total patients in {n_groups} groups"
      )
    n_generated = 0
    for total_count in range(1, n_total_patients + 1): #pylint: disable=too-many-nested-blocks
      for censored_counts in itertools.product(*censored_ranges):
        for died_counts in itertools.product(*died_ranges):
          n_generated += 1
          if verbose and (n_generated % 1000 == 0 or n_generated == n_trajectories):
            print(f"  {n_generated} / {n_trajectories}")
          if sum(censored_counts) + sum(died_counts) > total_count:
            continue

          expected_trajectory_probability = cls.calculate_KM_probability(
            total_count=total_count,
            censored_counts=censored_counts,
            died_counts=died_counts,
          )
          result.append((
            total_count,
            tuple(censored_counts),
            tuple(died_counts),
            expected_trajectory_probability,
          ))
    return result

  @functools.cached_property
  def valid_trajectories(self) -> list[tuple[int, tuple[int], tuple[int], float]]:
    """
    Get the valid trajectories for the current set of patients.
    This is a list of tuples (total_count, censored_counts, died_counts, expected_probability).
    """
    return self.calculate_valid_trajectories(
      n_total_patients=self.n_patients,
      n_censored_in_group=tuple(self.n_censored_in_group_total),
      n_died_in_group=tuple(self.n_died_in_group_total),
      verbose=False,
    )

  @functools.cached_property
  def n_trajectories(self) -> int:
    """
    The number of valid trajectories.
    """
    return len(self.valid_trajectories)

  @functools.cached_property
  def nll_penalty_for_patient_in_range(self) -> npt.NDArray[np.float64]:
    """
    Calculate the negative log-likelihood penalty for each patient
    if that patient is within the parameter range.
    This is negative if the patient's observed parameter is within the range
    and positive if it is outside the range.
    """
    sgn_nll_penalty_for_patient_in_range = 2 * self.parameter_in_range - 1
    observed_nll = np.array([
      p.parameter(p.observed_parameter)
      for p in self.all_patients
    ])
    if np.isfinite(self.parameter_min):
      parameter_min_nll = np.array([
        p.parameter(self.parameter_min)
        for p in self.all_patients
      ])
    else:
      parameter_min_nll = np.full(self.n_patients, np.inf)
    if np.isfinite(self.parameter_max):
      parameter_max_nll = np.array([
        p.parameter(self.parameter_max)
        for p in self.all_patients
      ])
    else:
      parameter_max_nll = np.full(self.n_patients, np.inf)

    range_boundary_nll = np.min(
      np.array([parameter_min_nll, parameter_max_nll]),
      axis=0
    )
    abs_nll_penalty_for_patient_in_range = observed_nll - range_boundary_nll

    nll_penalty_for_patient_in_range = (
      sgn_nll_penalty_for_patient_in_range
      * abs_nll_penalty_for_patient_in_range
    )

    return nll_penalty_for_patient_in_range

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

  def _make_gurobi_model(self):  #pylint: disable=too-many-locals
    """
    Create the Gurobi model for the ILP.
    This method constructs the model with decision variables, constraints,
    and the objective function.  It does NOT include the constraint for the
    expected probability, which is added in update_model_with_expected_probability.
    """
    model = gp.Model("Kaplan-Meier ILP")

    # Binary decision variables: x[i] = 1 if patient i is within the parameter range
    x = model.addVars(self.n_patients, vtype=GRB.BINARY, name="x")
    n_total = model.addVar(vtype=GRB.INTEGER, name="n_total")
    model.addConstr(
      n_total == gp.quicksum(x[i] for i in range(self.n_patients)),
      name="n_total_constraint",
    )

    # Integer vars to count totals
    n_censored_in_group = model.addVars(
      self.n_groups,
      vtype=GRB.INTEGER,
      name="n_censored_in_group",
    )
    n_died_in_group = model.addVars(
      self.n_groups,
      vtype=GRB.INTEGER,
      name="n_died_in_group",
    )
    n_at_risk = model.addVars(
      self.n_groups,
      vtype=GRB.INTEGER,
      name="n_at_risk",
    )
    n_survived_in_group = model.addVars(
      self.n_groups,
      vtype=GRB.INTEGER,
      name="n_survived_in_group",
    )

    # Constraints to link to totals
    for idx in range(self.n_groups):
      model.addConstr(
        n_censored_in_group[idx] == gp.quicksum(
          x[i] for i in range(self.n_patients) if self.censored_in_group[idx][i]
        ),
        name=f"n_censored_in_group_{idx}",
      )
      model.addConstr(
        n_died_in_group[idx] == gp.quicksum(
          x[i] for i in range(self.n_patients) if self.died_in_group[idx][i]
        ),
        name=f"n_died_in_group_{idx}",
      )
      if idx == 0:
        model.addConstr(
          n_at_risk[idx] == n_total - n_censored_in_group[idx],
          name=f"n_at_risk_{idx}"
        )
      else:
        model.addConstr(
          n_at_risk[idx]
            == n_at_risk[idx - 1] - n_died_in_group[idx - 1] - n_censored_in_group[idx],
          name=f"n_at_risk_{idx}",
        )
      model.addConstr(
        n_survived_in_group[idx] == n_at_risk[idx] - n_died_in_group[idx],
        name=f"n_survived_in_group_{idx}",
      )
    n_alive = model.addVar(vtype=GRB.INTEGER, name="n_alive")
    model.addConstr(
      n_alive == gp.quicksum(
        x[i] for i in range(self.n_patients) if self.patient_alive[i]
      ),
      name="n_alive_constraint",
    )

    # indicator variables for each trajectory
    traj_indicator_vars = model.addVars(
      self.n_trajectories,
      vtype=GRB.BINARY,
      name="indicator"
    )
    # Constraints to enforce trajectory selection
    for (
      traj_idx, (total_count, censored_counts, died_counts, _)
    ) in enumerate(self.valid_trajectories):
      # Sum of selected patients must match trajectory counts
      model.addGenConstrIndicator(
        traj_indicator_vars[traj_idx],
        True,
        n_total,
        GRB.EQUAL,
        total_count,
        name=f"traj_indicator_constraint_{traj_idx}_total",
      )
      for group_idx in range(self.n_groups):
        model.addGenConstrIndicator(
          traj_indicator_vars[traj_idx],
          True,
          gp.quicksum(x[i] for i in range(self.n_patients) if self.censored_in_group[group_idx][i]),
          GRB.EQUAL,
          censored_counts[group_idx],
          name=f"traj_indicator_constraint_{traj_idx}_censored_{group_idx}",
        )
        model.addGenConstrIndicator(
          traj_indicator_vars[traj_idx],
          True,
          gp.quicksum(x[i] for i in range(self.n_patients) if self.died_in_group[group_idx][i]),
          GRB.EQUAL,
          died_counts[group_idx],
          name=f"traj_indicator_constraint_{traj_idx}_died_{group_idx}",
        )
    #Only one trajectory can be selected
    model.addConstr(
      traj_indicator_vars.sum() == 1,
      name="one_trajectory_constraint",
    )

    #Setup for binomial penalty
    #There's a separate binomial term for each group
    #To complicate things, we only know the overall expected survival probability,
    #not the probability of survival in each group.
    #So we need to profile those.

    #p_i = probability of dying in group i
    p_died = model.addVars(
      self.n_groups,
      vtype=GRB.CONTINUOUS,
      name="p_died",
      lb=0,
      ub=1,
    )
    p_survived = model.addVars(
      self.n_groups,
      vtype=GRB.CONTINUOUS,
      name="p_survived",
      lb=0,
      ub=1,
    )
    log_p_bounds = np.array([-20, -1e-6])
    log_p_died = model.addVars(
      self.n_groups,
      vtype=GRB.CONTINUOUS,
      name="log_p_died",
      lb=log_p_bounds[0],
      ub=log_p_bounds[1],
    )
    log_p_survived = model.addVars(
      self.n_groups,
      vtype=GRB.CONTINUOUS,
      name="log_p_survived",
      lb=log_p_bounds[0],
      ub=log_p_bounds[1],
    )
    for i in range(self.n_groups):
      model.addGenConstrExp(log_p_died[i], p_died[i], name=f"log_p_died_constr_{i}")
      model.addGenConstrExp(log_p_survived[i], p_survived[i], name=f"log_p_survived_constr_{i}")
      model.addConstr(
        p_died[i] + p_survived[i] == 1,
        name=f"p_died_plus_p_survived_{i}"
      )

    #product of survival probabilities = the overall expected probability
    #we will set the expected probability via a constraint in update_model_with_expected_probability
    expected_probability = model.addVar(
      vtype=GRB.CONTINUOUS,
      name="expected_probability",
      lb=0,
      ub=1,
    )
    log_expected_probability = model.addVar(
      vtype=GRB.CONTINUOUS,
      name="log_expected_probability",
      lb=-20,
      ub=-1e-6,
    )
    model.addGenConstrExp(
      log_expected_probability,
      expected_probability,
      name="exp_log_expected_probability"
    )
    model.addConstr(
      log_expected_probability == log_p_survived.sum(),
      name="overall_expected_probability_constraint",
    )

    #Binomial terms
    #binomial probability = (n_at_risk choose n_died)
    #                       * dying probability ^ n_died
    #                       * surviving probability ^ n_survived
    #  ==> log likelihood = log(n_at_risk choose n_died)
    #                       + n_died * log(dying probability)
    #                       + (n_at_risk - n_died) * log(surviving probability)
    #                     = log(n_at_risk choose n_died)
    #                       + n_died * log_p_died
    #                       + (n_at_risk - n_died) * log_p_survived

    use_binomial_penalty_indicator = model.addVar(
      vtype=GRB.BINARY,
      name="use_binomial_penalty_indicator",
    )

    #n_at_risk choose n_died term
    n_choose_d_term_table = self.n_choose_d_term_table
    n_choose_d_indicator_vars = model.addVars(
      self.n_groups * len(n_choose_d_term_table),
      vtype=GRB.BINARY,
      name="n_choose_d_indicator",
    )
    binomial_terms = []
    n_choose_d_indicator_vars_by_group = collections.defaultdict(list)
    for indicator_var_idx, (group_idx, ((n, d), penalty)) in enumerate(
      itertools.product(
        range(self.n_groups),
        n_choose_d_term_table.items(),
      )
    ):
      indicator = n_choose_d_indicator_vars[indicator_var_idx]
      n_choose_d_indicator_vars_by_group[group_idx].append(indicator)
      binomial_terms.append(
        -penalty * n_choose_d_indicator_vars[indicator_var_idx]
      )
      model.addGenConstrIndicator(
        n_choose_d_indicator_vars[indicator_var_idx],
        True,
        n_at_risk[group_idx],
        GRB.EQUAL,
        n,
        name=f"n_choose_d_indicator_n_{group_idx}_{n}",
      )
      model.addGenConstrIndicator(
        n_choose_d_indicator_vars[indicator_var_idx],
        True,
        n_died_in_group[group_idx],
        GRB.EQUAL,
        d,
        name=f"n_choose_d_indicator_d_{group_idx}_{n}_{d}",
      )
    for group_idx in range(self.n_groups):
      # Ensure that exactly one n_choose_d_indicator is selected for each group
      model.addConstr(
        gp.quicksum(
          n_choose_d_indicator_vars_by_group[group_idx]
        ) == 1,
        name=f"one_n_choose_d_indicator_per_group_{group_idx}",
      )

    n_died_indicator_vars = model.addVars(
      int(sum(self.n_died_in_group_total + 1)),
      vtype=GRB.BINARY,
      name="n_died_indicator",
    )
    n_died_indicator_vars_by_group = collections.defaultdict(list)
    i = 0
    for group_idx in range(self.n_groups):
      for d in range(self.n_died_in_group_total[group_idx] + 1):
        n_died_indicator_vars_by_group[group_idx].append(n_died_indicator_vars[i])
        model.addGenConstrIndicator(
          n_died_indicator_vars[i],
          True,
          n_died_in_group[group_idx],
          GRB.EQUAL,
          d,
          name=f"n_died_indicator_{group_idx}_{d}",
        )
        binomial_terms.append(
          -d * log_p_died[group_idx] * n_died_indicator_vars[i]
        )
        i += 1
    assert i == len(n_died_indicator_vars)
    # Ensure that exactly one n_died_indicator is selected for each group
    for group_idx in range(self.n_groups):
      model.addConstr(
        gp.quicksum(
          n_died_indicator_vars_by_group[group_idx]
        ) == 1,
        name=f"one_n_died_indicator_per_group_{group_idx}",
      )

    n_survived_indicator_vars = model.addVars(
      self.n_groups * (self.n_patients + 1),
      #could probably have somewhat fewer of these: the maximum is n_patients,
      #but the minimum is not 0.
      vtype=GRB.BINARY,
      name="n_survived_indicator",
    )
    n_survived_indicator_vars_by_group = collections.defaultdict(list)
    i = 0
    for group_idx in range(self.n_groups):
      for s in range(self.n_patients + 1):
        n_survived_indicator_vars_by_group[group_idx].append(n_survived_indicator_vars[i])
        model.addGenConstrIndicator(
          n_survived_indicator_vars[i],
          True,
          n_survived_in_group[group_idx],
          GRB.EQUAL,
          s,
          name=f"n_survived_indicator_{group_idx}_{s}",
        )
        binomial_terms.append(
          -s * log_p_survived[group_idx] * n_survived_indicator_vars[i]
        )
        i += 1
    assert i == len(n_survived_indicator_vars)
    # Ensure that exactly one n_survived_indicator is selected for each group
    for group_idx in range(self.n_groups):
      model.addConstr(
        gp.quicksum(
          n_survived_indicator_vars_by_group[group_idx]
        ) == 1,
        name=f"one_n_survived_indicator_per_group_{group_idx}",
      )

    # Patient-wise penalties
    patient_penalties = []
    for i in range(self.n_patients):
      if np.isfinite(self.nll_penalty_for_patient_in_range[i]):
        patient_penalties.append(self.nll_penalty_for_patient_in_range[i] * x[i])
      elif np.isposinf(self.nll_penalty_for_patient_in_range[i]):
        #the patient must be selected, so we add a constraint
        model.addConstr(x[i] == 1)
      elif np.isneginf(self.nll_penalty_for_patient_in_range[i]):
        #the patient must not be selected, so we add a constraint
        model.addConstr(x[i] == 0)
      else:
        raise ValueError(
          f"Unexpected NLL penalty for patient {i}: "
          f"{self.nll_penalty_for_patient_in_range[i]}"
        )

    patient_penalty = gp.quicksum(patient_penalties)

    binom_penalty_expr = gp.quicksum(binomial_terms)
    binom_penalty = model.addVar(
      vtype=GRB.CONTINUOUS,
      name="binom_penalty",
    )
    model.addGenConstrIndicator(
      use_binomial_penalty_indicator,
      False,
      binom_penalty,
      GRB.EQUAL,
      0.0
    )
    #big M constraint to ensure binomial penalty is only used when the indicator is set
    max_penalty_term = max(
      abs(penalty) for penalty in self.n_choose_d_term_table.values()
    )
    max_d = max([*self.n_died_in_group_total, 1]) #avoid ValueError for empty n_died_in_group_total
    max_s = self.n_patients
    max_log_p = max(np.abs(log_p_bounds))
    safety_factor = 2
    big_M = safety_factor * self.n_groups * (
      max_penalty_term
      + max_d * max_log_p
      + max_s * max_log_p
    )
    model.addConstr(
      binom_penalty <= binom_penalty_expr + big_M * (1 - use_binomial_penalty_indicator),
      name="binomial_penalty_expr_upper_bound"
    )
    model.addConstr(
      binom_penalty >= binom_penalty_expr - big_M * (1 - use_binomial_penalty_indicator),
      name="binomial_penalty_expr_lower_bound"
    )

    # Objective: minimize total penalty
    model.setObjective(
      2 * (binom_penalty + patient_penalty),
      GRB.MINIMIZE,
    )
    model.update()

    return (
      model,
      traj_indicator_vars,
      expected_probability,
      use_binomial_penalty_indicator,
    )

  @functools.cached_property
  def gurobi_model(self):
    """
    Create the Gurobi model for the ILP.
    This is a cached property to avoid recreating the model multiple times.
    """
    return self._make_gurobi_model()

  def update_model_with_expected_probability( # pylint: disable=too-many-arguments
    self,
    *,
    model: gp.Model,
    expected_probability: float,
    patient_wise_only: bool,
    traj_indicator_vars: gp.tupledict[int, gp.Var],
    use_binomial_penalty_indicator: gp.Var,
    expected_probability_var: gp.Var,
  ):
    """
    Update the Gurobi model with the expected probability constraint.
    This is the only thing that changes between runs of the ILP.
    """
    #drop the previous constraints if they exist
    if self.__expected_probability_constraint is not None:
      model.remove(self.__expected_probability_constraint)
      self.__expected_probability_constraint = None
    if self.__binomial_penalty_constraint is not None:
      model.remove(self.__binomial_penalty_constraint)
      self.__binomial_penalty_constraint = None

    if not patient_wise_only:
      # ---------------------------
      # Indicator grid for binomial penalty
      # ---------------------------

      # Binomial penalty
      self.__binomial_penalty_constraint = model.addConstr(
        use_binomial_penalty_indicator == 1,
        name="use_binomial_penalty"
      )
      self.__expected_probability_constraint = model.addConstr(
        expected_probability_var == expected_probability,
        name="expected_probability_constraint",
      )
    else:
      #no binomial penalty means there's nothing to constrain the observed
      #probability to the expected probability.  In that case, what does
      #it mean to get an NLL for the expected probability?
      #Instead, we constrain the observed probability to be at least as
      #far from the nominal observed probability as the expected
      #and find the minimum patient-wise NLL.
      self.__binomial_penalty_constraint = model.addConstr(
        use_binomial_penalty_indicator == 0,
        name="use_binomial_penalty"
      )

      if expected_probability > self.observed_KM_probability:
        self.__expected_probability_constraint = model.addConstr(
          gp.quicksum(
            traj_indicator_vars[i] for i in range(self.n_trajectories)
            if self.valid_trajectories[i][3] >= expected_probability
          ) == 1
        )
      elif expected_probability < self.observed_KM_probability:
        self.__expected_probability_constraint = model.addConstr(
          gp.quicksum(
            traj_indicator_vars[i] for i in range(self.n_trajectories)
            if self.valid_trajectories[i][3] <= expected_probability
          ) == 1
        )
      else:
        assert expected_probability == self.observed_KM_probability

    model.update()

  def run_ILP( # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
    self,
    expected_probability: float,
    *,
    verbose=False,
    print_progress=False,
    binomial_only=False,
    patient_wise_only=False,
    gurobi_rtol=1e-6,
  ):
    """
    Run the ILP for the given time point.
    """
    if print_progress or verbose:
      print(
        "Running ILP for expected probability ", expected_probability,
        " at time point ", self.time_point, " at time ", datetime.datetime.now()
      )
    if not patient_wise_only and (expected_probability <= 0 or expected_probability >= 1):
      raise ValueError("expected_probability must be in (0, 1)")
    if expected_probability < 0 or expected_probability > 1:
      raise ValueError("expected_probability must be in [0, 1]")
    if binomial_only and patient_wise_only:
      raise ValueError("binomial_only and patient_wise_only cannot both be True")
    if not patient_wise_only and np.any(self.patient_censored):
      raise NotImplementedError(
        "Censored patients are not supported except in patient-wise-only mode"
      )

    binomial_penalty_table = self.create_binomial_penalty_table(
      n_patients=self.n_patients,
      expected_probability=expected_probability,
    )

    nll_penalty_for_patient_in_range = self.nll_penalty_for_patient_in_range

    if binomial_only or not any(np.isfinite(nll_penalty_for_patient_in_range)):
      if patient_wise_only:
        raise ValueError(
          "patient_wise_only cannot be True when binomial_only is True "
          "or when the parameter range is not finite"
        )
      binomial_penalty_val = binomial_penalty_table[(self.n_alive_obs, self.n_total_obs)]
      return scipy.optimize.OptimizeResult(
        x=2*binomial_penalty_val,
        success=True,
        n_total=self.n_total_obs,
        n_alive=self.n_alive_obs,
        binomial_2NLL=2*binomial_penalty_val,
        patient_2NLL=0,
        selected=self.parameter_in_range,
        model=None,
      )

    (
      model,
      traj_indicator_vars,
      expected_probability_var,
      use_binomial_penalty_indicator,
    ) = self.gurobi_model
    self.update_model_with_expected_probability(
      model=model,
      traj_indicator_vars=traj_indicator_vars,
      expected_probability=expected_probability,
      patient_wise_only=patient_wise_only,
      expected_probability_var=expected_probability_var,
      use_binomial_penalty_indicator=use_binomial_penalty_indicator,
    )

    if verbose:
      model.setParam('OutputFlag', 1)
      model.setParam('DisplayInterval', 1)
    else:
      # Suppress Gurobi output
      model.setParam('OutputFlag', 0)
    model.setParam("MIPGap", gurobi_rtol)

    model.optimize()

    if model.status != GRB.OPTIMAL:
      if model.status == GRB.INFEASIBLE and patient_wise_only:
        # If the model is infeasible, it means that no patients can be selected
        # while satisfying the constraints. This can happen if the expected
        # probability is too far from the observed probability and there are
        # some patients with infinite NLL penalties.
        return scipy.optimize.OptimizeResult(
          x=np.inf,
          success=False,
          n_total=0,
          n_alive=0,
          binomial_2NLL=np.inf,
          patient_2NLL=np.inf,
          patient_penalties=nll_penalty_for_patient_in_range,
          selected=[],
          model=model,
        )
      raise RuntimeError(
        f"Model optimization failed with status {model.status}. "
        "This may indicate an issue with the ILP formulation or the input data."
      )

    x: list[gp.Var] = []
    for i in range(self.n_patients):
      var = model.getVarByName(f"x[{i}]")
      assert var is not None
      x.append(var)
    n_total = model.getVarByName("n_total")
    n_alive = model.getVarByName("n_alive")
    assert n_total is not None
    assert n_alive is not None
    selected = [i for i in range(self.n_patients) if x[i].X > 0.5]
    n_alive_val = np.rint(n_alive.X)
    n_total_val = np.rint(n_total.X)

    patient_penalty_val = sum(
      nll_penalty_for_patient_in_range[i] * x[i].X
      for i in range(self.n_patients)
      if np.isfinite(nll_penalty_for_patient_in_range[i])
    )
    binom_penalty_var = model.getVarByName("binom_penalty")
    assert binom_penalty_var is not None
    binomial_penalty_val = binom_penalty_var.X
    if verbose:
      print("Selected patients:", selected)
      print("n_total:          ", int(n_total_val))
      print("n_alive:          ", int(n_alive_val))
      print("Binomial penalty: ", binomial_penalty_val)
      print("Patient penalty:  ", patient_penalty_val)
      print("Total penalty:    ", model.ObjVal)

    return scipy.optimize.OptimizeResult(
      x=model.ObjVal,
      success=model.status == GRB.OPTIMAL,
      n_total=n_total_val,
      n_alive=n_alive_val,
      binomial_2NLL=2*binomial_penalty_val,
      patient_2NLL=2*patient_penalty_val,
      patient_penalties=nll_penalty_for_patient_in_range,
      selected=selected,
      model=model,
    )
