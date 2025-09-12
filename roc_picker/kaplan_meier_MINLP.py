#pylint: disable=too-many-lines
"""
Mixed Integer Nonlinear Programming implementation for the Kaplan-Meier likelihood method.
"""

import collections.abc
import datetime
import functools
import itertools
import math
import os

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
from .utilities import LOG_ZERO_EPSILON_DEFAULT

def n_choose_d_term_table(n_patients) -> dict[tuple[int, int], float]:
  """
  Precompute the n choose d terms for the binomial penalty.
  """
  table = {}
  for n in range(n_patients + 1):
    for d in range(n + 1):
      table[(n, d)] = (
        math.lgamma(n + 1)
        - math.lgamma(d + 1)
        - math.lgamma(n - d + 1)
      )
  return table


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

  @staticmethod
  def _solve_0d(
    full_nll: collections.abc.Callable[[float], float],
  ) -> collections.abc.Callable[[float], float]:
    def wrapped(effective_param: float) -> float:
      return full_nll(effective_param)
    return wrapped

  @staticmethod
  def _solve_1d(
    full_nll: collections.abc.Callable[[float, float], float],
    *,
    var_type: str,  # 'theta' (ℝ) or 'positive' (>0 via exp)
  ) -> collections.abc.Callable[[float], float]:
    if var_type == 'theta':
      def map_var(s: float) -> float:
        return s
    elif var_type == 'positive':
      def map_var(s: float) -> float:
        return float(np.exp(s))
    else:
      raise ValueError(f"Unexpected var_type={var_type}")

    def wrapped(effective_param: float) -> float:
      def obj(s_arr: np.ndarray) -> float:
        s = float(s_arr[0])
        v = map_var(s)
        return full_nll(effective_param, v)
      res = scipy.optimize.minimize(
        obj,
        x0=np.array([0.0]),
        method='BFGS',
      )
      if not res.success:
        raise RuntimeError(f"Optimization failed:\n{res}")
      return res.fun
    return wrapped

  @staticmethod
  def _solve_nd(
    full_nll: collections.abc.Callable[[float, list[float]], float],
    *,
    var_types: list[str],  # each in {'theta','positive'}
  ) -> collections.abc.Callable[[float], float]:
    def map_vars(s_vec: np.ndarray) -> list[float]:
      out: list[float] = []
      for s, t in zip(s_vec, var_types, strict=True):
        if t == 'theta':
          out.append(float(s))
        elif t == 'positive':
          out.append(float(np.exp(s)))
        else:
          raise ValueError(f"Unexpected var_type={t}")
      return out

    def wrapped(effective_param: float) -> float:
      def obj(s_vec: np.ndarray) -> float:
        vars_ = map_vars(s_vec)
        return full_nll(effective_param, vars_)
      res = scipy.optimize.minimize(
        obj,
        x0=np.zeros(len(var_types)),
        method='BFGS',
        options={
          #loose tolerance - we don't actually care about the values of the nuisance parameters
          "gtol": 1e-3,
        }
      )
      if not res.success:
        raise RuntimeError(f"Optimization failed:\n{res}")
      return res.fun
    return wrapped

  # ---------- constructors with full_nll definitions ----------

  @classmethod
  def from_fixed_observable( # pylint: disable=too-many-arguments
    cls,
    time: float,
    censored: bool,
    observable: float,
    *,
    rel_epsilon: float = 1e-6,
    abs_epsilon: float = 1e-8,
    systematics: list[float] | None = None,
  ):
    """
    Create a KaplanMeierPatientNLL from a fixed observable.
    The NLL is a delta function, plus systematics.
    """
    systematics = systematics or []
    m = len(systematics)

    if m == 0 or observable == 0:
      # if observable == 0, then multiplicative systematics can't affect it
      # 0D: direct check
      def full_nll_0d(eff: float) -> float:
        return (
          0.0
          if np.isclose(eff, observable, rtol=rel_epsilon, atol=abs_epsilon)
          else float('inf')
        )
      wrapped = cls._solve_0d(full_nll_0d)

    elif m == 1:
      # analytic: eff = observable * a^theta  => theta = ln(eff/observable)/ln(a)
      a = systematics[0]
      if a <= 0:
        raise ValueError("Systematic base 'a' must be > 0")
      def full_nll_1d(eff: float) -> float:
        if eff <= 0:
          return float('inf')
        theta = np.log(eff / observable) / np.log(a)
        return 0.5 * float(theta * theta)
      wrapped = cls._solve_0d(full_nll_1d)

    else:
      # nD over thetas
      def full_nll_nd(eff: float, thetas_except_last: list[float]) -> float:
        if eff <= 0:
          return float('inf')
        last_theta = (
          np.log(eff / observable)
          - sum(theta * np.log(a) for theta, a in zip(thetas_except_last, systematics[:-1]))
        ) / np.log(systematics[-1])
        thetas = thetas_except_last + [last_theta]
        return 0.5 * float(np.sum(np.square(thetas)))
      wrapped = cls._solve_nd(full_nll_nd, var_types=['theta'] * (m-1))

    return cls(time, censored, wrapped, observable)

  @classmethod
  def from_count(
    cls,
    time: float,
    censored: bool,
    count: int,
    *,
    systematics: list[float] | None = None,
  ):
    """
    Create a KaplanMeierPatientNLL from a count.
    The parameter NLL gives the negative log-likelihood to observe the count
    given the parameter, which is the mean of the Poisson distribution.
    """
    systematics = systematics or []
    m = len(systematics)

    if m == 0:
      # 0D: parameter is the Poisson mean itself
      def full_nll_0d(eff: float) -> float:
        if eff == 0 and count == 0:
          return 0
        if eff <= 0:
          return float('inf')
        return -scipy.stats.poisson.logpmf(count, eff).item()
      wrapped = cls._solve_0d(full_nll_0d)

    elif m == 1:
      # 1D over theta
      a = systematics[0]
      if a <= 0:
        raise ValueError("Systematic base 'a' must be > 0")
      def full_nll_1d(eff: float, theta: float) -> float:
        if eff == 0 and count == 0:
          return 0
        if eff <= 0:
          return float('inf')
        nominal = eff / (a**theta)
        if nominal <= 0:
          return float('inf')
        base = -scipy.stats.poisson.logpmf(count, nominal).item()
        penalty = 0.5 * float(theta * theta)
        return base + penalty
      wrapped = cls._solve_1d(full_nll_1d, var_type='theta')

    else:
      # nD over thetas
      def full_nll_nd(eff: float, thetas: list[float]) -> float:
        if eff == 0 and count == 0:
          return 0
        if eff <= 0:
          return float('inf')
        prod_factor = 1.0
        for a, t in zip(systematics, thetas, strict=True):
          if a <= 0:
            return float('inf')
          prod_factor *= a**t
        nominal = eff / prod_factor
        if nominal <= 0:
          return float('inf')
        base = -scipy.stats.poisson.logpmf(count, nominal).item()
        penalty = 0.5 * float(np.sum(np.square(thetas)))
        return base + penalty
      wrapped = cls._solve_nd(full_nll_nd, var_types=['theta'] * m)

    return cls(time, censored, wrapped, count)

  @classmethod
  def from_poisson_density( # pylint: disable=too-many-arguments
    cls,
    time: float,
    censored: bool,
    numerator_count: int,
    denominator_area: float,
    *,
    systematics: list[float] | None = None,
  ):
    """
    Create a KaplanMeierPatientNLL from a Poisson count
    divided by an area that is known precisely.
    """
    if denominator_area <= 0:
      raise ValueError("denominator_area must be > 0")
    systematics = systematics or []
    m = len(systematics)

    if m == 0:
      # 0D: parameter is the density itself
      def full_nll_0d(eff_density: float) -> float:
        if eff_density == 0 and numerator_count == 0:
          return 0
        if eff_density <= 0:
          return float('inf')
        lam = eff_density * denominator_area
        return -scipy.stats.poisson.logpmf(numerator_count, lam).item()
      wrapped = cls._solve_0d(full_nll_0d)

    elif m == 1:
      # 1D over theta
      a = systematics[0]
      if a <= 0:
        raise ValueError("Systematic base 'a' must be > 0")
      def full_nll_1d(eff_density: float, theta: float) -> float:
        if eff_density == 0 and numerator_count == 0:
          return 0
        if eff_density <= 0:
          return float('inf')
        nominal = eff_density / (a**theta)
        if nominal <= 0:
          return float('inf')
        lam = nominal * denominator_area
        base = -scipy.stats.poisson.logpmf(numerator_count, lam).item()
        penalty = 0.5 * float(theta * theta)
        return base + penalty
      wrapped = cls._solve_1d(full_nll_1d, var_type='theta')

    else:
      # nD over thetas
      def full_nll_nd(eff_density: float, thetas: list[float]) -> float:
        if eff_density == 0 and numerator_count == 0:
          return 0
        if eff_density <= 0:
          return float('inf')
        prod_factor = 1.0
        for a, t in zip(systematics, thetas, strict=True):
          if a <= 0:
            return float('inf')
          prod_factor *= a**t
        nominal = eff_density / prod_factor
        if nominal <= 0:
          return float('inf')
        lam = nominal * denominator_area
        base = -scipy.stats.poisson.logpmf(numerator_count, lam).item()
        penalty = 0.5 * float(np.sum(np.square(thetas)))
        return base + penalty
      wrapped = cls._solve_nd(full_nll_nd, var_types=['theta'] * m)

    observed_density = numerator_count / denominator_area
    return cls(time, censored, wrapped, observed_density)

  @classmethod
  def from_poisson_ratio( # pylint: disable=too-many-arguments
    cls,
    time: float,
    censored: bool,
    numerator_count: int,
    denominator_count: int,
    *,
    systematics: list[float] | None = None,
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
    if denominator_count < 0 or numerator_count < 0:
      raise ValueError("Counts must be >= 0")
    systematics = systematics or []
    m = len(systematics)

    if m == 0:
      # 1D over lambda_d > 0 (no systematics)
      def full_nll_1d(eff_ratio: float, lambda_d: float) -> float:
        if eff_ratio == 0 and numerator_count == 0:
          return 0
        if eff_ratio <= 0 or lambda_d <= 0:
          return float('inf')
        lambda_n = eff_ratio * lambda_d
        nll_n = -scipy.stats.poisson.logpmf(numerator_count, lambda_n)
        nll_d = -scipy.stats.poisson.logpmf(denominator_count, lambda_d)
        return float((nll_n + nll_d).item())
      wrapped = cls._solve_1d(full_nll_1d, var_type='positive')

    else:
      # nD over [lambda_d (>0), thetas (ℝ)]
      def full_nll_nd(eff_ratio: float, vars_: list[float]) -> float:
        if not vars_ or len(vars_) != 1 + m:
          raise ValueError("Unexpected variables length in ratio nD")
        lambda_d = vars_[0]
        thetas = vars_[1:]
        if eff_ratio == 0 and numerator_count == 0:
          return 0
        if eff_ratio <= 0 or lambda_d <= 0:
          return float('inf')
        prod_factor = 1.0
        for a, t in zip(systematics, thetas, strict=True):
          if a <= 0:
            return float('inf')
          prod_factor *= a**t
        nominal_ratio = eff_ratio / prod_factor
        if nominal_ratio <= 0:
          return float('inf')
        lambda_n = nominal_ratio * lambda_d
        nll_n = -scipy.stats.poisson.logpmf(numerator_count, lambda_n)
        nll_d = -scipy.stats.poisson.logpmf(denominator_count, lambda_d)
        penalty = 0.5 * float(np.sum(np.square(thetas)))
        return float((nll_n + nll_d).item() + penalty)

      wrapped = cls._solve_nd(
        full_nll_nd,
        var_types=['positive'] + ['theta'] * m
      )

    if denominator_count <= 0:
      observed_ratio = float('inf')
    else:
      observed_ratio = numerator_count / denominator_count
    return cls(time, censored, wrapped, observed_ratio)

  @property
  def nominal(self) -> KaplanMeierPatient:
    """
    Returns the nominal Kaplan-Meier patient.
    """
    return KaplanMeierPatient(
      time=self.time,
      censored=self.censored,
      parameter=self.observed_parameter,
    )

class MINLPForKM:  # pylint: disable=too-many-public-methods, too-many-instance-attributes
  """
  Mixed Integer Nonlinear Programming for a point on the Kaplan-Meier curve.
  """
  __default_MIPGap = 1e-4
  __default_MIPGapAbs = 1e-7

  def __init__(  # pylint: disable=too-many-arguments
    self,
    all_patients: list[KaplanMeierPatientNLL],
    *,
    parameter_min: float,
    parameter_max: float,
    time_point: float,
    endpoint_epsilon: float = 1e-6,
    log_zero_epsilon: float = LOG_ZERO_EPSILON_DEFAULT, # New parameter for log arguments
    collapse_consecutive_deaths: bool = True,
  ):
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_max = parameter_max
    self.__time_point = time_point
    self.__endpoint_epsilon = endpoint_epsilon
    self.__log_zero_epsilon = log_zero_epsilon # Store the epsilon
    self.__collapse_consecutive_deaths = collapse_consecutive_deaths
    self.__expected_probability_constraint = None
    self.__binomial_penalty_constraint = None
    self.__patient_constraints_for_binomial_only = None
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
  @property
  def collapse_consecutive_deaths(self) -> bool:
    """
    Whether to collapse consecutive deaths with no intervening censoring.
    """
    return self.__collapse_consecutive_deaths
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
  def times_to_consider(self) -> npt.NDArray[np.float64]:
    """
    The unique sorted death times of all patients, plus the current time point.
    If collapse_consecutive_deaths is True, consecutive death times with no
    intervening censored patients are collapsed to reduce the number of 
    survival probability variables in the MINLP.
    """
    # Get all death times up to the time point
    death_mask = (~self.patient_censored) & (self.patient_times <= self.time_point)
    death_times = self.patient_times[death_mask]
    
    # Always include the time point itself
    all_times = list(death_times) + [self.time_point]
    unique_times = np.unique(all_times)
    
    if not self.collapse_consecutive_deaths:
      # Original behavior: return all unique times
      return np.sort(unique_times)
    
    # Collapse consecutive deaths logic
    # The key insight: we can collapse death times if no censoring occurs
    # between them (censoring at the same time doesn't count due to KM convention)
    collapsed_times = []
    
    i = 0
    while i < len(unique_times):
      current_time = unique_times[i]
      
      # Start a new group with the current time
      group_end = current_time
      
      # Look ahead to see if we can include more times in this group
      j = i + 1
      while j < len(unique_times):
        next_time = unique_times[j]
        
        # Check if there are any censored patients in the interval (group_end, next_time)
        # We use strict inequalities because:
        # - Censoring at group_end doesn't affect the next_time death (already processed)
        # - Censoring at next_time doesn't prevent the next_time death (death happens first)
        censored_between = np.any(
          self.patient_censored & 
          (self.patient_times > group_end) & 
          (self.patient_times < next_time)
        )
        
        # However, we need to check if there's censoring at group_end that would
        # affect the risk set for subsequent deaths
        if group_end != current_time:  # Not the first death in the group
          censored_at_group_end = np.any(
            self.patient_censored & (self.patient_times == group_end)
          )
          if censored_at_group_end:
            # Censoring at the end of the current group affects subsequent deaths
            break
        
        if censored_between:
          # Can't include next_time in this group due to intervening censoring
          break
        else:
          # No intervening censoring, extend the group
          group_end = next_time
          j += 1
      
      # Add the representative time for this group (use the last time in the group)
      collapsed_times.append(group_end)
      
      # Move to the next ungrouped time
      i = j
    
    return np.sort(np.array(collapsed_times))
  @functools.cached_property
  def n_times_to_consider(self) -> int:
    """
    The number of times to include in the calculation, which is
    the number of death times plus one if the time point is not
    itself a death time.
    """
    return len(self.times_to_consider)
  @functools.cached_property
  def _collapsed_time_groups(self) -> dict[float, list[float]]:
    """
    Map from representative time to list of original times in the collapsed group.
    Only used when collapse_consecutive_deaths is True.
    """
    if not self.collapse_consecutive_deaths:
      return {}
    
    # Get all death times up to the time point
    death_mask = (~self.patient_censored) & (self.patient_times <= self.time_point)
    death_times = self.patient_times[death_mask]
    all_times = list(death_times) + [self.time_point]
    unique_times = np.unique(all_times)
    
    groups = {}
    i = 0
    while i < len(unique_times):
      current_time = unique_times[i]
      group_end = current_time
      
      # Look ahead to see if we can include more times in this group
      j = i + 1
      while j < len(unique_times):
        next_time = unique_times[j]
        
        # Check if there are any censored patients in the interval (group_end, next_time)
        censored_between = np.any(
          self.patient_censored & 
          (self.patient_times > group_end) & 
          (self.patient_times < next_time)
        )
        
        # Check for censoring at group_end that would affect subsequent deaths
        if group_end != current_time:  # Not the first death in the group
          censored_at_group_end = np.any(
            self.patient_censored & (self.patient_times == group_end)
          )
          if censored_at_group_end:
            break
        
        if censored_between:
          break
        else:
          group_end = next_time
          j += 1
      
      # Store the group
      group_times = unique_times[i:j].tolist()
      groups[group_end] = group_times
      
      # Move to the next ungrouped time
      i = j
    
    return groups

  def patient_died(self, t) -> npt.NDArray[np.bool_]:
    """
    Returns a boolean array indicating which patients died at time t.
    If collapse_consecutive_deaths is True, for any t in a collapsed interval, anyone who died during the interval and before t is considered to have died at t.
    For any time t, a patient is considered to have died at t if:
    - Their time == t and not censored (no collapse)
    - If t is in a collapsed interval, their time is >= interval_start and < t, and not censored
    """
    if not self.collapse_consecutive_deaths:
      return (self.patient_times == t) & (~self.patient_censored)
    groups = self._collapsed_time_groups
    for group_end, group_times in groups.items():
      interval_start = min(group_times)
      if interval_start <= t <= group_end:
        return (self.patient_times >= interval_start) & (self.patient_times <= t) & (~self.patient_censored)
    return (self.patient_times == t) & (~self.patient_censored)

  def patient_still_at_risk(self, t) -> npt.NDArray[np.bool_]:
    """
    Returns a boolean array indicating which patients are still at risk at time t.
    If collapse_consecutive_deaths is True, anyone who died in the collapsed interval is considered at risk at any t in the interval.
    For any time t, a patient is at risk if:
    - Their time >= t (regardless of censored status)
    - If t is in a collapsed interval, their time is >= interval_start
    """
    if not self.collapse_consecutive_deaths:
      return self.patient_times >= t
    groups = self._collapsed_time_groups
    relevant_start_time = t
    for group_end, group_times in groups.items():
      interval_start = min(group_times)
      if interval_start <= t <= group_end:
        # Anyone who died in [interval_start, group_end) is at risk at any t in the interval
        relevant_start_time = interval_start
    return self.patient_times >= relevant_start_time

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

  @functools.cached_property
  def n_died_obs(self) -> npt.NDArray[np.int_]:
    """
    The number of patients who died at each time to consider using the observed parameters.
    """
    n_died = np.array([
      np.count_nonzero(
        self.patient_died(dt)
        & self.parameter_in_range
      )
      for dt in self.times_to_consider
    ], dtype=np.int_)
    return n_died
  @functools.cached_property
  def n_at_risk_obs(self) -> npt.NDArray[np.int_]:
    """
    The number of patients who were still at risk at each time to consider
    using the observed parameters.
    """
    n_at_risk = np.array([
      np.count_nonzero(
        self.patient_still_at_risk(dt)
        & self.parameter_in_range
      )
      for dt in self.times_to_consider
    ], dtype=np.int_)
    return n_at_risk
  @functools.cached_property
  def n_died_max(self) -> npt.NDArray[np.int_]:
    """
    The maximum number of patients who could have died at each death time.
    (regardless of parameter value)
    """
    n_died = np.array([
      np.count_nonzero(self.patient_died(dt))
      for dt in self.times_to_consider
    ], dtype=np.int_)
    return n_died
  @functools.cached_property
  def n_at_risk_max(self) -> npt.NDArray[np.int_]:
    """
    The maximum number of patients who could have been at risk at each death time.
    (regardless of parameter value)
    """
    n_at_risk = np.array([
      np.count_nonzero(self.patient_still_at_risk(dt))
      for dt in self.times_to_consider
    ], dtype=np.int_)
    return n_at_risk
  @functools.cached_property
  def n_censored_between_times_max(self) -> npt.NDArray[np.int_]:
    """
    The maximum number of patients who could have been censored between each pair of times.
    (regardless of parameter value)
    """
    n_censored = np.array([
      np.count_nonzero(
        (self.patient_times >= self.times_to_consider[i-1])
        & (self.patient_times < self.times_to_consider[i])
        & self.patient_censored
      )
      for i in range(1, self.n_times_to_consider)
    ], dtype=np.int_)
    return n_censored

  @classmethod
  def calculate_KM_probability(
    cls,
    n_at_risk: npt.NDArray[np.int_],
    n_died: npt.NDArray[np.int_],
  ) -> float:
    """
    Calculate the Kaplan-Meier probability at the time point.
    """
    if len(n_at_risk) != len(n_died):
      raise ValueError("At risk and died counts must have the same length")

    probability = 1.0
    for at_risk, died in zip(n_at_risk, n_died, strict=True):
      if at_risk > 0:
        probability *= (at_risk - died) / at_risk

    return probability

  @functools.cached_property
  def observed_KM_probability(self) -> float:
    """
    The observed Kaplan-Meier probability at the time point.
    This is calculated using the observed counts of patients who were censored or died.
    """
    return self.calculate_KM_probability(
      n_at_risk=self.n_at_risk_obs,
      n_died=self.n_died_obs,
    )

  @classmethod
  @functools.cache
  def calculate_possible_probabilities(
    cls,
    n_total_max: int,
    n_died_max: tuple[int],
    n_censored_between_times_max: tuple[int],
  ) -> set[float]:
    """
    Calculate possible probabilities based on the total number of patients
    who were censored or died in each group.
    The probabilities are calculated by iterating over all possible combinations
    of patients to be included or excluded.
    """
    if len(n_died_max) != len(n_censored_between_times_max)+1:
      raise ValueError("Died counts must be one more than censored counts")

    result = set()
    total_range = range(n_total_max)
    died_ranges = [range(nd + 1) for nd in n_died_max]
    censored_ranges = [range(nc + 1) for nc in n_censored_between_times_max]

    for total_count in total_range:
      for died_counts in itertools.product(*died_ranges):
        for censored_counts in itertools.product(*censored_ranges):
          at_risk_counts = [total_count]
          for i in range(1, len(n_died_max)):
            at_risk = (
              at_risk_counts[i-1]
              - died_counts[i-1]
              - censored_counts[i-1]
            )
            at_risk_counts.append(at_risk)
          if any(ar < 0 for ar in at_risk_counts):
            continue
          if total_count < sum(died_counts) + sum(censored_counts):
            continue
          km_probability = cls.calculate_KM_probability(
            n_at_risk=np.array(at_risk_counts, dtype=np.int_),
            n_died=np.array(died_counts, dtype=np.int_),
          )
          if not 0 <= km_probability <= 1:
            raise RuntimeError(
              f"Calculated KM probability {km_probability} is out of range [0,1]"
              f"for counts: total={total_count}, died={died_counts}, censored={censored_counts}"
            )
          result.add(km_probability)
    return result

  @functools.cached_property
  def possible_probabilities(self) -> set[float]:
    """
    Calculate the possible probabilities based on the total number of patients
    and the total number who were censored or died in each group.
    """
    return self.calculate_possible_probabilities(
      n_total_max=self.n_patients,
      n_died_max=tuple(self.n_died_max),
      n_censored_between_times_max=tuple(self.n_censored_between_times_max),
    )

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
    return n_choose_d_term_table(n_patients=self.n_patients)

  def add_counter_variables_and_constraints(
    self,
    model: gp.Model,
    a: gp.tupledict[int, gp.Var],
  ):
    """
    Add counter variables for the total number of patients,
    the number of patients who were censored or died or were at risk
    in each group, and the number of patients who are still alive.
    """
    n_total = model.addVar(vtype=GRB.INTEGER, name="n_total")
    model.addConstr(
      n_total == gp.quicksum(a[j] for j in range(self.n_patients)),
      name="n_total_constraint",
    )

    d = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.INTEGER,
      name="d",
    )
    r = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.INTEGER,
      name="r",
    )
    s = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.INTEGER,
      name="s",
    )

    # Constraints to link to totals
    for i, dt in enumerate(self.times_to_consider):
      model.addConstr(
        d[i] == gp.quicksum(
          a[j] for j in range(self.n_patients) if self.patient_died(dt)[j]
        ),
        name=f"d_{i}_definition",
      )
      model.addConstr(
        r[i] == gp.quicksum(
          a[j] for j in range(self.n_patients) if self.patient_still_at_risk(dt)[j]
        ),
        name=f"r_{i}_definition",
      )
      model.addConstr(
        s[i] == r[i] - d[i],
        name=f"s_{i}_definition",
      )

    return (
      d,
      r,
      s,
    )

  def add_kaplan_meier_probability_variables_and_constraints(
    self,
    model: gp.Model,
    r: gp.tupledict[int, gp.Var],
    s: gp.tupledict[int, gp.Var],
  ):
    """
    Add variables and constraints to calculate the Kaplan-Meier probability
    directly within the Gurobi model using logarithmic transformations.
    Handles the case where r for a group is 0.
    """
    # Variables for log of counts
    log_r_vars = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.CONTINUOUS,
      name="log_r",
      lb=-GRB.INFINITY,
      ub=np.log(self.n_patients + self.__log_zero_epsilon), # Max possible log(count)
    )
    log_n_survived_vars = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.CONTINUOUS,
      name="log_n_survived",
      lb=-GRB.INFINITY,
      ub=np.log(self.n_patients + self.__log_zero_epsilon), # Max possible log(count)
    )

    # Helper variables for log arguments (r + epsilon, n_survived + epsilon)
    r_plus_epsilon = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.CONTINUOUS,
      name="r_plus_epsilon",
      lb=self.__log_zero_epsilon, # Ensure strictly positive
    )
    n_survived_plus_epsilon = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.CONTINUOUS,
      name="n_survived_plus_epsilon",
      lb=self.__log_zero_epsilon, # Ensure strictly positive
    )

    # Constraints to link original counts to epsilon-added variables
    for i in range(self.n_times_to_consider):
      model.addConstr(
        r_plus_epsilon[i] == r[i] + self.__log_zero_epsilon,
        name=f"r_plus_epsilon_constr_{i}"
      )
      model.addConstr(
        n_survived_plus_epsilon[i] == s[i] + self.__log_zero_epsilon,
        name=f"n_survived_plus_epsilon_constr_{i}"
      )

      # Link count variables to their log counterparts using GenConstrLog
      model.addGenConstrLog(
        r_plus_epsilon[i],
        log_r_vars[i],
        name=f"log_r_constr_{i}"
      )
      model.addGenConstrLog(
        n_survived_plus_epsilon[i],
        log_n_survived_vars[i],
        name=f"log_n_survived_constr_{i}"
      )

    # Binary indicator for whether r for a group is zero
    is_r_zero = model.addVars(
        self.n_times_to_consider,
        vtype=GRB.BINARY,
        name="is_r_zero"
    )

    # Link is_r_zero to r using indicator constraint
    for i in range(self.n_times_to_consider):
      # If r[i] == 0, then is_r_zero[i] must be 1
      # If r[i] > 0, then is_r_zero[i] must be 0
      model.addGenConstrIndicator(
        is_r_zero[i], True, r[i], GRB.EQUAL, 0,
        name=f"is_r_zero_indicator_{i}"
      )

    # Kaplan-Meier log probability for each group term
    # This term will be 0 if r[i] is 0
    km_log_probability_per_group_terms = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.CONTINUOUS,
      name="km_log_prob_group_term",
      lb=-GRB.INFINITY,
      ub=0, # Log of a probability is always <= 0
    )

    # Use indicator constraints to set km_log_probability_per_group_terms[i]
    for i in range(self.n_times_to_consider):
      # If is_r_zero[i] is 0 (i.e., r[i] > 0)
      model.addGenConstrIndicator(
        is_r_zero[i], False,
        km_log_probability_per_group_terms[i] - (log_n_survived_vars[i] - log_r_vars[i]),
        GRB.EQUAL,
        0,
        name=f"km_log_prob_group_active_{i}"
      )
      # If is_r_zero[i] is 1 (i.e., r[i] == 0)
      model.addGenConstrIndicator(
        is_r_zero[i], True,
        km_log_probability_per_group_terms[i],
        GRB.EQUAL,
        0.0,
        name=f"km_log_prob_group_zero_r_{i}"
      )

    # Total Kaplan-Meier log probability: sum of log probabilities per group
    km_log_probability_total = model.addVar(
      vtype=GRB.CONTINUOUS,
      name="km_log_probability_total",
      lb=-GRB.INFINITY,
      ub=0,
    )
    model.addConstr(
      km_log_probability_total == km_log_probability_per_group_terms.sum(),
      name="km_log_probability_total_def"
    )

    # Kaplan-Meier probability variable (linear scale)
    km_probability_var = model.addVar(
      vtype=GRB.CONTINUOUS,
      name="km_probability",
      lb=0,
      ub=1,
    )
    # Link log probability to linear probability using GenConstrExp
    model.addGenConstrExp(
      km_log_probability_total,
      km_probability_var,
      name="exp_km_probability"
    )

    return km_probability_var

  def add_binomial_penalty(  # pylint: disable=too-many-locals, too-many-statements
    self,
    model: gp.Model,
    r: gp.tupledict[int, gp.Var],
    d: gp.tupledict[int, gp.Var],
    s: gp.tupledict[int, gp.Var],
  ):
    """
    Add the binomial penalty to the model.
    This penalty is based on the expected survival probability
    and the number of patients who died and who were at risk in each group.

    There's a separate binomial term for each group
    To complicate things, we only know the overall expected survival probability,
    not the probability of survival in each group.
    So we need to profile those.
    """

    #p_i = probability of dying at death time i
    p_died = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.CONTINUOUS,
      name="p_died",
      lb=0,
      ub=1,
    )
    p_survived = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.CONTINUOUS,
      name="p_survived",
      lb=0,
      ub=1,
    )
    divide = self.n_times_to_consider * 2
    log_p_bounds = np.array([
      np.log(self.__endpoint_epsilon / divide),
      np.log(1 - self.__endpoint_epsilon / divide),
    ])
    log_p_died = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.CONTINUOUS,
      name="log_p_died",
      lb=log_p_bounds[0],
      ub=log_p_bounds[1],
    )
    log_p_survived = model.addVars(
      self.n_times_to_consider,
      vtype=GRB.CONTINUOUS,
      name="log_p_survived",
      lb=log_p_bounds[0],
      ub=log_p_bounds[1],
    )
    for i in range(self.n_times_to_consider):
      model.addGenConstrExp(log_p_died[i], p_died[i], name=f"log_p_died_constr_{i}")
      model.addGenConstrExp(log_p_survived[i], p_survived[i], name=f"log_p_survived_constr_{i}")
      model.addConstr(
        p_died[i] + p_survived[i] == 1,
        name=f"p_died_plus_p_survived_{i}"
      )

    #product of survival probabilities = the overall expected probability
    #we will set the expected probability via a constraint in update_model_with_expected_probability
    expected_probability_var = model.addVar(
      vtype=GRB.CONTINUOUS,
      name="expected_probability",
      lb=0,
      ub=1,
    )
    log_expected_probability = model.addVar(
      vtype=GRB.CONTINUOUS,
      name="log_expected_probability",
      lb=np.log(self.__endpoint_epsilon),
      ub=np.log(1 - self.__endpoint_epsilon),
    )
    model.addGenConstrExp(
      log_expected_probability,
      expected_probability_var,
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
    n_choose_d_table = self.n_choose_d_term_table
    all_n_choose_d_indicator_vars = model.addVars(
      self.n_times_to_consider, len(n_choose_d_table),
      vtype=GRB.BINARY,
      name="n_choose_d_indicator",
    )
    n_choose_d_indicator_vars = {
      (i, n, d): all_n_choose_d_indicator_vars[i, idx]
      for i in range(self.n_times_to_consider)
      for idx, (n, d) in enumerate(n_choose_d_table.keys())
    }
    n_died_indicator_vars = model.addVars(
      self.n_times_to_consider, int(max(self.n_died_max)+1),
      vtype=GRB.BINARY,
      name="n_died_indicator",
    )
    n_survived_indicator_vars = model.addVars(
      self.n_times_to_consider, self.n_patients + 1,
      #could probably have somewhat fewer of these: the maximum is n_patients,
      #but the minimum is not 0.
      vtype=GRB.BINARY,
      name="n_survived_indicator",
    )
    binomial_terms = []
    for i in range(self.n_times_to_consider):
      for (r_value, d_value), penalty in n_choose_d_table.items():
        indicator = n_choose_d_indicator_vars[i, r_value, d_value]
        model.addGenConstrIndicator(
          indicator,
          True,
          r[i],
          GRB.EQUAL,
          r_value,
          name=f"n_choose_d_indicator_r_{i}_{r_value}_{d_value}",
        )
        model.addGenConstrIndicator(
          indicator,
          True,
          d[i],
          GRB.EQUAL,
          d_value,
          name=f"n_choose_d_indicator_d_{i}_{r_value}_{d_value}",
        )
        if (
          r_value > self.n_at_risk_max[i]
          or d_value > self.n_died_max[i]
          or d_value > r_value
        ):
          model.addConstr(
            indicator == 0,
            name=f"n_choose_d_indicator_impossible_{i}_{r_value}_{d_value}",
          )
        binomial_terms.append(-penalty * indicator)
      # Ensure that exactly one n_choose_d_indicator is selected for each death time
      indicators = [
        all_n_choose_d_indicator_vars[i, idx]
        for idx in range(len(n_choose_d_table))
      ]
      model.addConstr(
        gp.quicksum(indicators) == 1,
        name=f"one_n_choose_d_indicator_per_death_time_{i}",
      )

      for d_value in range(max(self.n_died_max) + 1):
        model.addGenConstrIndicator(
          n_died_indicator_vars[i, d_value],
          True,
          d[i],
          GRB.EQUAL,
          d_value,
          name=f"n_died_indicator_{i}_{d_value}",
        )
        if d_value > self.n_died_max[i]:
          model.addConstr(
            n_died_indicator_vars[i, d_value] == 0,
            name=f"n_died_indicator_impossible_{i}_{d_value}",
          )
        binomial_terms.append(
          -d_value * log_p_died[i] * n_died_indicator_vars[i, d_value]
        )
      # Ensure that exactly one n_died_indicator is selected for each group
      model.addConstr(
        gp.quicksum(
          n_died_indicator_vars[i, d_value]
          for d_value in range(max(self.n_died_max)+1)
        ) == 1,
        name=f"one_n_died_indicator_per_death_time_{i}",
      )

      for s_value in range(self.n_patients + 1):
        model.addGenConstrIndicator(
          n_survived_indicator_vars[i, s_value],
          True,
          s[i],
          GRB.EQUAL,
          s_value,
          name=f"n_survived_indicator_{i}_{s_value}",
        )
        if s_value > self.n_at_risk_max[i]:
          model.addConstr(
            n_survived_indicator_vars[i, s_value] == 0,
            name=f"n_survived_indicator_impossible_{i}_{s_value}",
          )
        binomial_terms.append(
          -s_value * log_p_survived[i] * n_survived_indicator_vars[i, s_value]
        )

      # Ensure that exactly one n_survived_indicator is selected for each group
      model.addConstr(
        gp.quicksum(
          n_survived_indicator_vars[i, s_value]
          for s_value in range(self.n_patients + 1)
        ) == 1,
        name=f"one_n_survived_indicator_per_death_time_{i}",
      )

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
      0.0,
      name="binomial_penalty_inactive",
    )

    #big M constraint to ensure binomial penalty is only used when the indicator is set
    max_penalty_term = max(
      abs(penalty) for penalty in self.n_choose_d_term_table.values()
    )
    max_d = max(self.n_died_max)
    max_s = self.n_patients
    max_log_p = max(np.abs(log_p_bounds))
    safety_factor = 2
    big_M = safety_factor * self.n_times_to_consider * (
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

    return binom_penalty, expected_probability_var, use_binomial_penalty_indicator

  def add_patient_wise_penalty(
    self,
    model: gp.Model,
    a: gp.tupledict[int, gp.Var],
  ):
    """
    Add the patient-wise penalty to the Gurobi model.
    This penalty is based on the negative log-likelihood of the patient's observed parameter
    being within the specified range.
    """
    # Patient-wise penalties
    patient_penalties = []
    for j in range(self.n_patients):
      if np.isfinite(self.nll_penalty_for_patient_in_range[j]):
        penalty = self.nll_penalty_for_patient_in_range[j] * a[j]
        if self.nll_penalty_for_patient_in_range[j] < 0:
          # If the penalty is negative, it means the patient is nominally within the range
          # We want the penalty to be 0 when all the patients are at their nominal values
          penalty -= self.nll_penalty_for_patient_in_range[j]
        patient_penalties.append(penalty)
      elif np.isneginf(self.nll_penalty_for_patient_in_range[j]):
        #the patient must be selected, so we add a constraint
        model.addConstr(
          a[j] == 1,
          name=f"patient_{j}_must_be_selected",
        )
      elif np.isposinf(self.nll_penalty_for_patient_in_range[j]):
        #the patient must not be selected, so we add a constraint
        model.addConstr(
          a[j] == 0,
          name=f"patient_{j}_must_not_be_selected",
        )
      else:
        raise ValueError(
          f"Unexpected NLL penalty for patient {j}: "
          f"{self.nll_penalty_for_patient_in_range[j]}"
        )

    patient_penalty = gp.quicksum(patient_penalties)
    return patient_penalty

  def _make_gurobi_model(self):  #pylint: disable=too-many-locals
    """
    Create the Gurobi model for the MINLP.
    This method constructs the model with decision variables, constraints,
    and the objective function.  It does NOT include the constraint for the
    expected probability, which is added in update_model_with_expected_probability.
    """
    model = gp.Model("Kaplan-Meier MINLP")

    # Binary decision variables: a[j] = 1 if patient j is within the parameter range
    a = model.addVars(self.n_patients, vtype=GRB.BINARY, name="a")

    (
      d,
      r,
      s,
    ) = self.add_counter_variables_and_constraints(
      model=model,
      a=a,
    )

    # Add Kaplan-Meier probability variables and constraints (replaces trajectory logic)
    km_probability_var = self.add_kaplan_meier_probability_variables_and_constraints(
      model=model,
      r=r,
      s=s,
    )

    (
      binom_penalty,
      expected_probability_var,
      use_binomial_penalty_indicator,
    ) = self.add_binomial_penalty(
      model=model,
      r=r,
      d=d,
      s=s,
    )

    patient_penalty = self.add_patient_wise_penalty(
      model=model,
      a=a,
    )

    # Objective: minimize total penalty
    model.setObjective(
      2 * (binom_penalty + patient_penalty),
      GRB.MINIMIZE,
    )
    model.update()

    return (
      model,
      a,
      km_probability_var,
      expected_probability_var,
      use_binomial_penalty_indicator,
    )

  @functools.cached_property
  def gurobi_model(self):
    """
    Create the Gurobi model for the MINLP.
    This is a cached property to avoid recreating the model multiple times.
    """
    return self._make_gurobi_model()

  def update_model_with_expected_probability( # pylint: disable=too-many-arguments, too-many-branches
    self,
    *,
    model: gp.Model,
    expected_probability: float | None,
    patient_wise_only: bool,
    binomial_only: bool,
    a: gp.tupledict[int, gp.Var],
    km_probability_var: gp.Var,
    use_binomial_penalty_indicator: gp.Var,
    expected_probability_var: gp.Var,
  ):
    """
    Update the Gurobi model with the expected probability constraint.
    This is the only thing that changes between runs of the MINLP.
    """
    #drop the previous constraints if they exist
    if self.__expected_probability_constraint is not None:
      model.remove(self.__expected_probability_constraint)
      self.__expected_probability_constraint = None
    if self.__binomial_penalty_constraint is not None:
      model.remove(self.__binomial_penalty_constraint)
      self.__binomial_penalty_constraint = None
    if self.__patient_constraints_for_binomial_only is not None:
      for constr in self.__patient_constraints_for_binomial_only:
        model.remove(constr)
      self.__patient_constraints_for_binomial_only = None

    if not patient_wise_only:
      # ---------------------------
      # Binomial penalty is active, constrain its expected probability
      # ---------------------------
      self.__binomial_penalty_constraint = model.addConstr(
        use_binomial_penalty_indicator == 1,
        name="use_binomial_penalty"
      )
      if expected_probability is not None:
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

      # Constrain the KM probability based on the expected_probability
      # If expected > observed, then KM_prob >= expected_probability
      # If expected < observed, then KM_prob <= expected_probability
      # If expected == observed or is None, then KM_prob is unconstrained
      if expected_probability is None:
        pass
      elif expected_probability > self.observed_KM_probability:
        self.__expected_probability_constraint = model.addConstr(
          km_probability_var >= expected_probability - self.__endpoint_epsilon,
          name="km_prob_ge_expected"
        )
      elif expected_probability < self.observed_KM_probability:
        self.__expected_probability_constraint = model.addConstr(
          km_probability_var <= expected_probability + self.__endpoint_epsilon,
          name="km_prob_le_expected"
        )
      else: # expected_probability == self.observed_KM_probability
        assert expected_probability == self.observed_KM_probability

    if binomial_only:
      self.__patient_constraints_for_binomial_only = []
      for j in range(self.n_patients):
        if self.parameter_in_range[j]:
          assert self.nll_penalty_for_patient_in_range[j] <= 0
          #the patient must be selected
          self.__patient_constraints_for_binomial_only.append(
            model.addConstr(
              a[j] == 1,
              name=f"patient_{j}_must_be_selected_binomial_only",
            )
          )
        else:
          assert self.nll_penalty_for_patient_in_range[j] >= 0
          #the patient must not be selected
          self.__patient_constraints_for_binomial_only.append(
            model.addConstr(
              a[j] == 0,
              name=f"patient_{j}_must_not_be_selected_binomial_only",
            )
          )

    model.update()

  def _set_gurobi_params(self, model: gp.Model, params: dict):
    """
    Helper function to set multiple Gurobi parameters from a dictionary.
    """
    for param, value in params.items():
      if value is not None:
        model.setParam(param, value)

  def _optimize_with_fallbacks(
    self,
    model: gp.Model,
    initial_params: dict,
    fallback_strategies: list[tuple[dict, str]],
    verbose: bool,
  ):
    """
    Attempts to optimize the Gurobi model, applying fallback strategies
    if the initial optimization is suboptimal.

    Args:
        model: The Gurobi model to optimize.
        initial_params: A dictionary of initial Gurobi parameters to apply.
        fallback_strategies: A list of tuples, where each tuple contains:
            - A dictionary of Gurobi parameters to apply for the fallback.
            - A string description of the fallback strategy.
        verbose: If True, print detailed optimization progress.

    Returns:
        The Gurobi model after optimization.
    """
    # Apply initial parameters
    self._set_gurobi_params(model, initial_params)

    if verbose:
      print("Attempting initial optimization...")
    model.optimize()

    # Check for suboptimal status and apply fallbacks
    if model.status == GRB.SUBOPTIMAL:
      for i, (fallback_params, description) in enumerate(fallback_strategies):
        if verbose:
          print(f"Model returned suboptimal solution. Applying fallback {i+1}: {description}")
          print(f"  New parameters: {fallback_params}")
        self._set_gurobi_params(model, fallback_params)
        model.optimize()
        if model.status == GRB.OPTIMAL:
          if verbose:
            print(f"Fallback {i+1} successful. Model is now optimal.")
          break
    return model

  def run_MINLP( # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
    self,
    expected_probability: float | None,
    *,
    verbose=False,
    print_progress=False,
    binomial_only=False,
    patient_wise_only=False,
    MIPGap: float | None = None,
    MIPGapAbs: float | None = None,
    TimeLimit: float | None = None,
    Threads: int | None = None,
    MIPFocus: int | None = None,
    LogFile: os.PathLike | None = None,
  ):
    """
    Run the MINLP for the given time point.
    """
    if print_progress or verbose:
      print(
        "Running MINLP for expected probability ", expected_probability,
        " at time point ", self.time_point, " at time ", datetime.datetime.now()
      )
    if expected_probability is not None:
      if not patient_wise_only and (expected_probability <= 0 or expected_probability >= 1):
        raise ValueError(f"expected_probability={expected_probability} must be in (0, 1) or None")
      if expected_probability < 0 or expected_probability > 1:
        raise ValueError(f"expected_probability={expected_probability} must be in [0, 1] or None")
    if binomial_only and patient_wise_only:
      raise ValueError("binomial_only and patient_wise_only cannot both be True")

    if MIPGap is None:
      MIPGap = self.__default_MIPGap
    if MIPGapAbs is None:
      MIPGapAbs = self.__default_MIPGapAbs

    nll_penalty_for_patient_in_range = self.nll_penalty_for_patient_in_range

    (
      model,
      a,
      km_probability_var,
      expected_probability_var,
      use_binomial_penalty_indicator,
    ) = self.gurobi_model
    self.update_model_with_expected_probability(
      model=model,
      a=a,
      km_probability_var=km_probability_var,
      expected_probability=expected_probability,
      patient_wise_only=patient_wise_only,
      binomial_only=binomial_only,
      expected_probability_var=expected_probability_var,
      use_binomial_penalty_indicator=use_binomial_penalty_indicator,
    )

    # Initial Gurobi parameters
    initial_gurobi_params = {
      'OutputFlag': 1 if verbose else 0,
      'DisplayInterval': 1,
      'MIPGap': MIPGap,
      'MIPGapAbs': MIPGapAbs,
      'NonConvex': 2,
      'NumericFocus': 3 if patient_wise_only else 0,
      'Seed': 123456,
      'TimeLimit': TimeLimit,
      'Threads': Threads,
      'MIPFocus': MIPFocus,
      'FuncPieces': 1000,
      'FuncPieceRatio': 0.5,
    }
    if LogFile is not None:
      initial_gurobi_params['LogFile'] = os.fspath(LogFile)

    # Define fallback strategies
    fallback_strategies = []

    # Fallback 1: Try MIPFocus 2 if initial was suboptimal and not already 2
    if MIPFocus != 2: # Only add this fallback if MIPFocus wasn't already 2
      fallback_strategies.append(
        ({'MIPFocus': 2}, "MIPFocus set to 2 (optimality focus)")
      )

    # Fallback 2: Increase TimeLimit if it was set and still suboptimal
    if TimeLimit is not None:
      fallback_strategies.append(
        ({'TimeLimit': TimeLimit * 1.5}, "Increased TimeLimit by 50%")
      )

    # Fallback 3: Increase FuncPieces
    current_func_pieces = initial_gurobi_params.get('FuncPieces', 0)
    if current_func_pieces < 2000: # Arbitrary upper limit to prevent excessive FuncPieces
      fallback_strategies.append(
        ({'FuncPieces': max(2000, int(current_func_pieces * 2))}, "Increased FuncPieces (doubled)")
      )
    if current_func_pieces < 5000:
      fallback_strategies.append(
        (
          {'FuncPieces': max(5000, int(current_func_pieces * 2.5)), 'FuncPieceRatio': 0.75},
          "Increased FuncPieces and adjusted FuncPieceRatio"
        )
      )

    # Fallback 4: Try different NumericFocus
    fallback_strategies.append(
      ({'NumericFocus': 2}, "Changed NumericFocus to 2 (accuracy)")
    )

    # Fallback 5: Experiment with Cuts (more aggressive)
    fallback_strategies.append(
      ({'Cuts': 2}, "Aggressive cut generation")
    )

    # Fallback 6: Experiment with Heuristics (less aggressive)
    fallback_strategies.append(
      ({'Heuristics': 0.5}, "Less aggressive heuristics")
    )

    #Fallback 7: Tighten barrier convergence tolerance
    fallback_strategies.append(
      ({'BarConvTol': 1e-8}, "Tightened barrier convergence tolerance")
    )

    #Fallback 8: Tighten feasibility tolerance
    fallback_strategies.append(
      ({'FeasibilityTol': 1e-8}, "Tightened feasibility tolerance")
    )

    # Fallback 9: Tighten optimality tolerance
    fallback_strategies.append(
      ({'OptimalityTol': 1e-8}, "Tightened optimality tolerance")
    )

    # Fallback 10: Use barrier method
    fallback_strategies.append(
      ({'Method': 2}, "Switched to barrier method")
    )

    #Fallback 11: Avoid switching back to simplex
    fallback_strategies.append(
      ({'CrossOver': 0}, "Avoided switching back to simplex method")
    )

    # Fallback 12: Barrier at all nodes
    fallback_strategies.append(
      ({'NodeMethod': 2}, "Used barrier method at all nodes")
    )

    # Last fallback: turn verbose output on
    # This is not going to work, but it will help us debug the issue
    if not verbose:
      fallback_strategies.append(
        (
          {
            'OutputFlag': 1,
            'DisplayInterval': 1,
            'InfUnbdInfo': 1,
          },
          "Turned verbose output on"
        )
      )

    # Optimize with fallbacks
    model = self._optimize_with_fallbacks(
        model, initial_gurobi_params, fallback_strategies, verbose
    )

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
          p_survived=[np.nan] * self.n_times_to_consider,
          binomial_2NLL=np.inf,
          patient_2NLL=np.inf,
          patient_penalties=nll_penalty_for_patient_in_range,
          selected=[],
          model=model,
          km_probability=np.nan,
        )
      raise RuntimeError(
        f"Model optimization failed with status {model.status}. "
        "This may indicate an issue with the MINLP formulation or the input data."
      )

    assert all(var is not None for var in a)
    selected = [j for j in range(self.n_patients) if a[j].X > 0.5]
    n_total_val = sum(selected)
    n_alive_val = sum(
      1 for j in selected
      if self.patient_still_at_risk(self.time_point)[j]
      and not self.patient_died(self.time_point)[j]
    )

    patient_penalty_val = sum(
      nll_penalty_for_patient_in_range[j] * (
        a[j].X
        - (1 if nll_penalty_for_patient_in_range[j] < 0 else 0)
      ) for j in range(self.n_patients)
      if np.isfinite(nll_penalty_for_patient_in_range[j])
    )
    binom_penalty_var = model.getVarByName("binom_penalty")
    assert binom_penalty_var is not None
    binomial_penalty_val = binom_penalty_var.X
    p_survived_val = []
    for i in range(self.n_times_to_consider):
      var = model.getVarByName(f"p_survived[{i}]")
      assert var is not None
      p_survived_val.append(var.X)
    if verbose:
      print("Selected patients:", selected)
      print("n_total:          ", int(n_total_val))
      print("Binomial penalty: ", 2*binomial_penalty_val)
      print("Patient penalty:  ", 2*patient_penalty_val)
      print("Total penalty:    ", model.ObjVal)

    return scipy.optimize.OptimizeResult(
      x=model.ObjVal,
      success=model.status == GRB.OPTIMAL,
      n_total=n_total_val,
      n_alive=n_alive_val,
      p_survived=p_survived_val,
      binomial_2NLL=2*binomial_penalty_val,
      patient_2NLL=2*patient_penalty_val,
      patient_penalties=nll_penalty_for_patient_in_range,
      selected=selected,
      model=model,
      km_probability=km_probability_var.X,
    )
