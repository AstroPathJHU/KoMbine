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
    log_zero_epsilon: float = 1e-10, # New parameter for log arguments
  ):
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_max = parameter_max
    self.__time_point = time_point
    self.__endpoint_epsilon = endpoint_epsilon
    self.__log_zero_epsilon = log_zero_epsilon # Store the epsilon
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
  def calculate_possible_probabilities(
    cls,
    n_total_patients: int,
    n_censored_in_group: tuple[int, ...],
    n_died_in_group: tuple[int, ...],
    verbose=False,
  ) -> set[float]:
    """
    Calculate possible probabilities based on the total number of patients and the
    total number who were censored or died in each group.
    The probabilities are calculated by iterating over all possible combinations
    of patients to be included or excluded.

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
    result = set()
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
        f"Calculating probabilities for {n_trajectories} trajectories for "
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
          result.add(expected_trajectory_probability)
    return result

  @functools.cached_property
  def possible_probabilities(self) -> set[float]:
    """
    Calculate the possible probabilities based on the total number of patients
    and the total number who were censored or died in each group.
    """
    return self.calculate_possible_probabilities(
      n_total_patients=self.n_patients,
      n_censored_in_group=tuple(self.n_censored_in_group_total),
      n_died_in_group=tuple(self.n_died_in_group_total),
      verbose=False,
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

    # Integer vars to count totals
    n_censored_in_group = model.addVars(
      self.n_groups,
      vtype=GRB.INTEGER,
      name="n_censored_in_group",
    )
    d = model.addVars(
      self.n_groups,
      vtype=GRB.INTEGER,
      name="d",
    )
    r = model.addVars(
      self.n_groups,
      vtype=GRB.INTEGER,
      name="r",
    )
    n_survived_in_group = model.addVars(
      self.n_groups,
      vtype=GRB.INTEGER,
      name="n_survived_in_group",
    )

    # Constraints to link to totals
    for i in range(self.n_groups):
      model.addConstr(
        n_censored_in_group[i] == gp.quicksum(
          a[j] for j in range(self.n_patients) if self.censored_in_group[i][j]
        ),
        name=f"n_censored_in_group_{i}",
      )
      model.addConstr(
        d[i] == gp.quicksum(
          a[j] for j in range(self.n_patients) if self.died_in_group[i][j]
        ),
        name=f"d_{i}",
      )
      if i == 0:
        model.addConstr(
          r[i] == n_total - n_censored_in_group[i],
          name=f"r_{i}"
        )
      else:
        model.addConstr(
          r[i]
            == r[i - 1] - d[i - 1] - n_censored_in_group[i],
          name=f"r_{i}",
        )
      model.addConstr(
        n_survived_in_group[i] == r[i] - d[i],
        name=f"n_survived_in_group_{i}",
      )
    n_alive = model.addVar(vtype=GRB.INTEGER, name="n_alive")
    model.addConstr(
      n_alive == gp.quicksum(
        a[j] for j in range(self.n_patients) if self.patient_alive[j]
      ),
      name="n_alive_constraint",
    )

    return (
      n_total,
      n_censored_in_group,
      d,
      r,
      n_survived_in_group,
      n_alive,
    )

  def add_kaplan_meier_probability_variables_and_constraints(
    self,
    model: gp.Model,
    r: gp.tupledict[int, gp.Var],
    n_survived_in_group: gp.tupledict[int, gp.Var],
  ):
    """
    Add variables and constraints to calculate the Kaplan-Meier probability
    directly within the Gurobi model using logarithmic transformations.
    Handles the case where r for a group is 0.
    """
    # Variables for log of counts
    log_r_vars = model.addVars(
      self.n_groups,
      vtype=GRB.CONTINUOUS,
      name="log_r",
      lb=-GRB.INFINITY,
      ub=np.log(self.n_patients + self.__log_zero_epsilon), # Max possible log(count)
    )
    log_n_survived_vars = model.addVars(
      self.n_groups,
      vtype=GRB.CONTINUOUS,
      name="log_n_survived",
      lb=-GRB.INFINITY,
      ub=np.log(self.n_patients + self.__log_zero_epsilon), # Max possible log(count)
    )

    # Helper variables for log arguments (r + epsilon, n_survived + epsilon)
    r_plus_epsilon = model.addVars(
      self.n_groups,
      vtype=GRB.CONTINUOUS,
      name="r_plus_epsilon",
      lb=self.__log_zero_epsilon, # Ensure strictly positive
    )
    n_survived_plus_epsilon = model.addVars(
      self.n_groups,
      vtype=GRB.CONTINUOUS,
      name="n_survived_plus_epsilon",
      lb=self.__log_zero_epsilon, # Ensure strictly positive
    )

    # Constraints to link original counts to epsilon-added variables
    for i in range(self.n_groups):
      model.addConstr(
        r_plus_epsilon[i] == r[i] + self.__log_zero_epsilon,
        name=f"r_plus_epsilon_constr_{i}"
      )
      model.addConstr(
        n_survived_plus_epsilon[i] == n_survived_in_group[i] + self.__log_zero_epsilon,
        name=f"n_survived_plus_epsilon_constr_{i}"
      )

    # Link count variables to their log counterparts using GenConstrLog
    for i in range(self.n_groups):
      # Use the new helper variables as arguments to GenConstrLog
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
        self.n_groups,
        vtype=GRB.BINARY,
        name="is_r_zero"
    )

    # Link is_r_zero to r using indicator constraint
    for i in range(self.n_groups):
      # If r[i] == 0, then is_r_zero[i] must be 1
      # If r[i] > 0, then is_r_zero[i] must be 0
      model.addGenConstrIndicator(
        is_r_zero[i], True, r[i], GRB.EQUAL, 0,
        name=f"is_r_zero_indicator_{i}"
      )

    # Kaplan-Meier log probability for each group term
    # This term will be 0 if r[i] is 0
    km_log_probability_per_group_terms = model.addVars(
      self.n_groups,
      vtype=GRB.CONTINUOUS,
      name="km_log_prob_group_term",
      lb=-GRB.INFINITY,
      ub=0, # Log of a probability is always <= 0
    )

    # Use indicator constraints to set km_log_probability_per_group_terms[i]
    for i in range(self.n_groups):
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
    n_survived_in_group: gp.tupledict[int, gp.Var],
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
    log_p_bounds = np.array([
      np.log(self.__endpoint_epsilon / self.n_groups / 2),
      np.log(1 - self.__endpoint_epsilon / self.n_groups / 2),
    ])
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
    n_choose_d_indicator_vars = model.addVars(
      self.n_groups * len(n_choose_d_table),
      vtype=GRB.BINARY,
      name="n_choose_d_indicator",
    )
    binomial_terms = []
    n_choose_d_indicator_vars_by_group = collections.defaultdict(list)
    for indicator_var_idx, (group_idx, ((n, d), penalty)) in enumerate(
      itertools.product(
        range(self.n_groups),
        n_choose_d_table.items(),
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
        r[group_idx],
        GRB.EQUAL,
        n,
        name=f"n_choose_d_indicator_r_{group_idx}_{n}",
      )
      model.addGenConstrIndicator(
        n_choose_d_indicator_vars[indicator_var_idx],
        True,
        d[group_idx],
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
          d[group_idx],
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
      _,
      _,
      d,
      r,
      n_survived_in_group,
      _,
    ) = self.add_counter_variables_and_constraints(
      model=model,
      a=a,
    )

    # Add Kaplan-Meier probability variables and constraints (replaces trajectory logic)
    km_probability_var = self.add_kaplan_meier_probability_variables_and_constraints(
      model=model,
      r=r,
      n_survived_in_group=n_survived_in_group,
    )

    (
      binom_penalty,
      expected_probability_var,
      use_binomial_penalty_indicator,
    ) = self.add_binomial_penalty(
      model=model,
      r=r,
      d=d,
      n_survived_in_group=n_survived_in_group,
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

    n_total = model.getVarByName("n_total")
    n_alive = model.getVarByName("n_alive")
    # Get the patient inclusion variables a[j]
    a = [model.getVarByName(f"a[{j}]") for j in range(self.n_patients)]
    assert n_total is not None
    assert n_alive is not None
    assert all(var is not None for var in a)
    selected = [j for j in range(self.n_patients) if a[j].X > 0.5]
    n_alive_val = np.rint(n_alive.X)
    n_total_val = np.rint(n_total.X)

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
    if verbose:
      print("Selected patients:", selected)
      print("n_total:          ", int(n_total_val))
      print("n_alive:          ", int(n_alive_val))
      print("Binomial penalty: ", 2*binomial_penalty_val)
      print("Patient penalty:  ", 2*patient_penalty_val)
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
      km_probability=km_probability_var.X,
    )
