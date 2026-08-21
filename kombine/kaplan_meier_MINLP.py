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
from .utilities import LOG_ZERO_EPSILON_DEFAULT, GurobiOptimizerMixin, validate_class_probs

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


def km_survival_from_risk_counts(
  r_vals: npt.ArrayLike,
  s_vals: npt.ArrayLike,
  *,
  log_zero_epsilon: float = LOG_ZERO_EPSILON_DEFAULT,
  cumulative: bool = False,
) -> float | list[float]:
  """
  Kaplan-Meier survival from at-risk and survived counts.

  Matches the Gurobi encoding: skip r == 0; otherwise use
  log(s + eps) - log(r + eps). If ``cumulative``, return the survival
  probability after each interval; otherwise the product over all intervals.
  """
  r_vals = np.asarray(r_vals, dtype=float)
  s_vals = np.asarray(s_vals, dtype=float)
  if r_vals.shape != s_vals.shape:
    raise ValueError("r_vals and s_vals must have the same shape")
  log_cum = 0.0
  probs: list[float] = []
  for r, s in zip(r_vals.tolist(), s_vals.tolist()):
    if r > 0.5:
      log_cum += math.log(s + log_zero_epsilon) - math.log(r + log_zero_epsilon)
    if cumulative:
      probs.append(float(math.exp(log_cum)))
  if cumulative:
    return probs
  return float(math.exp(log_cum))


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
        method='Powell',
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
        method='Powell',
        options={
          #loose tolerance - we don't actually care about the values of the nuisance parameters
          "ftol": 1e-3,
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

  @classmethod
  def from_discrete_class_probs(
    cls,
    time: float,
    censored: bool,
    class_probs: list[float],
    *,
    systematics: list[float] | None = None,
  ):
    """
    Create a KaplanMeierPatientNLL from discrete class probabilities.

    The parameter is the integer class index. The NLL is piecewise-constant
    over intervals [k, k+1) with value -log(p_k).
    """
    systematics = systematics or []
    if systematics:
      raise NotImplementedError(
        "Systematics are not supported for discrete class probabilities"
      )
    validate_class_probs(class_probs)
    log_probs = [
      float(np.log(prob)) if prob > 0 else -float('inf')
      for prob in class_probs
    ]

    def full_nll_0d(eff: float) -> float:
      if not np.isfinite(eff):
        return float('inf')
      if eff < 0:
        return float('inf')
      idx = int(math.floor(eff))
      if idx >= len(log_probs):
        return float('inf')
      if np.isclose(eff, len(log_probs)):
        return float('inf')
      log_prob = log_probs[idx]
      if not np.isfinite(log_prob):
        return float('inf')
      return -log_prob

    wrapped = cls._solve_0d(full_nll_0d)
    observed_param = float(int(np.argmax(class_probs)))
    return cls(time, censored, wrapped, observed_param)

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

class MINLPForKM(GurobiOptimizerMixin):  # pylint: disable=too-many-public-methods, too-many-instance-attributes
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
    binomial_only: bool = False,
    patient_wise_only: bool = False,
  ):
    if binomial_only and patient_wise_only:
      raise ValueError("binomial_only and patient_wise_only cannot both be True")
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_max = parameter_max
    self.__time_point = time_point
    self.__endpoint_epsilon = endpoint_epsilon
    self.__log_zero_epsilon = log_zero_epsilon # Store the epsilon
    self.__collapse_consecutive_deaths = collapse_consecutive_deaths
    self.__binomial_only = binomial_only
    self.__patient_wise_only = patient_wise_only
    self.__expected_probability_constraint = None
    self.__binomial_penalty_constraint = None
    self.__patient_constraints_for_binomial_only = None
    self.__risk_set_r_vars = None
    self.__risk_set_s_vars = None
    self.__profile_p_died = None
    self.__profile_p_survived = None
    self.__profile_log_p_died = None
    self.__profile_log_p_survived = None
    # MIP starts from the previous nearby expected_probability solve.
    self.__mip_start_a: dict[int, float] | None = None
    self.__mip_start_mode: tuple[bool, bool] | None = None
    self.__mip_start_profile: dict[str, list[float]] | None = None
    if not np.isfinite(self.__parameter_min and self.__parameter_min != -np.inf):
      raise ValueError("parameter_min must be finite or -inf")
    if not np.isfinite(self.__parameter_max and self.__parameter_max != np.inf):
      raise ValueError("parameter_max must be finite or inf")

  def seed_assignment_starts(self, starts: dict[int, float] | list[int]) -> None:
    """
    Seed binary assignment Starts from a previous solve (e.g. another time point).

    ``starts`` may be a dict of patient index -> 0/1, or a list of selected indices.
    """
    if isinstance(starts, dict):
      self.__mip_start_a = {int(k): float(v) for k, v in starts.items()}
    else:
      selected = set(int(j) for j in starts)
      self.__mip_start_a = {
        j: 1.0 if j in selected else 0.0 for j in range(self.n_patients)
      }
    # Mode unknown when seeded externally; allow first solve to use them.
    self.__mip_start_mode = None

  def _apply_assignment_mip_starts(
    self,
    a,
    *,
    binomial_only: bool,
    patient_wise_only: bool,
  ) -> None:
    """Apply cached assignment Starts when the constraint mode matches."""
    mode = (binomial_only, patient_wise_only)
    if self.__mip_start_mode is not None and self.__mip_start_mode != mode:
      self.__mip_start_a = None
      self.__mip_start_mode = None
      self.__mip_start_profile = None
    if self.__mip_start_a is None:
      for j in range(self.n_patients):
        a[j].Start = GRB.UNDEFINED
      return
    for j, value in self.__mip_start_a.items():
      if j < self.n_patients:
        a[j].Start = value

  def _store_assignment_mip_starts(
    self,
    a,
    *,
    binomial_only: bool,
    patient_wise_only: bool,
  ) -> None:
    """Cache assignment incumbents for the next nearby expected_probability."""
    self.__mip_start_a = {j: float(a[j].X) for j in range(self.n_patients)}
    self.__mip_start_mode = (binomial_only, patient_wise_only)

  def export_assignment_mip_starts(self) -> dict[int, float] | None:
    """Return the cached assignment Starts, if any."""
    if self.__mip_start_a is None:
      return None
    return dict(self.__mip_start_a)

  def _apply_profile_mip_starts(self) -> None:
    """Start continuous profile probabilities from the previous nearby solve."""
    starts = self.__mip_start_profile
    if starts is None:
      return
    var_map = {
      "p_died": self.__profile_p_died,
      "p_survived": self.__profile_p_survived,
      "log_p_died": self.__profile_log_p_died,
      "log_p_survived": self.__profile_log_p_survived,
    }
    for name, values in starts.items():
      vars_by_time = var_map.get(name)
      if vars_by_time is None:
        continue
      for i, value in enumerate(values):
        if i < len(vars_by_time):
          vars_by_time[i].Start = value

  def _store_profile_mip_starts(self) -> None:
    """Cache continuous profile incumbents for the next nearby expected_probability."""
    if self.__profile_p_survived is None:
      self.__mip_start_profile = None
      return
    self.__mip_start_profile = {
      "p_died": [float(self.__profile_p_died[i].X) for i in range(self.n_times_to_consider)],
      "p_survived": [
        float(self.__profile_p_survived[i].X) for i in range(self.n_times_to_consider)
      ],
      "log_p_died": [
        float(self.__profile_log_p_died[i].X) for i in range(self.n_times_to_consider)
      ],
      "log_p_survived": [
        float(self.__profile_log_p_survived[i].X) for i in range(self.n_times_to_consider)
      ],
    }

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
    # The key insight: we can collapse death times if no censoring occurs between them
    # (censoring at the same time as the last death doesn't count due to KM convention)
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
        # using strict inequalities so censoring exactly at next_time does not block
        # the death at next_time (KM convention: deaths before censoring at same time).
        censored_between = np.any(
          self.patient_censored &
          (self.patient_times > group_end) &
          (self.patient_times < next_time)
        )

        # Censoring at group_end always affects risk sets for subsequent death times,
        # so it blocks extending the collapsed interval.
        censored_at_group_end = np.any(
          self.patient_censored & (self.patient_times == group_end)
        )
        if censored_at_group_end:
          break

        if censored_between:
          # Can't include next_time in this group due to intervening censoring
          break
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
  def n_sub_times_to_consider(self) -> int:
    """
    The number of times to consider without collapsing consecutive deaths.
    """
    return sum(len(sub_times) for sub_times in self._collapsed_time_groups.values())
  @functools.cached_property
  def _collapsed_time_groups(self) -> dict[float, list[float]]:
    """
    Map from representative time to list of original times in the collapsed group.
    Only used when collapse_consecutive_deaths is True.
    """
    # Get all death times up to the time point
    death_mask = (~self.patient_censored) & (self.patient_times <= self.time_point)
    death_times = self.patient_times[death_mask]
    all_times = list(death_times) + [self.time_point]
    unique_times = np.unique(all_times)

    if not self.collapse_consecutive_deaths:
      return {time: [time] for time in unique_times}

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

        # Censoring at group_end always affects risk sets for subsequent death times,
        # so it blocks extending the collapsed interval.
        censored_at_group_end = np.any(
          self.patient_censored & (self.patient_times == group_end)
        )
        if censored_at_group_end:
          break

        if censored_between:
          break
        group_end = next_time
        j += 1

      # Store the group
      group_times = unique_times[i:j].tolist()
      groups[group_end] = group_times

      # Move to the next ungrouped time
      i = j

    return groups

  def patient_died(self, t, *, collapse_consecutive_deaths=True) -> npt.NDArray[np.bool_]:
    """
    Returns a boolean array indicating which patients died at time t.
    If collapse_consecutive_deaths is True, for any t in a collapsed interval,
      anyone who died during the interval and before t is considered to have died at t.
    For any time t, a patient is considered to have died at t if:
    - Their time == t and not censored (no collapse)
    - If t is in a collapsed interval, their time is >= interval_start and < t, and not censored

    If self.collapse_consecutive_deaths is False, the collapse_consecutive_deaths
    argument is ignored and no collapsing is done.
    """
    if not self.collapse_consecutive_deaths or not collapse_consecutive_deaths:
      return (self.patient_times == t) & (~self.patient_censored)
    groups = self._collapsed_time_groups
    for group_end, group_times in groups.items():
      interval_start = min(group_times)
      if interval_start <= t <= group_end:
        return (
          (self.patient_times >= interval_start)
          & (self.patient_times <= t)
          & (~self.patient_censored)
        )
    return (self.patient_times == t) & (~self.patient_censored)

  def patient_still_at_risk(self, t, *, collapse_consecutive_deaths=True) -> npt.NDArray[np.bool_]:
    """
    Returns a boolean array indicating which patients are still at risk at time t.
    If collapse_consecutive_deaths is True, anyone who died in the collapsed interval
      is considered at risk at any t in the interval.
    For any time t, a patient is at risk if:
    - Their time >= t (regardless of censored status)
    - If t is in a collapsed interval, their time is >= interval_start

    If self.collapse_consecutive_deaths is False, the collapse_consecutive_deaths
    argument is ignored and no collapsing is done.
    """
    if not self.collapse_consecutive_deaths or not collapse_consecutive_deaths:
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

  def _fixed_selected_counts(self) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    At-risk, died, survived, and collapsed sub-death counts for the nominal
    in-range assignment (used when ``binomial_only`` fixes ``a``).
    """
    selected = self.parameter_in_range
    r_vals: list[int] = []
    d_vals: list[int] = []
    s_vals: list[int] = []
    sub_d_vals: list[int] = []
    for dt in self.times_to_consider:
      r_value = int(np.count_nonzero(self.patient_still_at_risk(dt) & selected))
      d_value = int(np.count_nonzero(self.patient_died(dt) & selected))
      r_vals.append(r_value)
      d_vals.append(d_value)
      s_vals.append(r_value - d_value)
      for collapsed_time in self._collapsed_time_groups[dt]:
        sub_d_vals.append(int(np.count_nonzero(
          self.patient_died(collapsed_time, collapse_consecutive_deaths=False)
          & selected
        )))
    return r_vals, d_vals, s_vals, sub_d_vals

  def _rd_pair_is_feasible(self, i: int, r_value: int, d_value: int) -> bool:
    """Whether (r, d) at death-time index i can occur given risk-set flow.

    Survivors at time i cannot exceed the next risk-set maximum plus all
    censoring in ``[times[i], times[i+1])``, and r cannot exceed the previous
    risk-set maximum.  These bounds are valid for collapsed times because
    ``times_to_consider`` are group ends and intervening censoring is what
    splits groups.
    """
    if (
      r_value > self.n_at_risk_max[i]
      or d_value > self.n_died_max[i]
      or d_value > r_value
    ):
      return False
    s_value = r_value - d_value
    if i + 1 < self.n_times_to_consider:
      cmax = int(self.n_censored_between_times_max[i])
      if s_value > int(self.n_at_risk_max[i + 1]) + cmax:
        return False
    if i > 0 and r_value > int(self.n_at_risk_max[i - 1]):
      return False
    return True

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
    sub_d = model.addVars(
      self.n_sub_times_to_consider,
      vtype=GRB.INTEGER,
      name="sub_d",
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
    j = -1
    for i, dt in enumerate(self.times_to_consider):
      model.addConstr(
        d[i] == gp.quicksum(
          a[k] for k in range(self.n_patients) if self.patient_died(dt)[k]
        ),
        name=f"d_{i}_definition",
      )
      model.addConstr(
        r[i] == gp.quicksum(
          a[k] for k in range(self.n_patients) if self.patient_still_at_risk(dt)[k]
        ),
        name=f"r_{i}_definition",
      )
      model.addConstr(
        s[i] == r[i] - d[i],
        name=f"s_{i}_definition",
      )

      first_j = j+1
      for j, sub_dt in enumerate(self._collapsed_time_groups[dt], start=first_j):
        model.addConstr(
          sub_d[j] == gp.quicksum(
            a[k] for k in range(self.n_patients)
            if self.patient_died(sub_dt, collapse_consecutive_deaths=False)[k]
          ),
          name=f"sub_d_{j}_definition",
        )

      #Make sure that each d is the sum of its sub_ds.
      #This is here as a sanity check.  Gurobi should optimize it out.
      model.addConstr(
        d[i] == gp.quicksum(sub_d[j] for j in range(first_j, j+1)),
        name=f"d_{i}_from_sub_d",
      )

    return (
      d,
      sub_d,
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

  def add_binomial_penalty(  # pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-branches
    self,
    model: gp.Model,
    *,
    r: gp.tupledict[int, gp.Var],
    d: gp.tupledict[int, gp.Var],
    sub_d: gp.tupledict[int, gp.Var],
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

    Survived counts enter through ``r - d`` on the choose-(r, d) indicators;
    the ``s`` tupledict is unused here but kept for call-site compatibility.
    """
    _ = s  # survived encoded as r - d on choose-(r, d) indicators

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
    self.__profile_p_died = p_died
    self.__profile_p_survived = p_survived
    self.__profile_log_p_died = log_p_died
    self.__profile_log_p_survived = log_p_survived

    n_choose_d_table = self.n_choose_d_term_table
    binomial_terms = []

    if self.__binomial_only:
      _ = (r, d, sub_d)
      r_vals, d_vals, _s_vals, sub_d_vals = self._fixed_selected_counts()
      sub_d_offset = 0
      for i, time in enumerate(self.times_to_consider):
        r_value = r_vals[i]
        d_value = d_vals[i]
        s_value = r_value - d_value
        penalty = n_choose_d_table[(r_value, d_value)]
        binomial_terms.append(-penalty)
        binomial_terms.append(-d_value * log_p_died[i])
        binomial_terms.append(-s_value * log_p_survived[i])
        if d_value > 0:
          binomial_terms.append(
            -(math.lgamma(d_value + 1) - d_value * np.log(d_value))
          )
        n_sub = len(self._collapsed_time_groups[time])
        for sub_d_value in sub_d_vals[sub_d_offset:sub_d_offset + n_sub]:
          if sub_d_value > 0:
            binomial_terms.append(
              math.lgamma(sub_d_value + 1) - sub_d_value * np.log(sub_d_value)
            )
        sub_d_offset += n_sub
    else:
      # Binomial terms: one SOS1-style choose-(r,d) encoding per death time.
      # Only flow-feasible (r,d) indicators are created; d*log p_d and s*log p_s
      # are charged on the same indicator.
      sub_d_counter = -1
      for i, time in enumerate(self.times_to_consider):
        feasible_indicators = []
        for (r_value, d_value), penalty in n_choose_d_table.items():
          if not self._rd_pair_is_feasible(i, r_value, d_value):
            continue
          indicator = model.addVar(
            vtype=GRB.BINARY,
            name=f"n_choose_d_indicator_{i}_{r_value}_{d_value}",
          )
          feasible_indicators.append(indicator)
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
          s_value = r_value - d_value
          binomial_terms.append(-penalty * indicator)
          binomial_terms.append(-d_value * log_p_died[i] * indicator)
          binomial_terms.append(-s_value * log_p_survived[i] * indicator)
          if d_value > 0:
            binomial_terms.append(
              -indicator * (
                math.lgamma(d_value + 1) - d_value * np.log(d_value)
              )
            )

        if not feasible_indicators:
          raise RuntimeError(
            f"No feasible (r, d) pairs for death-time index {i} "
            f"(n_at_risk_max={self.n_at_risk_max[i]}, n_died_max={self.n_died_max[i]})"
          )
        model.addConstr(
          gp.quicksum(feasible_indicators) == 1,
          name=f"one_n_choose_d_indicator_per_death_time_{i}",
        )

        for sub_d_counter, collapsed_time in enumerate(
          self._collapsed_time_groups[time],
          start=sub_d_counter+1
        ):
          sub_d_var = sub_d[sub_d_counter]
          max_sub_d = np.count_nonzero(
            self.patient_died(collapsed_time, collapse_consecutive_deaths=False)
          )
          sub_d_indicators = []
          for sub_d_value in range(max_sub_d + 1):
            sub_d_indicator = model.addVar(
              vtype=GRB.BINARY,
              name=f"sub_d_indicator_{i}_{sub_d_counter}_{sub_d_value}",
            )
            sub_d_indicators.append(sub_d_indicator)
            model.addGenConstrIndicator(
              sub_d_indicator,
              True,
              sub_d_var,
              GRB.EQUAL,
              sub_d_value,
              name=f"sub_d_indicator_constr_{i}_{sub_d_counter}_{sub_d_value}",
            )
            if sub_d_value > 0:
              binomial_terms.append(
                sub_d_indicator * (
                  math.lgamma(sub_d_value + 1) - sub_d_value * np.log(sub_d_value)
                )
              )
          model.addConstr(
            gp.quicksum(sub_d_indicators) == 1,
            name=f"one_sub_d_indicator_per_sub_death_time_{i}_{sub_d_counter}",
          )

    binom_penalty_expr = gp.quicksum(binomial_terms)
    binom_penalty = model.addVar(
      vtype=GRB.CONTINUOUS,
      name="binom_penalty",
    )
    if self.__binomial_only:
      model.addConstr(
        binom_penalty == binom_penalty_expr,
        name="binomial_penalty_definition",
      )
      return binom_penalty, expected_probability_var, None

    # Full NLL: bilinear terms (indicator * log p). Keep the indicator + big-M
    # sandwich so Gurobi does not treat this as a quadratic equality (status 13).
    use_binomial_penalty_indicator = model.addVar(
      vtype=GRB.BINARY,
      name="use_binomial_penalty_indicator",
    )
    model.addGenConstrIndicator(
      use_binomial_penalty_indicator,
      False,
      binom_penalty,
      GRB.EQUAL,
      0.0,
      name="binomial_penalty_inactive",
    )
    max_penalty_term = max(
      abs(penalty) for penalty in n_choose_d_table.values()
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
      sub_d,
      r,
      s,
    ) = self.add_counter_variables_and_constraints(
      model=model,
      a=a,
    )

    km_probability_var = None
    expected_probability_var = None
    use_binomial_penalty_indicator = None
    binom_penalty = 0.0

    if self.__patient_wise_only:
      km_probability_var = self.add_kaplan_meier_probability_variables_and_constraints(
        model=model,
        r=r,
        s=s,
      )
    else:
      (
        binom_penalty,
        expected_probability_var,
        use_binomial_penalty_indicator,
      ) = self.add_binomial_penalty(
        model=model,
        r=r,
        d=d,
        sub_d=sub_d,
        s=s,
      )
      if use_binomial_penalty_indicator is not None:
        model.addConstr(
          use_binomial_penalty_indicator == 1,
          name="use_binomial_penalty",
        )
      if self.__binomial_only:
        for j in range(self.n_patients):
          if self.parameter_in_range[j]:
            assert self.nll_penalty_for_patient_in_range[j] <= 0
            model.addConstr(
              a[j] == 1,
              name=f"patient_{j}_must_be_selected_binomial_only",
            )
          else:
            assert self.nll_penalty_for_patient_in_range[j] >= 0
            model.addConstr(
              a[j] == 0,
              name=f"patient_{j}_must_not_be_selected_binomial_only",
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
    self.__risk_set_r_vars = r
    self.__risk_set_s_vars = s

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
    km_probability_var: gp.Var | None,
    expected_probability_var: gp.Var | None,
  ):
    """
    Update the Gurobi model with the expected probability constraint.
    This is the only thing that changes between runs of the MINLP.
    """
    if self.__expected_probability_constraint is not None:
      model.remove(self.__expected_probability_constraint)
      self.__expected_probability_constraint = None

    if not self.__patient_wise_only:
      if expected_probability is not None:
        assert expected_probability_var is not None
        self.__expected_probability_constraint = model.addConstr(
          expected_probability_var == expected_probability,
          name="expected_probability_constraint",
        )
    else:
      # Constrain the KM probability based on the expected_probability
      # If expected > observed, then KM_prob >= expected_probability
      # If expected < observed, then KM_prob <= expected_probability
      # If expected == observed or is None, then KM_prob is unconstrained
      assert km_probability_var is not None
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
      else:
        assert expected_probability == self.observed_KM_probability

    model.update()

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
    if binomial_only != self.__binomial_only or patient_wise_only != self.__patient_wise_only:
      raise ValueError(
        "run_MINLP mode must match MINLPForKM construction "
        f"(constructed binomial_only={self.__binomial_only}, "
        f"patient_wise_only={self.__patient_wise_only})"
      )
    if expected_probability is not None:
      if not patient_wise_only and (expected_probability <= 0 or expected_probability >= 1):
        raise ValueError(f"expected_probability={expected_probability} must be in (0, 1) or None")
      if expected_probability < 0 or expected_probability > 1:
        raise ValueError(f"expected_probability={expected_probability} must be in [0, 1] or None")

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
      _use_binomial_penalty_indicator,
    ) = self.gurobi_model
    self.update_model_with_expected_probability(
      model=model,
      km_probability_var=km_probability_var,
      expected_probability=expected_probability,
      expected_probability_var=expected_probability_var,
    )

    self._apply_assignment_mip_starts(
      a,
      binomial_only=binomial_only,
      patient_wise_only=patient_wise_only,
    )
    self._apply_profile_mip_starts()

    # Initial Gurobi parameters. FuncPieces starts at 1000; fallbacks may raise it.
    # Cuts=-1 restores the default after a fallback that set Cuts=2.
    initial_gurobi_params = {
      'OutputFlag': 1 if verbose else 0,
      'DisplayInterval': 1,
      'MIPGap': MIPGap,
      'MIPGapAbs': MIPGapAbs,
      'NonConvex': 2,
      'NumericFocus': 0,
      'Seed': 123456,
      'TimeLimit': TimeLimit,
      'Threads': Threads,
      'MIPFocus': MIPFocus,
      'FuncPieces': 1000,
      'FuncPieceRatio': 0.5,
    }
    if LogFile is not None:
      initial_gurobi_params['LogFile'] = os.fspath(LogFile)

    # Recovery fallbacks, then a weaker MIPGap certificate, then verbose debug.
    # Do not accept SUBOPTIMAL; TimeLimit retries are handled in _optimize_with_fallbacks.
    fallback_strategies = []
    if MIPFocus != 2:
      fallback_strategies.append(
        ({'MIPFocus': 2}, "MIPFocus set to 2 (optimality focus)")
      )
    if TimeLimit is not None:
      fallback_strategies.append(
        ({'TimeLimit': TimeLimit * 1.5}, "Increased TimeLimit by 50%")
      )
    fallback_strategies.append(
      ({'FuncPieces': 2000, 'FuncPieceRatio': 0.5}, "Increased FuncPieces to 2000")
    )
    fallback_strategies.append(
      ({'FuncPieces': 5000, 'FuncPieceRatio': 0.75}, "Increased FuncPieces to 5000")
    )
    fallback_strategies.append(
      ({'NumericFocus': 3}, "NumericFocus set to 3 (highest precision)")
    )
    fallback_strategies.append(
      ({'Cuts': 2}, "Aggressive cut generation")
    )
    if MIPGap < 1e-3:
      fallback_strategies.append(
        ({'MIPGap': 1e-3}, "Relaxed MIPGap to 1e-3")
      )
    if MIPGap < 1e-2:
      fallback_strategies.append(
        ({'MIPGap': 1e-2}, "Relaxed MIPGap to 1e-2")
      )
    if not verbose:
      fallback_strategies.append(
        (
          {'OutputFlag': 1, 'DisplayInterval': 1},
          "Enabled Gurobi output for debug after failed recovery",
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
    self._store_assignment_mip_starts(
      a,
      binomial_only=binomial_only,
      patient_wise_only=patient_wise_only,
    )
    self._store_profile_mip_starts()
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
    if patient_wise_only:
      binomial_penalty_val = 0.0
      p_survived_val = [np.nan] * self.n_times_to_consider
      assert km_probability_var is not None
      km_probability_val = km_probability_var.X
    else:
      binom_penalty_var = model.getVarByName("binom_penalty")
      assert binom_penalty_var is not None
      binomial_penalty_val = binom_penalty_var.X
      p_survived_val = []
      for i in range(self.n_times_to_consider):
        var = model.getVarByName(f"p_survived[{i}]")
        assert var is not None
        p_survived_val.append(var.X)
      assert self.__risk_set_r_vars is not None
      assert self.__risk_set_s_vars is not None
      km_probability_val = km_survival_from_risk_counts(
        [self.__risk_set_r_vars[i].X for i in range(self.n_times_to_consider)],
        [self.__risk_set_s_vars[i].X for i in range(self.n_times_to_consider)],
        log_zero_epsilon=self.__log_zero_epsilon,
      )
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
      km_probability=km_probability_val,
    )
