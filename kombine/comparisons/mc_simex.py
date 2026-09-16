"""
Küchenhoff MC-SIMEX for a binary high/low survival label.

This is a comparison estimator, not a KoMbine profile likelihood. Each
simulation-extrapolation step refits a naive hard-label Kaplan-Meier,
logrank, or Cox functional after extra misclassification; the
error-free functional is a quadratic extrapolation to lambda = -1.

Poisson or discrete-class measurements enter only through
``observable.probability_in_range``, which yields a per-patient flip
rate e_i. The survival model never sees the raw count. POI-SIMEX
(continuous Poisson density as a regression covariate) is intentionally
not implemented.

The Wald quadratic returned by ``McSimexForCoxPH.compute_2nll_at_hazard_ratio``
is not a profile likelihood. It is
(log H - log H_hat)^2 / sigma_hat^2, the same information as a Wald CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize
import scipy.stats

from ..kaplan_meier import KaplanMeierInstance, KaplanMeierPatient
from .yi_correction import YiCorrectionBase, YiCorrectionWithThreshold

if TYPE_CHECKING:
  from ..datacard import Patient

DEFAULT_LAMBDA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
DEFAULT_N_SIMULATIONS = 100
FLIP_RATE_CEILING = 0.5 - 1e-12
LOG_HAZARD_RATIO_BOUNDS = (-10.0, 10.0)
WALD_Z_95 = 1.96
VARIANCE_FLOOR = 1e-12


def flip_probability(misclassification_rate: float, simex_lambda: float) -> float:
  """
  Extra-flip probability Pi^lambda for a binary MC-SIMEX step.

  P(flip | lambda, e) = (1 - (1 - 2 e)^lambda) / 2.
  lambda = 0 is a no-op. Rates at or above 1/2 are clamped just below 1/2.
  """
  if simex_lambda == 0.0:
    return 0.0
  rate = min(max(float(misclassification_rate), 0.0), FLIP_RATE_CEILING)
  return 0.5 * (1.0 - (1.0 - 2.0 * rate) ** simex_lambda)


def extrapolate_quadratic(lambda_grid: np.ndarray, values: np.ndarray) -> float:
  """
  Quadratic (or linear if fewer than three finite points) fit in lambda,
  evaluated at lambda = -1.
  """
  finite = np.isfinite(values)
  x_fit = np.asarray(lambda_grid, dtype=float)[finite]
  y_fit = np.asarray(values, dtype=float)[finite]
  if x_fit.size < 2:
    raise ValueError(
      "Need at least two finite SIMEX lambda points to extrapolate, "
      f"got {x_fit.size}."
    )
  degree = 2 if x_fit.size >= 3 else 1
  coefficients = np.polyfit(x_fit, y_fit, deg=degree)
  return float(np.polyval(coefficients, -1.0))


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
  if rng is None:
    return np.random.default_rng()
  if isinstance(rng, np.random.Generator):
    return rng
  return np.random.default_rng(rng)


def _naive_logrank_statistic(  # pylint: disable=too-many-locals
  times: np.ndarray,
  censored: np.ndarray,
  is_high: np.ndarray,
) -> float | None:
  """
  Hard-label logrank statistic U^2 / V, or None if a group is empty.
  """
  if not np.any(is_high) or not np.any(~is_high):
    return None

  death_times = np.unique(times[~censored])
  if death_times.size == 0:
    return None

  statistic_u = 0.0
  statistic_v = 0.0
  for death_time in death_times:
    at_risk = times >= death_time
    n_low = int(np.count_nonzero(at_risk & ~is_high))
    n_high = int(np.count_nonzero(at_risk & is_high))
    n_total = n_low + n_high
    if n_total == 0:
      continue
    died = (times == death_time) & ~censored
    d_low = int(np.count_nonzero(died & ~is_high))
    d_high = int(np.count_nonzero(died & is_high))
    d_total = d_low + d_high
    if d_total == 0:
      continue
    expected_low = n_low * d_total / n_total
    if n_total > 1:
      variance_t = (
        n_low * n_high * d_total * (n_total - d_total)
      ) / (n_total * n_total * (n_total - 1))
    else:
      variance_t = 0.0
    statistic_u += d_low - expected_low
    statistic_v += variance_t

  if statistic_v <= 0.0:
    return 0.0
  return float(statistic_u * statistic_u / statistic_v)


def _risk_sets_at_death(
  times: np.ndarray,
  censored: np.ndarray,
  is_high: np.ndarray,
  death_time: float,
) -> tuple[int, int, int, int]:
  at_risk = times >= death_time
  r_low = int(np.count_nonzero(at_risk & ~is_high))
  r_high = int(np.count_nonzero(at_risk & is_high))
  died = (times == death_time) & ~censored
  d_low = int(np.count_nonzero(died & ~is_high))
  d_high = int(np.count_nonzero(died & is_high))
  return r_low, r_high, d_low, d_high


def breslow_2nll(
  times: np.ndarray,
  censored: np.ndarray,
  is_high: np.ndarray,
  hazard_ratio: float,
) -> float:
  """
  Breslow Cox partial 2NLL for a hard high/low label.
  """
  if hazard_ratio <= 0.0:
    raise ValueError(f"hazard_ratio must be positive, got {hazard_ratio}")
  death_times = np.unique(times[~censored])
  log_hazard_ratio = np.log(hazard_ratio)
  log_likelihood = 0.0
  for death_time in death_times:
    r_low, r_high, d_low, d_high = _risk_sets_at_death(
      times, censored, is_high, float(death_time)
    )
    d_total = d_low + d_high
    if d_total <= 0:
      continue
    denominator = r_low + hazard_ratio * r_high
    if denominator <= 0.0:
      denominator = VARIANCE_FLOOR
    log_likelihood += d_high * log_hazard_ratio - d_total * np.log(denominator)
  return float(-2.0 * log_likelihood)


def observed_information_log_hr(
  times: np.ndarray,
  censored: np.ndarray,
  is_high: np.ndarray,
  hazard_ratio: float,
) -> float:
  """
  Observed information for theta = log H in the Breslow partial likelihood.
  """
  death_times = np.unique(times[~censored])
  information = 0.0
  for death_time in death_times:
    r_low, r_high, d_low, d_high = _risk_sets_at_death(
      times, censored, is_high, float(death_time)
    )
    d_total = d_low + d_high
    if d_total <= 0 or r_low <= 0 or r_high <= 0:
      continue
    risk = r_low + hazard_ratio * r_high
    information += d_total * hazard_ratio * r_high * r_low / (risk * risk)
  return float(information)


def mle_log_hazard_ratio(
  times: np.ndarray,
  censored: np.ndarray,
  is_high: np.ndarray,
) -> tuple[float, float] | None:
  """
  Bounded MLE of log H and observed-information variance, or None if empty.
  """
  if not np.any(is_high) or not np.any(~is_high):
    return None
  if not np.any(~censored):
    return None

  def objective(log_hazard_ratio: float) -> float:
    return breslow_2nll(
      times, censored, is_high, float(np.exp(log_hazard_ratio))
    )

  result = scipy.optimize.minimize_scalar(
    objective,
    bounds=LOG_HAZARD_RATIO_BOUNDS,
    method='bounded',
  )
  log_hazard_ratio = float(result.x)
  hazard_ratio = float(np.exp(log_hazard_ratio))
  information = observed_information_log_hr(
    times, censored, is_high, hazard_ratio
  )
  if information <= 0.0:
    variance = np.nan
  else:
    variance = 1.0 / information
  return log_hazard_ratio, float(variance)


class _SimexMonteCarlo:  # pylint: disable=too-few-public-methods
  """
  Lambda grid, Monte Carlo size, and extra-flip draws shared by MC-SIMEX classes.
  """

  _lambda_grid: tuple[float, ...]
  _B: int
  _rng: np.random.Generator
  _prior_alpha: float
  _prior_beta: float

  def _init_simex(  # pylint: disable=too-many-arguments,invalid-name
    self,
    *,
    lambda_grid: tuple[float, ...] | list[float] | None,
    B: int,
    rng: np.random.Generator | int | None,
    prior_alpha: float,
    prior_beta: float,
  ) -> None:
    if B < 1:
      raise ValueError(f"B must be >= 1, got {B}")
    grid = DEFAULT_LAMBDA_GRID if lambda_grid is None else tuple(lambda_grid)
    if any(lam < 0.0 for lam in grid):
      raise ValueError(
        "lambda_grid must be non-negative (do not simulate lambda = -1), "
        f"got {grid}"
      )
    if 0.0 not in grid:
      grid = (0.0,) + grid
    self._lambda_grid = tuple(sorted(set(float(lam) for lam in grid)))
    self._B = int(B)
    self._rng = _as_rng(rng)
    self._prior_alpha = float(prior_alpha)
    self._prior_beta = float(prior_beta)

  def simulate_labels(
    self,
    observed_labels: np.ndarray,
    flip_rates: np.ndarray,
    simex_lambda: float,
  ) -> np.ndarray:
    """
    Flip observed binary labels with the MC-SIMEX probability at lambda.
    """
    if simex_lambda == 0.0:
      return observed_labels.copy()
    probabilities = np.array(
      [flip_probability(rate, simex_lambda) for rate in flip_rates],
      dtype=float,
    )
    flips = self._rng.random(observed_labels.shape[0]) < probabilities
    return observed_labels ^ flips


class McSimexBase(YiCorrectionBase, _SimexMonteCarlo):
  """
  Shared MC-SIMEX setup: patients, range, lambda grid, Monte Carlo size, RNG.
  """

  def __init__(  # pylint: disable=too-many-arguments,invalid-name
    self,
    patients: list[Patient],
    parameter_min: float = -np.inf,
    parameter_max: float = np.inf,
    *,
    lambda_grid: tuple[float, ...] | list[float] | None = None,
    B: int = DEFAULT_N_SIMULATIONS,
    rng: np.random.Generator | int | None = None,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.0,
  ):
    YiCorrectionBase.__init__(
      self,
      patients,
      parameter_min=parameter_min,
      parameter_max=parameter_max,
    )
    self._init_simex(
      lambda_grid=lambda_grid,
      B=B,
      rng=rng,
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )

  def observed_in_range(self, patient: Patient) -> bool:
    """
    Hard observed membership in [parameter_min, parameter_max).
    """
    return self._parameter_min <= patient.observed_parameter < self._parameter_max

  def flip_rate_for_range(  # pylint: disable=too-many-arguments
    self,
    patient: Patient,
    range_min: float,
    range_max: float,
    *,
    prior_alpha: float,
    prior_beta: float,
  ) -> float:
    """
    e_i = 1 - P(G = G*_i | data) for a single binary range membership.
    """
    observed = range_min <= patient.observed_parameter < range_max
    probability_observed = self.compute_patient_prob_in_range(
      patient,
      range_min,
      range_max,
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )
    if observed:
      return 1.0 - probability_observed
    return probability_observed


class McSimexWithThreshold(YiCorrectionWithThreshold, _SimexMonteCarlo):
  """
  MC-SIMEX for two-group (low / high) hard labels.
  """

  def __init__(  # pylint: disable=too-many-arguments,invalid-name
    self,
    patients: list[Patient],
    parameter_min: float = -np.inf,
    parameter_max: float = np.inf,
    parameter_threshold: float = 0.0,
    *,
    lambda_grid: tuple[float, ...] | list[float] | None = None,
    B: int = DEFAULT_N_SIMULATIONS,
    rng: np.random.Generator | int | None = None,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.0,
  ):
    YiCorrectionWithThreshold.__init__(
      self,
      patients,
      parameter_min=parameter_min,
      parameter_max=parameter_max,
      parameter_threshold=parameter_threshold,
    )
    self._init_simex(
      lambda_grid=lambda_grid,
      B=B,
      rng=rng,
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )

  def analysis_mask(self) -> np.ndarray:
    """
    Patients whose observed parameter lies in [parameter_min, parameter_max).
    """
    return np.array(
      [
        self._parameter_min <= patient.observed_parameter < self._parameter_max
        for patient in self._patients
      ],
      dtype=bool,
    )

  def observed_is_high(self) -> np.ndarray:
    """
    G*_i = 1 iff threshold <= observed_parameter < parameter_max.
    """
    return np.array(
      [
        self._parameter_threshold <= patient.observed_parameter < self._parameter_max
        for patient in self._patients
      ],
      dtype=bool,
    )

  def flip_rates_high_low(
    self,
    *,
    prior_alpha: float,
    prior_beta: float,
  ) -> np.ndarray:
    """
    Per-patient flip rates for the observed high/low label.
    """
    rates = []
    for patient, is_high in zip(self._patients, self.observed_is_high()):
      if is_high:
        probability_correct = self.compute_patient_prob_in_range(
          patient,
          self._parameter_threshold,
          self._parameter_max,
          prior_alpha=prior_alpha,
          prior_beta=prior_beta,
        )
      else:
        probability_correct = self.compute_patient_prob_in_range(
          patient,
          self._parameter_min,
          self._parameter_threshold,
          prior_alpha=prior_alpha,
          prior_beta=prior_beta,
        )
      rates.append(1.0 - probability_correct)
    return np.asarray(rates, dtype=float)

  def _survival_arrays(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.array(
      [patient.time for patient, keep in zip(self._patients, mask) if keep],
      dtype=float,
    )
    censored = np.array(
      [patient.censored for patient, keep in zip(self._patients, mask) if keep],
      dtype=bool,
    )
    return times, censored


class McSimexForKaplanMeier(McSimexBase):
  """
  MC-SIMEX point estimate of a Kaplan-Meier curve (no confidence bands).
  """

  def estimate_survival(  # pylint: disable=too-many-locals
    self,
    times_for_plot: list[float] | None = None,
    *,
    prior_alpha: float | None = None,
    prior_beta: float | None = None,
  ) -> dict:
    """
    Extrapolate the naive KM of the observed-range group to lambda = -1.
    """
    if prior_alpha is None:
      prior_alpha = self._prior_alpha
    if prior_beta is None:
      prior_beta = self._prior_beta
    observed_in_range = np.array(
      [self.observed_in_range(patient) for patient in self._patients],
      dtype=bool,
    )
    flip_rates = np.array(
      [
        self.flip_rate_for_range(
          patient,
          self._parameter_min,
          self._parameter_max,
          prior_alpha=prior_alpha,
          prior_beta=prior_beta,
        )
        for patient in self._patients
      ],
      dtype=float,
    )

    group_patients = [
      patient
      for patient, in_range in zip(self._patients, observed_in_range)
      if in_range
    ]
    death_times = sorted({
      patient.time for patient in group_patients if not patient.censored
    })
    if times_for_plot is None:
      if not death_times:
        raise ValueError("No death events found in the observed parameter range.")
      last_time = max(
        [*death_times, *(patient.time for patient in group_patients)]
      )
      times_for_plot = [0.0] + death_times + [1.1 * last_time]

    lambda_means = []
    for simex_lambda in self._lambda_grid:
      replicates = []
      n_draws = 1 if simex_lambda == 0.0 else self._B
      for _ in range(n_draws):
        labels = self.simulate_labels(
          observed_in_range, flip_rates, simex_lambda
        )
        survival = _naive_km_survival(
          self._patients, labels, times_for_plot
        )
        if survival is not None:
          replicates.append(survival)
      if not replicates:
        lambda_means.append(np.full(len(times_for_plot), np.nan))
      else:
        lambda_means.append(np.mean(np.vstack(replicates), axis=0))

    stacked = np.vstack(lambda_means)
    extrapolated = np.array([
      extrapolate_quadratic(np.asarray(self._lambda_grid), stacked[:, i])
      for i in range(stacked.shape[1])
    ])
    extrapolated = np.clip(extrapolated, 0.0, 1.0)
    for i, time in enumerate(times_for_plot):
      if time == 0.0:
        extrapolated[i] = 1.0

    return {
      'survival_probabilities': extrapolated,
      'times_for_plot': list(times_for_plot),
      'death_times': death_times,
      'method': 'mc_simex',
      'parameter_min': self._parameter_min,
      'parameter_max': self._parameter_max,
      'B': self._B,
      'lambda_grid': self._lambda_grid,
    }


def _naive_km_survival(
  patients: list[Patient],
  in_group: np.ndarray,
  times_for_plot: list[float],
) -> np.ndarray | None:
  if not np.any(in_group):
    return None
  km_patients = []
  for patient, flag in zip(patients, in_group):
    if patient.censored is None:
      raise ValueError("Censored status not set")
    km_patients.append(
      KaplanMeierPatient(
        time=patient.time,
        censored=patient.censored,
        parameter=1.0 if flag else 0.0,
      )
    )
  instance = KaplanMeierInstance(
    km_patients,
    parameter_min=0.5,
    parameter_max=np.inf,
  )
  if not instance.patients:
    return None
  return instance.survival_probabilities(times_for_plot=times_for_plot)


class McSimexForLogrank(McSimexWithThreshold):
  """
  MC-SIMEX for the two-sample logrank statistic, then a chi-square p-value.
  """

  def estimate_pvalue(  # pylint: disable=too-many-locals
    self,
    *,
    prior_alpha: float | None = None,
    prior_beta: float | None = None,
  ) -> dict:
    """
    Extrapolate U^2/V to lambda = -1 and convert to a chi-square(1) p-value.
    """
    if prior_alpha is None:
      prior_alpha = self._prior_alpha
    if prior_beta is None:
      prior_beta = self._prior_beta
    mask = self.analysis_mask()
    if not np.any(mask):
      raise ValueError("No patients in [parameter_min, parameter_max).")

    observed_high = self.observed_is_high()[mask]
    flip_rates = self.flip_rates_high_low(
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )[mask]
    times, censored = self._survival_arrays(mask)

    if not np.any(~censored):
      raise ValueError("No death events found in patient data.")

    n_low_observed = int(np.count_nonzero(~observed_high))
    n_high_observed = int(np.count_nonzero(observed_high))

    lambda_means = []
    for simex_lambda in self._lambda_grid:
      replicates = []
      n_draws = 1 if simex_lambda == 0.0 else self._B
      for _ in range(n_draws):
        labels = self.simulate_labels(observed_high, flip_rates, simex_lambda)
        statistic = _naive_logrank_statistic(times, censored, labels)
        if statistic is not None:
          replicates.append(statistic)
      if not replicates:
        lambda_means.append(np.nan)
      else:
        lambda_means.append(float(np.mean(replicates)))

    extrapolated = extrapolate_quadratic(
      np.asarray(self._lambda_grid),
      np.asarray(lambda_means, dtype=float),
    )
    logrank_statistic = max(extrapolated, 0.0)
    p_value = float(scipy.stats.chi2.sf(logrank_statistic, df=1))

    return {
      'p_value': p_value,
      'logrank_statistic': logrank_statistic,
      'n_low_observed': n_low_observed,
      'n_high_observed': n_high_observed,
      'method': 'mc_simex',
      'B': self._B,
      'lambda_grid': self._lambda_grid,
    }


class McSimexForCoxPH(McSimexWithThreshold):
  """
  MC-SIMEX for a Cox log hazard ratio, with a Wald interval and Wald Δ2NLL.
  """

  def __init__(self, *args, **kwargs):
    """
    Same arguments as McSimexWithThreshold, plus a cache for the Wald fit.
    """
    super().__init__(*args, **kwargs)
    self._cached_estimate: dict | None = None
    self._cached_priors: tuple[float, float] | None = None

  def estimate_hazard_ratio(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    self,
    *,
    prior_alpha: float | None = None,
    prior_beta: float | None = None,
  ) -> dict:
    """
    Extrapolate log H and its sampling variance to lambda = -1.

    Variance-versus-lambda uses the Breslow observed information of each
    successful replicate (Cook-Stefanski / R simex style), then a
    quadratic extrapolation. The interval is Wald, not a likelihood-ratio
    interval.
    """
    if prior_alpha is None:
      prior_alpha = self._prior_alpha
    if prior_beta is None:
      prior_beta = self._prior_beta
    priors = (prior_alpha, prior_beta)
    if self._cached_estimate is not None and self._cached_priors == priors:
      return self._cached_estimate

    mask = self.analysis_mask()
    if not np.any(mask):
      raise ValueError("No patients in [parameter_min, parameter_max).")

    observed_high = self.observed_is_high()[mask]
    flip_rates = self.flip_rates_high_low(
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )[mask]
    times, censored = self._survival_arrays(mask)

    lambda_log_hr = []
    lambda_variance = []
    naive_log_hr = None
    for simex_lambda in self._lambda_grid:
      log_hrs = []
      variances = []
      n_draws = 1 if simex_lambda == 0.0 else self._B
      for _ in range(n_draws):
        labels = self.simulate_labels(observed_high, flip_rates, simex_lambda)
        fit = mle_log_hazard_ratio(times, censored, labels)
        if fit is None:
          continue
        log_hrs.append(fit[0])
        if np.isfinite(fit[1]):
          variances.append(fit[1])
      if not log_hrs:
        lambda_log_hr.append(np.nan)
        lambda_variance.append(np.nan)
        continue
      mean_log_hr = float(np.mean(log_hrs))
      lambda_log_hr.append(mean_log_hr)
      if simex_lambda == 0.0:
        naive_log_hr = mean_log_hr
      if variances:
        lambda_variance.append(float(np.mean(variances)))
      else:
        lambda_variance.append(np.nan)

    log_hazard_ratio = extrapolate_quadratic(
      np.asarray(self._lambda_grid),
      np.asarray(lambda_log_hr, dtype=float),
    )
    variance = extrapolate_quadratic(
      np.asarray(self._lambda_grid),
      np.asarray(lambda_variance, dtype=float),
    )
    finite_variances = [
      value for value in lambda_variance if np.isfinite(value)
    ]
    if variance <= 0.0:
      if finite_variances:
        variance = max(min(finite_variances), VARIANCE_FLOOR)
      else:
        variance = VARIANCE_FLOOR
    se_log_hazard_ratio = float(np.sqrt(variance))
    hazard_ratio = float(np.exp(log_hazard_ratio))
    ci_lower = float(np.exp(log_hazard_ratio - WALD_Z_95 * se_log_hazard_ratio))
    ci_upper = float(np.exp(log_hazard_ratio + WALD_Z_95 * se_log_hazard_ratio))
    naive_hazard_ratio = (
      None if naive_log_hr is None else float(np.exp(naive_log_hr))
    )

    estimate = {
      'hazard_ratio': hazard_ratio,
      'log_hazard_ratio': log_hazard_ratio,
      'se_log_hazard_ratio': se_log_hazard_ratio,
      'ci_lower': ci_lower,
      'ci_upper': ci_upper,
      'confidence_level': 0.95,
      'naive_hazard_ratio': naive_hazard_ratio,
      'method': 'mc_simex',
      'B': self._B,
      'lambda_grid': self._lambda_grid,
    }
    self._cached_estimate = estimate
    self._cached_priors = priors
    return estimate

  def compute_2nll_at_hazard_ratio(
    self,
    hazard_ratio: float,
    *,
    prior_alpha: float | None = None,
    prior_beta: float | None = None,
  ) -> scipy.optimize.OptimizeResult:
    """
    Wald quadratic (log H - log H_hat)^2 / sigma_hat^2.

    This is not a profile likelihood and must not be labeled as one.
    """
    if hazard_ratio <= 0.0:
      raise ValueError(f"hazard_ratio must be positive, got {hazard_ratio}")
    estimate = self.estimate_hazard_ratio(
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )
    delta = (
      (np.log(hazard_ratio) - estimate['log_hazard_ratio']) ** 2
      / estimate['se_log_hazard_ratio'] ** 2
    )
    return scipy.optimize.OptimizeResult(
      x=float(delta),
      success=True,
      hazard_ratio=hazard_ratio,
      log_hazard_ratio=float(np.log(hazard_ratio)),
      fitted_hazard_ratio=estimate['hazard_ratio'],
      se_log_hazard_ratio=estimate['se_log_hazard_ratio'],
      ci_lower=estimate['ci_lower'],
      ci_upper=estimate['ci_upper'],
      cox_2NLL=float(delta),
      patient_2NLL=0.0,
      method='mc_simex_wald',
    )
