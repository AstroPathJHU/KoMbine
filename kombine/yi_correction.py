"""
Yi's misclassification correction methods for survival analysis.

This module implements Yi's correction method (Section 3.7.1 from
"Statistical Analysis with Measurement Error or Misclassification", 2017)
for discrete covariate misclassification in survival analysis.

Yi's methods use inverse probability weighting to account for measurement uncertainty,
providing an alternative to integer optimization (MINLP) approaches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize
import scipy.stats

from .utilities import prob_poisson_density_exceeds_threshold

if TYPE_CHECKING:
  from .datacard import Patient


class YiCorrectionBase:  # pylint: disable=too-few-public-methods
  """
  Base class for Yi's misclassification correction methods.

  Provides common functionality for computing patient probabilities
  of belonging to the high group based on their observable measurements.
  """

  def __init__(
    self,
    patients: list[Patient],
    parameter_threshold: float,
  ):
    """
    Initialize Yi's correction calculator.

    Parameters
    ----------
    patients : list[Patient]
        List of Patient objects with observable measurements.
    parameter_threshold : float
        Threshold separating "low" and "high" groups.
    """
    self._patients = patients
    self._parameter_threshold = parameter_threshold

  def compute_patient_prob_high(
    self,
    patient: Patient,
    *,
    method: str = 'bayesian',
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
  ) -> float:
    """
    Compute the probability that a patient belongs to the high group.

    This method uses the patient's observable measurement data to calculate
    a probability rather than making a deterministic classification.

    Parameters
    ----------
    patient : Patient
        The patient to compute probability for.
    method : str, optional
        Method for probability calculation ('bayesian' or 'normal_approx').
    prior_alpha : float, optional
        Alpha parameter for Bayesian prior.
    prior_beta : float, optional
        Beta parameter for Bayesian prior.

    Returns
    -------
    float
        Probability that patient is in the high group (0.0 to 1.0).
    """
    # Try probabilistic classification first (for Poisson density measurements)
    obs = getattr(patient, 'observable', None)
    if obs is not None and hasattr(obs, 'numerator') and hasattr(obs, 'denominator'):
      # Poisson density measurement - use probabilistic weighting
      return prob_poisson_density_exceeds_threshold(
        obs.numerator,
        obs.denominator,
        self._parameter_threshold,
        method=method,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
      )

    # Fall back to deterministic classification based on observed parameter
    return 1.0 if patient.observed_parameter > self._parameter_threshold else 0.0


class YiCorrectionForLogrank(YiCorrectionBase):
  """
  Yi's misclassification correction for the logrank test.

  This class implements the corrected logrank test using inverse probability
  weighting to account for measurement error in group assignment.
  """

  def compute_pvalue(  # pylint: disable=too-many-locals
    self,
    *,
    method: str = 'bayesian',
    prior_alpha: float = 0.5,
    prior_beta: float = 0.0,
  ) -> dict:
    """
    Calculate p-value using Yi's misclassification correction method.

    This implements Yi's correction for discrete covariate misclassification applied
    to the logrank test. Uses inverse probability weighting to account for
    measurement uncertainty.

    Parameters
    ----------
    method : str, optional
        Method for estimating misclassification probabilities:
        - 'bayesian': Full Bayesian posterior (default, more accurate)
        - 'normal_approx': Normal approximation (faster, less accurate for small counts)
    prior_alpha : float, optional
        Alpha parameter for Gamma prior (Bayesian method only). Default 0.5 (Jeffreys).
    prior_beta : float, optional
        Beta parameter for Gamma prior (Bayesian method only). Default 0.0.

    Returns
    -------
    dict
        Dictionary containing:
        - 'p_value' : float
            The p-value from the corrected logrank test.
        - 'logrank_statistic' : float
            The corrected logrank test statistic (chi-square distributed, df=1).
        - 'U' : float
            Sum of (observed - expected) weighted deaths for low group.
        - 'V' : float
            Sum of weighted variances.
        - 'n_low_observed' : int
            Number of patients observed in low group.
        - 'n_high_observed' : int
            Number of patients observed in high group.

    Notes
    -----
    Yi's method (Statistical Analysis with Measurement Error or Misclassification, 2017)
    uses inverse probability weighting to correct for misclassification.

    This improved implementation uses per-patient probability weighting:
    1. For each patient, compute P(true group = high | observed data)
    2. Weight patient's contributions by their individual probabilities
    3. Compute corrected logrank test using weighted risk sets and event counts

    The weighted logrank test formula:
    - At each death time t, compute weighted risk sets r_k^*(t) and deaths d_k^*(t)
    - U^* = sum_t [d_1^*(t) - r_1^*(t) * (d_0^* + d_1^*) / (r_0^* + r_1^*)]
    - V^* = sum_t [r_0^* * r_1^* * (d_0^* + d_1^*) * (r_0^* + r_1^* - d_0^* - d_1^*) /
                    ((r_0^* + r_1^*)^2 * (r_0^* + r_1^* - 1))]
    - Logrank statistic = (U^*)^2 / V^* ~ chi^2(1)

    This differs from KoMbine's MINLP approach:
    - Yi: Probabilistic weighting (no optimization, fractional assignments)
    - MINLP: Integer optimization over discrete assignments with NLL penalties

    The per-patient approach is more accurate than aggregate misclassification
    matrices, as it accounts for individual measurement uncertainty.

    See Section 3.7.1 of Yi's book for theoretical foundation. The logrank
    extension follows naturally from the weighted risk set principle (Equation 3.57).
    """

    # Count patients by observed group
    n_low_observed = sum(
      1 for p in self._patients
      if p.observed_parameter <= self._parameter_threshold
    )
    n_high_observed = len(self._patients) - n_low_observed

    # Get all unique death times
    all_death_times = sorted(set(
      p.time for p in self._patients if not p.censored
    ))

    if not all_death_times:
      raise ValueError("No death events found in patient data.")

    # Calculate weighted logrank test statistic using per-patient probabilities
    U = 0.0  # Sum of (observed - expected) for low group (weighted)
    V = 0.0  # Sum of variances (weighted)

    for death_time in all_death_times:
      # Compute weighted risk sets and death counts at this time
      r_low_weighted = 0.0
      r_high_weighted = 0.0
      d_low_weighted = 0.0
      d_high_weighted = 0.0

      for patient in self._patients:
        if patient.time < death_time:
          # Patient not at risk
          continue

        # Compute this patient's probability of being in high group
        prob_high = self.compute_patient_prob_high(
          patient,
          method=method,
          prior_alpha=prior_alpha,
          prior_beta=prior_beta,
        )

        prob_low = 1.0 - prob_high

        # Add to risk sets weighted by individual probabilities
        r_low_weighted += prob_low
        r_high_weighted += prob_high

        # If patient dies at this time, add to death counts
        if patient.time == death_time and not patient.censored:
          d_low_weighted += prob_low
          d_high_weighted += prob_high

      # Compute logrank components for this death time
      r_total_weighted = r_low_weighted + r_high_weighted
      d_total_weighted = d_low_weighted + d_high_weighted

      if r_total_weighted <= 0 or d_total_weighted <= 0:
        continue

      # Expected deaths in low group under null hypothesis
      expected_d_low = r_low_weighted * d_total_weighted / r_total_weighted

      # Variance for this time point
      if r_total_weighted > 1:
        variance_t = (
          r_low_weighted * r_high_weighted * d_total_weighted *
          (r_total_weighted - d_total_weighted)
        ) / (
          r_total_weighted * r_total_weighted * (r_total_weighted - 1)
        )
      else:
        variance_t = 0.0

      # Accumulate test statistic components
      U += d_low_weighted - expected_d_low
      V += variance_t

    if V <= 0:
      # No variance means no information for comparison
      return {
        'p_value': 1.0,
        'logrank_statistic': 0.0,
        'U': U,
        'V': V,
        'n_low_observed': n_low_observed,
        'n_high_observed': n_high_observed,
      }

    # Logrank test statistic
    logrank_statistic = U * U / V

    # Calculate p-value using chi-square distribution with 1 degree of freedom
    p_value = 1.0 - scipy.stats.chi2.cdf(logrank_statistic, df=1).item()

    return {
      'p_value': p_value,
      'logrank_statistic': logrank_statistic,
      'U': U,
      'V': V,
      'n_low_observed': n_low_observed,
      'n_high_observed': n_high_observed,
    }


class YiCorrectionForCoxPH(YiCorrectionBase):
  """
  Yi's misclassification correction for Cox proportional hazards model.

  This class implements the corrected Cox partial likelihood using inverse
  probability weighting to account for measurement error in covariate values.
  """

  LOG_ZERO_EPSILON_DEFAULT = 1e-6

  def compute_2nll_at_hazard_ratio(  # pylint: disable=too-many-locals
    self,
    hazard_ratio: float,
    *,
    method: str = 'bayesian',
    prior_alpha: float = 0.5,
    prior_beta: float = 0.0,
  ) -> scipy.optimize.OptimizeResult:
    """
    Compute 2NLL at a specific hazard ratio using Yi's misclassification correction.

    This implements Yi's misclassification correction method (Section 3.7.1) for
    the Cox proportional hazards model with discrete misclassified covariates.
    Instead of using integer optimization (MINLP), this method uses inverse probability
    weighting to account for measurement uncertainty.

    Parameters
    ----------
    hazard_ratio : float
        The hazard ratio value at which to evaluate the 2NLL.
        H = 1 corresponds to equal hazards (null hypothesis).
        H > 1 means the high group has higher hazard (worse outcomes).
        H < 1 means the low group has higher hazard.
    method : str, optional
        Method for estimating misclassification probabilities:
        - 'bayesian': Full Bayesian posterior (default, more accurate)
        - 'normal_approx': Normal approximation (faster, less accurate for small counts)
    prior_alpha : float, optional
        Alpha parameter for Gamma prior (Bayesian method only). Default 0.5 (Jeffreys).
    prior_beta : float, optional
        Beta parameter for Gamma prior (Bayesian method only). Default 0.0.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Optimization result with attributes:
        - x : float
            The 2NLL value at the specified hazard ratio using Yi's correction.
        - success : bool
            Always True for Yi's method (no optimization).
        - patients_low : list
            Indices of patients nominally assigned to low group.
        - patients_high : list
            Indices of patients nominally assigned to high group.
        - hazard_ratio : float
            The hazard ratio value (should equal the input).
        - log_hazard_ratio : float
            Natural logarithm of the hazard ratio.
        - cox_2NLL : float
            Twice the corrected Cox partial likelihood contribution.
        - patient_2NLL : float
            Always 0.0 for Yi's method (no patient-wise penalties).
        - method : str
            'yi_correction'

    Notes
    -----
    Yi's method (Statistical Analysis with Measurement Error or Misclassification, 2017)
    uses inverse probability weighting to correct for misclassification.

    This improved implementation uses per-patient probability weighting:
    1. For each patient, compute P(true group = high | observed data)
    2. Weight patient's contributions by their individual probabilities
    3. Compute corrected Cox partial likelihood using weighted risk sets

    This differs from KoMbine's MINLP approach:
    - Yi: Probabilistic weighting (no optimization, fractional assignments)
    - MINLP: Integer optimization over discrete assignments with NLL penalties

    The per-patient approach is more accurate than aggregate misclassification
    matrices, as it accounts for individual measurement uncertainty rather than
    assuming uniform misclassification probabilities within groups.

    See Section 3.7.1 of Yi's book for theoretical foundation.
    """
    log_hazard_ratio = np.log(hazard_ratio)

    # Separate patients by observed group for reporting
    patients_observed_low = []
    patients_observed_high = []

    for i, p in enumerate(self._patients):
      if p.observed_parameter > self._parameter_threshold:
        patients_observed_high.append(i)
      else:
        patients_observed_low.append(i)

    # Collect unique death times
    death_times = sorted(set(
      p.time for p in self._patients if not p.censored
    ))

    # Compute corrected Cox partial likelihood using per-patient weighted risk sets
    # Following Yi Section 3.7.1, but with individual patient probabilities

    log_likelihood = 0.0

    for t_death in death_times:
      # For each death time, compute weighted risk sets
      # Each patient contributes to both groups weighted by their probability

      # Weighted contributions for patients at risk at time t_death
      r_low_weighted = 0.0
      r_high_weighted = 0.0

      # Weighted death counts at this time
      d_low_weighted = 0.0
      d_high_weighted = 0.0

      for patient in self._patients:
        if patient.time < t_death:
          # Patient not at risk
          continue

        # Compute this patient's probability of being in high group
        prob_high = self.compute_patient_prob_high(
          patient,
          method=method,
          prior_alpha=prior_alpha,
          prior_beta=prior_beta,
        )

        prob_low = 1.0 - prob_high

        # Add to risk sets weighted by individual probabilities
        r_low_weighted += prob_low
        r_high_weighted += prob_high

        # If patient dies at this time, add to death counts
        if patient.time == t_death and not patient.censored:
          d_low_weighted += prob_low
          d_high_weighted += prob_high

      # Compute Cox partial likelihood contribution at this death time
      # Following Breslow approximation for tied deaths:
      # L_i = [H^d_high * (r_low + H*r_high)^{-d_total}]
      # where d_total = d_low + d_high

      d_total_weighted = d_low_weighted + d_high_weighted

      if d_total_weighted > 0:
        # Numerator: H^{d_high}
        numerator_log = d_high_weighted * log_hazard_ratio

        # Denominator: (r_low + H*r_high)^{d_total}
        denominator = r_low_weighted + hazard_ratio * r_high_weighted

        if denominator <= 0:
          # Avoid log(0) or log(negative)
          # This shouldn't happen with proper weighting, but handle gracefully
          denominator = self.LOG_ZERO_EPSILON_DEFAULT

        denominator_log = d_total_weighted * np.log(denominator)

        # Add to log likelihood
        log_likelihood += numerator_log - denominator_log

    # Convert to 2NLL
    cox_2nll = -2.0 * log_likelihood

    # For Yi's method, there are no patient-wise penalties (all uncertainty is in weights)
    patient_2nll = 0.0
    twonll = cox_2nll + patient_2nll

    result = scipy.optimize.OptimizeResult(
      x=twonll,
      success=True,
      patients_low=patients_observed_low,
      patients_high=patients_observed_high,
      hazard_ratio=hazard_ratio,
      log_hazard_ratio=log_hazard_ratio,
      cox_2NLL=cox_2nll,
      patient_2NLL=patient_2nll,
      method='yi_correction',
    )

    return result


class YiCorrectionForKaplanMeier(YiCorrectionBase):
  """
  Yi's misclassification correction applied to Kaplan-Meier curve estimation.

  This class implements weighted Kaplan-Meier survival estimation using inverse
  probability weighting to account for measurement error in group assignment.
  Unlike MINLP approaches, this provides point estimates without confidence intervals.
  """

  def compute_weighted_survival_probabilities(
    self,
    times_for_plot: list[float] | None = None,
    *,
    method: str = 'bayesian',
    prior_alpha: float = 0.5,
    prior_beta: float = 0.0,
  ) -> dict:
    """
    Calculate weighted Kaplan-Meier survival probabilities using Yi's correction.

    This implements the weighted Kaplan-Meier estimator using inverse probability
    weighting to account for measurement uncertainty in group assignment.
    Each patient contributes to the survival curve with weight equal to their
    probability of belonging to the observed (or alternative) group.

    Parameters
    ----------
    times_for_plot : list[float], optional
        Time points at which to calculate survival probabilities.
        If None, uses unique death times from patient data.
    method : str, optional
        Method for estimating misclassification probabilities:
        - 'bayesian': Full Bayesian posterior (default, more accurate)
        - 'normal_approx': Normal approximation (faster, less accurate for small counts)
    prior_alpha : float, optional
        Alpha parameter for Gamma prior (Bayesian method only). Default 0.5 (Jeffreys).
    prior_beta : float, optional
        Beta parameter for Gamma prior (Bayesian method only). Default 0.0.

    Returns
    -------
    dict
        Dictionary containing:
        - 'survival_probabilities' : np.ndarray
            Weighted survival probabilities at each time point in times_for_plot.
        - 'times_for_plot' : list[float]
            The time points where probabilities were calculated.
        - 'n_at_risk_weighted' : list[float]
            Weighted number of patients at risk at each death time.
        - 'n_deaths_weighted' : list[float]
            Weighted number of deaths at each death time.
        - 'n_at_risk' : list[int]
            Unweighted number of patients at risk at each death time.
        - 'n_deaths' : list[int]
            Unweighted number of deaths at each death time.
        - 'death_times' : list[float]
            Unique death times where events occurred.
        - 'method' : str
            'yi_correction'

    Notes
    -----
    Yi's method (Statistical Analysis with Measurement Error or Misclassification, 2017)
    uses inverse probability weighting to correct for misclassification.

    The weighted Kaplan-Meier estimator uses:
    - Weighted at-risk counts: n*(t) = sum of patient weights for patients at risk at t
    - Weighted death counts: d*(t) = sum of patient weights for patients who died at t
    - Weighted KM: S(t) = product of (1 - d*(t_i)/n*(t_i)) for all t_i <= t

    where each patient's weight depends on their probability of true group membership
    based on their observable measurements.

    This differs from KoMbine's MINLP approach:
    - Yi: Probabilistic weighting (point estimates only, no confidence intervals)
    - MINLP: Integer optimization with likelihood-based confidence intervals

    The per-patient weighting approach is more accurate than aggregate misclassification
    matrices, as it accounts for individual measurement uncertainty.

    See Section 3.7.1 of Yi's book for theoretical foundation. The weighted KM
    extension follows naturally from applying per-patient weights to the standard
    Kaplan-Meier formula.

    Examples
    --------
    >>> from kombine.yi_correction import YiCorrectionForKaplanMeier
    >>> from kombine.datacard import Patient
    >>> yi = YiCorrectionForKaplanMeier(patients=patient_list, parameter_threshold=0.5)
    >>> result = yi.compute_weighted_survival_probabilities()
    >>> print(result['survival_probabilities'])
    """
    # Get unique death times from all patients
    death_times = sorted(set(
      p.time for p in self._patients if not p.censored
    ))

    if not death_times:
      raise ValueError("No death events found in patient data.")

    # If times_for_plot not provided, use standard times based on death times
    if times_for_plot is None:
      times_for_plot = (
        [0.0]
        + death_times
        + [1.1 * max((
          max(death_times, default=0),
          max((p.time for p in self._patients if p.censored), default=0)
        ))]
      )

    # Track weighted and unweighted counts at each death time
    n_at_risk_by_time = []
    n_deaths_by_time = []
    n_at_risk_weighted_by_time = []
    n_deaths_weighted_by_time = []

    # Compute weighted and unweighted counts at each death time
    for t_death in death_times:
      n_at_risk = 0
      n_deaths = 0
      n_at_risk_weighted = 0.0
      n_deaths_weighted = 0.0

      for patient in self._patients:
        # Compute patient's probability of being in the high group
        prob_high = self.compute_patient_prob_high(
          patient,
          method=method,
          prior_alpha=prior_alpha,
          prior_beta=prior_beta,
        )

        # Patient is at risk if their time >= death_time
        if patient.time >= t_death:
          n_at_risk += 1
          n_at_risk_weighted += prob_high

        # Patient dies at this time if time == death_time and not censored
        if patient.time == t_death and not patient.censored:
          n_deaths += 1
          n_deaths_weighted += prob_high

      n_at_risk_by_time.append(n_at_risk)
      n_deaths_by_time.append(n_deaths)
      n_at_risk_weighted_by_time.append(n_at_risk_weighted)
      n_deaths_weighted_by_time.append(n_deaths_weighted)

    # Compute weighted Kaplan-Meier survival probabilities at requested times
    survival_probabilities = np.zeros(len(times_for_plot))
    survival_probabilities[0] = 1.0  # Probability at t=0 is 1.0

    for i, t in enumerate(times_for_plot):
      if t == 0.0:
        survival_probabilities[i] = 1.0
        continue

      # Find all death times <= t and accumulate survival probability
      s_t = 1.0
      for j, t_death in enumerate(death_times):
        if t_death > t:
          break

        # Weighted KM formula: S(t) = product of (1 - d*(t_i)/n*(t_i))
        n_at_risk_w = n_at_risk_weighted_by_time[j]
        n_deaths_w = n_deaths_weighted_by_time[j]

        if n_at_risk_w > 0 and n_deaths_w > 0:
          s_t *= (1.0 - n_deaths_w / n_at_risk_w)
        elif n_deaths_w > n_at_risk_w and n_at_risk_w > 0:
          # Handle edge case: weighted deaths > weighted at-risk
          # This can happen with probabilistic weights; clamp to 0
          s_t = 0.0
          break

      survival_probabilities[i] = s_t

    return {
      'survival_probabilities': survival_probabilities,
      'times_for_plot': times_for_plot,
      'n_at_risk_weighted': n_at_risk_weighted_by_time,
      'n_deaths_weighted': n_deaths_weighted_by_time,
      'n_at_risk': n_at_risk_by_time,
      'n_deaths': n_deaths_by_time,
      'death_times': death_times,
      'method': 'yi_correction',
    }
