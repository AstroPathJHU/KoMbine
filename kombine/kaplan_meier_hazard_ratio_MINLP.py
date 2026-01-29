"""
MINLP solver for calculating hazard ratios and likelihood scans for two Kaplan-Meier curves.
This module provides functionality to compute the negative log likelihood as a function
of the hazard ratio H, enabling profile likelihood analyses and confidence intervals.
"""
# pylint: disable=too-many-lines

import warnings
from typing import Optional

from gurobipy import GRB
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.stats

from .kaplan_meier_p_value_MINLP import MINLPforKMPValue
from .utilities import (
  LOG_ZERO_EPSILON_DEFAULT,
  estimate_misclassification_matrix,
  invert_misclassification_matrix,
)


class MINLPforKMHazardRatio(MINLPforKMPValue):
  """
  MINLP solver for calculating hazard ratios and likelihood scans.
  Extends MINLPforKMPValue to provide hazard ratio-specific functionality.
  """

  def __init__(  # pylint: disable=too-many-arguments
    self,
    all_patients,
    *,
    parameter_min: float = -np.inf,
    parameter_threshold: float,
    parameter_max: float = np.inf,
    log_zero_epsilon: float = LOG_ZERO_EPSILON_DEFAULT,
    tie_handling: str = "breslow",
    log_hazard_ratio_bounds: tuple[float, float] = (-10.0, 10.0),
  ):
    """
    Initialize the MINLP solver for hazard ratio calculation.

    Parameters
    ----------
    all_patients : list[KaplanMeierPatientNLL]
        List of all patients with their NLL penalties.
    parameter_min : float, optional
        Minimum parameter value for the "low" group. Default is -inf.
    parameter_threshold : float
        Threshold separating "low" and "high" groups.
    parameter_max : float, optional
        Maximum parameter value for the "high" group. Default is inf.
    log_zero_epsilon : float, optional
        Small epsilon value to avoid log(0). Default from utilities.
    tie_handling : str, optional
        Method for handling tied death times. Currently only "breslow" is supported.
    log_hazard_ratio_bounds : tuple[float, float], optional
        Bounds on log(hazard ratio) for the Gurobi model, as (lower_bound, upper_bound).
        These correspond to hazard ratio bounds of (exp(lb), exp(ub)).
        Default is (-10.0, 10.0), allowing HR in [0.000045, 22026].
        Increase these if you need to explore more extreme hazard ratios.
    """
    super().__init__(
      all_patients,
      parameter_min=parameter_min,
      parameter_threshold=parameter_threshold,
      parameter_max=parameter_max,
      log_zero_epsilon=log_zero_epsilon,
      tie_handling=tie_handling,
      log_hazard_ratio_bounds=log_hazard_ratio_bounds,
    )

  def compute_2nll_at_hazard_ratio(  # pylint: disable=too-many-locals
    self,
    hazard_ratio: float,
    *,
    cox_only: bool = False,
  ) -> scipy.optimize.OptimizeResult:
    """
    Compute the twice negative log likelihood (2NLL) at a specific hazard ratio.

    This method fixes the hazard ratio H to the specified value and optimizes
    over patient assignments to minimize the total NLL.

    Parameters
    ----------
    hazard_ratio : float
        The hazard ratio value at which to evaluate the 2NLL.
        H = 1 corresponds to equal hazards (null hypothesis).
        H > 1 means the high group has higher hazard (worse outcomes).
        H < 1 means the low group has higher hazard.
    cox_only : bool, optional
        If True, fix patient assignments to their nominal values (based on observed
        parameters) and only optimize the Cox likelihood. If False, allow patient
        assignments to float. Default is False.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Optimization result with attributes:
        - x : float
            The 2NLL value at the specified hazard ratio.
        - success : bool
            Whether the optimization succeeded.
        - patients_low : list
            Indices of patients assigned to the low group.
        - patients_high : list
            Indices of patients assigned to the high group.
        - n_total_low : int
            Total number of patients in the low group.
        - n_alive_low : int
            Number of alive patients in the low group at the end.
        - n_total_high : int
            Total number of patients in the high group.
        - n_alive_high : int
            Number of alive patients in the high group at the end.
        - km_probability_low : float
            Kaplan-Meier survival probability for the low group at the last time point.
        - km_probability_high : float
            Kaplan-Meier survival probability for the high group at the last time point.
        - cox_2NLL : float
            Twice the Cox partial likelihood contribution to the NLL.
        - patient_2NLL : float
            Twice the patient-wise penalty contribution to the NLL.
        - hazard_ratio : float
            The hazard ratio value (should equal the input).
        - log_hazard_ratio : float
            Natural logarithm of the hazard ratio.
        - model : gp.Model
            The Gurobi model (for advanced users).
    """
    (  # pylint: disable=duplicate-code
      model,
      null_hypothesis_indicator,
      a,
      km_probability_at_time_low,
      km_probability_at_time_high,
      beta,
      use_cox_penalty_indicator,
    ) = self.gurobi_model

    # Set the hazard ratio constraint
    log_hazard_ratio = np.log(hazard_ratio)
    beta_var = model.getVarByName("log_hazard_ratio")
    if beta_var is None:
      raise ValueError("Could not find log_hazard_ratio variable in model")

    # Remove existing hazard ratio constraint if any
    hazard_ratio_constraint_name = "fixed_hazard_ratio_constraint"
    existing_constr = model.getConstrByName(hazard_ratio_constraint_name)
    if existing_constr is not None:
      model.remove(existing_constr)

    # Add new constraint fixing the hazard ratio
    model.addConstr(
      beta_var == log_hazard_ratio,
      name=hazard_ratio_constraint_name
    )

    # Set null hypothesis indicator to False (we're fixing HR, not testing H=1)
    self.update_model_for_null_hypothesis_or_not(model, null_hypothesis_indicator, False)

    # Set cox_only mode
    self.update_model_with_cox_only_constraints(model, a, cox_only)

    # Set patient_wise_only to False (we want Cox penalty enabled)
    self.update_model_with_patient_wise_only_constraint(
      model,
      beta=beta,
      null_hypothesis_indicator=null_hypothesis_indicator,
      patient_wise_only=False,
      use_cox_penalty_indicator=use_cox_penalty_indicator,
    )

    # Optimize
    model.optimize()

    if model.status != GRB.OPTIMAL:
      raise ValueError(f"Optimization failed with status {model.status}")

    # Extract results
    twonll = model.ObjVal
    patients_low, patients_high = self._extract_patients_per_curve(a)
    patient_penalty = self._compute_patient_wise_penalty_value(a)
    cox_penalty = self._compute_cox_penalty(model)

    # Extract curve statistics
    (n_total_low, n_alive_low, km_prob_low,
     n_total_high, n_alive_high, km_prob_high) = (
      self._extract_curve_statistics(
        model, km_probability_at_time_low, km_probability_at_time_high
      )
    )

    result = scipy.optimize.OptimizeResult(
      x=twonll,
      success=True,
      patients_low=patients_low,
      patients_high=patients_high,
      n_total_low=n_total_low,
      n_alive_low=n_alive_low,
      n_total_high=n_total_high,
      n_alive_high=n_alive_high,
      km_probability_low=km_prob_low,
      km_probability_high=km_prob_high,
      cox_2NLL=2 * cox_penalty,
      patient_2NLL=2 * patient_penalty,
      hazard_ratio=hazard_ratio,
      log_hazard_ratio=log_hazard_ratio,
      model=model,
    )

    # Check if we're at or near the bounds and warn
    self._check_hazard_ratio_bounds(hazard_ratio, log_hazard_ratio)

    return result

  def compute_2nll_at_hazard_ratio_yi(  # pylint: disable=too-many-locals
    self,
    hazard_ratio: float,
    *,
    original_patients: Optional[list] = None,
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
        Beta parameter for Gamma prior (Bayesian method only). Default 0.5.
    
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
        - misclassification_matrix : ndarray
            Estimated 2x2 misclassification matrix Π.
        - inverse_misclassification_matrix : ndarray
            Inverse matrix Π^{-1} used for weighting.
    
    Notes
    -----
    Yi's method (Statistical Analysis with Measurement Error or Misclassification, 2017)
    uses inverse probability weighting to correct for misclassification:
    
    1. Estimate misclassification matrix Π where Π[i,j] = P(observed=j | true=i)
    2. Compute inverse matrix Π^{-1} for weighting
    3. Each patient contributes fractionally to both groups weighted by Π^{-1}
    4. Compute corrected Cox partial likelihood using weighted risk sets
    
    This differs from KoMbine's MINLP approach:
    - Yi: Probabilistic weighting (no optimization, fractional assignments)
    - MINLP: Integer optimization over discrete assignments with NLL penalties
    
    See Section 3.7.1 of Yi's book for theoretical foundation.
    """
    log_hazard_ratio = np.log(hazard_ratio)
    
    # Use original patients for misclassification estimation if provided
    # (needed to access observable counts/areas), otherwise fall back to NLL patients
    patients_for_estimation = original_patients if original_patients is not None else self.all_patients
    
    # Estimate misclassification matrix from patient data
    Pi = estimate_misclassification_matrix(
      patients_for_estimation,
      self.parameter_threshold,
      method=method,
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )
    
    # Compute inverse matrix for weighting
    Pi_inv = invert_misclassification_matrix(Pi)
    
    # Separate patients by observed group for initial assignment
    patients_observed_low = []
    patients_observed_high = []
    
    for i, p in enumerate(self.all_patients):
      if p.observed_parameter > self.parameter_threshold:
        patients_observed_high.append(i)
      else:
        patients_observed_low.append(i)
    
    # Collect unique death times
    death_times = sorted(set(
      p.time for p in self.all_patients if not p.censored
    ))
    
    # Compute corrected Cox partial likelihood using Yi's weighted risk sets
    # Following Yi Section 3.7.1, Equation 3.57-3.58
    
    log_likelihood = 0.0
    
    for t_death in death_times:
      # For each death time, compute weighted risk sets
      # r_k^*(t) = sum over patients at risk of their weighted contributions
      
      # Weighted contributions for patients at risk at time t_death
      r_low_weighted = 0.0
      r_high_weighted = 0.0
      
      # Weighted death counts at this time
      d_low_weighted = 0.0
      d_high_weighted = 0.0
      
      for i, p in enumerate(self.all_patients):
        if p.time < t_death:
          # Patient not at risk
          continue
        
        # Determine observed group
        observed_group = 1 if p.observed_parameter > self.parameter_threshold else 0
        
        # Inverse probability weights for this patient
        # If observed in group 0 (low):
        #   Weight for true=0: Pi_inv[0, 0]
        #   Weight for true=1: Pi_inv[1, 0]
        # If observed in group 1 (high):
        #   Weight for true=0: Pi_inv[0, 1]
        #   Weight for true=1: Pi_inv[1, 1]
        
        if observed_group == 0:
          weight_low = Pi_inv[0, 0]
          weight_high = Pi_inv[1, 0]
        else:
          weight_low = Pi_inv[0, 1]
          weight_high = Pi_inv[1, 1]
        
        # Add to risk sets
        r_low_weighted += weight_low
        r_high_weighted += weight_high
        
        # If patient dies at this time, add to death counts
        if p.time == t_death and not p.censored:
          d_low_weighted += weight_low
          d_high_weighted += weight_high
      
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
          denominator = LOG_ZERO_EPSILON_DEFAULT
        
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
      misclassification_matrix=Pi,
      inverse_misclassification_matrix=Pi_inv,
      method='yi_correction',
    )
    
    return result

  def likelihood_scan_hazard_ratio(  # pylint: disable=too-many-arguments,too-many-locals
    self,
    hazard_ratio_values: Optional[npt.NDArray[np.float64]] = None,
    *,
    cox_only: bool = False,
    n_points: int = 50,
    hazard_ratio_min: float = 0.1,
    hazard_ratio_max: float = 10.0,
  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], scipy.optimize.OptimizeResult]:
    """
    Perform a likelihood scan over a range of hazard ratio values.

    This computes the 2NLL as a function of the hazard ratio H, which can be used
    to construct profile likelihood confidence intervals or visualize the likelihood surface.

    Parameters
    ----------
    hazard_ratio_values : array-like, optional
        Specific hazard ratio values at which to evaluate the likelihood.
        If None, a logarithmically-spaced grid is generated.
    cox_only : bool, optional
        If True, fix patient assignments to nominal values. Default is False.
    n_points : int, optional
        Number of points in the likelihood scan if hazard_ratio_values is None.
        Default is 50.
    hazard_ratio_min : float, optional
        Minimum hazard ratio for the scan if hazard_ratio_values is None.
        Default is 0.1.
    hazard_ratio_max : float, optional
        Maximum hazard ratio for the scan if hazard_ratio_values is None.
        Default is 10.0.

    Returns
    -------
    hazard_ratios : ndarray
        Array of hazard ratio values.
    twonll_values : ndarray
        Array of 2NLL values corresponding to each hazard ratio.
    best_fit_result : scipy.optimize.OptimizeResult
        The optimization result at the best-fit hazard ratio (minimum 2NLL).
    """
    if hazard_ratio_values is None:
      # Create a logarithmically-spaced grid
      hazard_ratios = np.logspace(
        np.log10(hazard_ratio_min),
        np.log10(hazard_ratio_max),
        n_points
      )
    else:
      hazard_ratios = np.asarray(hazard_ratio_values)

    twonll_values = np.zeros_like(hazard_ratios)
    results = []

    for i, hr in enumerate(hazard_ratios):
      result = self.compute_2nll_at_hazard_ratio(hr, cox_only=cox_only)
      twonll_values[i] = result.x
      results.append(result)

    # Find the best-fit result (minimum 2NLL)
    best_idx = np.argmin(twonll_values)
    best_fit_result = results[best_idx]

    # Check if best fit from scan is at the scan boundaries
    if best_idx == 0:
      warnings.warn(
        f"Best-fit hazard ratio {hazard_ratios[0]:.6f} is at the lower "
        f"limit of the scan range. Consider extending the scan to lower "
        f"values with hazard_ratio_min < {hazard_ratios[0]:.6f}.",
        RuntimeWarning,
        stacklevel=2
      )
    elif best_idx == len(hazard_ratios) - 1:
      warnings.warn(
        f"Best-fit hazard ratio {hazard_ratios[-1]:.6f} is at the upper "
        f"limit of the scan range. Consider extending the scan to higher "
        f"values with hazard_ratio_max > {hazard_ratios[-1]:.6f}.",
        RuntimeWarning,
        stacklevel=2
      )

    return hazard_ratios, twonll_values, best_fit_result

  def hazard_ratio_confidence_interval(  # pylint: disable=too-many-arguments,too-many-locals
    self,
    *,
    cox_only: bool = False,
    confidence_level: float = 0.68,
    hazard_ratio_min: float = 0.01,
    hazard_ratio_max: float = 100.0,
    tolerance: float = 1e-3,
  ) -> tuple[float, float, float, scipy.optimize.OptimizeResult]:
    """
    Compute the confidence interval for the hazard ratio using profile likelihood.

    This method finds the best-fit hazard ratio and its confidence interval by
    identifying where the 2NLL crosses the threshold for the desired confidence level.

    Parameters
    ----------
    cox_only : bool, optional
        If True, fix patient assignments to nominal values. Default is False.
    confidence_level : float, optional
        Confidence level (e.g., 0.68 for 68% CI, 0.95 for 95% CI). Default is 0.68.
    hazard_ratio_min : float, optional
        Minimum hazard ratio to consider in the search. Default is 0.01.
    hazard_ratio_max : float, optional
        Maximum hazard ratio to consider in the search. Default is 100.0.
    tolerance : float, optional
        Tolerance for the confidence interval boundaries. Default is 1e-3.

    Returns
    -------
    best_fit_hr : float
        The best-fit (maximum likelihood) hazard ratio.
    lower_ci : float
        Lower bound of the confidence interval.
    upper_ci : float
        Upper bound of the confidence interval.
    best_fit_result : scipy.optimize.OptimizeResult
        The optimization result at the best-fit hazard ratio.
    """
    # First, find the best-fit hazard ratio by minimizing the 2NLL
    def objective(log_hr):
      hr = np.exp(log_hr)
      result = self.compute_2nll_at_hazard_ratio(hr, cox_only=cox_only)
      return result.x

    # Start from H=1 (log(H)=0)
    result_opt = scipy.optimize.minimize_scalar(
      objective,
      bounds=(np.log(hazard_ratio_min), np.log(hazard_ratio_max)),
      method='bounded',
      options={'xatol': tolerance * 0.1}
    )

    if not result_opt.success:
      raise ValueError("Could not find best-fit hazard ratio")

    best_fit_log_hr = float(result_opt.x)  # type: ignore[arg-type]
    best_fit_hr = np.exp(best_fit_log_hr)
    min_twonll = float(result_opt.fun)  # type: ignore[arg-type]

    # Get the best-fit result with full details
    best_fit_result = self.compute_2nll_at_hazard_ratio(best_fit_hr, cox_only=cox_only)

    # Compute the threshold for the confidence interval
    # For profile likelihood, the threshold is chi2.ppf(CL, df=1)
    chi2_threshold = scipy.stats.chi2.ppf(confidence_level, df=1)
    twonll_threshold = min_twonll + chi2_threshold

    # Find the lower and upper bounds where 2NLL crosses the threshold
    def twonll_minus_threshold(log_hr):
      hr = np.exp(log_hr)
      result = self.compute_2nll_at_hazard_ratio(hr, cox_only=cox_only)
      return result.x - twonll_threshold

    # Lower bound: search between min and best-fit
    try:
      lower_log_hr = scipy.optimize.brentq(
        twonll_minus_threshold,
        np.log(hazard_ratio_min),
        best_fit_log_hr,
        xtol=tolerance
      )
      lower_ci = np.exp(float(lower_log_hr))  # type: ignore[arg-type]
    except ValueError:
      # If no crossing found, the lower bound is at the search limit
      lower_ci = hazard_ratio_min

    # Upper bound: search between best-fit and max
    try:
      upper_log_hr = scipy.optimize.brentq(
        twonll_minus_threshold,
        best_fit_log_hr,
        np.log(hazard_ratio_max),
        xtol=tolerance
      )
      upper_ci = np.exp(float(upper_log_hr))  # type: ignore[arg-type]
    except ValueError:
      # If no crossing found, the upper bound is at the search limit
      upper_ci = hazard_ratio_max

    # Check if best fit is at or near the bounds
    self._check_hazard_ratio_bounds(best_fit_hr, best_fit_result.log_hazard_ratio)

    return best_fit_hr, lower_ci, upper_ci, best_fit_result

  def _check_hazard_ratio_bounds(self, hazard_ratio: float, log_hazard_ratio: float) -> None:
    """
    Check if the hazard ratio is at or near the model bounds and issue warnings if so.

    Parameters
    ----------
    hazard_ratio : float
        The hazard ratio value to check.
    log_hazard_ratio : float
        The log(hazard ratio) value to check.
    """
    log_hr_lb, log_hr_ub = self.log_hazard_ratio_bounds
    hr_lb = np.exp(log_hr_lb)
    hr_ub = np.exp(log_hr_ub)

    # Define "near the bound" as within 1% of the bound range
    tolerance = 0.01 * (log_hr_ub - log_hr_lb)

    if abs(log_hazard_ratio - log_hr_lb) < tolerance:
      warnings.warn(
        f"Hazard ratio {hazard_ratio:.6f} (log(HR)={log_hazard_ratio:.3f}) is at or near "
        f"the lower bound of the model (HR >= {hr_lb:.6f}, log(HR) >= {log_hr_lb:.1f}). "
        f"The true minimum may be below this bound. Consider increasing the lower bound by "
        f"passing log_hazard_ratio_bounds=({log_hr_lb-5:.1f}, {log_hr_ub:.1f}) to the "
        f"constructor.",
        RuntimeWarning,
        stacklevel=3
      )
    elif abs(log_hazard_ratio - log_hr_ub) < tolerance:
      warnings.warn(
        f"Hazard ratio {hazard_ratio:.6f} (log(HR)={log_hazard_ratio:.3f}) is at or near "
        f"the upper bound of the model (HR <= {hr_ub:.1f}, log(HR) <= {log_hr_ub:.1f}). "
        f"The true maximum may be above this bound. Consider increasing the upper bound by "
        f"passing log_hazard_ratio_bounds=({log_hr_lb:.1f}, {log_hr_ub+5:.1f}) to the "
        f"constructor.",
        RuntimeWarning,
        stacklevel=3
      )
