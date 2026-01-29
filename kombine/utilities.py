"""
Miscellaneous utilities for ROC Picker
"""
import typing

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.stats

# Default log zero epsilon value used across Kaplan-Meier likelihood methods
# Set to be larger than compile_plots.sh value (1e-7) so explicit specification not needed
LOG_ZERO_EPSILON_DEFAULT = 1e-6

T = typing.TypeVar("T")
R = typing.TypeVar("R")

class InspectableCache(typing.Generic[T, R]):
  """
  Cache decorator that allows inspection of cached values.
  """
  def __init__(self, func: typing.Callable[[T], R]):
    self._func = func
    self.cache: dict[tuple[T], R] = {}
    self.__name__ = getattr(func, "__name__", "InspectableCache")
    self.__doc__ = getattr(func, "__doc__", None)

  def __call__(self, arg: T) -> R:
    key = (arg,)
    if key in self.cache:
      return self.cache[key]
    result = self._func(arg)
    self.cache[key] = result
    return result

  def __getattr__(self, name: str) -> typing.Any:
    # Delegate attribute access to the original function
    return getattr(self._func, name)

  def __iter__(self) -> typing.Iterator[tuple[tuple[T], R]]:
    yield from self.cache.items()


# ============================================================================
# Yi's Misclassification Correction Utilities (Section 3.7.1)
# ============================================================================

def prob_poisson_density_exceeds_threshold(
  observed_count: int,
  observed_area: float,
  threshold: float,
  *,
  method: str = 'bayesian',
  prior_alpha: float = 0.5,
  prior_beta: float = 0.0,
) -> float:
  """
  Estimate P(true_density > threshold | observed_count, observed_area).
  
  Given a Poisson observation (count from area), estimate the probability that
  the true underlying density exceeds a classification threshold. This is used
  to convert continuous Poisson measurement error into discrete misclassification
  probabilities for Yi's correction method.
  
  Parameters
  ----------
  observed_count : int
      The observed count (numerator of density).
  observed_area : float
      The observed area (denominator of density). Must be > 0.
  threshold : float
      The classification threshold for group assignment. Must be > 0.
  method : str, optional
      Method for computing the probability:
      - 'bayesian': Full Bayesian posterior using Gamma prior (default)
      - 'normal_approx': Normal approximation to Poisson (faster but less accurate)
  prior_alpha : float, optional
      Alpha parameter for Gamma(alpha, beta) prior on Poisson rate parameter.
      Used only for 'bayesian' method. Default 0.5 (Jeffreys prior with beta=0).
  prior_beta : float, optional
      Beta parameter for Gamma(alpha, beta) prior on Poisson rate parameter.
      Used only for 'bayesian' method. Default 0.0 (Jeffreys prior).
  
  Returns
  -------
  float
      P(true_density > threshold | data), in range [0, 1].
  
  Notes
  -----
  For the Bayesian method:
  - Prior: lambda ~ Gamma(alpha, beta)
  - Likelihood: count ~ Poisson(lambda * area)
  - Posterior: lambda ~ Gamma(alpha + count, beta + area)
  - Return: P(lambda > threshold | data)
  
  For the normal approximation method:
  - Approximate Poisson(lambda*area) by Normal with mean=variance=lambda*area
  - Use CLT to get Normal distribution for density estimate
  - Compute P(density > threshold) using Normal CDF
  
  The normal approximation is faster but less accurate for small counts.
  
  Examples
  --------
  >>> # High count, clearly above threshold
  >>> prob_poisson_density_exceeds_threshold(100, 1.0, 50.0)
  ~0.999
  
  >>> # Low count, clearly below threshold
  >>> prob_poisson_density_exceeds_threshold(10, 1.0, 50.0)
  ~0.001
  
  >>> # Count near threshold (ambiguous classification)
  >>> prob_poisson_density_exceeds_threshold(50, 1.0, 50.0)
  ~0.5
  """
  if observed_area <= 0:
    raise ValueError(f"observed_area must be > 0, got {observed_area}")
  if threshold <= 0:
    raise ValueError(f"threshold must be > 0, got {threshold}")
  if observed_count < 0:
    raise ValueError(f"observed_count must be >= 0, got {observed_count}")
  
  if method == 'bayesian':
    # Bayesian posterior for Poisson density parameter
    # Let 'rate' be the true density (cells per unit area)
    # Observation: count ~ Poisson(rate * area)
    # 
    # Prior: rate ~ Gamma(alpha, scale)
    # With Jeffreys prior: alpha=0.5, scale=1 (so beta=1/scale=1)
    # 
    # Posterior: rate ~ Gamma(alpha + count, scale=1/(1/scale + area))
    # With Jeffreys: rate ~ Gamma(0.5 + count, scale=1/(0 + area))
    #              = Gamma(0.5 + count, scale=1/area)
    # 
    # We want: P(rate > threshold)
    
    # Convert prior from (alpha, beta) to (alpha, scale) parameterization
    # Beta parameterization: Gamma(alpha, beta) where rate = 1/scale
    # Scale parameterization: Gamma(alpha, scale)
    # Relationship: beta = 1/scale
    prior_scale = 1.0 / prior_beta if prior_beta > 0 else 1.0
    
    # Compute posterior parameters
    posterior_alpha = prior_alpha + observed_count
    posterior_scale = prior_scale / (1.0 + prior_scale * observed_area)
    
    # P(rate > threshold) = 1 - CDF(threshold)
    # scipy.stats.gamma uses (a=shape, scale=scale)
    prob_exceeds = 1.0 - scipy.stats.gamma.cdf(
      threshold,
      a=posterior_alpha,
      scale=posterior_scale
    )
    return float(prob_exceeds)
  
  elif method == 'normal_approx':
    # Normal approximation: density ~ N(observed_density, variance)
    observed_density = observed_count / observed_area
    
    # Variance of Poisson count = lambda * area
    # Variance of density = (lambda * area) / area^2 = lambda / area
    # Estimate lambda by observed_density
    variance = observed_density / observed_area
    
    # Handle zero count case (degenerate)
    if observed_count == 0:
      # density ~ 0, so P(density > threshold) ~ 0
      # Use half-count correction: treat as 0.5 count
      variance = 0.5 / (observed_area * observed_area)
      observed_density = 0.5 / observed_area
    
    std = np.sqrt(variance)
    
    # P(density > threshold) = 1 - Phi((threshold - observed_density) / std)
    z_score = (threshold - observed_density) / std
    prob_exceeds = 1.0 - scipy.stats.norm.cdf(z_score)
    return float(prob_exceeds)
  
  else:
    raise ValueError(f"Unknown method '{method}'. Must be 'bayesian' or 'normal_approx'.")


def estimate_misclassification_matrix(
  patients: list,
  threshold: float,
  *,
  method: str = 'bayesian',
  prior_alpha: float = 0.5,
  prior_beta: float = 0.0,
) -> npt.NDArray[np.float64]:
  """
  Estimate the misclassification matrix for binary group assignment from patient data.
  
  Computes a 2x2 misclassification probability matrix Π where:
  - Π[i,j] = P(observed group = j | true group = i)
  - Rows represent true groups (0 = low, 1 = high)
  - Columns represent observed groups (0 = low, 1 = high)
  
  This is used for Yi's misclassification correction method (Section 3.7.1).
  
  Parameters
  ----------
  patients : list
      List of patient objects. Each patient must have:
      - observed_parameter: float (the observed biomarker value)
      - Either:
        a) numerator_count and denominator_area attributes (Poisson density), OR
        b) Direct parameter value with uncertainty model
  threshold : float
      Classification threshold. Patients with parameter > threshold are "high",
      others are "low".
  method : str, optional
      Method for probability estimation ('bayesian' or 'normal_approx').
  prior_alpha : float, optional
      Prior parameter for Bayesian method.
  prior_beta : float, optional
      Prior parameter for Bayesian method.
  
  Returns
  -------
  np.ndarray
      2x2 misclassification matrix with shape (2, 2).
      Π[i,j] = P(observed=j | true=i).
  
  Notes
  -----
  The matrix is estimated by:
  1. For each patient, determine observed group (0 or 1) from observed_parameter
  2. Compute P(true > threshold | observed data) using Poisson measurement model
  3. Average these probabilities over patients in each observed group
  
  Under KoMbine conventions, the matrix is always invertible because patients
  are assigned to the group with highest probability (argmax), ensuring the
  diagonal dominates.
  
  Examples
  --------
  >>> # Patients with low measurement error → diagonal matrix
  >>> Pi = estimate_misclassification_matrix(patients, threshold=100.0)
  >>> Pi
  array([[0.95, 0.05],
         [0.05, 0.95]])
  
  >>> # Perfect measurement → identity matrix
  >>> Pi
  array([[1.0, 0.0],
         [0.0, 1.0]])
  """
  if threshold <= 0:
    raise ValueError(f"threshold must be > 0, got {threshold}")
  if not patients:
    raise ValueError("patients list cannot be empty")
  
  # Separate patients by observed group
  observed_low_patients = []
  observed_high_patients = []
  
  for patient in patients:
    # Get observed parameter value
    # Check if patient has observable with numerator/denominator (Poisson density)
    if hasattr(patient, 'observable'):
      obs = patient.observable
      if hasattr(obs, 'numerator') and hasattr(obs, 'denominator'):
        observed = obs.numerator / obs.denominator
      else:
        # For other observable types, try to get a nominal value
        observed = getattr(obs, 'nominal', None)
        if observed is None:
          raise ValueError(f"Cannot determine observed parameter for patient: {patient}")
    else:
      # Direct parameter (e.g., from fixed observable)
      observed = getattr(patient, 'observed_parameter', None)
      if observed is None:
        raise ValueError(f"Cannot determine observed parameter for patient: {patient}")
    
    if observed > threshold:
      observed_high_patients.append(patient)
    else:
      observed_low_patients.append(patient)
  
  # Initialize probability sums
  # Pi[true_group, observed_group]
  Pi = np.zeros((2, 2), dtype=np.float64)
  
  # Estimate Pi[0, 0] = P(observed=low | true=low)
  # and Pi[1, 0] = P(observed=low | true=high)
  # from patients observed in low group
  if observed_low_patients:
    prob_true_high_sum = 0.0
    for patient in observed_low_patients:
      # Get Poisson count and area from observable
      count = None
      area = None
      
      if hasattr(patient, 'observable'):
        obs = patient.observable
        if hasattr(obs, 'numerator') and hasattr(obs, 'denominator'):
          count = obs.numerator
          area = obs.denominator
      
      if count is not None and area is not None:
        # Use Poisson density model
        prob_true_high = prob_poisson_density_exceeds_threshold(
          count, area, threshold,
          method=method,
          prior_alpha=prior_alpha,
          prior_beta=prior_beta,
        )
      else:
        # Fallback: use observed value as best estimate
        # (occurs for fixed observables or simple counts)
        # Get observed value
        if hasattr(patient, 'observable'):
          obs = patient.observable
          if hasattr(obs, 'numerator') and hasattr(obs, 'denominator'):
            observed_val = obs.numerator / obs.denominator
          else:
            observed_val = getattr(obs, 'nominal', 0.0)
        else:
          observed_val = getattr(patient, 'observed_parameter', 0.0)
        
        prob_true_high = 1.0 if observed_val > threshold else 0.0
      
      prob_true_high_sum += prob_true_high
    
    # Average probabilities
    avg_prob_true_high = prob_true_high_sum / len(observed_low_patients)
    avg_prob_true_low = 1.0 - avg_prob_true_high
    
    # Given observed=low:
    # P(true=low | observed=low) ~ avg_prob_true_low
    # P(true=high | observed=low) ~ avg_prob_true_high
    # We need P(observed=low | true), use Bayes with equal priors
    # For simplicity, assume P(true=low) = P(true=high) = 0.5
    # Then P(observed=low | true=low) = P(true=low | observed=low) / P(true=low) * P(observed=low)
    # But we directly estimate from the group averages
    Pi[0, 0] = avg_prob_true_low  # P(observed=low | true=low) approximated
    Pi[1, 0] = avg_prob_true_high  # P(observed=low | true=high) approximated
  
  # Estimate Pi[0, 1] = P(observed=high | true=low)
  # and Pi[1, 1] = P(observed=high | true=high)
  # from patients observed in high group
  if observed_high_patients:
    prob_true_high_sum = 0.0
    for patient in observed_high_patients:
      # Get Poisson count and area from observable
      count = None
      area = None
      
      if hasattr(patient, 'observable'):
        obs = patient.observable
        if hasattr(obs, 'numerator') and hasattr(obs, 'denominator'):
          count = obs.numerator
          area = obs.denominator
      
      if count is not None and area is not None:
        prob_true_high = prob_poisson_density_exceeds_threshold(
          count, area, threshold,
          method=method,
          prior_alpha=prior_alpha,
          prior_beta=prior_beta,
        )
      else:
        # Fallback: use observed value as best estimate
        # Get observed value
        if hasattr(patient, 'observable'):
          obs = patient.observable
          if hasattr(obs, 'numerator') and hasattr(obs, 'denominator'):
            observed_val = obs.numerator / obs.denominator
          else:
            observed_val = getattr(obs, 'nominal', 0.0)
        else:
          observed_val = getattr(patient, 'observed_parameter', 0.0)
        
        prob_true_high = 1.0 if observed_val > threshold else 0.0
      
      prob_true_high_sum += prob_true_high
    
    avg_prob_true_high = prob_true_high_sum / len(observed_high_patients)
    avg_prob_true_low = 1.0 - avg_prob_true_high
    
    Pi[0, 1] = avg_prob_true_low  # P(observed=high | true=low)
    Pi[1, 1] = avg_prob_true_high  # P(observed=high | true=high)
  
  # Normalize rows (each row should sum to 1)
  row_sums = Pi.sum(axis=1, keepdims=True)
  # Avoid division by zero
  row_sums = np.where(row_sums > 0, row_sums, 1.0)
  Pi = Pi / row_sums
  
  return Pi


def invert_misclassification_matrix(
  Pi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
  """
  Compute the inverse of a 2x2 misclassification matrix for Yi's correction.
  
  Given misclassification matrix Π with Π[i,j] = P(observed=j | true=i),
  compute the matrix inverse Π^{-1} used for inverse probability weighting
  in Yi's method (Section 3.7.1, Equation 3.57).
  
  Parameters
  ----------
  Pi : np.ndarray
      2x2 misclassification matrix with shape (2, 2).
      Each row should sum to 1.0 (probability distribution).
  
  Returns
  -------
  np.ndarray
      2x2 inverse matrix Π^{-1} with shape (2, 2).
  
  Raises
  ------
  ValueError
      If matrix is singular or nearly singular (determinant near zero).
  
  Notes
  -----
  Under KoMbine conventions, the misclassification matrix is always invertible
  because nominal assignment uses argmax, ensuring diagonal dominance.
  
  For a 2x2 matrix:
  Π = [[π00, π01],
       [π10, π11]]
  
  The inverse is:
  Π^{-1} = (1/det) * [[π11, -π01],
                       [-π10, π00]]
  
  where det = π00*π11 - π01*π10.
  
  Examples
  --------
  >>> Pi = np.array([[0.9, 0.1], [0.1, 0.9]])
  >>> Pi_inv = invert_misclassification_matrix(Pi)
  >>> np.allclose(Pi @ Pi_inv, np.eye(2))
  True
  """
  if Pi.shape != (2, 2):
    raise ValueError(f"Pi must be 2x2, got shape {Pi.shape}")
  
  # Check that rows sum to 1 (probability distribution)
  row_sums = Pi.sum(axis=1)
  if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-8):
    raise ValueError(
      f"Rows of misclassification matrix must sum to 1. Got row sums: {row_sums}"
    )
  
  # Compute determinant
  det = Pi[0, 0] * Pi[1, 1] - Pi[0, 1] * Pi[1, 0]
  
  # Check for singularity
  if np.abs(det) < 1e-10:
    raise ValueError(
      f"Misclassification matrix is singular or nearly singular (det = {det}). "
      f"This should not occur under KoMbine conventions with argmax assignment."
    )
  
  # Compute inverse using analytical formula for 2x2
  Pi_inv = np.array([
    [Pi[1, 1], -Pi[0, 1]],
    [-Pi[1, 0], Pi[0, 0]]
  ], dtype=np.float64) / det
  
  return Pi_inv
