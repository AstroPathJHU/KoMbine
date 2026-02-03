"""
Miscellaneous utilities for ROC Picker
"""
import typing

import numpy as np
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

def prob_poisson_density_exceeds_threshold(  # pylint: disable=too-many-arguments
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

  if method == 'normal_approx':
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

  raise ValueError(f"Unknown method '{method}'. Must be 'bayesian' or 'normal_approx'.")
