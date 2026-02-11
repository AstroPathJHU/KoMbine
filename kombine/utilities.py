"""
Miscellaneous utilities for ROC Picker
"""
import os
import typing

import numpy as np
import scipy.stats

# Default log zero epsilon value used across Kaplan-Meier likelihood methods
# Set to be larger than compile_plots.sh value (1e-7)
# so explicit specification not needed
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
  prior_alpha : float, optional
      Alpha parameter for Gamma(alpha, beta) prior on Poisson rate parameter.
      Default 0.5 (Jeffreys prior with beta=0).
  prior_beta : float, optional
      Beta parameter for Gamma(alpha, beta) prior on Poisson rate parameter.
      Default 0.0.

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

  # Bayesian posterior for Poisson density parameter
  # Let 'rate' be the true density (cells per unit area)
  # Observation: count ~ Poisson(rate * area)
  #
  # Prior: rate ~ Gamma(alpha, beta) [rate parameterization]
  # With Jeffreys prior: alpha=0.5, beta=0
  #
  # Posterior: rate ~ Gamma(alpha + count, beta + area) [rate parameterization]
  # With Jeffreys: rate ~ Gamma(0.5 + count, 0 + area)
  #              = Gamma(0.5 + count, area)
  #
  # In scipy scale parameterization: scale = 1/rate_param
  # So posterior scale = 1/(beta + area)
  #
  # We want: P(rate > threshold)

  # Compute posterior parameters
  posterior_alpha = prior_alpha + observed_count
  # Correct Gamma-Poisson conjugate update: scale = 1/(beta + area)
  posterior_scale = 1.0 / (prior_beta + observed_area)

  # P(rate > threshold) = 1 - CDF(threshold)
  # scipy.stats.gamma uses (a=shape, scale=scale)
  prob_exceeds = 1.0 - scipy.stats.gamma.cdf(
    threshold,
    a=posterior_alpha,
    scale=posterior_scale
  )
  return float(prob_exceeds)


# ============================================================================
# Gurobi Optimization Helper Mixin
# ============================================================================

class GurobiOptimizerMixin:  # pylint: disable=too-few-public-methods
  """
  Mixin class providing common Gurobi optimization utilities.
  """

  def _set_gurobi_params(self, model, params: dict):
    """
    Helper function to set multiple Gurobi parameters from a dictionary.
    """
    for param, value in params.items():
      if value is not None:
        model.setParam(param, value)

  def _create_gurobi_params(  # pylint: disable=too-many-arguments
    self,
    *,
    verbose: bool = False,
    MIPGap: float | None = None,
    MIPGapAbs: float | None = None,
    TimeLimit: float | None = None,
    Threads: int | None = None,
    MIPFocus: int | None = None,
    LogFile=None,
  ) -> dict:
    """
    Create a standard Gurobi parameter dictionary.

    Args:
        verbose: If True, enable Gurobi output.
        MIPGap: Relative MIP optimality gap.
        MIPGapAbs: Absolute MIP optimality gap.
        TimeLimit: Time limit in seconds (optional).
        Threads: Number of threads to use (optional).
        MIPFocus: MIP focus setting (0=balanced, 1=feasibility, 2=optimality, 3=bound).
        LogFile: Path to log file (optional).

    Returns:
        Dictionary of Gurobi parameters.
    """
    params = {
      'OutputFlag': 1 if verbose else 0,
      'MIPGap': MIPGap,
      'MIPGapAbs': MIPGapAbs,
      'NonConvex': 2,
      'TimeLimit': TimeLimit,
      'Threads': Threads,
      'MIPFocus': MIPFocus,
    }
    if LogFile is not None:
      params['LogFile'] = os.fspath(LogFile)
    return params

  def _create_fallback_strategies(self) -> list[tuple[dict, str]]:
    """
    Create standard fallback strategies for Gurobi optimization.

    Returns:
        List of (parameters_dict, description) tuples.
    """
    return [
      ({'MIPFocus': 2}, "MIPFocus set to 2 (optimality focus)"),
      ({'NumericFocus': 3}, "NumericFocus set to 3 (highest precision)"),
    ]

  def _setup_and_optimize(  # pylint: disable=too-many-arguments
    self,
    model,
    *,
    verbose: bool = False,
    MIPGap: float | None = None,
    MIPGapAbs: float | None = None,
    TimeLimit: float | None = None,
    Threads: int | None = None,
    MIPFocus: int | None = None,
    LogFile=None,
  ):
    """
    Setup Gurobi parameters and optimize with fallback strategies.

    This combines parameter creation and optimization in one method to reduce
    code duplication in calling code.

    Args:
        model: The Gurobi model to optimize.
        verbose: If True, enable Gurobi output.
        MIPGap: Relative MIP optimality gap.
        MIPGapAbs: Absolute MIP optimality gap.
        TimeLimit: Time limit in seconds (optional).
        Threads: Number of threads to use (optional).
        MIPFocus: MIP focus setting (0=balanced, 1=feasibility, 2=optimality, 3=bound).
        LogFile: Path to log file (optional).

    Returns:
        The optimized Gurobi model.
    """
    initial_gurobi_params = self._create_gurobi_params(
      verbose=verbose,
      MIPGap=MIPGap,
      MIPGapAbs=MIPGapAbs,
      TimeLimit=TimeLimit,
      Threads=Threads,
      MIPFocus=MIPFocus,
      LogFile=LogFile,
    )
    fallback_strategies = self._create_fallback_strategies()

    return self._optimize_with_fallbacks(
      model, initial_gurobi_params, fallback_strategies, verbose
    )

  def _optimize_with_fallbacks(
    self,
    model,
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
    # Import GRB here to avoid circular imports
    from gurobipy import GRB  # pylint: disable=import-outside-toplevel

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
