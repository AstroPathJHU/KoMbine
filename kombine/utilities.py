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

def prob_poisson_density_in_range(  # pylint: disable=too-many-arguments
  observed_count: int,
  observed_area: float,
  range_min: float,
  range_max: float,
  *,
  prior_alpha: float = 0.5,
  prior_beta: float = 0.0,
) -> float:
  """
  Estimate P(range_min <= true_density < range_max | observed_count, observed_area).

  Given a Poisson observation (count from area), estimate the probability that
  the true underlying density lies within a specified range. This is used
  to convert continuous Poisson measurement error into misclassification
  probabilities for Yi's correction method, including "neither group" weights.

  Parameters
  ----------
  observed_count : int
      The observed count (numerator of density).
  observed_area : float
      The observed area (denominator of density). Must be > 0.
  range_min : float
      Lower bound for the density range. May be -np.inf.
  range_max : float
      Upper bound for the density range. May be np.inf.
  prior_alpha : float, optional
      Alpha parameter for Gamma(alpha, beta) prior on Poisson rate parameter.
      Default 0.5 (Jeffreys prior with beta=0).
  prior_beta : float, optional
      Beta parameter for Gamma(alpha, beta) prior on Poisson rate parameter.
      Default 0.0.

  Returns
  -------
  float
      P(range_min <= true_density < range_max | data), in range [0, 1].

  Notes
  -----
  For the Bayesian method:
  - Prior: lambda ~ Gamma(alpha, beta)
  - Likelihood: count ~ Poisson(lambda * area)
  - Posterior: lambda ~ Gamma(alpha + count, beta + area)
  - Return: P(range_min <= lambda < range_max | data)

  The rate parameter lambda is non-negative, so any portion of the range
  below 0 contributes zero probability.

  Examples
  --------
  >>> # High count, clearly within high range
  >>> prob_poisson_density_in_range(100, 1.0, 50.0, np.inf)
  ~0.999

  >>> # Low count, clearly below range
  >>> prob_poisson_density_in_range(10, 1.0, 50.0, np.inf)
  ~0.001

  >>> # Count near threshold (ambiguous classification)
  >>> prob_poisson_density_in_range(50, 1.0, 48.0, 52.0)
  ~0.5
  """
  if observed_area <= 0:
    raise ValueError(f"observed_area must be > 0, got {observed_area}")
  if observed_count < 0:
    raise ValueError(f"observed_count must be >= 0, got {observed_count}")
  if range_min >= range_max:
    raise ValueError(
      f"range_min must be < range_max, got {range_min} >= {range_max}"
    )

  # The Poisson rate is non-negative; clamp any negative range portion.
  if range_max <= 0:
    return 0.0
  if range_min < 0:
    range_min = 0.0

  if np.isneginf(range_min) and np.isposinf(range_max):
    return 1.0

  # Bayesian posterior for Poisson density parameter
  # Let 'rate' be the true density (cells per unit area)
  # Observation: count ~ Poisson(rate * area)
  #
  # Prior: rate ~ Gamma(alpha, beta) [rate parameterization]
  # With Jeffreys prior: alpha=0.5, beta=0
  #
  # Posterior: rate ~ Gamma(alpha + count, beta + area) [rate parameterization]
  # In scipy scale parameterization: scale = 1/(beta + area)
  #
  # We want: P(range_min <= rate < range_max)

  # Compute posterior parameters
  posterior_alpha = prior_alpha + observed_count
  # Correct Gamma-Poisson conjugate update: scale = 1/(beta + area)
  posterior_scale = 1.0 / (prior_beta + observed_area)

  # scipy.stats.gamma uses (a=shape, scale=scale)
  cdf_min = scipy.stats.gamma.cdf(
    range_min,
    a=posterior_alpha,
    scale=posterior_scale,
  ) if not np.isneginf(range_min) else 0.0
  cdf_max = scipy.stats.gamma.cdf(
    range_max,
    a=posterior_alpha,
    scale=posterior_scale,
  ) if not np.isposinf(range_max) else 1.0

  prob_in_range = max(0.0, min(1.0, float(cdf_max - cdf_min)))
  return float(prob_in_range)


def validate_class_probs(class_probs: list) -> None:
  """Raise ValueError if class_probs is not a valid probability distribution."""
  if not class_probs:
    raise ValueError("class_probs must be non-empty")
  for prob in class_probs:
    if not isinstance(prob, (int, float)) or prob < 0:
      raise ValueError(f"Invalid class probability: {prob}")
  total = float(sum(class_probs))
  if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
    raise ValueError(f"Class probabilities must sum to 1, got {total}")


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
    if the initial optimization is suboptimal or hits the time limit.

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

    needs_fallback = model.status in (GRB.SUBOPTIMAL, GRB.TIME_LIMIT)
    if needs_fallback:
      for i, (fallback_params, description) in enumerate(fallback_strategies):
        if verbose:
          print(
            f"Model returned status {model.status}. "
            f"Applying fallback {i+1}: {description}"
          )
          print(f"  New parameters: {fallback_params}")
        self._set_gurobi_params(model, fallback_params)
        model.optimize()
        if model.status == GRB.OPTIMAL:
          if verbose:
            print(f"Fallback {i+1} successful. Model is now optimal.")
          break
    return model
