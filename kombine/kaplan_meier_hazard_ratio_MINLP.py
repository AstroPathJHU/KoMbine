"""
MINLP solver for calculating hazard ratios and likelihood scans for two Kaplan-Meier curves.
This module provides functionality to compute the negative log likelihood as a function
of the hazard ratio H, enabling profile likelihood analyses and confidence intervals.
"""
# pylint: disable=too-many-lines

import warnings
import os
import datetime
import typing
from typing import Optional

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.stats

from .kaplan_meier_p_value_MINLP import MINLPforKMPValue
from .toy_calibration import (
  DEFAULT_CLS,
  SequentialToyCounter,
  ToyTestResult,
  as_generator,
  bisection_endpoint,
  hard_group_high,
  weighted_cox_permutation,
)
from .utilities import LOG_ZERO_EPSILON_DEFAULT, brentq_hazard_ratio_ci


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
    # MIP starts from the previous nearby-HR solve on this calculator.
    self.__mip_start_assignments: dict[tuple[int, int], float] | None = None
    self.__mip_start_cox_only: bool | None = None
    self.__mip_start_log_hr: float | None = None
    # Only reuse starts when |Δ log H| is below this (avoids trapping on distant jumps).
    self.__mip_start_max_log_hr_delta = 1.0
    self.__feasibility_cut_constraints: list = []
    self.last_oracle_work: float | None = None

  def _clear_hazard_ratio_mip_starts(self, a, beta_var=None) -> None:
    """Drop cached starts and reset Gurobi Start attributes."""
    self.__mip_start_assignments = None
    self.__mip_start_cox_only = None
    self.__mip_start_log_hr = None
    for var in a.values():
      var.Start = GRB.UNDEFINED
    if beta_var is not None:
      beta_var.Start = GRB.UNDEFINED

  def _apply_hazard_ratio_mip_starts(
    self,
    a,
    beta_var,
    log_hazard_ratio: float,
    cox_only: bool,
  ) -> None:
    """
    Seed the next solve from the previous incumbent when cox_only matches
    and the hazard ratio is close in log space.
    """
    if self.__mip_start_cox_only is not None and self.__mip_start_cox_only != cox_only:
      self._clear_hazard_ratio_mip_starts(a, beta_var)
      return

    if (
      self.__mip_start_assignments is None
      or self.__mip_start_log_hr is None
      or abs(log_hazard_ratio - self.__mip_start_log_hr)
        > self.__mip_start_max_log_hr_delta
    ):
      for var in a.values():
        var.Start = GRB.UNDEFINED
      beta_var.Start = GRB.UNDEFINED
      return

    for key, value in self.__mip_start_assignments.items():
      a[key].Start = value
    beta_var.Start = log_hazard_ratio

  def _store_hazard_ratio_mip_starts(
    self,
    a,
    cox_only: bool,
    log_hazard_ratio: float,
  ) -> None:
    """Cache assignment incumbents for the next nearby-HR solve."""
    self.__mip_start_assignments = {
      key: float(var.X) for key, var in a.items()
    }
    self.__mip_start_cox_only = cox_only
    self.__mip_start_log_hr = log_hazard_ratio

  def compute_2nll_at_hazard_ratio(  # pylint: disable=too-many-locals, too-many-arguments
    self,
    hazard_ratio: float,
    *,
    cox_only: bool = False,
    verbose: bool = False,
    print_progress: bool = False,
    MIPGap: float | None = None,
    MIPGapAbs: float | None = None,
    TimeLimit: float | None = None,
    Threads: int | None = None,
    MIPFocus: int | None = None,
    LogFile: Optional[os.PathLike] = None,
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
    if print_progress or verbose:
      print(
        f"[{datetime.datetime.now()}] Computing 2NLL at hazard ratio "
        f"{hazard_ratio}",
        flush=True,
      )

    (  # pylint: disable=duplicate-code
      model,
      null_hypothesis_indicator,
      a,
      beta,
      use_cox_penalty_indicator,
    ) = self.gurobi_model
    self._clear_feasibility_cuts(model)

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

    self._apply_hazard_ratio_mip_starts(a, beta_var, log_hazard_ratio, cox_only)

    # Setup and optimize with standard parameters and fallback strategies
    # pylint: disable=duplicate-code
    model = self._setup_and_optimize(
      model,
      verbose=verbose,
      MIPGap=MIPGap,
      MIPGapAbs=MIPGapAbs,
      TimeLimit=TimeLimit,
      Threads=Threads,
      MIPFocus=MIPFocus,
      LogFile=LogFile,
    )

    if model.status != GRB.OPTIMAL:
      raise ValueError(f"Optimization failed with status {model.status}")
    # pylint: enable=duplicate-code

    self._store_hazard_ratio_mip_starts(a, cox_only, log_hazard_ratio)

    result = self._extract_optimize_result(model, a)
    result.log_hazard_ratio = log_hazard_ratio

    # Check if we're at or near the bounds and warn
    self._check_hazard_ratio_bounds(hazard_ratio, log_hazard_ratio)

    return result

  def likelihood_scan_hazard_ratio(  # pylint: disable=too-many-arguments,too-many-locals
    self,
    hazard_ratio_values: Optional[npt.NDArray[np.float64]] = None,
    *,
    cox_only: bool = False,
    n_points: int = 50,
    hazard_ratio_min: float = 0.1,
    hazard_ratio_max: float = 10.0,
  ) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    scipy.optimize.OptimizeResult,
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
  ]:
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
    assignments_low : ndarray
      Boolean array of shape (n_points, n_patients) indicating low-group assignments.
    assignments_high : ndarray
      Boolean array of shape (n_points, n_patients) indicating high-group assignments.
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

    assignments_low = np.zeros((len(hazard_ratios), self.n_patients), dtype=bool)
    assignments_high = np.zeros((len(hazard_ratios), self.n_patients), dtype=bool)
    for idx, result in enumerate(results):
      assignments_low[idx, result.patients_low] = True
      assignments_high[idx, result.patients_high] = True
    return hazard_ratios, twonll_values, best_fit_result, assignments_low, assignments_high

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

    lower_ci, upper_ci = brentq_hazard_ratio_ci(
      twonll_minus_threshold,
      best_fit_log_hr=best_fit_log_hr,
      hazard_ratio_min=hazard_ratio_min,
      hazard_ratio_max=hazard_ratio_max,
      tolerance=tolerance,
    )

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

  def _clear_feasibility_cuts(self, model) -> None:
    """Remove temporary CL-feasibility cuts from a prior excess_at_most call."""
    for constr in self.__feasibility_cut_constraints:
      try:
        model.remove(constr)
      except gp.GurobiError:
        pass
    self.__feasibility_cut_constraints = []
    model.update()

  def _clear_fixed_hazard_ratio_constraint(self, model) -> None:
    """Drop a leftover locked-H constraint so H can float again."""
    existing = model.getConstrByName("fixed_hazard_ratio_constraint")
    if existing is not None:
      model.remove(existing)
      model.update()

  def _lock_hazard_ratio_for_solve(
    self,
    hazard_ratio: float,
    *,
    cox_only: bool,
  ):
    """Lock H, disable the H=1 indicator, and apply cox_only assignment bounds."""
    (
      model,
      null_hypothesis_indicator,
      a,
      beta,
      use_cox_penalty_indicator,
    ) = self.gurobi_model
    self._clear_feasibility_cuts(model)
    log_hazard_ratio = np.log(hazard_ratio)
    beta_var = model.getVarByName("log_hazard_ratio")
    if beta_var is None:
      raise ValueError("Could not find log_hazard_ratio variable in model")
    hazard_ratio_constraint_name = "fixed_hazard_ratio_constraint"
    existing_constr = model.getConstrByName(hazard_ratio_constraint_name)
    if existing_constr is not None:
      model.remove(existing_constr)
    model.addConstr(
      beta_var == log_hazard_ratio,
      name=hazard_ratio_constraint_name,
    )
    self.update_model_for_null_hypothesis_or_not(
      model, null_hypothesis_indicator, False,
    )
    self.update_model_with_cox_only_constraints(model, a, cox_only)
    self.update_model_with_patient_wise_only_constraint(
      model,
      beta=beta,
      null_hypothesis_indicator=null_hypothesis_indicator,
      patient_wise_only=False,
      use_cox_penalty_indicator=use_cox_penalty_indicator,
    )
    return model, a, beta_var, log_hazard_ratio

  def _install_toy_outcomes(
    self,
    times: list[float],
    censored: list[bool],
    *,
    first: bool,
  ) -> None:
    """Swap (time, censored) on this calculator, refreshing incidence if needed."""
    if first:
      self.set_patient_outcomes(times, censored, keep_model=False)
      return
    self.set_patient_outcomes(times, censored, keep_model=True)
    model, _null, a, _beta, _use_cox = self.gurobi_model
    r_vars, d_vars = self._get_risk_set_vars()
    assert r_vars is not None and d_vars is not None
    self._refresh_death_incidence_constraints(model, a, r_vars, d_vars)

  def excess_at_most(  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    self,
    hazard_ratio: float,
    *,
    twoNLL_min: float,
    level: float,
    cox_only: bool = False,
    verbose: bool = False,
    print_progress: bool = False,
    TimeLimit: float | None = None,
    Threads: int | None = None,
    LogFile: Optional[os.PathLike] = None,
  ) -> typing.Literal["inside", "outside", "unknown"]:
    """
    Ask whether profile excess 2NLL(H) - twoNLL_min can be <= ``level``.

    Adds an objective cut ``2NLL <= twoNLL_min + level`` at locked H and
    runs a feasibility-focused solve (first incumbent is enough).
    """
    if hazard_ratio <= 0:
      raise ValueError(f"hazard_ratio must be > 0, got {hazard_ratio}")
    if level < 0:
      raise ValueError(f"level must be >= 0, got {level}")
    if not np.isfinite(twoNLL_min):
      raise ValueError(f"twoNLL_min must be finite, got {twoNLL_min}")
    threshold = float(twoNLL_min + level)
    if print_progress or verbose:
      print(
        f"[{datetime.datetime.now()}] Feasibility excess_at_most "
        f"H={hazard_ratio} level={level} T={threshold}",
        flush=True,
      )
    model, a, beta_var, log_hazard_ratio = self._lock_hazard_ratio_for_solve(
      hazard_ratio, cox_only=cox_only,
    )
    self._apply_hazard_ratio_mip_starts(a, beta_var, log_hazard_ratio, cox_only)
    self.__feasibility_cut_constraints.append(
      model.addConstr(model.getObjective() <= threshold, name="feas_total_2nll_cut")
    )
    model.setParam("OutputFlag", 1 if verbose else 0)
    model.setParam("DisplayInterval", 1)
    model.setParam("NonConvex", 2)
    model.setParam("Seed", 123456)
    model.setParam("MIPFocus", 1)
    model.setParam("FuncPieces", 1000)
    model.setParam("FuncPieceRatio", 0.5)
    model.setParam("SolutionLimit", 1)
    if TimeLimit is not None:
      model.setParam("TimeLimit", TimeLimit)
    if Threads is not None:
      model.setParam("Threads", Threads)
    if LogFile is not None:
      model.setParam("LogFile", os.fspath(LogFile))
    try:
      model.optimize()
      status = model.status
      sol_count = int(model.SolCount)
      if sol_count > 0:
        self._store_hazard_ratio_mip_starts(a, cox_only, log_hazard_ratio)
        result: typing.Literal["inside", "outside", "unknown"] = "inside"
      elif status == GRB.INFEASIBLE:
        result = "outside"
      else:
        result = "unknown"
      try:
        self.last_oracle_work = float(model.Work)
      except (AttributeError, gp.GurobiError):
        self.last_oracle_work = None
      if print_progress or verbose:
        print(
          f"[{datetime.datetime.now()}] excess_at_most -> {result} "
          f"(status={status}, SolCount={sol_count})",
          flush=True,
        )
    finally:
      model.setParam("SolutionLimit", 2000000000)
      model.setParam("MIPFocus", 0)
      self._clear_feasibility_cuts(model)
    return result

  def hypothesized_hr_toy_test(  # pylint: disable=too-many-arguments, too-many-locals
    self,
    hazard_ratio: float,
    *,
    n_max: int = 19,
    rng: int | np.random.Generator | None = None,
    cox_only: bool = False,
    confidence_levels: tuple[float, ...] = DEFAULT_CLS,
    Threads: int | None = 1,
    verbose: bool = False,
    print_progress: bool = False,
  ) -> ToyTestResult:
    """
    Neyman test of H = ``hazard_ratio`` using weighted Cox outcome toys.

    Each toy is classified as more extreme than the data if
    ``excess_at_most`` returns ``outside`` (Δ2NLL* > Δ2NLL_obs).
    """
    generator = as_generator(rng)
    result_locked = self.compute_2nll_at_hazard_ratio(
      hazard_ratio, cox_only=cox_only, Threads=Threads,
      verbose=verbose, print_progress=print_progress,
    )
    self._clear_fixed_hazard_ratio_constraint(self.gurobi_model[0])
    _, result_alt = self._fit_null_and_alternative(
      cox_only=cox_only,
      patient_wise_only=False,
      verbose=verbose,
      print_progress=print_progress,
      solve_null=False,
      MIPGap=1e-6,
      MIPGapAbs=1e-8,
      TimeLimit=None,
      Threads=Threads,
      MIPFocus=None,
      LogFile=None,
    )
    t_obs = float(result_locked.x - result_alt.x)
    if t_obs < 0:
      t_obs = 0.0
    times0 = [patient.time for patient in self.all_patients]
    censored0 = [bool(patient.censored) for patient in self.all_patients]
    group_high = hard_group_high(
      self.observed_parameters,
      self.parameter_threshold,
      self.parameter_max,
    )
    counter = SequentialToyCounter(n_max=n_max, confidence_levels=confidence_levels)
    toy_calc = self._copy_for_outcome_shuffle()
    fit_kwargs = {
      "cox_only": cox_only,
      "patient_wise_only": False,
      "verbose": verbose,
      "print_progress": print_progress,
      "solve_null": False,
      "MIPGap": 1e-6,
      "MIPGapAbs": 1e-8,
      "TimeLimit": None,
      "Threads": Threads,
      "MIPFocus": None,
      "LogFile": None,
    }
    try:
      for i_toy in range(n_max):
        toy_times, toy_censored = weighted_cox_permutation(
          times0, censored0, group_high, hazard_ratio, rng=generator,
        )
        toy_calc._install_toy_outcomes(  # pylint: disable=protected-access
          toy_times, toy_censored, first=i_toy == 0,
        )
        toy_calc._clear_fixed_hazard_ratio_constraint(  # pylint: disable=protected-access
          toy_calc.gurobi_model[0],
        )
        _, toy_alt = toy_calc._fit_null_and_alternative(  # pylint: disable=protected-access
          **fit_kwargs,
        )
        status = toy_calc.excess_at_most(
          hazard_ratio,
          twoNLL_min=float(toy_alt.x),
          level=t_obs,
          cox_only=cox_only,
          verbose=verbose,
          print_progress=print_progress,
          Threads=Threads,
        )
        if status == "unknown":
          locked = toy_calc.compute_2nll_at_hazard_ratio(
            hazard_ratio, cox_only=cox_only, Threads=Threads,
            verbose=verbose, print_progress=print_progress,
          )
          extreme = float(locked.x - toy_alt.x) > t_obs
        else:
          extreme = status == "outside"
        counter.observe(extreme)
        if counter.all_decided():
          break
    finally:
      toy_calc._dispose_gurobi_model()  # pylint: disable=protected-access
    return counter.as_result(t_obs)

  def toy_calibrated_hazard_ratio_interval(  # pylint: disable=too-many-arguments, too-many-locals
    self,
    *,
    n_max: int = 19,
    rng: int | np.random.Generator | None = None,
    cox_only: bool = False,
    confidence_levels: tuple[float, ...] = DEFAULT_CLS,
    hazard_ratio_min: float = 0.1,
    hazard_ratio_max: float = 10.0,
    xtol: float = 0.05,
    Threads: int | None = 1,
    verbose: bool = False,
    print_progress: bool = False,
  ) -> dict[float, tuple[float, float, float]]:
    """
    Toy-calibrated HR interval via endpoint search from the χ² edges.

    Returns a dict mapping each CL to (best_fit, lower, upper).
    """
    generator = as_generator(rng)
    best_fit, chi2_lower, chi2_upper, _best = self.hazard_ratio_confidence_interval(
      cox_only=cox_only,
      confidence_level=max(confidence_levels),
      hazard_ratio_min=hazard_ratio_min,
      hazard_ratio_max=hazard_ratio_max,
    )
    cache: dict[float, ToyTestResult] = {}

    def test_at(hr: float) -> ToyTestResult:
      key = round(float(hr), 10)
      if key not in cache:
        cache[key] = self.hypothesized_hr_toy_test(
          hr,
          n_max=n_max,
          rng=int(generator.integers(0, 2**31 - 1)),
          cox_only=cox_only,
          confidence_levels=confidence_levels,
          Threads=Threads,
          verbose=verbose,
          print_progress=print_progress,
        )
      return cache[key]

    intervals: dict[float, tuple[float, float, float]] = {}
    for cl in confidence_levels:
      def inside(hr: float, _cl=cl) -> bool:
        return test_at(hr).decisions[_cl] == "accept"

      if not inside(best_fit):
        intervals[cl] = (float(best_fit), float(best_fit), float(best_fit))
        continue

      lower_probe = min(chi2_lower, best_fit)
      if lower_probe <= 0:
        lower_probe = max(hazard_ratio_min, best_fit * 0.5)
      lower = bisection_endpoint(
        inside, best_fit, lower_probe,
        xtol=xtol, log_scale=True,
      )
      if inside(lower) and lower > hazard_ratio_min:
        lower = bisection_endpoint(
          inside, best_fit, hazard_ratio_min,
          xtol=xtol, log_scale=True,
        )
      upper_probe = max(chi2_upper, best_fit)
      upper = bisection_endpoint(
        inside, best_fit, upper_probe,
        xtol=xtol, log_scale=True,
      )
      if inside(upper) and upper < hazard_ratio_max:
        upper = bisection_endpoint(
          inside, best_fit, hazard_ratio_max,
          xtol=xtol, log_scale=True,
        )
      intervals[cl] = (float(best_fit), float(lower), float(upper))
    return intervals
