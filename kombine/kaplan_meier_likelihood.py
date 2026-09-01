#pylint: disable=too-many-lines
"""
Kaplan-Meier curve with error bars calculated using the log-likelihood method.
"""

import collections.abc
import dataclasses
import datetime
import functools
import os
import typing
import pathlib

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.typing
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.stats

from .discrete_optimization import (
  binary_search_sign_change,
  cached_level_crossings,
  feasibility_assisted_level_crossings,
)
from .kaplan_meier import (
  KaplanMeierBase,
  KaplanMeierInstance,
)
from .kaplan_meier_MINLP import GurobiWorkStats, MINLPForKM, KaplanMeierPatientNLL
from .utilities import InspectableCache, LOG_ZERO_EPSILON_DEFAULT

@dataclasses.dataclass
class KaplanMeierPlotConfig:  #pylint: disable=too-many-instance-attributes
  """
  Configuration for Kaplan-Meier likelihood plots.

  Attributes:
  times_for_plot: Sequence of time points for plotting the survival probabilities.
  xmax: Maximum time for x-axis range. If provided, limits the plot to [0, xmax].
  include_binomial_only: If True, include error bands for the binomial error alone.
  include_exponential_greenwood: If True, include error bands for the binomial error
                                 using the exponential Greenwood method.
  include_patient_wise_only: If True, include error bands for the patient-wise error alone.
  include_full_NLL: If True, include error bands for the full negative log-likelihood.
  include_best_fit: If True, include the best fit curve in the plot.
  include_nominal: If True, include the nominal Kaplan-Meier curve.
  nominal_label: Label for the nominal curve.
  nominal_color: Color for the nominal curve.
  best_label: Label for the best fit curve.
  best_color: Color for the best fit curve.
  patient_wise_only_suffix: Suffix for the patient-wise only error bands.
  binomial_only_suffix: Suffix for the binomial-only error bands.
  full_NLL_suffix: Suffix for the full NLL error bands.
  exponential_greenwood_suffix: Suffix for the exponential Greenwood error bands.
  CLs: List of confidence levels for the error bands.
  CL_colors: List of colors for the confidence levels.
  CL_colors_greenwood: List of colors for the Greenwood confidence levels.
  CL_hatches: List of hatches for the confidence levels
              for the binomial-only or patient-wise-only error bands.
  create_figure: If True, create a new matplotlib figure for the plot.
  close_figure: If True, close the figure after saving or showing.
  show: If True, display the plot.
  saveas: Path to save the plot image.
  legend_saveas: Path to save the legend separately, or None.
                 If provided, the legend will be left off the main plot.
  print_progress: If True, print progress messages during calculations.
  MIPGap: Relative MIP gap for the optimization solver.
  MIPGapAbs: Absolute MIP gap for the optimization solver.
  rerun_until_convergence: If True, rerun the MINLP optimization until the result
                          converges within tolerances between consecutive iterations.
  include_median_survival: If True, include the median survival time in the legend.
  title: Title for the plot.
  xlabel: Label for the x-axis. If provided, this overrides time_unit.
  time_unit: Unit for the x-axis time label (e.g., "months", "years").
              If provided and xlabel is not explicitly set, the label will be
              "Time (unit)". If neither is provided, the label defaults to "Time".
  ylabel: Label for the y-axis.
  show_grid: If True, display a grid on the plot.
  figsize: Size of the figure as a tuple (width, height).
  tight_layout: If True, use tight layout for the plot.
  legend_fontsize: Font size for the legend.
  label_fontsize: Font size for the axis labels.
  title_fontsize: Font size for the plot title.
  tick_fontsize: Font size for the tick labels.
  legend_loc: Location of the legend in the plot.
  dpi: Dots per inch for the figure resolution.
  pvalue_fontsize: Font size for the p-value text.
  pvalue_format: Format string for p-value display (e.g., '.3g', '.2f').
  """
  times_for_plot: typing.Sequence[float] | None = None
  xmax: float | None = None
  include_binomial_only: bool = False
  include_exponential_greenwood: bool = False
  include_patient_wise_only: bool = False
  include_full_NLL: bool = True
  include_best_fit: bool = True
  include_nominal: bool = True
  nominal_label: str = 'Nominal'
  nominal_color: str = 'red'
  best_label: str = 'Best Fit'
  best_color: str = 'blue'
  patient_wise_only_suffix: str = 'Patient-wise only'
  binomial_only_suffix: str = 'Binomial only'
  full_NLL_suffix: str = ''
  exponential_greenwood_suffix: str = 'Binomial only, exp. Greenwood'
  CLs: list[float] = dataclasses.field(default_factory=lambda: [0.68, 0.95])
  CL_colors: list[str] = dataclasses.field(
    default_factory=lambda: ['dodgerblue', 'skyblue', 'lightblue', 'lightcyan']
  )
  CL_colors_greenwood: list[str] = dataclasses.field(
    default_factory=lambda: ['darkorange', 'gold', 'khaki', 'lightyellow']
  )
  CL_hatches: list[str] = dataclasses.field(
    default_factory=lambda: ['//', '\\\\', 'xx', '++']
  )
  create_figure: bool = True
  close_figure: bool | None = None
  show: bool = False
  saveas: os.PathLike | str | None = None
  legend_saveas: os.PathLike | str | None = None
  print_progress: bool = False
  MIPGap: float | None = None
  MIPGapAbs: float | None = None
  rerun_until_convergence: bool = False
  include_median_survival: bool = False
  title: str | None = "Kaplan-Meier Curves"
  xlabel: str | None = None
  time_unit: str | None = None
  ylabel: str = "Survival Probability"
  show_grid: bool = True
  figsize: tuple[float, float] = (10, 7)
  tight_layout: bool = True
  legend_fontsize: int = 10
  label_fontsize: int = 12
  title_fontsize: int = 14
  tick_fontsize: int = 10
  legend_loc: matplotlib.typing.LegendLocType | None = None
  dpi: int = 100
  pvalue_fontsize: int = 12
  pvalue_format: str = '.3g'

  def __post_init__(self):
    """
    Post-initialization validation and default adjustments.
    """
    if self.include_binomial_only and self.include_patient_wise_only:
      raise ValueError("include_binomial_only and include_patient_wise_only cannot both be True")
    if not (
      self.include_binomial_only
      or self.include_patient_wise_only
      or self.include_full_NLL
      or self.include_exponential_greenwood
      or self.include_nominal
    ):
      raise ValueError(
        "At least one of include_binomial_only, include_patient_wise_only, "
        "include_full_NLL, include_exponential_greenwood, or include_nominal must be True"
      )

    # Helper variable for whether error bands will be computed
    include_error_bands = (
      self.include_full_NLL
      or self.include_patient_wise_only
      or self.include_binomial_only
      or self.include_exponential_greenwood
    )

    include_hatched_error_bands = (
      self.include_full_NLL and (
        self.include_patient_wise_only
        or self.include_binomial_only
      )
    )

    # Error if best fit is requested but no error band options are available
    if self.include_best_fit and not include_error_bands:
      raise ValueError(
        "include_best_fit=True requires at least one of include_full_NLL, "
        "include_patient_wise_only, include_binomial_only, or "
        "include_exponential_greenwood to be True"
      )

    # Only validate CL_colors length when error bands will be computed
    if include_error_bands and len(self.CLs) > len(self.CL_colors):
      raise ValueError(
        f"Not enough colors provided for {len(self.CLs)} CLs, "
        f"got {len(self.CL_colors)} colors"
      )
    self.CL_colors = self.CL_colors[:len(self.CLs)]

    if (
      include_hatched_error_bands
      and len(self.CLs) > len(self.CL_hatches)
    ):
      raise ValueError(
        f"Not enough hatches provided for {len(self.CLs)} CLs, "
        f"got {len(self.CL_hatches)} hatches"
      )
    self.CL_hatches = self.CL_hatches[:len(self.CLs)]

class KaplanMeierLikelihood(KaplanMeierBase):
  """
  Kaplan-Meier curve with error bars calculated using the log-likelihood method.
  """
  __default_MIPGap = 1e-4
  __default_MIPGapAbs = 1e-7

  def __init__( # pylint: disable=too-many-arguments
    self,
    *,
    all_patients: list[KaplanMeierPatientNLL],
    parameter_min: float,
    parameter_max: float,
    endpoint_epsilon: float = 1e-6,
    log_zero_epsilon: float = LOG_ZERO_EPSILON_DEFAULT,
    collapse_consecutive_deaths: bool = True,
    time_unit: str | None = None,
  ):
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_max = parameter_max
    self.__endpoint_epsilon = endpoint_epsilon
    self.__log_zero_epsilon = log_zero_epsilon
    self.__collapse_consecutive_deaths = collapse_consecutive_deaths
    self.__time_unit = time_unit
    self._last_gurobi_work_stats: GurobiWorkStats | None = None

  @property
  def last_gurobi_work_stats(self) -> GurobiWorkStats | None:
    """Work totals from the most recent ``survival_probabilities_likelihood`` call."""
    return self._last_gurobi_work_stats

  @property
  def all_patients(self) -> list[KaplanMeierPatientNLL]:
    """
    The list of all patients.
    """
    return self.__all_patients

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
  def time_unit(self) -> str | None:
    """
    The time unit for the x-axis label, inherited from the datacard.
    """
    return self.__time_unit

  @property
  def patient_death_times(self) -> frozenset:
    """
    The survival times of the patients who died.
    (excludes censored patients)
    """
    return frozenset(p.time for p in self.all_patients if not p.censored)
  @property
  def patient_censored_times(self) -> frozenset:
    """
    The survival times of the patients who were censored.
    """
    return frozenset(p.time for p in self.all_patients if p.censored)

  @functools.cached_property
  def nominalkm(self) -> KaplanMeierInstance:
    """
    The nominal Kaplan-Meier curve.
    """
    return KaplanMeierInstance(
      all_patients=[p.nominal for p in self.all_patients],
      parameter_min=self.parameter_min,
      parameter_max=self.parameter_max,
    )

  def minlp_for_km(
    self,
    time_point: float,
    *,
    binomial_only: bool = False,
    patient_wise_only: bool = False,
  ):
    """
    Get the MINLP for the given time point and likelihood mode.
    """
    return MINLPForKM(
      all_patients=self.all_patients,
      parameter_min=self.parameter_min,
      parameter_max=self.parameter_max,
      time_point=time_point,
      endpoint_epsilon=self.__endpoint_epsilon,
      log_zero_epsilon=self.__log_zero_epsilon,
      collapse_consecutive_deaths=self.__collapse_consecutive_deaths,
      binomial_only=binomial_only,
      patient_wise_only=patient_wise_only,
    )

  def get_twoNLL_function( # pylint: disable=too-many-arguments
    self,
    time_point: float,
    *,
    binomial_only=False,
    patient_wise_only=False,
    verbose=False,
    print_progress=False,
    MIPGap=None,
    MIPGapAbs=None,
    rerun_until_convergence=False,
    assignment_starts=None,
    minlp_time_limit: float | None = None,
  ) -> tuple[
    InspectableCache[float | None, scipy.optimize.OptimizeResult],
    InspectableCache[float | None, float],
    MINLPForKM,
  ]:
    """
    Get the twoNLL function for the given time point.

    Args:
      rerun_until_convergence: If True, run the MINLP at least twice, repeating
                              until result.x converges within MIPGap and MIPGapAbs
                              tolerances between consecutive iterations.
    """
    if MIPGap is None:
      MIPGap = self.__default_MIPGap
    if MIPGapAbs is None:
      MIPGapAbs = self.__default_MIPGapAbs

    minlp = self.minlp_for_km(
      time_point=time_point,
      binomial_only=binomial_only,
      patient_wise_only=patient_wise_only,
    )
    if assignment_starts is not None:
      minlp.seed_assignment_starts(assignment_starts)

    @InspectableCache
    def run_MINLP(expected_probability: float | None) -> scipy.optimize.OptimizeResult:
      """
      Run the MINLP for the given expected probability.
      """
      result = minlp.run_MINLP(
        expected_probability=expected_probability,
        binomial_only=binomial_only,
        patient_wise_only=patient_wise_only,
        verbose=verbose,
        print_progress=print_progress,
        MIPGap=MIPGap,
        MIPGapAbs=MIPGapAbs,
        TimeLimit=minlp_time_limit,
      )

      if not rerun_until_convergence or not result.success:
        return result

      # Rerun until convergence: repeat until result.x is stable between iterations
      prev_x = result.x
      last_successful_result = result
      max_iterations = 10  # Safety limit to prevent infinite loops
      iteration = 1

      while iteration < max_iterations:
        try:
          result = minlp.run_MINLP(
            expected_probability=expected_probability,
            binomial_only=binomial_only,
            patient_wise_only=patient_wise_only,
            verbose=verbose,
            print_progress=print_progress,
            MIPGap=MIPGap,
            MIPGapAbs=MIPGapAbs,
            TimeLimit=minlp_time_limit,
          )
        except Exception: #pylint: disable=broad-exception-caught
          # If rerun raises an exception, return the last successful result
          return last_successful_result

        if not result.success:
          # If rerun fails, return the last successful result
          return last_successful_result

        # Check convergence using both relative and absolute tolerances
        abs_diff = abs(result.x - prev_x)
        rel_diff = abs_diff / max(abs(prev_x), 1e-10)  # Avoid division by zero

        if abs_diff <= MIPGapAbs and rel_diff <= MIPGap:
          # Converged
          break

        prev_x = result.x
        last_successful_result = result
        iteration += 1

      return result

    @InspectableCache
    def twoNLL(expected_probability: float | None) -> float:
      """
      The negative log-likelihood function.
      """
      result = run_MINLP(expected_probability)
      if not result.success:
        return np.inf
      return result.x
    return run_MINLP, twoNLL, minlp

  def calculate_possible_probabilities(self, time_point: float) -> np.ndarray:
    """
    Get the possible probabilities for the given patients.
    """
    return np.array(sorted(self.minlp_for_km(time_point).possible_probabilities))

  @functools.cached_property
  def __possible_probabilities(self) -> dict[float, np.ndarray]:
    return {}

  def possible_probabilities(self, time_point: float) -> np.ndarray:
    """
    Get the possible probabilities for the given time point.
    This is a cached property to avoid recalculating the probabilities multiple times.
    """
    if time_point not in self.__possible_probabilities:
      self.__possible_probabilities[time_point] = self.calculate_possible_probabilities(time_point)
    return self.__possible_probabilities[time_point]

  def best_probability( #pylint: disable=too-many-arguments
    self,
    run_MINLP: collections.abc.Callable[[float | None], scipy.optimize.OptimizeResult],
    time_point: float | None = None,
  ) -> tuple[float, float]:
    """
    Find the expected probability that minimizes the negative log-likelihood
    for the given time point.
    """
    result = run_MINLP(None)
    if not result.success:
      raise RuntimeError(
        f"Failed to find the best probability for time point {time_point}"
      )
    best_prob = result.km_probability
    twoNLL_min = result.x
    if not 0 <= best_prob <= 1:
      raise ValueError(
        f"Best probability {best_prob} is not in [0, 1] for time point {time_point}"
      )
    return best_prob, twoNLL_min

  def survival_probabilities_exponential_greenwood(
    self,
    CLs: list[float],
    times_for_plot: typing.Sequence[float],
    *,
    binomial_only=False,
    patient_wise_only=False,
  ):
    """
    Calculate the survival probabilities using the exponential Greenwood method.
    """
    if patient_wise_only or not binomial_only:
      raise ValueError(
        "Exponential Greenwood confidence intervals"
        "can only include the binomial error"
      )
    return self.nominalkm.survival_probabilities_exponential_greenwood(
      CLs=CLs,
      times_for_plot=times_for_plot,
    )

  def survival_probabilities_likelihood( # pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-arguments
    self,
    CLs: list[float],
    times_for_plot: typing.Sequence[float],
    *,
    binomial_only=False,
    patient_wise_only=False,
    gurobi_verbose=False,
    optimize_verbose=False,
    print_progress=False,
    MIPGap=None,
    MIPGapAbs=None,
    rerun_until_convergence=False,
    crossing_mode: typing.Literal["full", "feasibility"] = "full",
    component_cuts: bool = False,
    binom_time_cuts: bool = False,
    cost_biased_bisection: bool = True,
    r_prior: float = 0.5,
    kappa: float = 4.0,
    minlp_time_limit: float | None = None,
    oracle_time_limit: float | None = None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the survival probabilities for the given quantiles.
    
    Parameters
    ----------
    CLs : list[float]
        Confidence levels for the survival probabilities
    times_for_plot : sequence of float
        Time points for which to calculate survival probabilities
    binomial_only : bool, default False
        If True, only use binomial constraints
    patient_wise_only : bool, default False  
        If True, only use patient-wise constraints
    gurobi_verbose : bool, default False
        If True, enable verbose Gurobi output
    optimize_verbose : bool, default False
        If True, enable verbose optimization output
    print_progress : bool, default False
        If True, print progress information
    MIPGap : float, optional
        Gurobi MIP gap tolerance (used for objective function tolerance)
    MIPGapAbs : float, optional
        Gurobi absolute MIP gap tolerance (used for objective function tolerance)
    rerun_until_convergence : bool, default False
        If True, rerun the MINLP until the result converges within tolerances
    crossing_mode : {"full", "feasibility"}, default "full"
        How to locate profile-likelihood CL endpoints. ``full`` minimizes
        2NLL at every trial ``p`` (legacy). ``feasibility`` uses
        ``excess_at_most`` sign tests for bracketing, with full minimizes
        only for MLE, oracle fallbacks, and brentq polish.
    component_cuts : bool, default False
        When ``crossing_mode="feasibility"``, also add redundant
        ``2*binom <= T`` and ``2*patient <= T`` cuts. Off by default;
        sum-only was faster with matching endpoints in hard-case ablations.
    binom_time_cuts : bool, default False
        When ``crossing_mode="feasibility"``, also add redundant
        ``2*binom_piece[i] <= T`` per death time.
    cost_biased_bisection : bool, default True
        When ``crossing_mode="feasibility"``, use cost-biased oracle bisection
        (prior probe fraction ``r_prior``; adapts from oracle timings).
    r_prior : float, default 0.5
        Initial probe fraction between inside and outside endpoints (0.5 =
        midpoint bisection).
    kappa : float, default 4.0
        Prior strength (pseudo-counts) for the Beta prior on ``r``.
    minlp_time_limit : float, optional
        Gurobi TimeLimit (seconds) for each full ``run_MINLP`` call. None = no limit.
    oracle_time_limit : float, optional
        Gurobi TimeLimit (seconds) for each ``excess_at_most`` oracle call. None = no limit.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Best probabilities and survival probabilities for each confidence level
    """
    if crossing_mode not in {"full", "feasibility"}:
      raise ValueError(
        f"crossing_mode must be 'full' or 'feasibility', got {crossing_mode!r}"
      )
    # Set default tolerance values if not provided
    if MIPGap is None:
      MIPGap = self.__default_MIPGap
    if MIPGapAbs is None:
      MIPGapAbs = self.__default_MIPGapAbs

    best_probabilities = []
    survival_probabilities = []
    for i, t in enumerate(times_for_plot, start=1):
      if print_progress:
        print(
          f"[{datetime.datetime.now()}] Calculating survival probabilities "
          f"for time point {t:.2f} ({i} / {len(times_for_plot)})",
          flush=True,
        )
      survival_probabilities_time_point = []
      survival_probabilities.append(survival_probabilities_time_point)
      run_MINLP, twoNLL, minlp = self.get_twoNLL_function(
        time_point=t,
        binomial_only=binomial_only,
        patient_wise_only=patient_wise_only,
        verbose=gurobi_verbose,
        print_progress=print_progress,
        MIPGap=MIPGap,
        MIPGapAbs=MIPGapAbs,
        rerun_until_convergence=rerun_until_convergence,
        minlp_time_limit=minlp_time_limit,
      )
      # Find the expected probability that minimizes the negative log-likelihood
      # for the given time point
      try:
        best_prob, twoNLL_min = self.best_probability(
          run_MINLP=run_MINLP,
          time_point=t,
        )
      except Exception as e:
        raise RuntimeError(
          f"Failed to find the best probability for time point {t}"
        ) from e
      best_probabilities.append(best_prob)
      if patient_wise_only:
        best_prob_clipped = best_prob
      else:
        best_prob_clipped = np.clip(
          best_prob,
          self.__endpoint_epsilon,
          1 - self.__endpoint_epsilon,
        )

      if patient_wise_only:
        for CL in CLs:
          if t < min(self.patient_death_times):
            survival_probabilities_time_point.append((1, 1))
            continue

          d2NLLcut = scipy.stats.chi2.ppf(CL, 1).item()
          def objective_function(
            expected_probability: float,
            twoNLL=twoNLL, twoNLL_min=twoNLL_min, d2NLLcut=d2NLLcut
          ) -> float:
            return twoNLL(expected_probability) - twoNLL_min - d2NLLcut
          if best_prob == best_prob_clipped:
            np.testing.assert_allclose(
              objective_function(best_prob),
              -d2NLLcut,
              atol=1e-2,
            )

          probs = self.possible_probabilities(time_point=t)
          if best_prob not in probs:
            probs = np.append(probs, best_prob)
            probs = np.sort(probs)
          i_best = int(np.searchsorted(probs, best_prob))
          np.testing.assert_equal(
            probs[i_best],
            best_prob,
            err_msg=f"Best probability {best_prob} not found in possible probabilities {probs}",
          )

          if objective_function(probs[-1]) < 0:
            upper_bound = 1
          else:
            upper = binary_search_sign_change(
              objective_function=objective_function,
              probs=probs,
              lo=i_best,
              hi=len(probs) - 1,
              verbose=optimize_verbose,
              MIPGap=MIPGap,
              MIPGapAbs=MIPGapAbs,
            )
            if upper is None:
              raise RuntimeError("No upper sign change found")
            upper_bound = upper

          if objective_function(probs[0]) < 0:
            lower_bound = 0
          else:
            lower = binary_search_sign_change(
              objective_function=objective_function,
              probs=probs,
              lo=0,
              hi=i_best,
              verbose=optimize_verbose,
              MIPGap=MIPGap,
              MIPGapAbs=MIPGapAbs,
            )
            if lower is None:
              raise RuntimeError("No lower sign change found")
            lower_bound = lower
          survival_probabilities_time_point.append((lower_bound, upper_bound))
      else:
        levels = [scipy.stats.chi2.ppf(cl, 1).item() for cl in CLs]

        def profile_excess(
          expected_probability: float,
          _twoNLL=twoNLL,
          _twoNLL_min=twoNLL_min,
        ) -> float:
          return _twoNLL(expected_probability) - _twoNLL_min

        if best_prob == best_prob_clipped:
          np.testing.assert_allclose(profile_excess(best_prob), 0.0, atol=1e-2)

        brentq_xtol = 1e-4
        brentq_rtol = 1e-4
        eps = self.__endpoint_epsilon
        one_minus_eps = 1.0 - eps

        def resolve_crossings(
          x_outer: float,
          x_inner: float,
        ) -> tuple[list[float], list[typing.Literal["inside", "outside"] | None]]:
          if crossing_mode == "full":
            return cached_level_crossings(
              profile_excess,
              x_outer,
              x_inner,
              levels,
              xtol=brentq_xtol,
              rtol=brentq_rtol,
            ), [None] * len(levels)

          def sign_oracle(
            p: float,
            level: float,
            _minlp=minlp,
            _twoNLL_min=twoNLL_min,
          ) -> tuple[typing.Literal["inside", "outside", "unknown"], float]:
            status = _minlp.excess_at_most(
              p,
              twoNLL_min=_twoNLL_min,
              level=level,
              component_cuts=component_cuts,
              binom_time_cuts=binom_time_cuts,
              verbose=gurobi_verbose,
              print_progress=print_progress,
              TimeLimit=oracle_time_limit,
            )
            # Prefer Gurobi Work (sleep-immune). Fall back to a tiny positive
            # cost so the tracker still gets an outcome-count update.
            work = _minlp.last_oracle_work
            return status, float(work) if work is not None and work > 0.0 else 1e-9

          outer_oracle: list[typing.Literal["inside", "outside"] | None] = []
          crossings = feasibility_assisted_level_crossings(
            profile_excess,
            sign_oracle,
            x_outer,
            x_inner,
            levels,
            xtol=brentq_xtol,
            rtol=brentq_rtol,
            cost_biased_bisection=cost_biased_bisection,
            r_prior=r_prior,
            kappa=kappa,
            outer_oracle=outer_oracle,
          )
          return crossings, outer_oracle

        def _clip_lower_endpoint(
          crossings: list[float],
          outer_oracle: list[typing.Literal["inside", "outside"] | None],
        ) -> list[float]:
          clipped: list[float] = []
          for x, level, ostatus in zip(crossings, levels, outer_oracle, strict=True):
            if ostatus == "inside":
              clipped.append(0.0)
            elif ostatus == "outside":
              clipped.append(float(x))
            else:
              f_outer = profile_excess(eps)
              clipped.append(0.0 if f_outer <= level else float(x))
          return clipped

        def _clip_upper_endpoint(
          crossings: list[float],
          outer_oracle: list[typing.Literal["inside", "outside"] | None],
        ) -> list[float]:
          clipped: list[float] = []
          for x, level, ostatus in zip(crossings, levels, outer_oracle, strict=True):
            if ostatus == "inside":
              clipped.append(1.0)
            elif ostatus == "outside":
              clipped.append(float(x))
            else:
              f_outer = profile_excess(one_minus_eps)
              clipped.append(1.0 if f_outer <= level else float(x))
          return clipped

        if best_prob <= eps:
          lowers = [0.0] * len(CLs)
        else:
          lowers, lower_outer = resolve_crossings(eps, best_prob_clipped)
          lowers = _clip_lower_endpoint(lowers, lower_outer)

        if best_prob >= one_minus_eps:
          uppers = [1.0] * len(CLs)
        else:
          uppers, upper_outer = resolve_crossings(one_minus_eps, best_prob_clipped)
          uppers = _clip_upper_endpoint(uppers, upper_outer)

        if print_progress:
          ws = minlp.work_stats
          print(
            f"[{datetime.datetime.now()}]   CL crossings unique p "
            f"evaluations={len(twoNLL.cache)} for time point {t} "
            f"(mode={crossing_mode}); "
            f"Work oracle={ws.oracle_work:.3f} minimize={ws.minimize_work:.3f} "
            f"total={ws.total_work:.3f} "
            f"(oracle_calls={ws.oracle_calls}, minimize_calls={ws.minimize_calls}, "
            f"oracle_outside={ws.oracle_outside_calls})",
            flush=True,
          )
        for lower_bound, upper_bound in zip(lowers, uppers, strict=True):
          survival_probabilities_time_point.append((lower_bound, upper_bound))
      self._last_gurobi_work_stats = minlp.work_stats
    return np.array(best_probabilities), np.array(survival_probabilities)

  def plot(self, config: KaplanMeierPlotConfig | None = None, **kwargs) -> dict:
    """
    Plots the Kaplan-Meier curves based on the provided configuration.

    The time_unit priority is:
    1. kwargs["time_unit"] if explicitly provided and not None
    2. config.time_unit if config is provided and its time_unit is not None
    3. self.time_unit as fallback
    """
    # Determine the effective time_unit following priority:
    # kwargs["time_unit"] > config.time_unit > self.time_unit
    effective_time_unit = None
    if kwargs.get('time_unit', None) is not None:
      effective_time_unit = kwargs['time_unit']
    elif config is not None and config.time_unit is not None:
      effective_time_unit = config.time_unit
    else:
      effective_time_unit = self.time_unit

    kwargs['time_unit'] = effective_time_unit

    if config is None:
      # Build kwargs with the effective time_unit for config creation
      config = KaplanMeierPlotConfig(**kwargs)
    else:
      # If config is provided and kwargs are also given, update config with kwargs
      # Only override config fields with kwargs values
      config = dataclasses.replace(config, **kwargs)
    # Use config.times_for_plot, falling back to self.get_times_for_plot(xmax) if None
    times_for_plot = config.times_for_plot
    if times_for_plot is None:
      times_for_plot = self.get_times_for_plot(xmax=config.xmax)

    fig, ax = self._prepare_figure(config)

    # Plot nominal curve and censored points
    results: dict[str, npt.NDArray[np.float64]] = self._plot_nominal(ax, config, times_for_plot)

    # Calculate and plot confidence bands and best fit curve
    results.update(self._calculate_and_plot_confidence_bands(ax, config, times_for_plot))

    self._plot_censored(
      ax,
      config,
      results["x"],
      results["nominal"] if config.include_nominal else results["best_fit"],
    )

    # Finalize plot elements (legend, labels, grid, save/show/close)
    self._finalize_plot(fig, ax, config)

    # Return results for further inspection if needed
    return results

  def _prepare_figure(
    self,
    config: KaplanMeierPlotConfig,
  ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Prepares the matplotlib figure and axes."""
    if config.create_figure:
      fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
      fig = plt.gcf() # Get current figure
      ax = plt.gca() # Get current axes if figure already exists
    return fig, ax

  def _plot_nominal(
    self,
    ax: matplotlib.axes.Axes,
    config: KaplanMeierPlotConfig,
    times_for_plot: typing.Sequence[float],
  ):
    """Plots the nominal Kaplan-Meier curve and censored patient markers."""
    nominal_x, nominal_y = self.nominalkm.points_for_plot(times_for_plot=times_for_plot)
    label = config.nominal_label
    if config.include_median_survival:
      MST = self.nominalkm.median_survival_time(
        times_for_plot=nominal_x,
        survival_probabilities=nominal_y,
      )
      label += f" (MST={MST:.1f})".replace("inf", r"$\infty$")
    if config.include_nominal:
      ax.plot(
        nominal_x,
        nominal_y,
        label=label,
        color=config.nominal_color,
        linestyle='--'
      )

    return {
      "x": nominal_x,
      "nominal": nominal_y,
    }

  def _plot_censored(
    self,
    ax: matplotlib.axes.Axes,
    config: KaplanMeierPlotConfig,
    x_for_plot: typing.Sequence[float] | npt.NDArray[np.float64],
    y_for_plot: typing.Sequence[float] | npt.NDArray[np.float64],
  ):
    patient_censored_times = sorted(self.nominalkm.patient_censored_times)
    censored_times_probabilities = [
      y_for_plot[
        max(i for i, t in enumerate(x_for_plot) if t <= patient_censored_time)
      ]
      for patient_censored_time in patient_censored_times
    ]
    ax.plot(
      patient_censored_times,
      censored_times_probabilities,
      marker='|',
      color=config.nominal_color if config.include_nominal else config.best_color,
      markersize=8,
      markeredgewidth=1.5,
      linestyle="",
    )

  def _plot_confidence_band_fill( # pylint: disable=too-many-arguments, too-many-locals
    self,
    ax: matplotlib.axes.Axes,
    config: KaplanMeierPlotConfig,
    times_for_plot: typing.Sequence[float],
    CL_probabilities_data: np.ndarray,
    *,
    label_suffix: str = "",
    use_hatches: bool = False,
    colors: list[str] | None = None,
  ):
    """
    Helper to plot confidence bands using fill_between.
    """
    results = {}
    if colors is None:
      colors = config.CL_colors
    for CL, color, hatch, (p_minus, p_plus) in zip(
      config.CLs,
      colors,
      config.CL_hatches,
      CL_probabilities_data.transpose(1, 2, 0),
      strict=True,
    ):
      x_minus, y_minus = self.get_points_for_plot(times_for_plot, p_minus)
      x_plus, y_plus = self.get_points_for_plot(times_for_plot, p_plus)
      np.testing.assert_array_equal(x_minus, x_plus)

      if CL > 0.9999:
        label = f'{CL:.6%} CL'
      elif CL > 0.99:
        label = f'{CL:.2%} CL'
      else:
        label = f'{CL:.0%} CL'

      if label_suffix:
        label += f' ({label_suffix})'

      if config.include_median_survival:
        MST_low = self.median_survival_time(
          times_for_plot=x_minus,
          survival_probabilities=y_minus,
        )
        MST_high = self.median_survival_time(
          times_for_plot=x_plus,
          survival_probabilities=y_plus,
        )
        label += f" (MST$\\in$({MST_low:.1f}, {MST_high:.1f}))".replace("inf", r"$\infty$")

      if use_hatches:
        ax.fill_between(
          x_minus,
          y_minus,
          y_plus,
          edgecolor=color,
          facecolor='none',
          hatch=hatch,
          alpha=0.5,
          label=label,
        )
      else:
        ax.fill_between(
          x_minus,
          y_minus,
          y_plus,
          color=color,
          alpha=0.5,
          label=label,
        )
      results[label] = (y_minus, y_plus)
    return results

  def _calculate_and_plot_confidence_bands( # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    self,
    ax: matplotlib.axes.Axes,
    config: KaplanMeierPlotConfig,
    times_for_plot: typing.Sequence[float]
  ):
    """Calculates and plots the confidence bands and best-fit curve."""

    # --- storage for computed results (no plotting yet) ---
    best_probabilities = None
    CL_probabilities = None
    results = {}

    best_prob_full = None
    CL_prob_full = None

    best_prob_binomial = None
    CL_prob_binomial = None

    best_prob_greenwood = None
    CL_prob_greenwood = None

    best_prob_patient = None
    CL_prob_patient = None

    # --- compute required probability sets (no fills plotted here) ---
    if config.include_full_NLL:
      best_prob_full, CL_prob_full = self.survival_probabilities_likelihood(
        CLs=config.CLs,
        times_for_plot=times_for_plot,
        print_progress=config.print_progress,
        MIPGap=config.MIPGap,
        MIPGapAbs=config.MIPGapAbs,
        rerun_until_convergence=config.rerun_until_convergence,
      )

    if config.include_binomial_only:
      best_prob_binomial, CL_prob_binomial = self.survival_probabilities_likelihood(
        CLs=config.CLs,
        times_for_plot=times_for_plot,
        binomial_only=True,
        print_progress=config.print_progress,
        MIPGap=config.MIPGap,
        MIPGapAbs=config.MIPGapAbs,
        rerun_until_convergence=config.rerun_until_convergence,
      )

    if config.include_exponential_greenwood:
      best_prob_greenwood, CL_prob_greenwood = self.survival_probabilities_exponential_greenwood(
        CLs=config.CLs,
        times_for_plot=times_for_plot,
        binomial_only=True,
      )

    if config.include_patient_wise_only:
      best_prob_patient, CL_prob_patient = self.survival_probabilities_likelihood(
        CLs=config.CLs,
        times_for_plot=times_for_plot,
        patient_wise_only=True,
        print_progress=config.print_progress,
        MIPGap=config.MIPGap,
        MIPGapAbs=config.MIPGapAbs,
        rerun_until_convergence=config.rerun_until_convergence,
      )

    # --- determine which set is the 'best' (preserve original precedence) ---
    if config.include_full_NLL:
      best_probabilities = best_prob_full
      CL_probabilities = CL_prob_full

    if config.include_binomial_only:
      if not config.include_full_NLL:
        best_probabilities = best_prob_binomial
        CL_probabilities = CL_prob_binomial

    if config.include_exponential_greenwood:
      # does not override an explicit full/binomial preference
      if not config.include_full_NLL and not config.include_binomial_only:
        best_probabilities = best_prob_greenwood
        CL_probabilities = CL_prob_greenwood

    if config.include_patient_wise_only:
      if not config.include_full_NLL:
        best_probabilities = best_prob_patient
        CL_probabilities = CL_prob_patient

    # --- fail fast if we need error bands but couldn't determine a best probability set ---
    has_error_band_option = (
      config.include_full_NLL
      or config.include_binomial_only
      or config.include_exponential_greenwood
      or config.include_patient_wise_only
    )
    if has_error_band_option and (best_probabilities is None or CL_probabilities is None):
      raise ValueError(
        "Could not determine best_probabilities or CL_probabilities. "
        "Check config flags and data returned by likelihood/greenwood calls."
      )

    # --- PLOT PHASE: plot best-fit first (so it appears above fills added here) ---
    if config.include_best_fit:
      best_x, best_y = self.get_points_for_plot(times_for_plot, best_probabilities)
      label = config.best_label
      if config.include_median_survival:
        MST = self.median_survival_time(
          times_for_plot=best_x,
          survival_probabilities=best_y,
        )
        label += f" (MST={MST:.1f})"
      ax.plot(
        best_x,
        best_y,
        label=label,
        color=config.best_color,
        linestyle='--'
      )
      results["best_fit"] = best_y

    # --- now plot confidence-band fills in the original sequence ---
    if config.include_full_NLL:
      assert CL_prob_full is not None
      CL_results = self._plot_confidence_band_fill(
        ax, config, times_for_plot, CL_prob_full, use_hatches=False
      )
      results.update(CL_results)

    if config.include_binomial_only:
      assert CL_prob_binomial is not None
      if config.include_full_NLL:
        CL_results = self._plot_confidence_band_fill(
          ax, config, times_for_plot, CL_prob_binomial,
          label_suffix=config.binomial_only_suffix, use_hatches=True
        )
      else:
        CL_results = self._plot_confidence_band_fill(
          ax, config, times_for_plot, CL_prob_binomial,
          label_suffix=config.binomial_only_suffix, use_hatches=False
        )
      results.update(CL_results)

    if config.include_exponential_greenwood:
      assert CL_prob_greenwood is not None
      CL_results = self._plot_confidence_band_fill(
        ax, config, times_for_plot, CL_prob_greenwood,
        label_suffix=config.exponential_greenwood_suffix, use_hatches=False,
        colors=config.CL_colors_greenwood[:len(config.CLs)],
      )
      results.update(CL_results)

    if config.include_patient_wise_only:
      assert CL_prob_patient is not None
      if config.include_full_NLL:
        CL_results = self._plot_confidence_band_fill(
          ax, config, times_for_plot, CL_prob_patient,
          label_suffix=config.patient_wise_only_suffix, use_hatches=True
        )
      else:
        CL_results = self._plot_confidence_band_fill(
          ax, config, times_for_plot, CL_prob_patient,
          label_suffix=config.patient_wise_only_suffix, use_hatches=False
        )
      results.update(CL_results)

    return results

  def _finalize_plot(  #pylint: disable=too-many-branches
    self,
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    config: KaplanMeierPlotConfig,
  ):
    """Adds final plot elements and handles saving/showing/closing."""
    # Resolve xlabel: explicit xlabel > time_unit > default "Time"
    xlabel = config.xlabel
    if xlabel is None:
      if config.time_unit is not None:
        xlabel = f"Time ({config.time_unit})"
      else:
        xlabel = "Time"
    ax.set_xlabel(xlabel, fontsize=config.label_fontsize)
    ax.set_ylabel(config.ylabel, fontsize=config.label_fontsize)
    if config.title is not None:
      ax.set_title(config.title, fontsize=config.title_fontsize)
    ax.grid(visible=config.show_grid)
    ax.set_ylim(0, 1.05) # Ensure y-axis is from 0 to 1.05 for survival probability

    # Set x-axis limits if xmax is specified
    if config.xmax is not None:
      ax.set_xlim(0, config.xmax)

    #set font sizes
    ax.tick_params(labelsize=config.tick_fontsize)
    if config.title is not None:
      ax.title.set_fontsize(config.title_fontsize)

    if config.tight_layout:
      fig.tight_layout()

    if config.saveas is not None:
      save_path = pathlib.Path(config.saveas)
      save_path.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(save_path, bbox_inches='tight', dpi=config.dpi)

    if config.legend_saveas is None:
      ax.legend(fontsize=config.legend_fontsize, loc=config.legend_loc)
    elif config.legend_saveas == os.devnull:
      #don't add a legend to the main plot
      pass
    else:
      handles, labels = ax.get_legend_handles_labels()
      fig_legend, ax_legend = plt.subplots(figsize=config.figsize, dpi=config.dpi)
      ax_legend.axis("off")
      legend = ax_legend.legend(
        handles, labels,
        fontsize=config.legend_fontsize,
        loc="center"
      )
      #crop whitespace
      fig_legend.canvas.draw()
      bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())
      fig_legend.set_size_inches(bbox.width, bbox.height)

      fig_legend.savefig(config.legend_saveas, bbox_inches="tight")
      plt.close(fig_legend)

    if config.show:
      plt.show()

    if config.close_figure is None: # Default behavior: close if saving, don't close if showing
      if config.saveas is not None:
        plt.close(fig)
    elif config.close_figure:
      plt.close(fig)
