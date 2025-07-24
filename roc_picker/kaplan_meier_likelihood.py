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
import numpy as np
import scipy.optimize
import scipy.stats

from .discrete_optimization import (
  binary_search_sign_change,
  minimize_discrete_single_minimum,
)
from .kaplan_meier import (
  KaplanMeierBase,
  KaplanMeierInstance,
)
from .kaplan_meier_MINLP import MINLPForKM, KaplanMeierPatientNLL
from .utilities import InspectableCache

@dataclasses.dataclass
class KaplanMeierPlotConfig:  #pylint: disable=too-many-instance-attributes
  """
  Configuration for Kaplan-Meier likelihood plots.
  """
  times_for_plot: typing.Sequence[float] | None = None
  include_binomial_only: bool = False
  include_patient_wise_only: bool = False
  include_full_NLL: bool = True
  include_best_fit: bool = True
  include_nominal: bool = True
  nominal_label: str = 'Nominal'
  nominal_color: str = 'blue'
  best_label: str = 'Best Fit'
  best_color: str = 'red'
  CLs: list[float] = dataclasses.field(default_factory=lambda: [0.68, 0.95])
  CL_colors: list[str] = dataclasses.field(
    default_factory=lambda: ['dodgerblue', 'skyblue', 'lightblue', 'lightcyan']
  )
  CL_hatches: list[str] = dataclasses.field(
    default_factory=lambda: ['//', '\\\\', 'xx', '++']
  )
  create_figure: bool = True
  close_figure: typing.Optional[bool] = None
  show: bool = False
  saveas: typing.Optional[os.PathLike] = None
  print_progress: bool = False
  MIPGap: typing.Optional[float] = None
  MIPGapAbs: typing.Optional[float] = None
  include_median_survival: bool = False
  title: typing.Optional[str] = "Kaplan-Meier Curves"
  xlabel: str = "Time"
  ylabel: str = "Survival Probability"
  show_grid: bool = True
  figsize: tuple[float, float] = (10, 7)
  dpi: int = 100

  def __post_init__(self):
    """
    Post-initialization validation and default adjustments.
    """
    if self.include_binomial_only and self.include_patient_wise_only:
      raise ValueError("include_binomial_only and include_patient_wise_only cannot both be True")
    if not (self.include_binomial_only or self.include_patient_wise_only or self.include_full_NLL):
      raise ValueError(
        "At least one of include_binomial_only, include_patient_wise_only, "
        "or include_full_NLL must be True"
      )
    if len(self.CLs) > len(self.CL_colors):
      raise ValueError(
        f"Not enough colors provided for {len(self.CLs)} CLs, "
        f"got {len(self.CL_colors)} colors"
      )
    self.CL_colors = self.CL_colors[:len(self.CLs)]

    if (
      len(self.CLs) > len(self.CL_hatches)
      and self.include_full_NLL
      and (self.include_binomial_only or self.include_patient_wise_only)
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
    log_zero_epsilon: float = 1e-10,
  ):
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_max = parameter_max
    self.__endpoint_epsilon = endpoint_epsilon
    self.__log_zero_epsilon = log_zero_epsilon

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
  ):
    """
    Get the MINLP for the given time point.
    """
    return MINLPForKM(
      all_patients=self.all_patients,
      parameter_min=self.parameter_min,
      parameter_max=self.parameter_max,
      time_point=time_point,
      endpoint_epsilon=self.__endpoint_epsilon,
      log_zero_epsilon=self.__log_zero_epsilon,
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
  ) -> collections.abc.Callable[[float], float]:
    """
    Get the twoNLL function for the given time point.
    """
    if MIPGap is None:
      MIPGap = self.__default_MIPGap
    if MIPGapAbs is None:
      MIPGapAbs = self.__default_MIPGapAbs

    minlp = self.minlp_for_km(time_point=time_point)
    @InspectableCache
    def twoNLL(expected_probability: float) -> float:
      """
      The negative log-likelihood function.
      """
      result = minlp.run_MINLP(
        expected_probability=expected_probability,
        binomial_only=binomial_only,
        patient_wise_only=patient_wise_only,
        verbose=verbose,
        print_progress=print_progress,
        MIPGap=MIPGap,
        MIPGapAbs=MIPGapAbs,
      )
      if not result.success:
        return np.inf
      return result.x
    return twoNLL

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
    twoNLL: collections.abc.Callable[[float], float],
    time_point: float,
    *,
    patient_wise_only=False,
    optimize_verbose=False,
    MIPGap: float | None = None,
    MIPGapAbs: float | None = None,
    _force_minimization: bool = False,
  ) -> tuple[float, float]:
    """
    Find the expected probability that minimizes the negative log-likelihood
    for the given time point.
    """
    if patient_wise_only:
      # if patient_wise_only is True, the best probability is the
      # nominal probability, by construction
      if _force_minimization:
        #for debug purposes: actually minimize the twoNLL
        if MIPGap is None:
          MIPGap = self.__default_MIPGap
        if MIPGapAbs is None:
          MIPGapAbs = self.__default_MIPGapAbs

        # Set atol using MIPGapAbs and rtol using MIPGap
        atol_for_discrete_min = MIPGapAbs * 1.1 # Or some other factor for safety
        rtol_for_discrete_min = MIPGap * 1.1 # Use relative MIPGap for rtol

        return minimize_discrete_single_minimum(
          objective_function=twoNLL,
          possible_values=self.possible_probabilities(time_point),
          verbose=optimize_verbose,
          atol=atol_for_discrete_min,
          rtol=rtol_for_discrete_min,
        )
      expected_probability = self.nominalkm.survival_probability(time_point)
      return expected_probability, twoNLL(expected_probability)
    def vectorized_twoNLL(expected_probability: float) -> float:
      return twoNLL(float(expected_probability))
    vectorized_twoNLL = np.vectorize(vectorized_twoNLL, otypes=[float])
    result: scipy.optimize.OptimizeResult = scipy.optimize.differential_evolution(
      vectorized_twoNLL,
      bounds=np.array([[self.__endpoint_epsilon, 1 - self.__endpoint_epsilon]]),
      rng=123456,
    )
    assert isinstance(result, scipy.optimize.OptimizeResult)
    if not result.success:
      raise RuntimeError("Failed to find the best probability")
    x, = result.x
    return x, result.fun

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
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the survival probabilities for the given quantiles.
    """
    best_probabilities = []
    survival_probabilities = []
    for i, t in enumerate(times_for_plot, start=1):
      if print_progress:
        print(
          f"Calculating survival probabilities for time point {t:.2f} "
          f"({i} / {len(times_for_plot)}) at time {datetime.datetime.now()}"
        )
      survival_probabilities_time_point = []
      survival_probabilities.append(survival_probabilities_time_point)
      twoNLL = self.get_twoNLL_function(
        time_point=t,
        binomial_only=binomial_only,
        patient_wise_only=patient_wise_only,
        verbose=gurobi_verbose,
        print_progress=print_progress,
        MIPGap=MIPGap,
        MIPGapAbs=MIPGapAbs,
      )
      # Find the expected probability that minimizes the negative log-likelihood
      # for the given time point
      try:
        best_prob, twoNLL_min = self.best_probability(
          twoNLL=twoNLL,
          time_point=t,
          patient_wise_only=patient_wise_only,
          optimize_verbose=optimize_verbose,
          MIPGap=MIPGap,
          MIPGapAbs=MIPGapAbs,
        )
      except Exception as e:
        raise RuntimeError(
          f"Failed to find the best probability for time point {t}"
        ) from e
      best_probabilities.append(best_prob)

      for CL in CLs:
        if patient_wise_only and t < min(self.patient_death_times):
          # If the time point is outside the range of patient times, we cannot
          # calculate a patient-wise survival probability.
          survival_probabilities_time_point.append((1, 1))
          continue

        d2NLLcut = scipy.stats.chi2.ppf(CL, 1).item()
        def objective_function(
          expected_probability: float,
          twoNLL=twoNLL, twoNLL_min=twoNLL_min, d2NLLcut=d2NLLcut
        ) -> float:
          return twoNLL(expected_probability) - twoNLL_min - d2NLLcut
        np.testing.assert_almost_equal(
          objective_function(best_prob),
          -d2NLLcut,
        )

        if patient_wise_only:
          probs = self.possible_probabilities(time_point=t)
          i_best = int(np.searchsorted(probs, best_prob))

          # Check edge case: upper bound
          if objective_function(probs[-1]) < 0:
            upper_bound = 1
          else:
            upper = binary_search_sign_change(
              objective_function=objective_function,
              probs=probs,
              lo=i_best,
              hi=len(probs) - 1,
              verbose=optimize_verbose,
              # No explicit atol/rtol for binary_search_sign_change, it relies on exact sign change
            )
            if upper is None:
              raise RuntimeError("No upper sign change found")
            upper_bound = upper

          # Check edge case: lower bound
          if objective_function(probs[0]) < 0:
            lower_bound = 0
          else:
            lower = binary_search_sign_change(
              objective_function=objective_function,
              probs=probs,
              lo=0,
              hi=i_best,
              verbose=optimize_verbose,
              # No explicit atol/rtol for binary_search_sign_change, it relies on exact sign change
            )
            if lower is None:
              raise RuntimeError("No lower sign change found")
            lower_bound = lower

        else:
          if objective_function(self.__endpoint_epsilon) < 0:
            lower_bound = 0
          else:
            lower_bound = scipy.optimize.brentq(
              objective_function,
              self.__endpoint_epsilon,
              best_prob,
              xtol=1e-6,
            )
          if objective_function(1 - self.__endpoint_epsilon) < 0:
            upper_bound = 1
          else:
            upper_bound = scipy.optimize.brentq(
              objective_function,
              best_prob,
              1 - self.__endpoint_epsilon,
              xtol=1e-6,
            )

        survival_probabilities_time_point.append((lower_bound, upper_bound))
    return np.array(best_probabilities), np.array(survival_probabilities)

  def plot(self, config: KaplanMeierPlotConfig):
    """
    Plots the Kaplan-Meier curves based on the provided configuration.
    """
    # Use config.times_for_plot, falling back to self.times_for_plot if None
    times_for_plot = config.times_for_plot
    if times_for_plot is None:
      times_for_plot = self.times_for_plot

    fig, ax = self._prepare_figure(config)

    # Plot nominal curve and censored points
    self._plot_nominal_and_censored(ax, config, times_for_plot)

    # Calculate and plot confidence bands and best fit curve
    self._calculate_and_plot_confidence_bands(ax, config, times_for_plot)

    # Finalize plot elements (legend, labels, grid, save/show/close)
    self._finalize_plot(fig, ax, config)

    # Return results if needed, similar to original plot method
    # For now, we'll just return an empty dict as the original returned `results`
    # was not explicitly used outside the method. If you need it, we can re-add.
    return {} # Placeholder for results dictionary if needed later

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

  def _plot_nominal_and_censored(
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

    patient_censored_times = sorted(self.nominalkm.patient_censored_times)
    censored_times_probabilities = self.nominalkm.survival_probabilities(
      patient_censored_times,
    )
    ax.plot(
      patient_censored_times,
      censored_times_probabilities,
      marker='|',
      color=config.nominal_color,
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
  ):
    """
    Helper to plot confidence bands using fill_between.
    """
    for CL, color, hatch, (p_minus, p_plus) in zip(
      config.CLs,
      config.CL_colors,
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

  def _calculate_and_plot_confidence_bands(
    self,
    ax: matplotlib.axes.Axes,
    config: KaplanMeierPlotConfig,
    times_for_plot: typing.Sequence[float]
  ):
    """Calculates and plots the confidence bands and best-fit curve."""
    best_probabilities = None
    CL_probabilities = None
    # For binomial_only or patient_wise_only if full NLL is also included
    CL_probabilities_subset = None

    # Calculate and plot Full NLL
    if config.include_full_NLL:
      best_probabilities, CL_probabilities = self.survival_probabilities_likelihood(
        CLs=config.CLs,
        times_for_plot=times_for_plot,
        print_progress=config.print_progress,
        MIPGap=config.MIPGap,
        MIPGapAbs=config.MIPGapAbs,
      )
      # Plot the full NLL confidence bands
      self._plot_confidence_band_fill(
        ax, config, times_for_plot, CL_probabilities, use_hatches=False
      )

    # Calculate and plot Binomial Only
    if config.include_binomial_only:
      (
        best_probabilities_binomial, CL_probabilities_binomial
      ) = self.survival_probabilities_likelihood(
        CLs=config.CLs,
        times_for_plot=times_for_plot,
        binomial_only=True,
        print_progress=config.print_progress,
        MIPGap=config.MIPGap,
        MIPGapAbs=config.MIPGapAbs,
      )
      if config.include_full_NLL:
        CL_probabilities_subset = CL_probabilities_binomial
        self._plot_confidence_band_fill(
          ax, config, times_for_plot, CL_probabilities_subset,
          label_suffix="Binomial only", use_hatches=True
        )
      else:
        best_probabilities = best_probabilities_binomial
        CL_probabilities = CL_probabilities_binomial
        self._plot_confidence_band_fill(
          ax, config, times_for_plot, CL_probabilities,
          label_suffix="Binomial only", use_hatches=False # No hatches if it's the primary CL
        )

    # Calculate and plot Patient-Wise Only
    if config.include_patient_wise_only:
      (
        best_probabilities_patient_wise, CL_probabilities_patient_wise
      ) = self.survival_probabilities_likelihood(
        CLs=config.CLs,
        times_for_plot=times_for_plot,
        patient_wise_only=True,
        print_progress=config.print_progress,
        MIPGap=config.MIPGap,
        MIPGapAbs=config.MIPGapAbs,
      )
      if config.include_full_NLL:
        CL_probabilities_subset = CL_probabilities_patient_wise
        self._plot_confidence_band_fill(
          ax, config, times_for_plot, CL_probabilities_subset,
          label_suffix="Patient-wise only", use_hatches=True
        )
      else:
        best_probabilities = best_probabilities_patient_wise
        CL_probabilities = CL_probabilities_patient_wise
        self._plot_confidence_band_fill(
          ax, config, times_for_plot, CL_probabilities,
          label_suffix="Patient-wise only", use_hatches=False # No hatches if it's the primary CL
        )

    assert best_probabilities is not None
    # This assertion might fail if only subset is plotted without full NLL
    # assert CL_probabilities is not None

    # Plot best fit curve
    if config.include_best_fit and best_probabilities is not None:
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
      np.testing.assert_array_equal(best_x, times_for_plot)

  def _finalize_plot(
    self,
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    config: KaplanMeierPlotConfig,
  ):
    """Adds final plot elements and handles saving/showing/closing."""
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    if config.title is not None:
      ax.set_title(config.title)
    ax.legend()
    if config.show_grid:
      ax.grid()
    ax.set_ylim(0, 1.05) # Ensure y-axis is from 0 to 1.05 for survival probability

    if config.saveas is not None:
      save_path = pathlib.Path(config.saveas)
      save_path.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(save_path, bbox_inches='tight', dpi=config.dpi)
      print(f"Plot saved to {save_path}")

    if config.show:
      plt.show()

    if config.close_figure is None: # Default behavior: close if saving, don't close if showing
      if config.saveas is not None:
        plt.close(fig)
    elif config.close_figure:
      plt.close(fig)
