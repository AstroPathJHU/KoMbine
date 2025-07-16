"""
Kaplan-Meier curve with error bars calculated using the log-likelihood method.
"""

import collections.abc
import datetime
import functools
import typing

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
from .kaplan_meier_ILP import ILPForKM, KaplanMeierPatientNLL
from .utilities import InspectableCache

class KaplanMeierLikelihood(KaplanMeierBase):
  """
  Kaplan-Meier curve with error bars calculated using the log-likelihood method.
  """
  def __init__(
    self,
    *,
    all_patients: list[KaplanMeierPatientNLL],
    parameter_min: float,
    parameter_max: float,
    endpoint_epsilon: float = 1e-6,
  ):
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_max = parameter_max
    self.__endpoint_epsilon = endpoint_epsilon

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

  def ilp_for_km(
    self,
    time_point: float,
  ):
    """
    Get the ILP for the given time point.
    """
    return ILPForKM(
      all_patients=self.all_patients,
      parameter_min=self.parameter_min,
      parameter_max=self.parameter_max,
      time_point=time_point,
      endpoint_epsilon=self.__endpoint_epsilon,
    )

  def ilps_for_km(
    self,
    times_for_plot: typing.Sequence[float] | None,
  ):
    """
    Get the ILPs for the given time points.
    """
    if times_for_plot is None:
      times_for_plot = self.times_for_plot
    return [
      self.ilp_for_km(time_point=t)
      for t in times_for_plot
    ]

  def get_twoNLL_function( # pylint: disable=too-many-arguments
    self,
    time_point: float,
    *,
    binomial_only=False,
    patient_wise_only=False,
    verbose=False,
    print_progress=False,
    MIPGap=None,
    fallback_MIPGap=None,
  ) -> collections.abc.Callable[[float], float]:
    """
    Get the twoNLL function for the given time point.
    """
    ilp = self.ilp_for_km(time_point=time_point)
    @InspectableCache
    def twoNLL(expected_probability: float) -> float:
      """
      The negative log-likelihood function.
      """
      result = ilp.run_ILP(
        expected_probability=expected_probability,
        binomial_only=binomial_only,
        patient_wise_only=patient_wise_only,
        verbose=verbose,
        print_progress=print_progress,
        MIPGap=MIPGap,
        fallback_MIPGap=fallback_MIPGap,
      )
      if not result.success:
        return np.inf
      return result.x
    return twoNLL

  def calculate_possible_probabilities(self, time_point: float) -> np.ndarray:
    """
    Get the possible probabilities for the given patients.
    """
    return np.array(sorted(self.ilp_for_km(time_point).possible_probabilities))

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
  ) -> tuple[float, float]:
    """
    Find the expected probability that minimizes the negative log-likelihood
    for the given time point.
    """
    if patient_wise_only:
      return minimize_discrete_single_minimum(
        objective_function=twoNLL,
        possible_values=self.possible_probabilities(time_point),
        verbose=optimize_verbose,
      )
    result = scipy.optimize.minimize_scalar(
      twoNLL,
      bounds=(self.__endpoint_epsilon, 1 - self.__endpoint_epsilon),
      method='bounded',
    )
    assert isinstance(result, scipy.optimize.OptimizeResult)
    if not result.success:
      raise RuntimeError("Failed to find the best probability")
    return result.x, result.fun

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
    fallback_MIPGap=None,
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
        fallback_MIPGap=fallback_MIPGap,
      )
      # Find the expected probability that minimizes the negative log-likelihood
      # for the given time point
      try:
        best_prob, twoNLL_min = self.best_probability(
          twoNLL=twoNLL,
          time_point=t,
          patient_wise_only=patient_wise_only,
          optimize_verbose=optimize_verbose,
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

  def plot( # pylint: disable=too-many-arguments, too-many-branches, too-many-statements
    self,
    *,
    times_for_plot: typing.Sequence[float] | None = None,
    include_binomial_only=False,
    include_patient_wise_only=False,
    include_full_NLL=True,
    include_best_fit=True,
    nominal_label='Nominal',
    nominal_color='blue',
    CLs=None,
    CL_colors=None,
    CL_hatches=None,
    create_figure=True,
    close_figure=None,
    show=False,
    saveas=None,
    print_progress=False,
    MIPGap=None,
    fallback_MIPGap=None,
    include_median_survival=False,
  ): #pylint: disable=too-many-locals
    """
    Plots the Kaplan-Meier curves.
    """
    if include_binomial_only and include_patient_wise_only:
      raise ValueError("include_binomial_only and include_patient_wise_only cannot both be True")
    if not (include_binomial_only or include_patient_wise_only or include_full_NLL):
      raise ValueError(
        "At least one of include_binomial_only, include_patient_wise_only, "
        "or include_full_NLL must be True"
      )
    if times_for_plot is None:
      times_for_plot = self.times_for_plot
    results = {}
    if create_figure:
      plt.figure()
    nominal_x, nominal_y = self.nominalkm.points_for_plot(times_for_plot=times_for_plot)
    label = nominal_label
    if include_median_survival:
      label += " (MST={:.1f})".format(  #pylint: disable=consider-using-f-string
        self.median_survival_time(
          times_for_plot=nominal_x,
          survival_probabilities=nominal_y,
        )
      ).replace("inf", r"$\infty$")
    plt.plot(
      nominal_x,
      nominal_y,
      label=label,
      color=nominal_color,
      linestyle='--'
    )
    results["x"] = nominal_x
    results["nominal"] = nominal_y

    if CLs is None:
      CLs = [0.68, 0.95]

    if CL_colors is None:
      CL_colors = ['dodgerblue', 'skyblue', 'lightblue', 'lightcyan']
    if len(CLs) > len(CL_colors):
      raise ValueError(
        f"Not enough colors provided for {len(CLs)} CLs, "
        f"got {len(CL_colors)} colors"
      )
    CL_colors = CL_colors[:len(CLs)]

    if CL_hatches is None:
      CL_hatches = ['//', '\\\\', 'xx', '++']
    if (
      len(CLs) > len(CL_hatches)
      and include_full_NLL
      and (include_binomial_only or include_patient_wise_only)
    ):
      raise ValueError(
        f"Not enough hatches provided for {len(CLs)} CLs, "
        f"got {len(CL_hatches)} hatches"
      )
    CL_hatches = CL_hatches[:len(CLs)]

    CL_probabilities_subset = None
    best_probabilities = None
    CL_probabilities = None
    if include_full_NLL:
      best_probabilities, CL_probabilities = self.survival_probabilities_likelihood(
        CLs=CLs,
        times_for_plot=times_for_plot,
        print_progress=print_progress,
        MIPGap=MIPGap,
        fallback_MIPGap=fallback_MIPGap,
      )
    if include_binomial_only:
      (
        best_probabilities_binomial, CL_probabilities_binomial
      ) = self.survival_probabilities_likelihood(
        CLs=CLs,
        times_for_plot=times_for_plot,
        binomial_only=True,
        print_progress=print_progress,
        MIPGap=MIPGap,
        fallback_MIPGap=fallback_MIPGap,
      )
      if include_full_NLL:
        CL_probabilities_subset = CL_probabilities_binomial
      else:
        best_probabilities = best_probabilities_binomial
        CL_probabilities = CL_probabilities_binomial
    if include_patient_wise_only:
      (
        best_probabilities_patient_wise, CL_probabilities_patient_wise
      ) = self.survival_probabilities_likelihood(
        CLs=CLs,
        times_for_plot=times_for_plot,
        patient_wise_only=True,
        print_progress=print_progress,
        MIPGap=MIPGap,
        fallback_MIPGap=fallback_MIPGap,
      )
      if include_full_NLL:
        CL_probabilities_subset = CL_probabilities_patient_wise
      else:
        best_probabilities = best_probabilities_patient_wise
        CL_probabilities = CL_probabilities_patient_wise

    assert best_probabilities is not None
    assert CL_probabilities is not None

    best_x, best_y = self.get_points_for_plot(times_for_plot, best_probabilities)
    if include_best_fit:
      label = "Best Probability"
      if include_median_survival:
        label += " (MST={:.1f})".format(  #pylint: disable=consider-using-f-string
          self.nominalkm.median_survival_time(
            times_for_plot=best_x,
            survival_probabilities=best_y,
          )
        )
      plt.plot(
        best_x,
        best_y,
        label=label,
        color='red',
        linestyle='--'
      )
      np.testing.assert_array_equal(best_x, nominal_x)
      results["best_fit"] = best_y

    for CL, color, (p_minus, p_plus) in zip(
      CLs,
      CL_colors,
      CL_probabilities.transpose(1, 2, 0),
      strict=True,
    ):
      x_minus, y_minus = self.get_points_for_plot(times_for_plot, p_minus)
      x_plus, y_plus = self.get_points_for_plot(times_for_plot, p_plus)
      np.testing.assert_array_equal(x_minus, x_plus)
      np.testing.assert_array_equal(x_minus, best_x)
      results[f'CL_{CL}'] = (y_minus, y_plus)

      if CL > 0.9999:
        label = f'{CL:.6%} CL'
      elif CL > 0.99:
        label = f'{CL:.2%} CL'
      else:
        label = f'{CL:.0%} CL'
      if include_median_survival:
        label += " (MST$\\in$({:.1f}, {:.1f}))".format(  #pylint: disable=consider-using-f-string
          self.nominalkm.median_survival_time(
            times_for_plot=x_minus,
            survival_probabilities=y_minus,
          ),
          self.nominalkm.median_survival_time(
            times_for_plot=x_plus,
            survival_probabilities=y_plus,
          ),
        ).replace("inf", r"$\infty$")
      plt.fill_between(
        x_minus,
        y_minus,
        y_plus,
        color=color,
        alpha=0.5,
        label=label,
      )

    if CL_probabilities_subset is not None:
      subset_label = "Binomial only" if include_binomial_only else "Patient-wise only"
      for CL, color, hatch, (p_minus_subset, p_plus_subset) in zip(
        CLs,
        CL_colors,
        CL_hatches,
        CL_probabilities_subset.transpose(1, 2, 0),
        strict=True,
      ):
        x_minus_subset, y_minus_subset = self.get_points_for_plot(times_for_plot, p_minus_subset)
        x_plus_subset, y_plus_subset = self.get_points_for_plot(times_for_plot, p_plus_subset)
        np.testing.assert_array_equal(x_minus_subset, x_plus_subset)
        np.testing.assert_array_equal(x_minus_subset, best_x)
        results[f'CL_{CL}_subset'] = (y_minus_subset, y_plus_subset)

        if CL > 0.9999:
          label = f'{CL:.6%} CL ({subset_label})'
        elif CL > 0.99:
          label = f'{CL:.2%} CL ({subset_label})'
        else:
          label = f'{CL:.0%} CL ({subset_label})'
        if include_median_survival:
          label += r" (MST$\elem$({:.1f}, {:.1f}))".format(  #pylint: disable=consider-using-f-string
            self.nominalkm.median_survival_time(
              times_for_plot=x_minus_subset,
              survival_probabilities=y_minus_subset,
            ),
            self.nominalkm.median_survival_time(
              times_for_plot=x_plus_subset,
              survival_probabilities=y_plus_subset,
            ),
          )
        plt.fill_between(
          x_minus_subset,
          y_minus_subset,
          y_plus_subset,
          edgecolor=color,
          facecolor='none',
          hatch=hatch,
          alpha=0.5,
          label=label,
        )

    if create_figure:
      plt.xlabel("Time")
      plt.ylabel("Survival Probability")
      plt.legend()
      plt.title("Kaplan-Meier Curves")
      plt.grid()
      if saveas is not None:
        plt.savefig(saveas)
      if show:
        plt.show()
      if close_figure:
        plt.close()

    return results
