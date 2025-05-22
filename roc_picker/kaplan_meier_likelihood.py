"""
Kaplan-Meier curve with error bars calculated using the log-likelihood method.
"""

import collections.abc
import functools

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats

from .kaplan_meier import (
  KaplanMeierBase,
  KaplanMeierInstance,
  KaplanMeierPatient,
  KaplanMeierPatientBase
)

class KaplanMeierPatientNLL(KaplanMeierPatientBase):
  """
  A patient with a time and a parameter.
  The parameter is a log-likelihood function.
  """
  def __init__(
    self,
    time: float,
    parameter_nll: collections.abc.Callable[[float], float],
    observed_parameter: float,
  ):
    super().__init__(time=time, parameter=parameter_nll)
    self.__observed_parameter = observed_parameter

  @property
  def parameter(self) -> collections.abc.Callable[[float], float]:
    """
    The parameter is a log-likelihood function.
    """
    return super().parameter

  @property
  def observed_parameter(self) -> float:
    """
    The observed value of the parameter.
    """
    return self.__observed_parameter

  @classmethod
  def from_count(
    cls,
    time: float,
    count: int,
  ):
    """
    Create a KaplanMeierPatientNLL from a count.
    The gives the negative log-likelihood to observe the count
    given the parameter, which is the mean of the Poisson distribution.
    """
    def parameter_nll(x: float) -> float:
      """
      The parameter is a log-likelihood function.
      """
      return -scipy.stats.poisson.logpmf(count, x).item()
    return cls(
      time=time,
      parameter_nll=parameter_nll,
      observed_parameter=count,
    )

  @property
  def nominal(self) -> KaplanMeierPatient:
    """
    The nominal value of the parameter.
    """
    return KaplanMeierPatient(
      time=self.time,
      parameter=self.observed_parameter,
    )

class ILPForKM:
  """
  Integer Linear Programming for a point on the Kaplan-Meier curve.
  """
  def __init__(
    self,
    all_patients: list[KaplanMeierPatientNLL],
    parameter_min: float,
    parameter_max: float,
    time_point: float,
  ):
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_max = parameter_max
    self.__time_point = time_point
    if not np.isfinite(self.__parameter_min and self.__parameter_min != -np.inf):
      raise ValueError("parameter_min must be finite or -inf")
    if not np.isfinite(self.__parameter_max and self.__parameter_max != np.inf):
      raise ValueError("parameter_max must be finite or inf")

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
  def time_point(self) -> float:
    """
    The time point for the Kaplan-Meier curve.
    """
    return self.__time_point

  def run_ILP(self, expected_probability: float, verbose=False, binomial_only=False): # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    """
    Run the ILP for the given time point.
    """
    n_patients = len(self.all_patients)
    patient_times = np.array([p.time for p in self.all_patients])
    patient_alive = patient_times > self.time_point
    observed_parameters = np.array([p.observed_parameter for p in self.all_patients])

    parameter_in_range = (
      (observed_parameters > self.parameter_min)
      & (observed_parameters < self.parameter_max)
    )
    sgn_nll_penalty_for_patient_in_range = 2 * parameter_in_range - 1
    observed_nll = np.array([
      p.parameter(p.observed_parameter)
      for p in self.all_patients
    ])
    if np.isfinite(self.parameter_min):
      parameter_min_nll = np.array([
        p.parameter(self.parameter_min)
        for p in self.all_patients
      ])
    else:
      parameter_min_nll = np.full(n_patients, np.inf)
    if np.isfinite(self.parameter_max):
      parameter_max_nll = np.array([
        p.parameter(self.parameter_max)
        for p in self.all_patients
      ])
    else:
      parameter_max_nll = np.full(n_patients, np.inf)

    range_boundary_nll = np.min(
      np.array([parameter_min_nll, parameter_max_nll]),
      axis=0
    )
    if np.isfinite(range_boundary_nll).any() and not binomial_only:
      abs_nll_penalty_for_patient_in_range = observed_nll - range_boundary_nll
    else:
      abs_nll_penalty_for_patient_in_range = np.full(n_patients, 5*scipy.stats.binom.logpmf(0, n_patients, 0.99999).item())

    nll_penalty_for_patient_in_range = (
      sgn_nll_penalty_for_patient_in_range
      * abs_nll_penalty_for_patient_in_range
    )

    binomial_penalty_table = {}
    for n_total in range(n_patients + 1):
      for n_alive in range(n_total + 1):
        penalty = -scipy.stats.binom.logpmf(n_alive, n_total, expected_probability)
        binomial_penalty_table[(n_alive, n_total)] = penalty.item()

    # ---------------------------
    # Gurobi model
    # ---------------------------

    model = gp.Model("Kaplan-Meier ILP")

    # Binary decision variables: x[i] = 1 if patient i is within the parameter range
    x = model.addVars(n_patients, vtype=GRB.BINARY, name="x")

    # Integer vars to count totals
    n_total = model.addVar(vtype=GRB.INTEGER, name="n_total")
    n_alive = model.addVar(vtype=GRB.INTEGER, name="n_alive")

    # Constraints to link to totals
    model.addConstr(n_total == gp.quicksum(x[i] for i in range(n_patients)))
    model.addConstr(n_alive == gp.quicksum(x[i] for i in range(n_patients) if patient_alive[i]))

    # ---------------------------
    # Indicator grid for binomial penalty
    # ---------------------------

    # Create binary indicators for each valid (n_alive, n_total)
    indicator_vars = {}
    for n_total_val in range(n_patients + 1):
      for n_alive_val in range(n_total_val + 1):
        ind = model.addVar(vtype=GRB.BINARY, name=f"ind_{n_alive_val}_{n_total_val}")
        indicator_vars[(n_alive_val, n_total_val)] = ind
        # Add indicator constraint: if active, enforce n_alive and n_total match
        model.addGenConstrIndicator(ind, True, n_alive - n_alive_val, GRB.EQUAL, 0)
        model.addGenConstrIndicator(ind, True, n_total - n_total_val, GRB.EQUAL, 0)

    # Only one pair can be active
    model.addConstr(gp.quicksum(indicator_vars.values()) == 1)

    # Binomial penalty
    binom_penalty = gp.quicksum(
      binomial_penalty_table[(na, nt)] * ind
      for (na, nt), ind in indicator_vars.items()
    )

    # Patient-wise penalties
    patient_penalty = gp.quicksum(
      nll_penalty_for_patient_in_range[i] * x[i] for i in range(n_patients)
    )

    # Objective: minimize total penalty
    model.setObjective(
      2 * (binom_penalty + patient_penalty),
      GRB.MINIMIZE,
    )

    model.optimize()

    selected = [i for i in range(n_patients) if x[i].X > 0.5]
    binomial_penalty_val = binomial_penalty_table[(n_alive.X, n_total.X)]
    patient_penalty_val = sum(
      nll_penalty_for_patient_in_range[i] * x[i].X for i in range(n_patients)
    )
    if verbose:
      if model.status == GRB.OPTIMAL:
        print("Selected patients:", selected)
        print("n_total:          ", int(n_total.X))
        print("n_alive:          ", int(n_alive.X))
        print("Binomial penalty: ", binomial_penalty_val)
        print("Patient penalty:  ", patient_penalty_val)
        print("Total penalty:    ", model.ObjVal)
      else:
        print("No optimal solution found.")

    return scipy.optimize.OptimizeResult(
      x=model.ObjVal,
      success=model.status == GRB.OPTIMAL,
      n_total=int(n_total.X),
      n_alive=int(n_alive.X),
      binomial_penalty=binomial_penalty_val,
      patient_penalty=patient_penalty_val,
      selected=selected,
      model=model,
    )

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
  def patient_times(self) -> frozenset:
    """
    The set of all patient times.
    """
    return frozenset(p.time for p in self.all_patients)

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
    )

  def ilps_for_km(
    self,
    times_for_plot: np.ndarray | None,
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

  def get_twoNLL_function(self, time_point: float, binomial_only=False):
    """
    Get the twoNLL function for the given time point.
    """
    ilp = self.ilp_for_km(time_point=time_point)
    def twoNLL(expected_probability: float) -> float:
      """
      The negative log-likelihood function.
      """
      result = ilp.run_ILP(expected_probability=expected_probability, binomial_only=binomial_only)
      if not result.success:
        return np.inf
      return result.x
    return twoNLL

  def best_probability(
    self,
    time_point: float,
    binomial_only=False,
  ) -> tuple[float, float]:
    """
    Find the probability that minimizes the negative log-likelihood
    for the given time point.
    """
    # Find the expected probability that minimizes the negative log-likelihood
    # for the given time point
    twoNLL = self.get_twoNLL_function(time_point=time_point, binomial_only=binomial_only)
    result = scipy.optimize.minimize_scalar(
      twoNLL,
      bounds=(self.__endpoint_epsilon, 1 - self.__endpoint_epsilon),
      method='bounded',
    )
    if not result.success:
      raise RuntimeError("Failed to find the best probability")
    return result.x, result.fun

  def survival_probabilities_likelihood( # pylint: disable=too-many-locals
    self,
    CLs: list[float],
    times_for_plot: np.ndarray,
    binomial_only=False,
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the survival probabilities for the given quantiles.
    """
    best_probabilities = []
    survival_probabilities = []
    for t in times_for_plot:
      survival_probabilities_time_point = []
      survival_probabilities.append(survival_probabilities_time_point)
      twoNLL = self.get_twoNLL_function(time_point=t, binomial_only=binomial_only)
      # Find the expected probability that minimizes the negative log-likelihood
      # for the given time point
      best_prob, twoNLL_min = self.best_probability(time_point=t, binomial_only=binomial_only)
      best_probabilities.append(best_prob)

      for CL in CLs:
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

  def plot(
    self,
    times_for_plot=None,
    include_binomial_only=False,
    show=False,
    saveas=None,
  ): #pylint: disable=too-many-locals
    """
    Plots the Kaplan-Meier curves.
    """
    if times_for_plot is None:
      times_for_plot = self.times_for_plot
    plt.figure()
    nominal_x, nominal_y = self.nominalkm.points_for_plot(times_for_plot=times_for_plot)
    plt.plot(
      nominal_x,
      nominal_y,
      label="Nominal",
      color='black',
      linestyle='--'
    )

    CLs = [0.68, 0.95]
    best_probabilities, survival_probabilities = self.survival_probabilities_likelihood(
      CLs=CLs,
      times_for_plot=self.times_for_plot,
    )
    if include_binomial_only:
      _, survival_probabilities_binomial = self.survival_probabilities_likelihood(
        CLs=CLs,
        times_for_plot=self.times_for_plot,
        binomial_only=True,
      )
    else:
      survival_probabilities_binomial = None

    best_x, best_y = self.get_points_for_plot(times_for_plot, best_probabilities)
    plt.plot(
      best_x,
      best_y,
      label="Best Probability",
      color='red',
      linestyle='--'
    )

    print(survival_probabilities.shape)
    (p_m68, p_p68), (p_m95, p_p95) = survival_probabilities.transpose(1, 2, 0)
    x_m95, y_m95 = self.get_points_for_plot(times_for_plot, p_m95)
    x_m68, y_m68 = self.get_points_for_plot(times_for_plot, p_m68)
    x_p68, y_p68 = self.get_points_for_plot(times_for_plot, p_p68)
    x_p95, y_p95 = self.get_points_for_plot(times_for_plot, p_p95)

    np.testing.assert_array_equal(x_m95, x_p95)
    np.testing.assert_array_equal(x_m68, x_p68)

    plt.fill_between(
      x_m68,
      y_m68,
      y_p68,
      color='dodgerblue',
      alpha=0.5,
      label='68% CL',
    )
    plt.fill_between(
      x_m95,
      y_m95,
      y_p95,
      color='skyblue',
      alpha=0.5,
      label='95% CL',
    )

    if survival_probabilities_binomial is not None:
      (p_m68_binomial, p_p68_binomial), (p_m95_binomial, p_p95_binomial) = survival_probabilities_binomial.transpose(1, 2, 0)
      x_m95_binomial, y_m95_binomial = self.get_points_for_plot(times_for_plot, p_m95_binomial)
      x_m68_binomial, y_m68_binomial = self.get_points_for_plot(times_for_plot, p_m68_binomial)
      x_p68_binomial, y_p68_binomial = self.get_points_for_plot(times_for_plot, p_p68_binomial)
      x_p95_binomial, y_p95_binomial = self.get_points_for_plot(times_for_plot, p_p95_binomial)

      plt.fill_between(
        x_m68_binomial,
        y_m68_binomial,
        y_p68_binomial,
        edgecolor='dodgerblue',
        facecolor='none',
        hatch='\\\\',
        alpha=0.5,
        label='68% CL (Binomial only)',
      )
      plt.fill_between(
        x_m95_binomial,
        y_m95_binomial,
        y_p95_binomial,
        edgecolor='skyblue',
        facecolor='none',
        hatch='//',
        alpha=0.5,
        label='95% CL (Binomial only)',
      )

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.title("Kaplan-Meier Curves")
    plt.grid()
    if saveas is not None:
      plt.savefig(saveas)
    if show:
      plt.show()
    plt.close()
