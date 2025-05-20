"""
Kaplan-Meier curve with error bars calculated using the log-likelihood method.
"""

import collections.abc

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.stats

from .kaplan_meier import KaplanMeierPatientBase

class KaplanMeierPatientNLL(KaplanMeierPatientBase):
  """
  A patient with a time and a parameter.
  The parameter is a log-likelihood function.
  """
  def __init__(
    self,
    time: float,
    parameter_nll: collections.abc.Callable[[float], float],
    nominal_parameter: float,
  ):
    super().__init__(time=time, parameter=parameter_nll)
    self.__observed_parameter = nominal_parameter

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
      nominal_parameter=count,
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

  def run_ILP(self, expected_probability: float): # pylint: disable=too-many-locals
    """
    Run the ILP for the given time point.
    """
    n_patients = len(self.__all_patients)
    patient_times = np.array([p.time for p in self.__all_patients])
    patient_alive = patient_times > self.__time_point
    parameters = np.array([p.parameter for p in self.__all_patients])

    parameter_in_range = (
      (parameters > self.__parameter_min)
      & (parameters < self.__parameter_max)
    )
    sgn_nll_penalty_for_patient_in_range = 2 * parameter_in_range - 1
    abs_nll_penalty_for_patient_in_range = np.array([
      p.parameter(p.observed_parameter)
       - min(
         p.parameter(self.__parameter_min),
         p.parameter(self.__parameter_max),
       )
      for p in self.__all_patients
    ])

    nll_penalty_for_patient_in_range = (
      sgn_nll_penalty_for_patient_in_range
      * abs_nll_penalty_for_patient_in_range
    )

    #Gurobi implementation from ChatGPT

    binomial_penalty_table = {}
    for n_total in range(n_patients + 1):
      for n_alive in range(n_total + 1):
        penalty = -scipy.stats.binom.logpmf(n_alive, n_total, expected_probability)
        binomial_penalty_table[(n_alive, n_total)] = penalty.item()

    # Create arrays for Gurobi PWL
    # Flatten by row-major order: (n_total, n_alive) -> penalty
    alive_vals = []
    penalty_vals = []

    for n_total in range(n_patients + 1):
      for n_alive in range(n_total + 1):
        alive_vals.append((n_total, n_alive))
        penalty_vals.append(binomial_penalty_table[(n_alive, n_total)])

    # We'll do PWL on just `n_alive` for fixed `n_total`, so build separate slices during modeling

    # ---------------------------
    # Gurobi model
    # ---------------------------

    model = gp.Model("Kaplan-Meier ILP")

    # Binary decision variables: x[i] = 1 if patient i is within the parameter range
    x = model.addVars(n_patients, vtype=GRB.BINARY, name="x")

    # Integer vars to count totals
    n_total = model.addVar(vtype=GRB.INTEGER, name="n_total")
    n_alive = model.addVar(vtype=GRB.INTEGER, name="n_alive")

    # Piecewise-linear penalty var
    binom_penalty = model.addVar(lb=0.0, name="binom_penalty")

    # Constraint: link n_total to selected patients
    model.addConstr(n_total == gp.quicksum(x[i] for i in range(n_patients)))

    # Constraint: link n_alive to selected & alive patients
    model.addConstr(
        n_alive == gp.quicksum(x[i] for i in range(n_patients) if patient_alive[i])
    )

    # Piecewise approximation: for each possible n_total, define a separate PWL segment
    indicator_vars = []
    for n_total_val in range(n_patients + 1):
      # Build x-y curve for this n_total
      x_vals = list(range(n_total_val + 1))
      y_vals = [
        binomial_penalty_table[(n_alive_val, n_total_val)]
        for n_alive_val in x_vals
      ]

      # Binary indicator if n_total equals this value
      indicator = model.addVar(vtype=GRB.BINARY, name=f"ind_ntotal_{n_total_val}")
      indicator_vars.append(indicator)
      model.addConstr((indicator == 1) >> (n_total == n_total_val))

      # Temp var for penalty at this n_total
      binom_piece = model.addVar(lb=0.0, name=f"binom_piece_{n_total_val}")
      model.addGenConstrPWL(n_alive, binom_piece, x_vals, y_vals)

      # Only include penalty if indicator is on
      model.addConstr(binom_piece <= binom_penalty + (1 - indicator) * 1e6)
      model.addConstr(binom_piece >= binom_penalty - (1 - indicator) * 1e6)

    # Only one n_total active
    model.addConstr(gp.quicksum(indicator_vars) == 1)

    # Patient-wise penalties
    patient_penalty = gp.quicksum(
      nll_penalty_for_patient_in_range[i] * x[i] for i in range(n_patients)
    )

    # Objective: minimize total penalty
    model.setObjective(
      binom_penalty + patient_penalty,
      GRB.MINIMIZE,
    )

    return model.optimize()
