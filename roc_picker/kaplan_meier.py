"""
Kaplan-Meier curves with systematic uncertainties, using the Monte Carlo method.
"""

import abc
import functools
import numbers

import matplotlib.pyplot as plt
import numpy as np
import scipy.special

from .systematics_mc import DistributionBase, DummyDistribution

class KaplanMeierPatientBase(abc.ABC):
  """
  Base class for Kaplan-Meier patients.
  It contains the survival time and the parameter used to group the patients.
  """
  def __init__(self, time: float, censored: bool, parameter):
    self.__time = time
    self.__censored = censored
    self.__parameter = parameter
  @property
  def time(self):
    """
    Returns the survival time of the patient.
    """
    return self.__time
  @property
  def censored(self) -> bool:
    """
    Returns True if the patient is censored, False otherwise.
    """
    return self.__censored
  @property
  def parameter(self):
    """
    Returns the parameter used to group the patients.
    """
    return self.__parameter

class KaplanMeierPatient(KaplanMeierPatientBase):
  """
  Class to represent a patient with their survival time and parameter.
  """
  def __init__(self, time: float, censored: bool, parameter: float):
    super().__init__(time=time, censored=censored, parameter=parameter)
    if not isinstance(parameter, (numbers.Number)):
      raise TypeError("Parameter must be a number")

  @property
  def parameter(self) -> float:
    """
    Returns the parameter used to group the patients.
    """
    return super().parameter



class KaplanMeierPatientDistribution(KaplanMeierPatientBase):
  """
  Class to represent a patient with their survival time and parameter,
  but with a probability distribution for the parameter.
  """
  def __init__(self, time: float, censored: bool, parameter : DistributionBase | float):
    if isinstance(parameter, (int, float)):
      parameter = DummyDistribution(parameter)
    if not isinstance(parameter, DistributionBase):
      raise TypeError("Parameter must be a DistributionBase or a number")
    super().__init__(time=time, censored=censored, parameter=parameter)

  @property
  def parameter(self) -> DistributionBase:
    """
    Returns the parameter used to group the patients.
    """
    return super().parameter

  @property
  def nominal(self):
    """
    Returns the nominal patient.
    """
    return KaplanMeierPatient(
      time=self.time,
      censored=self.censored,
      parameter=self.parameter.nominal
    )

  def rvs(self, size, random_state):
    """
    Returns a random sample of patients using the probability distribution.
    """
    return np.vectorize(KaplanMeierPatient)(
      self.time,
      self.parameter.rvs(size=size, random_state=random_state)
    )

class KaplanMeierBase(abc.ABC):
  """
  Base class for Kaplan-Meier curves with some utility functions.
  """
  @property
  @abc.abstractmethod
  def patient_times(self) -> frozenset[float]:
    """
    Returns the survival times of the patients.
    """
  @functools.cached_property
  def times_for_plot(self):
    """
    Returns the survival times for the Kaplan-Meier curve.
    The times are the unique survival times of the patients,
    plus a point at 0 and a point beyond the last time.
    """
    times_for_plot = sorted(self.patient_times)
    times_for_plot = np.array([0] + times_for_plot + [times_for_plot[-1] * 1.1])
    return times_for_plot

  @staticmethod
  def get_points_for_plot(times_for_plot, survival_probabilities):
    """
    Return (x, y) points for the Kaplan-Meier curve based on the
    survival probabilities at each time.
    Each time enters twice in the plot in order to have a step function.
    """
    x = [times_for_plot[0]]
    y = [survival_probabilities[0]]
    for prevprob, time, prob in zip(
      survival_probabilities[:-1],
      times_for_plot[1:],
      survival_probabilities[1:],
      strict=True
    ):
      x.append(time)
      y.append(prevprob)
      x.append(time)
      y.append(prob)
    return np.array(x), np.array(y)

class KaplanMeierInstance(KaplanMeierBase):
  """
  Class to represent a Kaplan-Meier curve.
  It contains a list of patients with their survival times and parameters.
  The patients are filtered based on a parameter range.

  Parameters
  ----------
  all_patients : list of KMPatient
    List of all patients with their survival times and parameters.
  """
  def __init__(
    self,
    all_patients: list[KaplanMeierPatient],
    parameter_min: float = -np.inf,
    parameter_max: float = np.inf
  ):
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_max = parameter_max


  @property
  def all_patients(self):
    """
    Returns all the patients before filtering.
    """
    return self.__all_patients
  @property
  def patients(self):
    """
    Returns the patients who enter the Kaplan-Meier curve.
    The patients are filtered based on the parameter range.
    """
    return [
      p for p in self.all_patients
      if self.__parameter_min <= p.parameter < self.__parameter_max
    ]

  @property
  def patient_times(self):
    """
    Returns the survival times of the patients.
    """
    patients = self.patients
    patient_times = frozenset({p.time for p in patients})
    return patient_times

  def survival_probabilities(self, times_for_plot=None):
    """
    Returns the points for the Kaplan-Meier curve.
    The points are the survival times and the survival probabilities.
    """
    patients = self.patients
    patient_times = np.array([p.time for p in patients])
    patient_censored = np.array([p.censored for p in patients])
    if times_for_plot is None:
      times_for_plot = self.times_for_plot

    survival_probabilities = np.zeros(len(times_for_plot))
    for i, t in enumerate(times_for_plot):
      n_patients = np.count_nonzero(patient_times > t | ~patient_censored)
      still_alive = np.count_nonzero(patient_times > t)
      survival_probabilities[i] = still_alive / n_patients

    return survival_probabilities

  def points_for_plot(self, times_for_plot=None):
    """
    Returns the points for the Kaplan-Meier curve.
    The points are the survival times and the survival probabilities.
    """
    if times_for_plot is None:
      times_for_plot = self.times_for_plot
    survival_probabilities = self.survival_probabilities(times_for_plot)
    return self.get_points_for_plot(times_for_plot, survival_probabilities)

class KaplanMeierPlot(KaplanMeierBase):
  """
  Class to represent a set of Kaplan-Meier curves.
  It contains a list of patients with their survival times and parameters.
  The patients are filtered based on a parameter range, and each patient
  enters a different Kaplan-Meier curve.

  Parameters
  ----------
  all_patients : list of KMPatient
    List of all patients with their survival times and parameters.
  thresholds : list of float
    List of thresholds to filter the patients.
  """
  def __init__(
    self,
    all_patients: list[KaplanMeierPatient],
    thresholds: list[float],
  ):
    self.__all_patients = all_patients
    self.__thresholds = [-np.inf] + sorted(thresholds) + [np.inf]
    self.__curves = []
    for i in range(len(self.__thresholds) - 1):
      self.__curves.append(
        KaplanMeierInstance(
          all_patients,
          self.__thresholds[i],
          self.__thresholds[i + 1],
        )
      )

  @property
  def all_patients(self):
    """
    Returns the patients with their survival times and parameters.
    """
    return self.__all_patients

  def plot(self):
    """
    Plots the Kaplan-Meier curves.
    """
    plt.figure()
    for i, curve in enumerate(self.__curves):
      x, y = curve.points_for_plot(times_for_plot=self.times_for_plot)
      print(x, y)
      plt.plot(
        x,
        y,
        #where='post',
        label=f"Curve {i + 1}: {self.__thresholds[i]} <= parameter < {self.__thresholds[i + 1]}"
      )

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan-Meier Curves")
    plt.legend()
    plt.grid()
    plt.show()

  @property
  def patient_times(self):
    """
    Returns the survival times of the patients.
    """
    return frozenset.union(
      *[kmi.patient_times for kmi in self.__curves]
    )

class KaplanMeierCollection(KaplanMeierBase):
  """
  A Monte-Carlo-generated collection of Kaplan-Meier curves.

  Parameters
  ----------
  kminstances : list of KaplanMeierInstance
    The generated Kaplan-Meier curves.
  nominalkm : KaplanMeierInstance
    The nominal Kaplan-Meier curve.
  """
  def __init__(
    self,
    kminstances: list[KaplanMeierInstance],
    nominalkm: KaplanMeierInstance,
  ):
    self.__kminstances = kminstances
    self.__nominalkm = nominalkm

  @property
  def kminstances(self):
    """
    Returns the generated Kaplan-Meier curves.
    """
    return self.__kminstances
  @property
  def nominalkm(self):
    """
    Returns the nominal Kaplan-Meier curve.
    """
    return self.__nominalkm

  @property
  def patient_times(self):
    """
    Returns the survival times of the patients.
    """
    return frozenset.union(
      *[kmi.patient_times for kmi in self.kminstances]
    )

  def survival_probabilities_quantiles(self, quantiles: list[float], times_for_plot=None):
    """
    Returns the survival probabilities for the Kaplan-Meier curves.
    The probabilities are the quantiles of the survival probabilities.
    """
    if times_for_plot is None:
      times_for_plot = self.times_for_plot
    survival_probabilities = np.zeros((len(self.kminstances), len(times_for_plot)))
    for i, kmi in enumerate(self.kminstances):
      try:
        survival_probabilities[i] = kmi.survival_probabilities(times_for_plot)
      except ZeroDivisionError:
        survival_probabilities[i] = np.nan

    return np.nanquantile(survival_probabilities, quantiles, axis=0)

  # pylint: disable=similarities
  def plot(self, times_for_plot=None, show=False, saveas=None): #pylint: disable=too-many-locals
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

    sigmas = [-2, -1, 0, 1, 2]
    quantiles = [(1 + scipy.special.erf(nsigma/np.sqrt(2))) / 2 for nsigma in sigmas]
    p_m95, p_m68, _, p_p68, p_p95 = self.survival_probabilities_quantiles(
      quantiles=quantiles,
      times_for_plot=times_for_plot,
    )

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
      label='68% CL'
    )
    plt.fill_between(
      x_m95,
      y_m95,
      y_p95,
      color='skyblue',
      alpha=0.5,
      label='95% CL'
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

class KaplanMeierDistributions(KaplanMeierBase):
  """
  Class to represent a set of Kaplan-Meier curves.
  It contains a list of patients with their survival times and parameters.
  The patients are filtered based on a parameter range.

  Parameters
  ----------
  all_patients : list of KMPatient
    List of all patients with their survival times and parameters.
  """
  def __init__(
    self,
    all_patients: list[KaplanMeierPatientDistribution],
    parameter_min: float = -np.inf,
    parameter_max: float = np.inf,
  ):
    self.__all_patients = all_patients
    self.__parameter_min = parameter_min
    self.__parameter_max = parameter_max
  @property
  def all_patients(self):
    """
    Returns the patients with their survival times and parameters.
    The parameters are probability distributions.
    """
    return self.__all_patients
  @property
  def all_patients_nominal(self):
    """
    Returns the nominal patients
    (i.e the patients with their nominal parameter values).
    """
    return [p.nominal for p in self.all_patients]
  @property
  def patient_times(self):
    """
    Returns the survival times of the patients.
    """
    patients = self.all_patients
    patient_times = frozenset({p.time for p in patients})
    return patient_times
  def generate(self, size, random_state):
    """
    Generates a list of Kaplan-Meier curves from the probability distributions.
    """
    patients = np.array([p.rvs(size=size, random_state=random_state) for p in self.all_patients])
    patients = np.transpose(patients)
    return KaplanMeierCollection(
      [
        KaplanMeierInstance(
          generated_patients, self.__parameter_min, self.__parameter_max
        ) for generated_patients in patients
      ],
      KaplanMeierInstance(
        self.all_patients_nominal, self.__parameter_min, self.__parameter_max
      )
    )

class KaplanMeierDistributionsPlot(KaplanMeierBase):
  """
  Class to represent a plot of Kaplan-Meier curves with uncertainties.
  It contains a list of patients with their survival times and parameters.
  The patients are filtered based on a parameter range, and each patient
  enters a different Kaplan-Meier curve.  The same patient will enter
  different curves for each value of the random number.
  """

  def __init__(
    self,
    all_patients: list[KaplanMeierPatientDistribution],
    thresholds: list[float],
  ):
    self.__all_patients = all_patients
    self.__thresholds = [-np.inf] + sorted(thresholds) + [np.inf]
    self.__curves: list[KaplanMeierDistributions] = []
    for i in range(len(self.__thresholds) - 1):
      self.__curves.append(
        KaplanMeierDistributions(
          all_patients,
          self.__thresholds[i],
          self.__thresholds[i + 1],
        )
      )

  @property
  def all_patients(self):
    """
    Returns the patients with their survival times and parameters.
    The parameters are probability distributions.
    """
    return self.__all_patients

  def plot(self, size=1000, random_state=None, show=False, saveas=None): #pylint: disable=too-many-locals
    """
    Plots the Kaplan-Meier curves.
    """
    colors = (
      ('blue', 'dodgerblue', 'skyblue'),
      ('red', 'orangered', 'salmon'),
      ('green', 'limegreen', 'lightgreen'),
      ('purple', 'mediumpurple', 'plum'),
      ('orange', 'darkorange', 'peachpuff'),
      ('brown', 'saddlebrown', 'tan'),
      ('pink', 'deeppink', 'lightpink'),
      ('gray', 'dimgray', 'lightgray'),
    )
    if len(self.__curves) > len(colors):
      raise ValueError(
        f"Too many curves ({len(self.__curves)}) for the available colors ({len(colors)})"
      )
    times_for_plot = self.times_for_plot
    for i, (curve, (color_nominal, color_68, color_95)) in enumerate(zip(self.__curves, colors)):
      kmc = curve.generate(size=size, random_state=random_state)
      nominal_x, nominal_y = kmc.nominalkm.points_for_plot(times_for_plot=times_for_plot)
      plt.plot(
        nominal_x,
        nominal_y,
        label=f"Nominal {i + 1}: {self.__thresholds[i]} <= parameter < {self.__thresholds[i + 1]}",
        color=color_nominal,
        linestyle='--'
      )
      sigmas = [-2, -1, 0, 1, 2]
      quantiles = [(1 + scipy.special.erf(nsigma/np.sqrt(2))) / 2 for nsigma in sigmas]
      p_m95, p_m68, _, p_p68, p_p95 = kmc.survival_probabilities_quantiles(
        times_for_plot=times_for_plot,
        quantiles=quantiles,
      )
      x_m95, y_m95 = kmc.get_points_for_plot(times_for_plot, p_m95)
      x_m68, y_m68 = kmc.get_points_for_plot(times_for_plot, p_m68)
      x_p68, y_p68 = kmc.get_points_for_plot(times_for_plot, p_p68)
      x_p95, y_p95 = kmc.get_points_for_plot(times_for_plot, p_p95)
      np.testing.assert_array_equal(x_m95, x_p95)
      np.testing.assert_array_equal(x_m68, x_p68)
      plt.fill_between(
        x_m68,
        y_m68,
        y_p68,
        color=color_68,
        alpha=0.5,
        label=f'68% CL {i + 1}'
      )
      plt.fill_between(
        x_m95,
        y_m95,
        y_p95,
        color=color_95,
        alpha=0.5,
        label=f'95% CL {i + 1}'
      )

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan-Meier Curves")
    plt.legend()
    plt.grid()
    if saveas is not None:
      plt.savefig(saveas)
    if show:
      plt.show()
    plt.close()

  @property
  def patient_times(self):
    """
    Returns the survival times of the patients.
    """
    return frozenset.union(
      *[kmd.patient_times for kmd in self.__curves]
    )
  #pylint: enable=similarities
