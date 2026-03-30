#pylint: disable=too-many-lines
"""
A datacard class to specify the inputs to KoMbine and ROC Picker.
This is heavily modeled after the datacard format used in the Higgs Combine Tool.
"""

import abc
import functools
import itertools
import os

import numpy as np
import scipy.stats

from roc_picker.delta_functions import DeltaFunctionsROC
from roc_picker.discrete import DiscreteROC
from roc_picker.systematics_mc import (
  DistributionBase,
  DummyDistribution,
  ROCDistributions,
  ScipyDistribution,
)
from .kaplan_meier_likelihood import (
  KaplanMeierLikelihood,
  KaplanMeierPatientNLL,
)
from .kaplan_meier_p_value_MINLP import MINLPforKMPValue
from .kaplan_meier_hazard_ratio_MINLP import MINLPforKMHazardRatio
from .yi_correction import YiCorrectionForLogrank, YiCorrectionForCoxPH, YiCorrectionForKaplanMeier
from .utilities import LOG_ZERO_EPSILON_DEFAULT, prob_poisson_density_in_range, validate_class_probs

def _parse_prob_class_label(label: str) -> int | None:
  """
  Parse labels like "prob0" or "prob12" into an integer class index.
  """
  if not label.startswith("prob"):
    return None
  suffix = label[len("prob"):]
  if not suffix.isdigit():
    return None
  return int(suffix)

class Response:
  """
  A class to represent the response of a patient.
  """
  def __init__(self, response):
    self.response = response
    if self.response not in ["responder", "non-responder"]:
      raise ValueError(f"Invalid response: {self.response}")

  def __repr__(self):
    return f"Response(response={self.response})"

  def __str__(self):
    return f"Response: {self.response}"

class Observable(abc.ABC): # pylint: disable=too-few-public-methods
  """
  An abstract base class for observables.
  """
  @abc.abstractmethod
  def _create_observable_distribution(self) -> DistributionBase:
    """
    Abstract method to get the observable distribution.
    """

  @functools.cached_property
  def observable_distribution(self) -> DistributionBase:
    """
    Get the observable distribution.
    """
    return self._create_observable_distribution()

  @abc.abstractmethod
  def patient_nll(self, time, censored, *, systematics) -> KaplanMeierPatientNLL:
    """
    Get the patient NLL for the likelihood method.
    """

  @abc.abstractmethod
  def observed_parameter(self) -> float:
    """
    Get the observed parameter value.

    This is the scalar value used for classification and threshold comparison.
    Each observable type computes this differently based on its measurement type.
    """

  def probability_in_range(
    self,
    range_min: float,
    range_max: float,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
  ) -> float:
    """
    Compute P(range_min <= true_parameter < range_max | observed data).

    Concrete observables must implement this to reflect their measurement
    uncertainty model. The default raises NotImplementedError.
    """
    raise NotImplementedError(
      f"probability_in_range not implemented for {type(self).__name__}"
    )

class FixedObservable(Observable):
  """
  A class to represent a fixed observable.
  """
  def __init__(self, value: float):
    self.value = value

  def __repr__(self):
    return f"{type(self).__name__}(value={self.value})"

  def _create_observable_distribution(self):
    """
    Get the observable distribution for a fixed observable.
    """
    return DummyDistribution(self.value)

  def __eq__(self, other):
    if not isinstance(other, FixedObservable):
      return NotImplemented
    return self.value == other.value

  def __str__(self):
    return str(self.value)

  def patient_nll(
    self,
    time: float,
    censored: bool,
    *,
    systematics: list[float] | None
  ) -> KaplanMeierPatientNLL:
    """
    Get the patient NLL for the likelihood method.
    """
    return KaplanMeierPatientNLL.from_fixed_observable(
      observable=self.value,
      censored=censored,
      time=time,
      systematics=systematics,
    )

  def observed_parameter(self) -> float:
    """
    Get the observed parameter value (the fixed value itself).
    """
    return self.value

  def probability_in_range(
    self,
    range_min: float,
    range_max: float,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
  ) -> float:
    """
    Fixed observables have no measurement uncertainty.
    """
    _ = prior_alpha
    _ = prior_beta
    return 1.0 if range_min <= self.value < range_max else 0.0

class DiscreteClassObservable(Observable):
  """
  A class to represent discrete-class probabilities for a patient.

  The parameter is the integer class index. Probabilities correspond to
  P(true_class = k | observed data).
  """
  def __init__(
    self,
    class_probs: list[float] | None = None,
    *,
    class_index: int | None = None,
    class_prob: float | None = None,
  ):
    self.__class_probs_by_index: dict[int, float] = {}
    self.__class_probs: tuple[float, ...] | None = None

    if class_probs is not None:
      self._set_class_probs_from_list(class_probs)

    if class_index is not None or class_prob is not None:
      if class_index is None or class_prob is None:
        raise ValueError("class_index and class_prob must be provided together")
      self.set_class_prob(class_index, class_prob)

  def __repr__(self):
    if self.__class_probs is not None:
      return f"{type(self).__name__}(class_probs={list(self.__class_probs)})"
    return f"{type(self).__name__}(class_probs=unfinalized)"

  def _set_class_probs_from_list(self, class_probs: list[float]):
    if self.__class_probs is not None or self.__class_probs_by_index:
      raise ValueError("Class probabilities already set")
    validate_class_probs(class_probs)
    self.__class_probs = tuple(float(p) for p in class_probs)

  def set_class_prob(self, class_index: int, class_prob: float) -> None:
    """
    Set the probability for a specific class index (used in datacard parsing).
    """
    if self.__class_probs is not None:
      raise ValueError("Class probabilities already finalized")
    if not isinstance(class_index, int) or class_index < 0:
      raise ValueError(f"Invalid class_index: {class_index}")
    if not isinstance(class_prob, (int, float)) or class_prob < 0:
      raise ValueError(f"Invalid class probability: {class_prob}")
    if class_index in self.__class_probs_by_index:
      existing = self.__class_probs_by_index[class_index]
      if existing != class_prob:
        raise ValueError(
          f"Class probability for index {class_index} already set to {existing}"
        )
      return
    self.__class_probs_by_index[class_index] = float(class_prob)

  def merge_from(self, other: "DiscreteClassObservable") -> None:
    """
    Merge class probabilities from another DiscreteClassObservable.
    """
    if not isinstance(other, DiscreteClassObservable):
      raise ValueError("Can only merge from DiscreteClassObservable")
    if other.__class_probs is not None:  # pylint: disable=protected-access
      raise ValueError("Cannot merge from finalized class probabilities")
    for class_index, class_prob in other.__class_probs_by_index.items():  # pylint: disable=protected-access
      self.set_class_prob(class_index, class_prob)

  def finalize_class_probs(self, n_classes: int) -> None:
    """
    Finalize class probabilities after all class lines have been parsed.
    """
    if self.__class_probs is not None:
      return
    if not isinstance(n_classes, int) or n_classes <= 0:
      raise ValueError(f"Invalid n_classes: {n_classes}")
    expected_indices = set(range(n_classes))
    if set(self.__class_probs_by_index.keys()) != expected_indices:
      missing = sorted(expected_indices - set(self.__class_probs_by_index.keys()))
      extra = sorted(set(self.__class_probs_by_index.keys()) - expected_indices)
      raise ValueError(
        f"Class probabilities missing indices {missing} or extra indices {extra}"
      )
    probs = [self.__class_probs_by_index[i] for i in range(n_classes)]
    total = float(sum(probs))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
      raise ValueError(f"Class probabilities must sum to 1, got {total}")
    self.__class_probs = tuple(float(p) for p in probs)

  def _require_finalized(self) -> tuple[float, ...]:
    if self.__class_probs is None:
      raise ValueError("Class probabilities not finalized")
    return self.__class_probs

  def _create_observable_distribution(self):
    """
    Discrete class observables do not currently support systematics MC.
    """
    raise NotImplementedError(
      "observable_distribution not supported for DiscreteClassObservable"
    )

  def patient_nll(
    self,
    time: float,
    censored: bool,
    *,
    systematics: list[float] | None
  ) -> KaplanMeierPatientNLL:
    """
    Get the patient NLL for the likelihood method.
    """
    if systematics:
      raise NotImplementedError(
        "Systematics are not supported for DiscreteClassObservable"
      )
    class_probs = list(self._require_finalized())
    return KaplanMeierPatientNLL.from_discrete_class_probs(
      class_probs=class_probs,
      time=time,
      censored=censored,
    )

  def observed_parameter(self) -> float:
    """
    Get the observed parameter value (argmax class index).
    """
    class_probs = self._require_finalized()
    return float(int(np.argmax(class_probs)))

  def probability_in_range(
    self,
    range_min: float,
    range_max: float,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
  ) -> float:
    """
    Sum class probabilities for indices within [range_min, range_max).
    """
    _ = prior_alpha
    _ = prior_beta
    class_probs = self._require_finalized()
    n_classes = len(class_probs)
    if not np.isfinite(range_min):
      range_min = 0
    if not np.isfinite(range_max):
      range_max = n_classes
    total = 0.0
    for idx, prob in enumerate(class_probs):
      if range_min <= idx < range_max:
        total += prob
    return float(total)

class PoissonObservable(Observable):
  """
  A class to represent a Poisson observable.
  """
  def __init__(self, count: int, unique_id: int):
    self.count = count
    if not isinstance(self.count, int) or self.count < 0:
      raise ValueError(f"Invalid count: {self.count}")
    self.unique_id = unique_id
    if not isinstance(self.unique_id, int):
      raise ValueError(f"Invalid unique_id: {self.unique_id}")

  def __repr__(self):
    return f"{type(self).__name__}(count={self.count})"

  def _create_observable_distribution(self):
    """
    Get the observable distribution for a Poisson observable.
    """
    return ScipyDistribution(
      nominal=self.count,
      scipydistribution=scipy.stats.poisson(mu=self.count),
      unique_id=self.unique_id,
    )

  def patient_nll(
    self,
    time: float,
    censored: bool, *,
    systematics: list[float] | None
  ) -> KaplanMeierPatientNLL:
    """
    Get the patient NLL for the likelihood method.
    """
    return KaplanMeierPatientNLL.from_count(
      count=self.count,
      censored=censored,
      time=time,
      systematics=systematics,
    )

  def observed_parameter(self) -> float:
    """
    Get the observed parameter value (the count itself).
    """
    return float(self.count)

  def probability_in_range(
    self,
    range_min: float,
    range_max: float,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
  ) -> float:
    """
    Treat the Poisson rate as the parameter and integrate its posterior.
    """
    return prob_poisson_density_in_range(
      observed_count=self.count,
      observed_area=1.0,
      range_min=range_min,
      range_max=range_max,
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )

class PoissonDensityObservable(Observable):
  """
  A class to represent a Poisson density observable:
  a count divided by a fixed area.

  Parameters:
  -----------
  numerator (int): The count for the numerator.
  area (float): The fixed area for the denominator.
  unique_id_numerator (int): A unique ID for the numerator distribution.
  """

  def __init__(
    self,
    *,
    numerator: int | None = None,
    denominator: float | None = None,
    unique_id_numerator: int
  ):
    self.__numerator = None
    self.__denominator = None
    self.numerator = numerator
    self.denominator = denominator
    self.unique_id_numerator = unique_id_numerator

    if not isinstance(unique_id_numerator, int):
      raise ValueError(f"Invalid unique_id_numerator: {unique_id_numerator}")

  def __repr__(self):
    return f"{type(self).__name__}(numerator={self.numerator}, area={self.denominator})"

  @property
  def numerator(self):
    """
    Get the count for the numerator.
    """
    return self.__numerator
  @numerator.setter
  def numerator(self, value):
    if value is None:
      return
    if not isinstance(value, int) or value < 0:
      raise ValueError(f"Invalid numerator: {value}")
    if self.__numerator is not None and self.__numerator != value:
      raise ValueError("Numerator already set")
    self.__numerator = value

  @property
  def denominator(self):
    """
    Get the fixed area for the denominator.
    """
    return self.__denominator
  @denominator.setter
  def denominator(self, value):
    if value is None:
      return
    if not isinstance(value, (int, float)) or value <= 0:
      raise ValueError(f"Invalid denominator: {value}")
    if self.__denominator is not None and self.__denominator != value:
      raise ValueError("Denominator already set")
    self.__denominator = value
  def _create_observable_distribution(self):
    """
    Get the observable distribution for a Poisson density observable.
    """
    if self.numerator is None or self.denominator is None:
      raise ValueError("Numerator and denominator must be set")
    return ScipyDistribution(
      nominal=self.numerator,
      scipydistribution=scipy.stats.poisson(mu=self.numerator),
      unique_id=self.unique_id_numerator,
    ) / self.denominator

  def patient_nll(
    self,
    time: float,
    censored: bool, *,
    systematics: list[float] | None
  ) -> KaplanMeierPatientNLL:
    """
    Get the patient NLL for the likelihood method.
    """
    if self.numerator is None or self.denominator is None:
      raise ValueError("Numerator and denominator must be set")
    return KaplanMeierPatientNLL.from_poisson_density(
      numerator_count=self.numerator,
      denominator_area=self.denominator,
      time=time,
      censored=censored,
      systematics=systematics,
    )

  def observed_parameter(self) -> float:
    """
    Get the observed parameter value (density = count / area).
    """
    if self.numerator is None or self.denominator is None:
      raise ValueError("Numerator and denominator must be set")
    if self.denominator == 0:
      return float('inf')
    return self.numerator / self.denominator

  def probability_in_range(
    self,
    range_min: float,
    range_max: float,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
  ) -> float:
    """
    Compute probability using the Poisson density posterior.
    """
    if self.numerator is None or self.denominator is None:
      raise ValueError("Numerator and denominator must be set")
    return prob_poisson_density_in_range(
      observed_count=self.numerator,
      observed_area=self.denominator,
      range_min=range_min,
      range_max=range_max,
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )

class PoissonRatioObservable(Observable):
  """
  A class to represent a ratio of two Poisson observables.

  This class is used to create a ratio of two Poisson distributions,
  which is useful for modeling the ratio of two counts.
  The numerator and denominator are specified as integers, and the
  unique IDs are used to identify the distributions in the datacard.

  Parameters:
  -----------
  numerator (int): The count for the numerator.
  denominator (int): The count for the denominator.
  unique_id_numerator (int): A unique ID for the numerator distribution.
  unique_id_denominator (int): A unique ID for the denominator distribution.
  """
  def __init__(
    self,
    *,
    numerator: int | None = None,
    denominator: int | None = None,
    unique_id_numerator: int,
    unique_id_denominator: int
  ):
    self.__numerator = None
    self.__denominator = None
    self.numerator = numerator
    self.denominator = denominator
    self.unique_id_numerator = unique_id_numerator
    self.unique_id_denominator = unique_id_denominator

    if not isinstance(unique_id_numerator, int):
      raise ValueError(f"Invalid unique_id_numerator: {unique_id_numerator}")
    if not isinstance(unique_id_denominator, int):
      raise ValueError(f"Invalid unique_id_denominator: {unique_id_denominator}")


  def __repr__(self):
    return f"{type(self).__name__}(numerator={self.numerator}, denominator={self.denominator})"

  @property
  def numerator(self):
    """
    Get the count for the numerator.
    """
    return self.__numerator
  @numerator.setter
  def numerator(self, value):
    if value is None:
      return
    if not isinstance(value, int) or value < 0:
      raise ValueError(f"Invalid numerator: {value}")
    if self.__numerator is not None and self.__numerator != value:
      raise ValueError("Numerator already set")
    self.__numerator = value
  @property
  def denominator(self):
    """
    Get the count for the denominator.
    """
    return self.__denominator
  @denominator.setter
  def denominator(self, value):
    if value is None:
      return
    if not isinstance(value, int) or value < 0:
      raise ValueError(f"Invalid denominator: {value}")
    if self.__denominator is not None and self.__denominator != value:
      raise ValueError("Denominator already set")
    self.__denominator = value

  def _create_observable_distribution(self):
    """
    Get the observable distribution for a ratio of two Poisson observables.
    """
    if self.numerator is None or self.denominator is None:
      raise ValueError("Numerator and denominator must be set")
    return ScipyDistribution(
      nominal=self.numerator,
      scipydistribution=scipy.stats.poisson(mu=self.numerator),
      unique_id=self.unique_id_numerator,
    ) / ScipyDistribution(
      nominal=self.denominator,
      scipydistribution=scipy.stats.poisson(mu=self.denominator),
      unique_id=self.unique_id_denominator,
    )

  def patient_nll(
    self,
    time: float,
    censored: bool,
    *,
    systematics: list[float] | None
  ) -> KaplanMeierPatientNLL:
    """
    Get the patient NLL for the likelihood method.
    """
    if self.numerator is None or self.denominator is None:
      raise ValueError("Numerator and denominator must be set")
    return KaplanMeierPatientNLL.from_poisson_ratio(
      numerator_count=self.numerator,
      denominator_count=self.denominator,
      time=time,
      censored=censored,
      systematics=systematics,
    )

  def observed_parameter(self) -> float:
    """
    Get the observed parameter value (ratio = numerator / denominator).
    """
    if self.numerator is None or self.denominator is None:
      raise ValueError("Numerator and denominator must be set")
    if self.denominator == 0:
      return float('inf')
    return self.numerator / self.denominator

  def probability_in_range(
    self,
    range_min: float,
    range_max: float,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
  ) -> float:
    """
    Yi correction probability is not implemented for Poisson ratios.
    """
    _ = range_min
    _ = range_max
    _ = prior_alpha
    _ = prior_beta
    raise NotImplementedError(
      "probability_in_range not implemented for PoissonRatioObservable"
    )


class Systematic:
  """
  A class to represent a systematic uncertainty.

  This class is used to apply systematic uncertainties to the observable
  distributions. The systematic type is specified as a string, and the
  unique ID is used to identify the systematic in the datacard.
  The only supported systematic type is "lnN", which represents a
  log-normal distribution.

  Parameters:
  name (str): The name of the systematic.
  systematic_type (str): The type of the systematic. Currently, only "lnN" is supported.
  unique_id (int): A unique ID for the systematic.
  """
  def __init__(self, name, systematic_type: str, unique_id: int):
    self.name = name
    if systematic_type not in ["lnN"]:
      raise ValueError(f"Invalid systematic type: {systematic_type}")
    self.systematic_type = systematic_type
    self.unique_id = unique_id
    self.__patients: list[Patient] = []

  def __repr__(self):
    return (
      f"Systematic(name={self.name}, systematic_type={self.systematic_type}, "
      f"unique_id={self.unique_id})"
    )

  @functools.cached_property
  def random_distribution(self):
    """
    Generate a random distribution for the systematic.
    """
    if self.systematic_type == "lnN":
      return ScipyDistribution(
        nominal=0,
        scipydistribution=scipy.stats.norm(),
        unique_id=self.unique_id
      )
    raise ValueError(f"Invalid systematic type: {self.systematic_type}")

  def apply(self, nominal, value):
    """
    Apply the systematic to a nominal value.
    """
    if self.systematic_type == "lnN":
      return nominal * value ** self.random_distribution
    raise ValueError(f"Invalid systematic type: {self.systematic_type}")

  def __eq__(self, other):
    if not isinstance(other, Systematic):
      return NotImplemented
    if self.name == other.name:
      if self.unique_id != other.unique_id:
        raise ValueError(
          f"Systematic {self.name} has different unique IDs: "
          f"{self.unique_id} and {other.unique_id}"
        )
      if self.systematic_type != other.systematic_type:
        raise ValueError(
          f"Systematic {self.name} has different types: "
          f"{self.systematic_type} and {other.systematic_type}"
        )
      return True
    return False

  def __hash__(self):
    return hash((self.name, self.systematic_type, self.unique_id))

  @property
  def patients(self):
    """
    Returns the patients that this systematic is applied to.
    """
    return tuple(self.__patients)

  def mark_as_applied_to_patient(self, patient: "Patient"):
    """
    Mark this systematic as applied to a patient.
    """
    self.__patients.append(patient)

class Patient: # pylint: disable=too-many-instance-attributes
  """
  A class to represent a patient.
  """
  def __init__( # pylint: disable=too-many-arguments
    self,
    *,
    response: Response | None = None,
    survival_time: float | None = None,
    censored: bool | None = None,
    observable: Observable | None = None,
    systematics: list[tuple[Systematic, float]] | None = None,
  ):
    self.__response = None
    self.__survival_time = None
    self.__censored = None
    self.__observable = None
    self.__systematics : list[tuple[Systematic, float]] = []
    self.response = response
    self.survival_time = survival_time
    self.censored = censored
    self.observable = observable
    if systematics is None:
      systematics = []
    for systematic, value in systematics:
      self.add_systematic(systematic, value)

  def __repr__(self):
    return f"Patient(response={self.response}, observable={self.observable})"

  @property
  def response(self):
    """
    Get the response for the patient.
    """
    return self.__response
  @response.setter
  def response(self, value):
    if value is not None and not isinstance(value, Response):
      raise ValueError(f"Invalid response: {value}")
    if self.__response is not None:
      raise ValueError("Response already set")
    self.__response = value
  @property
  def is_responder(self):
    """
    Check if the patient is a responder.
    """
    if self.response is None:
      raise ValueError("Response not set")
    return {
      "responder": True,
      "non-responder": False,
    }[self.response.response]

  @property
  def survival_time(self):
    """
    Get the survival time for the patient.
    """
    return self.__survival_time
  @survival_time.setter
  def survival_time(self, value):
    if value is not None and not isinstance(value, (int, float)):
      raise ValueError(f"Invalid survival time: {value}")
    if self.__survival_time is not None:
      raise ValueError("Survival time already set")
    self.__survival_time = value

  @property
  def censored(self):
    """
    Get the censored status for the patient.
    """
    return self.__censored
  @censored.setter
  def censored(self, value):
    if value is not None and not isinstance(value, bool):
      raise ValueError(f"Invalid censored status: {value}")
    if self.__censored is not None:
      raise ValueError("Censored status already set")
    self.__censored = value

  @property
  def observable(self) -> Observable | None:
    """
    Get the observable for the patient.
    """
    return self.__observable
  @observable.setter
  def observable(self, value):
    if value is not None and not isinstance(value, Observable):
      raise ValueError(f"Invalid observable: {value}")
    if self.__observable is not None:
      if (
        isinstance(value, PoissonRatioObservable)
        and isinstance(self.__observable, PoissonRatioObservable)
      ):
        self.__observable.numerator = value.numerator
        self.__observable.denominator = value.denominator
      elif (
        isinstance(value, PoissonDensityObservable)
        and isinstance(self.__observable, PoissonDensityObservable)
      ):
        self.__observable.numerator = value.numerator
        self.__observable.denominator = value.denominator
      elif (
        isinstance(value, DiscreteClassObservable)
        and isinstance(self.__observable, DiscreteClassObservable)
      ):
        self.__observable.merge_from(value)
      else:
        raise ValueError("Observable already set")
    else:
      self.__observable = value

  @property
  def systematics(self):
    """
    Get the systematics for the patient.
    """
    return self.__systematics

  def add_systematic(self, systematic: Systematic, value: float | None):
    """
    Add a systematic to the patient.
    """
    for s, v in self.__systematics:
      if s == systematic:
        raise ValueError(f"Systematic {systematic} already added with value {v}")
    if value is not None:
      self.__systematics.append((systematic, value))
      systematic.mark_as_applied_to_patient(self)

  def get_distribution(self) -> DistributionBase:
    """
    Get the distribution for the patient.
    """
    if self.observable is None:
      raise ValueError("Observable not set")
    result = self.observable.observable_distribution
    for systematic, value in self.__systematics:
      if value is not None:
        result = systematic.apply(result, value)
    return result

  def get_nll(self) -> KaplanMeierPatientNLL:
    """
    Get the NLL for the patient.
    """
    if self.observable is None:
      raise ValueError("Observable not set")
    if self.survival_time is None:
      raise ValueError("Survival time not set")
    if self.censored is None:
      raise ValueError("Censored status not set")
    systematics = []
    for systematic, value in self.__systematics:
      if len(systematic.patients) > 1:
        raise NotImplementedError("Correlated systematics among patients are not supported")
      if systematic.systematic_type == "lnN":
        systematics.append(value)
      else:
        raise NotImplementedError(f"Systematic type {systematic.systematic_type} not supported")
    result = self.observable.patient_nll(
      time=self.survival_time,
      censored=self.censored,
      systematics=systematics,
    )
    return result

  @property
  def time(self) -> float:
    """
    Alias for survival_time to conform to PatientLike protocol.
    """
    if self.survival_time is None:
      raise ValueError("Survival time not set")
    return self.survival_time

  @property
  def observed_parameter(self) -> float:
    """
    Get the observed parameter value from the observable.

    This property enables the Patient class to conform to the PatientLike protocol,
    allowing it to be used interchangeably with KaplanMeierPatientNLL in methods
    like compute_patient_prob_high.

    Returns:
        float: The observed parameter value from the observable.

    Raises:
        ValueError: If observable is not set.
    """
    if self.observable is None:
      raise ValueError("Observable not set")
    return self.observable.observed_parameter()

class Datacard:
  """
  A datacard class to specify the inputs to ROC Picker.
  Refer to docs/03_examples.md for usage examples.
  """
  def __init__(self, patients: list[Patient]):
    """
    Initialize a datacard.
    This function should not be called directly. Use `parse_datacard` instead.
    """
    self.__patients = patients

  @property
  def patients(self):
    """
    Get the patients in the datacard.
    """
    return self.__patients

  @property
  def observable_type(self):
    """
    Get the observable type for the datacard.
    """
    if not self.__patients:
      raise ValueError("No patients found")
    observable_types = {type(p.observable) for p in self.__patients}
    if len(observable_types) != 1:
      raise ValueError("Mismatched observable types")
    result, = observable_types
    return result

  @property
  def systematics(self):
    """
    Get the systematics for the datacard.
    """
    systematics = set()
    for p in self.__patients:
      for systematic, _ in p.systematics:
        systematics.add(systematic)
    return systematics

  @classmethod
  def parse_datacard(cls, file_path: os.PathLike): # pylint: disable=too-many-branches, too-many-statements, too-many-locals
    #disable warnings because this function is just parsing a file and is not too complex
    """
    Parse a datacard file and return a Datacard object.

    Parameters:
    file_path (os.PathLike): Path to the datacard file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
      lines = file.readlines()

    observable_type = None
    patients = None
    discrete_class_indices: set[int] = set()

    unique_id_generator = itertools.count(0)

    for line in lines:
      line = line.strip()
      if not line or line.startswith('#') or line.startswith('---'):
        continue

      split = line.split()
      if split[0] == "observable_type":
        observable_type = split[1]
        if observable_type not in [
          "fixed",
          "poisson",
          "poisson_density",
          "poisson_ratio",
          "discrete_classes",
        ]:
          raise ValueError(f"Invalid observable_type: {observable_type}")
      elif split[0] == "bin":
        pass
      elif split[0] in ["response", "survival_time"]:
        if patients is not None:
          raise ValueError("Multiple 'response' lines found")
        patients = cls.process_response_line(
          split=split,
        )
      elif split[0] == "censored":
        if patients is None:
          raise ValueError("No 'response' line found before 'censored' line")
        if len(split) != len(patients) + 1:
          raise ValueError(
            f"Number of censored values ({len(split) - 1}) "
            f"does not match number of patients ({len(patients)})"
          )
        for patient, censored in zip(patients, split[1:], strict=True):
          patient.censored = {
            0: False,
            1: True,
          }[int(censored)]
      elif (
        split[0] in ["observable", "count", "num", "denom", "area"]
        or _parse_prob_class_label(split[0]) is not None
      ):
        if observable_type is None:
          raise ValueError(f"No 'observable_type' line found before '{split[0]}' line")
        if patients is None:
          raise ValueError(f"No 'response' line found before '{split[0]}' line")

        prob_index = _parse_prob_class_label(split[0])
        if observable_type == "discrete_classes" and prob_index is None:
          raise ValueError(f"Expected probability line 'probN', got '{split[0]}'")
        if observable_type != "discrete_classes" and prob_index is not None:
          raise ValueError(
            f"Unexpected probability line '{split[0]}' for observable_type '{observable_type}'"
          )
        if prob_index is not None:
          discrete_class_indices.add(prob_index)

        observables = cls.process_observable_line(
          split=split,
          observable_type=observable_type,
          prob_index=prob_index,
          unique_id_generator=(
            unique_id_generator
            if patients[0].observable is None #pylint: disable=unsubscriptable-object
            #if the observable is already set, then the new Observable
            #object is not used and so we just use a dummy.
            else itertools.count(0)
          ),
        )
        if len(observables) != len(patients):
          raise ValueError(
            f"Number of {split[0]} values ({len(observables)}) "
            f"does not match number of patients ({len(patients)})"
          )
        for patient, observable in zip(patients, observables, strict=True):
          patient.observable = observable
      elif split[1] in ["lnN"]:
        if observable_type is None:
          raise ValueError(f"No 'observable_type' line found before '{split[0]}' line")
        if patients is None:
          raise ValueError(f"No 'response' line found before '{split[0]}' line")
        systematic, systematic_values = cls.process_systematic_line(
          split=split,
          unique_id_generator=unique_id_generator,
        )
        if len(systematic_values) != len(patients):
          raise ValueError(
            f"Number of systematic values ({len(systematic_values)}) "
            f"does not match number of patients ({len(patients)})"
          )
        for patient, value in zip(patients, systematic_values, strict=True):
          if value is not None:
            patient.add_systematic(systematic, value)
      else:
        raise ValueError(f"Unexpected line format: {line}")

    if observable_type is None:
      raise ValueError("No 'observable_type' line found")
    if patients is None:
      raise ValueError("No 'response' line found")

    if observable_type == "discrete_classes":
      if not discrete_class_indices:
        raise ValueError("No 'probN' lines found for discrete_classes observable")
      n_classes = max(discrete_class_indices) + 1
      if discrete_class_indices != set(range(n_classes)):
        raise ValueError(
          f"Discrete class indices must be contiguous from 0 to {n_classes - 1}"
        )
      for patient in patients:
        if not isinstance(patient.observable, DiscreteClassObservable):
          raise ValueError("Discrete classes require DiscreteClassObservable")
        patient.observable.finalize_class_probs(n_classes)
    return Datacard(
      patients=patients,
    )

  @classmethod
  def process_response_line(cls, split: list[str]):
    """
    Process a line of the datacard that specifies responses.
    This function is used to create the appropriate response objects.
    """
    if len(split) < 2:
      raise ValueError(f"Invalid response line: {split}")
    if split[0] == "response":
      responses = [Response(response) for response in split[1:]]
      patients = [Patient(response=response) for response in responses]
    elif split[0] == "survival_time":
      survival_times = [float(x) for x in split[1:]]
      patients = [Patient(survival_time=survival_time) for survival_time in survival_times]
    else:
      raise ValueError(f"Invalid response line: {split}")
    return patients

  @classmethod
  def process_observable_line(
    cls,
    *,
    split: list[str],
    observable_type: str,
    prob_index: int | None,
    unique_id_generator: itertools.count
  ):
    """
    Process a line of the datacard that specifies observables.
    This function is used to create the appropriate observable objects.
    """
    if (observable_type, split[0]) not in (
      ("fixed", "observable"),
      ("poisson", "count"),
      ("poisson_density", "num"),
      ("poisson_density", "area"),
      ("poisson_ratio", "num"),
      ("poisson_ratio", "denom"),
      ("discrete_classes", split[0]),
    ):
      raise ValueError(
        f"Unexpected '{split[0]}' line for observable_type '{observable_type}'"
      )
    if observable_type == "discrete_classes":
      if prob_index is None:
        raise ValueError("Discrete class lines require prob_index")
      values = [float(_) for _ in split[1:]]
    else:
      value_type = {
        ("fixed", "observable"): float,
        ("poisson", "count"): int,
        ("poisson_density", "num"): int,
        ("poisson_density", "area"): float,
        ("poisson_ratio", "num"): int,
        ("poisson_ratio", "denom"): int,
      }[observable_type, split[0]]
      values = [value_type(_) for _ in split[1:]]

    if observable_type == "fixed":
      observables = [FixedObservable(value) for value in values]
    elif observable_type == "poisson":
      observables = [
        PoissonObservable(
          int(value),
          unique_id=next(unique_id_generator)
        ) for value in values
      ]
    elif observable_type == "poisson_density":
      kw = {"num": "numerator", "area": "denominator"}[split[0]]
      observables = [
        PoissonDensityObservable(
          **{
            kw: value,
            "unique_id_numerator": next(unique_id_generator),
          },
        )
        for value in values
      ]
    elif observable_type == "poisson_ratio":
      kw = {"num": "numerator", "denom": "denominator"}[split[0]]
      observables = [
        PoissonRatioObservable(
          **{
            kw: int(value),
          },
          unique_id_numerator=next(unique_id_generator),
          unique_id_denominator=next(unique_id_generator),
        )
        for value in values
      ]
    elif observable_type == "discrete_classes":
      observables = [
        DiscreteClassObservable(
          class_index=prob_index,
          class_prob=value,
        )
        for value in values
      ]
    else:
      assert False, f"Unexpected observable_type: {observable_type}"

    return observables

  @classmethod
  def process_systematic_line(
    cls,
    *,
    split: list[str],
    unique_id_generator: itertools.count,
  ):
    """
    Process a line of the datacard that specifies systematics.
    This function is used to create the appropriate systematic objects.
    """
    systematic_name = split[0]
    systematic_type = split[1]
    systematic_values = [float(x) if x != '-' else None for x in split[2:]]
    systematic = Systematic(
      name=systematic_name,
      systematic_type=systematic_type,
      unique_id=next(unique_id_generator),
    )
    return systematic, systematic_values


  def systematics_mc_roc(self, *, flip_sign=False):
    """
    Generate a set of ROCDistributions for generating ROC curve
    error bands using the MC method.  See docs/02_rocpicker.tex for
    math details and docs/03_examples.md for usage examples.
    """

    responders = [
      p.get_distribution()
      for p in self.patients
      if p.is_responder
    ]
    nonresponders = [
      p.get_distribution()
      for p in self.patients
      if not p.is_responder
    ]

    return ROCDistributions(responders=responders, nonresponders=nonresponders, flip_sign=flip_sign)

  def discrete_roc(self, **kwargs):
    """
    Generate a DiscreteROC object for the discrete method.
    See docs/02_rocpicker.tex for math details and docs/03_examples.md
    for usage examples.
    """
    if self.observable_type != FixedObservable:
      raise ValueError(f"Invalid observable_type {self.observable_type} for discrete")
    if self.systematics:
      raise ValueError("Can't do systematics for discrete")

    responders: list[float] = []
    nonresponders: list[float] = []
    dct = {
      True: responders,
      False: nonresponders,
    }
    for p in self.patients:
      if not isinstance(p.observable, FixedObservable):
        raise ValueError(f"Invalid observable type {type(p.observable)} for discrete")
      distribution = p.get_distribution()
      if not isinstance(distribution, DummyDistribution):
        assert False
      dct[p.is_responder].append(float(distribution))

    return DiscreteROC(responders=responders, nonresponders=nonresponders, **kwargs)

  def delta_functions_roc(self, **kwargs):
    """
    Generate a DeltaFunctions object for the delta_functions method.
    See docs/02_rocpicker.tex for math details and docs/03_examples.md
    for usage examples.
    """
    if self.observable_type != FixedObservable:
      raise ValueError(f"Invalid observable_type {self.observable_type} for discrete")
    if self.systematics:
      raise ValueError("Can't do systematics for discrete")

    responders: list[float] = []
    nonresponders: list[float] = []
    dct = {
      True: responders,
      False: nonresponders,
    }
    for p in self.patients:
      if not isinstance(p.observable, FixedObservable):
        raise ValueError(f"Invalid observable type {type(p.observable)} for discrete")
      distribution = p.get_distribution()
      if not isinstance(distribution, DummyDistribution):
        assert False
      dct[p.is_responder].append(distribution.nominal)

    return DeltaFunctionsROC(responders=responders, nonresponders=nonresponders, **kwargs)

  def km_likelihood( # pylint: disable=too-many-arguments
    self,
    parameter_min: float,
    parameter_max: float,
    *,
    endpoint_epsilon: float = 1e-6,
    log_zero_epsilon: float = LOG_ZERO_EPSILON_DEFAULT,
    collapse_consecutive_deaths: bool = True,
  ) -> KaplanMeierLikelihood:
    """
    Generate a KaplanMeierLikelihood object for generating Kaplan-Meier
    error bands using the likelihood method.
    """
    patients = []
    for p in self.patients:
      nll = p.get_nll()
      patients.append(nll)
    return KaplanMeierLikelihood(
      all_patients=patients,
      parameter_min=parameter_min,
      parameter_max=parameter_max,
      endpoint_epsilon=endpoint_epsilon,
      log_zero_epsilon=log_zero_epsilon,
      collapse_consecutive_deaths=collapse_consecutive_deaths,
    )

  def km_p_value( #pylint: disable=too-many-arguments
    self,
    *,
    parameter_min: float = -np.inf,
    parameter_threshold: float,
    parameter_max: float = np.inf,
    log_zero_epsilon: float = LOG_ZERO_EPSILON_DEFAULT,
    tie_handling: str = "breslow",
  ) -> MINLPforKMPValue:
    """
    Generate a MINLPforKMPValue object for calculating p-values for Kaplan-Meier curves
    using the likelihood method.
    """
    patients = []
    for p in self.patients:
      nll = p.get_nll()
      patients.append(nll)
    return MINLPforKMPValue(
      all_patients=patients,
      parameter_min=parameter_min,
      parameter_threshold=parameter_threshold,
      parameter_max=parameter_max,
      log_zero_epsilon=log_zero_epsilon,
      tie_handling=tie_handling,
    )

  def km_hazard_ratio( #pylint: disable=too-many-arguments
    self,
    *,
    parameter_min: float = -np.inf,
    parameter_threshold: float,
    parameter_max: float = np.inf,
    log_zero_epsilon: float = LOG_ZERO_EPSILON_DEFAULT,
    tie_handling: str = "breslow",
    log_hazard_ratio_bounds: tuple[float, float] = (-10.0, 10.0),
  ) -> "MINLPforKMHazardRatio":
    """
    Generate a MINLPforKMHazardRatio object for calculating hazard ratios for Kaplan-Meier curves
    using the likelihood method.

    This method creates a hazard ratio calculator that can compute the best-fit hazard ratio,
    confidence intervals, and perform likelihood scans over hazard ratio values.

    Parameters
    ----------
    parameter_min : float, optional
        The minimum parameter value for the "low" group. Default is -inf.
    parameter_threshold : float
        The threshold separating the "low" and "high" groups.
    parameter_max : float, optional
        The maximum parameter value for the "high" group. Default is +inf.
    log_zero_epsilon : float, optional
        Small epsilon value to avoid log(0). Default from utilities.
    tie_handling : str, optional
        Method for handling tied death times. Currently only "breslow" is
        supported. Default is "breslow".
    log_hazard_ratio_bounds : tuple[float, float], optional
        Bounds on log(hazard ratio) for the Gurobi model, as
        (lower_bound, upper_bound).
        These correspond to hazard ratio bounds of (exp(lb), exp(ub)).
        Default is (-10.0, 10.0), allowing HR in [0.000045, 22026].
        Increase these if you need to explore more extreme hazard ratios.

    Returns
    -------
    MINLPforKMHazardRatio
        A hazard ratio calculator object with methods for computing hazard
        ratios, confidence intervals, and likelihood scans.

    Examples
    --------
    >>> from kombine.datacard import Datacard
    >>> datacard = Datacard.parse_datacard("datacard.txt")
    >>> hr_calc = datacard.km_hazard_ratio(parameter_threshold=0.5)
    >>> best_fit, lower_ci, upper_ci, result = hr_calc.hazard_ratio_confidence_interval()
    >>> print(f"Hazard ratio: {best_fit:.2f} [{lower_ci:.2f}, {upper_ci:.2f}]")
    """
    # Import here to avoid circular import
    from .kaplan_meier_hazard_ratio_MINLP import MINLPforKMHazardRatio  # pylint: disable=redefined-outer-name,import-outside-toplevel

    patients = []
    for p in self.patients:
      nll = p.get_nll()
      patients.append(nll)
    return MINLPforKMHazardRatio(
      all_patients=patients,
      parameter_min=parameter_min,
      parameter_threshold=parameter_threshold,
      parameter_max=parameter_max,
      log_zero_epsilon=log_zero_epsilon,
      tie_handling=tie_handling,
      log_hazard_ratio_bounds=log_hazard_ratio_bounds,
    )

  def km_p_value_logrank(
    self,
    *,
    parameter_threshold: float,
    parameter_min: float = -np.inf,
    parameter_max: float = np.inf,
    cox_only: bool = True,
  ) -> float:
    """
    Calculate p-value for comparing two Kaplan-Meier curves using the conventional
    logrank test method.

    This is a convenience method that creates a MINLPforKMPValue object
    and calls its survival_curves_pvalue_logrank method.

    Parameters
    ----------
    parameter_threshold : float
        The threshold value that separates the two groups.
    parameter_min : float, optional
        The minimum parameter value to include in the analysis. Default is -inf.
    parameter_max : float, optional
        The maximum parameter value to include in the analysis. Default is +inf.
    cox_only : bool, optional
        If True, only include patients whose observed parameter is within the
        specified ranges. Default is True.

    Returns
    -------
    float
        The p-value from the logrank test.
    """
    minlp_pvalue = self.km_p_value(
      parameter_min=parameter_min,
      parameter_threshold=parameter_threshold,
      parameter_max=parameter_max,
    )

    return minlp_pvalue.survival_curves_pvalue_logrank(
      cox_only=cox_only,
    )

  def km_p_value_logrank_yi(  #pylint: disable=too-many-arguments
    self,
    *,
    parameter_threshold: float,
    parameter_min: float = -np.inf,
    parameter_max: float = np.inf,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
  ) -> dict:
    """
    Calculate p-value using Yi's misclassification correction method (Section 3.7.1).

    Yi's method uses inverse probability weighting to account for measurement uncertainty,
    providing an alternative to KoMbine's MINLP optimization approach.

    Parameters
    ----------
    parameter_threshold : float
        The threshold value that separates the two groups.
    parameter_min : float, optional
        The minimum parameter value to include in the analysis. Default is -inf.
    parameter_max : float, optional
        The maximum parameter value to include in the analysis. Default is +inf.
    prior_alpha : float, optional
        Alpha parameter for Gamma prior. Default 0.5 (Jeffreys).
    prior_beta : float, optional
        Beta parameter for Gamma prior. Default 0.5.

    Returns
    -------
    dict
        Dictionary containing:
        - 'p_value' : float - The p-value from the corrected logrank test.
        - 'logrank_statistic' : float - The corrected test statistic.
        - 'U' : float - Weighted observed minus expected.
        - 'V' : float - Weighted variance.
        - 'n_low_observed' : int - Patients observed in low group.
        - 'n_high_observed' : int - Patients observed in high group.

    Notes
    -----
    See Yi (2017) "Statistical Analysis with Measurement Error or Misclassification",
    Section 3.7.1 for theoretical foundation.

    Patients can contribute to the low group, high group, or neither depending on
    their probability of lying within the requested parameter ranges.

    Examples
    --------
    >>> from kombine.datacard import Datacard
    >>> datacard = Datacard.parse_datacard("datacard.txt")
    >>> result = datacard.km_p_value_logrank_yi(parameter_threshold=0.5)
    >>> print(f"Yi's corrected p-value: {result['p_value']:.4f}")
    """
    yi_correction = YiCorrectionForLogrank(
      patients=self.patients,
      parameter_min=parameter_min,
      parameter_max=parameter_max,
      parameter_threshold=parameter_threshold,
    )

    return yi_correction.compute_pvalue(
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )

  def km_hazard_ratio_yi(  # pylint: disable=too-many-arguments,unused-argument
    self,
    *,
    parameter_threshold: float,
    hazard_ratio: float,
    parameter_min: float = -np.inf,
    parameter_max: float = np.inf,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.0,
    log_hazard_ratio_bounds: tuple[float, float] = (-10.0, 10.0),
  ) -> scipy.optimize.OptimizeResult:
    """
    Compute 2NLL at a hazard ratio using Yi's misclassification correction.

    Yi's method uses inverse probability weighting to account for measurement uncertainty,
    providing an alternative to KoMbine's MINLP optimization approach.

    Parameters
    ----------
    parameter_threshold : float
        The threshold value that separates the two groups.
    hazard_ratio : float
        The hazard ratio value at which to evaluate the 2NLL.
        H = 1 corresponds to equal hazards (null hypothesis).
    parameter_min : float, optional
        The minimum parameter value to include in the analysis. Default is -inf.
    parameter_max : float, optional
        The maximum parameter value to include in the analysis. Default is +inf.
    prior_alpha : float, optional
        Alpha parameter for Gamma prior. Default 0.5 (Jeffreys).
    prior_beta : float, optional
        Beta parameter for Gamma prior. Default 0.0.
    log_hazard_ratio_bounds : tuple[float, float], optional
        Bounds on log(hazard ratio) for compatibility. Not used in Yi's method.
        Default is (-10.0, 10.0).

    Returns
    -------
    scipy.optimize.OptimizeResult
        Optimization result with attributes:
        - x : float - The 2NLL value.
        - success : bool - Always True for Yi's method.
        - hazard_ratio : float - The hazard ratio value.
        - log_hazard_ratio : float - Natural log of hazard ratio.
        - cox_2NLL : float - Twice the corrected Cox partial likelihood.
        - patient_2NLL : float - Always 0.0 for Yi's method.

    Notes
    -----
    See Yi (2017) "Statistical Analysis with Measurement Error or Misclassification",
    Section 3.7.1 for theoretical foundation.

    Patients can contribute to the low group, high group, or neither depending on
    their probability of lying within the requested parameter ranges.

    Examples
    --------
    >>> from kombine.datacard import Datacard
    >>> datacard = Datacard.parse_datacard("datacard.txt")
    >>> result = datacard.km_hazard_ratio_yi(
    ...     parameter_threshold=0.5,
    ...     hazard_ratio=2.0
    ... )
    >>> print(f"2NLL at HR=2.0: {result.x:.2f}")
    """
    yi_correction = YiCorrectionForCoxPH(
      patients=self.patients,
      parameter_min=parameter_min,
      parameter_max=parameter_max,
      parameter_threshold=parameter_threshold,
    )

    return yi_correction.compute_2nll_at_hazard_ratio(
      hazard_ratio=hazard_ratio,
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )

  def km_survival_yi(  # pylint: disable=too-many-arguments
    self,
    *,
    parameter_min: float = -np.inf,
    parameter_max: float = np.inf,
    times_for_plot: list[float] | None = None,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.0,
  ) -> dict:
    """
    Calculate weighted Kaplan-Meier survival probabilities using Yi's correction.

    Yi's method uses inverse probability weighting to account for measurement uncertainty,
    providing point estimates of the best-fit Kaplan-Meier curve without confidence intervals.
    This implements the weighted KM estimator where each patient contributes based on their
    probability of belonging to the parameter range.

    Parameters
    ----------
    parameter_min : float, optional
        The minimum parameter value to include in the analysis. Default is -inf.
    parameter_max : float, optional
        The maximum parameter value to include in the analysis. Default is +inf.
    times_for_plot : list[float], optional
      Time points to use when evaluating the survival probabilities.
    prior_alpha : float, optional
        Alpha parameter for Gamma prior (Bayesian method only). Default 0.5 (Jeffreys).
    prior_beta : float, optional
        Beta parameter for Gamma prior (Bayesian method only). Default 0.0.

    Returns
    -------
    dict
        Dictionary containing:
        - 'survival_probabilities' : np.ndarray
            Weighted survival probabilities at each time point.
        - 'times_for_plot' : list[float]
            Time points where probabilities were calculated.
        - 'n_at_risk_weighted' : list[float]
            Weighted number of patients at risk at each death time.
        - 'n_deaths_weighted' : list[float]
            Weighted number of deaths at each death time.
        - 'n_at_risk' : list[int]
            Unweighted number of patients at risk at each death time.
        - 'n_deaths' : list[int]
            Unweighted number of deaths at each death time.
        - 'death_times' : list[float]
            Unique death times where events occurred.
        - 'method' : str
            'yi_correction'

    Notes
    -----
    See Yi (2017) "Statistical Analysis with Measurement Error or Misclassification",
    Section 3.7.1 for theoretical foundation. The weighted Kaplan-Meier extension
    applies per-patient probability weights to the standard KM formula.

    Unlike MINLP approaches, Yi's method provides point estimates only without
    confidence intervals around the survival curve.

    Examples
    --------
    >>> from kombine.datacard import Datacard
    >>> datacard = Datacard.parse_datacard("datacard.txt")
    >>> result = datacard.km_survival_yi(parameter_min=0.0, parameter_max=1.0)
    >>> survival_probs = result['survival_probabilities']
    >>> times = result['times_for_plot']
    """
    yi_correction = YiCorrectionForKaplanMeier(
      patients=self.patients,
      parameter_min=parameter_min,
      parameter_max=parameter_max,
    )

    return yi_correction.compute_weighted_survival_probabilities(
      times_for_plot=times_for_plot,
      prior_alpha=prior_alpha,
      prior_beta=prior_beta,
    )

  def clear_distributions(self):
    """
    Delete the distributions for all patients.
    This is useful for clearing the unique_ids so that they can be
    regenerated.  You can always rerun systematics_mc_roc()
    to regenerate the distributions.
    """
    for p in self.patients:
      if p.observable is not None:
        del p.observable.observable_distribution
    for s in self.systematics:
      del s.random_distribution
