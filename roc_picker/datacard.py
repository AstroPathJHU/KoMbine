"""
A datacard class to specify the inputs to ROC Picker.
This is heavily modeled after the datacard format used in the Higgs Combine Tool.
"""

import abc
import argparse
import functools
import itertools
import pathlib
import scipy.stats
from .delta_functions import DeltaFunctionsROC
from .discrete import DiscreteROC
from .systematics_mc import DistributionBase, ROCDistributions, ScipyDistribution

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
  def _create_observable_distribution(self) -> DistributionBase | float:
    """
    Abstract method to get the observable distribution.
    """

  @functools.cached_property
  def observable_distribution(self) -> DistributionBase | float:
    """
    Get the observable distribution.
    """
    return self._create_observable_distribution()

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
    return self.value

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

class Patient:
  """
  A class to represent a patient.
  """
  def __init__(
    self,
    response: Response | None = None,
    observable: Observable | None = None,
    systematics: list[tuple[Systematic, float]] | None = None
  ):
    self.__response = None
    self.__observable = None
    self.__systematics = []
    self.response = response
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
    if not isinstance(value, Response):
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
  def observable(self):
    """
    Get the observable for the patient.
    """
    return self.__observable
  @observable.setter
  def observable(self, value):
    if not isinstance(value, Observable):
      raise ValueError(f"Invalid observable: {value}")
    if self.__observable is not None:
      if (
        isinstance(value, PoissonRatioObservable)
        and isinstance(self.__observable, PoissonRatioObservable)
      ):
        self.__observable.numerator = value.numerator
        self.__observable.denominator = value.denominator
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

  def add_systematic(self, systematic: Systematic, value: float):
    """
    Add a systematic to the patient.
    """
    for s, v in self.__systematics:
      if s == systematic:
        raise ValueError(f"Systematic {systematic} already added with value {v}")
    self.__systematics.append((systematic, value))

  def get_distribution(self):
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
  def parse_datacard(cls, file_path): # pylint: disable=too-many-branches, too-many-statements
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

    unique_id_generator = itertools.count(0)

    for line in lines:
      line = line.strip()
      if not line or line.startswith('#') or line.startswith('---'):
        continue

      split = line.split()
      if split[0] == "observable_type":
        observable_type = split[1]
        if observable_type not in ["fixed", "poisson", "poisson_ratio"]:
          raise ValueError(f"Invalid observable_type: {observable_type}")
      elif split[0] == "bin":
        pass
      elif split[0] == "response":
        if patients is not None:
          raise ValueError("Multiple 'response' lines found")
        patients = cls.process_response_line(
          split=split,
        )
      elif split[0] in ["observable", "count", "num", "denom"]:
        if observable_type is None:
          raise ValueError(f"No 'observable_type' line found before '{split[0]}' line")
        if patients is None:
          raise ValueError(f"No 'response' line found before '{split[0]}' line")
        observables = cls.process_observable_line(
          split=split,
          observable_type=observable_type,
          unique_id_generator=unique_id_generator
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
    if split[0] != "response":
      raise ValueError(f"Invalid response line: {split}")
    responses = [Response(response) for response in split[1:]]
    patients = [Patient(response=response) for response in responses]
    return patients

  @classmethod
  def process_observable_line(
    cls,
    *,
    split: list[str],
    observable_type: str,
    unique_id_generator: itertools.count
  ):
    """
    Process a line of the datacard that specifies observables.
    This function is used to create the appropriate observable objects.
    """
    if (observable_type, split[0]) not in (
      ("fixed", "observable"),
      ("poisson", "count"),
      ("poisson_ratio", "num"),
      ("poisson_ratio", "denom"),
    ):
      raise ValueError(
        f"Unexpected '{split[0]}' line for observable_type '{observable_type}'"
      )
    value_type = {
      "fixed": float,
      "poisson": int,
      "poisson_ratio": int,
    }[observable_type]
    values = [value_type(_) for _ in split[1:]]

    if observable_type == "fixed":
      observables = [FixedObservable(value) for value in values]
    elif observable_type == "poisson":
      observables = [
        PoissonObservable(
          value,
          unique_id=next(unique_id_generator)
        ) for value in values
      ]
    elif observable_type == "poisson_ratio":
      kw = {"num": "numerator", "denom": "denominator"}[split[0]]
      unique_id_kw = "unique_id_" + kw
      observables = [
        PoissonRatioObservable(
          **{
            kw: value,
            unique_id_kw: next(unique_id_generator),
          },
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
      if not isinstance(distribution, float):
        assert False
      dct[p.is_responder].append(distribution)

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
      if not isinstance(distribution, float):
        assert False
      dct[p.is_responder].append(distribution)

    return DeltaFunctionsROC(responders=responders, nonresponders=nonresponders, **kwargs)

def plot_systematics_mc_roc():
  """
  Run MC method from a datacard.
  """
  # pylint: disable=C0301
  parser = argparse.ArgumentParser(description="Run MC method from a datacard.")
  parser.add_argument("datacard", type=pathlib.Path, help="Path to the datacard file.")
  parser.add_argument("output_file", type=pathlib.Path, help="Path to the output file for the plot.")
  parser.add_argument("--nrocs", type=int, help="Number of MC samples to generate.", default=10000, dest="size")
  parser.add_argument("--random-seed", type=int, help="Random seed for generation", dest="random_state", default=123456)
  parser.add_argument("--flip-sign", action="store_true", help="flip the sign of the observable (use this if AUC is < 0.5 and you want it to be > 0.5)")
  # pylint: enable=C0301

  args = parser.parse_args()
  datacard = Datacard.parse_datacard(args.__dict__.pop("datacard"))
  rd = datacard.systematics_mc_roc(flip_sign=args.__dict__.pop("flip_sign"))
  rocs = rd.generate(size=args.__dict__.pop("size"), random_state=args.__dict__.pop("random_state"))
  rocs.plot(saveas=args.__dict__.pop("output_file"))
  if args.__dict__:
    raise ValueError(f"Unused arguments: {args.__dict__}")

def plot_discrete_roc():
  """
  Run discrete method from a datacard.
  """
  # pylint: disable=C0301
  parser = argparse.ArgumentParser(description="Run discrete method from a datacard.")
  parser.add_argument("datacard", type=pathlib.Path, help="Path to the datacard file.")
  parser.add_argument("--roc-filename", type=pathlib.Path, help="Path to the output file for the ROC curve.", dest="rocfilename")
  parser.add_argument("--roc-errors-filename", type=pathlib.Path, help="Path to the output file for the ROC curve with error bands.", dest="rocerrorsfilename")
  parser.add_argument("--scan-filename", type=pathlib.Path, help="Path to the output file for the likelihood scan", dest="scanfilename")
  parser.add_argument("--y-upper-limit", type=float, help="y axis upper limit of the likelihood scan plot", dest="yupperlim")
  parser.add_argument("--npoints", type=int, help="number of points in the likelihood scan", dest="npoints")
  parser.add_argument("--flip-sign", action="store_true", help="flip the sign of the observable (use this if AUC is < 0.5 and you want it to be > 0.5)")
  # pylint: enable=C0301

  args = parser.parse_args()
  datacard = Datacard.parse_datacard(args.__dict__.pop("datacard"))
  discrete = datacard.discrete_roc(flip_sign=args.__dict__.pop("flip_sign"))
  discrete.make_plots(
    filenames=[
      args.__dict__.pop("rocfilename"),
      args.__dict__.pop("scanfilename"),
      args.__dict__.pop("rocerrorsfilename"),
    ],
    yupperlim=args.__dict__.pop("yupperlim"),
    npoints=args.__dict__.pop("npoints"),
  )
  if args.__dict__:
    raise ValueError(f"Unused arguments: {args.__dict__}")

def plot_delta_functions_roc():
  """
  Run delta functions method from a datacard.
  """
  # pylint: disable=C0301
  parser = argparse.ArgumentParser(description="Run delta functions method from a datacard.")
  parser.add_argument("datacard", type=pathlib.Path, help="Path to the datacard file.")
  parser.add_argument("--roc-filename", type=pathlib.Path, help="Path to the output file for the ROC curve.", dest="rocfilename")
  parser.add_argument("--roc-errors-filename", type=pathlib.Path, help="Path to the output file for the ROC curve with error bands.", dest="rocerrorsfilename")
  parser.add_argument("--scan-filename", type=pathlib.Path, help="Path to the output file for the likelihood scan", dest="scanfilename")
  parser.add_argument("--y-upper-limit", type=float, help="y axis upper limit of the likelihood scan plot", dest="yupperlim")
  parser.add_argument("--npoints", type=int, help="number of points in the likelihood scan", dest="npoints")
  parser.add_argument("--flip-sign", action="store_true", help="flip the sign of the observable (use this if AUC is < 0.5 and you want it to be > 0.5)")
  # pylint: enable=C0301

  args = parser.parse_args()
  datacard = Datacard.parse_datacard(args.__dict__.pop("datacard"))
  deltafunctions = datacard.delta_functions_roc(flip_sign=args.__dict__.pop("flip_sign"))
  deltafunctions.make_plots(
    filenames=[
      args.__dict__.pop("rocfilename"),
      args.__dict__.pop("scanfilename"),
      args.__dict__.pop("rocerrorsfilename"),
    ],
    yupperlim=args.__dict__.pop("yupperlim"),
    npoints=args.__dict__.pop("npoints"),
  )
  if args.__dict__:
    raise ValueError(f"Unused arguments: {args.__dict__}")
