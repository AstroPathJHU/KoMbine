#This code was generated with the help of ChatGPT.

"""
A datacard class to specify the inputs to ROC Picker.
This is heavily modeled after the datacard format used in the Higgs Combine Tool.
"""

import argparse
import itertools
import pathlib
import re
import scipy.stats
from .delta_functions import DeltaFunctions
from .discrete import DiscreteROC
from .systematics_mc import ROCDistributions, ScipyDistribution

class Datacard:
  """
  A datacard class to specify the inputs to ROC Picker.
  Refer to docs/03_examples.md for usage examples.
  """
  def __init__(self, patients=None, systematics=None, observable_type=None):
    """
    Initialize a datacard.
    This function should not be called directly. Use `parse_datacard` instead.
    """
    if patients is None:
      patients = []
    if systematics is None:
      systematics = []
    if observable_type is None:
      raise ValueError("observable_type must be provided")

    self.patients = patients
    self.systematics = systematics
    self.observable_type = observable_type

  @staticmethod
  def parse_datacard(file_path): # pylint: disable=too-many-branches, too-many-statements
    #disable warnings because this function is just parsing a file and is not too complex
    """
    Parse a datacard file and return a Datacard object.

    Parameters:
    file_path (os.PathLike): Path to the datacard file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
      lines = file.readlines()

    data = {
      "patients": [],
      "systematics": [],
      "observable_type": None
    }

    responses, numerators, denominators = [], [], []

    for line in lines:
      line = line.strip()
      if not line or line.startswith('#') or line.startswith('---'):
        continue

      if line.startswith("observable_type"):
        data["observable_type"] = line.split()[1]
      elif line.startswith("bin"):
        continue
      elif line.startswith("response"):
        responses = line.split()[1:]
        continue
      elif line.startswith("observable"):
        if data["observable_type"] != "fixed":
          raise ValueError(
            f"Unexpected 'observable' line for observable_type '{data['observable_type']}'"
          )
        values = list(map(float, line.split()[1:]))
        try:
          for response, value in zip(responses, values, strict=True):
            data["patients"].append({
              "response": response,
              "value": value
            })
        except ValueError as e:
          raise ValueError("Mismatched lengths in responses and values") from e
        continue
      elif line.startswith("count"):
        if data["observable_type"] != "poisson":
          raise ValueError(
            f"Unexpected 'count' line for observable_type '{data['observable_type']}'"
          )
        values = list(map(int, line.split()[1:]))
        try:
          for response, value in zip(responses, values, strict=True):
            data["patients"].append({
              "response": response,
              "value": value
            })
        except ValueError as e:
          raise ValueError("Mismatched lengths in responses and values") from e
        continue
      elif line.startswith("num"):
        if data["observable_type"] != "poisson_ratio":
          raise ValueError(f"Unexpected 'num' line for observable_type '{data['observable_type']}'")
        numerators = list(map(int, line.split()[1:]))
        continue
      elif line.startswith("denom"):
        if data["observable_type"] != "poisson_ratio":
          raise ValueError(
            f"Unexpected 'denom' line for observable_type '{data['observable_type']}'"
          )
        denominators = list(map(int, line.split()[1:]))
        try:
          for response, num, denom in zip(responses, numerators, denominators, strict=True):
            data["patients"].append({
              "response": response,
              "numerator": num,
              "denominator": denom
            })
        except ValueError as e:
          raise ValueError("Mismatched lengths in responses, numerators, and denominators") from e
        continue
      elif re.match(r'.*\s+lnN\s+.*', line):
        tokens = line.split()
        data["systematics"].append({
          "type": tokens[0],
          "method": tokens[1],
          "values": [float(x) if x != '-' else None for x in tokens[2:]]
        })
        continue
      else:
        raise ValueError(f"Unexpected line format: {line}")

    if data["observable_type"] not in ["fixed", "poisson", "poisson_ratio"]:
      raise ValueError(f"Invalid observable_type: {data['observable_type']}")

    return Datacard(
      patients=data["patients"],
      systematics=data["systematics"],
      observable_type=data["observable_type"]
    )

  def systematics_mc(self, *, id_start=0, flip_sign=False):
    """
    Generate a set of ROCDistributions for generating ROC curve
    error bands using the MC method.  See docs/02_rocpicker.tex for 
    math details and docs/03_examples.md for usage examples.
    """
    id_generator = itertools.count(id_start)
    patient_distributions = []

    if self.observable_type == "fixed":
      for p in self.patients:
        observable = p["value"]
        patient_distributions.append({
          "response": p["response"],
          "observable": observable
        })

    elif self.observable_type == "poisson":
      for p in self.patients:
        observable = ScipyDistribution(
          nominal=p["value"],
          scipydistribution=scipy.stats.poisson(mu=p["value"]),
          unique_id=next(id_generator)
        )
        patient_distributions.append({
          "response": p["response"],
          "observable": observable
        })

    elif self.observable_type == "poisson_ratio":
      for p in self.patients:
        observable = ScipyDistribution(
          nominal=p["numerator"],
          scipydistribution=scipy.stats.poisson(mu=p["numerator"]),
          unique_id=next(id_generator)
        ) / ScipyDistribution(
          nominal=p["denominator"],
          scipydistribution=scipy.stats.poisson(mu=p["denominator"]),
          unique_id=next(id_generator)
        )
        patient_distributions.append({
          "response": p["response"],
          "observable": observable
        })

    # Apply log-normal systematics
    for systematic in self.systematics:
      if systematic["method"] == "lnN":
        log_norm_factor = ScipyDistribution(
          nominal=0,
          scipydistribution=scipy.stats.norm(),
          unique_id=next(id_generator)
        )
        try:
          for patient, value in zip(patient_distributions, systematic["values"], strict=True):
            if value is not None:
              patient["observable"] *= value ** log_norm_factor
        except ValueError as e:
          raise ValueError(
            "Mismatched lengths in patient distributions and systematic values"
          ) from e
      else:
        raise ValueError(f"Unknown systematic method: {systematic['method']}")

    responders = [
      p["observable"]
      for p in patient_distributions
      if p["response"] == "responder"
    ]
    nonresponders = [
      p["observable"]
      for p in patient_distributions
      if p["response"] == "non-responder"
    ]

    return ROCDistributions(responders=responders, nonresponders=nonresponders, flip_sign=flip_sign)

  def discrete(self, **kwargs):
    """
    Generate a DiscreteROC object for the discrete method.
    See docs/02_rocpicker.tex for math details and docs/03_examples.md
    for usage examples.
    """
    if self.observable_type != "fixed":
      raise ValueError(f"Invalid observable_type {self.observable_type} for discrete")
    if self.systematics:
      raise ValueError("Can't do systematics for discrete")

    responders = []
    nonresponders = []
    dct = {
      "responder": responders,
      "non-responder": nonresponders,
    }
    for p in self.patients:
      dct[p["response"]].append(p["value"])

    return DiscreteROC(responders=responders, nonresponders=nonresponders, **kwargs)

  def delta_functions(self, **kwargs):
    """
    Generate a DeltaFunctions object for the delta_functions method.
    See docs/02_rocpicker.tex for math details and docs/03_examples.md
    for usage examples.
    """
    if self.observable_type != "fixed":
      raise ValueError(f"Invalid observable_type {self.observable_type} for delta_functions")
    if self.systematics:
      raise ValueError("Can't do systematics for delta_functions")

    responders = []
    nonresponders = []
    dct = {
      "responder": responders,
      "non-responder": nonresponders,
    }
    for p in self.patients:
      dct[p["response"]].append(p["value"])

    return DeltaFunctions(responders=responders, nonresponders=nonresponders, **kwargs)

def plot_systematics_mc():
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
  datacard = Datacard.parse_datacard(args.datacard)
  rd = datacard.systematics_mc(flip_sign=args.__dict__.pop("flip_sign"))
  rocs = rd.generate(size=args.size, random_state=args.random_state)
  rocs.plot(saveas=args.output_file)

def plot_discrete():
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
  discrete = datacard.discrete(flip_sign=args.__dict__.pop("flip_sign"))
  discrete.make_plots(**args.__dict__)

def plot_delta_functions():
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
  deltafunctions = datacard.delta_functions(flip_sign=args.__dict__.pop("flip_sign"))
  deltafunctions.make_plots(**args.__dict__)
