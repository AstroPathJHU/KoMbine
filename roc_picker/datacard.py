import argparse, itertools, pathlib, re, scipy.stats
from .delta_functions import DeltaFunctions
from .discrete import DiscreteROC
from .systematics_mc import ROCDistributions, ScipyDistribution

class Datacard:
  """
  Generated with the help of ChatGPT
  """
  def __init__(self, patients=None, systematics=None, observable_type=None):
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
  def parse_datacard(file_path):
    with open(file_path, 'r') as file:
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
          raise ValueError(f"Unexpected 'observable' line for observable_type '{data['observable_type']}'")
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
          raise ValueError(f"Unexpected 'count' line for observable_type '{data['observable_type']}'")
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
          raise ValueError(f"Unexpected 'denom' line for observable_type '{data['observable_type']}'")
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

    return Datacard(patients=data["patients"], systematics=data["systematics"], observable_type=data["observable_type"])

  def systematics_mc(self, saveas=None, *, id_start=0):
    id_generator = itertools.count(id_start)
    patient_distributions = []

    if self.observable_type == "fixed":
      for p in self.patients:
        fixed_value = p["value"]
        patient_distributions.append({
          "response": p["response"],
          "ratio": fixed_value
        })

    elif self.observable_type == "poisson":
      for p in self.patients:
        count = ScipyDistribution(nominal=p["value"], scipydistribution=scipy.stats.poisson(mu=p["value"]), id=next(id_generator))
        patient_distributions.append({
          "response": p["response"],
          "ratio": count
        })

    elif self.observable_type == "poisson_ratio":
      for p in self.patients:
        numerator = ScipyDistribution(nominal=p["numerator"], scipydistribution=scipy.stats.poisson(mu=p["numerator"]), id=next(id_generator))
        denominator = ScipyDistribution(nominal=p["denominator"], scipydistribution=scipy.stats.poisson(mu=p["denominator"]), id=next(id_generator))
        ratio = numerator / denominator
        patient_distributions.append({
          "response": p["response"],
          "ratio": ratio
        })

    # Apply log-normal systematics
    for systematic in self.systematics:
      if systematic["method"] == "lnN":
        log_norm_factor = ScipyDistribution(
          nominal=0,
          scipydistribution=scipy.stats.norm(),
          id=next(id_generator)
        )
        try:
          for patient, value in zip(patient_distributions, systematic["values"], strict=True):
            if value is not None:
              patient["ratio"] *= value ** log_norm_factor
        except ValueError as e:
          raise ValueError("Mismatched lengths in patient distributions and systematic values") from e

    responders = [p["ratio"] for p in patient_distributions if p["response"] == "responder"]
    nonresponders = [p["ratio"] for p in patient_distributions if p["response"] == "non-responder"]

    return ROCDistributions(responders=responders, nonresponders=nonresponders, flip_sign=True)

  def discrete(self, **kwargs):
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
  parser = argparse.ArgumentParser(description="Run MC method from a datacard.")
  parser.add_argument("datacard", type=pathlib.Path, help="Path to the datacard file.")
  parser.add_argument("output_file", type=pathlib.Path, help="Path to the output file for the plot.")
  parser.add_argument("--nrocs", type=int, help="Number of MC samples to generate.", default=10000, dest="size")
  parser.add_argument("--random-seed", type=int, help="Random seed for generation", dest="random_state")

  args = parser.parse_args()
  datacard = Datacard.parse_datacard(args.datacard)
  rd = datacard.systematics_mc()
  rocs = rd.generate(size=args.size, random_state=args.random_state)
  rocs.plot(saveas=args.output_file)

def plot_discrete():
  parser = argparse.ArgumentParser(description="Run discrete method from a datacard.")
  parser.add_argument("datacard", type=pathlib.Path, help="Path to the datacard file.")
  parser.add_argument("--roc-filename", type=pathlib.Path, help="Path to the output file for the ROC curve.", target="rocfilename")
  parser.add_argument("--roc-errors-filename", type=pathlib.Path, help="Path to the output file for the ROC curve with error bands.", target="rocerrorsfilename")
  parser.add_argument("--scan-filename", type=pathlib.Path, help="Path to the output file for the likelihood scan", target="scanfilename")
  parser.add_argument("--y-upper-limit", type=float, help="y axis upper limit of the likelihood scan plot", target="yupperlimit")
  parser.add_argument("--npoints", type=int, help="number of points in the likelihood scan", target="npoints")
  parser.add_argument("--flip-sign", action="store_true", help="flip the sign of the observable (use this if AUC is < 0.5 and you want it to be > 0.5)")

  args = parser.parse_args()
  datacard = Datacard.parse_datacard(args.__dict__.pop("datacard"))
  discrete = datacard.discrete()
  discrete.plot(**args.__dict__)

def plot_delta_functions():
  parser = argparse.ArgumentParser(description="Run discrete method from a datacard.")
  parser.add_argument("datacard", type=pathlib.Path, help="Path to the datacard file.")
  parser.add_argument("--roc-filename", type=pathlib.Path, help="Path to the output file for the ROC curve.", target="rocfilename")
  parser.add_argument("--roc-errors-filename", type=pathlib.Path, help="Path to the output file for the ROC curve with error bands.", target="rocerrorsfilename")
  parser.add_argument("--scan-filename", type=pathlib.Path, help="Path to the output file for the likelihood scan", target="scanfilename")
  parser.add_argument("--y-upper-limit", type=float, help="y axis upper limit of the likelihood scan plot", target="yupperlimit")
  parser.add_argument("--npoints", type=int, help="number of points in the likelihood scan", target="npoints")
  parser.add_argument("--flip-sign", action="store_true", help="flip the sign of the observable (use this if AUC is < 0.5 and you want it to be > 0.5)")

  args = parser.parse_args()
  datacard = Datacard.parse_datacard(args.__dict__.pop("datacard"))
  deltafunctions = datacard.delta_functions()
  deltafunctions.plot(**args.__dict__)
