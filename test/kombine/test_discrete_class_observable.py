"""
Tests for DiscreteClassObservable.
"""

import math
import pathlib
import tempfile

import numpy as np

import kombine.datacard
from kombine.datacard import DiscreteClassObservable


def test_discrete_class_probability_in_range():
  """
  Discrete class observables should sum probabilities in ranges.
  """
  obs = DiscreteClassObservable(class_probs=[0.1, 0.7, 0.2])
  assert np.isclose(obs.probability_in_range(0.0, 1.0), 0.1)
  assert np.isclose(obs.probability_in_range(1.0, 3.0), 0.9)
  assert np.isclose(obs.probability_in_range(1.5, 2.5), 0.2)


def test_discrete_class_observed_parameter_tie_break():
  """
  Argmax should pick the lowest index on ties.
  """
  obs = DiscreteClassObservable(class_probs=[0.5, 0.5])
  assert obs.observed_parameter() == 0.0


def test_discrete_class_patient_nll():
  """
  NLL should be -log(p_k) for class intervals.
  """
  obs = DiscreteClassObservable(class_probs=[0.2, 0.8])
  nll = obs.patient_nll(time=1.0, censored=False, systematics=None)
  assert np.isclose(nll.parameter(0.1), -math.log(0.2))
  assert np.isclose(nll.parameter(1.9), -math.log(0.8))
  assert np.isinf(nll.parameter(2.0))


def test_discrete_class_datacard_parsing():
  """
  Datacard parsing should assemble prob lines into discrete classes.
  """
  content = """
observable_type discrete_classes
response responder non-responder responder
prob0 0.2 0.7 0.5
prob1 0.8 0.3 0.5
""".strip()
  with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
    f.write(content)
    temp_path = pathlib.Path(f.name)

  datacard = kombine.datacard.Datacard.parse_datacard(temp_path)
  temp_path.unlink()

  patients = datacard.patients
  assert isinstance(patients[0].observable, DiscreteClassObservable)
  assert patients[0].observable.observed_parameter() == 1.0
  assert isinstance(patients[1].observable, DiscreteClassObservable)
  assert np.isclose(patients[1].observable.probability_in_range(0.0, 1.0), 0.7)
  assert isinstance(patients[2].observable, DiscreteClassObservable)
  assert np.isclose(patients[2].observable.probability_in_range(1.0, 2.0), 0.5)
