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


def test_discrete_class_two_group_penalty_uses_class_bins():
  """
  Two-group NLL penalties must compare class bins, not the integer cut.

  Evaluating NLL only at threshold=1 would give class-1 patients a zero
  penalty for both groups, so KoMbine could reassign them for free.
  """
  datacard = kombine.datacard.Datacard.parse_datacard(
    pathlib.Path(__file__).parent / "datacards" / "simple_examples"
    / "discrete_classes_hr_example_small.txt"
  )
  hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=1.0,
    parameter_min=-np.inf,
    parameter_max=np.inf,
  )
  penalties = hr_calc.nll_penalty_for_patient_in_range
  flip_cost = -math.log(0.05) + math.log(0.95)
  for i, patient in enumerate(hr_calc.all_patients):
    observed_high = patient.observed_parameter >= 1.0
    if observed_high:
      np.testing.assert_allclose(penalties[i, 0], flip_cost, rtol=1e-10)
      np.testing.assert_allclose(penalties[i, 1], -flip_cost, rtol=1e-10)
    else:
      np.testing.assert_allclose(penalties[i, 0], -flip_cost, rtol=1e-10)
      np.testing.assert_allclose(penalties[i, 1], flip_cost, rtol=1e-10)


def test_discrete_class_hazard_ratio_stays_near_hard_labels():
  """
  At e=0.05, KoMbine's HR should stay near the hard-label Cox MLE.
  """
  datacard = kombine.datacard.Datacard.parse_datacard(
    pathlib.Path(__file__).parent / "datacards" / "simple_examples"
    / "discrete_classes_hr_example_small.txt"
  )
  hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=1.0,
    parameter_min=-np.inf,
    parameter_max=np.inf,
  )
  hard_hr, _, _, _ = hr_calc.hazard_ratio_confidence_interval(
    cox_only=True,
    confidence_level=0.95,
    hazard_ratio_min=0.01,
    hazard_ratio_max=100.0,
  )
  kombine_hr, lower, upper, _ = hr_calc.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.95,
    hazard_ratio_min=0.01,
    hazard_ratio_max=100.0,
  )
  assert 0.5 < kombine_hr < 10.0, (
    f"Expected a finite HR near the hard-label fit, got {kombine_hr}"
  )
  assert abs(np.log(kombine_hr) - np.log(hard_hr)) < 0.5, (
    f"KoMbine HR {kombine_hr:.3f} drifted from hard-label {hard_hr:.3f}"
  )
  assert lower < kombine_hr < upper


if __name__ == "__main__":
  test_discrete_class_probability_in_range()
  test_discrete_class_observed_parameter_tie_break()
  test_discrete_class_patient_nll()
  test_discrete_class_datacard_parsing()
  test_discrete_class_two_group_penalty_uses_class_bins()
  test_discrete_class_hazard_ratio_stays_near_hard_labels()
  print("[SUCCESS] Discrete class observable tests passed")
