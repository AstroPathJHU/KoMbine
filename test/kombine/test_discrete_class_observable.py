"""
Tests for DiscreteClassObservable.
"""

import math
import pathlib
import tempfile

import numpy as np
import scipy.stats

import kombine.datacard
from kombine.datacard import DiscreteClassObservable
from kombine.kaplan_meier_p_value_MINLP import MINLPforKMPValue


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
      np.testing.assert_allclose(penalties[i, 1], 0.0, atol=1e-12)
    else:
      np.testing.assert_allclose(penalties[i, 0], 0.0, atol=1e-12)
      np.testing.assert_allclose(penalties[i, 1], flip_cost, rtol=1e-10)
  assert np.all(np.isposinf(hr_calc.nll_penalty_for_unassigned))


def test_two_group_neither_bin_uses_tail_nll():
  """
  Exclusion must cost the tail NLL, not a group-to-group flip.

  With three classes and finite bounds, neither is class 2. That cost
  equals m_neither - m_best, which need not be flip_cost or 2*flip_cost.
  """
  obs_low = DiscreteClassObservable(class_probs=[0.90, 0.05, 0.05])
  patient = obs_low.patient_nll(time=1.0, censored=False, systematics=None)
  calc = MINLPforKMPValue(
    [patient],
    parameter_min=0.0,
    parameter_threshold=1.0,
    parameter_max=2.0,
  )
  m_low, m_high, m_neither = calc.assignment_nlls[0]
  np.testing.assert_allclose(m_low, -math.log(0.90), rtol=1e-10)
  np.testing.assert_allclose(m_high, -math.log(0.05), rtol=1e-10)
  np.testing.assert_allclose(m_neither, -math.log(0.05), rtol=1e-10)
  np.testing.assert_allclose(calc.nll_penalty_for_patient_in_range[0, 0], 0.0, atol=1e-12)
  np.testing.assert_allclose(
    calc.nll_penalty_for_patient_in_range[0, 1],
    m_high - m_low,
    rtol=1e-10,
  )
  np.testing.assert_allclose(
    calc.nll_penalty_for_unassigned[0],
    m_neither - m_low,
    rtol=1e-10,
  )
  flip_to_high = m_high - m_low
  assert not np.isclose(calc.nll_penalty_for_unassigned[0], 2.0 * flip_to_high)


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


def test_permutation_pvalue_agrees_with_chi2_at_small_e():
  """
  At e=0.05 assignments do not move, so permutation p stays large.
  """
  datacard = kombine.datacard.Datacard.parse_datacard(
    pathlib.Path(__file__).parent / "datacards" / "simple_examples"
    / "discrete_classes_hr_example_small.txt"
  )
  calc = datacard.km_p_value(
    parameter_threshold=1.0,
    parameter_min=-np.inf,
    parameter_max=np.inf,
  )
  p_perm, result_null, result_alt = calc.solve_and_pvalue(
    cox_only=False,
    n_permutations=19,
    rng=0,
  )
  chi2_p = scipy.stats.chi2.sf(result_null.x - result_alt.x, 1)
  assert chi2_p > 0.1, f"Expected a large chi2 p at e=0.05, got {chi2_p}"
  assert p_perm > 0.1, f"Expected a large permutation p at e=0.05, got {p_perm}"


def test_permutation_pvalue_not_tiny_when_chi2_is():
  """
  At large e the profile LRT chi2 p-value is tiny, but the permutation
  p-value must not treat reassignment search as extra evidence.
  """
  datacard = kombine.datacard.Datacard.parse_datacard(
    pathlib.Path(__file__).parent / "datacards" / "simple_examples"
    / "discrete_classes_hr_example_very_large.txt"
  )
  calc = datacard.km_p_value(
    parameter_threshold=1.0,
    parameter_min=-np.inf,
    parameter_max=np.inf,
  )
  p_perm, result_null, result_alt = calc.solve_and_pvalue(
    cox_only=False,
    n_permutations=39,
    rng=0,
  )
  chi2_p = scipy.stats.chi2.sf(result_null.x - result_alt.x, 1)
  assert chi2_p < 0.01, f"Expected a tiny chi2 p at e=0.40, got {chi2_p}"
  assert p_perm > 0.05, f"Permutation p should not be tiny, got {p_perm}"
  p_cox, _, _ = calc.solve_and_pvalue(cox_only=True)
  assert p_cox > 0.1, f"Cox-only chi2 p should stay near the hard-label LRT, got {p_cox}"


def test_solve_and_pvalue_rejects_negative_permutations():
  """
  n_permutations must be non-negative.
  """
  datacard = kombine.datacard.Datacard.parse_datacard(
    pathlib.Path(__file__).parent / "datacards" / "simple_examples"
    / "discrete_classes_hr_example_small.txt"
  )
  calc = datacard.km_p_value(
    parameter_threshold=1.0,
    parameter_min=-np.inf,
    parameter_max=np.inf,
  )
  try:
    calc.solve_and_pvalue(n_permutations=-1)
  except ValueError as exc:
    assert "n_permutations" in str(exc)
  else:
    raise AssertionError("Expected ValueError for n_permutations=-1")


if __name__ == "__main__":
  test_discrete_class_probability_in_range()
  test_discrete_class_observed_parameter_tie_break()
  test_discrete_class_patient_nll()
  test_discrete_class_datacard_parsing()
  test_discrete_class_two_group_penalty_uses_class_bins()
  test_two_group_neither_bin_uses_tail_nll()
  test_discrete_class_hazard_ratio_stays_near_hard_labels()
  test_permutation_pvalue_agrees_with_chi2_at_small_e()
  test_permutation_pvalue_not_tiny_when_chi2_is()
  test_solve_and_pvalue_rejects_negative_permutations()
  print("[SUCCESS] Discrete class observable tests passed")
