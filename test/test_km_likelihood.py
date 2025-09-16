"""
Test the Kaplan-Meier likelihood method.
"""

import argparse
import pathlib
import json
import math
import warnings

import numpy as np

import roc_picker.datacard
from roc_picker.kaplan_meier_MINLP import KaplanMeierPatientNLL, MINLPForKM
from .utility_testing_functions import format_value_for_json, Tolerance

warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here / "datacards" / "simple_examples"

def runtest(
  censoring=False,
): #pylint: disable=too-many-locals, too-many-statements, too-many-branches
  """
  Test the Kaplan-Meier likelihood method.
  """

  if censoring:
    dcfile = datacards / "poisson_ratio_km_censoring.txt"
    # These are the same datacard as poisson_ratio_km_censoring, but with
    # the observable type set to other types instead of poisson_ratio.
    alt_datacards = {
      "fixed": datacards / "fixed_km_censoring.txt",
      "density": datacards / "poisson_density_km_censoring.txt",
      "count": datacards / "poisson_km_censoring.txt",
      "systematic": datacards / "poisson_ratio_km_censoring_systematic.txt",
      "systematic_small": datacards / "poisson_ratio_km_censoring_systematic_small.txt",
    }
    reffile = here / "reference" / "km_likelihood_with_censoring.json"
  else:
    dcfile = datacards / "poisson_ratio_km.txt"
    alt_datacards = None
    reffile = here / "reference" / "km_likelihood.json"

  tolerance: Tolerance = {"atol": 2e-4, "rtol": 2e-4}
  nominal_hazard_ratio_tolerance: Tolerance = {"atol": 2e-3, "rtol": 2e-3}

  # Calculate precision for JSON output based on rtol
  rtol_value = tolerance["rtol"]
  # Handle case where rtol is 0 to avoid math domain error
  if rtol_value > 0:
    json_precision = int(abs(math.log10(rtol_value))) + 1
  else:
    json_precision = 4 # Default precision if rtol is 0 or undefined

  # Higher precision for patient-wise p-values to avoid rounding small values to 0
  patient_wise_precision = 8

  datacard = roc_picker.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)
  times_for_plot = kml.times_for_plot

  # Test with no parameter limits
  # Only the binomial distribution enters the likelihood
  nominal_probabilities_fullrange = kml.nominalkm.survival_probabilities(
    times_for_plot=times_for_plot
  )
  CLs = [0.68, 0.95]
  best_probabilities_fullrange, CL_probabilities_fullrange = kml.survival_probabilities_likelihood(
    CLs=CLs,
    times_for_plot=times_for_plot,
  )
  np.testing.assert_allclose(
    best_probabilities_fullrange,
    nominal_probabilities_fullrange,
    **tolerance
  )

  # Now test with parameter limits
  kml2 = datacard.km_likelihood(parameter_min=0.19, parameter_max=0.79, endpoint_epsilon=1e-4)
  nominal_probabilities = kml2.nominalkm.survival_probabilities(times_for_plot=times_for_plot)
  best_probabilities, CL_probabilities = kml2.survival_probabilities_likelihood(
    CLs=CLs,
    times_for_plot=times_for_plot,
  )
  best_probabilities_binomial, CL_probabilities_binomial = kml2.survival_probabilities_likelihood(
    CLs=CLs,
    times_for_plot=times_for_plot,
    binomial_only=True,
  )
  (
    best_probabilities_patient_wise, CL_probabilities_patient_wise
  ) = kml2.survival_probabilities_likelihood(
    CLs=CLs,
    times_for_plot=times_for_plot,
    patient_wise_only=True,
  )

  kml2_no_collapse = datacard.km_likelihood(
    parameter_min=0.19,
    parameter_max=0.79,
    endpoint_epsilon=1e-4,
    collapse_consecutive_deaths=False
  )
  (
    best_probabilities_no_collapse, CL_probabilities_no_collapse
  ) = kml2_no_collapse.survival_probabilities_likelihood(
    CLs=CLs,
    times_for_plot=times_for_plot,
  )

  np.testing.assert_allclose(
    best_probabilities_binomial,
    nominal_probabilities,
    **tolerance
  )

  try:
    np.testing.assert_allclose(
      nominal_probabilities_fullrange,
      nominal_probabilities,
      **tolerance,
    )
  except AssertionError:
    pass
  else:
    raise AssertionError(
      "Nominal probabilities should not be the same "
      "with and without parameter limits."
    )

  try:
    np.testing.assert_allclose(
      best_probabilities,
      nominal_probabilities,
      **tolerance
    )
  except AssertionError:
    pass
  else:
    raise AssertionError(
      "Best probabilities with both penalties "
      "don't have to be the same as the nominal probabilities. "
      "(and in this case they are not)."
    )


  try:
    np.testing.assert_allclose(
      CL_probabilities,
      CL_probabilities_binomial,
      **tolerance
    )
  except AssertionError:
    pass
  else:
    raise AssertionError(
      "CL probabilities shouldn't be the same "
      "with only the binomial penalty and with the full likelihood."
    )

  np.testing.assert_allclose(
    best_probabilities_patient_wise,
    nominal_probabilities,
    **tolerance
  )

  try:
    np.testing.assert_allclose(
      CL_probabilities_patient_wise,
      CL_probabilities,
      **tolerance
    )
  except AssertionError:
    pass
  else:
    raise AssertionError(
      "CL probabilities shouldn't be the same "
      "with only the patient-wise penalty and with the full likelihood."
    )

  np.testing.assert_allclose(
    best_probabilities_binomial,
    nominal_probabilities,
    **tolerance
  )

  try:
    np.testing.assert_allclose(
      CL_probabilities_patient_wise,
      CL_probabilities_binomial,
      **tolerance
    )
  except AssertionError:
    pass
  else:
    raise AssertionError(
      "CL probabilities shouldn't be the same "
      "with only the binomial penalty and with the patient-wise penalty."
    )
  
  np.testing.assert_allclose(
    best_probabilities_no_collapse,
    best_probabilities,
    **tolerance,
  )
  np.testing.assert_allclose(
    CL_probabilities_no_collapse,
    CL_probabilities,
    **tolerance,
  )

  # Calculate p-value using the likelihood method
  # Use the middle of the parameter range as threshold for the p-value calculation
  parameter_min = 0.0001
  parameter_max = 100
  parameter_threshold = 0.45

  km_p_value_minlp_breslow = datacard.km_p_value(
    parameter_min=parameter_min,
    parameter_threshold=parameter_threshold,
    parameter_max=parameter_max,
    tie_handling="breslow",
  )

  nominal_hazard_ratio_breslow = km_p_value_minlp_breslow.nominal_hazard_ratio
  p_value_breslow, _, _ = km_p_value_minlp_breslow.solve_and_pvalue()
  p_value_cox_breslow, _, _ = km_p_value_minlp_breslow.solve_and_pvalue(cox_only=True)

  try:
    _, _, _ = km_p_value_minlp_breslow.solve_and_pvalue(patient_wise_only=True)
  except NotImplementedError:
    pass
  else:
    raise AssertionError(
      "Patient-wise p-value is not implemented "
      "and should have raised a NotImplementedError"
    )

  # Test mutual exclusion of options
  try:
    km_p_value_minlp_breslow.solve_and_pvalue(cox_only=True, patient_wise_only=True)
    raise AssertionError(
      "Should have raised ValueError for both cox_only and patient_wise_only being True"
    )
  except ValueError as e:
    if "cox_only and patient_wise_only cannot both be True" not in str(e):
      raise AssertionError("Wrong error message") from e

  # Calculate p-value using the conventional logrank method for comparison
  p_value_logrank = datacard.km_p_value_logrank(
    parameter_threshold=parameter_threshold,
    parameter_min=parameter_min,
    parameter_max=parameter_max,
    cox_only=True,
  )

  # Test that logrank method requires cox_only=True
  try:
    datacard.km_p_value_logrank(
      parameter_threshold=parameter_threshold,
      parameter_min=parameter_min,
      parameter_max=parameter_max,
      cox_only=False,
    )
    raise AssertionError("Should have raised ValueError for cox_only=False")
  except ValueError as e:
    if "only supports cox_only=True" not in str(e):
      raise AssertionError("Wrong error message for cox_only=False") from e

  alt_results = {}
  if alt_datacards is not None:
    kml3 = datacard.km_likelihood(
      parameter_min=0.25,
      parameter_max=0.75,
      endpoint_epsilon=1e-4,
    )
    nominal_probabilities_noboundary = kml3.nominalkm.survival_probabilities(
      times_for_plot=times_for_plot,
    )
    (
      best_probabilities_noboundary_binomial,
      CL_probabilities_noboundary_binomial,
    ) = kml3.survival_probabilities_likelihood(
      CLs=CLs,
      times_for_plot=times_for_plot,
      binomial_only=True,
    )
    (
      best_probabilities_noboundary,
      CL_probabilities_noboundary
    ) = kml3.survival_probabilities_likelihood(
      CLs=CLs,
      times_for_plot=times_for_plot,
    )

    for name, dcfile_alt in alt_datacards.items():
      factor = {
        "fixed": 1,
        "density": 1,
        "count": 100,
        "systematic": 1,
        "systematic_small": 1,
      }[name]
      datacard_alt = roc_picker.datacard.Datacard.parse_datacard(dcfile_alt)
      kml3_alt = datacard_alt.km_likelihood(
        parameter_min=0.25*factor,
        parameter_max=0.75*factor,
        endpoint_epsilon=1e-4,
      )
      nominal_probabilities_noboundary_alt = kml3_alt.nominalkm.survival_probabilities(
        times_for_plot=times_for_plot,
      )
      (
        best_probabilities_noboundary_alt,
        CL_probabilities_noboundary_alt,
      ) = kml3_alt.survival_probabilities_likelihood(
        CLs=CLs,
        times_for_plot=times_for_plot,
      )

      alt_results[name] = {
        "nominal_probabilities": nominal_probabilities_noboundary_alt,
        "best_probabilities": best_probabilities_noboundary_alt,
        "CL_probabilities": CL_probabilities_noboundary_alt,
      }

      np.testing.assert_allclose(
        nominal_probabilities_noboundary,
        nominal_probabilities_noboundary_alt,
        **tolerance,
      )
      if name == "fixed":
        np.testing.assert_allclose(
          best_probabilities_noboundary_alt,
          best_probabilities_noboundary_binomial,
          **tolerance,
        )
        np.testing.assert_allclose(
          CL_probabilities_noboundary_alt,
          CL_probabilities_noboundary_binomial,
          **tolerance,
        )
      elif name == "count":
        best_probabilities_noboundary_density = alt_results["density"]["best_probabilities"]
        CL_probabilities_noboundary_density = alt_results["density"]["CL_probabilities"]
        np.testing.assert_allclose(
          best_probabilities_noboundary_density,
          best_probabilities_noboundary_alt,
          **tolerance,
        )
        np.testing.assert_allclose(
          CL_probabilities_noboundary_density,
          CL_probabilities_noboundary_alt,
          **tolerance,
        )
      elif name == "systematic_small":
        np.testing.assert_allclose(
          best_probabilities_noboundary,
          best_probabilities_noboundary_alt,
          **tolerance,
        )
        np.testing.assert_allclose(
          CL_probabilities_noboundary,
          CL_probabilities_noboundary_alt,
          **tolerance,
        )
      elif name == "systematic":
        try:
          np.testing.assert_allclose(
            CL_probabilities_noboundary,
            CL_probabilities_noboundary_alt,
            **tolerance,
          )
        except AssertionError:
          pass
        else:
          raise AssertionError(
            "Probabilities should not be unchanged when applying a large systematic uncertainty"
          )

  # Define the arrays to be compared in the desired order
  ordered_array_data = {
    "nominal_probabilities_fullrange": nominal_probabilities_fullrange,
    "best_probabilities_fullrange": best_probabilities_fullrange,
    "CL_probabilities_fullrange": CL_probabilities_fullrange,
    "nominal_probabilities": nominal_probabilities,
    "best_probabilities": best_probabilities,
    "CL_probabilities": CL_probabilities,
    "best_probabilities_binomial": best_probabilities_binomial,
    "CL_probabilities_binomial": CL_probabilities_binomial,
    "best_probabilities_patient_wise": best_probabilities_patient_wise,
    "CL_probabilities_patient_wise": CL_probabilities_patient_wise,
    "nominal_hazard_ratio_breslow": np.array([nominal_hazard_ratio_breslow]),
    "p_value_breslow": np.array([p_value_breslow]),
    "p_value_cox_breslow": np.array([p_value_cox_breslow]),
    #"p_value_patient_wise_breslow": np.array([p_value_patient_wise_breslow]),
    "p_value_logrank": np.array([p_value_logrank]),
  }
  for name, alt_data in alt_results.items():
    ordered_array_data[f"nominal_probabilities_{name}"] = alt_data["nominal_probabilities"]
    ordered_array_data[f"best_probabilities_{name}"] = alt_data["best_probabilities"]
    ordered_array_data[f"CL_probabilities_{name}"] = alt_data["CL_probabilities"]

  try:
    with open(reffile, "r", encoding="utf-8") as f:
      loaded_data = json.load(f)
      reference_data = {k: np.asarray(v) for k, v in loaded_data.items()}

    # Check for missing or extra keys
    testing_keys = set(ordered_array_data.keys())
    reference_keys = set(reference_data.keys())

    missing_keys = testing_keys - reference_keys
    extra_keys = reference_keys - testing_keys

    if missing_keys:
      raise AssertionError(f"Keys missing in reference file: {', '.join(sorted(missing_keys))}")
    if extra_keys:
      raise AssertionError(f"Extra keys found in reference file: {', '.join(sorted(extra_keys))}")

    # Compare arrays in the defined order
    for name, array in ordered_array_data.items():
      ref = reference_data[name]
      np.testing.assert_allclose(
        array,
        ref,
        **(nominal_hazard_ratio_tolerance if "nominal_hazard_ratio" in name else tolerance),
        err_msg=f"Array '{name}' does not match the reference."
      )
  except Exception:
    with open(here / "test_output" / reffile.name, "w", encoding="utf-8") as f:
      formatted_data_for_json = {}
      for k, v in ordered_array_data.items():
        # Use higher precision for patient-wise p-values
        precision_to_use = patient_wise_precision if "patient_wise" in k else json_precision
        formatted_data_for_json[k] = format_value_for_json(v.tolist(), precision_to_use)

      json.dump(
        formatted_data_for_json,
        f,
        indent=2,
        sort_keys=True,
      )
    raise

def test_times_to_consider_collapse_logic(): # pylint: disable=too-many-locals
  """
  Comprehensive tests for the times_to_consider collapsing logic.
  Tests various scenarios to ensure the collapsing algorithm works correctly.
  """
  # Test 1: Simple consecutive deaths without censoring
  # Should collapse all consecutive deaths

  # Create test patients with consecutive deaths
  # pylint: disable=line-too-long
  patients = [
    KaplanMeierPatientNLL(time=1.0, censored=False, parameter_nll=lambda x: 0, observed_parameter=0.5),
    KaplanMeierPatientNLL(time=2.0, censored=False, parameter_nll=lambda x: 0, observed_parameter=0.5),
    KaplanMeierPatientNLL(time=3.0, censored=False, parameter_nll=lambda x: 0, observed_parameter=0.5),
    KaplanMeierPatientNLL(time=4.0, censored=False, parameter_nll=lambda x: 0, observed_parameter=0.5),
  ]
  # pylint: enable=line-too-long

  # Test with collapse=True
  minlp_collapse = MINLPForKM(
    patients,
    parameter_min=0.1,
    parameter_max=0.9,
    time_point=5.0,
    collapse_consecutive_deaths=True
  )
  times_collapsed = minlp_collapse.times_to_consider

  # Test with collapse=False
  minlp_no_collapse = MINLPForKM(
    patients,
    parameter_min=0.1,
    parameter_max=0.9,
    time_point=5.0,
    collapse_consecutive_deaths=False
  )
  times_no_collapse = minlp_no_collapse.times_to_consider

  # Should have fewer times with collapse
  assert len(times_collapsed) <= len(times_no_collapse)
  # For consecutive deaths, should collapse to just the final time
  expected_collapsed = np.array([5.0])  # Only the time_point
  np.testing.assert_allclose(times_collapsed, expected_collapsed, rtol=1e-10)

  # Test 2: Deaths with intervening censoring
  # pylint: disable=line-too-long
  patients_with_censoring = [
    KaplanMeierPatientNLL(time=1.0, censored=False, parameter_nll=lambda x: 0, observed_parameter=0.5),
    KaplanMeierPatientNLL(time=2.0, censored=False, parameter_nll=lambda x: 0, observed_parameter=0.5),
    KaplanMeierPatientNLL(time=2.5, censored=True, parameter_nll=lambda x: 0, observed_parameter=0.5),  # Censoring between 2 and 3
    KaplanMeierPatientNLL(time=3.0, censored=False, parameter_nll=lambda x: 0, observed_parameter=0.5),
    KaplanMeierPatientNLL(time=4.0, censored=False, parameter_nll=lambda x: 0, observed_parameter=0.5),
  ]
  # pylint: enable=line-too-long

  minlp_censoring_collapse = MINLPForKM(
    patients_with_censoring,
    parameter_min=0.1,
    parameter_max=0.9,
    time_point=5.0,
    collapse_consecutive_deaths=True
  )
  times_censoring_collapsed = minlp_censoring_collapse.times_to_consider

  minlp_censoring_no_collapse = MINLPForKM(
    patients_with_censoring,
    parameter_min=0.1,
    parameter_max=0.9,
    time_point=5.0,
    collapse_consecutive_deaths=False
  )
  times_censoring_no_collapse = minlp_censoring_no_collapse.times_to_consider

  # Should have more times due to censoring preventing collapse
  # Expected: deaths at 1,2 can be collapsed to 2, then censoring at 2.5 prevents
  # collapsing with deaths at 3,4, so we get [2.0, 5.0] (deaths 3,4 collapse with time_point)
  expected_with_censoring = np.array([2.0, 5.0])
  np.testing.assert_allclose(times_censoring_collapsed, expected_with_censoring, rtol=1e-10)

  expected_no_collapse = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  np.testing.assert_allclose(times_censoring_no_collapse, expected_no_collapse, rtol=1e-10)

  # Test 3: Censoring at same time as death (death should happen first per KM convention)
  # pylint: disable=line-too-long
  patients_same_time = [
    KaplanMeierPatientNLL(time=1.0, censored=False, parameter_nll=lambda x: 0, observed_parameter=0.5),
    KaplanMeierPatientNLL(time=2.0, censored=False, parameter_nll=lambda x: 0, observed_parameter=0.5),
    KaplanMeierPatientNLL(time=2.0, censored=True, parameter_nll=lambda x: 0, observed_parameter=0.5),   # Censoring at same time as death
    KaplanMeierPatientNLL(time=3.0, censored=False, parameter_nll=lambda x: 0, observed_parameter=0.5),
  ]
  # pylint: enable=line-too-long

  minlp_same_time_collapse = MINLPForKM(
    patients_same_time,
    parameter_min=0.1,
    parameter_max=0.9,
    time_point=4.0,
    collapse_consecutive_deaths=True
  )
  times_same_time_collapsed = minlp_same_time_collapse.times_to_consider

  # Deaths at 1,2 can be collapsed, but censoring at 2.0 affects subsequent deaths
  # So death at 3 should be separate
  expected_same_time = np.array([2.0, 4.0])  # 1,2 collapse to 2.0; 3 collapses with time_point 4.0
  np.testing.assert_allclose(times_same_time_collapsed, expected_same_time, rtol=1e-10)

  # Additional test: patient_died and patient_still_at_risk logic for collapse
  patient_times = [1, 1, 2, 3, 3, 4, 5, 5, 6, 7]
  patient_censored = [True, False, True, False, False, False, False, True, False, True]
  patients = [
    KaplanMeierPatientNLL(time=t, censored=c, parameter_nll=lambda x: 0, observed_parameter=0.5)
    for t, c in zip(patient_times, patient_censored)
  ]
  minlp = MINLPForKM(
    patients,
    parameter_min=0.1,
    parameter_max=0.9,
    time_point=8.0,
    collapse_consecutive_deaths=True
  )
  # Test cases
  # pylint: disable=line-too-long
  test_cases = [
    (3, [False, False, False, True, True, True, True, True, True, True], [False, False, False, True, True, False, False, False, False, False]),
    (3.5, [False, False, False, True, True, True, True, True, True, True], [False, False, False, True, True, False, False, False, False, False]),
    (4, [False, False, False, True, True, True, True, True, True, True], [False, False, False, True, True, True, False, False, False, False]),
    (4.5, [False, False, False, True, True, True, True, True, True, True], [False, False, False, True, True, True, False, False, False, False]),
    (5, [False, False, False, True, True, True, True, True, True, True], [False, False, False, True, True, True, True, False, False, False]),
  ]
  # pylint: enable=line-too-long
  for t, expected_at_risk, expected_died in test_cases:
    at_risk = minlp.patient_still_at_risk(t).tolist()
    died = minlp.patient_died(t).tolist()
    assert at_risk == expected_at_risk, \
      f"patient_still_at_risk({t}) failed: {at_risk} != {expected_at_risk}"
    assert died == expected_died, f"patient_died({t}) failed: {died} != {expected_died}"

def main(args=None):
  """
  Main function to run the test.
  By default, it runs both with and without censoring, plus additional tests.
  You can specify --censoring or --no-censoring to run only one of them.
  """
  p = argparse.ArgumentParser(
    description="Test the Kaplan-Meier likelihood method."
  )
  g = p.add_mutually_exclusive_group()
  g.add_argument(
    "--censoring",
    action="store_true",
    help="Test with censoring.",
  )
  g.add_argument(
    "--no-censoring",
    action="store_true",
    help="Test without censoring.",
  )
  g.add_argument(
    "--collapse-logic-only",
    action="store_true",
    help="Test only the collapse logic.",
  )
  args = p.parse_args(args)

  if args.collapse_logic_only:
    test_times_to_consider_collapse_logic()
    return

  if not args.censoring and not args.no_censoring:
    args.censoring = args.no_censoring = True

  if args.no_censoring:
    runtest(censoring=False)
  if args.censoring:
    runtest(censoring=True)

  # Run additional tests
  test_times_to_consider_collapse_logic()

if __name__ == "__main__":
  main()
