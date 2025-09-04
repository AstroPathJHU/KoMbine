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
  p_value_binomial_breslow, _, _ = km_p_value_minlp_breslow.solve_and_pvalue(binomial_only=True)

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
    km_p_value_minlp_breslow.solve_and_pvalue(binomial_only=True, patient_wise_only=True)
    raise AssertionError(
      "Should have raised ValueError for both binomial_only and patient_wise_only being True"
    )
  except ValueError as e:
    if "binomial_only and patient_wise_only cannot both be True" not in str(e):
      raise AssertionError("Wrong error message") from e

  # Calculate p-value using the conventional logrank method for comparison
  p_value_logrank = datacard.km_p_value_logrank(
    parameter_threshold=parameter_threshold,
    parameter_min=parameter_min,
    parameter_max=parameter_max,
    binomial_only=True,
  )

  # Test that logrank method requires binomial_only=True
  try:
    datacard.km_p_value_logrank(
      parameter_threshold=parameter_threshold,
      parameter_min=parameter_min,
      parameter_max=parameter_max,
      binomial_only=False,
    )
    raise AssertionError("Should have raised ValueError for binomial_only=False")
  except ValueError as e:
    if "only supports binomial_only=True" not in str(e):
      raise AssertionError("Wrong error message for binomial_only=False") from e

  alt_results = {}
  if alt_datacards is not None:
    kml3 = datacard.km_likelihood(
      parameter_min=0.25,
      parameter_max=0.75,
      endpoint_epsilon=1e-4
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
        endpoint_epsilon=1e-4
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
    "p_value_binomial_breslow": np.array([p_value_binomial_breslow]),
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

def main(args=None):
  """
  Main function to run the test.
  By default, it runs both with and without censoring.
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
  args = p.parse_args(args)

  if not args.censoring and not args.no_censoring:
    args.censoring = args.no_censoring = True

  if args.no_censoring:
    runtest(censoring=False)
  if args.censoring:
    runtest(censoring=True)

if __name__ == "__main__":
  main()
