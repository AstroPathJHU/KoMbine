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
    dcfile_fixed = datacards / "fixed_km_censoring.txt"
    # This is the same datacard as poisson_ratio_km_censoring, but with
    # the observable type set to fixed instead of poisson_ratio.
    reffile = here / "reference" / "km_likelihood_with_censoring.json"
  else:
    dcfile = datacards / "poisson_ratio_km.txt"
    dcfile_fixed = None
    reffile = here / "reference" / "km_likelihood.json"

  tolerance: Tolerance = {"atol": 2e-4, "rtol": 2e-4}

  # Calculate precision for JSON output based on rtol
  rtol_value = tolerance["rtol"]
  # Handle case where rtol is 0 to avoid math domain error
  if rtol_value > 0:
    json_precision = int(abs(math.log10(rtol_value))) + 1
  else:
    json_precision = 4 # Default precision if rtol is 0 or undefined

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
  kml2 = datacard.km_likelihood(parameter_min=0.2, parameter_max=0.8, endpoint_epsilon=1e-4)
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
      best_probabilities_binomial,
      **tolerance
    )
  except AssertionError:
    pass
  else:
    raise AssertionError(
      "Best probabilities don't have to be the same "
      "with only the binomial penalty and with the full likelihood "
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

  try:
    np.testing.assert_allclose(
      best_probabilities_patient_wise,
      best_probabilities,
      **tolerance
    )
  except AssertionError:
    pass
  else:
    raise AssertionError(
      "Best probabilities don't have to be the same "
      "with only the patient-wise penalty and with the full likelihood "
      "and in this case they are not."
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

  try:
    np.testing.assert_allclose(
      best_probabilities_patient_wise,
      best_probabilities_binomial,
      **tolerance
    )
  except AssertionError:
    pass
  else:
    raise AssertionError(
      "Best probabilities don't have to be the same "
      "with only the binomial penalty and with the patient-wise penalty "
      "and in this case they are not."
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

  if dcfile_fixed is not None:
    datacard_fixed = roc_picker.datacard.Datacard.parse_datacard(dcfile_fixed)
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

    kml3_fixed = datacard_fixed.km_likelihood(
      parameter_min=0.25,
      parameter_max=0.75,
      endpoint_epsilon=1e-4
    )
    nominal_probabilities_noboundary_fixed = kml3_fixed.nominalkm.survival_probabilities(
      times_for_plot=times_for_plot,
    )
    (
      best_probabilities_noboundary_fixed,
      CL_probabilities_noboundary_fixed,
    ) = kml3_fixed.survival_probabilities_likelihood(
      CLs=CLs,
      times_for_plot=times_for_plot,
    )

    np.testing.assert_allclose(
      nominal_probabilities_noboundary,
      nominal_probabilities_noboundary_fixed,
      **tolerance,
    )
    np.testing.assert_allclose(
      best_probabilities_noboundary_fixed,
      best_probabilities_noboundary_binomial,
      **tolerance,
    )
    np.testing.assert_allclose(
      CL_probabilities_noboundary_fixed,
      CL_probabilities_noboundary_binomial,
      **tolerance,
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
  }

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
        **tolerance,
        err_msg=f"Array '{name}' does not match the reference."
      )
  except Exception:
    with open(here / "test_output" / reffile.name, "w", encoding="utf-8") as f:
      # Convert NumPy arrays to lists and format floats before dumping as a dictionary,
      # ensuring sorted keys and reduced indentation.
      formatted_data_for_json = {
        k: format_value_for_json(v.tolist(), json_precision)
        for k, v in ordered_array_data.items()
      }
      json.dump(
        formatted_data_for_json,
        f,
        indent=2, # Reduced indent to 2 spaces
        sort_keys=True # Ensure deterministic output order
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
