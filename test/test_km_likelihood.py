"""
Test the Kaplan-Meier likelihood method.
"""

import argparse
import pathlib
import pickle
import warnings

import numpy as np

import roc_picker.datacard
from .utility_testing_functions import Tolerance

warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here / "datacards" / "simple_examples"

def runtest(
  censoring=False,
):  #pylint: disable=too-many-locals, too-many-statements, too-many-branches
  """
  Test the Kaplan-Meier likelihood method.
  """
  if censoring:
    dcfile = datacards / "datacard_example_6.txt"
    reffile = here / "reference" / "km_likelihood_with_censoring.pkl"
  else:
    dcfile = datacards / "datacard_example_5.txt"
    reffile = here / "reference" / "km_likelihood.pkl"

  tolerance: Tolerance = {"atol": 2e-4, "rtol": 2e-4}
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

  array_names = (
    "nominal_probabilities_fullrange",
    "best_probabilities_fullrange",
    "CL_probabilities_fullrange",
    "nominal_probabilities",
    "best_probabilities",
    "CL_probabilities",
    "best_probabilities_binomial",
    "CL_probabilities_binomial",
    "best_probabilities_patient_wise",
    "CL_probabilities_patient_wise",
  )
  to_compare_to_reference = (
    nominal_probabilities_fullrange,
    best_probabilities_fullrange,
    CL_probabilities_fullrange,
    nominal_probabilities,
    best_probabilities,
    CL_probabilities,
    best_probabilities_binomial,
    CL_probabilities_binomial,
    best_probabilities_patient_wise,
    CL_probabilities_patient_wise,
  )
  try:
    with open(reffile, "rb") as f:
      reference = pickle.load(f)
      for name, array, ref in zip(array_names, to_compare_to_reference, reference, strict=True):
        np.testing.assert_allclose(
          array,
          ref,
          **tolerance,
          err_msg=f"Array {name} does not match the reference."
        )
  except:
    with open(here / "test_output" / reffile.name, "wb") as f:
      pickle.dump(to_compare_to_reference, f)
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
