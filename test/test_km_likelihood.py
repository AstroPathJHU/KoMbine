"""
Test the Kaplan-Meier likelihood method.
"""

import pathlib
import pickle
import warnings

import numpy as np

import roc_picker.datacard
from .utility_testing_functions import Tolerance

warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here / "datacards" / "simple_examples"

def main():  #pylint: disable=too-many-locals
  """
  Test the Kaplan-Meier likelihood method.
  """
  tolerance: Tolerance = {"atol": 1e-5, "rtol": 1e-5}
  datacard = roc_picker.datacard.Datacard.parse_datacard(datacards / "datacard_example_5.txt")

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
  kml2 = datacard.km_likelihood(parameter_min=0.2, parameter_max=0.8)
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
      "with and without parameter limits (and in this case they are not)."
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
  )
  try:
    with open(here / "reference" / "km_likelihood.pkl", "rb") as f:
      reference = pickle.load(f)
      for name, array, ref in zip(array_names, to_compare_to_reference, reference):
        np.testing.assert_allclose(
          array,
          ref,
          **tolerance,
          err_msg=f"Array {name} does not match the reference."
        )
  except:
    with open(here / "test_output" / "km_likelihood.pkl", "wb") as f:
      pickle.dump(to_compare_to_reference, f)
    raise

if __name__ == "__main__":
  main()
