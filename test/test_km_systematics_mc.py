"""
Test the discrete module, and generate the figures for that section of the documentation.
"""

import pathlib
import pickle
import warnings

import numpy as np
import scipy.special

import roc_picker.datacard
from .utility_testing_functions import Tolerance

warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here/"datacards"/"simple_examples"

def main():
  """
  Test the Kaplan-Meier systematics module.
  """
  sigmas = [-2, -1, 0, 1, 2]
  quantiles = [(1 + scipy.special.erf(nsigma/np.sqrt(2))) / 2 for nsigma in sigmas]

  datacard = roc_picker.datacard.Datacard.parse_datacard(datacards/"datacard_example_4.txt")

  #First try with no parameter limits
  #All the generated KM distributions should be the same
  #as the nominal distribution
  kmdistributions = datacard.systematics_mc_km()
  kmcollection = kmdistributions.generate(size=1000, random_state=123456)
  km_quantiles = kmcollection.survival_probabilities_quantiles(quantiles)
  nominal = kmcollection.nominalkm.survival_probabilities()
  for km in km_quantiles:
    np.testing.assert_array_equal(km, nominal)

  #Now try with parameter limits
  #The quantiles are now nontrivial but should be the same as the reference
  kmdistributions = datacard.systematics_mc_km(parameter_min=70, parameter_max=np.inf)
  kmcollection = kmdistributions.generate(size=1000, random_state=123456)
  km_quantiles = kmcollection.survival_probabilities_quantiles(quantiles)
  #test that (x, y) are the same as the reference
  #this serves to test that the plotting part of the code works correctly
  km_quantiles_xy = np.array([
    kmcollection._points_for_plot(kmcollection.times_for_plot, km)
    for km in km_quantiles
  ])


  tolerance: Tolerance = {"atol": 1e-6, "rtol": 1e-6}

  try:
    with open(here/"reference"/"km_systematics_mc.pkl", "rb") as f:
      ref = pickle.load(f)
      np.testing.assert_allclose(km_quantiles_xy, ref, **tolerance)
  except:
    with open(here/"test_output"/"km_systematics_mc.pkl", "wb") as f:
      pickle.dump(km_quantiles_xy, f)
    raise

if __name__ == "__main__":
  main()
