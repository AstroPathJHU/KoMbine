"""
Test the discrete module, and generate the figures for that section of the documentation.
"""

import pathlib
import pickle
import typing
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.special

import roc_picker.datacard
from roc_picker.systematics_mc import AUC
warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here/"datacards"/"lung"

def main():
  """
  Test the systematics_mc module.
  """
  datacard = roc_picker.datacard.Datacard.parse_datacard(datacards/"datacard_cells.txt")
  rocdistributions = datacard.systematics_mc_roc(flip_sign=False)
  rocs = rocdistributions.generate(
    size=100,
    random_state=123456
  )

  sigmas = [-2, -1, 0, 1, 2]
  quantiles = [(1 + scipy.special.erf(nsigma/np.sqrt(2))) / 2 for nsigma in sigmas]

  x_quantiles = {}
  y_quantiles = {}
  (x_quantiles["m95"], x_quantiles["m68"], x_quantiles["nominal"], x_quantiles["p68"], x_quantiles["p95"]), (y_quantiles["m95"], y_quantiles["m68"], y_quantiles["nominal"], y_quantiles["p68"], y_quantiles["p95"]) = rocs.roc_quantiles(quantiles)

  del rocdistributions, rocs

  rocdistributions_flip = datacard.systematics_mc_roc(flip_sign=True)
  rocs_flip = rocdistributions_flip.generate(
    size=100,
    random_state=123456
  )

  x_quantiles_flip = {}
  y_quantiles_flip = {}
  (x_quantiles_flip["m95"], x_quantiles_flip["m68"], x_quantiles_flip["nominal"], x_quantiles_flip["p68"], x_quantiles_flip["p95"]), (y_quantiles_flip["m95"], y_quantiles_flip["m68"], y_quantiles_flip["nominal"], y_quantiles_flip["p68"], y_quantiles_flip["p95"]) = rocs_flip.roc_quantiles(quantiles)

  AUCs = {
    k: AUC(x_quantiles[k], y_quantiles[k])
    for k in x_quantiles.keys()
  }
  AUCs_flip = {
    k: AUC(x_quantiles_flip[k], y_quantiles_flip[k])
    for k in x_quantiles_flip.keys()
  }

  class Tolerance(typing.TypedDict):
    "typed class for atol and rtol to pass to np.testing.assert_allclose"
    rtol: float
    atol: float
  tolerance: Tolerance = {"atol": 1e-6, "rtol": 1e-6}

  for k in sorted(set(x_quantiles) | set(x_quantiles_flip)):
    x = x_quantiles[k]
    y = y_quantiles[k]
    auc = AUCs[k]
    flipk = {
      "nominal": "nominal",
      "p68": "m68",
      "p95": "m95",
      "m68": "p68",
      "m95": "p95",
    }[k]
    x_flip = x_quantiles_flip[flipk]
    y_flip = y_quantiles_flip[flipk]
    auc_flip = AUCs_flip[flipk]
    try:
      np.testing.assert_allclose(
        np.array([x, y]),
        1-np.array([x_flip, y_flip])[:,::-1],
        **tolerance,
      )
    except AssertionError:
      plt.figure()
      plt.plot(x, y, label="unflipped")
      plt.plot(1-x_flip, 1-y_flip, label="flipped twice")
      plt.legend()
      plt.savefig(here/"test_output"/f"roc_{k}.png")
      plt.close()
      raise
    np.testing.assert_allclose(auc, 1-auc_flip, **tolerance)

  try:
    with open(here/"reference"/"systematics_mc.pkl", "rb") as f:
      x_quantiles_ref, y_quantiles_ref = pickle.load(f)
    for k in set(x_quantiles) | set(y_quantiles):
      x = x_quantiles[k]
      y = y_quantiles[k]
      x_ref = x_quantiles_ref[k]
      y_ref = y_quantiles_ref[k]
      np.testing.assert_allclose(np.array([x, y]), np.array([x_ref, y_ref]), **tolerance)
  except:
    with open(here/"test_output"/"systematics_mc.pkl", "wb") as f:
      pickle.dump((x_quantiles, y_quantiles), f)
    raise

if __name__ == "__main__":
  main()
