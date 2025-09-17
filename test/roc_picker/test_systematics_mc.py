"""
Test the systematics_mc module, and generate the figures for that section of the documentation.
"""

import argparse
import json
import math
import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.special

import roc_picker.datacard
from roc_picker.systematics_mc import AUC
from .utility_testing_functions import (
  compare_dict_keys,
  flip_sign_curve,
  format_value_for_json,
  Tolerance,
)

warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here/"datacards"/"lung"

def test_systematics_mc(make_plots=False): #pylint: disable=too-many-locals, too-many-statements
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
  quantiles = [
    (1 + scipy.special.erf(nsigma/np.sqrt(2))) / 2 # pylint: disable=no-member
    for nsigma in sigmas
  ]

  x_quantiles = {}
  y_quantiles = {}
  (
    (
      x_quantiles["m95"],
      x_quantiles["m68"],
      x_quantiles["nominal"],
      x_quantiles["p68"],
      x_quantiles["p95"],
    ), (
      y_quantiles["m95"],
      y_quantiles["m68"],
      y_quantiles["nominal"],
      y_quantiles["p68"],
      y_quantiles["p95"],
    )
  ) = rocs.roc_quantiles(quantiles)

  del rocdistributions, rocs

  rocdistributions_flip = datacard.systematics_mc_roc(flip_sign=True)
  rocs_flip = rocdistributions_flip.generate(
    size=100,
    random_state=123456
  )

  x_quantiles_flip = {}
  y_quantiles_flip = {}
  (
    (
      x_quantiles_flip["m95"],
      x_quantiles_flip["m68"],
      x_quantiles_flip["nominal"],
      x_quantiles_flip["p68"],
      x_quantiles_flip["p95"],
    ), (
      y_quantiles_flip["m95"],
      y_quantiles_flip["m68"],
      y_quantiles_flip["nominal"],
      y_quantiles_flip["p68"],
      y_quantiles_flip["p95"],
    )
  ) = rocs_flip.roc_quantiles(quantiles)

  AUCs = {
    k: AUC(x_quantiles[k], y_quantiles[k])
    for k in x_quantiles | y_quantiles
  }
  AUCs_flip = {
    k: AUC(x_quantiles_flip[k], y_quantiles_flip[k])
    for k in x_quantiles_flip | y_quantiles_flip
  }

  tolerance: Tolerance = {"atol": 1e-6, "rtol": 1e-6}

  # Calculate precision for JSON output based on rtol
  rtol_value = tolerance["rtol"]
  json_precision = int(abs(math.log10(rtol_value))) + 1 if rtol_value > 0 else 4

  for k in sorted(set(x_quantiles) | set(x_quantiles_flip)):
    x = x_quantiles[k]
    y = y_quantiles[k]
    auc = AUCs[k]
    flipk = flip_sign_curve(k)
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
      if make_plots: # Conditionally make plots
        plt.figure()
        plt.plot(x, y, label="unflipped")
        plt.plot(1-x_flip, 1-y_flip, label="flipped twice")
        plt.legend()
        plt.savefig(here/"test_output"/f"roc_{k}.png")
        plt.close()
      raise
    np.testing.assert_allclose(auc, 1-auc_flip, **tolerance)

  # Prepare current quantiles data for comparison/saving
  current_quantiles_data = {}
  for k in x_quantiles | y_quantiles:
    current_quantiles_data[k] = {
      "x": x_quantiles[k].tolist(),
      "y": y_quantiles[k].tolist(),
    }

  try:
    with open(here/"reference"/"systematics_mc.json", "r", encoding="utf-8") as f:
      loaded_data = json.load(f)

    compare_dict_keys(current_quantiles_data, loaded_data)

    for k, current_data in current_quantiles_data.items():
      ref_data = loaded_data[k]
      np.testing.assert_allclose(
        np.array(current_data["x"]),
        np.array(ref_data["x"]),
        **tolerance,
        err_msg=f"x_quantiles[{k}] does not match the reference."
      )
      np.testing.assert_allclose(
        np.array(current_data["y"]),
        np.array(ref_data["y"]),
        **tolerance,
        err_msg=f"y_quantiles[{k}] does not match the reference."
      )
  except Exception:
    with open(here/"test_output"/"systematics_mc.json", "w", encoding="utf-8") as f:
      formatted_quantiles_data = {
        name: format_value_for_json(data, json_precision)
        for name, data in current_quantiles_data.items()
      }
      json.dump(
        formatted_quantiles_data,
        f,
        indent=2, # Use 2-space indent
        sort_keys=True # Ensure deterministic output order
      )
    raise

def main(): # New main function with argparse
  """
  Main function to run the systematics_mc test with optional plotting.
  """
  p = argparse.ArgumentParser(
    description="Test the systematics_mc module with optional plotting."
  )
  p.add_argument(
    "--make-plots",
    action="store_true",
    help="Generate ROC plots if assertions fail.",
  )
  args = p.parse_args()
  test_systematics_mc(make_plots=args.make_plots)

if __name__ == "__main__":
  main()
