"""
Test the discrete module, and generate the figures for that section of the documentation.
"""

import pathlib
import json
import math
import warnings

import numpy as np

import roc_picker.datacard
from ..utility_testing_functions import (
  compare_dict_keys,
  flip_sign_curve,
  format_value_for_json,
  Tolerance,
)

warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here/"datacards"/"simple_examples"

def main(): #pylint: disable=too-many-locals
  """
  Test the discrete module.
  """
  datacard = roc_picker.datacard.Datacard.parse_datacard(datacards/"example_roc.txt")
  discrete = datacard.discrete_roc(flip_sign=False, check_validity=True)
  rocs = discrete.make_plots(
    npoints=100,
    yupperlim=20,
    show=False,
  )

  discrete_flip = datacard.discrete_roc(flip_sign=True, check_validity=True)
  rocs_flip = discrete_flip.make_plots(
    npoints=100,
    yupperlim=20,
    show=False,
  )

  tolerance: Tolerance = {"atol": 1e-6, "rtol": 1e-6}

  # Calculate precision for JSON output based on rtol
  rtol_value = tolerance["rtol"]
  json_precision = int(abs(math.log10(rtol_value))) + 1 if rtol_value > 0 else 4

  for k in set(rocs) | set(rocs_flip):
    roc = rocs[k]
    flipk = flip_sign_curve(k)
    flip = rocs_flip[flipk]
    np.testing.assert_allclose(
      np.array([roc.x, roc.y]),
      1-np.array([flip.x, flip.y])[:,::-1],
      **tolerance,
    )
    np.testing.assert_allclose(roc.AUC, 1-flip.AUC, **tolerance)
    np.testing.assert_allclose(roc.NLL, flip.NLL, **tolerance)

  # Prepare current ROCs for comparison/saving
  current_rocs_data = {}
  for name, roc_obj in rocs.items():
    current_rocs_data[name] = {
      "x": roc_obj.x.tolist(),
      "y": roc_obj.y.tolist(),
      "AUC": roc_obj.AUC,
      "NLL": roc_obj.NLL,
    }

  try:
    # Changed to .json reference file
    with open(here/"reference"/"discrete.json", "r", encoding="utf-8") as f:
      loaded_data = json.load(f)
      # No need to create refs_data, use loaded_data directly

    # Check for missing or extra keys as per previous refactoring
    compare_dict_keys(current_rocs_data, loaded_data)

    for k, roc in current_rocs_data.items():
      ref = loaded_data[k]
      np.testing.assert_allclose(
        np.array([roc["x"], roc["y"]]),
        np.array([ref["x"], ref["y"]]),
        **tolerance
      )
      np.testing.assert_allclose(roc["AUC"], ref["AUC"], **tolerance)
      np.testing.assert_allclose(roc["NLL"], ref["NLL"], **tolerance)
  except Exception:
    with open(here/"test_output"/"discrete.json", "w", encoding="utf-8") as f: # Changed to .json
      formatted_rocs_data = {
        name: format_value_for_json(data, json_precision)
        for name, data in current_rocs_data.items()
      }
      json.dump(
        formatted_rocs_data,
        f,
        indent=2, # Use 2-space indent
        sort_keys=True # Ensure deterministic output order
      )
    raise

if __name__ == "__main__":
  main()
