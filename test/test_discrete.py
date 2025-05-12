"""
Test the discrete module, and generate the figures for that section of the documentation.
"""

import pathlib
import pickle
import typing
import warnings

import numpy as np
import roc_picker.datacard
warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here/"datacards"/"simple_examples"

def main():
  """
  Test the discrete module, and generate the figures for that section of the documentation.
  """
  datacard = roc_picker.datacard.Datacard.parse_datacard(datacards/"datacard_example_1.txt")
  discrete = datacard.discrete(flip_sign=False, check_validity=True)
  rocs = discrete.make_plots(
    npoints=100,
    yupperlim=20,
    show=False,
  )

  discrete_flip = datacard.discrete(flip_sign=True, check_validity=True)
  rocs_flip = discrete_flip.make_plots(
    npoints=100,
    yupperlim=20,
    show=False,
  )

  def hack_fix_roc(roc):
    if roc.x[0] == roc.x[1] and roc.y[0] == roc.y[1]:
      roc["x"] = roc.x[1:]
      roc["y"] = roc.y[1:]
    if roc.x[-1] == roc.x[-2] and roc.y[-1] == roc.y[-2]:
      roc["x"] = roc.x[:-1]
      roc["y"] = roc.y[:-1]

  class Tolerance(typing.TypedDict):
    rtol: float
    atol: float
  tolerance: Tolerance = {"atol": 1e-6, "rtol": 1e-6}

  for k in set(rocs) | set(rocs_flip):
    roc = rocs[k]
    flipk = {
      "nominal": "nominal",
      "p68": "m68",
      "p95": "m95",
      "m68": "p68",
      "m95": "p95",
    }[k]
    flip = rocs_flip[flipk]
    np.testing.assert_allclose(
      np.array([roc.x, roc.y]),
      1-np.array([flip.x, flip.y])[:,::-1],
      **tolerance,
    )
    np.testing.assert_allclose(roc.AUC, 1-flip.AUC, **tolerance)
    np.testing.assert_allclose(roc.NLL, flip.NLL, **tolerance)

  try:
    with open(here/"reference"/"discrete.pkl", "rb") as f:
      refs = pickle.load(f)
    for k in set(rocs) | set(refs):
      roc = rocs[k]
      ref = refs[k]
      hack_fix_roc(ref)
      np.testing.assert_allclose(np.array([roc.x, roc.y]), np.array([ref.x, ref.y]), **tolerance)
      np.testing.assert_allclose(roc.AUC, ref.AUC, **tolerance)
      np.testing.assert_allclose(roc.NLL, ref.NLL, **tolerance)
  except:
    with open(here/"test_output"/"discrete.pkl", "wb") as f:
      pickle.dump(rocs, f)
    raise

if __name__ == "__main__":
  main()
