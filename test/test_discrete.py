import numpy as np, pathlib, pickle
import roc_picker.discrete

here = pathlib.Path(__file__).parent
docsfolder = here.parent/"docs"

responders = [1, 1, 2, 2, 3, 9, 10]
nonresponders = [2, 3, 3, 4, 6, 8, 9, 10, 10, 10, 10, 11, 12, 13]

def main():
  rocs = roc_picker.discrete.DiscreteROC(
    responders=responders,
    nonresponders=nonresponders,
    flip_sign=False,
    check_validity=True,
  ).plot_roc(
    npoints=100,
    yupperlim=20,
    rocfilename=docsfolder/"discrete_exampleroc.pdf",
    scanfilename=docsfolder/"discrete_scan.pdf",
    rocerrorsfilename=docsfolder/"discrete_exampleroc_errors.pdf",
    show=False,
  )

  try:
    with open(here/"reference"/"discrete.pkl", "rb") as f:
      refs = pickle.load(f)
    for k in set(rocs) | set(refs):
      roc = rocs[k]
      ref = refs[k]
      np.testing.assert_allclose(roc.x, ref.x)
      np.testing.assert_allclose(roc.y, ref.y)
      np.testing.assert_allclose(roc.AUC, ref.AUC)
      np.testing.assert_allclose(roc.NLL, ref.NLL)
  except:
    with open(here/"test_output"/"discrete.pkl", "wb") as f:
      pickle.dump(rocs, f)
    raise

if __name__ == "__main__":
  main()
