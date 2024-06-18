import numpy as np, pathlib
import roc_picker.delta_functions, roc_picker.discrete
from . import test_discrete

here = pathlib.Path(__file__).parent
docsfolder = here.parent/"docs"

responders = np.linspace(-10, 10, 3)
nonresponders = responders+5

def main():
  roc_picker.delta_functions.DeltaFunctions(
    responders=responders,
    nonresponders=nonresponders,
    flip_sign=False,
  ).plot_roc(
    npoints=100,
    yupperlim=20,
    rocfilename=docsfolder/"deltafunctions_exampleroc.pdf",
    scanfilename=docsfolder/"deltafunctions_scan.pdf",
    rocerrorsfilename=docsfolder/"deltafunctions_exampleroc_errors.pdf",
    show=False,
  )

  roc_picker.discrete.DiscreteROC(
    responders=responders,
    nonresponders=nonresponders,
    flip_sign=False,
  ).plot_roc(
    npoints=100,
    yupperlim=20,
    scanfilename=docsfolder/"discrete_scan_compare_to_delta_functions.pdf",
    show=False,
  )

  roc_picker.delta_functions.DeltaFunctions(
    responders=test_discrete.responders,
    nonresponders=test_discrete.nonresponders,
    flip_sign=False,
  ).plot_roc(
    npoints=100,
    yupperlim=20,
    scanfilename=docsfolder/"deltafunctions_scan_compare_to_discrete.pdf",
    show=False,
  )

if __name__ == "__main__":
  main()
