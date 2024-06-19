import pathlib
import roc_picker.discrete

here = pathlib.Path(__file__).parent
docsfolder = here.parent/"docs"

responders = [1, 1, 2, 2, 3, 9, 10]
nonresponders = [2, 3, 3, 4, 6, 8, 9, 10, 10, 10, 10, 11, 12, 13]

def main():
  roc_picker.discrete.DiscreteROC(
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

if __name__ == "__main__":
  main()
