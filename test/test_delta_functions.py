import pathlib
import roc_picker.datacard

here = pathlib.Path(__file__).parent
datacards = here/"datacards"
docsfolder = here.parent/"docs"

def main():
  datacard = roc_picker.datacard.Datacard.parse_datacard(datacards/"datacard_example_2.txt")
  delta_functions = datacard.delta_functions(flip_sign=False)
  delta_functions.plot_roc(
    npoints=100,
    yupperlim=20,
    rocfilename=docsfolder/"deltafunctions_exampleroc.pdf",
    scanfilename=docsfolder/"deltafunctions_scan.pdf",
    rocerrorsfilename=docsfolder/"deltafunctions_exampleroc_errors.pdf",
    show=False,
  )

  discrete = datacard.discrete(flip_sign=False)
  discrete.plot_roc(
    npoints=100,
    yupperlim=20,
    scanfilename=docsfolder/"discrete_scan_compare_to_delta_functions.pdf",
    show=False,
  )

  discrete_datacard = roc_picker.datacard.Datacard.parse_datacard(datacards/"datacard_example_1.txt")
  delta_functions_2 = discrete_datacard.delta_functions(flip_sign=False)
  delta_functions_2.plot_roc(
    npoints=100,
    yupperlim=20,
    scanfilename=docsfolder/"deltafunctions_scan_compare_to_discrete.pdf",
    show=False,
  )

if __name__ == "__main__":
  main()
