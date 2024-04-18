# ROC Picker

Welcome!

ROC Picker is a software package for propagating statistical and systematic
uncertainties in a biomedical analysis.

To install ROC Picker, clone the repository, enter its folder, and do
```
pip install .
```

Quick start:
```
from roc_picker.discrete import DiscreteROC
responders = [1, 1, 2, 3, 9, 10]
nonresponders = [2, 3, 3, 4, 6, 8, 9, 10, 10, 10, 10, 11, 12, 13]
DiscreteROC(
  responders=responders,
  nonresponders=nonresponders,
).plot_roc(
  yupperlim=20,
  #if you want to save the output plots
  rocfilename="roc.pdf",
  scanfilename="auc_scan.pdf",
  rocerrorsfilename="roc_errors.pdf",
  #if you're running in a jupyter notebook or similar, and want to see the plots
  show=True,
)
```

For more detailed information, please see the latex document in the `docs/` folder.
(The plots can be compiled by running the scripts in the test folder.)

