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
  npoints=100,
  yupperlim=20,
  #if you want to save the output plots
  rocfilename="roc.pdf",
  scanfilename="auc_scan.pdf",
  rocerrorsfilename="roc_errors.pdf",
  #if you're running in a jupyter notebook or similar, and want to see the plots
  show=True,
)
```

For more detailed information, please see the documentation.
The source is in the `docs/` folder, and you can download the output
(latest version from the `main` branch, compiled by Github Actions) from
[this link](https://nightly.link/AstroPathJHU/ROCPicker/workflows/test_and_docs/main/docs.zip).

It contains:
 - LaTeX:
   - `rocpicker.tex` (compiled: `.pdf`): a detailed explanation of the math that
      goes into all the methods used in ROC Picker.  To compile the plots, first run
      the scripts in the `test/` folder, then compile the LaTeX document
      using xelatex and biber.
 - Jupyter notebooks:
   - These are all committed as `.md`.  To convert them to `.ipynb`, use
     `jupytext --sync *.md`.  They are stored as `.html` in the zip file in
     the link above.
   - `examples.md`: examples of how to run the various methods included in ROC Picker
     using the datacard interface.
   - `small_perturbation.md`: an illustration of small perturbations to one of the
     observables and how that affects the results.
   - `lung_example.md`: an example analysis of statistical and systematic uncertainties
     using AstroPath lung cancer data
