# ROC Picker and KoMbine

![ROC Picker logo](logo.png)

Welcome!

ROC Picker is a software package for propagating statistical and systematic
uncertainties in a biomedical analysis using ROC curves.

KoMbine provides Kaplan-Meier curve analysis functionality with uncertainty
propagation using likelihood-based methods.

## Full documentation

For detailed information and examples, please see the documentation.
The source is in the `docs/` folder, and you can download the output
(latest version from the `main` branch, compiled by Github Actions) from
[this link](https://nightly.link/AstroPathJHU/ROCPicker/workflows/test_and_docs/main/docs.zip).

## Quick start

To install ROC Picker and KoMbine, clone the repository, enter its folder, and do
```
pip install .
```

Here is a simple ROC Picker example:
```
from roc_picker.discrete import DiscreteROC
responders = [1, 1, 2, 3, 9, 10]
nonresponders = [2, 3, 3, 4, 6, 8, 9, 10, 10, 10, 10, 11, 12, 13]
DiscreteROC(
  responders=responders,
  nonresponders=nonresponders,
).make_plots(
  npoints=100,
  yupperlim=20,
  #if you want to save the output plots
  filenames=("roc.pdf", "auc_scan.pdf", "roc_errors.pdf"),
  #if you're running in a jupyter notebook or similar, and want to see the plots
  show=True,
)
```

Here is a simple KoMbine example:
```
from kombine.datacard import Datacard
import pathlib

# Load a datacard with Kaplan-Meier data
datacardfile = pathlib.Path("test/kombine/datacards/simple_examples/simple_km_few_deaths.txt")
datacard = Datacard.parse_datacard(datacardfile)

# Create Kaplan-Meier likelihood curves for low and high parameter groups
kml_low = datacard.km_likelihood(parameter_min=-float('inf'), parameter_max=0.45)
kml_high = datacard.km_likelihood(parameter_min=0.45, parameter_max=float('inf'))

# Plot the results
kml_low.plot()
kml_high.plot()
```
