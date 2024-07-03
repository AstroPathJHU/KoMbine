---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import warnings
warnings.simplefilter("error")
```

In this notebook I want to explore what happens if you make a small perturbation to one of the values for the responders or non-responders.  Is the behavior stable?  This is a sanity check on the method.

```python
import copy, numpy as np, pathlib
from roc_picker.datacard import Datacard
```

# Discrete

For this method, the only thing that matters is the ordering of the responders and non-responders with respect to each other.  In this example I take a non-responder who has the same observable as one of the responders and move the value up or down a little bit.  The results show that the error regions and AUCs change a bit, but not by that much.

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent/"test"/"datacards"/"simple_examples"/"datacard_example_1.txt"
datacard = Datacard.parse_datacard(datacardfile)
```

```python
shift_up = copy.deepcopy(datacard.patients)
shift_down = copy.deepcopy(datacard.patients)
np.testing.assert_equal(datacard.patients[13]["value"], 9)
shift_up[13]["value"] = 9.1
shift_down[13]["value"] = 8.9

for patient, up, down in zip(datacard.patients, shift_up, shift_down, strict=True):
    toprint = patient["response"], patient["value"]
    if up["value"] != patient["value"]:
        toprint = (*toprint, "shift to", up["value"], "or", down["value"])
    print(*toprint)

datacard_shift_up = Datacard(patients=shift_up, systematics=[], observable_type="fixed")
datacard_shift_down = Datacard(patients=shift_down, systematics=[], observable_type="fixed")
```

```python
# _ = datacard.discrete().plot_roc(show=[False, False, True])
# _ = datacard_shift_up.discrete().plot_roc(show=[False, False, True])
# _ = datacard_shift_down.discrete().plot_roc(show=[False, False, True])
```

# Systematics MC

Now the actual value does matter (especially relative to the error).

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent/"test"/"datacards"/"simple_examples"/"datacard_example_3.txt"
datacard = Datacard.parse_datacard(datacardfile)
```

```python
shift_up = copy.deepcopy(datacard.patients)
shift_down = copy.deepcopy(datacard.patients)
np.testing.assert_equal(datacard.patients[6]["value"], 30)
shift_up[6]["value"] = 31
shift_down[6]["value"] = 29

for patient, up, down in zip(datacard.patients, shift_up, shift_down, strict=True):
    toprint = patient["response"], patient["value"]
    if up["value"] != patient["value"]:
        toprint = (*toprint, "shift to", up["value"], "or", down["value"])
    print(*toprint)

datacard_shift_up = Datacard(patients=shift_up, systematics=[], observable_type="poisson")
datacard_shift_down = Datacard(patients=shift_down, systematics=[], observable_type="poisson")
```

```python
datacard.systematics_mc(flip_sign=False).generate(size=10000, random_state=123456).plot(show=True)
datacard_shift_up.systematics_mc(flip_sign=False).generate(size=10000, random_state=123456).plot(show=True)
datacard_shift_down.systematics_mc(flip_sign=False).generate(size=10000, random_state=123456).plot(show=True)
```

```python

```
