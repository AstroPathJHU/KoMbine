---
jupyter:
  jupytext:
    formats: ipynb,md,py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: rocpicker
    language: python
    name: python3
---

```python
import warnings
warnings.simplefilter("error")
```

In this notebook I want to explore what happens if you make a small perturbation to one of the values for the responders or non-responders.  Is the behavior stable?  This is a sanity check on the method.

```python
import copy
import pathlib

import numpy as np

from roc_picker.datacard import Datacard, FixedObservable, Patient, PoissonObservable
from roc_picker.systematics_mc import ScipyDistribution
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
np.testing.assert_equal(datacard.patients[13].observable, FixedObservable(9.0))
shift_up[13] = Patient(response=datacard.patients[13].response, observable=FixedObservable(9.1))
shift_down[13] = Patient(response=datacard.patients[13].response, observable=FixedObservable(8.9))

for patient, up, down in zip(datacard.patients, shift_up, shift_down, strict=True):
    toprint = patient.response, patient.observable
    if up.observable != patient.observable:
        toprint = (*toprint, "shift to", up.observable, "or", down.observable)
    print(*toprint)

datacard_shift_up = Datacard(patients=shift_up)
datacard_shift_down = Datacard(patients=shift_down)
```

```python
_ = datacard.discrete_roc().make_plots(show=[False, False, True])
_ = datacard_shift_up.discrete_roc().make_plots(show=[False, False, True])
_ = datacard_shift_down.discrete_roc().make_plots(show=[False, False, True])
```

# Systematics MC

Now the actual value does matter (relative to the error).  For this example I use Poisson uncertainties.  Again, one of the non-responders nominally has the same count as a responder, and I shift it up or down by 1.  We expect the nominal ROC curve to change, but minimal change to the error bands.  And this is exactly what we get.

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent/"test"/"datacards"/"simple_examples"/"datacard_example_3.txt"
datacard = Datacard.parse_datacard(datacardfile)
```

```python
shift_up = copy.deepcopy(datacard.patients)
shift_down = copy.deepcopy(datacard.patients)
patient = datacard.patients[6]
distribution = patient.get_distribution()
np.testing.assert_equal(distribution.nominal, 30)
assert isinstance(patient.observable, PoissonObservable)
assert len(patient.systematics) == 0
assert isinstance(distribution, ScipyDistribution)
shift_up[6] = Patient(response=datacard.patients[6].response, observable=PoissonObservable(31, unique_id=distribution.unique_id))
shift_down[6] = Patient(response=datacard.patients[6].response, observable=PoissonObservable(29, unique_id=distribution.unique_id))
del distribution

for patient, up, down in zip(datacard.patients, shift_up, shift_down, strict=True):
    toprint = patient.response, patient.observable
    assert isinstance(patient.observable, PoissonObservable)
    assert isinstance(up.observable, PoissonObservable)
    if up.observable.count != patient.observable.count:
        toprint = (*toprint, "shift to", up.observable, "or", down.observable)
    print(*toprint)

datacard_shift_up = Datacard(patients=shift_up)
datacard_shift_down = Datacard(patients=shift_down)
```

```python
datacard.systematics_mc_roc(flip_sign=False).generate(size=10000, random_state=123456).plot(show=True)
datacard.clear_distributions()
datacard_shift_up.systematics_mc_roc(flip_sign=False).generate(size=10000, random_state=123456).plot(show=True)
datacard_shift_up.clear_distributions()
datacard_shift_down.systematics_mc_roc(flip_sign=False).generate(size=10000, random_state=123456).plot(show=True)
datacard_shift_down.clear_distributions()
```

```python

```
