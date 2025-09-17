"""
A datacard re-export for ROC Picker compatibility.
The main Datacard class now lives in the kombine package.
"""

# Re-export the Datacard class from kombine for backward compatibility
from kombine.datacard import Datacard, FixedObservable, Patient, PoissonObservable

# Re-export common classes and functions that ROC Picker needs
__all__ = ['Datacard', 'FixedObservable', 'Patient', 'PoissonObservable']
