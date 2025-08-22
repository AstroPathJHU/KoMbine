"""
Test patient_wise_only with the debug info
"""

import numpy as np
from roc_picker.kaplan_meier_p_value_MINLP import MINLPforKMPValue
from roc_picker.kaplan_meier_MINLP import KaplanMeierPatientNLL

def simple_nll(observed_param):
    def nll_func(param):
        return 0.5 * (param - observed_param) ** 2
    return nll_func

# Create test with different survival times to get non-zero probabilities
patients = [
    KaplanMeierPatientNLL(time=2.0, censored=False, parameter_nll=simple_nll(0.5), observed_parameter=0.5),  # low group, dies at time 2
    KaplanMeierPatientNLL(time=5.0, censored=True,  parameter_nll=simple_nll(0.7), observed_parameter=0.7),  # low group, censored at time 5
    KaplanMeierPatientNLL(time=1.0, censored=False, parameter_nll=simple_nll(1.5), observed_parameter=1.5), # high group, dies at time 1
    KaplanMeierPatientNLL(time=3.0, censored=True,  parameter_nll=simple_nll(1.8), observed_parameter=1.8), # high group, censored at time 3
]

minlp = MINLPforKMPValue(
    all_patients=patients,
    parameter_threshold=1.0,
)

print(f"Number of patients: {len(patients)}")
print(f"Death times: {minlp.all_death_times}")
print(f"Nominal probabilities: {minlp.nominal_curves_probabilities}")

try:
    # Test patient_wise_only
    p_value, result_null, result_alt = minlp.solve_and_pvalue(patient_wise_only=True)
    print(f"✓ Patient-wise-only p-value: {p_value}")
    print(f"  Null -2LL: {result_null.x}")
    print(f"  Alt -2LL: {result_alt.x}")
except Exception as e:
    print(f"✗ Patient-wise-only failed: {e}")

try:
    # Test regular for comparison  
    p_value, result_null, result_alt = minlp.solve_and_pvalue()
    print(f"✓ Regular p-value: {p_value}")
    print(f"  Null -2LL: {result_null.x}")
    print(f"  Alt -2LL: {result_alt.x}")
except Exception as e:
    print(f"✗ Regular p-value failed: {e}")