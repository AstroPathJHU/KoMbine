"""
Test the patient_wise_only=True functionality in kaplan_meier_p_value_MINLP.py
"""

import numpy as np
from roc_picker.kaplan_meier_p_value_MINLP import MINLPforKMPValue
from roc_picker.kaplan_meier_MINLP import KaplanMeierPatientNLL

def create_test_patients():
    """Create a simple set of test patients for validation."""
    
    def simple_nll(observed_param):
        """Simple quadratic NLL function centered around the observed parameter."""
        def nll_func(param):
            return 0.5 * (param - observed_param) ** 2
        return nll_func
    
    # Create patients with different survival times and parameters
    # Use only 2 patients to stay within Gurobi license limits
    patients = [
        # Low group patient - dies early (worse survival)
        KaplanMeierPatientNLL(time=1.0, censored=False, parameter_nll=simple_nll(0.3), observed_parameter=0.3),
        # High group patient - survives longer (better survival)
        KaplanMeierPatientNLL(time=3.0, censored=True, parameter_nll=simple_nll(1.5), observed_parameter=1.5),
    ]
    
    return patients

def test_nominal_curves_probabilities():
    """Test the calculation of nominal curve probabilities."""
    patients = create_test_patients()
    
    minlp = MINLPforKMPValue(
        all_patients=patients,
        parameter_threshold=1.0,  # Split at 1.0
    )
    
    # Test that we can calculate nominal probabilities
    low_prob, high_prob = minlp.nominal_curves_probabilities
    
    # Basic sanity checks
    assert 0 <= low_prob <= 1, f"Low probability {low_prob} should be between 0 and 1"
    assert 0 <= high_prob <= 1, f"High probability {high_prob} should be between 0 and 1"
    
    # For our test case, expect low curve to have worse survival (0.0) and high curve better (1.0)
    assert low_prob == 0.0, f"Expected low curve probability to be 0.0, got {low_prob}"
    assert high_prob == 1.0, f"Expected high curve probability to be 1.0, got {high_prob}"

def test_patient_wise_only_functionality():
    """Test that patient_wise_only produces different results from regular p-value."""
    patients = create_test_patients()
    
    minlp = MINLPforKMPValue(
        all_patients=patients,
        parameter_threshold=1.0,
    )
    
    # Test patient_wise_only p-value
    p_value_pw, result_null_pw, result_alt_pw = minlp.solve_and_pvalue(patient_wise_only=True)
    
    # Test regular p-value
    p_value_reg, result_null_reg, result_alt_reg = minlp.solve_and_pvalue()
    
    # Basic sanity checks
    assert 0 <= p_value_pw <= 1, f"Patient-wise-only p-value {p_value_pw} should be between 0 and 1"
    assert 0 <= p_value_reg <= 1, f"Regular p-value {p_value_reg} should be between 0 and 1"
    assert result_null_pw.success, "Patient-wise-only null hypothesis optimization should succeed"
    assert result_alt_pw.success, "Patient-wise-only alternative hypothesis optimization should succeed"
    assert result_null_reg.success, "Regular null hypothesis optimization should succeed"
    assert result_alt_reg.success, "Regular alternative hypothesis optimization should succeed"
    
    # The key test: patient_wise_only should produce different results
    p_value_diff = abs(p_value_pw - p_value_reg)
    assert p_value_diff > 1e-6, f"Patient-wise-only should produce different results, but difference was only {p_value_diff}"

def test_mutually_exclusive_options():
    """Test that binomial_only and patient_wise_only are mutually exclusive."""
    patients = create_test_patients()
    
    minlp = MINLPforKMPValue(
        all_patients=patients,
        parameter_threshold=1.0,
    )
    
    try:
        # This should raise an error
        minlp.solve_and_pvalue(binomial_only=True, patient_wise_only=True)
        assert False, "Should have raised ValueError for both binomial_only and patient_wise_only being True"
    except ValueError as e:
        assert "binomial_only and patient_wise_only cannot both be True" in str(e)

def test_basic_functionality_still_works():
    """Test that the basic functionality still works after our changes."""
    patients = create_test_patients()
    
    minlp = MINLPforKMPValue(
        all_patients=patients,
        parameter_threshold=1.0,
    )
    
    # Test that regular p-value still works
    p_value, result_null, result_alt = minlp.solve_and_pvalue()
    
    assert 0 <= p_value <= 1, f"P-value {p_value} should be between 0 and 1"
    assert result_null.success, "Null hypothesis optimization should succeed"
    assert result_alt.success, "Alternative hypothesis optimization should succeed"

def runtest():
    """Run all patient_wise_only tests."""
    test_nominal_curves_probabilities()
    test_basic_functionality_still_works()
    test_patient_wise_only_functionality()
    test_mutually_exclusive_options()

if __name__ == "__main__":
    runtest()
    print("✓ All patient_wise_only tests passed!")