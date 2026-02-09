---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: rocpicker
    language: python
    name: python3
---

## Summary: Yi's Method Characteristics

| Feature | Yi's Method | MINLP Method |
|---------|-------------|-------------| 
| **Weighting** | Probabilistic (0-1) | Integer (0/1) |
| **Speed** | Fast | Slower (optimization) |
| **Output** | Point estimates | Point + CI |
| **Applicability** | Poisson/measurement error | General uncertainties |
| **Assumption** | Known error distribution | General NLL penalties |

### When to use Yi's Method:
- Fast preliminary analysis with Poisson measurements
- Research with clear measurement error assumptions
- When probability-weighted estimates are sufficient

### When to use MINLP:
- When confidence intervals are essential
- More general uncertainty structures
- Publication-ready results with rigorous CIs

```python
# Plot Yi's profile likelihood for hazard ratio
hr_fine = np.linspace(0.3, 5.0, 50)
yi_2nlls_fine = []

for hr in hr_fine:
    result = datacard.km_hazard_ratio_yi(
        parameter_threshold=threshold,
        hazard_ratio=hr,
        parameter_min=-np.inf,
        parameter_max=np.inf,
        method='bayesian',
    )
    yi_2nlls_fine.append(result.x)

plt.figure(figsize=(10, 6))
plt.plot(hr_fine, yi_2nlls_fine, 'b-', linewidth=2, label="Yi's Profile Likelihood")
plt.axvline(x=best_hr_yi, color='r', linestyle='--', label=f"Best-fit: {best_hr_yi:.3f}")
plt.axhline(y=min(yi_2nlls_fine) + 2.706, color='g', linestyle=':', alpha=0.7, label='95% CI threshold')
plt.xlabel('Hazard Ratio')
plt.ylabel('2NLL')
plt.title("Yi's Profile Likelihood for Hazard Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

```python
# Calculate Yi's corrected 2NLL at different hazard ratio values
hazard_ratios = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
yi_2nlls = []

for hr in hazard_ratios:
    result = datacard.km_hazard_ratio_yi(
        parameter_threshold=threshold,
        hazard_ratio=hr,
        parameter_min=-np.inf,
        parameter_max=np.inf,
        method='bayesian',
    )
    yi_2nlls.append(result.x)
    print(f"HR = {hr:.1f}: 2NLL = {result.x:.4f}, Cox 2NLL = {result.cox_2NLL:.4f}")

# Find best-fit HR
best_idx = np.argmin(yi_2nlls)
best_hr_yi = hazard_ratios[best_idx]
print(f"\nBest-fit HR (Yi): {best_hr_yi:.2f}")
```

## Yi's Method for Hazard Ratios

Yi's correction for Cox proportional hazards uses probabilistic weights in the partial likelihood calculation.

```python
# Calculate Yi's corrected logrank test p-value
result_yi_pvalue = datacard.km_p_value_logrank_yi(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
    method='bayesian',
    prior_alpha=0.5,
    prior_beta=0.0,
)

print(f"Yi's Corrected Logrank Test:")
print(f"  P-value: {result_yi_pvalue['p_value']:.4e}")
print(f"  Test statistic: {result_yi_pvalue['logrank_statistic']:.4f}")
print(f"  U (observed - expected): {result_yi_pvalue['U']:.4f}")
print(f"  V (variance): {result_yi_pvalue['V']:.4f}")
print(f"  Patients observed in low group: {result_yi_pvalue['n_low_observed']}")
print(f"  Patients observed in high group: {result_yi_pvalue['n_high_observed']}")
```

## Yi's Method for P-Values (Logrank Test)

Yi's correction for the logrank test accounts for measurement uncertainty in patient group assignments when computing the test statistic.

```python
# Plot Yi's weighted KM curves
plt.figure(figsize=(10, 6))

# For low group
times_low = result_yi_low['times_for_plot']
surv_low = result_yi_low['survival_probabilities']
plt.step(times_low, surv_low, where='post', label=f'Low group (Yi corrected)', linewidth=2, color='red')

# For high group
times_high = result_yi_high['times_for_plot']
surv_high = result_yi_high['survival_probabilities']
plt.step(times_high, surv_high, where='post', label=f'High group (Yi corrected)', linewidth=2, color='blue')

plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title("Yi's Weighted Kaplan-Meier Curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])
plt.show()
```

```python
# Define threshold
threshold = 0.5

# Calculate Yi's weighted Kaplan-Meier for the low group
result_yi_low = datacard.km_survival_yi(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=threshold,  # Only low group
    method='bayesian',
)

# Calculate Yi's weighted Kaplan-Meier for the high group
result_yi_high = datacard.km_survival_yi(
    parameter_threshold=threshold,
    parameter_min=threshold,
    parameter_max=np.inf,  # Only high group
    method='bayesian',
)

print(f"Yi's KM for low group (parameter < {threshold}):")
print(f"  Times: {result_yi_low['times_for_plot'][:5]}...")
print(f"  Survival probs: {result_yi_low['survival_probabilities'][:5]}...")
print(f"  Death times: {result_yi_low['death_times']}")

print(f"\nYi's KM for high group (parameter >= {threshold}):")
print(f"  Times: {result_yi_high['times_for_plot'][:5]}...")
print(f"  Survival probs: {result_yi_high['survival_probabilities'][:5]}...")
print(f"  Death times: {result_yi_high['death_times']}")
```

## Yi's Method for Kaplan-Meier Curves

Using Yi's correction, we estimate the best-fit Kaplan-Meier curve by weighting each patient according to their probability of belonging to each group.

```python
here = pathlib.Path(".").resolve()
# Load a datacard with Poisson density observables (measurement error)
datacardfile = here.parent.parent / "test" / "kombine" / "datacards" / "simple_examples" / "poisson_km_censoring.txt"

# Parse the datacard
datacard = Datacard.parse_datacard(datacardfile)
print(f"Loaded {len(datacard.patients)} patients")
print(f"Deaths: {sum(1 for p in datacard.patients if not p.censored)}")
print(f"Censored: {sum(1 for p in datacard.patients if p.censored)}")

# Check observable type
obs_types = {type(p.observable).__name__ for p in datacard.patients if p.observable}
print(f"Observable types: {obs_types}")
```

## Loading Data with Measurement Error

To illustrate Yi's method, we use patient data with Poisson density measurements (measurement error in biomarker values).


## Yi's Method Overview

Yi's correction method (Section 3.7.1 of "Statistical Analysis with Measurement Error or Misclassification", 2017) uses inverse probability weighting:

1. For each patient, compute: P(true group = high | observed data)
2. Weight patient contributions by their individual probability
3. Apply standard formulas (KM, logrank, Cox) with weighted counts

**Key differences from MINLP:**
- Yi: Probabilistic weighting (fractional assignments)
- MINLP: Integer optimization (0/1 assignments with penalties)
- Yi: Point estimates only
- MINLP: Point estimates + confidence intervals
- Yi: Fast computation
- MINLP: More computationally intensive

```python
import warnings
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from kombine.datacard import Datacard, PoissonDensityObservable
```

# Yi's Misclassification Correction: Comprehensive Comparison

This notebook demonstrates Yi's method for correcting measurement error in group assignment across all three analyses:
1. **Kaplan-Meier curves** - Weighted survival probability estimates
2. **P-values** - Logrank test with probabilistic weighting
3. **Hazard ratios** - Cox partial likelihood with uncertainty weighting

Yi's method uses inverse probability weighting to account for measurement uncertainty, providing an alternative to KoMbine's MINLP optimization approach.


# Yi's Misclassification Correction: Comprehensive Comparison

This notebook demonstrates Yi's method for correcting measurement error in group assignment across all three analyses:
1. **Kaplan-Meier curves** - Weighted survival probability estimates
2. **P-values** - Logrank test with probabilistic weighting
3. **Hazard ratios** - Cox partial likelihood with uncertainty weighting

Yi's method uses inverse probability weighting to account for measurement uncertainty, providing an alternative to KoMbine's MINLP optimization approach.

```python
import warnings
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from kombine.datacard import Datacard, PoissonDensityObservable
```

## Yi's Method Overview

Yi's correction method (Section 3.7.1 of "Statistical Analysis with Measurement Error or Misclassification", 2017) uses inverse probability weighting:

1. For each patient, compute: P(true group = high | observed data)
2. Weight patient contributions by their individual probability
3. Apply standard formulas (KM, logrank, Cox) with weighted counts

**Key differences from MINLP:**
- Yi: Probabilistic weighting (fractional assignments)
- MINLP: Integer optimization (0/1 assignments with penalties)
- Yi: Point estimates only
- MINLP: Point estimates + confidence intervals
- Yi: Fast computation
- MINLP: More computationally intensive


## Loading Data with Measurement Error

To illustrate Yi's method, we use patient data with Poisson density measurements (measurement error in biomarker values).

```python
here = pathlib.Path(".").resolve()
# Load a datacard with Poisson density observables (measurement error)
datacardfile = here.parent.parent / "test" / "kombine" / "datacards" / "simple_examples" / "poisson_km_censoring.txt"

# Parse the datacard
datacard = Datacard.parse_datacard(datacardfile)
print(f"Loaded {len(datacard.patients)} patients")
print(f"Deaths: {sum(1 for p in datacard.patients if not p.censored)}")
print(f"Censored: {sum(1 for p in datacard.patients if p.censored)}")

# Check observable type
obs_types = {type(p.observable).__name__ for p in datacard.patients if p.observable}
print(f"Observable types: {obs_types}")
```

## Yi's Method for Kaplan-Meier Curves

Using Yi's correction, we estimate the best-fit Kaplan-Meier curve by weighting each patient according to their probability of belonging to each group.

```python
# Define threshold
threshold = 0.5

# Calculate Yi's weighted Kaplan-Meier for the low group
result_yi_low = datacard.km_survival_yi(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=threshold,  # Only low group
    method='bayesian',
)

# Calculate Yi's weighted Kaplan-Meier for the high group
result_yi_high = datacard.km_survival_yi(
    parameter_threshold=threshold,
    parameter_min=threshold,
    parameter_max=np.inf,  # Only high group
    method='bayesian',
)

print(f"Yi's KM for low group (parameter < {threshold}):")
print(f"  Times: {result_yi_low['times_for_plot'][:5]}...")
print(f"  Survival probs: {result_yi_low['survival_probabilities'][:5]}...")
print(f"  Death times: {result_yi_low['death_times']}")

print(f"\nYi's KM for high group (parameter >= {threshold}):")
print(f"  Times: {result_yi_high['times_for_plot'][:5]}...")
print(f"  Survival probs: {result_yi_high['survival_probabilities'][:5]}...")
print(f"  Death times: {result_yi_high['death_times']}")
```

```python
# Plot Yi's weighted KM curves
plt.figure(figsize=(10, 6))

# For low group
times_low = result_yi_low['times_for_plot']
surv_low = result_yi_low['survival_probabilities']
plt.step(times_low, surv_low, where='post', label=f'Low group (Yi corrected)', linewidth=2, color='red')

# For high group
times_high = result_yi_high['times_for_plot']
surv_high = result_yi_high['survival_probabilities']
plt.step(times_high, surv_high, where='post', label=f'High group (Yi corrected)', linewidth=2, color='blue')

plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title("Yi's Weighted Kaplan-Meier Curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])
plt.show()
```

## Yi's Method for P-Values (Logrank Test)

Yi's correction for the logrank test accounts for measurement uncertainty in patient group assignments when computing the test statistic.

```python
# Calculate Yi's corrected logrank test p-value
result_yi_pvalue = datacard.km_p_value_logrank_yi(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
    method='bayesian',
    prior_alpha=0.5,
    prior_beta=0.0,
)

print(f"Yi's Corrected Logrank Test:")
print(f"  P-value: {result_yi_pvalue['p_value']:.4e}")
print(f"  Test statistic: {result_yi_pvalue['logrank_statistic']:.4f}")
print(f"  U (observed - expected): {result_yi_pvalue['U']:.4f}")
print(f"  V (variance): {result_yi_pvalue['V']:.4f}")
print(f"  Patients observed in low group: {result_yi_pvalue['n_low_observed']}")
print(f"  Patients observed in high group: {result_yi_pvalue['n_high_observed']}")
```

## Yi's Method for Hazard Ratios

Yi's correction for Cox proportional hazards uses probabilistic weights in the partial likelihood calculation.

```python
# Calculate Yi's corrected 2NLL at different hazard ratio values
hazard_ratios = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
yi_2nlls = []

for hr in hazard_ratios:
    result = datacard.km_hazard_ratio_yi(
        parameter_threshold=threshold,
        hazard_ratio=hr,
        parameter_min=-np.inf,
        parameter_max=np.inf,
        method='bayesian',
    )
    yi_2nlls.append(result.x)
    print(f"HR = {hr:.1f}: 2NLL = {result.x:.4f}, Cox 2NLL = {result.cox_2NLL:.4f}")

# Find best-fit HR
best_idx = np.argmin(yi_2nlls)
best_hr_yi = hazard_ratios[best_idx]
print(f"\nBest-fit HR (Yi): {best_hr_yi:.2f}")
```

```python
# Plot Yi's profile likelihood for hazard ratio
hr_fine = np.linspace(0.3, 5.0, 50)
yi_2nlls_fine = []

for hr in hr_fine:
    result = datacard.km_hazard_ratio_yi(
        parameter_threshold=threshold,
        hazard_ratio=hr,
        parameter_min=-np.inf,
        parameter_max=np.inf,
        method='bayesian',
    )
    yi_2nlls_fine.append(result.x)

plt.figure(figsize=(10, 6))
plt.plot(hr_fine, yi_2nlls_fine, 'b-', linewidth=2, label="Yi's Profile Likelihood")
plt.axvline(x=best_hr_yi, color='r', linestyle='--', label=f"Best-fit: {best_hr_yi:.3f}")
plt.axhline(y=min(yi_2nlls_fine) + 2.706, color='g', linestyle=':', alpha=0.7, label='95% CI threshold')
plt.xlabel('Hazard Ratio')
plt.ylabel('2NLL')
plt.title("Yi's Profile Likelihood for Hazard Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Summary: Yi's Method Characteristics

| Feature | Yi's Method | MINLP Method |
|---------|-------------|-------------|
| **Weighting** | Probabilistic (0-1) | Integer (0/1) |
| **Speed** | Fast | Slower (optimization) |
| **Output** | Point estimates | Point + CI |
| **Applicability** | Poisson/measurement error | General uncertainties |
| **Assumption** | Known error distribution | General NLL penalties |

### When to use Yi's Method:
- Fast preliminary analysis with Poisson measurements
- Research with clear measurement error assumptions
- When probability-weighted estimates are sufficient

### When to use MINLP:
- When confidence intervals are essential
- More general uncertainty structures
- Publication-ready results with rigorous CIs
