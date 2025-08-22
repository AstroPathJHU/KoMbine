# Kaplan-Meier P-Value Implementation

This implementation adds conventional p-value calculations to Kaplan-Meier curves using the exponential Greenwood method, similar to how confidence intervals are calculated.

## What it does

- Calculates p-values for each time point on the Kaplan-Meier curve
- Tests the null hypothesis that the survival probability equals a specified value (default 0.5)
- Uses the same Greenwood variance formula as confidence intervals
- Displays significant p-values as annotations on the plot when Greenwood intervals are shown

## Usage

### Python API

```python
import roc_picker.datacard
from roc_picker.kaplan_meier_likelihood import KaplanMeierPlotConfig

# Parse datacard and create Kaplan-Meier likelihood object
datacard = roc_picker.datacard.Datacard.parse_datacard("your_datacard.txt")
kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

# Calculate p-values
best_probabilities, p_values = kml.survival_probabilities_pvalues_greenwood(
    times_for_plot=kml.times_for_plot,
    null_survival_probability=0.5,  # Test against H0: S(t) = 0.5
    binomial_only=True,
)

# Create plot with Greenwood confidence intervals and p-values
config = KaplanMeierPlotConfig(
    include_exponential_greenwood=True,
    include_greenwood_pvalues=True,
    null_survival_probability=0.5,
    pvalue_significance_threshold=0.05,  # Significance level for annotations
    saveas="kaplan_meier_with_pvalues.png"
)

results = kml.plot(config=config)
```

### Command Line

```bash
# Basic usage with Greenwood intervals and p-values
kombine datacard.txt output.png --include-exponential-greenwood --include-greenwood-pvalues

# Custom null hypothesis and significance threshold
kombine datacard.txt output.png \
    --include-exponential-greenwood \
    --include-greenwood-pvalues \
    --null-survival-probability 0.7 \
    --pvalue-significance-threshold 0.01
```

## Technical Details

- Uses log(-log(S)) transformation for approximate normality
- Standard error calculated using Greenwood's formula
- Two-sided z-test for hypothesis testing
- Only works with `binomial_only=True` (same as Greenwood confidence intervals)
- P-values < significance threshold are annotated on the plot with red arrows

## New Parameters

- `--include-greenwood-pvalues`: Enable p-value calculation and display
- `--null-survival-probability`: Null hypothesis value for S(t) (default: 0.5)
- `--pvalue-significance-threshold`: Threshold for highlighting significant p-values (default: 0.05)

## Integration

P-values are automatically calculated and displayed when both `--include-exponential-greenwood` and `--include-greenwood-pvalues` flags are used together, as requested in the issue.