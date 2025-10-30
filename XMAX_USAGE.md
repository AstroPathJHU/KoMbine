# X-Axis Range Control for Kaplan-Meier Plots

## Overview

This feature adds the ability to limit the x-axis range of Kaplan-Meier plots by specifying a maximum time value (`xmax`).

## Usage

### Command Line Interface

Both `kombine` and `kombine_twogroups` commands now accept the `--xmax` parameter:

```bash
# Basic usage with xmax
kombine datacard.txt output.pdf --xmax 10.0

# Two-groups plot with xmax
kombine_twogroups datacard.txt output.pdf --parameter-threshold 0.5 --xmax 50.0

# Without xmax (default behavior - full range)
kombine datacard.txt output.pdf
```

### Python API

```python
from kombine.datacard import Datacard
from kombine.kaplan_meier_likelihood import KaplanMeierPlotConfig

# Parse datacard
datacard = Datacard.parse_datacard("datacard.txt")
kml = datacard.km_likelihood()

# Create plot with xmax
config = KaplanMeierPlotConfig(
    xmax=10.0,
    saveas="output.pdf"
)
kml.plot(config=config)

# Or pass xmax directly to plot()
kml.plot(xmax=10.0, saveas="output.pdf")
```

### Accessing Time Points

The new `get_times_for_plot(xmax)` method can be used to get time points with xmax applied:

```python
# Get times for plotting without xmax (full range)
times_full = kml.times_for_plot

# Get times for plotting with xmax
times_limited = kml.get_times_for_plot(xmax=10.0)
```

## Implementation Details

When `xmax` is specified:

1. **Time Point Selection**: The algorithm includes:
   - All death times ≤ xmax
   - The first death time > xmax (for curve continuity and interpolation)
   - xmax itself (to ensure the curve ends exactly at the boundary)

2. **X-Axis Limits**: The matplotlib x-axis is set to [0, xmax] using `ax.set_xlim(0, xmax)`

3. **Curve Continuity**: The survival probability at xmax is calculated using the next time point after xmax, ensuring a smooth curve ending at the boundary

## Edge Cases

The implementation correctly handles:

- **xmax beyond all death times**: Includes all death times and xmax
- **xmax exactly at a death time**: No duplication, includes next time for interpolation
- **xmax before first death time**: Includes only first death time and xmax
- **No xmax specified**: Full-range plot (backward compatible with existing behavior)

## Testing

Comprehensive tests are included in:
- `test/kombine/test_xmax.py` - Basic functionality tests
- `test/kombine/test_xmax_edge_cases.py` - Edge case tests

Run tests with:
```bash
python test/kombine/test_xmax.py
python test/kombine/test_xmax_edge_cases.py
```

## Backward Compatibility

This feature is fully backward compatible:
- Omitting `--xmax` produces plots identical to previous versions
- The cached `times_for_plot` property remains unchanged
- All existing tests continue to pass
