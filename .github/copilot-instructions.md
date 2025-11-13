# ROC Picker and KoMbine Copilot Instructions

## Repository Overview

This repository contains two distinct Python packages for biomedical analysis:

- **ROC Picker**: ROC curve analysis with statistical and systematic uncertainty propagation
- **KoMbine**: Kaplan-Meier curve analysis with likelihood-based uncertainty methods

**Repository structure**: ~6,000 lines of Python code across two main packages
**Languages**: Python (primary), LaTeX (documentation), Shell scripts (build automation)
**Key frameworks**: NumPy/SciPy for numerical computing, Matplotlib for plotting, Gurobi for optimization (KoMbine only)
**Target runtime**: Python 3.11+ (tested with 3.12)

## Package Structure

### Core Packages
- **`roc_picker/`**: ROC analysis functionality
  - `discrete.py`, `systematics_mc.py`, `delta_functions.py` - Main ROC methods
  - `discrete_base.py`, `continuous_distributions.py` - Supporting classes
  - `command_line_interface.py` - ROC-specific CLI functions
  - `datacard.py` - Re-exports Datacard from kombine for compatibility
  
- **`kombine/`**: Kaplan-Meier analysis functionality
  - `kaplan_meier*.py` - KM curve and likelihood methods
  - `discrete_optimization.py`, `utilities.py` - Optimization utilities (used by KM methods)
  - `datacard.py` - Main Datacard class (core data parsing)
  - `command_line_interface.py` - KM-specific CLI functions

### Test Structure
- **`test/roc_picker/`**: ROC Picker tests, datacards, and reference data
- **`test/kombine/`**: KoMbine tests, datacards, and reference data
- **Shared files**: `utility_testing_functions.py`, `test_continuous_distributions.py` (copied to both)

### Documentation Structure  
- **`docs/roc_picker/`**: ROC Picker documentation, LaTeX files, and plotting scripts
- **`docs/kombine/`**: KoMbine documentation, LaTeX files, and plotting scripts
- Each has independent numbering starting from 01, with separate `compile_*_plots.sh` scripts

**KoMbine documentation files**:
- `01_table_of_contents.md` - Index of all documentation files
- `02_kombine.tex` - LaTeX paper with mathematical details (JSS submission)
- `03_kaplan_meier_example.md` - Jupyter notebook showing Python API usage examples
- `04_compare_to_lifelines.md` - Jupyter notebook comparing to `lifelines` package
- `05_command_line_interface.md` - **Pure Markdown** (no Python cells) documenting all CLI options for `kombine` and `kombine_twogroups` commands

**Documentation style guidelines**:
- Files `03_*.md` and `04_*.md` are Jupytext notebooks with Python cells for interactive examples
- File `05_command_line_interface.md` is pure Markdown without Python cells - it documents the CLI, not the Python API
- All CLI options must be documented in `05_command_line_interface.md` and verified by `test/kombine/test_cli_documentation.py`
- When adding new CLI arguments, update `05_command_line_interface.md` and run the documentation test

## Critical Build Information

### Installation and Environment Setup

**Always run installation in this exact sequence:**
1. `pip install .` - installs both roc_picker and kombine packages
2. `pip install pylint pyflakes texoutparse` - installs required linting and development tools
3. `rm -rf build` - clean up build artifacts (if needed for clean builds)

**Dependencies installed**: gurobipy, matplotlib, numpy, scipy>=1.15
**Additional development tools required**: pylint, pyflakes, texoutparse

### Command Line Interface

**ROC Picker CLI tools**:
```bash
rocpicker_discrete datacard.txt [options]     # Discrete ROC analysis
rocpicker_mc datacard.txt output.pdf [options] # Monte Carlo systematics  
rocpicker_delta_functions datacard.txt [options] # Delta function analysis
```

**KoMbine CLI tools** (require Gurobi):
```bash
kombine datacard.txt output.pdf [options]      # KM likelihood 
kombine_twogroups datacard.txt output.pdf [options] # KM two-group analysis
```

### Basic Functionality Tests

**ROC Picker test** (~4 seconds):
```python
from roc_picker.discrete import DiscreteROC
responders = [1, 1, 2, 3, 9, 10]
nonresponders = [2, 3, 3, 4, 6, 8, 9, 10, 10, 10, 10, 11, 12, 13]
result = DiscreteROC(responders=responders, nonresponders=nonresponders).make_plots(npoints=100, yupperlim=20, show=False)
# Returns dict with keys: ['nominal', 'm68', 'p68', 'm95', 'p95']
```

**KoMbine test** (~4 seconds):
```python
from kombine.datacard import Datacard
datacard = Datacard.parse_datacard("test/kombine/datacards/simple_examples/simple_km_few_deaths.txt")
kml = datacard.km_likelihood(parameter_min=-float('inf'), parameter_max=0.45)
# Returns KaplanMeierLikelihood object
```

### Gurobi License Limitation (CRITICAL for KoMbine)

**WARNING**: KoMbine's Kaplan-Meier likelihood methods require Gurobi optimizer. The restricted Gurobi license causes failures on large models with the error:
```
GurobiError: Model too large for size-limited license
```

**What works without full Gurobi license**:
- All ROC Picker functionality (`test/roc_picker/` tests)
- Basic KoMbine discrete optimization (`test/kombine/test_discrete_optimization.py`)
- Small KoMbine datasets with few patients

**What requires full Gurobi license**:
- Large KoMbine likelihood tests (`test/kombine/test_km_likelihood.py` with many patients)
- Full KoMbine documentation compilation (some plots)

## Testing Commands

### ROC Picker Tests (no Gurobi license required)
```bash
python -m test.roc_picker.test_discrete              # ~5 seconds
python -m test.roc_picker.test_systematics_mc        # ~10 seconds
python test/roc_picker/test_continuous_distributions.py  # ~5 seconds
```

### KoMbine Tests
```bash
python -m test.kombine.test_discrete_optimization    # ~90 seconds, works with restricted license
python -m test.kombine.test_km_likelihood           # May fail with "Model too large" on restricted license
```

### Linting and Code Quality
```bash
python -m pyflakes .        # Should pass (may show f-string warnings in generated docs/)
python -m pylint .          # Should score ~10/10
```

## GitHub Actions Workflows

The repository now uses three separate workflows:

1. **`.github/workflows/linting.yml`**: Linting and type checking (pyflakes, pylint, pyright)
2. **`.github/workflows/test_roc_picker.yml`**: ROC Picker testing and documentation
3. **`.github/workflows/test_kombine.yml`**: KoMbine testing and documentation (requires Gurobi secrets)

**Gurobi secrets** (for KoMbine workflow): `GUROBI_WLSACCESSID`, `GUROBI_WLSSECRET`, `GUROBI_LICENSEID`

## Development Workflow

### For ROC Picker changes:
1. **Installation**: `pip install . && pip install pylint pyflakes texoutparse`
2. **Linting**: `python -m pyflakes . && python -m pylint .`
3. **Testing**: Run ROC Picker tests from `test/roc_picker/`
4. **Documentation**: Use `docs/roc_picker/compile_roc_plots.sh`

### For KoMbine changes:
1. **Installation**: Same as above
2. **Linting**: Same as above  
3. **Testing**: Run KoMbine tests from `test/kombine/` (may need Gurobi license)
4. **Documentation**: Use `docs/kombine/compile_km_plots.sh`

### For cross-package changes:
- Test both packages since ROC Picker imports from KoMbine
- The `Datacard` class lives in `kombine/datacard.py` but is re-exported by `roc_picker/datacard.py`

## File Locations Quick Reference

**ROC Picker code**: `roc_picker/` (6 Python modules)
**KoMbine code**: `kombine/` (8 Python modules)
**ROC Picker tests**: `test/roc_picker/` with `datacards/`, `reference/`, `test_output/`
**KoMbine tests**: `test/kombine/` with `datacards/`, `reference/`, `test_output/`
**ROC Picker docs**: `docs/roc_picker/` (LaTeX + Jupyter notebooks)
**KoMbine docs**: `docs/kombine/` (LaTeX + Jupyter notebooks, includes JSS class files)

## Common Patterns

- **Error handling**: Code uses numpy testing utilities for numerical comparisons with specified tolerances
- **Reference testing**: Tests compare outputs to reference JSON files using `np.testing.assert_allclose()`
- **Configuration**: Heavily configuration-driven via datacard files (text format similar to Higgs Combine Tool)
- **Plotting**: Matplotlib-based with configurable output formats (PDF default)
- **CLI**: Entry points defined in pyproject.toml provide command-line interfaces
- **Confidence intervals**: Results include nominal, ±68%, and ±95% confidence levels (keys: 'nominal', 'p68', 'm68', 'p95', 'm95')

## Troubleshooting

**Import issues**: Remember that `roc_picker.datacard` now re-exports from `kombine.datacard`
**Missing test data**: Check if you're in the right test subdirectory (`test/roc_picker/` vs `test/kombine/`)
**Gurobi license errors**: Expected for large KoMbine models - document as known limitation
**Build directory issues**: Run `rm -rf build` before reinstalling
**Missing linting tools**: Run `pip install pylint pyflakes texoutparse` after main installation