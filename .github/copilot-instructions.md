# ROC Picker Copilot Instructions

## Repository Overview

ROC Picker is a Python package for propagating statistical and systematic uncertainties in biomedical analysis. It provides tools for ROC curve analysis using various methods including discrete optimization, Monte Carlo with systematics, and Kaplan-Meier likelihood methods.

**Repository size**: ~6,000 lines of Python code across 12 main modules
**Languages**: Python (primary), LaTeX (documentation), Shell scripts (build automation)
**Key frameworks**: NumPy/SciPy for numerical computing, Matplotlib for plotting, Gurobi for optimization
**Target runtime**: Python 3.11+ (tested with 3.12)

## Critical Build Information

### Installation and Environment Setup

**Always run installation in this exact sequence:**
1. `pip install .` - installs the package and dependencies
2. `pip install pylint pyflakes texoutparse` - installs required linting and development tools
3. `rm -rf build` - clean up build artifacts (if needed for clean builds)

**Dependencies installed**: gurobipy, matplotlib, numpy, scipy>=1.15
**Additional development tools required**: pylint, pyflakes, texoutparse (for linting and code quality checks)

### Command Line Interface

**Available CLI tools** (installed via pip install):
```bash
rocpicker_discrete datacard.txt [options]  # Discrete ROC analysis
rocpicker_mc datacard.txt output.pdf [options]  # Monte Carlo systematics  
rocpicker_delta_functions datacard.txt [options]  # Delta function analysis
kombine datacard.txt output.pdf [options]  # KM likelihood (requires full Gurobi)
kombine_twogroups datacard.txt output.pdf [options]  # KM two-group analysis (requires full Gurobi)
```

**Basic functionality test** (~4 seconds):
```python
from roc_picker.discrete import DiscreteROC
responders = [1, 1, 2, 3, 9, 10]
nonresponders = [2, 3, 3, 4, 6, 8, 9, 10, 10, 10, 10, 11, 12, 13]
result = DiscreteROC(responders=responders, nonresponders=nonresponders).make_plots(npoints=100, yupperlim=20, show=False)
# Returns dict with keys: ['nominal', 'm68', 'p68', 'm95', 'p95']
```

### Gurobi License Limitation (CRITICAL)

**WARNING**: The Kaplan-Meier likelihood methods require Gurobi optimizer. The restricted Gurobi license causes failures on large models with the error:
```
GurobiError: Model too large for size-limited license
```

**What works without full Gurobi license**:
- Discrete ROC analysis (`test_discrete.py`, `rocpicker_discrete`)
- Monte Carlo systematics (`test_systematics_mc.py`, `rocpicker_mc`) 
- Delta functions (`rocpicker_delta_functions`)
- Discrete optimization (`test_discrete_optimization.py`)

**What requires full Gurobi license**:
- Kaplan-Meier likelihood tests (`test_km_likelihood.py`)
- `kombine` and `kombine_twogroups` command-line tools

### Gurobi License Configuration for Copilot Agents

**IMPORTANT**: GitHub Copilot agents do not have direct access to repository secrets like GitHub Actions workflows do. This is by design for security reasons.

**If you need to test Gurobi functionality as a Copilot agent:**

1. **Limited license testing**: Most functionality works with Gurobi's restricted license that comes with the gurobipy package installation. Only large models will fail.

2. **For full license access**: The repository owner would need to provide Gurobi license information through alternative means:
   - **Option A**: Provide a temporary license file that can be placed in the repository (not recommended for security)
   - **Option B**: Set up environment variables manually in the session:
     ```bash
     export GUROBI_WLSACCESSID="your_access_id"
     export GUROBI_WLSSECRET="your_secret" 
     export GUROBI_LICENSEID="your_license_id"
     cat <<EOF > ~/gurobi.lic
     WLSACCESSID=$GUROBI_WLSACCESSID
     WLSSECRET=$GUROBI_WLSSECRET
     LICENSEID=$GUROBI_LICENSEID
     EOF
     ```
   - **Option C**: Focus testing on methods that don't require full Gurobi license

**Recommendation**: Start with testing the methods that work with restricted license, and document any Gurobi license limitations as expected behavior. The repository owner can manually test the full Gurobi functionality if needed.

**Note**: The GitHub Actions workflow uses these secrets for Gurobi WLS license:
- `GUROBI_WLSACCESSID`
- `GUROBI_WLSSECRET` 
- `GUROBI_LICENSEID`

### Testing Commands

**Prerequisites**: Ensure you have installed linting and development tools: `pip install pylint pyflakes texoutparse`

**Run tests in this order** (some will fail due to Gurobi license):
```bash
# These work with restricted Gurobi license:
python -m test.test_discrete_optimization  # ~90 seconds
python -m test.test_discrete              # ~5 seconds  
python -m test.test_systematics_mc        # ~10 seconds
python test/test_continuous_distributions.py  # ~5 seconds

# This fails with restricted Gurobi license:
python -m test.test_km_likelihood  # Will fail with "Model too large for size-limited license"
```

**Test validation**: Tests use reference JSON files in `test/reference/` to validate numerical outputs against expected results with specified tolerances (typically 1e-6 absolute/relative tolerance).

### Linting and Code Quality

**Always run linting before committing**:
```bash
python -m pyflakes .        # Should pass (may show f-string warnings in generated docs/)
python -m pylint .          # Should score ~10/10
```

**Linting configuration**: `.pylintrc` configures 2-space indentation, ignores invalid variable names (C0103), and has scipy.special whitelist.

### Documentation Build

**Documentation tools needed**:
```bash
pip install jupytext nbconvert  # For Jupyter notebook conversion
```

**Optional tools for comprehensive development environment** (matching GitHub Actions):
```bash
pip install ipykernel lifelines  # Additional tools used in CI
```

**Build documentation**:
```bash
cd docs
jupytext --sync *.md        # Convert markdown to notebooks
./compile_plots.sh          # Generate plots (requires full Gurobi license for some plots)
```

**LaTeX compilation** (if needed):
- Uses XeLaTeX with biber for bibliography
- Main documents: `02_rocpicker.tex`, `08_kaplan_meier_paper.tex`
- Configuration: `.latexmkrc` sets XeLaTeX as default

## Project Architecture

### Core Module Structure (`roc_picker/`)

**Main entry point**: `datacard.py` (1,184 lines) - Parses datacard configuration files and orchestrates analysis
**Key analysis modules**:
- `discrete.py` (207 lines) - Discrete ROC optimization
- `systematics_mc.py` (567 lines) - Monte Carlo with systematics
- `kaplan_meier_likelihood.py` (868 lines) - KM likelihood (requires Gurobi)
- `kaplan_meier_MINLP.py` (1,814 lines) - Mixed-integer optimization (requires Gurobi)
- `delta_functions.py` (187 lines) - Delta function analysis
- `discrete_optimization.py` (315 lines) - Core optimization routines

**Supporting modules**:
- `discrete_base.py` (372 lines) - Base class for discrete methods
- `continuous_distributions.py` (213 lines) - Continuous distribution handling
- `kaplan_meier.py` (360 lines) - KM curve utilities
- `utilities.py` (32 lines) - Common utilities

### Test Structure (`test/`)

**Test modules**: Each major component has corresponding test file
**Test data**: `test/datacards/` contains example configurations
- `simple_examples/` - Basic test cases
- `lung/` - Real biomedical data examples

**Reference data**: `test/reference/` contains JSON files with expected numerical results

### Configuration Files

**Build**: `setup.py` + `pyproject.toml` (setuptools with setuptools_scm for versioning)
**Linting**: `.pylintrc` 
**Git**: `.gitignore` excludes build artifacts, PDFs, logs, notebooks
**CI/CD**: `.github/workflows/test_and_docs.yml` runs full test and documentation pipeline

### GitHub Actions Workflow

The CI pipeline (`.github/workflows/test_and_docs.yml`) runs:
1. Python 3.11 setup and pip install
2. pyflakes and pylint checking
3. Jupyter notebook conversion
4. Pyright type checking
5. Gurobi installation and license configuration (with secrets)
6. README code example test
7. All test modules
8. Plot compilation and LaTeX document generation
9. Artifact upload

**Timing**: Full CI run takes ~10-15 minutes with LaTeX compilation

### Documentation (`docs/`)

**Structure**: Mixed LaTeX papers + Jupyter notebooks
**Build script**: `compile_plots.sh` generates all figures
**Output**: GitHub Actions compiles to PDF and uploads as artifacts

## Datacard Format

The package uses a text-based datacard format (similar to Higgs Combine Tool) to specify analysis configurations. Example format:
```
observable_type  fixed         
------------
# List of patients
------------
bin inclusive inclusive ...
response responder non-responder ...
observable 1 2 3 ...
```

**Key datacard locations**:
- `test/datacards/simple_examples/` - Basic examples for testing
- `test/datacards/lung/` - Real biomedical data for realistic testing

**Example datacard files**: `example_roc.txt`, `poisson_ratio_km.txt`, `symmetric_roc.txt`

## Workflow for Code Changes

1. **Installation**: 
   ```bash
   pip install .
   pip install pylint pyflakes texoutparse
   ```
2. **Linting**: `python -m pyflakes . && python -m pylint .`
3. **Test relevant modules**: Run appropriate test modules (avoid KM tests if no Gurobi license)
4. **Validation**: Use reference data tests to ensure numerical stability
5. **Documentation**: If changing docs, run `jupytext --sync docs/*.md`

## Troubleshooting

**Gurobi license errors**: Expected for KM likelihood methods - document as known limitation
**Build directory issues**: Run `rm -rf build` before reinstalling
**Linting f-string warnings**: Acceptable in generated `docs/*.py` files from jupytext
**Test timing**: `test_discrete_optimization` is slow (~90s) - this is normal
**Network timeouts during install**: Retry pip install if PyPI connection fails
**Missing linting tools**: If you get "No module named pylint", "No module named pyflakes", or "No module named texoutparse", make sure to run `pip install pylint pyflakes texoutparse` after the main installation

## Common Patterns

- **Error handling**: Code uses numpy testing utilities for numerical comparisons with specified tolerances
- **Reference testing**: Tests compare outputs to reference JSON files using `np.testing.assert_allclose()`
- **Configuration**: Heavily configuration-driven via datacard files (text format similar to Higgs Combine Tool)
- **Plotting**: Matplotlib-based with configurable output formats (PDF default)
- **CLI**: Entry points defined in setup.py provide command-line interfaces
- **Confidence intervals**: Results include nominal, ±68%, and ±95% confidence levels (keys: 'nominal', 'p68', 'm68', 'p95', 'm95')
- **Data validation**: `flip_sign` parameter allows inverting analysis for AUC < 0.5 cases

## File Locations Quick Reference

**Source code**: `roc_picker/` (12 Python modules)
**Tests**: `test/` with subdirectories:
- `test/datacards/simple_examples/` - Basic test configurations  
- `test/datacards/lung/` - Real biomedical data examples
- `test/reference/` - Expected numerical results (JSON format)
**Documentation**: `docs/` (LaTeX + Jupyter notebooks)
**Build artifacts**: Excluded by `.gitignore` (build/, *.pdf, *.html, *.ipynb)

## Trust These Instructions

These instructions have been validated by running the build, test, and linting processes. Only search for additional information if these instructions are incomplete or found to be incorrect.