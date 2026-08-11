# ROC Picker and KoMbine Copilot Instructions

## Repository Overview

This repository contains two distinct Python packages for biomedical analysis:

- **ROC Picker**: ROC curve analysis with statistical and systematic uncertainty propagation
- **KoMbine**: Kaplan-Meier curve analysis with likelihood-based uncertainty methods

**Repository structure**: ~6,000 lines of Python code across two main packages
**Languages**: Python (primary), LaTeX (documentation), Shell scripts (build automation)
**Key frameworks**: NumPy/SciPy for numerical computing, Matplotlib for plotting, Gurobi for optimization (KoMbine only)
**Target runtime**: Python 3.11+ (tested with 3.12)

## Code Style Guidelines

**CRITICAL: Indentation and Formatting**
- **Use 2-space indentation** for all Python code (not 4 spaces or tabs)
- **No trailing whitespace** - remove all trailing spaces at the end of lines
- **Always run pylint after editing** and fix any errors before committing
- Run `python -m pylint <file>` to check for issues
- **Pylint must return perfect results (10.00/10)** - all warnings and errors must be addressed
- Nontrivial errors that would require major refactoring can be ignored via `# pylint: disable=error-name` comments, but this is less preferable
- Target pylint score: 10.00/10
- **Pyright must not have any errors** - run `pyright <file>` to check for type errors
- Before running pyright, install any missing packages (e.g., `pip install matplotlib numpy scipy`) to properly evaluate real type errors, as missing import errors can mask actual issues
- Expected import errors in CI environments without packages installed are acceptable
- Fix all other pyright errors before committing

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
  - `comparisons/` - published alternatives (Yi weights, Küchenhoff MC-SIMEX)

### Test Structure
- **`test/roc_picker/`**: ROC Picker tests, datacards, and reference data
- **`test/kombine/`**: KoMbine tests, datacards, and reference data
- **Shared utilities**: `test/utility_testing_functions.py` (shared helpers used by both)
- `test_continuous_distributions.py` lives in `test/roc_picker/`

### Documentation Structure
- **`docs/roc_picker/`**: ROC Picker documentation, LaTeX files, and plotting scripts
- **`docs/kombine/`**: KoMbine documentation, LaTeX files, and plotting scripts
- Each has independent numbering starting from 01
- ROC Picker uses `docs/roc_picker/compile_roc_plots.sh`; KoMbine uses `python -m docs.kombine.compile_km_plots` (Python script, not shell)

**KoMbine documentation files**:
- `01_table_of_contents.md` - Index of all documentation files (pure Markdown, synced with Jupytext)
- `02_kombine.tex` - LaTeX paper with mathematical details (JSS submission)
- `03_kaplan_meier_example.md` - Jupyter notebook showing Python API usage examples
- `04_analysis_demo.md` - Jupyter notebook with analysis demonstration
- `05_compare_lifelines_greenwood.md` - Jupyter notebook comparing to `lifelines` and Greenwood method
- `06_compare_thomas_grunkemeier.md` - Jupyter notebook comparing Thomas–Grunkemeier intervals
- `07_previous_methods_comparison.md` - Jupyter notebook comparing Yi, MC-SIMEX, and KoMbine
- `08_command_line_interface.md` - **Pure Markdown** (no Python cells) documenting all CLI options for `kombine` and `kombine_twogroups` commands (synced with Jupytext)

**Documentation style guidelines**:
- Files `03_*.md` through `07_*.md` are Jupytext notebooks with Python cells for interactive examples
- Files `01_*.md` and `08_*.md` are pure Markdown with Jupytext headers but no Python cells
- All documentation markdown files must have Jupytext headers to allow `jupytext --sync` to process them
- All CLI options must be documented in `08_command_line_interface.md` and verified by `test/kombine/test_ci_and_documentation.py`
- When adding new CLI arguments, update `08_command_line_interface.md` and run the documentation test
- The table of contents (`01_table_of_contents.md`) must list all numbered documentation files

**Compiling KoMbine LaTeX documentation (`02_kombine.tex`)**:
- **CRITICAL**: Before compiling LaTeX, you must either:
  1. Run `python -m docs.kombine.compile_km_plots --testing` (from repo root) to generate test plots (works with restricted Gurobi license), OR
  2. Temporarily remove figure `\includegraphics` commands from the LaTeX file (do NOT commit this removal)
- Without plots, LaTeX compilation will fail with missing file errors
- The `--testing` flag generates smaller datasets: 10 patients for p-value plots, 3 patients for lung dataset
- Full plot generation requires a full Gurobi license

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

### Gurobi License Setup (CRITICAL for KoMbine)

**Gurobi License Configuration**:
KoMbine's Kaplan-Meier likelihood methods require Gurobi optimizer. The Copilot GitHub Actions environment has access to Gurobi Web License Service (WLS) academic license via environment variables:

- `GUROBI_WLSACCESSID` - WLS access ID
- `GUROBI_WLSSECRET` - WLS secret key
- `GUROBI_LICENSEID` - License ID

**Setting up the unrestricted academic license in Copilot environment**:

Create a license file in the home directory:

```bash
cat <<EOF > ~/gurobi.lic
WLSACCESSID=${GUROBI_WLSACCESSID}
WLSSECRET=${GUROBI_WLSSECRET}
LICENSEID=${GUROBI_LICENSEID}
EOF
```

This activates the WLS academic license (requires network access to `token.gurobi.com`, which is now allowed in the Copilot environment). When you import gurobipy, it will automatically use this license:

```python
import gurobipy
env = gurobipy.Env()  # Uses WLS academic license from ~/gurobi.lic
# Academic license <LICENSE_ID> - for non-commercial use only - registered to <EMAIL>
```

**License capabilities**:
With the unrestricted WLS academic license, you can:
- Run all KoMbine tests including `test_km_likelihood.py` with large datasets
- Generate full KoMbine documentation plots without size restrictions
- Run MINLP optimizations on models with hundreds of variables and constraints
- Execute the full lung cancer dataset analysis with 100+ patients

**What works without setting up the license file**:
If you don't create the license file, Gurobi falls back to a temporary restricted license (expires 2027-11-29) that works for:
- All ROC Picker functionality (`test/roc_picker/` tests)
- Basic KoMbine discrete optimization (`test/kombine/test_discrete_optimization.py`)
- Small KoMbine datasets (< 5 patients) for testing plot generation

**CI/CD License Setup**:
In `.github/workflows/ci-cd.yml`, the license is set up similarly using GitHub secrets:
```yaml
- name: Set up Gurobi license
  run: |
    cat <<EOF > ~/gurobi.lic
    WLSACCESSID=${{ secrets.GUROBI_WLSACCESSID }}
    WLSSECRET=${{ secrets.GUROBI_WLSSECRET }}
    LICENSEID=${{ secrets.GUROBI_LICENSEID }}
    EOF
```

**Test Datacards for Development**:

Small datacards for quick testing (work even with restricted license):
- `test/kombine/datacards/simple_examples/simple_km_few_deaths.txt` - 4 timepoints, minimal test case
- `test/kombine/datacards/lung/test_small_dataset/test_lung_cells.txt` - 3 patients for cells plots
- `test/kombine/datacards/lung/test_small_dataset/test_lung_donuts.txt` - 3 patients for DONUTS plots

Production datacards (require unrestricted license):
- `test/kombine/datacards/simple_examples/fixed_km_censoring.txt` - 12 patients
- `test/kombine/datacards/simple_examples/fixed_km_censoring_many_patients.txt` - 100 patients
- `test/kombine/datacards/lung/lung_cells.txt` - Full lung cancer cells dataset
- `test/kombine/datacards/lung/lung_donuts.txt` - Full lung cancer DONUTS dataset

**compile_km_plots.py modes**:
- `--testing`: Uses small test datasets, works with restricted license, generates km_example and greenwood plots only
- Production mode (default): Uses full datasets, requires unrestricted license, generates all plots including p-value comparison and lung_km_RFS
- `--lung-production-panel A B ...`: Mixed mode - specified panels use production data, others use test data (useful for debugging specific panels)

## Testing Commands

### ROC Picker Tests (no Gurobi license required)
```bash
python -m test.roc_picker.test_discrete              # ~5 seconds
python -m test.roc_picker.test_systematics_mc        # ~10 seconds
python test/roc_picker/test_continuous_distributions.py  # ~5 seconds
```

### KoMbine Tests (require unrestricted Gurobi license)
```bash
python -m test.kombine.test_discrete_optimization    # ~90 seconds
python -m test.kombine.test_km_likelihood           # ~2 minutes, requires unrestricted license
python -m test.kombine.test_hazard_ratio
python -m test.kombine.test_km_plotting
python -m test.kombine.test_yi_correction
python -m test.kombine.test_mc_simex
python -m test.kombine.test_discrete_class_observable
python -m test.kombine.test_ci_and_documentation
```

**Note**: Always set up the Gurobi license file (see above) before running KoMbine tests.

### Linting and Code Quality
```bash
python -m pyflakes .        # Should pass (may show f-string warnings in generated docs/)
python -m pylint .          # Should score ~10/10
```

## GitHub Actions Workflows

The repository uses a single combined workflow:

- **`.github/workflows/ci-cd.yml`**: Linting, type checking, ROC Picker and KoMbine testing, and documentation (requires Gurobi secrets)

**Gurobi secrets**: `GUROBI_WLSACCESSID`, `GUROBI_WLSSECRET`, `GUROBI_LICENSEID`

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
4. **Documentation**: Use `python -m docs.kombine.compile_km_plots` (from repo root)

### For cross-package changes:
- Test both packages since ROC Picker imports from KoMbine
- The `Datacard` class lives in `kombine/datacard.py` but is re-exported by `roc_picker/datacard.py`

## File Locations Quick Reference

**ROC Picker code**: `roc_picker/` (7 Python modules + `__init__.py`)
**KoMbine code**: `kombine/` (10 Python modules + `__init__.py`)
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
- **Exception handling**: Do not catch exceptions and convert them to warnings. If there is an error, it should be raised as an error so the user is aware of the problem.

## Troubleshooting

**Import issues**: Remember that `roc_picker.datacard` now re-exports from `kombine.datacard`
**Missing test data**: Check if you're in the right test subdirectory (`test/roc_picker/` vs `test/kombine/`)
**Gurobi license errors**: Expected for large KoMbine models - document as known limitation
**Build directory issues**: Run `rm -rf build` before reinstalling
**Missing linting tools**: Run `pip install pylint pyflakes texoutparse` after main installation