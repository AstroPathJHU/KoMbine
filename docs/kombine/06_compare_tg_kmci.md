---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
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

```python
"""Compare KoMbine binomial-only KM intervals against Thomas-Grunkemeier via R."""

# pylint: disable=wrong-import-position
# pyright: reportMissingImports=false
```

```python
import warnings

warnings.simplefilter("error")
```

# KoMbine (Binomial-Only) vs Thomas and Grunkemeier

This notebook compares KoMbine's binomial-only Kaplan-Meier confidence intervals
to Thomas and Grunkemeier's method as implemented in R package `km.ci`
(`method="grunkemeier"`).

Goal: verify that KoMbine reproduces the same results when only binomial
uncertainty is enabled.


## Requirements

- Python environment with `kombinekm`, `numpy`, `pandas`, `matplotlib`
- R installation with `Rscript` available on PATH
- R packages `survival` and `km.ci`

If R packages are missing, in R run:

```r
install.packages(c("survival", "km.ci"))
```

```python
import importlib.metadata
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kombine.datacard import Datacard
```

```python
rscript_path_candidate = shutil.which("Rscript")
if rscript_path_candidate is None and os.name == "nt":
  # Common default R installation path on Windows.
  windows_candidates = sorted(pathlib.Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"))
  if windows_candidates:
    rscript_path_candidate = str(windows_candidates[-1])

if rscript_path_candidate is None:
  raise RuntimeError(
    "Rscript was not found on PATH. Install R and ensure Rscript is available."
  )

rscript_path = rscript_path_candidate


def run_r_expr(expr):
  """Run an R expression and return stdout text."""
  completed = subprocess.run(
    [rscript_path, "-e", expr],
    check=False,
    capture_output=True,
    text=True,
  )
  if completed.returncode != 0:
    raise RuntimeError(
      "R command failed.\n"
      f"Expression: {expr}\n"
      f"stdout:\n{completed.stdout}\n"
      f"stderr:\n{completed.stderr}"
    )
  return completed.stdout.strip()


# Check required R packages once and fail early with clear instructions.
run_r_expr(
  "if (!requireNamespace('survival', quietly=TRUE) || "
  "!requireNamespace('km.ci', quietly=TRUE)) "
  "stop('Install required packages: survival and km.ci')"
)
```

```python
print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("Rscript:", rscript_path)
print("R:", run_r_expr("cat(R.version.string)"))
print("survival:", run_r_expr("cat(as.character(packageVersion('survival')))"))
print("km.ci:", run_r_expr("cat(as.character(packageVersion('km.ci')))"))

try:
  print("kombinekm:", importlib.metadata.version("kombinekm"))
except importlib.metadata.PackageNotFoundError:
  print("kombinekm: version metadata not found (editable/local install)")
```

```python
here = pathlib.Path(".").resolve()
simple_examples = here.parent.parent / "test" / "kombine" / "datacards" / "simple_examples"

comparison_datacards = [
  simple_examples / "fixed_km_censoring.txt",
  simple_examples / "simple_km_few_deaths.txt",
]

for dc_path in comparison_datacards:
  print(dc_path)
```

```python
def tg_grunkemeier_from_r(durations, events, conf_level=0.95):
  """Run Thomas and Grunkemeier CIs via R km.ci(method='grunkemeier')."""
  if durations.shape != events.shape:
    raise ValueError("durations and events must have identical shape")

  with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = pathlib.Path(tmpdir)
    input_csv = tmpdir_path / "input.csv"
    output_csv = tmpdir_path / "output.csv"
    script_path = tmpdir_path / "run_tg.R"

    pd.DataFrame(
      {
        "time": durations.astype(float),
        "status": events.astype(int),
      }
    ).to_csv(input_csv, index=False)

    script_path.write_text(
      textwrap.dedent(
        """
        args <- commandArgs(trailingOnly = TRUE)
        input_csv <- args[1]
        output_csv <- args[2]
        conf_level <- as.numeric(args[3])

        suppressPackageStartupMessages(library(survival))
        suppressPackageStartupMessages(library(km.ci))

        dat <- read.csv(input_csv)
        fit <- survfit(Surv(time, status) ~ 1, data = dat)
        fit_tg <- km.ci(fit, conf.level = conf_level, method = "grunkemeier")

        result <- data.frame(
          time = fit_tg$time,
          surv_tg = fit_tg$surv,
          lower_tg = fit_tg$lower,
          upper_tg = fit_tg$upper,
          n_event = fit_tg$n.event
        )

        result <- result[result$n_event > 0, c("time", "surv_tg", "lower_tg", "upper_tg")]
        write.csv(result, output_csv, row.names = FALSE)
        """
      ).strip()
      + "\n",
      encoding="utf-8",
    )

    completed = subprocess.run(
      [rscript_path, str(script_path), str(input_csv), str(output_csv), str(conf_level)],
      check=False,
      capture_output=True,
      text=True,
    )
    if completed.returncode != 0:
      raise RuntimeError(
        "Rscript failed while running Thomas-Grunkemeier comparison.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
      )

    return pd.read_csv(output_csv)
```

```python
def kombine_binomial_only(datacard_path, conf_level=0.95):
  """Run KoMbine with binomial-only uncertainty on all patients."""
  datacard = Datacard.parse_datacard(datacard_path)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  times = np.asarray(sorted(kml.patient_death_times), dtype=float)
  best, ci = kml.survival_probabilities_likelihood(
    CLs=[conf_level],
    times_for_plot=list(times),
    binomial_only=True,
  )

  best = np.asarray(best, dtype=float)
  ci = np.asarray(ci, dtype=float)

  kombine_df = pd.DataFrame(
    {
      "time": times,
      "surv_kombine": best,
      "lower_kombine": ci[:, 0, 0],
      "upper_kombine": ci[:, 0, 1],
    }
  )

  return kombine_df, datacard
```

```python
def compare_tg_vs_kombine(datacard_path, conf_level=0.95, atol_surv=1e-6, atol_ci=2e-5):
  """Compare TG (R) and KoMbine (Python), return merged table and summary."""
  kombine_df, datacard = kombine_binomial_only(datacard_path, conf_level=conf_level)

  durations = np.asarray([patient.time for patient in datacard.patients], dtype=float)
  events = np.asarray([0 if patient.censored else 1 for patient in datacard.patients], dtype=int)
  tg_df = tg_grunkemeier_from_r(durations, events, conf_level=conf_level)

  merged = kombine_df.merge(tg_df, on="time", how="outer", indicator=True, sort=True)
  if not np.all(merged["_merge"] == "both"):
    missing = merged.loc[merged["_merge"] != "both", ["time", "_merge"]]
    raise ValueError(
      "Time grids do not match between KoMbine and TG outputs. "
      f"Unmatched rows:\n{missing.to_string(index=False)}"
    )
  merged = merged.drop(columns="_merge")

  merged["delta_surv"] = merged["surv_kombine"] - merged["surv_tg"]

  finite_lower = np.isfinite(merged["lower_kombine"]) & np.isfinite(merged["lower_tg"])
  finite_upper = np.isfinite(merged["upper_kombine"]) & np.isfinite(merged["upper_tg"])

  merged["delta_lower"] = np.where(
    finite_lower,
    merged["lower_kombine"] - merged["lower_tg"],
    np.nan,
  )
  merged["delta_upper"] = np.where(
    finite_upper,
    merged["upper_kombine"] - merged["upper_tg"],
    np.nan,
  )

  np.testing.assert_allclose(merged["surv_kombine"], merged["surv_tg"], rtol=0.0, atol=atol_surv)
  if np.any(finite_lower):
    np.testing.assert_allclose(
      merged.loc[finite_lower, "lower_kombine"],
      merged.loc[finite_lower, "lower_tg"],
      rtol=0.0,
      atol=atol_ci,
    )
  if np.any(finite_upper):
    np.testing.assert_allclose(
      merged.loc[finite_upper, "upper_kombine"],
      merged.loc[finite_upper, "upper_tg"],
      rtol=0.0,
      atol=atol_ci,
    )

  def nanmax_abs_or_zero(values):
    arr = np.asarray(values, dtype=float)
    if np.all(~np.isfinite(arr)):
      return 0.0
    return float(np.nanmax(np.abs(arr)))

  result_summary = {
    "datacard": datacard_path.name,
    "n_timepoints": len(merged),
    "n_finite_lower_points": int(np.sum(finite_lower)),
    "n_finite_upper_points": int(np.sum(finite_upper)),
    "max_abs_delta_surv": nanmax_abs_or_zero(merged["delta_surv"]),
    "max_abs_delta_lower": nanmax_abs_or_zero(merged["delta_lower"]),
    "max_abs_delta_upper": nanmax_abs_or_zero(merged["delta_upper"]),
  }

  return merged, result_summary
```

```python
comparison_tables = {}
comparison_summaries = []

for dc_path in comparison_datacards:
  merged_table, one_summary = compare_tg_vs_kombine(dc_path, conf_level=0.95)
  comparison_tables[dc_path.name] = merged_table
  comparison_summaries.append(one_summary)

  print(f"[PASS] {dc_path.name}")
  print(f"  timepoints: {one_summary['n_timepoints']}")
  print(f"  finite lower CI points compared: {one_summary['n_finite_lower_points']}")
  print(f"  finite upper CI points compared: {one_summary['n_finite_upper_points']}")
  print(f"  max |delta surv| : {one_summary['max_abs_delta_surv']:.3e}")
  print(f"  max |delta lower|: {one_summary['max_abs_delta_lower']:.3e}")
  print(f"  max |delta upper|: {one_summary['max_abs_delta_upper']:.3e}")
```

```python
summary_df = pd.DataFrame(comparison_summaries)
print(summary_df.to_string(index=False))
```

## Example row-level comparison table

This table shows KoMbine and Thomas-Grunkemeier values side-by-side at each
event time, including deltas. At boundaries where one method reports an
undefined CI endpoint (NA/NaN), those CI points are excluded from strict
numerical assertions and counted separately in the summary.

```python
example_name = comparison_datacards[0].name
example_table = comparison_tables[example_name]
print(example_table.to_string(index=False))
```

```python
def plot_overlay(comparison_df, title):
  """Overlay survival curves and CIs from KoMbine and TG for visual confirmation."""
  t = comparison_df["time"].to_numpy()

  t_plot = np.r_[0.0, t]

  surv_k = np.r_[1.0, comparison_df["surv_kombine"].to_numpy()]
  lower_k = np.r_[1.0, comparison_df["lower_kombine"].to_numpy()]
  upper_k = np.r_[1.0, comparison_df["upper_kombine"].to_numpy()]

  surv_tg = np.r_[1.0, comparison_df["surv_tg"].to_numpy()]
  lower_tg = np.r_[1.0, comparison_df["lower_tg"].to_numpy()]
  upper_tg = np.r_[1.0, comparison_df["upper_tg"].to_numpy()]

  plt.figure(figsize=(9, 5.5))
  plt.step(
    t_plot,
    surv_k,
    where="post",
    color="tab:blue",
    lw=2.2,
    label="KoMbine (binomial only)",
  )
  plt.step(
    t_plot,
    surv_tg,
    where="post",
    color="tab:orange",
    lw=1.8,
    linestyle="--",
    label="TG via R km.ci",
  )

  plt.step(t_plot, lower_k, where="post", color="tab:blue", alpha=0.75, lw=1.2)
  plt.step(t_plot, upper_k, where="post", color="tab:blue", alpha=0.75, lw=1.2)

  plt.step(
    t_plot,
    lower_tg,
    where="post",
    color="tab:orange",
    alpha=0.75,
    lw=1.2,
    linestyle="--",
  )
  plt.step(
    t_plot,
    upper_tg,
    where="post",
    color="tab:orange",
    alpha=0.75,
    lw=1.2,
    linestyle="--",
  )

  plt.ylim(-0.05, 1.05)
  plt.xlabel("Time")
  plt.ylabel("Survival probability")
  plt.title(title)
  plt.legend()
  plt.tight_layout()
  plt.show()
```

```python
for dc_path in comparison_datacards:
  one_table = comparison_tables[dc_path.name]
  plot_overlay(one_table, f"KoMbine vs Thomas-Grunkemeier: {dc_path.name}")
```

## Conclusion

If all assertion checks above pass, then on these datasets KoMbine with
`binomial_only=True` reproduces Thomas and Grunkemeier results from `km.ci`
(up to numerical tolerance at machine precision scale).
