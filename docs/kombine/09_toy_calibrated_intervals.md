---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
    kernelspec:
      display_name: Python 3
      language: python
      name: python3
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
---

```python
# pylint: disable=bad-indentation,line-too-long,missing-module-docstring,redefined-outer-name,wrong-import-order,wrong-import-position,duplicate-code,too-many-locals,too-many-statements,too-many-branches
```

# Toy-calibrated HR intervals and KM bands

Permutation p-values in KoMbine shuffle `(time, censored)` under **no association** ($H=1$). Likelihood scans still cut $\Delta 2\mathrm{NLL}$ at $\chi^2_1$ (1 and 3.84). Those cuts ignore assignment search, so at large measurement error a $\chi^2$ 95% HR interval can exclude $H=1$ while the permutation test does not reject it.

This notebook inverts the same profile LRT with **toys** on the n=20 discrete-class cards from the methods-comparison notebook ($e=0.20$, $0.25$, $0.40$):

1. Pretend a candidate $\theta$ (an $H$, or $S(t)$) is true.
2. Simulate cohorts with biomarkers held fixed (weighted Cox permutation of the observed times for HR; binomial redraws on the observed death-time grid for KM).
3. Ask only whether each toy's $\Delta 2\mathrm{NLL}$ is larger than the observed value (`excess_at_most`), and stop when remaining toys cannot change the 68%/95% decision.
4. Search only interval **endpoints**, starting from a known-inside $H$ (the MLE if toys accept it, otherwise $H=1$, then a log-spaced grid). Interior points are not toy-tested.

The HR figure matches notebook 07's discrete row: KoMbine $\chi^2$ $\Delta 2\mathrm{NLL}$ vs $H$ (0.01–100, 25 scan points) with toy 68%/95% intervals overlaid as shaded ranges. Yi / MC-SIMEX are omitted here; toys calibrate KoMbine, not those approximations.

CI sets `KOMBINE_SKIP_TOY_CALIBRATION=1` and skips the MINLP cells (same pattern as notebook 07 skipping n=50).

## What model dependence the toys add

Coverage is exact only if the toys are draws from the true experiment. $\chi^2$ does not avoid model dependence; it assumes regularity and 1 df instead. These toys swap that for **the DGP we simulate is the one KoMbine fitted.**

Three layers:

1. **The likelihood model itself (already in KoMbine).** Proportional hazards for HR; binomial deaths given risk sets for KM; known measurement-error penalties; one discrete group per patient. Using that model as a simulator is the usual parametric-bootstrap assumption, not a new structural story.

2. **Plug-in of things the likelihood treats as unknown.** The HR generator uses the **observed hard label** vs threshold. That ignores measurement error in the *generation* of times (the fit still pays the measurement penalty). KM binomial draws use the **constrained MLE** at $S(t)=s_0$ (assignments and interval $p_i^s$). Specifying only $S(t^*)=s_0$ does not determine the rest of the curve; the rest is filled in from that profiled fit.

3. **Frozen event-time grid.** We re-pair or redraw deaths on the **observed times**, we do not draw new calendars. Censoring is kept as observed.

**Special case $H=1$.** Uniform permutation of `(time, censored)` is essentially nonparametric given exchangeability. Extra dependence appears only for $H\neq 1$ and for KM.

**What this does not add.** No exponential/Weibull baseline. No redraw of biomarkers.

At large $e$ the plug-in labeling / constrained MLE can be a poor stand-in for the true DGP, so toy coverage can still be off even though $\chi^2$ is also wrong. Do not treat these bands as assumption-free.

```python
import os
import sys
import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt

_repo_root = pathlib.Path(".").resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from kombine.datacard import Datacard

SKIP = bool(os.environ.get("KOMBINE_SKIP_TOY_CALIBRATION"))
N_MAX = 19
RNG = 0
# Local runs use Gurobi's default thread count. CI goldens keep Threads=1.
THREADS = None
N_HR_SCAN = 25
HAZARD_RATIO_MIN = 0.01
HAZARD_RATIO_MAX = 100.0
HAZARD_RATIOS_SCAN = np.logspace(-2, 2, N_HR_SCAN)
DATACARDS = _repo_root / "test" / "kombine" / "datacards" / "simple_examples"
CARDS = {
    "e=0.20": {
        "file": DATACARDS / "discrete_classes_hr_example_moderate.txt",
        "label": "Disc. Classes (e=0.20)",
    },
    "e=0.25": {
        "file": DATACARDS / "discrete_classes_hr_example_large.txt",
        "label": "Disc. Classes (e=0.25)",
    },
    "e=0.40": {
        "file": DATACARDS / "discrete_classes_hr_example_very_large.txt",
        "label": "Disc. Classes (e=0.40)",
    },
}
hr_results = {}

print("skip toy calibration MINLPs:", SKIP)
print("n=20 discrete-class HR mosaic; N_MAX=", N_MAX, "Threads=", THREADS)
```

```python
if SKIP:
    print("Skipping n=20 toy HR mosaic (KOMBINE_SKIP_TOY_CALIBRATION=1).")
else:
    for key, info in CARDS.items():
        started = time.perf_counter()
        dc = Datacard.parse_datacard(info["file"])
        hr_calc = dc.km_hazard_ratio(
            parameter_threshold=1.0,
            parameter_min=-np.inf,
            parameter_max=np.inf,
        )
        chi2 = hr_calc.hazard_ratio_confidence_interval(
            cox_only=False, confidence_level=0.95,
            hazard_ratio_min=HAZARD_RATIO_MIN,
            hazard_ratio_max=HAZARD_RATIO_MAX,
        )
        scan_2nll = []
        for hr in HAZARD_RATIOS_SCAN:
            locked = hr_calc.compute_2nll_at_hazard_ratio(
                float(hr), cox_only=False, Threads=THREADS,
            )
            scan_2nll.append(float(locked.x))
        toy_intervals = hr_calc.toy_calibrated_hazard_ratio_interval(
            n_max=N_MAX,
            rng=RNG,
            confidence_levels=(0.68, 0.95),
            hazard_ratio_min=HAZARD_RATIO_MIN,
            hazard_ratio_max=HAZARD_RATIO_MAX,
            Threads=THREADS,
            print_progress=True,
        )
        elapsed = time.perf_counter() - started
        hr_results[key] = {
            "label": info["label"],
            "chi2_best": float(chi2[0]),
            "chi2_lo": float(chi2[1]),
            "chi2_hi": float(chi2[2]),
            "scan_2nll": scan_2nll,
            "toy": toy_intervals,
            "wall": elapsed,
        }
        toy_68 = toy_intervals[0.68]
        toy_95 = toy_intervals[0.95]
        print(f"\n{info['label']}: chi2 95% HR = {chi2[0]:.3g} [{chi2[1]:.3g}, {chi2[2]:.3g}]")
        print(f"  toy 68% [{toy_68[1]:.3g}, {toy_68[2]:.3g}]  (MLE {toy_68[0]:.3g})")
        print(f"  toy 95% [{toy_95[1]:.3g}, {toy_95[2]:.3g}]  (MLE {toy_95[0]:.3g})")
        print(f"  wall {elapsed:.1f}s")
```

```python
if SKIP:
    print("Skipping KM toy band probe (HR-only pass).")
else:
    print("KM toy bands are out of scope for this pass; skipping.")
```

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, key in zip(axes, CARDS):
    info = CARDS[key]
    ax.set_title(info["label"], fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(HAZARD_RATIO_MIN, HAZARD_RATIO_MAX)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Hazard Ratio", fontsize=10)
    ax.axhline(3.84, color="gray", linestyle=":", alpha=0.6, linewidth=2.0,
               label="95% CL (chi2=3.84)", zorder=1)
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1.0,
               label="H = 1", zorder=1)
    ax.grid(True, alpha=0.3, which="both")
    result = hr_results.get(key)
    if result is None:
        ax.text(0.5, 0.5, "KOMBINE_SKIP_TOY_CALIBRATION=1",
                ha="center", va="center", transform=ax.transAxes)
        continue
    delta = np.array(result["scan_2nll"]) - min(result["scan_2nll"])
    ax.plot(HAZARD_RATIOS_SCAN, delta, color="#d32f2f", linewidth=2.5,
            marker="s", markersize=3, label="KoMbine chi2", zorder=3)
    ax.axvline(result["chi2_best"], color="#d32f2f", linestyle="--",
               alpha=0.6, linewidth=1.5, zorder=2)
    toy_68 = result["toy"][0.68]
    toy_95 = result["toy"][0.95]
    ax.axvspan(toy_95[1], toy_95[2], color="#d32f2f", alpha=0.12,
               label="toy 95%", zorder=0)
    ax.axvspan(toy_68[1], toy_68[2], color="#d32f2f", alpha=0.22,
               label="toy 68%", zorder=0)
    ax.legend(fontsize=7, loc="upper left")

axes[0].set_ylabel(r"$-2 \Delta \ln L$", fontsize=10)
fig.suptitle("n=20 discrete classes: chi2 HR scan vs toy 68%/95% intervals",
             fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()
```
