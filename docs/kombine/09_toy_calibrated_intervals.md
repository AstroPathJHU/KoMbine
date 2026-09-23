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

This notebook inverts the same profile LRT with **toys** on the n=20 cards from the methods-comparison notebook (Fixed, discrete $e=0.20/0.25/0.40$, and Poisson large/moderate/small counts):

1. Pretend a candidate $\theta$ (an $H$, or $S(t)$) is true.
2. Simulate cohorts with biomarkers held fixed (weighted Cox permutation of the observed times for HR; binomial redraws on the observed death-time grid for KM).
3. Ask only whether each toy's $\Delta 2\mathrm{NLL}$ is larger than the observed value (`excess_at_most`), and stop when remaining toys cannot change the 68%/95% decision.
4. Search only interval **endpoints**, starting from a known-inside point (the MLE if toys accept it; otherwise $H=1$ / $S=0.5$, then a probe grid). Interior points are not toy-tested.

The HR and KM figures match notebook 07's mosaic layout. Yi / MC-SIMEX are omitted here; toys calibrate KoMbine, not those approximations.

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

**Monotone KM edges.** Pointwise toy endpoint search is omitted as the primary band. We run an S-grid of toys (all $B$ toys, no early stop) and fit decreasing $lo(t)$, $hi(t)$ by a binomial MLE: $n_{\mathrm{extreme}}\sim\mathrm{Binomial}(B,\pi_{\mathrm{in}})$ inside the band and $\pi_{\mathrm{out}}>\pi_{\mathrm{in}}$ outside. The same call also returns the usual $\chi^2$ profile bands. A later notebook cell compares that monotone fit to **naive** per-time intervals from the same grid (min/max accepted $S$ at each $t$, not forced decreasing)—no extra toys.

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
from kombine.toy_calibration import naive_pointwise_band_edges_from_grid

SKIP = bool(os.environ.get("KOMBINE_SKIP_TOY_CALIBRATION"))
N_MAX = 19
N_S_GRID = 11
RNG = 0
# Local runs use Gurobi's default thread count. CI goldens keep Threads=1.
THREADS = None
N_HR_SCAN = 25
HAZARD_RATIO_MIN = 0.01
HAZARD_RATIO_MAX = 100.0
HAZARD_RATIOS_SCAN = np.logspace(-2, 2, N_HR_SCAN)
DATACARDS = _repo_root / "test" / "kombine" / "datacards" / "simple_examples"

# Same n=20 family and mosaic layout as notebook 07.
MOSAIC_LAYOUT = [
    ['.', 'fixed', '.'],
    ['dc_small', 'dc_moderate', 'dc_large'],
    ['pois_large', 'pois_moderate', 'pois_small'],
]
MOSAIC_TO_SCENARIO = {
    'fixed': 'fixed',
    'dc_small': 'misclass_small',
    'dc_moderate': 'misclass_moderate',
    'dc_large': 'misclass_large',
    'pois_large': 'large',
    'pois_moderate': 'moderate',
    'pois_small': 'small',
}
SCENARIOS = {
    'fixed': {
        'file': DATACARDS / 'fixed_hr_example.txt',
        'label': 'Fixed Observable',
        'threshold': 0.5001,
    },
    'misclass_small': {
        'file': DATACARDS / 'discrete_classes_hr_example_moderate.txt',
        'label': 'Disc. Classes (e=0.20)',
        'threshold': 1.0,
    },
    'misclass_moderate': {
        'file': DATACARDS / 'discrete_classes_hr_example_large.txt',
        'label': 'Disc. Classes (e=0.25)',
        'threshold': 1.0,
    },
    'misclass_large': {
        'file': DATACARDS / 'discrete_classes_hr_example_very_large.txt',
        'label': 'Disc. Classes (e=0.40)',
        'threshold': 1.0,
    },
    'large': {
        'file': DATACARDS / 'poisson_density_hr_example_large.txt',
        'label': 'Poisson (large counts)',
        'threshold': 0.5001,
    },
    'moderate': {
        'file': DATACARDS / 'poisson_density_hr_example_moderate.txt',
        'label': 'Poisson (moderate counts)',
        'threshold': 0.5001,
    },
    'small': {
        'file': DATACARDS / 'poisson_density_hr_example_small.txt',
        'label': 'Poisson (small counts)',
        'threshold': 0.5001,
    },
}
hr_results = {}
km_results = {}

COLORS_PALETTE = {
    ('fixed', 'low'): '#1565c0',
    ('fixed', 'high'): '#c62828',
    ('misclass_small', 'low'): '#2e7d32',
    ('misclass_small', 'high'): '#c62828',
    ('misclass_moderate', 'low'): '#43a047',
    ('misclass_moderate', 'high'): '#b71c1c',
    ('misclass_large', 'low'): '#66bb6a',
    ('misclass_large', 'high'): '#e57373',
    ('large', 'low'): '#1976d2',
    ('large', 'high'): '#e53935',
    ('moderate', 'low'): '#26a69a',
    ('moderate', 'high'): '#fb8c00',
    ('small', 'low'): '#80cbc4',
    ('small', 'high'): '#ffd54f',
}

print("skip toy calibration MINLPs:", SKIP)
print("n=20 Fixed / discrete / Poisson HR+KM mosaics; N_MAX=", N_MAX,
      "N_S_GRID=", N_S_GRID, "Threads=", THREADS)
```

```python
if SKIP:
    print("Skipping n=20 toy HR mosaic (KOMBINE_SKIP_TOY_CALIBRATION=1).")
else:
    for key, info in SCENARIOS.items():
        if key in hr_results:
            print(f"skipping {info['label']} HR (already in memory)")
            continue
        started = time.perf_counter()
        dc = Datacard.parse_datacard(info["file"])
        hr_calc = dc.km_hazard_ratio(
            parameter_threshold=info["threshold"],
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
    print("Skipping n=20 KM monotone mosaic (KOMBINE_SKIP_TOY_CALIBRATION=1).")
else:
    for key, info in SCENARIOS.items():
        if key in km_results:
            print(f"skipping {info['label']} KM (already in memory)")
            continue
        started = time.perf_counter()
        dc = Datacard.parse_datacard(info["file"])
        threshold = info["threshold"]
        binomial_only = key == "fixed"
        km_low = dc.km_likelihood(parameter_min=-np.inf, parameter_max=threshold)
        km_high = dc.km_likelihood(parameter_min=threshold, parameter_max=np.inf)
        times_low = sorted(km_low.patient_death_times)
        times_high = sorted(km_high.patient_death_times)
        print(f"\n[{info['label']}] KM low={len(times_low)} death times, "
              f"high={len(times_high)} death times; "
              f"S-grid={N_S_GRID}, B={N_MAX}, early_stop=False")
        best_low, chi2_low, fitted_low, n_ext_low, fit_low = (
            km_low.toy_monotone_fit_survival_bands(
                times_low,
                n_max=N_MAX,
                n_s_grid=N_S_GRID,
                rng=RNG,
                binomial_only=binomial_only,
                Threads=THREADS,
                print_progress=True,
            )
        )
        best_high, chi2_high, fitted_high, n_ext_high, fit_high = (
            km_high.toy_monotone_fit_survival_bands(
                times_high,
                n_max=N_MAX,
                n_s_grid=N_S_GRID,
                rng=RNG + 1,
                binomial_only=binomial_only,
                Threads=THREADS,
                print_progress=True,
            )
        )
        elapsed = time.perf_counter() - started
        km_results[key] = {
            "label": info["label"],
            "low": {
                "times": times_low,
                "best": best_low,
                "chi2": chi2_low,
                "fitted": fitted_low,
                "n_extreme": n_ext_low,
                "s_grid": fit_low.s_grid,
                "pi_in": fit_low.pi_in,
                "pi_out": fit_low.pi_out,
                "loglik": fit_low.loglik,
            },
            "high": {
                "times": times_high,
                "best": best_high,
                "chi2": chi2_high,
                "fitted": fitted_high,
                "n_extreme": n_ext_high,
                "s_grid": fit_high.s_grid,
                "pi_in": fit_high.pi_in,
                "pi_out": fit_high.pi_out,
                "loglik": fit_high.loglik,
            },
            "wall": elapsed,
        }
        print(
            f"{info['label']}: KM monotone fit wall {elapsed:.1f}s "
            f"(low pi_in={fit_low.pi_in:.2f}/pi_out={fit_low.pi_out:.2f}, "
            f"high pi_in={fit_high.pi_in:.2f}/pi_out={fit_high.pi_out:.2f})"
        )
```
```python
_, axes_dict = plt.subplot_mosaic(
    MOSAIC_LAYOUT, figsize=(14, 13),
    gridspec_kw={'hspace': 0.52, 'wspace': 0.35},
)
for panel_key, scenario_key in MOSAIC_TO_SCENARIO.items():
    ax = axes_dict[panel_key]
    info = SCENARIOS[scenario_key]
    ax.set_title(info["label"], fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(HAZARD_RATIO_MIN, HAZARD_RATIO_MAX)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Hazard Ratio", fontsize=10)
    ax.set_ylabel(r"$-2 \Delta \ln L$", fontsize=10)
    ax.axhline(3.84, color="gray", linestyle=":", alpha=0.6, linewidth=2.0,
               label="95% CL (chi2=3.84)", zorder=1)
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1.0,
               label="H = 1", zorder=1)
    ax.grid(True, alpha=0.3, which="both")
    result = hr_results.get(scenario_key)
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

for panel_key, header in zip(
    ['dc_small', 'dc_moderate', 'dc_large'],
    ['Small Uncertainty', 'Medium Uncertainty', 'Large Uncertainty'],
):
    axes_dict[panel_key].annotate(
        header, xy=(0.5, 1.0), xytext=(0, 30),
        xycoords='axes fraction', textcoords='offset points',
        ha='center', va='bottom', fontsize=12, fontweight='bold',
        color='#333333', annotation_clip=False,
    )
for panel_key, row_label in zip(
    ['dc_small', 'pois_large'],
    ['Discrete\nClasses', 'Poisson\nCounts'],
):
    axes_dict[panel_key].annotate(
        row_label, xy=(0, 0.5), xytext=(-52, 0),
        xycoords='axes fraction', textcoords='offset points',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color='#333333', rotation=90, annotation_clip=False,
    )

plt.suptitle(
    "n=20: chi2 HR scan vs toy 68%/95% intervals (Fixed / discrete / Poisson)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
plt.show()
```

```python
def _km_step_arrays(times, best, lo, hi):
    if len(times) == 0:
        return [0.0], [1.0], [1.0], [1.0]
    times_plot = [times[0]]
    best_plot = [1.0]
    lo_plot = [1.0]
    hi_plot = [1.0]
    for i, t in enumerate(times):
        times_plot.append(t)
        best_plot.append(best[i])
        lo_plot.append(lo[i])
        hi_plot.append(hi[i])
    return times_plot, best_plot, lo_plot, hi_plot


_, axes_dict = plt.subplot_mosaic(
    MOSAIC_LAYOUT, figsize=(14, 13),
    gridspec_kw={'hspace': 0.52, 'wspace': 0.35},
)
for panel_key, scenario_key in MOSAIC_TO_SCENARIO.items():
    ax = axes_dict[panel_key]
    info = SCENARIOS[scenario_key]
    ax.set_title(info["label"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Survival Probability", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    result = km_results.get(scenario_key)
    if result is None:
        ax.text(0.5, 0.5, "KOMBINE_SKIP_TOY_CALIBRATION=1",
                ha="center", va="center", transform=ax.transAxes)
        continue
    for arm, color in (
        ("low", COLORS_PALETTE[(scenario_key, "low")]),
        ("high", COLORS_PALETTE[(scenario_key, "high")]),
    ):
        arm_result = result[arm]
        times = arm_result["times"]
        best = arm_result["best"]
        # Single CL=0.95 from toy_monotone_fit_survival_bands
        chi2_lo = arm_result["chi2"][:, 0, 0]
        chi2_hi = arm_result["chi2"][:, 0, 1]
        fitted = arm_result["fitted"]
        t_plot, best_plot, chi2_lo_plot, chi2_hi_plot = _km_step_arrays(
            times, best, chi2_lo, chi2_hi,
        )
        _, _, mono_lo_plot, mono_hi_plot = _km_step_arrays(
            times, best, fitted[:, 0], fitted[:, 1],
        )
        ax.fill_between(
            t_plot, chi2_lo_plot, chi2_hi_plot, step="post",
            alpha=0.12, color=color, label=f"{arm} chi2 95%",
        )
        ax.fill_between(
            t_plot, mono_lo_plot, mono_hi_plot, step="post",
            alpha=0.28, color=color, label=f"{arm} toy monotone fit",
        )
        ax.step(
            t_plot, best_plot, where="post", linewidth=2.5, color=color,
            label=f"{arm} MLE", zorder=3,
        )
    ax.legend(fontsize=7, loc="lower left")

for panel_key, header in zip(
    ['dc_small', 'dc_moderate', 'dc_large'],
    ['Small Uncertainty', 'Medium Uncertainty', 'Large Uncertainty'],
):
    axes_dict[panel_key].annotate(
        header, xy=(0.5, 1.0), xytext=(0, 30),
        xycoords='axes fraction', textcoords='offset points',
        ha='center', va='bottom', fontsize=12, fontweight='bold',
        color='#333333', annotation_clip=False,
    )
for panel_key, row_label in zip(
    ['dc_small', 'pois_large'],
    ['Discrete\nClasses', 'Poisson\nCounts'],
):
    axes_dict[panel_key].annotate(
        row_label, xy=(0, 0.5), xytext=(-52, 0),
        xycoords='axes fraction', textcoords='offset points',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color='#333333', rotation=90, annotation_clip=False,
    )

plt.suptitle(
    "n=20: chi2 vs binomial monotone toy KM bands (Fixed / discrete / Poisson)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
plt.show()
```

```python
# Compare monotone binomial fit vs naive per-time intervals from the same
# S-grid (no extra toys). Naive edges are min/max accepted S at each t and
# need not decrease.
_, axes_dict = plt.subplot_mosaic(
    MOSAIC_LAYOUT, figsize=(14, 13),
    gridspec_kw={'hspace': 0.52, 'wspace': 0.35},
)
for panel_key, scenario_key in MOSAIC_TO_SCENARIO.items():
    ax = axes_dict[panel_key]
    info = SCENARIOS[scenario_key]
    ax.set_title(info["label"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Survival Probability", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    result = km_results.get(scenario_key)
    if result is None:
        ax.text(0.5, 0.5, "KOMBINE_SKIP_TOY_CALIBRATION=1",
                ha="center", va="center", transform=ax.transAxes)
        continue
    for arm, color in (
        ("low", COLORS_PALETTE[(scenario_key, "low")]),
        ("high", COLORS_PALETTE[(scenario_key, "high")]),
    ):
        arm_result = result[arm]
        times = arm_result["times"]
        best = arm_result["best"]
        fitted = arm_result["fitted"]
        naive = naive_pointwise_band_edges_from_grid(
            arm_result["s_grid"],
            arm_result["n_extreme"],
            N_MAX,
            confidence_level=0.95,
            best=best,
        )
        t_plot, best_plot, mono_lo_plot, mono_hi_plot = _km_step_arrays(
            times, best, fitted[:, 0], fitted[:, 1],
        )
        _, _, naive_lo_plot, naive_hi_plot = _km_step_arrays(
            times, best, naive[:, 0], naive[:, 1],
        )
        ax.fill_between(
            t_plot, naive_lo_plot, naive_hi_plot, step="post",
            alpha=0.18, color=color, label=f"{arm} naive pointwise",
        )
        ax.step(
            t_plot, mono_lo_plot, where="post", linewidth=2.0,
            color=color, linestyle="--", label=f"{arm} monotone fit", zorder=4,
        )
        ax.step(
            t_plot, mono_hi_plot, where="post", linewidth=2.0,
            color=color, linestyle="--", zorder=4,
        )
        ax.step(
            t_plot, best_plot, where="post", linewidth=2.5, color=color,
            label=f"{arm} MLE", zorder=3,
        )
    ax.legend(fontsize=6, loc="lower left")

for panel_key, header in zip(
    ['dc_small', 'dc_moderate', 'dc_large'],
    ['Small Uncertainty', 'Medium Uncertainty', 'Large Uncertainty'],
):
    axes_dict[panel_key].annotate(
        header, xy=(0.5, 1.0), xytext=(0, 30),
        xycoords='axes fraction', textcoords='offset points',
        ha='center', va='bottom', fontsize=12, fontweight='bold',
        color='#333333', annotation_clip=False,
    )
for panel_key, row_label in zip(
    ['dc_small', 'pois_large'],
    ['Discrete\nClasses', 'Poisson\nCounts'],
):
    axes_dict[panel_key].annotate(
        row_label, xy=(0, 0.5), xytext=(-52, 0),
        xycoords='axes fraction', textcoords='offset points',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color='#333333', rotation=90, annotation_clip=False,
    )

plt.suptitle(
    "n=20: monotone binomial fit vs naive per-time toy intervals (same S-grid)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
plt.show()
```
