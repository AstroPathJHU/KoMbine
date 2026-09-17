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

This notebook inverts the same profile LRT with **toys** on the n=20 discrete-class cards from the methods-comparison notebook ($e=0.20$ vs $e=0.40$):

1. Pretend a candidate $\theta$ (an $H$, or $S(t)$) is true.
2. Simulate cohorts with biomarkers held fixed (weighted Cox permutation of the observed times for HR; binomial redraws on the observed death-time grid for KM).
3. Ask only whether each toy's $\Delta 2\mathrm{NLL}$ is larger than the observed value (`excess_at_most`), and stop when remaining toys cannot change the 68%/95% decision.
4. Search only interval **endpoints**, starting from the $\chi^2$ edges. Interior points are not toy-tested.

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
THREADS = 1
DATACARDS = _repo_root / "test" / "kombine" / "datacards" / "simple_examples"
CARDS = {
    "e=0.20": DATACARDS / "discrete_classes_hr_example_moderate.txt",
    "e=0.40": DATACARDS / "discrete_classes_hr_example_very_large.txt",
}
hr_rows = []
km_info = None

print("skip toy calibration MINLPs:", SKIP)
print("n=20 discrete-class prototype; N_MAX=", N_MAX)
```

```python
if SKIP:
    print("Skipping n=20 toy calibration (KOMBINE_SKIP_TOY_CALIBRATION=1).")
else:
    for label, path in CARDS.items():
        dc = Datacard.parse_datacard(path)
        hr_calc = dc.km_hazard_ratio(
            parameter_threshold=1.0,
            parameter_min=-np.inf,
            parameter_max=np.inf,
        )
        chi2 = hr_calc.hazard_ratio_confidence_interval(
            cox_only=False, confidence_level=0.95,
            hazard_ratio_min=0.05, hazard_ratio_max=20.0,
        )
        toy_h1 = hr_calc.hypothesized_hr_toy_test(
            1.0, n_max=N_MAX, rng=RNG, Threads=THREADS,
        )
        toy_lo = hr_calc.hypothesized_hr_toy_test(
            float(chi2[1]), n_max=N_MAX, rng=RNG + 1, Threads=THREADS,
        )
        toy_hi = hr_calc.hypothesized_hr_toy_test(
            float(chi2[2]), n_max=N_MAX, rng=RNG + 2, Threads=THREADS,
        )
        row = {
            "label": label,
            "chi2_best": float(chi2[0]),
            "chi2_lo": float(chi2[1]),
            "chi2_hi": float(chi2[2]),
            "p_h1": float(toy_h1.p_value),
            "dec_h1": dict(toy_h1.decisions),
            "t_obs_h1": float(toy_h1.t_obs),
            "n_ext_h1": (toy_h1.n_extreme, toy_h1.n_run),
            "dec_lo": dict(toy_lo.decisions),
            "dec_hi": dict(toy_hi.decisions),
        }
        hr_rows.append(row)
        print(f"\n{label}: chi2 95% HR = {chi2[0]:.3g} [{chi2[1]:.3g}, {chi2[2]:.3g}]")
        print(f"  toy test H=1: p={toy_h1.p_value:.3g} "
              f"n_ext={toy_h1.n_extreme}/{toy_h1.n_run} "
              f"decisions={toy_h1.decisions} d2NLL_obs={toy_h1.t_obs:.3g}")
        print(f"  toy at chi2 lower {chi2[1]:.3g}: {toy_lo.decisions}")
        print(f"  toy at chi2 upper {chi2[2]:.3g}: {toy_hi.decisions}")
```

```python
if SKIP:
    print("Skipping KM toy band probe at e=0.40.")
    km_info = None
else:
    dc = Datacard.parse_datacard(CARDS["e=0.40"])
    km = dc.km_likelihood(parameter_min=-np.inf, parameter_max=1.0)
    times = sorted(p.time for p in km.all_patients if not p.censored)
    t0 = times[min(2, len(times) - 1)]
    best, chi2_bands = km.survival_probabilities_likelihood(
        CLs=[0.68, 0.95], times_for_plot=[t0], crossing_mode="feasibility",
    )
    s_mle = float(np.clip(best[0], 1e-3, 1 - 1e-3))
    chi2_95 = chi2_bands[0][1]
    toy_mle = km.hypothesized_s_toy_test(
        t0, s_mle, n_max=N_MAX, rng=RNG, Threads=THREADS,
    )
    s_lo = float(np.clip(chi2_95[0], 1e-3, 1 - 1e-3))
    s_hi = float(np.clip(chi2_95[1], 1e-3, 1 - 1e-3))
    toy_lo = km.hypothesized_s_toy_test(
        t0, s_lo, n_max=N_MAX, rng=RNG + 1, Threads=THREADS,
    )
    toy_hi = km.hypothesized_s_toy_test(
        t0, s_hi, n_max=N_MAX, rng=RNG + 2, Threads=THREADS,
    )
    km_info = {
        "t0": t0,
        "best": float(best[0]),
        "chi2_bands": chi2_bands[0],
        "toy_mle": toy_mle,
        "toy_lo": toy_lo,
        "toy_hi": toy_hi,
        "s_lo": s_lo,
        "s_hi": s_hi,
    }
    print(f"KM low group e=0.40 at t={t0:g}: best S={best[0]:.3g}")
    print(f"  chi2 68/95 bands={chi2_bands[0]}")
    print(f"  toy at MLE S: p={toy_mle.p_value:.3g} "
          f"n_ext={toy_mle.n_extreme}/{toy_mle.n_run} "
          f"decisions={toy_mle.decisions}")
    print(f"  toy at chi2 95 lower {s_lo:.3g}: {toy_lo.decisions}")
    print(f"  toy at chi2 95 upper {s_hi:.3g}: {toy_hi.decisions}")
```

```python
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
ax_hr, ax_km = axes

if not hr_rows:
    ax_hr.set_title("HR (skipped)")
    ax_hr.text(0.5, 0.5, "KOMBINE_SKIP_TOY_CALIBRATION=1", ha="center", va="center")
    ax_hr.set_xticks([])
    ax_hr.set_yticks([])
else:
    ys = np.arange(len(hr_rows))
    bests = [row["chi2_best"] for row in hr_rows]
    xerr = np.array([
        [row["chi2_best"] - row["chi2_lo"] for row in hr_rows],
        [row["chi2_hi"] - row["chi2_best"] for row in hr_rows],
    ])
    ax_hr.errorbar(bests, ys, xerr=xerr, fmt="o", capsize=4, label="chi2 95% HR")
    ax_hr.axvline(1.0, color="gray", ls=":", label="H = 1")
    for y, row in zip(ys, hr_rows):
        mark = "accept" if row["dec_h1"].get(0.95) == "accept" else "reject"
        ax_hr.text(max(row["chi2_hi"] * 1.05, 1.2), y, f"toy H=1 95%: {mark}", va="center")
    ax_hr.set_yticks(ys)
    ax_hr.set_yticklabels([row["label"] for row in hr_rows])
    ax_hr.set_xscale("log")
    ax_hr.set_xlabel("hazard ratio")
    ax_hr.set_title("n=20 chi2 95% HR vs toy test at H=1")
    ax_hr.legend(loc="lower right", fontsize=8)

if km_info is None:
    ax_km.set_title("KM e=0.40 (skipped)")
    ax_km.text(0.5, 0.5, "KOMBINE_SKIP_TOY_CALIBRATION=1", ha="center", va="center")
    ax_km.set_xticks([])
    ax_km.set_yticks([])
else:
    chi2_68, chi2_95 = km_info["chi2_bands"]
    ax_km.plot(km_info["best"], 0.0, "ko", label="MLE S")
    ax_km.hlines(0.15, chi2_68[0], chi2_68[1], colors="C0", lw=6, label="chi2 68%")
    ax_km.hlines(-0.15, chi2_95[0], chi2_95[1], colors="C1", lw=6, label="chi2 95%")
    for s_val, toy, name in (
        (km_info["best"], km_info["toy_mle"], "MLE"),
        (km_info["s_lo"], km_info["toy_lo"], "chi2 lo"),
        (km_info["s_hi"], km_info["toy_hi"], "chi2 hi"),
    ):
        color = "green" if toy.decisions.get(0.95) == "accept" else "red"
        ax_km.plot(s_val, -0.45, "v", color=color)
        ax_km.text(s_val, -0.7, name, ha="center", fontsize=8, color=color)
    ax_km.set_xlim(-0.05, 1.05)
    ax_km.set_ylim(-1.0, 0.6)
    ax_km.set_yticks([])
    ax_km.set_xlabel(f"S(t={km_info['t0']:g}) low group")
    ax_km.set_title("e=0.40 KM: chi2 bands vs toy 95% at edges")
    ax_km.legend(loc="upper right", fontsize=8)

fig.tight_layout()
plt.show()
```
