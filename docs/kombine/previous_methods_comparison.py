"""Helpers for the Yi / MC-SIMEX / KoMbine comparison in notebook 07.

Scenario wiring, the toy-result cache, and the figures. The solvers live in kombine.
"""
# pylint: disable=line-too-long,too-many-arguments,too-many-branches,too-many-lines,too-many-locals,too-many-positional-arguments,too-many-statements

import datetime
import json
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from kombine.comparisons import YiCorrectionForCoxPH
from kombine.datacard import Datacard
from kombine.toy_calibration import (
  fit_monotone_band_edges_binomial,
  naive_pointwise_band_edges_from_grid,
)

# Single notebook budget. Edit these if you want denser scans.
N_PERMUTATIONS = 99
N_HR_SCAN = 25
SIMEX_B = 20
LABEL_WIDTH = 9
HAZARD_RATIOS_SCAN = np.logspace(-2, 2, N_HR_SCAN)  # 0.01 to 100
HAZARD_RATIO_MIN = 0.01
HAZARD_RATIO_MAX = 100.0

# Default: n=20 then n=50. Set KOMBINE_QUICK_COMPARISON=1 to skip n=50 (CI).
QUICK_COMPARISON = bool(os.environ.get("KOMBINE_QUICK_COMPARISON"))
# Skip toy MINLPs. A cache, if loaded, is still plotted. Fixed-card χ² still runs.
SKIP_TOY = bool(os.environ.get("KOMBINE_SKIP_TOY_CALIBRATION"))
SIMEX_RNG = 0
N_MAX = 19
N_S_GRID = 11
TOY_RNG = 0
# Local runs use Gurobi's default thread count.
THREADS = None
LOAD_TOY_CACHE = True
CACHE_DIR = pathlib.Path(__file__).resolve().parent / "_toy_calibration_cache"
CACHE_PATH = CACHE_DIR / "methods_comparison_toys.json"

TITLE_QUICK = "n=20, discrete e=0.20/0.25/0.40"
TITLE_FULL = "n=50, discrete e=0.05/0.10/0.25"

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

SCENARIOS_QUICK = {
  'fixed': {
    'file': 'fixed_hr_example.txt',
    'label': 'Fixed Observable',
    'description': 'no measurement error',
    'threshold': 0.5001,
  },
  'misclass_small': {
    'file': 'discrete_classes_hr_example_moderate.txt',
    'label': 'Disc. Classes (e=0.20)',
    'description': 'e = 0.20',
    'threshold': 1.0,
  },
  'misclass_moderate': {
    'file': 'discrete_classes_hr_example_large.txt',
    'label': 'Disc. Classes (e=0.25)',
    'description': 'e = 0.25',
    'threshold': 1.0,
  },
  'misclass_large': {
    'file': 'discrete_classes_hr_example_very_large.txt',
    'label': 'Disc. Classes (e=0.40)',
    'description': 'e = 0.40',
    'threshold': 1.0,
  },
  'large': {
    'file': 'poisson_density_hr_example_large.txt',
    'label': 'Poisson (large counts)',
    'description': '~2-3% relative error',
    'threshold': 0.5001,
  },
  'moderate': {
    'file': 'poisson_density_hr_example_moderate.txt',
    'label': 'Poisson (moderate counts)',
    'description': '~5-7% relative error',
    'threshold': 0.5001,
  },
  'small': {
    'file': 'poisson_density_hr_example_small.txt',
    'label': 'Poisson (small counts)',
    'description': '~25-70% relative error',
    'threshold': 0.5001,
  },
}

SCENARIOS_FULL = {
  'fixed': {
    'file': 'methods_comparison_fixed.txt',
    'label': 'Fixed Observable',
    'description': 'no measurement error',
    'threshold': 0.5001,
  },
  'misclass_small': {
    'file': 'methods_comparison_discrete_e05.txt',
    'label': 'Disc. Classes (e=0.05)',
    'description': 'e = 0.05',
    'threshold': 1.0,
  },
  'misclass_moderate': {
    'file': 'methods_comparison_discrete_e10.txt',
    'label': 'Disc. Classes (e=0.10)',
    'description': 'e = 0.10',
    'threshold': 1.0,
  },
  'misclass_large': {
    'file': 'methods_comparison_discrete_e25.txt',
    'label': 'Disc. Classes (e=0.25)',
    'description': 'e = 0.25',
    'threshold': 1.0,
  },
  'large': {
    'file': 'methods_comparison_poisson_large.txt',
    'label': 'Poisson (large counts)',
    'description': '~2-5% relative error',
    'threshold': 0.5001,
  },
  'moderate': {
    'file': 'methods_comparison_poisson_moderate.txt',
    'label': 'Poisson (moderate counts)',
    'description': '~10-20% relative error',
    'threshold': 0.5001,
  },
  'small': {
    'file': 'methods_comparison_poisson_small.txt',
    'label': 'Poisson (small counts)',
    'description': '~25-70% relative error',
    'threshold': 0.5001,
  },
}

# Same low/high colors in every mosaic panel (panels are separate plots).
COLOR_LOW = "#1565c0"
COLOR_HIGH = "#c62828"


def progress(msg: str) -> None:
  """Print a progress line with a wall-clock timestamp."""
  print(f"[{datetime.datetime.now()}] {msg}", flush=True)


def format_pvalue(p: float) -> str:
  """Ordinary decimals for p >= 1e-3; scientific notation below that."""
  if not np.isfinite(p):
    return str(p)
  if p >= 1e-3:
    return f'{p:.4g}'
  return f'{p:.3e}'


def skip_n50(what: str) -> None:
  """Print why the n=50 family is omitted under KOMBINE_QUICK_COMPARISON."""
  progress(
    f"Skipping n=50 {what} because KOMBINE_QUICK_COMPARISON is set. "
    "Unset it to run both the n=20 (e=0.20/0.25/0.40) and n=50 families."
  )


def skip_toys(what: str) -> None:
  """Print why a toy solve was omitted."""
  progress(
    f"Skipping {what} because KOMBINE_SKIP_TOY_CALIBRATION is set. "
    "χ² is not plotted in its place."
  )


def _json_encode(obj):
  if isinstance(obj, dict):
    return {str(key): _json_encode(value) for key, value in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_json_encode(value) for value in obj]
  if isinstance(obj, np.ndarray):
    return {
      "__ndarray__": True,
      "dtype": str(obj.dtype),
      "shape": list(obj.shape),
      "data": obj.tolist(),
    }
  if isinstance(obj, (np.floating, np.integer)):
    return obj.item()
  if isinstance(obj, pathlib.Path):
    return str(obj)
  return obj


def _json_decode(obj):
  if isinstance(obj, dict):
    if obj.get("__ndarray__"):
      return np.asarray(obj["data"], dtype=np.dtype(obj["dtype"]))
    return {key: _json_decode(value) for key, value in obj.items()}
  if isinstance(obj, list):
    return [_json_decode(value) for value in obj]
  return obj


TOY_CACHE = {
  "n20": {"hr": {}, "km": {}, "pv": {}},
  "n50": {"hr": {}, "km": {}, "pv": {}},
}


def save_toy_cache(path=CACHE_PATH):
  """Write toy HR/KM results and permutation p-values. Skip placeholders are not stored."""
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "format": "kombine_methods_comparison_toys_v1",
    "N_MAX": int(N_MAX),
    "N_S_GRID": int(N_S_GRID),
    "N_PERMUTATIONS": int(N_PERMUTATIONS),
    "RNG": int(TOY_RNG),
    "families": _json_encode(TOY_CACHE),
  }
  path.write_text(json.dumps(payload), encoding="utf-8")
  return path


def load_toy_cache(path=CACHE_PATH):
  """Load a matching cache. Returns False when the file is missing or stale."""
  path = pathlib.Path(path)
  if not path.is_file():
    return False
  payload = json.loads(path.read_text(encoding="utf-8"))
  if payload.get("format") != "kombine_methods_comparison_toys_v1":
    progress(f"Ignoring toy cache with format {payload.get('format')!r}")
    return False
  if int(payload.get("N_MAX", -1)) != N_MAX or int(payload.get("N_S_GRID", -1)) != N_S_GRID:
    progress(
      f"Ignoring toy cache at {path.name}: "
      f"N_MAX/N_S_GRID {payload.get('N_MAX')}/{payload.get('N_S_GRID')} "
      f"!= {N_MAX}/{N_S_GRID}"
    )
    return False
  loaded = _json_decode(payload.get("families", {}))
  # Older files have no N_PERMUTATIONS. Keep their HR/KM entries and leave p-values empty.
  use_pvalues = int(payload.get("N_PERMUTATIONS", -1)) == N_PERMUTATIONS
  if "N_PERMUTATIONS" in payload and not use_pvalues:
    progress(
      f"Ignoring cached p-values at {path.name}: "
      f"N_PERMUTATIONS {payload.get('N_PERMUTATIONS')} != {N_PERMUTATIONS}"
    )
  for family in ("n20", "n50"):
    family_payload = loaded.get(family, {})
    TOY_CACHE[family]["hr"] = family_payload.get("hr", {})
    TOY_CACHE[family]["km"] = family_payload.get("km", {})
    TOY_CACHE[family]["pv"] = family_payload.get("pv", {}) if use_pvalues else {}
  return True


def _refit_cached_km(km_dict):
  """Re-fit monotone edges from stored S-grid counts. No new MINLPs."""
  n_refit = 0
  for result in km_dict.values():
    kombine = result.get("kombine", {})
    for arm in ("low", "high"):
      arm_result = kombine.get(arm, {})
      if "n_extreme" not in arm_result:
        continue
      fit = fit_monotone_band_edges_binomial(
        arm_result["s_grid"],
        arm_result["n_extreme"],
        N_MAX,
        best=arm_result["best"],
      )
      fitted = np.column_stack([fit.lo, fit.hi])
      arm_result["fitted"] = fitted
      arm_result["ci"] = fitted[:, None, :]
      arm_result["pi_in"] = fit.pi_in
      arm_result["pi_out"] = fit.pi_out
      arm_result["loglik"] = fit.loglik
      n_refit += 1
  return n_refit


def _band_from_profile(best, ci):
  """Profile CI is already (n_times, n_cl, 2)."""
  return np.asarray(best, dtype=float), np.asarray(ci, dtype=float)


def _band_from_monotone(best, fitted, n_extreme, s_grid, fit):
  """Store the monotone toy band in the same CI layout as a one-CL profile."""
  fitted = np.asarray(fitted, dtype=float)
  return {
    "best": np.asarray(best, dtype=float),
    "ci": fitted[:, None, :],
    "fitted": fitted,
    "n_extreme": np.asarray(n_extreme),
    "s_grid": np.asarray(s_grid, dtype=float),
    "pi_in": fit.pi_in,
    "pi_out": fit.pi_out,
    "loglik": fit.loglik,
    "skipped": False,
  }


def load_datacards(directory, scenarios):
  """Parse each scenario datacard and print n / deaths / distinct times."""
  loaded = {}
  for key, info in scenarios.items():
    datacard = Datacard.parse_datacard(directory / info['file'])
    loaded[key] = datacard
    n_patients = len(datacard.patients)
    n_deaths = sum(1 for patient in datacard.patients if not patient.censored)
    uniq_deaths = len({
      round(patient.time, 10)
      for patient in datacard.patients if not patient.censored
    })
    progress(
      f"{info['label']}: {n_patients} patients, {n_deaths} deaths, "
      f"{uniq_deaths} distinct death times"
    )
  return loaded


def run_km_analysis(scenarios, datacards, family):
  """Yi, MC-SIMEX, and KoMbine KM for every scenario.

  The fixed card uses a binomial χ² band. Other cards use monotone toy
  bands unless toys are skipped and this family has no cache entry.
  """
  km_results = {}
  for scenario_key, scenario_info in scenarios.items():
    cached = TOY_CACHE[family]["km"].get(scenario_key)
    if cached is not None:
      progress(f"[{scenario_info['label']}] Analysis 1 from toy cache")
      km_results[scenario_key] = cached
      continue

    progress(f"[{scenario_info['label']}] Analysis 1 starting…")
    dc = datacards[scenario_key]
    threshold = scenario_info['threshold']

    km_low = dc.km_likelihood(
      parameter_min=-np.inf,
      parameter_max=threshold,
    )
    km_high = dc.km_likelihood(
      parameter_min=threshold,
      parameter_max=np.inf,
    )

    times_low = sorted(km_low.patient_death_times)
    times_high = sorted(km_high.patient_death_times)
    times_low_plot = [0.0] + times_low
    times_high_plot = [0.0] + times_high
    progress(
      f"  death times: low={len(times_low)} high={len(times_high)}"
    )

    progress("  Yi…")
    result_low_yi = dc.km_survival_yi(
      parameter_min=-np.inf,
      parameter_max=threshold,
      times_for_plot=times_low_plot,
    )
    result_high_yi = dc.km_survival_yi(
      parameter_min=threshold,
      parameter_max=np.inf,
      times_for_plot=times_high_plot,
    )

    progress(f"  MC-SIMEX (B={SIMEX_B})…")
    result_low_simex = dc.km_survival_mc_simex(
      parameter_min=-np.inf,
      parameter_max=threshold,
      times_for_plot=times_low_plot,
      rng=SIMEX_RNG,
      B=SIMEX_B,
    )
    result_high_simex = dc.km_survival_mc_simex(
      parameter_min=threshold,
      parameter_max=np.inf,
      times_for_plot=times_high_plot,
      rng=SIMEX_RNG,
      B=SIMEX_B,
    )

    use_toys = scenario_key != "fixed"
    if use_toys and SKIP_TOY:
      skip_toys(f"{scenario_info['label']} KoMbine KM bands")
      kombine_low = {
        "times": times_low, "best": None, "ci": None, "skipped": True,
      }
      kombine_high = {
        "times": times_high, "best": None, "ci": None, "skipped": True,
      }
      store_cache = False
    elif use_toys:
      progress(
        f"  KoMbine toy monotone KM (B={N_MAX}, S-grid={N_S_GRID})…"
      )
      started = time.perf_counter()
      best_low, _chi2_low, fitted_low, n_ext_low, fit_low = (
        km_low.toy_monotone_fit_survival_bands(
          times_low,
          n_max=N_MAX,
          n_s_grid=N_S_GRID,
          rng=TOY_RNG,
          binomial_only=False,
          Threads=THREADS,
          print_progress=True,
        )
      )
      best_high, _chi2_high, fitted_high, n_ext_high, fit_high = (
        km_high.toy_monotone_fit_survival_bands(
          times_high,
          n_max=N_MAX,
          n_s_grid=N_S_GRID,
          rng=TOY_RNG + 1,
          binomial_only=False,
          Threads=THREADS,
          print_progress=True,
        )
      )
      kombine_low = _band_from_monotone(
        best_low, fitted_low, n_ext_low, fit_low.s_grid, fit_low,
      )
      kombine_high = _band_from_monotone(
        best_high, fitted_high, n_ext_high, fit_high.s_grid, fit_high,
      )
      kombine_low["times"] = times_low
      kombine_high["times"] = times_high
      progress(f"  toy KM wall {time.perf_counter() - started:.1f}s")
      store_cache = True
    else:
      progress("  KoMbine fixed card: binomial χ² bands…")
      best_low, ci_low = km_low.survival_probabilities_likelihood(
        CLs=[0.95],
        times_for_plot=times_low,
        binomial_only=True,
        print_progress=True,
        crossing_mode="feasibility",
      )
      best_high, ci_high = km_high.survival_probabilities_likelihood(
        CLs=[0.95],
        times_for_plot=times_high,
        binomial_only=True,
        print_progress=True,
        crossing_mode="feasibility",
      )
      best_low, ci_low = _band_from_profile(best_low, ci_low)
      best_high, ci_high = _band_from_profile(best_high, ci_high)
      kombine_low = {
        "times": times_low, "best": best_low, "ci": ci_low, "skipped": False,
      }
      kombine_high = {
        "times": times_high, "best": best_high, "ci": ci_high, "skipped": False,
      }
      store_cache = True

    km_results[scenario_key] = {
      'yi': {
        'low': result_low_yi,
        'high': result_high_yi,
      },
      'simex': {
        'low': result_low_simex,
        'high': result_high_simex,
      },
      'kombine': {
        'low': kombine_low,
        'high': kombine_high,
      },
    }
    if store_cache:
      TOY_CACHE[family]["km"][scenario_key] = km_results[scenario_key]
      save_toy_cache()

    print(f"\n{scenario_info['label']}:")
    print(f"  {'Yi':<{LABEL_WIDTH}} - Low group final survival:  {result_low_yi['survival_probabilities'][-1]:.4f}")
    print(f"  {'Yi':<{LABEL_WIDTH}} - High group final survival: {result_high_yi['survival_probabilities'][-1]:.4f}")
    print(f"  {'MC-SIMEX':<{LABEL_WIDTH}} - Low group final survival:  {result_low_simex['survival_probabilities'][-1]:.4f}")
    print(f"  {'MC-SIMEX':<{LABEL_WIDTH}} - High group final survival: {result_high_simex['survival_probabilities'][-1]:.4f}")

    for arm_name, arm in (("Low", kombine_low), ("High", kombine_high)):
      if arm.get("skipped") or arm.get("best") is None or len(arm["best"]) == 0:
        print(f"  {'KoMbine':<{LABEL_WIDTH}} - {arm_name} group: toys skipped")
        continue
      ci = arm["ci"]
      if getattr(ci, "size", 0):
        print(
          f"  {'KoMbine':<{LABEL_WIDTH}} - {arm_name} group final survival:  "
          f"{arm['best'][-1]:.4f} [{ci[-1, 0, 0]:.4f}, {ci[-1, 0, 1]:.4f}]"
        )
      else:
        print(
          f"  {'KoMbine':<{LABEL_WIDTH}} - {arm_name} group final survival:  "
          f"{arm['best'][-1]:.4f}"
        )
  return km_results


def _plot_km_in_ax(ax, scenario_info, result):
  """Plot KM curves (Yi dashed, SIMEX dotted, KoMbine solid + CI shading)."""
  color_low = COLOR_LOW
  color_high = COLOR_HIGH

  times_low_yi = result['yi']['low']['times_for_plot']
  surv_low_yi = result['yi']['low']['survival_probabilities']
  ax.step(times_low_yi, surv_low_yi, where='post', linewidth=2.5,
      color=color_low, alpha=0.7, linestyle='--', label='Yi: Low group')

  times_high_yi = result['yi']['high']['times_for_plot']
  surv_high_yi = result['yi']['high']['survival_probabilities']
  ax.step(times_high_yi, surv_high_yi, where='post', linewidth=2.5,
      color=color_high, alpha=0.7, linestyle='--', label='Yi: High group')

  times_low_simex = result['simex']['low']['times_for_plot']
  surv_low_simex = result['simex']['low']['survival_probabilities']
  ax.step(times_low_simex, surv_low_simex, where='post', linewidth=2.0,
      color=color_low, alpha=0.85, linestyle=':', label='MC-SIMEX: Low group')

  times_high_simex = result['simex']['high']['times_for_plot']
  surv_high_simex = result['simex']['high']['survival_probabilities']
  ax.step(times_high_simex, surv_high_simex, where='post', linewidth=2.0,
      color=color_high, alpha=0.85, linestyle=':', label='MC-SIMEX: High group')

  times_low_kombine = result['kombine']['low']['times']
  best_low_kombine = result['kombine']['low']['best']
  ci_low_kombine = result['kombine']['low']['ci']
  if result['kombine']['low'].get('skipped') or best_low_kombine is None:
    ax.text(
      0.98, 0.98, "KoMbine toys skipped",
      ha="right", va="top", transform=ax.transAxes, fontsize=8,
    )
  else:
    times_plot_low = [times_low_kombine[0]]
    best_plot_low = [1.0]
    for i, t in enumerate(times_low_kombine):
      times_plot_low.append(t)
      best_plot_low.append(best_low_kombine[i])
    ax.step(times_plot_low, best_plot_low, where='post', linewidth=2.5,
        color=color_low, alpha=0.9, label='KoMbine: Low group', zorder=3)
    if getattr(ci_low_kombine, 'size', 0):
      ci_lower_plot_low = [1.0]
      ci_upper_plot_low = [1.0]
      for i, _t in enumerate(times_low_kombine):
        ci_lower_plot_low.append(ci_low_kombine[i, 0, 0])
        ci_upper_plot_low.append(ci_low_kombine[i, 0, 1])
      ax.fill_between(times_plot_low, ci_lower_plot_low, ci_upper_plot_low,
              step='post', alpha=0.15, color=color_low,
              label='KoMbine: Low 95% CI', zorder=2)

  times_high_kombine = result['kombine']['high']['times']
  best_high_kombine = result['kombine']['high']['best']
  ci_high_kombine = result['kombine']['high']['ci']
  if not result['kombine']['high'].get('skipped') and best_high_kombine is not None:
    times_plot_high = [times_high_kombine[0]]
    best_plot_high = [1.0]
    for i, t in enumerate(times_high_kombine):
      times_plot_high.append(t)
      best_plot_high.append(best_high_kombine[i])
    ax.step(times_plot_high, best_plot_high, where='post', linewidth=2.5,
        color=color_high, alpha=0.9, label='KoMbine: High group', zorder=3)
    if getattr(ci_high_kombine, 'size', 0):
      ci_lower_plot_high = [1.0]
      ci_upper_plot_high = [1.0]
      for i, _t in enumerate(times_high_kombine):
        ci_lower_plot_high.append(ci_high_kombine[i, 0, 0])
        ci_upper_plot_high.append(ci_high_kombine[i, 0, 1])
      ax.fill_between(times_plot_high, ci_lower_plot_high, ci_upper_plot_high,
              step='post', alpha=0.15, color=color_high,
              label='KoMbine: High 95% CI', zorder=2)

  ax.set_xlabel('Time', fontsize=10)
  ax.set_ylabel('Survival Probability', fontsize=10)
  ax.set_title(scenario_info['label'], fontsize=11, fontweight='bold')
  ax.legend(fontsize=8, loc='lower left')
  ax.grid(True, alpha=0.3)
  ax.set_ylim([0, 1.05])


def _annotate_comparison_mosaic(axes_dict):
  """Column headers and row labels shared by KM and HR mosaics."""
  col_headers_list = ['Small Uncertainty', 'Medium Uncertainty', 'Large Uncertainty']
  for panel_key, header in zip(['dc_small', 'dc_moderate', 'dc_large'], col_headers_list):
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


def plot_km_mosaic(scenarios, km_results, title):
  """3-row comparison mosaic of KM curves."""
  _, axes_dict = plt.subplot_mosaic(MOSAIC_LAYOUT, figsize=(14, 13),  # pyright: ignore[reportCallIssue, reportArgumentType]
    gridspec_kw={'hspace': 0.52, 'wspace': 0.35},
  )
  for panel_key, scenario_key in MOSAIC_TO_SCENARIO.items():
    _plot_km_in_ax(
      axes_dict[panel_key],
      scenarios[scenario_key],
      km_results[scenario_key],
    )
  _annotate_comparison_mosaic(axes_dict)
  plt.suptitle(title, fontsize=14, fontweight='bold')
  plt.tight_layout()
  plt.show()


def plot_monotone_vs_naive(scenarios, km_results, title):
  """Monotone toy edges vs naive per-time intervals on the stored S-grid."""
  _, axes_dict = plt.subplot_mosaic(MOSAIC_LAYOUT, figsize=(14, 13),  # pyright: ignore[reportCallIssue, reportArgumentType]
    gridspec_kw={'hspace': 0.52, 'wspace': 0.35},
  )
  for panel_key, scenario_key in MOSAIC_TO_SCENARIO.items():
    ax = axes_dict[panel_key]
    info = scenarios[scenario_key]
    ax.set_title(info['label'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Survival Probability', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    result = km_results[scenario_key]['kombine']
    if 'n_extreme' not in result['low']:
      note = "toys skipped" if result['low'].get('skipped') else "χ² band (fixed card)"
      ax.text(0.5, 0.5, note, ha='center', va='center', transform=ax.transAxes)
      continue
    for arm, color in (('low', COLOR_LOW), ('high', COLOR_HIGH)):
      arm_result = result[arm]
      times = arm_result['times']
      best = arm_result['best']
      fitted = arm_result['fitted']
      naive = naive_pointwise_band_edges_from_grid(
        arm_result['s_grid'],
        arm_result['n_extreme'],
        N_MAX,
        confidence_level=0.95,
        best=best,
      )
      times_plot = [times[0]]
      best_plot = [1.0]
      mono_lo = [1.0]
      mono_hi = [1.0]
      naive_lo = [1.0]
      naive_hi = [1.0]
      for i, t in enumerate(times):
        times_plot.append(t)
        best_plot.append(best[i])
        mono_lo.append(fitted[i, 0])
        mono_hi.append(fitted[i, 1])
        naive_lo.append(naive[i, 0])
        naive_hi.append(naive[i, 1])
      ax.fill_between(
        times_plot, naive_lo, naive_hi, step='post',
        alpha=0.18, color=color, label=f'{arm} naive pointwise',
      )
      ax.step(
        times_plot, mono_lo, where='post', linewidth=2.0,
        color=color, linestyle='--', label=f'{arm} monotone fit', zorder=4,
      )
      ax.step(
        times_plot, mono_hi, where='post', linewidth=2.0,
        color=color, linestyle='--', zorder=4,
      )
      ax.step(
        times_plot, best_plot, where='post', linewidth=2.5, color=color,
        label=f'{arm} MLE', zorder=3,
      )
    ax.legend(fontsize=6, loc='lower left')
  _annotate_comparison_mosaic(axes_dict)
  plt.suptitle(title, fontsize=14, fontweight='bold')
  plt.tight_layout()
  plt.show()


def _print_pvalue_scenario(label, result):
  """Print one scenario's Yi / MC-SIMEX / KoMbine p-values."""
  print(f"\n{label}:")
  print(f"  {'Yi':<{LABEL_WIDTH}} p-value: {format_pvalue(result['yi'])}")
  print(f"  {'MC-SIMEX':<{LABEL_WIDTH}} p-value: {format_pvalue(result['simex'])}")
  print(f"  {'KoMbine':<{LABEL_WIDTH}} p-value: {format_pvalue(result['kombine'])}")


def run_pvalue_analysis(scenarios, datacards, family):
  """Yi / MC-SIMEX logrank p-values and KoMbine permutation LRT.

  A cached scenario is reprinted and not solved again. Entries are reused only
  when the file was written with this ``N_PERMUTATIONS``.
  """
  pvalue_results = {}
  for scenario_key, scenario_info in scenarios.items():
    cached = TOY_CACHE[family]["pv"].get(scenario_key)
    if cached is not None:
      progress(f"[{scenario_info['label']}] Analysis 2 from cache")
      pvalue_results[scenario_key] = cached
      _print_pvalue_scenario(scenario_info['label'], cached)
      continue

    dc = datacards[scenario_key]
    threshold = scenario_info['threshold']
    progress(f"[{scenario_info['label']}] Analysis 2 starting…")

    progress("  Yi logrank…")
    yi_result = dc.km_p_value_logrank_yi(
      parameter_threshold=threshold,
      parameter_min=-np.inf,
      parameter_max=np.inf,
    )

    progress(f"  MC-SIMEX logrank (B={SIMEX_B})…")
    simex_result = dc.km_p_value_logrank_mc_simex(
      parameter_threshold=threshold,
      parameter_min=-np.inf,
      parameter_max=np.inf,
      rng=SIMEX_RNG,
      B=SIMEX_B,
    )

    progress(f"  KoMbine permutation LRT (B={N_PERMUTATIONS})…")
    kombine_calc = dc.km_p_value(
      parameter_threshold=threshold,
      parameter_min=-np.inf,
      parameter_max=np.inf,
    )
    pval_kombine, _, _ = kombine_calc.solve_and_pvalue(
      cox_only=False,
      n_permutations=N_PERMUTATIONS,
      rng=SIMEX_RNG,
      print_progress=True,
    )

    pvalue_results[scenario_key] = {
      'yi': yi_result['p_value'],
      'simex': simex_result['p_value'],
      'kombine': pval_kombine,
    }
    TOY_CACHE[family]["pv"][scenario_key] = pvalue_results[scenario_key]
    save_toy_cache()
    _print_pvalue_scenario(scenario_info['label'], pvalue_results[scenario_key])
  return pvalue_results


def plot_pvalue_bars(scenarios, pvalue_results, title):
  """Grouped bar chart of Yi / MC-SIMEX / KoMbine p-values."""
  _, ax1 = plt.subplots(figsize=(14, 5))
  scenario_keys = list(scenarios.keys())
  scenario_labels = [scenarios[k]['label'] for k in scenario_keys]
  yi_pvals = [pvalue_results[k]['yi'] for k in scenario_keys]
  simex_pvals = [pvalue_results[k]['simex'] for k in scenario_keys]
  kombine_pvals = [pvalue_results[k]['kombine'] for k in scenario_keys]

  x = np.arange(len(scenario_labels))
  width = 0.25
  bars1 = ax1.bar(x - width, yi_pvals, width, label="Yi's Method", color='steelblue')
  bars2 = ax1.bar(x, simex_pvals, width, label='MC-SIMEX', color='mediumpurple')
  bars3 = ax1.bar(x + width, kombine_pvals, width, label='KoMbine', color='coral')

  ax1.set_ylabel('P-value', fontsize=12)
  ax1.set_title(title, fontsize=13, fontweight='bold')
  ax1.set_xticks(x)
  ax1.set_xticklabels(scenario_labels, rotation=15, ha='right')
  ax1.legend(fontsize=11)
  ax1.grid(True, alpha=0.3, axis='y')

  for bar_group in [bars1, bars2, bars3]:
    for rect in bar_group:
      height = rect.get_height()
      ax1.text(rect.get_x() + rect.get_width()/2., height,
          format_pvalue(height), ha='center', va='bottom', fontsize=8)

  plt.tight_layout()
  plt.show()


def run_hr_analysis(scenarios, datacards, family):
  """Yi Breslow profile, MC-SIMEX Wald, and KoMbine HR.

  The KoMbine curve is the profile Δ2NLL. The reported interval is a toy
  95% interval when assignments are free, and a Wilks cut on the fixed card.
  """
  hr_results = {}
  for scenario_key, scenario_info in scenarios.items():
    cached = TOY_CACHE[family]["hr"].get(scenario_key)
    if cached is not None:
      progress(f"[{scenario_info['label']}] Analysis 3 from toy cache")
      hr_results[scenario_key] = cached
      continue

    dc = datacards[scenario_key]
    hr_threshold = scenario_info['threshold']
    progress(f"[{scenario_info['label']}] Analysis 3 starting…")

    progress("  Yi HR CI…")
    yi_calc = YiCorrectionForCoxPH(
      patients=dc.patients,
      parameter_min=-np.inf,
      parameter_max=np.inf,
      parameter_threshold=hr_threshold,
    )
    best_hr_yi, yi_lower_ci, yi_upper_ci, yi_best_fit = (
      yi_calc.hazard_ratio_confidence_interval(
        confidence_level=0.95,
        hazard_ratio_min=HAZARD_RATIO_MIN,
        hazard_ratio_max=HAZARD_RATIO_MAX,
      )
    )
    yi_2nlls = [
      yi_calc.compute_2nll_at_hazard_ratio(hr).x
      for hr in HAZARD_RATIOS_SCAN
    ]

    progress(f"  MC-SIMEX HR (B={SIMEX_B})…")
    simex_calc = dc.km_hazard_ratio_mc_simex(
      parameter_threshold=hr_threshold,
      parameter_min=-np.inf,
      parameter_max=np.inf,
      rng=SIMEX_RNG,
      B=SIMEX_B,
    )
    simex_estimate = simex_calc.estimate_hazard_ratio()
    simex_2nlls = [
      simex_calc.compute_2nll_at_hazard_ratio(hr).x
      for hr in HAZARD_RATIOS_SCAN
    ]

    hr_calc = dc.km_hazard_ratio(
      parameter_threshold=hr_threshold,
      parameter_min=-np.inf,
      parameter_max=np.inf,
    )

    progress(f"  KoMbine HR 2NLL scan ({N_HR_SCAN} points)…")
    kombine_2nlls = []
    for hr in HAZARD_RATIOS_SCAN:
      result = hr_calc.compute_2nll_at_hazard_ratio(
        hr, cox_only=False, verbose=False, print_progress=True,
      )
      kombine_2nlls.append(result.x)

    use_toys = scenario_key != "fixed"
    toy_68 = None
    if use_toys and SKIP_TOY:
      skip_toys(f"{scenario_info['label']} KoMbine HR interval")
      grid_idx = int(np.argmin(kombine_2nlls))
      best_hr_kombine = float(HAZARD_RATIOS_SCAN[grid_idx])
      lower_ci = np.nan
      upper_ci = np.nan
      store_cache = False
    elif use_toys:
      progress(f"  KoMbine toy HR interval (B={N_MAX})…")
      toy_intervals = hr_calc.toy_calibrated_hazard_ratio_interval(
        n_max=N_MAX,
        rng=TOY_RNG,
        cox_only=False,
        confidence_levels=(0.68, 0.95),
        hazard_ratio_min=HAZARD_RATIO_MIN,
        hazard_ratio_max=HAZARD_RATIO_MAX,
        Threads=THREADS,
        print_progress=True,
      )
      best_hr_kombine, lower_ci, upper_ci = toy_intervals[0.95]
      toy_68 = (toy_intervals[0.68][1], toy_intervals[0.68][2])
      store_cache = True
    else:
      progress("  KoMbine fixed card: χ² HR interval…")
      best_hr_kombine, lower_ci, upper_ci, _ = hr_calc.hazard_ratio_confidence_interval(
        cox_only=False,
        method="chi2",
        confidence_level=0.95,
        hazard_ratio_min=HAZARD_RATIO_MIN,
        hazard_ratio_max=HAZARD_RATIO_MAX,
      )
      store_cache = True

    hr_results[scenario_key] = {
      'yi_best': best_hr_yi,
      'yi_2nlls': yi_2nlls,
      'yi_min_2nll': yi_best_fit.x,
      'yi_lower': yi_lower_ci,
      'yi_upper': yi_upper_ci,
      'simex_best': simex_estimate['hazard_ratio'],
      'simex_2nlls': simex_2nlls,
      'simex_lower': simex_estimate['ci_lower'],
      'simex_upper': simex_estimate['ci_upper'],
      'kombine_best': best_hr_kombine,
      'kombine_2nlls': kombine_2nlls,
      'kombine_lower': lower_ci,
      'kombine_upper': upper_ci,
      'kombine_toy_68': toy_68,
      'kombine_skipped': bool(use_toys and SKIP_TOY),
    }
    if store_cache:
      TOY_CACHE[family]["hr"][scenario_key] = hr_results[scenario_key]
      save_toy_cache()

    print(f"\n{scenario_info['label']}:")
    print(f"  {'Yi':<{LABEL_WIDTH}} best-fit HR: {best_hr_yi:.3f} [{yi_lower_ci:.3f}, {yi_upper_ci:.3f}]")
    print(f"  {'MC-SIMEX':<{LABEL_WIDTH}} best-fit HR: {simex_estimate['hazard_ratio']:.3f} [{simex_estimate['ci_lower']:.3f}, {simex_estimate['ci_upper']:.3f}]")
    if np.isfinite(lower_ci) and np.isfinite(upper_ci):
      print(f"  {'KoMbine':<{LABEL_WIDTH}} best-fit HR: {best_hr_kombine:.3f} [{lower_ci:.3f}, {upper_ci:.3f}]")
    else:
      print(f"  {'KoMbine':<{LABEL_WIDTH}} best-fit HR: {best_hr_kombine:.3f} [toys skipped]")
  return hr_results


def plot_hr_mosaic(scenarios, hr_results, title):
  """3-row comparison mosaic of HR Δ2NLL scans."""
  _, axes_dict = plt.subplot_mosaic(MOSAIC_LAYOUT, figsize=(14, 13),  # pyright: ignore[reportCallIssue, reportArgumentType]
    gridspec_kw={'hspace': 0.52, 'wspace': 0.35},
  )
  for panel_key, scenario_key in MOSAIC_TO_SCENARIO.items():
    ax = axes_dict[panel_key]
    result = hr_results[scenario_key]
    info = scenarios[scenario_key]

    yi_2nlls = result['yi_2nlls']
    delta_yi = np.array(yi_2nlls) - result['yi_min_2nll']
    delta_simex = np.array(result['simex_2nlls'])
    kombine_2nlls = result['kombine_2nlls']
    delta_kombine = np.array(kombine_2nlls) - min(kombine_2nlls)

    ax.plot(HAZARD_RATIOS_SCAN, delta_yi, color='#1976d2', linewidth=2.5, marker='o', markersize=3,
        label="Yi's Method", zorder=3)
    ax.plot(HAZARD_RATIOS_SCAN, delta_simex, color='#7b1fa2', linewidth=2.5, marker='^', markersize=3,
        linestyle=':', label='MC-SIMEX (Wald)', zorder=3)
    ax.plot(HAZARD_RATIOS_SCAN, delta_kombine, color='#d32f2f', linewidth=2.5, marker='s', markersize=3,
        label='KoMbine profile', zorder=3)
    ax.axvline(result['yi_best'], color='#1976d2', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
    ax.axvline(result['simex_best'], color='#7b1fa2', linestyle=':', alpha=0.6, linewidth=1.5, zorder=2)
    ax.axvline(result['kombine_best'], color='#d32f2f', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
    lower = result['kombine_lower']
    upper = result['kombine_upper']
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
      ax.axvspan(lower, upper, color='#d32f2f', alpha=0.12, label='KoMbine 95%', zorder=0)
    toy_68 = result.get('kombine_toy_68')
    if toy_68 is not None:
      ax.axvspan(toy_68[0], toy_68[1], color='#d32f2f', alpha=0.22, label='KoMbine 68%', zorder=0)
    elif result.get('kombine_skipped'):
      ax.text(0.98, 0.98, "KoMbine toys skipped",
          ha="right", va="top", transform=ax.transAxes, fontsize=8)

    ax.set_xlabel('Hazard Ratio', fontsize=10)
    ax.set_ylabel(r'$-2 \Delta \ln L$', fontsize=10)
    ax.set_title(info['label'], fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_xlim([0.01, 100.0])
    ax.set_ylim([0, 10])

  _annotate_comparison_mosaic(axes_dict)
  plt.suptitle(title, fontsize=14, fontweight='bold')
  plt.tight_layout()
  plt.show()


# Wilks cuts for one degree of freedom. The 95% line is the usual 3.84.
CHI2_68 = float(scipy.stats.chi2.ppf(0.68, 1))
CHI2_95 = float(scipy.stats.chi2.ppf(0.95, 1))


def _delta_2nll(twonlls):
  """Profile Δ2NLL, recentered at the scan minimum."""
  values = np.asarray(twonlls, dtype=float)
  return values - np.min(values)


def _wilks_runs(delta, level):
  """Contiguous grid-index runs where the profile is at or below ``level``."""
  inside = np.asarray(delta, dtype=float) <= level
  runs = []
  start = None
  for i, flag in enumerate(inside):
    if flag and start is None:
      start = i
    elif not flag and start is not None:
      runs.append((start, i - 1))
      start = None
  if start is not None:
    runs.append((start, len(inside) - 1))
  return runs


def _cross_log_hr(delta, i_in, i_out, level):
  """Log-linear HR where the profile crosses ``level`` between two grid points."""
  d_in = float(delta[i_in])
  d_out = float(delta[i_out])
  if d_out == d_in:
    return float(HAZARD_RATIOS_SCAN[i_out])
  t = (level - d_in) / (d_out - d_in)
  log_in = np.log(HAZARD_RATIOS_SCAN[i_in])
  log_out = np.log(HAZARD_RATIOS_SCAN[i_out])
  return float(np.exp(log_in + t * (log_out - log_in)))


def _chi2_interval_around_minimum(delta, level):
  """χ²₁ edges on either side of the scan minimum.

  Same rule as ``brentq_hazard_ratio_ci``: a side that never crosses stays
  at the search bound. Returns ``(lower, upper, n_extra_runs)``. An extra
  run is a grid segment at or below ``level`` that does not contain the
  minimum. The reported interval is still the one around the minimum.
  """
  delta = np.asarray(delta, dtype=float)
  imin = int(np.argmin(delta))
  runs = _wilks_runs(delta, level)
  containing = [run for run in runs if run[0] <= imin <= run[1]]
  if not containing:
    hr_at_min = float(HAZARD_RATIOS_SCAN[imin])
    return hr_at_min, hr_at_min, len(runs)
  start, end = containing[0]
  if start == 0:
    lower = float(HAZARD_RATIOS_SCAN[0])
  else:
    lower = _cross_log_hr(delta, start, start - 1, level)
  if end == len(delta) - 1:
    upper = float(HAZARD_RATIOS_SCAN[-1])
  else:
    upper = _cross_log_hr(delta, end, end + 1, level)
  return lower, upper, len(runs) - 1


def _format_hr_interval(lower, upper):
  """Print a finite HR interval, or a placeholder."""
  if lower is None or upper is None:
    return 'n/a'
  if not (np.isfinite(lower) and np.isfinite(upper)):
    return 'n/a'
  return f'[{lower:.3f}, {upper:.3f}]'


def _toy_interval_strings(result):
  """Cached toy 68% and 95% intervals. The fixed card has no toys."""
  toy_68 = result.get('kombine_toy_68')
  if toy_68 is None:
    return 'n/a', 'n/a'
  return (
    _format_hr_interval(toy_68[0], toy_68[1]),
    _format_hr_interval(result['kombine_lower'], result['kombine_upper']),
  )


def plot_chi2_vs_toy_mosaic(scenarios, family, title):
  """Floating profile, χ²₁ lines, and cached toy spans.

  The horizontal lines are the Wilks cut of this same profile. Toy spans are
  drawn only when the cache stored a toy interval. An extra sublevel run
  that does not contain the minimum is not painted as part of one bar.
  """
  hr_cache = TOY_CACHE[family]['hr']
  _, axes_dict = plt.subplot_mosaic(MOSAIC_LAYOUT, figsize=(14, 13),  # pyright: ignore[reportCallIssue, reportArgumentType]
    gridspec_kw={'hspace': 0.52, 'wspace': 0.35},
  )
  for panel_key, scenario_key in MOSAIC_TO_SCENARIO.items():
    ax = axes_dict[panel_key]
    info = scenarios[scenario_key]
    result = hr_cache[scenario_key]
    delta = _delta_2nll(result['kombine_2nlls'])
    _lo95, _hi95, n_extra_95 = _chi2_interval_around_minimum(delta, CHI2_95)

    ax.plot(
      HAZARD_RATIOS_SCAN, delta, color='#d32f2f', linewidth=2.2,
      marker='s', markersize=3, label='profile (assignments free)', zorder=3,
    )
    ax.axhline(
      CHI2_68, color='#37474f', linestyle=':', linewidth=1.2,
      label=r'$\chi^2_1$ 68%', zorder=1,
    )
    ax.axhline(
      CHI2_95, color='#37474f', linestyle='--', linewidth=1.2,
      label=r'$\chi^2_1$ 95%', zorder=1,
    )
    toy_68 = result.get('kombine_toy_68')
    lower = result['kombine_lower']
    upper = result['kombine_upper']
    if toy_68 is not None and np.isfinite(lower) and np.isfinite(upper) and upper > lower:
      ax.axvspan(lower, upper, color='#d32f2f', alpha=0.12, label='toy 95%', zorder=0)
      if toy_68[1] > toy_68[0]:
        ax.axvspan(
          toy_68[0], toy_68[1], color='#d32f2f', alpha=0.22,
          label='toy 68%', zorder=0,
        )
    if n_extra_95:
      ax.text(
        0.98, 0.02, 'extra chi2 run',
        ha='right', va='bottom', transform=ax.transAxes, fontsize=8,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none', 'pad': 1.5},
      )
    ax.set_xlabel('Hazard Ratio', fontsize=10)
    ax.set_ylabel(r'$-2 \Delta \ln L$', fontsize=10)
    ax.set_title(info['label'], fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_xlim([0.01, 100.0])
    ax.set_ylim([0, 10])

  _annotate_comparison_mosaic(axes_dict)
  plt.suptitle(title, fontsize=14, fontweight='bold')
  plt.tight_layout()
  plt.show()


def print_chi2_vs_toy_table(scenarios, family):
  """χ²₁ edges around the scan minimum, and cached toy intervals."""
  hr_cache = TOY_CACHE[family]['hr']
  header = (
    f"{'Scenario':<36} {'chi2 68%':>22} {'chi2 95%':>22}"
    f"  {'toy 68%':>22}  {'toy 95%':>22}  note"
  )
  print(header)
  print('-' * len(header))
  for key, info in scenarios.items():
    result = hr_cache[key]
    delta = _delta_2nll(result['kombine_2nlls'])
    lo68, hi68, n_extra_68 = _chi2_interval_around_minimum(delta, CHI2_68)
    lo95, hi95, n_extra_95 = _chi2_interval_around_minimum(delta, CHI2_95)
    toy_68_str, toy_95_str = _toy_interval_strings(result)
    notes = []
    if n_extra_68:
      notes.append(f'extra 68% run ({n_extra_68})')
    if n_extra_95:
      notes.append(f'extra 95% run ({n_extra_95})')
    note = ', '.join(notes)
    print(
      f"{info['label']:<36} {_format_hr_interval(lo68, hi68):>22}"
      f" {_format_hr_interval(lo95, hi95):>22}"
      f"  {toy_68_str:>22}  {toy_95_str:>22}  {note}"
    )


def show_chi2_vs_toys(scenarios, family, title):
  """Plot and tabulate χ²₁ vs toys for one cached family. No new solves."""
  hr_cache = TOY_CACHE[family]['hr']
  if any(key not in hr_cache for key in scenarios):
    progress(f'chi2 vs toys: cache for {family} is incomplete; skipping')
    return
  progress(f'chi2 vs toys — {title} (cache only)')
  plot_chi2_vs_toy_mosaic(scenarios, family, title)
  print_chi2_vs_toy_table(scenarios, family)


def print_summary_table(scenarios, pvalue_results, hr_results):
  """Live p-value / HR table for one card family."""
  header = (
    f"{'Scenario':<36} {'Yi p':>9} {'SIMEX p':>9} {'KoMbine p':>11}"
    f"  {'Yi HR [95% CI]':>20}  {'SIMEX HR [Wald]':>20}  {'KoMbine HR [95% CI]':>22}"
  )
  print(header)
  print('-' * len(header))
  for key, info in scenarios.items():
    pv = pvalue_results[key]
    hr = hr_results[key]
    yi_ci = (
      f"[{hr['yi_lower']:.3f}, {hr['yi_upper']:.3f}]"
      if not np.isnan(hr['yi_lower']) else '[n/a]'
    )
    simex_ci = f"[{hr['simex_lower']:.3f}, {hr['simex_upper']:.3f}]"
    ko_ci = (
      f"[{hr['kombine_lower']:.3f}, {hr['kombine_upper']:.3f}]"
      if np.isfinite(hr['kombine_lower']) else '[toys skipped]'
    )
    yi_hr_str = f"{hr['yi_best']:.3f} {yi_ci}"
    simex_hr_str = f"{hr['simex_best']:.3f} {simex_ci}"
    ko_hr_str = f"{hr['kombine_best']:.3f} {ko_ci}"
    ko_p_str = 'n/a' if pv['kombine'] is None else format_pvalue(pv['kombine'])
    print(
      f"{info['label']:<36} {format_pvalue(pv['yi']):>9} {format_pvalue(pv['simex']):>9} {ko_p_str:>11}"
      f"  {yi_hr_str:>20}  {simex_hr_str:>20}  {ko_hr_str:>22}"
    )
