"""Benchmark feasibility CL-band crossings vs full minimize.

Compares crossing_mode=full vs feasibility (component cuts on/off) on hard
death times for selected methods_comparison cards.

Not part of the default CI suite; run manually:
  python -m test.kombine.bench_feasibility_cl_oracle
  python -m test.kombine.bench_feasibility_cl_oracle --cards discrete_e10 poisson_small
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

_repo = pathlib.Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
  sys.path.insert(0, str(_repo))

from docs.kombine.rebin_methods_comparison_times import ensure_rebinned
from kombine.datacard import Datacard

CARD_SPECS: dict[str, dict] = {
  "discrete_e10": {
    "file": "methods_comparison_discrete_e10.txt",
    "label": "Disc. Classes (e=0.10)",
    "threshold": 1.0,
    "arm": "low",
    "targets": [3.6905, 9.4637],
  },
  "poisson_small": {
    "file": "methods_comparison_poisson_small.txt",
    "label": "Poisson (small counts)",
    "threshold": 0.5001,
    "arm": "low",
    "targets": [3.6905, 9.4637],
  },
}


def _nearest(times: list[float], target: float) -> float:
  return min(times, key=lambda t: abs(t - target))


def _run_one(km, t: float, *, crossing_mode: str, component_cuts: bool) -> dict:
  t0 = time.perf_counter()
  best, ci = km.survival_probabilities_likelihood(
    CLs=[0.95],
    times_for_plot=[t],
    print_progress=True,
    crossing_mode=crossing_mode,
    component_cuts=component_cuts,
  )
  elapsed = time.perf_counter() - t0
  return {
    "mode": crossing_mode,
    "component_cuts": component_cuts,
    "t": t,
    "best": float(best[0]),
    "ci_lo": float(ci[0, 0, 0]),
    "ci_hi": float(ci[0, 0, 1]),
    "seconds": elapsed,
  }


def _bench_card(spec: dict) -> list[dict]:
  datacards_dir = ensure_rebinned(n_bins=4)
  path = datacards_dir / spec["file"]
  dc = Datacard.parse_datacard(path)
  threshold = float(spec["threshold"])
  if spec["arm"] == "low":
    km = dc.km_likelihood(parameter_min=-np.inf, parameter_max=threshold)
  else:
    km = dc.km_likelihood(parameter_min=threshold, parameter_max=np.inf)
  times = sorted(km.patient_death_times)
  targets = [_nearest(times, target) for target in spec["targets"]]
  print(
    f"\n{'='*72}\n{spec['label']} ({path.name})\n"
    f"times={times} targets={targets}\n{'='*72}",
    flush=True,
  )

  rows: list[dict] = []
  for t in targets:
    for label, mode, cuts in (
      ("full", "full", True),
      ("feasibility+cuts", "feasibility", True),
      ("feasibility sum-only", "feasibility", False),
    ):
      print(f"\n=== {spec['label']} t={t} {label} ===", flush=True)
      row = _run_one(km, t, crossing_mode=mode, component_cuts=cuts)
      row["label"] = spec["label"]
      row["variant"] = label
      rows.append(row)
  return rows


def _print_summary(all_rows: list[dict]) -> None:
  print("\n" + "=" * 72, flush=True)
  print("SUMMARY", flush=True)
  print("=" * 72, flush=True)
  for label in dict.fromkeys(r["label"] for r in all_rows):
    print(f"\n{label}", flush=True)
    print(
      f"{'t':>8} {'variant':<22} {'time(s)':>8} {'ci_lo':>10} {'ci_hi':>10} "
      f"{'dlo':>10} {'dhi':>10} {'speedup':>8}",
      flush=True,
    )
    label_rows = [r for r in all_rows if r["label"] == label]
    for t in dict.fromkeys(r["t"] for r in label_rows):
      full = next(r for r in label_rows if r["t"] == t and r["mode"] == "full")
      for r in label_rows:
        if r["t"] != t:
          continue
        dlo = abs(r["ci_lo"] - full["ci_lo"]) if r["mode"] != "full" else 0.0
        dhi = abs(r["ci_hi"] - full["ci_hi"]) if r["mode"] != "full" else 0.0
        speedup = full["seconds"] / max(r["seconds"], 1e-9)
        print(
          f"{r['t']:8.4f} {r['variant']:<22} {r['seconds']:8.1f} "
          f"{r['ci_lo']:10.6f} {r['ci_hi']:10.6f} "
          f"{dlo:10.2e} {dhi:10.2e} {speedup:8.2f}x",
          flush=True,
        )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--cards",
    nargs="+",
    choices=list(CARD_SPECS),
    default=list(CARD_SPECS),
    help="Which rebinned methods_comparison cards to benchmark",
  )
  args = parser.parse_args()

  all_rows: list[dict] = []
  for key in args.cards:
    all_rows.extend(_bench_card(CARD_SPECS[key]))
  _print_summary(all_rows)

  for row in all_rows:
    if row["mode"] == "full":
      continue
    full = next(
      r for r in all_rows
      if r["label"] == row["label"] and r["t"] == row["t"] and r["mode"] == "full"
    )
    if abs(row["ci_lo"] - full["ci_lo"]) > 1e-3 or abs(row["ci_hi"] - full["ci_hi"]) > 1e-3:
      raise SystemExit(
        f"Endpoint mismatch beyond 1e-3 for {row['label']} t={row['t']} {row['variant']}"
      )


if __name__ == "__main__":
  main()
