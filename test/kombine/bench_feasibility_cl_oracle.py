"""Benchmark feasibility CL-band crossings vs full minimize.

Compares crossing_mode=full vs feasibility (component cuts on/off) on hard
death times for selected methods_comparison cards.

Not part of the default CI suite; run manually:
  python -m test.kombine.bench_feasibility_cl_oracle
  python -m test.kombine.bench_feasibility_cl_oracle --cards discrete_e10 poisson_small
"""

from __future__ import annotations

import argparse
import json
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


def _run_one(
  km,
  t: float,
  *,
  crossing_mode: str,
  component_cuts: bool,
  binom_time_cuts: bool = False,
  cost_biased_bisection: bool = True,
  minlp_time_limit: float | None = None,
  oracle_time_limit: float | None = None,
) -> dict:
  t0 = time.perf_counter()
  best, ci = km.survival_probabilities_likelihood(
    CLs=[0.95],
    times_for_plot=[t],
    print_progress=True,
    crossing_mode=crossing_mode,
    component_cuts=component_cuts,
    binom_time_cuts=binom_time_cuts,
    cost_biased_bisection=cost_biased_bisection,
    minlp_time_limit=minlp_time_limit,
    oracle_time_limit=oracle_time_limit,
  )
  elapsed = time.perf_counter() - t0
  ws = km.last_gurobi_work_stats
  return {
    "mode": crossing_mode,
    "component_cuts": component_cuts,
    "binom_time_cuts": binom_time_cuts,
    "cost_biased_bisection": cost_biased_bisection,
    "t": t,
    "best": float(best[0]),
    "ci_lo": float(ci[0, 0, 0]),
    "ci_hi": float(ci[0, 0, 1]),
    "seconds": elapsed,
    "oracle_work": float(ws.oracle_work) if ws is not None else 0.0,
    "minimize_work": float(ws.minimize_work) if ws is not None else 0.0,
    "total_work": float(ws.total_work) if ws is not None else 0.0,
    "oracle_calls": int(ws.oracle_calls) if ws is not None else 0,
    "minimize_calls": int(ws.minimize_calls) if ws is not None else 0,
    "oracle_outside_calls": int(ws.oracle_outside_calls) if ws is not None else 0,
  }


def _bench_card(
  spec: dict,
  *,
  include_binom_time: bool,
  compare_cost_bias: bool,
  skip_full: bool,
  minlp_time_limit: float | None,
  oracle_time_limit: float | None,
) -> list[dict]:
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
    variants: list[tuple[str, str, bool, bool, bool]] = [
      ("full", "full", True, False, True),
      ("feasibility+cuts", "feasibility", True, False, True),
      ("feasibility sum-only", "feasibility", False, False, True),
    ]
    if compare_cost_bias:
      variants.append(
        ("feasibility+cuts midpoint", "feasibility", True, False, False),
      )
    if skip_full:
      variants = [v for v in variants if v[0] != "full"]
    if include_binom_time:
      variants.extend(
        (
          ("feasibility+cuts+time", "feasibility", True, True, True),
          ("feasibility sum+time", "feasibility", False, True, True),
        )
      )
    for label, mode, cuts, time_cuts, cost_bias in variants:
      print(f"\n=== {spec['label']} t={t} {label} ===", flush=True)
      row = _run_one(
        km,
        t,
        crossing_mode=mode,
        component_cuts=cuts,
        binom_time_cuts=time_cuts,
        cost_biased_bisection=cost_bias,
        minlp_time_limit=minlp_time_limit,
        oracle_time_limit=oracle_time_limit,
      )
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
      f"{'t':>8} {'variant':<22} {'wall(s)':>8} {'oracle_W':>9} {'min_W':>9} "
      f"{'total_W':>9} {'Wspd':>6} {'oci':>4} {'minc':>5} "
      f"{'ci_lo':>10} {'ci_hi':>10} {'dlo':>10} {'dhi':>10}",
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
        w_speedup = full["total_work"] / max(r["total_work"], 1e-9)
        print(
          f"{r['t']:8.4f} {r['variant']:<22} {r['seconds']:8.1f} "
          f"{r['oracle_work']:9.1f} {r['minimize_work']:9.1f} {r['total_work']:9.1f} "
          f"{w_speedup:6.2f}x {r['oracle_calls']:4d} {r['minimize_calls']:5d} "
          f"{r['ci_lo']:10.6f} {r['ci_hi']:10.6f} "
          f"{dlo:10.2e} {dhi:10.2e}",
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
  parser.add_argument(
    "--include-binom-time-cuts",
    action="store_true",
    help="Also benchmark per-death-time binom_piece cuts",
  )
  parser.add_argument(
    "--targets",
    nargs="+",
    type=float,
    default=None,
    help="Override death times to benchmark (nearest match per card)",
  )
  parser.add_argument(
    "--compare-cost-bias",
    action="store_true",
    help="Also run feasibility+cuts with midpoint bisection (cost_biased=False)",
  )
  parser.add_argument(
    "--skip-full",
    action="store_true",
    help="Skip crossing_mode=full (inject baseline rows via --full-baseline)",
  )
  parser.add_argument(
    "--full-baseline",
    type=pathlib.Path,
    default=None,
    help="JSON list of full-mode result rows to prepend when --skip-full",
  )
  parser.add_argument(
    "--minlp-time-limit",
    type=float,
    default=None,
    help="Gurobi TimeLimit (seconds) per full minimize; bench-only safety valve",
  )
  parser.add_argument(
    "--oracle-time-limit",
    type=float,
    default=None,
    help="Gurobi TimeLimit (seconds) per excess_at_most oracle call",
  )
  args = parser.parse_args()

  all_rows: list[dict] = []
  if args.skip_full and args.full_baseline is not None:
    baseline_rows = json.loads(args.full_baseline.read_text(encoding="utf-8"))
    all_rows.extend(baseline_rows)
  for key in args.cards:
    spec = dict(CARD_SPECS[key])
    if args.targets is not None:
      spec["targets"] = args.targets
    all_rows.extend(
      _bench_card(
        spec,
        include_binom_time=args.include_binom_time_cuts,
        compare_cost_bias=args.compare_cost_bias,
        skip_full=args.skip_full,
        minlp_time_limit=args.minlp_time_limit,
        oracle_time_limit=args.oracle_time_limit,
      )
    )
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
