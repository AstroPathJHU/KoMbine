#!/usr/bin/env python3
"""Rebin survival times on the methods_comparison_* card family.

Snaps every patient's survival_time onto K quantile-bin medians taken from the
fixed baseline card, then writes the same survival_time / censored rows into
every sibling methods_comparison_*.txt (observables unchanged).

Notebook 07's full-comparison mode expects K=4 (chosen after capped probes:
K=6 censored a KM arm past 15 min; K=4 finished a hard-card arm in ~8 min).

Usage (from repo root)::

    python test/kombine/datacards/simple_examples/rebin_methods_comparison_times.py
    python test/kombine/datacards/simple_examples/rebin_methods_comparison_times.py --n-bins 4
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BASELINE = HERE / "methods_comparison_fixed.txt"
SIBLINGS = sorted(HERE.glob("methods_comparison_*.txt"))


def rebin_times(times: np.ndarray, n_bins: int) -> np.ndarray:
  """Snap each time onto the median of its quantile bin."""
  if n_bins < 1:
    raise ValueError(f"n_bins must be >= 1, got {n_bins}")
  if n_bins >= len(times):
    return times.copy()
  edges = np.quantile(times, np.linspace(0.0, 1.0, n_bins + 1))
  edges[-1] += 1e-9
  idx = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, n_bins - 1)
  return np.array([float(np.median(times[idx == b])) for b in idx])


def parse_rows(text: str) -> dict[str, list[str]]:
  rows: dict[str, list[str]] = {}
  for line in text.splitlines():
    parts = line.split("\t")
    if len(parts) > 1 and parts[0] in {"survival_time", "censored"}:
      rows[parts[0]] = parts[1:]
  if "survival_time" not in rows or "censored" not in rows:
    raise ValueError("datacard missing survival_time or censored row")
  return rows


def rewrite_card(
  path: pathlib.Path,
  times: np.ndarray,
  censored: list[str],
  *,
  n_bins: int,
  n_unique: int,
) -> None:
  text = path.read_text(encoding="utf-8")
  lines = text.splitlines()
  out: list[str] = []
  header_done = False
  for line in lines:
    if line.startswith("survival_time"):
      out.append("\t".join(["survival_time"] + [f"{t:.4f}" for t in times]))
      continue
    if line.startswith("censored"):
      out.append("\t".join(["censored"] + list(censored)))
      continue
    if (
      not header_done
      and line.startswith("#")
      and ("n=50" in line or "deaths=" in line or "Notebook 07" in line)
    ):
      # Replace the stats comment once with the rebinned description.
      if "n=50" in line or "deaths=" in line:
        out.append(
          f"# n=50, n_death_times={n_unique} (quantile-bin medians, K={n_bins}), "
          "target HR=6, n_per_group=25, seed=42"
        )
        out.append(
          "# Survival times rebinned from methods_comparison_fixed.txt; "
          "shared across sibling methods_comparison_* cards"
        )
        header_done = True
        continue
    out.append(line)
  # If no stats comment was found, insert after the first # block opener.
  if not header_done:
    inserted = False
    final: list[str] = []
    for line in out:
      final.append(line)
      if not inserted and line.startswith("#") and "Notebook 07" in line:
        final.append(
          f"# n=50, n_death_times={n_unique} (quantile-bin medians, K={n_bins}), "
          "target HR=6, n_per_group=25, seed=42"
        )
        final.append(
          "# Survival times rebinned from methods_comparison_fixed.txt; "
          "shared across sibling methods_comparison_* cards"
        )
        inserted = True
    out = final
  path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--n-bins", type=int, default=4,
    help="quantile bins (default 4; notebook full-comparison mode)",
  )
  args = parser.parse_args()

  baseline_text = BASELINE.read_text(encoding="utf-8")
  rows = parse_rows(baseline_text)
  raw_times = np.array([float(t) for t in rows["survival_time"]])
  censored = rows["censored"]
  binned = rebin_times(raw_times, args.n_bins)
  # Death times among uncensored patients drive the KM grid.
  death_mask = [c not in ("1", "True", "true") for c in censored]
  n_unique = len({round(t, 10) for t, d in zip(binned, death_mask) if d})
  print(
    f"baseline n={len(raw_times)} raw_unique_deaths="
    f"{len({round(t, 10) for t, d in zip(raw_times, death_mask) if d})} "
    f"-> K={args.n_bins} unique_death_times={n_unique}",
    flush=True,
  )

  for path in SIBLINGS:
    rewrite_card(
      path, binned, censored, n_bins=args.n_bins, n_unique=n_unique,
    )
    print(f"wrote {path.name}", flush=True)


if __name__ == "__main__":
  main()
