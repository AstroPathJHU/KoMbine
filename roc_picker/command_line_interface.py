"""
Command line interface functions for ROC Picker.
"""

import argparse
import pathlib

from kombine.datacard import Datacard


def plot_systematics_mc_roc():
  """
  Run MC method from a datacard.
  """
  # pylint: disable=line-too-long
  parser = argparse.ArgumentParser(description="Run MC method from a datacard.")
  parser.add_argument("datacard", type=pathlib.Path, help="Path to the datacard file.")
  parser.add_argument("output_file", type=pathlib.Path, help="Path to the output file for the plot.")
  parser.add_argument("--nrocs", type=int, help="Number of MC samples to generate.", default=10000, dest="size")
  parser.add_argument("--random-seed", type=int, help="Random seed for generation", dest="random_state", default=123456)
  parser.add_argument("--flip-sign", action="store_true", help="flip the sign of the observable (use this if AUC is < 0.5 and you want it to be > 0.5)")
  # pylint: enable=line-too-long

  args = parser.parse_args()
  datacard = Datacard.parse_datacard(args.__dict__.pop("datacard"))
  rd = datacard.systematics_mc_roc(flip_sign=args.__dict__.pop("flip_sign"))
  rocs = rd.generate(size=args.__dict__.pop("size"), random_state=args.__dict__.pop("random_state"))
  rocs.plot(saveas=args.__dict__.pop("output_file"))
  if args.__dict__:
    raise ValueError(f"Unused arguments: {args.__dict__}")


def plot_discrete_roc():
  """
  Run discrete method from a datacard.
  """
  # pylint: disable=line-too-long
  parser = argparse.ArgumentParser(description="Run discrete method from a datacard.")
  parser.add_argument("datacard", type=pathlib.Path, help="Path to the datacard file.")
  parser.add_argument("--roc-filename", type=pathlib.Path, help="Path to the output file for the ROC curve.", dest="rocfilename")
  parser.add_argument("--roc-errors-filename", type=pathlib.Path, help="Path to the output file for the ROC curve with error bands.", dest="rocerrorsfilename")
  parser.add_argument("--scan-filename", type=pathlib.Path, help="Path to the output file for the likelihood scan", dest="scanfilename")
  parser.add_argument("--y-upper-limit", type=float, help="y axis upper limit of the likelihood scan plot", dest="yupperlim")
  parser.add_argument("--npoints", type=int, help="number of points in the likelihood scan", dest="npoints")
  parser.add_argument("--flip-sign", action="store_true", help="flip the sign of the observable (use this if AUC is < 0.5 and you want it to be > 0.5)")
  # pylint: enable=line-too-long

  args = parser.parse_args()
  datacard = Datacard.parse_datacard(args.__dict__.pop("datacard"))
  discrete = datacard.discrete_roc(flip_sign=args.__dict__.pop("flip_sign"))
  discrete.make_plots(
    filenames=[
      args.__dict__.pop("rocfilename"),
      args.__dict__.pop("scanfilename"),
      args.__dict__.pop("rocerrorsfilename"),
    ],
    yupperlim=args.__dict__.pop("yupperlim"),
    npoints=args.__dict__.pop("npoints"),
  )
  if args.__dict__:
    raise ValueError(f"Unused arguments: {args.__dict__}")


def plot_delta_functions_roc():
  """
  Run delta functions method from a datacard.
  """
  # pylint: disable=line-too-long
  parser = argparse.ArgumentParser(description="Run delta functions method from a datacard.")
  parser.add_argument("datacard", type=pathlib.Path, help="Path to the datacard file.")
  parser.add_argument("--roc-filename", type=pathlib.Path, help="Path to the output file for the ROC curve.", dest="rocfilename")
  parser.add_argument("--roc-errors-filename", type=pathlib.Path, help="Path to the output file for the ROC curve with error bands.", dest="rocerrorsfilename")
  parser.add_argument("--scan-filename", type=pathlib.Path, help="Path to the output file for the likelihood scan", dest="scanfilename")
  parser.add_argument("--y-upper-limit", type=float, help="y axis upper limit of the likelihood scan plot", dest="yupperlim")
  parser.add_argument("--npoints", type=int, help="number of points in the likelihood scan", dest="npoints")
  parser.add_argument("--flip-sign", action="store_true", help="flip the sign of the observable (use this if AUC is < 0.5 and you want it to be > 0.5)")
  # pylint: enable=line-too-long

  args = parser.parse_args()
  datacard = Datacard.parse_datacard(args.__dict__.pop("datacard"))
  deltafunctions = datacard.delta_functions_roc(flip_sign=args.__dict__.pop("flip_sign"))
  deltafunctions.make_plots(
    filenames=[
      args.__dict__.pop("rocfilename"),
      args.__dict__.pop("scanfilename"),
      args.__dict__.pop("rocerrorsfilename"),
    ],
    yupperlim=args.__dict__.pop("yupperlim"),
    npoints=args.__dict__.pop("npoints"),
  )
  if args.__dict__:
    raise ValueError(f"Unused arguments: {args.__dict__}")