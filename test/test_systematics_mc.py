import os, pathlib
from roc_picker.systematics_mc import plot_from_datacard

here = pathlib.Path(__file__).parent
datacards = here/"datacards"
docs = here.parent/"docs"

# Define paths to datacards
datacards = {
  "neighborhoods_no_systematics": datacards/"datacard_neighborhoods_no_systematics.txt",
  "neighborhoods_no_poisson_uncertainty": datacards/"datacard_neighborhoods_no_poisson_uncertainty.txt",
  "cells_no_systematics": datacards/"datacard_cells_no_systematics.txt",
  "cells_no_poisson_uncertainty": datacards/"datacard_cells_no_poisson_uncertainty.txt"
}

# Define output paths for plots
output_paths = {
  "poisson_roc_neighborhoods": docs/"poisson_roc_neighborhoods.pdf",
  "lognormal_roc_neighborhoods": docs/"lognormal_roc_neighborhoods.pdf",
  "poisson_roc_cells": docs/"poisson_roc_cells.pdf",
  "lognormal_roc_cells": docs/"lognormal_roc_cells.pdf"
}

if __name__ == "__main__":
  # Generate plots for neighborhoods without systematics (Poisson uncertainty)
  plot_from_datacard(datacards["neighborhoods_no_systematics"], output_paths["poisson_roc_neighborhoods"])

  # Generate plots for neighborhoods with systematics but without Poisson uncertainty
  plot_from_datacard(datacards["neighborhoods_no_poisson_uncertainty"], output_paths["lognormal_roc_neighborhoods"], id_start=100)

  # Generate plots for cells without systematics (Poisson uncertainty)
  plot_from_datacard(datacards["cells_no_systematics"], output_paths["poisson_roc_cells"], id_start=200)

  # Generate plots for cells with systematics but without Poisson uncertainty
  plot_from_datacard(datacards["cells_no_poisson_uncertainty"], output_paths["lognormal_roc_cells"], id_start=300)
