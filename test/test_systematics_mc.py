#this script was generated using ChatGPT

import pathlib
from roc_picker.datacard import Datacard

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

def plot(datacard, output, id_start=0, size=10000, random_state=123456):
  d = Datacard.parse_datacard(datacard)
  rd = d.systematics_mc(id_start=id_start)
  rocs = rd.generate(size=size, random_state=random_state)
  rocs.plot(output)

if __name__ == "__main__":
  # Generate plots for neighborhoods without systematics (Poisson uncertainty)
  plot(datacards["neighborhoods_no_systematics"], output_paths["poisson_roc_neighborhoods"])

  # Generate plots for neighborhoods with systematics but without Poisson uncertainty
  plot(datacards["neighborhoods_no_poisson_uncertainty"], output_paths["lognormal_roc_neighborhoods"], id_start=100)

  # Generate plots for cells without systematics (Poisson uncertainty)
  plot(datacards["cells_no_systematics"], output_paths["poisson_roc_cells"], id_start=200)

  # Generate plots for cells with systematics but without Poisson uncertainty
  plot(datacards["cells_no_poisson_uncertainty"], output_paths["lognormal_roc_cells"], id_start=300)
