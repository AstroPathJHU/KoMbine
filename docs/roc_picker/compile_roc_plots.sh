#!/bin/bash

set -euxo pipefail

export PYTHONWARNINGS=error
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd $(dirname $0)
cp ../../logo.png .
rocpicker_mc ../../test/roc_picker/datacards/lung/datacard_donuts_poisson.txt ./poisson_roc_donuts.pdf --flip-sign
rocpicker_mc ../../test/roc_picker/datacards/lung/datacard_donuts_systematics.txt ./lognormal_roc_donuts.pdf --flip-sign
rocpicker_mc ../../test/roc_picker/datacards/lung/datacard_cells_poisson.txt ./poisson_roc_cells.pdf --flip-sign
rocpicker_mc ../../test/roc_picker/datacards/lung/datacard_cells_systematics.txt ./lognormal_roc_cells.pdf --flip-sign

rocpicker_discrete ../../test/roc_picker/datacards/simple_examples/example_roc.txt --roc-filename discrete_exampleroc.pdf --scan-filename discrete_scan.pdf --roc-errors-filename discrete_exampleroc_errors.pdf --y-upper-limit 20 --npoints 100
rocpicker_discrete ../../test/roc_picker/datacards/simple_examples/symmetric_roc.txt --scan-filename discrete_scan_compare_to_delta_functions.pdf --y-upper-limit 20 --npoints 100

rocpicker_delta_functions ../../test/roc_picker/datacards/simple_examples/symmetric_roc.txt --roc-filename deltafunctions_exampleroc.pdf --scan-filename deltafunctions_scan.pdf --roc-errors-filename deltafunctions_exampleroc_errors.pdf --y-upper-limit 20 --npoints 100
rocpicker_delta_functions ../../test/roc_picker/datacards/simple_examples/example_roc.txt --scan-filename delta_functions_scan_compare_to_discrete.pdf --y-upper-limit 20 --npoints 100

python ../../test/roc_picker/test_continuous_distributions.py