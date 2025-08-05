#!/bin/bash

set -euxo pipefail

export PYTHONWARNINGS=error
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd $(dirname $0)
cp ../logo.png .
rocpicker_mc ../test/datacards/lung/datacard_neighborhoods_poisson.txt ./poisson_roc_neighborhoods.pdf --flip-sign
rocpicker_mc ../test/datacards/lung/datacard_neighborhoods_systematics.txt ./lognormal_roc_neighborhoods.pdf --flip-sign
rocpicker_mc ../test/datacards/lung/datacard_cells_poisson.txt ./poisson_roc_cells.pdf --flip-sign
rocpicker_mc ../test/datacards/lung/datacard_cells_systematics.txt ./lognormal_roc_cells.pdf --flip-sign

rocpicker_discrete ../test/datacards/simple_examples/example_roc.txt --roc-filename discrete_exampleroc.pdf --scan-filename discrete_scan.pdf --roc-errors-filename discrete_exampleroc_errors.pdf --y-upper-limit 20 --npoints 100
rocpicker_discrete ../test/datacards/simple_examples/symmetric_roc.txt --scan-filename discrete_scan_compare_to_delta_functions.pdf --y-upper-limit 20 --npoints 100

rocpicker_delta_functions ../test/datacards/simple_examples/symmetric_roc.txt --roc-filename deltafunctions_exampleroc.pdf --scan-filename deltafunctions_scan.pdf --roc-errors-filename deltafunctions_exampleroc_errors.pdf --y-upper-limit 20 --npoints 100
rocpicker_delta_functions ../test/datacards/simple_examples/example_roc.txt --scan-filename delta_functions_scan_compare_to_discrete.pdf --y-upper-limit 20 --npoints 100

python ../test/test_continuous_distributions.py

kombine ../test/datacards/simple_examples/poisson_ratio_km_censoring.txt km_example.pdf --parameter-min 0.45
kombine_twogroups ../test/datacards/lung/datacard_cells_OS.txt lung_cells_km_OS.pdf --parameter-threshold 0.4 --print-progress
kombine_twogroups ../test/datacards/lung/datacard_donuts_OS.txt lung_donuts_km_OS.pdf --parameter-threshold 350 --print-progress
kombine_twogroups ../test/datacards/lung/datacard_cells_OS.txt lung_cells_km_OS_patient_wise.pdf --parameter-threshold 0.4 --include-patient-wise-only --exclude-full-nll --print-progress
kombine_twogroups ../test/datacards/lung/datacard_donuts_OS.txt lung_donuts_km_OS_patient_wise.pdf --parameter-threshold 350 --include-patient-wise-only --exclude-full-nll --print-progress
kombine_twogroups ../test/datacards/lung/datacard_cells_OS.txt lung_cells_km_OS_binomial.pdf --parameter-threshold 0.4 --include-binomial-only --exclude-full-nll --print-progress
kombine_twogroups ../test/datacards/lung/datacard_donuts_OS.txt lung_donuts_km_OS_binomial.pdf --parameter-threshold 350 --include-binomial-only --exclude-full-nll --print-progress

kombine ../test/datacards/simple_examples/fixed_km_censoring.txt comparison_to_greenwood_small_n.pdf --exclude-nominal --include-exponential-greenwood --include-binomial-only --exclude-full-nll
kombine ../test/datacards/simple_examples/fixed_km_censoring_many_patients.txt comparison_to_greenwood_large_n.pdf --exclude-nominal --include-exponential-greenwood --include-binomial-only --exclude-full-nll --print-progress
