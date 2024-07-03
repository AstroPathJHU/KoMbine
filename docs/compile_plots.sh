set -euxo pipefail

export PYTHONWARNINGS=error

cd $(dirname $0)
rocpicker_mc ../test/datacards/lung/datacard_neighborhoods_poisson.txt ./poisson_roc_neighborhoods.pdf --flip-sign
rocpicker_mc ../test/datacards/lung/datacard_neighborhoods_systematics.txt ./lognormal_roc_neighborhoods.pdf --flip-sign
rocpicker_mc ../test/datacards/lung/datacard_cells_poisson.txt ./poisson_roc_cells.pdf --flip-sign
rocpicker_mc ../test/datacards/lung/datacard_cells_systematics.txt ./lognormal_roc_cells.pdf --flip-sign

rocpicker_discrete ../test/datacards/simple_examples/datacard_example_1.txt --roc-filename discrete_exampleroc.pdf --scan-filename discrete_scan.pdf --roc-errors-filename discrete_exampleroc_errors.pdf --y-upper-limit 20 --npoints 100
rocpicker_discrete ../test/datacards/simple_examples/datacard_example_2.txt --scan-filename discrete_scan_compare_to_delta_functions.pdf --y-upper-limit 20 --npoints 100

rocpicker_delta_functions ../test/datacards/simple_examples/datacard_example_2.txt --roc-filename deltafunctions_exampleroc.pdf --scan-filename deltafunctions_scan.pdf --roc-errors-filename deltafunctions_exampleroc_errors.pdf --y-upper-limit 20 --npoints 100
rocpicker_delta_functions ../test/datacards/simple_examples/datacard_example_2.txt --scan-filename delta_functions_scan_compare_to_discrete.pdf --y-upper-limit 20 --npoints 100

python ../test/test_continuous_distributions.py
