#!/bin/bash

set -euxo pipefail

export PYTHONWARNINGS=error
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd $(dirname $0)

COMMON_ARGS=(--figsize 7 7 --legend-fontsize 16 --title-fontsize 16 --label-fontsize 16 --tick-fontsize 16)

kombine ../../test/kombine/datacards/simple_examples/poisson_ratio_km_censoring.txt km_example.pdf --parameter-min 0.45 "${COMMON_ARGS[@]}" --title "Kaplan–Meier Example"

# Note: lung data would normally be in ../../test/kombine/datacards/lung/ but we need to check if this data should be shared or duplicated
SURVIVAL_TYPE=RFS
if [ $SURVIVAL_TYPE = RFS ]; then
  YLABEL="Regression-Free Survival Probability"
  CELL_THRESHOLD=0.4
  DONUT_THRESHOLD=1130
elif [ $SURVIVAL_TYPE = OS ]; then
  YLABEL="Overall Survival Probability"
  CELL_THRESHOLD=0.4
  DONUT_THRESHOLD=350
fi
COMMON_TWOGROUPS_ARGS=(--exclude-nominal --print-progress  --xlabel 'Time (Months)' --ylabel "$YLABEL" --legend-loc 'lower right'  --patient-wise-only-suffix '' --binomial-only-suffix '' --pvalue-fontsize 16 "${COMMON_ARGS[@]}")

# Using lung data from main test directory for now - may need to be duplicated or referenced differently
kombine_twogroups ../../test/datacards/lung/datacard_cells_${SURVIVAL_TYPE}.txt lung_cells_km_${SURVIVAL_TYPE}.pdf --parameter-threshold "$CELL_THRESHOLD" "${COMMON_TWOGROUPS_ARGS[@]}" --title "CD8+FoxP3+ Cells" --pvalue-format '.2f'
kombine_twogroups ../../test/datacards/lung/datacard_donuts_${SURVIVAL_TYPE}.txt lung_donuts_km_${SURVIVAL_TYPE}.pdf --parameter-threshold "$DONUT_THRESHOLD" "${COMMON_TWOGROUPS_ARGS[@]}" --title "DONUTS" --pvalue-format '.2f'
kombine_twogroups ../../test/datacards/lung/datacard_cells_${SURVIVAL_TYPE}.txt lung_cells_km_${SURVIVAL_TYPE}_patient_wise.pdf --parameter-threshold "$CELL_THRESHOLD" --include-patient-wise-only --exclude-full-nll "${COMMON_TWOGROUPS_ARGS[@]}" --title "CD8+FoxP3+ Cells, Patient-Wise Errors" --pvalue-format '.2f'
kombine_twogroups ../../test/datacards/lung/datacard_donuts_${SURVIVAL_TYPE}.txt lung_donuts_km_${SURVIVAL_TYPE}_patient_wise.pdf --parameter-threshold "$DONUT_THRESHOLD" --include-patient-wise-only --exclude-full-nll "${COMMON_TWOGROUPS_ARGS[@]}" --title "DONUTS, Patient-Wise Errors" --pvalue-format '.2f'
kombine_twogroups ../../test/datacards/lung/datacard_cells_${SURVIVAL_TYPE}.txt lung_cells_km_${SURVIVAL_TYPE}_binomial.pdf --parameter-threshold "$CELL_THRESHOLD" --include-binomial-only --exclude-full-nll "${COMMON_TWOGROUPS_ARGS[@]}" --title "CD8+FoxP3+ Cells, Binomial Errors" --pvalue-format '.2f'
kombine_twogroups ../../test/datacards/lung/datacard_donuts_${SURVIVAL_TYPE}.txt lung_donuts_km_${SURVIVAL_TYPE}_binomial.pdf --parameter-threshold "$DONUT_THRESHOLD" --include-binomial-only --exclude-full-nll "${COMMON_TWOGROUPS_ARGS[@]}" --title "DONUTS, Binomial Errors" --pvalue-format '.2f'

COMMON_EXPONENTIAL_GREENWOOD_ARGS=(--exclude-nominal --include-exponential-greenwood --include-binomial-only --binomial-only-suffix 'KoMbine' --exponential-greenwood-suffix 'e. G.' --exclude-full-nll "${COMMON_ARGS[@]}")
kombine ../../test/kombine/datacards/simple_examples/fixed_km_censoring.txt comparison_to_greenwood_small_n.pdf "${COMMON_EXPONENTIAL_GREENWOOD_ARGS[@]}" --title 'Comparison to Greenwood, $N=12$'
kombine ../../test/kombine/datacards/simple_examples/fixed_km_censoring_many_patients.txt comparison_to_greenwood_large_n.pdf "${COMMON_EXPONENTIAL_GREENWOOD_ARGS[@]}" --title 'Comparison to Greenwood, $N=100$'

COMMON_P_VALUE_ARGS=(--exclude-nominal --include-binomial-only --exclude-full-nll --pvalue-fontsize 16 --include-logrank-pvalue --parameter-threshold 0.5 "${COMMON_ARGS[@]}")
kombine_twogroups ../../test/kombine/datacards/simple_examples/fixed_km_censoring.txt comparison_to_conventional_p_value_small_n.pdf "${COMMON_P_VALUE_ARGS[@]}" --title 'Comparison to conventional $p$ value method, $N=12$'
kombine_twogroups ../../test/kombine/datacards/simple_examples/fixed_km_censoring_many_patients.txt comparison_to_conventional_p_value_large_n.pdf "${COMMON_P_VALUE_ARGS[@]}" --title 'Comparison to conventional $p$ value method, $N=100$'

python compare_p_value.py --n-patients 10 --n-trials 100 --allow-ties --save-as p_value_comparison_10_patients.pdf "${COMMON_ARGS[@]}"
python compare_p_value.py --n-patients 100 --n-trials 100 --allow-ties --save-as p_value_comparison_100_patients.pdf "${COMMON_ARGS[@]}"
python compare_p_value.py --n-patients 1000 --n-trials 100 --allow-ties --save-as p_value_comparison_1000_patients.pdf "${COMMON_ARGS[@]}"
python compare_p_value.py --n-patients 10 --n-trials 100 --no-ties --save-as p_value_comparison_10_patients_no_ties.pdf "${COMMON_ARGS[@]}"
python compare_p_value.py --n-patients 100 --n-trials 100 --no-ties --save-as p_value_comparison_100_patients_no_ties.pdf "${COMMON_ARGS[@]}"
#exclude this plot - it takes too long and doesn't add much value
#python compare_p_value.py --n-patients 1000 --n-trials 100 --no-ties --save-as p_value_comparison_1000_patients_no_ties.pdf "${COMMON_ARGS[@]}"