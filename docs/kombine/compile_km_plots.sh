#!/bin/bash

#This script requires a Gurobi license.  They are available for free for academic use.
#Otherwise, all that's needed is to pip install kombinekm.

set -euxo pipefail

export PYTHONWARNINGS=error
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd $(dirname $0)

FONT_SIZES=(--legend-fontsize 16 --title-fontsize 16 --label-fontsize 16 --tick-fontsize 16)
COMMON_ARGS_BIG_PLOT=(--figsize 7 7 ${FONT_SIZES[@]})
COMMON_ARGS_SMALL_PLOT=(--figsize 5 5 ${FONT_SIZES[@]})

kombine ../../test/kombine/datacards/simple_examples/poisson_ratio_km_censoring.txt km_example.pdf --parameter-min 0.45 "${COMMON_ARGS_BIG_PLOT[@]}" --title "Kaplan–Meier Example"

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
COMMON_TWOGROUPS_ARGS=(--exclude-nominal --print-progress  --xlabel 'Time (Months)' --ylabel "$YLABEL" --patient-wise-only-suffix '' --binomial-only-suffix '' --pvalue-fontsize 16 "${COMMON_ARGS_SMALL_PLOT[@]}")

kombine_twogroups ../../test/kombine/datacards/lung/datacard_cells_${SURVIVAL_TYPE}.txt lung_cells_km_${SURVIVAL_TYPE}.pdf --parameter-threshold "$CELL_THRESHOLD" "${COMMON_TWOGROUPS_ARGS[@]}" --title "CD8+FoxP3+ Cells" --pvalue-format '.2f' --legend-saveas lung_cells_km_${SURVIVAL_TYPE}_legend.pdf
kombine_twogroups ../../test/kombine/datacards/lung/datacard_donuts_${SURVIVAL_TYPE}.txt lung_donuts_km_${SURVIVAL_TYPE}.pdf --parameter-threshold "$DONUT_THRESHOLD" "${COMMON_TWOGROUPS_ARGS[@]}" --title "DONUTS" --pvalue-format '.2f' --legend-saveas lung_donuts_km_${SURVIVAL_TYPE}_legend.pdf
kombine_twogroups ../../test/kombine/datacards/lung/datacard_cells_${SURVIVAL_TYPE}.txt lung_cells_km_${SURVIVAL_TYPE}_patient_wise.pdf --parameter-threshold "$CELL_THRESHOLD" --include-patient-wise-only --exclude-full-nll "${COMMON_TWOGROUPS_ARGS[@]}" --title "CD8+FoxP3+ Cells, Patient-Wise Errors" --pvalue-format '.2f' --legend-saveas lung_cells_km_${SURVIVAL_TYPE}_legend.pdf
kombine_twogroups ../../test/kombine/datacards/lung/datacard_donuts_${SURVIVAL_TYPE}.txt lung_donuts_km_${SURVIVAL_TYPE}_patient_wise.pdf --parameter-threshold "$DONUT_THRESHOLD" --include-patient-wise-only --exclude-full-nll "${COMMON_TWOGROUPS_ARGS[@]}" --title "DONUTS, Patient-Wise Errors" --pvalue-format '.2f' --legend-saveas lung_donuts_km_${SURVIVAL_TYPE}_legend.pdf
kombine_twogroups ../../test/kombine/datacards/lung/datacard_cells_${SURVIVAL_TYPE}.txt lung_cells_km_${SURVIVAL_TYPE}_binomial.pdf --parameter-threshold "$CELL_THRESHOLD" --include-binomial-only --exclude-full-nll "${COMMON_TWOGROUPS_ARGS[@]}" --title "CD8+FoxP3+ Cells, Binomial Errors" --pvalue-format '.2f' --legend-saveas lung_cells_km_${SURVIVAL_TYPE}_legend.pdf
kombine_twogroups ../../test/kombine/datacards/lung/datacard_donuts_${SURVIVAL_TYPE}.txt lung_donuts_km_${SURVIVAL_TYPE}_binomial.pdf --parameter-threshold "$DONUT_THRESHOLD" --include-binomial-only --exclude-full-nll "${COMMON_TWOGROUPS_ARGS[@]}" --title "DONUTS, Binomial Errors" --pvalue-format '.2f' --legend-saveas lung_donuts_km_${SURVIVAL_TYPE}_legend.pdf

COMMON_EXPONENTIAL_GREENWOOD_ARGS=(--exclude-nominal --include-exponential-greenwood --include-binomial-only --binomial-only-suffix 'KoMbine' --exponential-greenwood-suffix 'e. G.' --exclude-full-nll "${COMMON_ARGS_BIG_PLOT[@]}")
kombine ../../test/kombine/datacards/simple_examples/fixed_km_censoring.txt comparison_to_greenwood_small_n.pdf "${COMMON_EXPONENTIAL_GREENWOOD_ARGS[@]}" --title 'Comparison to Greenwood, $N=12$'
kombine ../../test/kombine/datacards/simple_examples/fixed_km_censoring_many_patients.txt comparison_to_greenwood_large_n.pdf "${COMMON_EXPONENTIAL_GREENWOOD_ARGS[@]}" --title 'Comparison to Greenwood, $N=100$'

python compare_p_value.py --n-patients 10 --n-trials 100 --allow-ties --save-as p_value_comparison_10_patients.pdf "${COMMON_ARGS_SMALL_PLOT[@]}"
python compare_p_value.py --n-patients 100 --n-trials 100 --allow-ties --save-as p_value_comparison_100_patients.pdf "${COMMON_ARGS_SMALL_PLOT[@]}"
python compare_p_value.py --n-patients 1000 --n-trials 100 --allow-ties --save-as p_value_comparison_1000_patients.pdf "${COMMON_ARGS_SMALL_PLOT[@]}"
python compare_p_value.py --n-patients 10 --n-trials 100 --no-ties --save-as p_value_comparison_10_patients_no_ties.pdf "${COMMON_ARGS_SMALL_PLOT[@]}"
python compare_p_value.py --n-patients 100 --n-trials 100 --no-ties --save-as p_value_comparison_100_patients_no_ties.pdf "${COMMON_ARGS_SMALL_PLOT[@]}"
#exclude this plot - it takes too long and doesn't add much value
#python compare_p_value.py --n-patients 1000 --n-trials 100 --no-ties --save-as p_value_comparison_1000_patients_no_ties.pdf "${COMMON_ARGS_SMALL_PLOT[@]}"