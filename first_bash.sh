#!/bin/bash

OUTPUT_FILE="U1_test_run"
array=(1 2 3 4 5 6 7)

LATTICE_SIZE=16

STEP_SIZE=0.05
THERM_STEPS=512
GEN_STEPS=10000
SAVE_INTERVAL=10

mkdir -p $OUTPUT_FILE

COUNT=0
for BETA in "${array[@]}"; do
    python3 u1_test.py -L $LATTICE_SIZE -b $BETA --step_size $STEP_SIZE --therm_steps $THERM_STEPS --gen_steps $GEN_STEPS --save_interval $SAVE_INTERVAL -o "$OUTPUT_FILE/configs_$COUNT"

    ((COUNT++))
done

echo "${array[@]}" | jq -s '{beta: .}' > $OUTPUT_FILE/parameters.json

json_string=$(jq --argjson lattice_size $LATTICE_SIZE '.lattice_size = $lattice_size' $OUTPUT_FILE/parameters.json)
echo "$json_string" > $OUTPUT_FILE/parameters.json

python3 u1_data_analysis.py -i $OUTPUT_FILE