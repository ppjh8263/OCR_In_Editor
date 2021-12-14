#!/bin/bash
CONFS=("optuna_config/base_rnet.json" "optuna_config/base_crnn_1lstm.json" "optuna_config/base_crnn_4lstm.json")
for i in ${CONFS[@]}
do
    python tune.py \
    --config $i \
    --storage "<db storage>"
done