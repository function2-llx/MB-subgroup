#!/usr/bin/env zsh

set -x

for model in SVM LR KNN RF XGB MLP; do
    python scripts/cls/metrics.py radiomics/case-output/$model.xlsx --output_dir radiomics/plot/$model --all
done
