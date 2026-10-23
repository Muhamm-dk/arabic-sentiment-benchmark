#!/usr/bin/env bash
set -e
for ds in ajgt astd; do
  for model in arabert marbert xlmr; do
    for seed in 0 1 2; do
      [ -f results/runs/$ds-$model-s$seed.json ] && continue
      uv run python -m arabic_sentiment.finetune --dataset $ds --model $model --seed $seed --epochs 5
    done
  done
done
