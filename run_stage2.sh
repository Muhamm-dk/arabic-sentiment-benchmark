#!/usr/bin/env bash
set -e
for model in arabert marbert xlmr; do
  [ -f results/runs/labr-$model-s0.json ] && continue
  extra=""
  [ "$model" = "xlmr" ] && extra="--batch_size 16 --grad_accum 2"
  uv run python -m arabic_sentiment.finetune --dataset labr --model $model --seed 0 --epochs 4 $extra
done
