#!/usr/bin/env bash
set -e
for ds in hard hard_notitle; do
  for model in arabert marbert xlmr; do
    [ -f results/runs/$ds-$model-s0-n20000.json ] && continue
    extra=""
    [ "$model" = "xlmr" ] && extra="--batch_size 16 --grad_accum 2"
    uv run python -m arabic_sentiment.finetune --dataset $ds --model $model --seed 0 --epochs 3 --train_subsample 20000 $extra
  done
done
