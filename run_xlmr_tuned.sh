#!/usr/bin/env bash
set -e
for ds in ajgt astd; do
  for seed in 0 1 2; do
    [ -f results/runs/$ds-xlmr-s$seed-tuned.json ] && continue
    uv run python -m arabic_sentiment.finetune --dataset $ds --model xlmr --seed $seed --epochs 10 --lr 3e-5 --batch_size 16 --run_tag tuned
  done
done
