#!/usr/bin/env bash
# usage: ./run_llm_queue.sh qwen|gptoss|allam   (rerun after a quota reset, it resumes)
m=$1
for spec in "ajgt" "astd" "labr --n 300" "hard --n 300" "hard_notitle --n 300"; do
  uv run python -m arabic_sentiment.llm_eval --model $m --dataset $spec --split test --prompt en
done
