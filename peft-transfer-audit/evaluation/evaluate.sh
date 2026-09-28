#!/usr/bin/env bash
# Evaluation commands for the audit (LM Evaluation Harness, vLLM backend).
#
# Usage:
#   bash evaluation/evaluate.sh <MODEL_DIR> <RUN_NAME>
#
# Example:
#   bash evaluation/evaluate.sh exports/pilot_math_pissa_s42 pilot_math_pissa_s42
set -e
MODEL_DIR=$1
RUN_NAME=$2

# Main evaluation. GSM8K is 5-shot and reports both built-in answer filters
# (strict-match and flexible-extract); MMLU is 0-shot (acc);
# ARC-E and ARC-C are 0-shot (acc_norm).
lm_eval --model vllm \
  --model_args pretrained=${MODEL_DIR},dtype=bfloat16,gpu_memory_utilization=0.85 \
  --tasks gsm8k,mmlu,arc_easy,arc_challenge \
  --batch_size 32 \
  --output_path results/${RUN_NAME}

# ---------------------------------------------------------------------------
# Raw-output inspection (needed for the termination statistics of mode 5).
# Rerun GSM8K with sample logging, then count the '####' markers:
#
#   lm_eval --model vllm \
#     --model_args pretrained=${MODEL_DIR},dtype=bfloat16,gpu_memory_utilization=0.85 \
#     --tasks gsm8k --batch_size 32 --log_samples \
#     --output_path results/${RUN_NAME}_rerun
#   python scripts/count_broken_terminations.py results/${RUN_NAME}_rerun
#
# Cross-engine replication (HuggingFace backend, 200-example subset):
#
#   lm_eval --model hf \
#     --model_args pretrained=${MODEL_DIR},dtype=bfloat16 \
#     --tasks gsm8k --batch_size 8 --limit 200 \
#     --output_path results/${RUN_NAME}_hf200
# ---------------------------------------------------------------------------
