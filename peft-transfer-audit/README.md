# peft-transfer-audit

Replication materials for the paper
**"Do Parameter-Efficient Fine-Tuning Findings Transfer to Small Language Models?"**

The entire audit runs on a single consumer GPU (NVIDIA RTX 4090, 24 GB).

## Repository contents

| Path | What it is |
| --- | --- |
| `configs/train_a1_lora.yaml` | Master training configuration (LLaMA-Factory). Every run in the paper uses this file with only the fields listed in `configs/RUNS.md` changed. |
| `configs/RUNS.md` | The run matrix: which fields change for each of the pilot, scale-sweep, and robustness runs. |
| `evaluation/evaluate.sh` | The exact LM Evaluation Harness commands (vLLM backend), including the raw-output logging and the HuggingFace-backend cross-check. |
| `scripts/merge_check.py` | The algebraic merge-correctness check for SVD-initialized (PiSSA-style) adapters described in Section III-F of the paper. |
| `scripts/count_broken_terminations.py` | The raw-output analysis used to quantify broken termination (mode 5 in Table VI of the paper). |

## Environment

- LLaMA-Factory (training and adapter export)
- LM Evaluation Harness with the vLLM backend (evaluation)
- PyTorch + safetensors (the two scripts)
- bfloat16 throughout

## Reproducing a run

1. Train, e.g. the 1.5B LoRA general-instruction run:

   ```bash
   llamafactory-cli train configs/train_a1_lora.yaml
   ```

   For any other run, copy the file and change only the fields listed in
   `configs/RUNS.md` (model, method switch, dataset, seed, learning rate,
   and the batch split that fits 24 GB).

2. Merge the adapter (required for PiSSA; the merge check below assumes a
   merged export):

   ```bash
   llamafactory-cli export --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
       --adapter_name_or_path saves/pilot/math/pissa \
       --template qwen --finetuning_type lora \
       --export_dir exports/pilot_math_pissa_s42
   ```

3. Evaluate:

   ```bash
   bash evaluation/evaluate.sh exports/pilot_math_pissa_s42 pilot_math_pissa_s42
   ```

4. Run the two checks:

   ```bash
   python scripts/merge_check.py \
       --base Qwen/Qwen2.5-1.5B-Instruct \
       --merged exports/pilot_math_pissa_s42 \
       --adapter saves/pilot/math/pissa \
       --layer model.layers.0.self_attn.q_proj

   python scripts/count_broken_terminations.py results/pilot_math_pissa_s42_rerun
   ```

## Notes

- All runs use seed 42 unless stated otherwise; the robustness runs use
  seeds 43 and 44 and a halved learning rate (5e-5), see `configs/RUNS.md`.
- GSM8K is evaluated 5-shot with both built-in answer filters of the harness
  (`strict-match` and `flexible-extract`); MMLU is 0-shot (`acc`);
  ARC-E/ARC-C are 0-shot (`acc_norm`).
