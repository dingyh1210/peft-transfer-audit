# Run matrix

Every run starts from `train_a1_lora.yaml` and changes only the fields below.
All other settings stay untouched: `lora_rank: 8`, `lora_alpha: 16`,
`lora_dropout: 0`, `cutoff_len: 2048`, `max_samples: 10000`,
`num_train_epochs: 2`, `lr_scheduler_type: cosine`, `warmup_steps: 40`,
`bf16: true`, `gradient_checkpointing: true`, `template: qwen`.

## Method switches (finetuning section)

| Method | YAML fields |
| --- | --- |
| LoRA | `finetuning_type: lora`, `use_dora: false`, `pissa_init: false` |
| DoRA | `finetuning_type: lora`, `use_dora: true` |
| LoRA+ | `finetuning_type: lora`, `loraplus_lr_ratio: 16` (B learns 16x faster than A) |
| PiSSA | `finetuning_type: lora`, `pissa_init: true` |

## Pilot at 1.5B (8 runs, seed 42, learning_rate 1.0e-4)

`model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct`

| Dataset field | Methods |
| --- | --- |
| `dataset: alpaca_gpt4_en` | LoRA, DoRA, LoRA+, PiSSA |
| `dataset: metamathqa` | LoRA, DoRA, LoRA+, PiSSA |

## Scale sweep on MetaMathQA (seed 42, learning_rate 1.0e-4)

| Model | Methods |
| --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | LoRA, PiSSA |
| `Qwen/Qwen2.5-3B-Instruct` | LoRA, PiSSA |
| `Qwen/Qwen2.5-7B-Instruct` | LoRA, PiSSA |

(The 1.5B rows of the sweep are the pilot math runs above.)

## Robustness runs (all PiSSA on MetaMathQA)

| Run | Changes |
| --- | --- |
| 1.5B, seed 43 | `seed: 43` |
| 1.5B, seed 44 | `seed: 44` |
| 1.5B, half learning rate | `learning_rate: 5.0e-5` (seed 42) |
| 7B, seed 43 | `model_name_or_path: Qwen/Qwen2.5-7B-Instruct`, `seed: 43` |

## Batch split (fits a single 24 GB GPU)

`per_device_train_batch_size x gradient_accumulation_steps = 16` in every
run; only the split changes with model size, e.g. `4 x 4` at 1.5B and
`1 x 16` at 7B. Training loss is unaffected because the effective batch
is constant.
