---
name: search-config-agent
description: Use when measuring GPU/NPU memory for EMFoundation configs, choosing batch sizes, comparing default.json variants, or regenerating code/configs/grid.json (at most 20 configs).
---

# Search Config Agent

Measure first, never estimate VRAM with formulas.

## Tools
- `python utils/gen_configs.py --d_model ... --n_layers ... --neck ...` — enumerates configs,
  counts exact params/shared ratio on the meta device, filters <1B and >80% shared, writes grid.json.
- `python utils/measure_vram.py --grid code/configs/grid.json --budget 76` — real pretrain step
  (with MAE decoder, visible tokens only) and finetune step (3 tasks, no decoder); reports peaks.
- `python utils/measure_vram.py --config <cfg> --find_max_bs` — max batch per stage.

## Workflow
1. Preflight: `nvidia-smi` (GPU idle), `git status --short`.
2. Generate grid (<= 20 configs) with gen_configs; keep `grad_checkpoint: true` for d_model >= 1024.
3. Measure with measure_vram; drop configs over budget; set batch sizes from `max_bs_*`.
4. Record the table (GPU, torch version, date, peaks) in `docs/gpu_memory_analysis.md`.

## Rules
- Conformer layers belong in `neck_layer_types`, never in `layer_types` (model raises).
- Never overwrite `code/configs/default.json` while experimenting; copy it.
