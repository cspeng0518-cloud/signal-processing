# Initial Training Guide for the Electromagnetic Signal Foundation Model

> Repo: `signal-processing` (competition tasks: protocol / modulation / parameter)
> Training machine: Linux + GPU (Ampere or newer with bf16 support; Ascend NPU also works)

---

## 1. Environment Setup

```bash
# Python 3.8+ / PyTorch 2.x
pip install torch safetensors pandas pyarrow numpy tqdm tensorboard matplotlib
```

**No CUDA-specific operator libraries required**: attention / MLA / GDA / Mamba are
all pure-torch native implementations with no dependency on `mamba-ssm`,
`causal-conv1d`, or `flash-attn` — friendly to domestic accelerator migration.
`tqdm` (progress bars) and `tensorboard` (training curves) are hard dependencies;
make sure they are installed on the training machine. Pinned versions recommended:

```bash
pip install -r environment/requirements.txt
```

---

## 2. Code Structure and Data Layout

```
signal-processing/
├── code/
│   ├── configs/default.json      # model/training config (two top-level sections)
│   ├── models/                   # competition-specific: foundation / tokenizer / heads / frequency / backbone
│   │   └── layers/               # generic components: attention / gated_delta / mamba / hybrid
│   ├── training/
│   │   ├── pretrain_mae.py       # stage 1 entry: MAE self-supervised pretraining
│   │   ├── finetune_multitask.py # stage 2 entry: multi-task finetuning
│   │   ├── trainer.py            # training lifecycle (scheduling / checkpointing / export)
│   │   ├── tasks.py              # task losses / evaluation (polymorphic)
│   │   └── common.py             # shared scaffolding (config normalization / MAE loading)
│   ├── utils/                    # dataset / logging / checkpointing / metrics
│   └── infer.py                  # official inference entry
├── datasets/                     # official data: raw chunks (01/02/03, zips extracted)
│   │                              # + prepared memmap ({task}/iq.memap + meta.json,
│   │                              #   produced by utils/prepare_official.py)
│   └── simulated/                 # legacy simulated parquets (fallback only)
├── weights/                      # output directory (checkpoints, exported weights)
└── logs/                         # training metrics (jsonl)
```

Data loading priority: if `datasets/{task}/meta.json` exists (prepared memmap),
it is used automatically; otherwise the loader falls back to
`{split}_{task}.parquet` in the data dir (legacy `train_system.parquet` naming
is also accepted). Keep `--data_dir datasets` in all commands below.

---

## 3. Model Configuration (configs/default.json)

| Key | Default | Notes |
|---|---|---|
| `model.backbone` | `hybrid` | hybrid architecture (GLM-5.2-style per-layer layout) |
| `model.d_model / nhead` | 512 / 8 | head_dim = 64 |
| `model.n_layers` | 27 | layer count kept in the 20-30 range |
| `model.ffn_hidden` | 2048 | SwiGLU FFN width per layer (default 4*d_model) |
| `model.layer_types` | 27-entry list | `full_attention : mla = 1 : 2` (every group of 3: F, M, M) |
| `model.pos_encoding` | `rope` | hybrid backbone requires rope |
| `model.mla_kv/q_lora_rank` | 128 / 128 | MLA KV/Q low-rank compression ranks |
| `training.amp` | `true` | bf16 mixed precision (no loss scaling, quantization-friendly) |

- FFN is **SwiGLU** (Llama/GLM-style gating: `down(silu(gate(x)) * up(x))`, no bias)
- Total parameters: **~118M** (~240MB in bf16), far below the 1B competition limit
- Pretraining and finetuning share the backbone; weight sharing ratio > 95%,
  satisfying the > 80% requirement (validated automatically at the finetune entry)

---

## 4. Training Workflow

### 4.0 Data Preparation (one-time, before any training)

The raw official chunks live under
`datasets/01|02|03/rf_signal_benchmark_train_v1.0/` (~2 GB each). Convert them
into the fast memmap training format first:

```bash
cd signal-processing

# Smoke-scale conversion (first 3,000 samples per task, minutes)
python utils/prepare_official.py --limit 3000

# Full conversion (1,179,000 rows -> ~9.7 GB memmap, streaming row-group scan)
python utils/prepare_official.py
```

This produces `datasets/{task}/iq.memap + meta.json` with a stratified three-way
split recorded in `meta.json`: **train/val/test = 80%/10%/10%** (stratified by
label for protocol/modulation, by comm/radar pattern for parameter). The val
split drives early stopping and best-weight export; the test split never
participates in any training-time decision and is only used for the unbiased
post-training evaluation (`--eval_test`, Section 4.3). All commands below
keep `--data_dir datasets` — the loader picks the memmap up automatically
(highest priority). Chunk discovery handles any extraction nesting depth,
rejects duplicated part files (same part extracted/uploaded twice), and
errors out unless exactly 3 distinct parts are found.

### 4.1 Smoke Test (run this before real training)

```bash
cd signal-processing

# Stage 1 smoke: only 2 steps per epoch, evaluation skipped
python code/training/pretrain_mae.py --data_dir datasets --out_dir weights --smoke_steps 2

# Stage 2 smoke (auto-initializes from scratch if no MAE weights exist)
python code/training/finetune_multitask.py --data_dir datasets --out_dir weights --smoke_steps 2
```

Pass criteria: both scripts exit without errors, checkpoint files appear under
`weights/`, and the log ends with `recon_mse=...` / `train_*_weighted_loss=...`.

### 4.1.1 Multi-Card Launch (single A800 GPU / multi-card 910C NPU)

Both stages run unchanged on a single card (plain `python`, as shown above).
On multiple cards launch with `torchrun` — the backend is picked automatically
(`nccl` on CUDA GPUs, `hccl` on Ascend NPUs):

```bash
# 8x Ascend 910C example (use --nproc_per_node=<number of cards>)
torchrun --nproc_per_node=8 code/training/pretrain_mae.py \
    --data_dir datasets --out_dir weights

torchrun --nproc_per_node=8 code/training/finetune_multitask.py \
    --data_dir datasets --out_dir weights --eval_test
```

How multi-card training works here:

- `training.batch_size` is the **per-card** batch; the effective batch is
  `batch_size × nproc_per_node` — scale `training.lr` accordingly if you change
  the card count
- Train splits are sharded with `DistributedSampler`; gradients are averaged
  across cards after every step (manual all-reduce, no DDP wrapper — the model
  has multiple custom forward entries and MAE leaves task heads unused)
- All ranks use the same seed, so weights stay bit-identical across cards
- Only rank 0 writes logs / TensorBoard / checkpoints / exports; the validation
  set is **sharded across all ranks** (each card evaluates its slice in
  parallel, metrics are all-reduced with sample-count weighting — exact for
  accuracy/precision), so epoch-end evaluation stays fast and symmetric even
  on 16 cards. The early-stopping decision is identical on every rank and the
  stop flag is broadcast so every rank exits in step

### 4.2 Stage 1: MAE Self-Supervised Pretraining (task-agnostic)

```bash
python code/training/pretrain_mae.py \
    --data_dir datasets \
    --out_dir weights \
    --config code/configs/default.json
```

- Uses the mixed training sets of all tasks with only iq/sr (no labels)
- A rolling checkpoint is **overwritten** to `weights/mae_resume.pt` at the end of
  every epoch (a single file, not one per epoch)
- Shared weights are exported once at the end to `weights/mae_pretrained.pt`
- Epoch cap `epochs_pretrain` (default 200); no validation split exists for MAE,
  so early stopping tracks the **training loss** (`recon_mse`): after
  `patience_pretrain` (default 20) epochs without a new low, training stops
  (counter round-trips through the resume checkpoint)

**Resume from interruption**:

```bash
python code/training/pretrain_mae.py --data_dir datasets --out_dir weights \
    --resume weights/mae_resume.pt
```

### 4.3 Stage 2: Single-Backbone Multi-Task Finetuning

```bash
# Automatically loads weights/mae_pretrained.pt as initialization
python code/training/finetune_multitask.py \
    --data_dir datasets \
    --out_dir weights
```

- Three tasks trained in rotation (protocol / modulation / parameter updated in
  turn within each step)
- Validation every epoch; proxy score = 0.3*protocol_acc + 0.3*modulation_acc
  + 0.4*parameter_precision
- Whenever the score reaches a new best, official inference artifacts are exported
  to `weights/`: `model.safetensors` + `config.json` + `param_stats.json`
  (again a single rolling best — not one file per epoch)
- Epoch cap `epochs_finetune` (default 200)
- Early stopping: stops after `patience` (default 10) epochs without improvement

**Specify MAE weights / resume**:

```bash
python code/training/finetune_multitask.py --data_dir datasets --out_dir weights \
    --from_mae weights/mae_pretrained.pt        # use a specific pretrained checkpoint
python code/training/finetune_multitask.py --data_dir datasets --out_dir weights \
    --resume weights/finetune_resume.pt         # resume (keeps weights, skips MAE reload)
python code/training/finetune_multitask.py --data_dir datasets --out_dir weights \
    --eval_test                                 # after fit(): unbiased eval on the
                                                # test split (loads exported best
                                                # weights; run it ONCE per model)
```

### 4.4 Grid Search and Batch Training (gen_configs + run_grid)

Sections 4.2/4.3 train **one** config by hand. The recommended sweep workflow is:
generate a grid of legal configs first, then run them all through the same
two-stage pipeline with one command.

**Step 1 — generate the grid** (`utils/gen_configs.py`, single output file
`code/configs/grid.json`):

```bash
python utils/gen_configs.py          # defaults, writes grid.json directly
python utils/gen_configs.py --dry    # preview only

# custom sweep axes (all accept multiple values; 'auto' allowed for bs/lr)
python utils/gen_configs.py \
    --d_model 512 768 1024 --n_layers 24 27 30 \
    --batch_size auto 128 --lr auto \
    --shared_ratio 0.85 0.90 0.95 \
    --max_configs 20                 # keep only the 20 largest by params
```

Every entry is a complete, directly-trainable config. The generator enforces the
competition red lines (**total params < 1B** via real model instantiation,
**shared ratio > 80%**, `d_model % nhead == 0` for RoPE) and a VRAM ceiling
(default 64 GB peak estimate) per config; illegal combos are skipped with a
printed reason. Head capacity is swept via the shared-ratio tiers — head widths
are solved so that shared ratio lands on 0.85/0.90/0.95 (names carry
`_shared85/90/95` suffixes).

**Step 2 — batch training** (`utils/run_grid.py`):

```bash
python utils/run_grid.py --dry        # list the pending configs
python utils/run_grid.py             # run all pending configs sequentially
python utils/run_grid.py --only 1024d_27l   # subset by name substring
python utils/run_grid.py --retry      # rerun configs that failed last time
```

Each grid entry runs `pretrain_mae.py` then `finetune_multitask.py`
(sequential, single GPU) with `--out_dir weights/<name>`. Per-config artifacts
are archived to `logs/grid_runs/<name>/` (config.json, full stage logs,
epoch/step metrics jsonl, plots). After each config finishes, the runner reads
its best `val_proxy_score` and manages weights globally:

- only the **single best-scoring set** in history is kept, in `weights/best/`
  (`model.safetensors` + `config.json` + `param_stats.json`, official inference
  format) plus `weights/best/best_config.json` (feed-back-to-training format)
- non-best configs have their weights deleted right after their run finishes
  (MAE checkpoints and resume files included) — pass `--keep_weights` to retain

Batch-level outputs: `logs/grid_runs/best.json` (best config summary),
`state.json` (progress; reruns skip done configs automatically),
`summary.jsonl` (one line per config), and comparison plots
`logs/grid_runs/plots/proxy_curves.png` + `best_bar.png`.

**Log isolation note**: each training process still writes the shared
`logs/{pretrain,finetune}_metrics.jsonl` and `logs/plots/` (they are reset at
the start of a fresh run and archived per config immediately after it
finishes). Do **not** start a manual training run while a batch is in
progress, or the two would interleave in those shared files. TensorBoard event
files under `logs/tb/` accumulate across runs (never reset) — use the archived
per-config jsonl/plots as the source of truth after a batch.

### 4.5 Inference (official format)

```bash
python code/infer.py \
    --input datasets/01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet \
    --output out/pred.jsonl \
    --weights_dir weights \
    --batch_size 32
```

Output JSONL has one `{task, sample_id, prediction}` per line, strictly preserving
input row order; parameter samples are hard-masked by the pattern head
(non-applicable parameters are set to 0.0).

---

## 5. Key Tuning Suggestions

| Symptom | Fix |
|---|---|
| Out of memory (OOM) | Halve `training.batch_size` (32→16→8); then reduce `model.ffn_hidden` (2048→1536) or `model.d_model` (512→384) |
| Want a larger model | Raise `model.d_model` (512→768) and `model.ffn_hidden` (keep ~4x relation); still far below 1B |
| Pretraining loss drops too slowly | Increase `epochs_pretrain` or `training.lr` (1e-4 → 2e-4) |
| Persistent grad-norm spikes | Reduce `training.grad_clip` (1.0→0.5) or `training.lr` |
| Finetuning overfits | Increase `training.weight_decay` (0.05→0.1) or reduce `epochs_finetune` |
| Slow data loading | Set `training.num_workers` to the CPU core count (e.g. 8) |
| Old GPU without bf16 | Set `training.amp` to `false` (falls back to fp32) |
| Want a different layer layout | Edit the `model.layer_types` list (e.g. adding mamba: a short list like `["full_attention","mla","mamba"]` is tiled cyclically to fill n_layers) |

**Viewing training metrics** (four visualizations in parallel):

```bash
# 1) Terminal progress bar: shown automatically by tqdm during training

# 2) Per-epoch jsonl metrics (one line per epoch)
tail -f logs/pretrain_metrics.jsonl    # recon_mse / grad_norm / lr
tail -f logs/finetune_metrics.jsonl    # train_*_loss / val_* / val_proxy_score
tail -f logs/pretrain_steps.jsonl      # per-step records (step / epoch / loss ...)

# 3) TensorBoard curves (step-level loss / grad norm / lr + epoch-level val metrics)
tensorboard --logdir logs/tb --port 6006
# open http://<training-machine-IP>:6006 in a browser

# 4) Automatic plots after training finishes: PNGs under logs/plots/
#    - <stage>_<group>.png       per-epoch curves (loss / grad_norm / lr / val)
#    - <stage>_step_<group>.png  per-step curves, ALL epochs on one figure,
#      with dashed vertical lines marking epoch boundaries; the loss figure
#      also overlays val metrics ("x" markers on a secondary right axis,
#      placed at each epoch's last step) so train and val share one chart
ls logs/plots/
```

TensorBoard scalar reference:

| Tag | Level | Meaning |
|---|---|---|
| `train/loss`, `train/loss_{task}` | step | training loss (per-task curves when finetuning) |
| `train/grad_norm` | step | global grad norm before clipping (spikes = instability, persistently small = converged) |
| `train/lr` | step | learning rate (warmup + cosine) |
| `epoch/recon_mse`, `epoch/train_*_loss` | epoch | per-epoch averages |
| `epoch/val_*`, `epoch/val_proxy_score` | epoch | validation metrics (finetuning) |

---

## 6. Competition Constraint Self-Checks

The finetune entry validates and prints these automatically, and errors out on
violation:

- Total parameters < 1B (`MAX_TOTAL_PARAMS`)
- Cross-task weight sharing ratio > 80% (`MIN_SHARED_RATIO`)

---

## 7. FAQ

1. **Error `no training data under ...`**: run `python utils/prepare_official.py`
   first (Section 4.0) or check `--data_dir`; official data requires the memmap
   under `datasets/{task}/`, the parquet fallback only covers simulated data.
2. **Error about unknown layer type**: `layer_types` only accepts `full_attention` /
   `mla` / `gda` / `mamba` (legacy `attn` is also accepted and normalized).
3. **Hybrid backbone error asking for `pos_encoding='rope'`**: by design — MLA /
   attention layers rely on rotary positions; set `model.pos_encoding` to `rope`.
4. **Learning rate looks odd after resuming**: the checkpoint contains scheduler
   state, so the schedule resumes exactly where it left off — expected behavior.
