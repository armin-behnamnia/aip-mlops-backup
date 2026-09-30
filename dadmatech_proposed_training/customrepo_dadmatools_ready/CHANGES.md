# Review Corrections — dadmatech_proposed_training

Corrections applied to the DadmaTech-proposed training repo after a platform
compatibility review against the `sample_training` template, the platform API
doc, and the client's requirement document. Each change lists the file(s)
touched and the reason.

## Blockers fixed

### 1. Removed a live bearer token committed as `.env`
- **Removed:** `.env` (contained a real, expired platform JWT).
- **Why:** secrets must never be pushed to Git. The JWT had already expired
  (2026-08-29), so it was both a leak and a guaranteed failure.
- **Action for operator:** supply a fresh `BEARER_TOKEN` as a runtime
  environment variable (or an untracked `.env` created after clone).

### 2. Removed the 241 MB bundled dataset
- **Removed:** `dataset/` (`dadmatools_trainset_new.json`,
  `dadmatools_testset_new.json`, `SHA256SUMS.txt`; ~241 MB total).
- **Why:** the platform clones the repo before every training run; a bundled
  dataset inflates clone time and may exceed Git size limits. The data already
  lives in the platform data registry.
- **Code change:** `DATASET_DIR` now defaults to **unset** in `train.py`, so the
  dataset is downloaded from the registry via `s3_download` using `DATASET_NAME`
  (same as the platform template). Setting `DATASET_DIR` still skips the
  download for local runs. `TRAIN_FILE`/`TEST_FILE` auto-detection is unchanged.

### 3. Registered artifact is now a self-contained model
- **File:** `train.py` (`train_variant`, `main`).
- **Was:** `manager.add_model` was pointed at the whole run directory
  (checkpoints, logs, inference JSONL, adapter only) — no `config.json` at its
  root, so the registered model could not be re-downloaded by this very repo's
  `get_model` logic (which requires exactly one `config.json`), nor served
  standalone.
- **Now:** after training, the LoRA adapter **and** the fully trained
  `modules_to_save` decoder blocks are merged into a standalone 16-bit model at
  `outputs/<run>/final_model/` via `save_pretrained_merged`. The code asserts
  `final_model/config.json` exists (fails loudly otherwise) and registers
  `final_model/` only. `final_adapter/` is still written to disk for local use
  but is not registered.

### 4. Collapsed the two thinking runs into an auto-selected single run
- **File:** `train.py` (CONFIG, `resolve_thinking_modes`, `dataset_has_reasoning`,
  `_apply_template`).
- **Was:** always trained `thinking_true` and `thinking_false` adapters. The
  client's dataset contains **zero** reasoning traces (verified: 0
  `reasoning_content` fields and 0 thinking tags across all 17,797 train rows),
  so both runs would train on essentially identical targets — double GPU time,
  double registry uploads, for no benefit.
- **Now:** `THINKING_MODES=auto` (default) inspects the dataset: a single
  non-thinking run when no reasoning exists, a single thinking run when it
  does. `THINKING_MODES=true|false|both` forces a mode (including the original
  two-run behavior).
- **Template check:** the old `STRICT_THINKING_TEMPLATE` env var is removed.
  `enable_thinking` is now passed to the chat template only when the template
  supports it; a thinking run on a non-supporting template (e.g. gpt-oss)
  fails immediately with an explicit error, and a non-thinking run simply
  omits the kwarg. This replaces the previous all-or-nothing strict check and
  removes the `.env` ↔ code mismatch that would have crashed run 2 on gpt-oss.

## Warnings fixed

### 5. Optimizer/training defaults realigned with the platform doc
- **File:** `train.py` (CONFIG, `_sft_config`).
- **Was:** `epochs=3`, and batch size / gradient accumulation / optimizer left
  at TRL defaults (8/1/fp32 AdamW) — deviating from the requirement doc's
  "defaults from the document, except the listed changes".
- **Now:** `epochs=1`, `per_device_train_batch_size=2`,
  `gradient_accumulation_steps=4`, `optim="adamw_8bit"` (same effective batch 8,
  lower memory). All overridable via `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM`,
  `OPTIM` env vars and via the ClearML hyperparameters entry point
  (`train_hyperparameter.py`, which now also connects/casts `batch_size` and
  `grad_accum`).

### 6. Checkpoint bloat removed
- **File:** `train.py` (`_sft_config`).
- **Was:** `save_strategy="epoch"` with full state. With ~1.7B fully trainable
  parameters (unfrozen blocks), optimizer/scheduler states are ~14 GB per
  checkpoint; 3 epochs x 2 modes would have produced 80+ GB per run.
- **Now:** `save_only_model=True` (no optimizer/scheduler state) and
  `save_total_limit=1` (only the latest epoch checkpoint is kept). The
  per-epoch inference callback still fires on every save.

### 7. Dataset parsed once, shared across runs
- **File:** `train.py` (`main`, `train_variant`).
- **Was:** each `train_variant` call re-loaded and re-validated the train and
  test JSON files.
- **Now:** `main` loads both splits and builds test cases once; `train_variant`
  receives them. (Each thinking mode still gets its own tokenization, which is
  mode-dependent by design.)

### 8. transformers floor raised for gpt-oss compatibility
- **File:** `requirements.txt`: `transformers>=4.51,<5` → `>=4.52,<5`
  (gpt-oss architectures require 4.52+; harmless for Qwen-style bases).

### 9. `.env.example` rewritten
- **File:** `.env.example`.
- Removed `STRICT_THINKING_TEMPLATE`; `DATASET_DIR` now commented out
  (registry download is the default); documented `THINKING_MODES`, `EPOCHS`,
  `BATCH_SIZE`, `GRAD_ACCUM`, `OPTIM`; recommends setting `MAX_TEST_SAMPLES`.

### 10. readme updated
- **File:** `readme.md` — rewritten to describe the new defaults, the
  auto-thinking selection, the registry-based dataset, the `final_model`
  registration behavior, and the checkpoint policy. Persian, matching the
  template's documentation language.

### 11. ShareGPT support removed (platform ingestion owns reformatting)
- **Files:** `dataset_utils.py`, `tests/test_dataset_utils.py`,
  `examples/sharegpt_toolcall_sample.jsonl` (deleted), `train.py` (docstring),
  `train_hyperparameter.py` (task name), `readme.md`, this file.
- **Why:** the platform's data ingestion module already restructures
  ShareGPT/DADMAtools-style inputs before they are registered, so the training
  code only needs the canonical format. This shrinks the normalizer and removes
  a large untested conversion surface.
- **What changed:**
  - `dataset_utils.py` accepts only canonical `messages` rows (plus the Alpaca
    fallback). The `conversations` key and ShareGPT roles (`human`, `gpt`,
    `chatgpt`, `bot`, `observation`) and `from`/`value` field handling are
    removed; `ROLE_MAP` keeps `user`, `assistant`, `system`, `tool`,
    `function`, plus a defensive `function_call` role (see below).
    Unsupported roles/schemas now raise `DatasetFormatError` with a message
    pointing to the data ingestion module.
  - Defensive fallbacks for ingestion output variations (both verified against
    all 20,340 real DADMAtools rows converted two ways, 0 invalid, 77,010 tool
    calls preserved):
    - a message with `role: "function_call"` and the call JSON in `content`
      (single or parallel array) is converted to an assistant message with
      structured `tool_calls`;
    - a `tool_calls` element that is a JSON string holding one call object or
      an array of parallel calls is parsed and flattened.
  - Kept: `tools`/`functions` normalization, structured `tool_calls` /
    `function_call` conversion, `reasoning_content` handling, Alpaca fallback,
    `without_reasoning`, `final_inference_case`.
  - Tests rewritten for the canonical format (including rejection of ShareGPT
    `conversations` rows, the defensive `function_call` paths, and the Alpaca
    fallback).
  - `examples/sharegpt_toolcall_sample.jsonl` deleted (ShareGPT sample no
    longer relevant).

## Verification performed after the edits (no GPU runtime available)

- `python3 -m py_compile train.py train_hyperparameter.py dataset_utils.py` — clean.
- Existing unit tests (`tests/test_dataset_utils.py`): 5/5 pass.
- New unit tests (`tests/test_review_fixes.py`) cover every new/changed pure
  function: template-capability detection, `_apply_template` kwarg behavior
  (including the hard error for unsupported thinking runs), reasoning
  detection, `resolve_thinking_modes`, default CONFIG alignment with the doc,
  `_sft_config` fields (when `trl` is importable), and `find_transformer_layers`
  selection (when `torch` is importable). All pure-logic tests pass locally;
  GPU-dependent paths are untested.
- The removed `dataset/` files and `.env` were backed up to
  `/tmp/opencode/dadmatech_backup/` (outside the repo) before deletion.

## Known remaining constraints (documented, not changed)

- Full-layer unfreezing requires `LOAD_IN_4BIT=0` (kept as default); a 20B-class
  base model at 16-bit with fp32 master weights needs an ~80 GB GPU.
- `save_pretrained_merged` on a model with `modules_to_save` blocks is the
  assumed unsloth/PEFT behavior (merge restores trained modules into the base
  weights). The code verifies `config.json` exists post-merge but the merge
  itself could not be exercised without a GPU — watch the first run's logs.
