"""Unsloth SFT for platform-canonical chat datasets with structured tool calls.

The entry point downloads the requested dataset/model from the platform
registries (the data ingestion module has already restructured ShareGPT-style
inputs into the canonical ``messages`` format), trains the selected thinking
adapter, saves a checkpoint after every epoch, and runs the test split before
training and after every saved epoch. The merged final model is pushed back to
the model registry.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from dataset_utils import DatasetFormatError, final_inference_case, normalize_example, without_reasoning


# The platform already provides the model/tokenizer locally.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value in (None, "", "none", "null") else int(value)


CONFIG: dict[str, Any] = {
    # Platform access
    "token": os.environ.get("BEARER_TOKEN"),
    "clearml_api_host": os.getenv("CLEARML_API_HOST"),
    "user_management_api": os.getenv("USER_MANAGEMENT_API"),
    "s3_endpoint_url": os.getenv("CEPH_ENDPOINT_URL"),
    # Change these names/paths for your own assets. Local paths skip downloading.
    "dataset_name": os.environ.get("DATASET_NAME", "CHANGE_ME_DATASET"),
    # By default the dataset is pulled from the platform data registry; set
    # DATASET_DIR to a local folder to skip the download instead.
    "dataset_dir": os.environ.get("DATASET_DIR"),
    "train_file": os.environ.get("TRAIN_FILE", "dadmatools_trainset_new.json"),
    "test_file": os.environ.get("TEST_FILE", "dadmatools_testset_new.json"),
    "base_model_name": os.environ.get("BASE_MODEL_NAME", "CHANGE_ME_MODEL"),
    "base_model_dir": os.environ.get("BASE_MODEL_DIR"),
    "save_model_name": os.environ.get("SAVE_MODEL_NAME", "dadmatools_toolcall_lora"),
    # Requested training settings (optimizer defaults follow the platform doc)
    "epochs": float(os.environ.get("EPOCHS", "1")),
    "batch_size": int(os.environ.get("BATCH_SIZE", "2")),
    "grad_accum": int(os.environ.get("GRAD_ACCUM", "4")),
    "optim": os.environ.get("OPTIM", "adamw_8bit"),
    "learning_rate": 1e-4,  # 10e-5
    "lora_rank": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "extra_unfrozen_mode": os.environ.get("EXTRA_UNFROZEN_MODE", "first2_last"),
    "warmup_epochs": 0.1,
    # "true" trains a thinking adapter, "false" a non-thinking adapter, and
    # "auto" inspects the dataset + chat template and picks a single mode.
    "thinking_mode": os.environ.get("THINKING_MODES", "auto"),
    # Data/inference controls (not optimizer hyperparameters)
    "system_prompt": os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant."),
    "max_seq_length": int(os.environ.get("MAX_SEQ_LENGTH", "2048")),
    "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "512")),
    "max_test_samples": _optional_int("MAX_TEST_SAMPLES"),  # None = the full test file
    "output_dir": os.environ.get("OUTPUT_DIR", "./outputs"),
    "seed": int(os.environ.get("SEED", "3407")),
    # Full trainable decoder blocks are incompatible with QLoRA's quantized weights.
    "load_in_4bit": _env_bool("LOAD_IN_4BIT", False),
}


def _resolve_explicit_file(dataset_root: Path, value: str | None, label: str) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = dataset_root / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {candidate}")
    return candidate.resolve()


def _auto_find_split(dataset_root: Path, split: str) -> Path:
    suffixes = {".json", ".jsonl"}
    exact_stems = {split, f"{split}_data", f"{split}-data", f"data_{split}"}
    files = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    exact = [path for path in files if path.stem.lower() in exact_stems]
    fuzzy = [path for path in files if split in path.stem.lower()]
    matches = exact or fuzzy
    if len(matches) != 1:
        choices = ", ".join(str(path.relative_to(dataset_root)) for path in matches or files)
        raise FileNotFoundError(
            f"Could not select exactly one {split} file under {dataset_root}. "
            f"Set {split.upper()}_FILE explicitly. Candidates: {choices or 'none'}"
        )
    return matches[0].resolve()


def resolve_data_files(dataset_root: Path, cfg: dict[str, Any]) -> tuple[Path, Path]:
    train_file = _resolve_explicit_file(dataset_root, cfg.get("train_file"), "train")
    test_file = _resolve_explicit_file(dataset_root, cfg.get("test_file"), "test")
    return train_file or _auto_find_split(dataset_root, "train"), test_file or _auto_find_split(
        dataset_root, "test"
    )


def find_transformer_layers(model: Any, mode: str) -> list[str]:
    """Find the decoder ModuleList and select its first two and optional last block."""
    import torch

    if mode in {"", "none", "off", "0"}:
        return []
    if mode not in {"first2", "first2_last"}:
        raise ValueError("EXTRA_UNFROZEN_MODE must be first2_last, first2, or none")

    candidates: list[tuple[int, int, str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) < 2:
            continue
        leaf = name.rsplit(".", 1)[-1]
        preferred = 1 if leaf in {"layers", "h", "blocks", "block"} else 0
        if all(any(True for _ in child.parameters(recurse=True)) for child in module[:2]):
            candidates.append((preferred, len(module), name, module))
    if not candidates:
        raise RuntimeError(
            "Could not locate the transformer's decoder layer stack; set "
            "EXTRA_UNFROZEN_MODE=none or adapt find_transformer_layers for this architecture."
        )

    _, count, name, _ = max(candidates, key=lambda item: (item[0], item[1]))
    indices = [0, 1]
    if mode == "first2_last" and count - 1 not in indices:
        indices.append(count - 1)
    return [f"{name}.{index}" for index in indices]


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def _template_supports_enable_thinking(tokenizer: Any) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    serialized = json.dumps(template, ensure_ascii=False) if isinstance(template, dict) else str(template)
    return "enable_thinking" in serialized


def _apply_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    if _template_supports_enable_thinking(tokenizer):
        kwargs["enable_thinking"] = enable_thinking
    elif enable_thinking:
        raise RuntimeError(
            "The selected tokenizer chat template does not reference "
            "enable_thinking, so a thinking run is impossible with this base "
            "model. Use an enable_thinking-compatible model (e.g. Qwen3) or "
            "set THINKING_MODES=false."
        )
    return _as_token_ids(tokenizer.apply_chat_template(messages, **kwargs))


def build_training_dataset(
    raw_dataset: Any,
    tokenizer: Any,
    cfg: dict[str, Any],
    enable_thinking: bool,
) -> Any:
    """Create one supervised sample for every assistant turn in each conversation."""
    from datasets import Dataset

    features: list[dict[str, list[int]]] = []
    truncated = 0
    skipped = 0
    for row_index, row in enumerate(raw_dataset):
        try:
            messages, tools = normalize_example(dict(row), cfg["system_prompt"])
        except DatasetFormatError as exc:
            raise DatasetFormatError(f"train row {row_index}: {exc}") from exc

        for target_index, message in enumerate(messages):
            if message["role"] != "assistant" or target_index == 0:
                continue
            history = messages[:target_index]
            target = message if enable_thinking else without_reasoning(message)
            prompt_ids = _apply_template(
                tokenizer,
                history,
                tools,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            full_ids = _apply_template(
                tokenizer,
                history + [target],
                tools,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            common = 0
            for prompt_token, full_token in zip(prompt_ids, full_ids):
                if prompt_token != full_token:
                    break
                common += 1
            if common != len(prompt_ids):
                raise RuntimeError(
                    f"train row {row_index}, assistant turn {target_index}: the chat template's "
                    "generation prompt is not a prefix of the completed conversation. "
                    "Use the tokenizer supplied with the base model."
                )

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            max_length = int(cfg["max_seq_length"])
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
                labels = labels[:max_length]
                truncated += 1
            if not any(label != -100 for label in labels):
                skipped += 1
                continue
            features.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )

    if not features:
        raise RuntimeError("No trainable assistant responses remained after formatting/truncation")
    print(
        f"[dataset] enable_thinking={enable_thinking}: {len(features)} assistant turns; "
        f"truncated={truncated}; skipped={skipped}"
    )
    return Dataset.from_list(features)


def build_test_cases(raw_dataset: Any, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    limit = cfg.get("max_test_samples")
    cases: list[dict[str, Any]] = []
    for row_index, row in enumerate(raw_dataset):
        if limit is not None and len(cases) >= limit:
            break
        try:
            history, expected, tools = final_inference_case(dict(row), cfg["system_prompt"])
        except DatasetFormatError as exc:
            raise DatasetFormatError(f"test row {row_index}: {exc}") from exc
        cases.append(
            {"row_index": row_index, "messages": history, "expected": expected, "tools": tools}
        )
    if not cases:
        raise RuntimeError("The test file is empty")
    return cases


def dataset_has_reasoning(raw_dataset: Any) -> bool:
    """True if any assistant turn carries reasoning (fields or <think> tags)."""
    for row in raw_dataset:
        messages, _ = normalize_example(dict(row), None)
        for message in messages:
            if message["role"] != "assistant":
                continue
            if message.get("reasoning_content"):
                return True
            if "<think>" in message.get("content", ""):
                return True
    return False


def resolve_thinking_modes(setting: str, dataset_has_reasoning_flag: bool) -> list[bool]:
    """Map the THINKING_MODES env value to the list of runs to perform."""
    normalized = str(setting).strip().lower()
    if normalized in {"both", "true,false", "2"}:
        return [True, False]
    if normalized in {"true", "thinking", "1"}:
        return [True]
    if normalized in {"false", "none", "off", "0"}:
        return [False]
    if normalized in {"auto", ""}:
        if dataset_has_reasoning_flag:
            print(
                "[thinking] auto: dataset contains reasoning traces -> "
                "single enable_thinking=True run"
            )
            return [True]
        print(
            "[thinking] auto: dataset contains no reasoning traces -> "
            "single enable_thinking=False run (set THINKING_MODES=both to force both)"
        )
        return [False]
    raise ValueError("THINKING_MODES must be one of: auto, true, false, both")


def _model_device(model: Any) -> Any:
    for parameter in model.parameters():
        if getattr(parameter, "device", None) is not None and parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine model device")


def run_inference(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    output_file: Path,
    enable_thinking: bool,
    cfg: dict[str, Any],
    checkpoint_label: str,
) -> None:
    """Generate and persist JSONL predictions while preserving the training RNG state."""
    import torch
    from unsloth import FastLanguageModel

    output_file.parent.mkdir(parents=True, exist_ok=True)
    python_rng = random.getstate()
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    device = _model_device(model)

    try:
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg["seed"])
        FastLanguageModel.for_inference(model)
        model.eval()
        with output_file.open("w", encoding="utf-8") as handle, torch.inference_mode():
            for case in cases:
                prompt_ids = _apply_template(
                    tokenizer,
                    case["messages"],
                    case["tools"],
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                generation_kwargs = {
                    "max_new_tokens": cfg["max_new_tokens"],
                    "do_sample": True,
                    "temperature": 0.6 if enable_thinking else 0.7,
                    "top_p": 0.95 if enable_thinking else 0.8,
                    "top_k": 20,
                    "pad_token_id": (
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                }
                generated = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs
                )
                new_tokens = generated[0, input_ids.shape[1] :]
                record = {
                    "row_index": case["row_index"],
                    "checkpoint": checkpoint_label,
                    "enable_thinking": enable_thinking,
                    "messages": case["messages"],
                    "tools": case["tools"],
                    "expected": case["expected"],
                    "generated_text": tokenizer.decode(new_tokens, skip_special_tokens=True),
                    "generation": generation_kwargs,
                }
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        print(f"[inference] wrote {len(cases)} predictions to {output_file}")
    finally:
        random.setstate(python_rng)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if was_training:
            FastLanguageModel.for_training(model)
            model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _sft_config(cfg: dict[str, Any], run_dir: Path) -> Any:
    from trl import SFTConfig

    # 0.1 epoch of warmup across an epochs-epoch run = warmup_epochs/epochs of all steps.
    parameters = inspect.signature(SFTConfig.__init__).parameters
    if cfg["epochs"] <= 0:
        raise ValueError("epochs must be positive")
    warmup_fraction = cfg["warmup_epochs"] / cfg["epochs"]
    kwargs: dict[str, Any] = {
        "output_dir": str(run_dir),
        "num_train_epochs": cfg["epochs"],
        "learning_rate": cfg["learning_rate"],
        "per_device_train_batch_size": cfg["batch_size"],
        "gradient_accumulation_steps": cfg["grad_accum"],
        "optim": cfg["optim"],
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "epoch",
        # Keep only the last checkpoint and skip optimizer/scheduler state:
        # fully unfrozen decoder blocks make optimizer states several GB each.
        "save_only_model": True,
        "save_total_limit": 1,
        "report_to": "none",
        "seed": cfg["seed"],
        "data_seed": cfg["seed"],
    }
    if "warmup_ratio" in parameters:
        kwargs["warmup_ratio"] = warmup_fraction
    else:
        # New Transformers versions accept a float <1 as a ratio in warmup_steps.
        kwargs["warmup_steps"] = warmup_fraction
    if "max_length" in parameters:
        kwargs["max_length"] = cfg["max_seq_length"]
    elif "max_seq_length" in parameters:
        kwargs["max_seq_length"] = cfg["max_seq_length"]
    if "dataset_kwargs" in parameters:
        kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    return SFTConfig(**kwargs)


def train_variant(
    cfg: dict[str, Any],
    raw_train: Any,
    test_cases: list[dict[str, Any]],
    base_model: str,
    enable_thinking: bool,
) -> Path:
    import torch
    from unsloth import FastLanguageModel
    from transformers import DataCollatorForSeq2Seq, TrainerCallback
    from trl import SFTTrainer

    label = "thinking_true" if enable_thinking else "thinking_false"
    run_dir = Path(cfg["output_dir"]) / label
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    modules_to_save = find_transformer_layers(model, cfg["extra_unfrozen_mode"])
    if modules_to_save and cfg["load_in_4bit"]:
        raise RuntimeError(
            "Full extra layers cannot be trained from 4-bit quantized base weights. "
            "Keep LOAD_IN_4BIT=0, or set EXTRA_UNFROZEN_MODE=none for pure QLoRA."
        )
    peft_kwargs = {
        "r": cfg["lora_rank"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "target_modules": cfg["lora_target_modules"],
        "use_gradient_checkpointing": "unsloth",
        "random_state": cfg["seed"],
    }
    if modules_to_save:
        peft_kwargs["modules_to_save"] = modules_to_save
    model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

    selection = {
        "enable_thinking": enable_thinking,
        "modules_to_save": modules_to_save,
        "lora_target_modules": cfg["lora_target_modules"],
        "load_in_4bit": cfg["load_in_4bit"],
    }
    (log_dir / "trainable_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_dataset = build_training_dataset(raw_train, tokenizer, cfg, enable_thinking)

    before_file = run_dir / "inference" / "before_training.jsonl"
    run_inference(
        model,
        tokenizer,
        test_cases,
        before_file,
        enable_thinking,
        cfg,
        "before_training",
    )

    class PersistentLossCallback(TrainerCallback):
        def __init__(self) -> None:
            self.jsonl = log_dir / "loss.jsonl"
            self.csv_file = log_dir / "loss.csv"
            self.jsonl.write_text("", encoding="utf-8")
            with self.csv_file.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    ["timestamp_utc", "step", "epoch", "loss", "learning_rate", "grad_norm"]
                )

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
            if not state.is_world_process_zero or not logs or "loss" not in logs:
                return
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "step": state.global_step,
                "epoch": state.epoch,
                **dict(logs),
            }
            with self.jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            with self.csv_file.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [
                        record["timestamp_utc"],
                        record["step"],
                        record["epoch"],
                        record.get("loss"),
                        record.get("learning_rate"),
                        record.get("grad_norm"),
                    ]
                )

    class CheckpointInferenceCallback(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if not state.is_world_process_zero:
                return
            epoch = max(1, int(math.ceil(float(state.epoch or 0))))
            checkpoint = f"checkpoint-{state.global_step}"
            output_file = run_dir / "inference" / f"after_epoch_{epoch}_{checkpoint}.jsonl"
            run_inference(
                kwargs.get("model") or model,
                tokenizer,
                test_cases,
                output_file,
                enable_thinking,
                cfg,
                checkpoint,
            )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "args": _sft_config(cfg, run_dir),
        "data_collator": DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100
        ),
        "callbacks": [PersistentLossCallback(), CheckpointInferenceCallback()],
    }
    trainer_parameters = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_state()

    final_adapter_dir = run_dir / "final_adapter"
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # Merge LoRA + the fully trained modules_to_save decoder blocks into a
    # standalone 16-bit model so the registry artifact is directly servable
    # and re-usable as a base model (its config.json is verified below).
    final_model_dir = run_dir / "final_model"
    model.save_pretrained_merged(str(final_model_dir), tokenizer, save_method="merged_16bit")
    if not (final_model_dir / "config.json").is_file():
        raise RuntimeError(
            f"merged save did not produce {final_model_dir / 'config.json'}; "
            "refusing to register a non-self-contained artifact"
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_dir


def main(config_overrides: dict[str, Any] | None = None) -> None:
    import gc

    from aipmodel.model_registry import MLOpsManager
    from data.sdk.download_sdk import s3_download
    from datasets import load_dataset

    cfg = dict(CONFIG)
    if config_overrides:
        cfg.update(config_overrides)
    cfg["epochs"] = float(cfg["epochs"])
    if cfg["extra_unfrozen_mode"] != "none" and cfg["load_in_4bit"]:
        raise ValueError("LOAD_IN_4BIT must be 0 when extra decoder layers are unfrozen")
    if cfg["epochs"] <= 0:
        raise ValueError("epochs must be positive")
    if not cfg.get("token"):
        raise ValueError("Set BEARER_TOKEN in .env before starting training")

    manager = MLOpsManager(user_token=cfg["token"])

    if cfg.get("dataset_dir"):
        dataset_root = Path(cfg["dataset_dir"]).expanduser().resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"DATASET_DIR does not exist: {dataset_root}")
    else:
        if cfg["dataset_name"] == "CHANGE_ME_DATASET":
            raise ValueError("Set DATASET_NAME (or DATASET_DIR) before starting training")
        dataset_root = Path("./dataset").resolve()
        s3_download(
            dataset_name=cfg["dataset_name"],
            absolute_path=str(dataset_root),
            token=cfg["token"],
            clearml_api_host=cfg["clearml_api_host"],
            user_management_url=cfg["user_management_api"],
            s3_endpoint_url=cfg["s3_endpoint_url"],
        )
    train_file, test_file = resolve_data_files(dataset_root, cfg)
    print(f"[dataset] train={train_file}")
    print(f"[dataset] test={test_file}")

    if cfg.get("base_model_dir"):
        base_model = str(Path(cfg["base_model_dir"]).expanduser().resolve())
    else:
        if cfg["base_model_name"] == "CHANGE_ME_MODEL":
            raise ValueError("Set BASE_MODEL_NAME (or BASE_MODEL_DIR) before starting training")
        manager.get_model(model_name=cfg["base_model_name"], local_dest="./base_model")
        configs = list(Path("./base_model").rglob("config.json"))
        if len(configs) != 1:
            raise FileNotFoundError(
                f"Expected one model config under ./base_model, found {len(configs)}"
            )
        base_model = str(configs[0].parent.resolve())

    raw_train = load_dataset("json", data_files=str(train_file), split="train")
    raw_test = load_dataset("json", data_files=str(test_file), split="train")
    test_cases = build_test_cases(raw_test, cfg)
    thinking_modes = resolve_thinking_modes(cfg["thinking_mode"], dataset_has_reasoning(raw_train))

    registered: dict[str, Any] = {}
    for enable_thinking in thinking_modes:
        run_dir = train_variant(cfg, raw_train, test_cases, base_model, enable_thinking)
        suffix = "thinking_true" if enable_thinking else "thinking_false"
        model_name = f"{cfg['save_model_name']}_{suffix}"
        model_id = manager.add_model(
            source_type="local",
            source_path=str(run_dir / "final_model"),
            model_name=model_name,
        )
        registered[suffix] = {"model_name": model_name, "model_id": model_id}
        print(f"[registry] {suffix}: model_id={model_id}")
        # Ensure the first run's model/optimizer cycles are gone before a second run.
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    summary_file = Path(cfg["output_dir"]) / "registered_models.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(registered, indent=2, default=str), encoding="utf-8")
    print(f"done: {json.dumps(registered, default=str)}")


if __name__ == "__main__":
    main()
