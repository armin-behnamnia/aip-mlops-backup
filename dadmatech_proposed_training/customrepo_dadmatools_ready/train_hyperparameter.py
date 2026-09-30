"""Optional ClearML-hyperparameter entry point for the shared training pipeline."""

from __future__ import annotations

from clearml import Task

from train import CONFIG, main


def run() -> None:
    task = Task.init(project_name="custom-repo", task_name="DADMAtools tool-call LoRA")
    hparams = {
        "dataset_name": CONFIG["dataset_name"],
        "base_model_name": CONFIG["base_model_name"],
        "save_model_name": CONFIG["save_model_name"],
        "epochs": CONFIG["epochs"],
        "batch_size": CONFIG["batch_size"],
        "grad_accum": CONFIG["grad_accum"],
        "learning_rate": CONFIG["learning_rate"],
        "lora_rank": CONFIG["lora_rank"],
        "lora_alpha": CONFIG["lora_alpha"],
        "lora_dropout": CONFIG["lora_dropout"],
        "warmup_epochs": CONFIG["warmup_epochs"],
        "extra_unfrozen_mode": CONFIG["extra_unfrozen_mode"],
    }
    task.connect(hparams)
    overrides = {
        **hparams,
        "epochs": float(hparams["epochs"]),
        "batch_size": int(hparams["batch_size"]),
        "grad_accum": int(hparams["grad_accum"]),
        "learning_rate": float(hparams["learning_rate"]),
        "lora_rank": int(hparams["lora_rank"]),
        "lora_alpha": int(hparams["lora_alpha"]),
        "lora_dropout": float(hparams["lora_dropout"]),
        "warmup_epochs": float(hparams["warmup_epochs"]),
    }
    main(overrides)


if __name__ == "__main__":
    run()
