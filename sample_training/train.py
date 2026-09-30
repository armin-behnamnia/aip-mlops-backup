"""
نمونه آموزش سفارشی روی پلتفرم MLOps — فاین‌تیون مدل زبانی با unsloth.

این اسکریپت چهار گام پایه را نشان می‌دهد:
  ۱) احراز هویت            -> MLOpsManager
  ۲) دریافت دیتاست         -> s3_download
  ۳) دریافت مدل پایه        -> manager.get_model
  ۴) آموزش، سپس ذخیرهٔ مدل  -> manager.add_model

توضیح کامل هر گام در README.md آمده است.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Force Hugging Face offline BEFORE importing unsloth/transformers — otherwise
# they reach out to huggingface.co (revision HEAD checks / unsloth helper repos)
# and retry with backoff. The model + tokenizer are already local.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ---- platform SDKs -----------------------------------------------------------
from aipmodel.model_registry import MLOpsManager   # model registry (pull/push)
from data.sdk.download_sdk import s3_download       # dataset download

load_dotenv()

CONFIG = {
    # توکن شما — از «عملیات یادگیری ماشین > کلید اپلیکیشن» بردارید (README)
    "token":               os.environ.get("BEARER_TOKEN"),

    # این سه مورد توسط پلتفرم فراهم می‌شوند؛ نیازی به تنظیم آن‌ها ندارید
    "clearml_api_host":    os.getenv("CLEARML_API_HOST"),
    "user_management_api": os.getenv("USER_MANAGEMENT_API"),
    "s3_endpoint_url":     os.getenv("CEPH_ENDPOINT_URL"),

    # چه چیزی گرفته/ذخیره شود
    "dataset_name":        "alpaca-mini-6",
    "base_model_name":     "smoltest02",
    "save_model_name":     "smoltest02_custom_repo",

    # پارامترهای آموزش
    "system_prompt":  "You are a helpful assistant.",
    "max_seq_length": 2048,
    "epochs":         1,
    "batch_size":     2,
    "learning_rate":  2e-4,
    "output_dir":     "./outputs",
}


def to_messages(row, system_prompt):
    """هر ردیف دیتاست را به قالب پیام‌های گفتگو تبدیل می‌کند
    (پشتیبانی از حالت alpaca و حالت آمادهٔ messages)."""
    if "messages" in row:
        msgs = row["messages"]
    elif "instruction" in row and ("output" in row or "response" in row):
        msgs = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row.get("response") or row.get("output")},
        ]
    else:
        raise ValueError(f"Unsupported row schema: {list(row.keys())}")
    if system_prompt and (not msgs or msgs[0]["role"] != "system"):
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    return msgs


def train(cfg, data_file, base_model):
    """آموزش با unsloth؛ مسیر پوشهٔ مدل نهایی را برمی‌گرداند."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    ds = load_dataset("json", data_files=str(data_file), split="train")

    def render(row):
        msgs = to_messages(row, cfg["system_prompt"])
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False,
                                                      add_generation_prompt=False)}
    ds = ds.map(render, remove_columns=ds.column_names)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,   # trl>=0.13؛ در نسخه‌های قدیمی‌تر tokenizer=
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=cfg["max_seq_length"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=4,
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            logging_steps=1,
            optim="adamw_8bit",
            output_dir=cfg["output_dir"],
            save_strategy="epoch",
            report_to="none",
        ),
    )
    trainer.train()

    save_dir = os.path.join(cfg["output_dir"], "final")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir


def main():
    cfg = CONFIG

    # ۱) احراز هویت با توکن شما
    manager = MLOpsManager(user_token=cfg["token"])

    # ۲) دریافت دیتاست به پوشهٔ ./dataset
    s3_download(
        dataset_name=cfg["dataset_name"],
        absolute_path="./dataset",
        token=cfg["token"],
        clearml_api_host=cfg["clearml_api_host"],
        user_management_url=cfg["user_management_api"],
        s3_endpoint_url=cfg["s3_endpoint_url"],
    )
    data_file = next(Path("./dataset").rglob("*.jsonl"))

    # ۳) دریافت مدل پایه به پوشهٔ ./base_model
    manager.get_model(model_name=cfg["base_model_name"], local_dest="./base_model")
    base_model = str(next(Path("./base_model").rglob("config.json")).parent)

    # ۴) آموزش
    save_dir = train(cfg, data_file, base_model)

    # ۵) بارگذاری مدل خروجی در رجیستری
    model_id = manager.add_model(
        source_type="local",
        source_path=save_dir,
        model_name=cfg["save_model_name"],
    )
    print(f"done: pushed model_id={model_id}")


if __name__ == "__main__":
    main()
