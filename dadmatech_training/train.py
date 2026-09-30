"""
آموزش سفارشی DadmaTech روی پلتفرم MLOps — فاین‌تیون مدل زبانی با unsloth + LoRA.

جریان کلی (مطابق قالب sample_training پلتفرم):
  ۱) احراز هویت            -> MLOpsManager
  ۲) دریافت دیتاست         -> s3_download
  ۳) دریافت مدل پایه        -> manager.get_model
  ۴) آموزش، سپس ذخیرهٔ مدل  -> manager.add_model

نام مدل و دیتاست از متغیرهای محیطی خوانده می‌شوند (README):
    DATASET_NAME        نام دیتاست در رجیستری داده        (الزامی)
    BASE_MODEL_NAME     نام مدل پایه در رجیستری مدل       (الزامی)
    OUTPUT_MODEL_NAME   نام مدل خروجی در رجیستری مدل      (اختیاری)

مقادیر آموزش از پیش‌فرض‌های داکیومنت پلتفرم گرفته شده‌اند، با تغییرات
درخواست‌شدهٔ کارفرما:

    learning_rate          = 10e-5
    lora_rank              = 32
    lora_alpha             = 16
    lora_dropout           = 0.05
    lora_target_modules    = q_proj,o_proj,k_proj,v_proj,gate_proj,up_proj,down_proj
    extra_unfreezed_layers = دو لایهٔ اول مدل + لایهٔ آخر مدل
    warmup_steps           = ۰/۱ ایپاک (۱۰٪ِ گام‌های یک ایپاک)

نکتهٔ مهم دربارهٔ لایه‌های آزادشده:
  برای اینکه لایه‌های آزادشده واقعاً آموزش ببینند، مدل به‌صورت ۱۶بیت بارگذاری
  می‌شود (نه ۴بیت؛ وزن‌های کوانتیزهٔ ۴بیت قابل بهینه‌سازی نیستند) و در انتها به‌جای
  ذخیرهٔ صرفِ آداپتور، مدل merge‌شدهٔ کامل ذخیره و در رجیستری ثبت می‌شود تا
  تغییرات آن لایه‌ها از دست نرود.

هایپرپارامترها (learning_rate, epochs, lora_rank و ...) از طریق ClearML ثبت
می‌شوند تا هنگام ساخت تسک از فیلد hyperparams در بدنهٔ درخواست API
قابل بازنویسی باشند (نمونهٔ payload در README).
"""

import math
import os
import re
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
from clearml import Task                            # <-- برای هایپرپارامتر

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

    # چه چیزی گرفته/ذخیره شود — نام‌ها از متغیرهای محیطی خوانده می‌شوند
    "dataset_name":        os.environ.get("DATASET_NAME"),                 # الزامی
    "base_model_name":     os.environ.get("BASE_MODEL_NAME"),              # الزامی
    "save_model_name":     os.environ.get("OUTPUT_MODEL_NAME",
                                          "dadmatech_finetuned"),          # اختیاری

    # ---- پارامترهای آموزش (پیش‌فرض داکیومنت + تغییرات درخواستی کارفرما) ----
    "system_prompt":  "You are a helpful assistant.",
    "max_seq_length": 2048,
    "epochs":         1,
    "batch_size":     2,
    "grad_accum":     4,                       # همان پیش‌فرض داکیومنت
    "learning_rate":  10e-5,                   # درخواست کارفرما: lr = 10e-5
    "output_dir":     "./outputs",

    # ---- تنظیمات LoRA (درخواست کارفرما) ----
    "lora_rank":      32,
    "lora_alpha":     16,
    "lora_dropout":   0.05,                    # درخواست کارفرما: 5%
    "lora_target_modules": ["q_proj", "o_proj", "k_proj", "v_proj",
                            "gate_proj", "up_proj", "down_proj"],

    # ---- لایه‌های آزادشدهٔ اضافی (درخواست کارفرما) ----
    # دو لایهٔ اول مدل + لایهٔ آخر مدل؛ وزن‌های این لایه‌ها علاوه بر LoRA
    # به‌صورت کامل آموزش می‌بینند.
    "unfreeze_first_n_layers": 2,
    "unfreeze_last_layer":     True,
    "load_in_4bit":            False,   # آموزش لایه‌های آزاد با وزن ۴بیت ممکن نیست

    # ---- warmup (درخواست کارفرما: ۰/۱ ایپاک) ----
    # مقدار دقیق گام‌ها بعد از دانلود دیتاست محاسبه می‌شود:
    # warmup_steps = warmup_epochs * ceil(len(dataset) / (batch_size * grad_accum))
    "warmup_epochs":  0.1,
}


def to_messages(row, system_prompt):
    """هر ردیف دیتاست را به قالب پیام‌های گفتگو تبدیل می‌کند
    (پشتیبانی از حالت messages، حالت alpaca و فیلد text آماده)."""
    if "messages" in row and row["messages"] is not None:
        msgs = row["messages"]
    elif "instruction" in row and ("output" in row or "response" in row):
        user = row["instruction"]
        if row.get("input"):                     # alpaca با ورودی اختیاری
            user = f"{user}\n{row['input']}"
        msgs = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row.get("response") or row.get("output")},
        ]
    elif "text" in row and row["text"]:
        # متن از قبل رندرشده؛ بدون تغییر استفاده می‌شود
        return None
    else:
        raise ValueError(f"Unsupported row schema: {list(row.keys())}")
    if system_prompt and (not msgs or msgs[0]["role"] != "system"):
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    return msgs


def find_data_file(dataset_dir):
    """نخستین فایل دادهٔ دانلودشده را پیدا می‌کند (jsonl/json/parquet/csv)."""
    for pattern, fmt in (("*.jsonl", "json"), ("*.json", "json"),
                         ("*.parquet", "parquet"), ("*.csv", "csv")):
        found = next(Path(dataset_dir).rglob(pattern), None)
        if found is not None:
            return found, fmt
    raise FileNotFoundError(
        f"No .jsonl/.json/.parquet/.csv file found under {dataset_dir}")


def unfreeze_extra_layers(peft_model, first_n=2, include_last=True):
    """آزاد کردن کامل چند لایهٔ ترنسفورمر (علاوه بر LoRA) برای آموزش.

    وزن‌های لایه‌های هدف requires_grad می‌گیرند؛ چون مدل ۱۶بیت بارگذاری شده،
    این لایه‌ها واقعاً بهینه‌سازی می‌شوند و در خروجی merge‌شده حفظ می‌مانند.
    """
    layer_ids = set()
    for name, _ in peft_model.named_parameters():
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            layer_ids.add(int(m.group(1)))
    if not layer_ids:
        print("[unfreeze] WARNING: no transformer layers found; skipped.")
        return

    last = max(layer_ids)
    targets = set(range(min(first_n, last + 1)))     # لایه‌های ۰ تا first_n-1
    if include_last:
        targets.add(last)

    unfrozen = 0
    for name, param in peft_model.named_parameters():
        m = re.search(r"layers\.(\d+)\.", name)
        if not m or int(m.group(1)) not in targets:
            continue
        if "lora_" in name:                          # پارامترهای LoRA خودشان آموزش‌پذیرند
            continue
        param.requires_grad = True
        unfrozen += 1
    print(f"[unfreeze] layers {sorted(targets)} (of 0..{last}) -> "
          f"{unfrozen} weight tensors set trainable")


def train(cfg, data_file, data_format, base_model):
    """آموزش با unsloth؛ مسیر پوشهٔ مدل نهایی را برمی‌گرداند."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],   # ۱۶بیت تا لایه‌های آزادشده آموزش ببینند
        dtype=None,                         # bf16 در صورت پشتیبانی GPU، وگرنه fp16
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        use_gradient_checkpointing="unsloth",
    )

    # لایه‌های آزادشدهٔ اضافی (دو لایهٔ اول + لایهٔ آخر)
    unfreeze_extra_layers(model,
                          first_n=cfg["unfreeze_first_n_layers"],
                          include_last=cfg["unfreeze_last_layer"])
    model.print_trainable_parameters()

    ds = load_dataset(data_format, data_files=str(data_file), split="train")

    def render(row):
        if "text" in row and row["text"] and "messages" not in row \
                and "instruction" not in row:
            return {"text": row["text"]}
        msgs = to_messages(row, cfg["system_prompt"])
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False,
                                                      add_generation_prompt=False)}
    ds = ds.map(render, remove_columns=ds.column_names)

    # warmup_steps = ۰/۱ ایپاک (درخواست کارفرما)
    steps_per_epoch = math.ceil(len(ds) / (cfg["batch_size"] * cfg["grad_accum"]))
    warmup_steps = max(1, int(cfg["warmup_epochs"] * steps_per_epoch))
    print(f"[warmup] dataset={len(ds)} rows, steps/epoch={steps_per_epoch}, "
          f"warmup_steps={warmup_steps} (0.1 epoch)")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,   # trl>=0.13؛ در نسخه‌های قدیمی‌تر tokenizer=
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=cfg["max_seq_length"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            warmup_steps=warmup_steps,
            logging_steps=1,
            optim="adamw_8bit",
            output_dir=cfg["output_dir"],
            save_strategy="epoch",
            report_to="none",
        ),
    )
    trainer.train()

    # ذخیرهٔ مدل merge‌شدهٔ کامل (نه فقط آداپتور) تا تغییرات لایه‌های
    # آزادشده هم در مدل خروجی حفظ شود.
    save_dir = os.path.join(cfg["output_dir"], "final")
    model.save_pretrained_merged(save_dir, tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained(save_dir)
    return save_dir


def _as_int(value):
    return int(float(value))


def _as_float(value):
    return float(value)


def main():
    cfg = CONFIG

    # --- اعتبارسنجی متغیرهای محیطی الزامی -----------------------------------
    missing = [name for name, key in (
        ("BEARER_TOKEN", "token"),
        ("DATASET_NAME", "dataset_name"),
        ("BASE_MODEL_NAME", "base_model_name"),
    ) if not cfg[key]]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + " — آن‌ها را در فایل .env کنار کد یا در محیط اجرا تنظیم کنید (README)."
        )

    # --- هایپرپارامترهای ClearML -------------------------------------------
    # به تسک در حال اجرا وصل می‌شویم و مقادیر را ثبت می‌کنیم؛ اگر در بدنهٔ
    # درخواستِ API مقداری برای این‌ها بفرستید، همان جایگزین پیش‌فرض می‌شود.
    task = Task.init(project_name="dadmatech",
                     task_name="dadmatech custom finetuning")
    hparams = {
        "learning_rate":       cfg["learning_rate"],
        "epochs":              cfg["epochs"],
        "batch_size":          cfg["batch_size"],
        "grad_accum":          cfg["grad_accum"],
        "max_seq_length":      cfg["max_seq_length"],
        "lora_rank":           cfg["lora_rank"],
        "lora_alpha":          cfg["lora_alpha"],
        "lora_dropout":        cfg["lora_dropout"],
    }
    task.connect(hparams)                 # روی agent، مقادیر API اینجا اعمال می‌شوند

    # اعمال روی cfg + کست به نوع درست
    cfg["learning_rate"]       = _as_float(hparams["learning_rate"])
    cfg["epochs"]              = _as_int(hparams["epochs"])
    cfg["batch_size"]          = _as_int(hparams["batch_size"])
    cfg["grad_accum"]          = _as_int(hparams["grad_accum"])
    cfg["max_seq_length"]      = _as_int(hparams["max_seq_length"])
    cfg["lora_rank"]           = _as_int(hparams["lora_rank"])
    cfg["lora_alpha"]          = _as_int(hparams["lora_alpha"])
    cfg["lora_dropout"]        = _as_float(hparams["lora_dropout"])

    # چاپ پیکربندی مؤثر تا در لاگ اجرا قابل راستی‌آزمایی باشد
    print("=" * 60)
    for key in ("dataset_name", "base_model_name", "save_model_name",
                "system_prompt", "max_seq_length", "epochs", "batch_size",
                "grad_accum", "learning_rate", "lora_rank", "lora_alpha",
                "lora_dropout", "lora_target_modules",
                "unfreeze_first_n_layers", "unfreeze_last_layer",
                "load_in_4bit", "warmup_epochs"):
        print(f"[config] {key:24s} = {cfg[key]}")
    print("=" * 60)

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
    data_file, data_format = find_data_file("./dataset")
    print(f"[dataset] file={data_file} format={data_format}")

    # ۳) دریافت مدل پایه به پوشهٔ ./base_model
    manager.get_model(model_name=cfg["base_model_name"], local_dest="./base_model")
    base_model = str(next(Path("./base_model").rglob("config.json")).parent)

    # ۴) آموزش
    save_dir = train(cfg, data_file, data_format, base_model)

    # ۵) بارگذاری مدل خروجی در رجیستری
    model_id = manager.add_model(
        source_type="local",
        source_path=save_dir,
        model_name=cfg["save_model_name"],
    )
    print(f"done: pushed model_id={model_id}")


if __name__ == "__main__":
    main()
