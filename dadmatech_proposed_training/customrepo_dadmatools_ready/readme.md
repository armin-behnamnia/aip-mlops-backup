# فاین‌تیون ShareGPT دارای tool call با Unsloth

نسخهٔ بازبینی‌شده (تغییرات نسبت به نسخهٔ اولیه در `CHANGES.md`). به‌صورت پیش‌فرض یک اجرا انجام می‌شود؛ حالت آن (`enable_thinking`) با `THINKING_MODES=auto` به‌صورت خودکار از روی داده و chat template انتخاب می‌شود:

- اگر داده حاوی reasoning باشد (فیلد `reasoning_content` یا تگ‌های thinking) یا صریحاً `THINKING_MODES=true` باشد → اجرای thinking
- در غیر این صورت → یک اجرای non-thinking واحد
- `THINKING_MODES=both` هر دو اجرا را مانند نسخهٔ قبلی انجام می‌دهد

روی فایل train، مطابق پارامترها آموزش انجام می‌شود، loss در CSV و JSONL ذخیره می‌شود، پایان هر epoch چک‌پوینت (فقط وزن‌ها، بدون state بهینه‌ساز، فقط آخرین چک‌پوینت نگه داشته می‌شود) گرفته می‌شود و روی فایل test قبل از آموزش و بعد از هر چک‌پوینت inference انجام می‌شود.

## پارامترهای اعمال‌شده

| پارامتر | مقدار |
|---|---:|
| learning rate | `1e-4` (همان `10e-5`) |
| LoRA rank | `32` |
| LoRA alpha | `16` |
| LoRA dropout | `0.05` |
| LoRA target modules | `q_proj,o_proj,k_proj,v_proj,gate_proj,up_proj,down_proj` |
| extra unfrozen layers | دو بلاک اول + آخرین بلاک decoder |
| epochs | `1` (پیش‌فرض داکیومنت) |
| batch size / grad accum | `2 / 4` (پیش‌فرض داکیومنت) |
| optimizer | `adamw_8bit` (پیش‌فرض داکیومنت) |
| warmup | `0.1` epoch (به‌صورت `warmup_ratio = 0.1 / epochs`) |

مقادیر epochs، batch size و gradient accumulation با متغیرهای محیطی `EPOCHS`، `BATCH_SIZE` و `GRAD_ACCUM` قابل تغییرند.

## مواردی که شما باید تنظیم کنید

فایل `.env.example` را به `.env` کپی کنید و این مقادیر را تکمیل کنید (`.env` در `.gitignore` است و نباید در گیت باشد):

```env
BEARER_TOKEN=...
DATASET_NAME=نام_دیتاست_در_رجیستری
BASE_MODEL_NAME=نام_مدل_در_پلتفرم
```

دیتاست و مدل پایه به‌صورت پیش‌فرض از رجیستری‌های پلتفرم دانلود می‌شوند. اگر مدل یا دیتاست را روی دیسک دارید، با `BASE_MODEL_DIR` و `DATASET_DIR` می‌توانید دانلود مربوطه را رد کنید. برای پیدا کردن فایل‌های train/test، ابتدا مقادیر `TRAIN_FILE` و `TEST_FILE` و در غیر این صورت جستجوی خودکار (نام شامل train/test) استفاده می‌شود.

اگر `enable_thinking` برای اجرای thinking لازم است ولی chat template مدل پایه از آن پشتیبانی نمی‌کند، کد با خطای واضح متوقف می‌شود؛ در آن صورت یا مدلی سازگار (مثلاً Qwen3) انتخاب کنید یا `THINKING_MODES=false` بگذارید.

## پیشنهادهای اجرا

- `MAX_TEST_SAMPLES` را تنظیم کنید (مثلاً `100`)؛ پیش‌فرض، کل فایل test در هر نقطهٔ زمانی ارزیابی می‌شود و پرهزینه است.
- با لایه‌های کامل آزادشده و بارگذاری ۱۶بیت، مدل‌های بزرگ به GPU با حافظهٔ زیاد نیاز دارند. اگر کم آوردید: `EXTRA_UNFROZEN_MODE=first2` یا `none` + `LOAD_IN_4BIT=1` (حالت QLoRA خالص؛ دیگر خواستهٔ «دو لایهٔ اول + آخر» را اجرا نمی‌کند).

## قالب داده

کد فقط قالب استاندارد (canonical) خروجی ماژول Data Ingestion پلتفرم را می‌پذیرد؛ ورودی‌های ShareGPT/DADMAtools قبل از ثبت در رجیستری، توسط همان ماژول بازساختار می‌شوند:

- OpenAI chat: `messages` با `role/content` (قالب اصلی)
- Alpaca: `instruction` به‌همراه `output` یا `response` (حالت جایگزین)
- ابزارها در ستون `tools` یا `functions`
- فراخوانی ساختاریافته در `tool_calls` یا `function_call`
- پاسخ ابزار با roleهای `tool` یا `function`
- قالب `conversations` (ShareGPT) پذیرفته نمی‌شود و خطای واضح می‌دهد؛ اگر چنین داده‌ای دارید، ابتدا آن را از طریق Data Ingestion پلتفرم ثبت کنید.

در هر مکالمه یک نمونهٔ آموزشی برای هر پاسخ assistant ساخته می‌شود و loss فقط روی پاسخ همان assistant محاسبه می‌شود؛ بنابراین turnهای tool call میانی نیز آموزش داده می‌شوند.

### نکته دربارهٔ thinking

`enable_thinking` یک کلید chat template است، نه پارامتری که به‌تنهایی reasoning تولید کند. در اجرای non-thinking، `reasoning_content` یا بخش thinking محتوا از target حذف می‌شود. در اجرای thinking، reasoning موجود حفظ می‌شود. اگر دیتاست reasoning واقعی نداشته باشد، حالت `auto` به‌صورت خودکار سراغ اجرای thinking نمی‌رود تا دو آموزش یکسان و بی‌فایده اجرا نشود.

## لایه‌های کاملاً قابل‌آموزش و حافظهٔ GPU

دو بلاک اول و بلاک آخر decoder با `modules_to_save` به‌طور کامل آموزش داده و همراه adapter ذخیره می‌شوند. این کار با وزن پایهٔ 4-bit سازگار نیست، بنابراین مقدار پیش‌فرض زیر لازم است:

```env
EXTRA_UNFROZEN_MODE=first2_last
LOAD_IN_4BIT=0
```

## خروجی‌ها

برای هر mode یک پوشه ساخته می‌شود:

```text
outputs/
├── thinking_false/                  # (یا thinking_true، بسته به mode)
│   ├── checkpoint-.../              # فقط آخرین چک‌پوینت، فقط وزن‌ها
│   ├── final_adapter/               # adapter (LoRA + modules_to_save)
│   ├── final_model/                 # مدل کامل merge‌شده ۱۶بیتی — این در رجیستری ثبت می‌شود
│   ├── logs/loss.csv
│   ├── logs/loss.jsonl
│   ├── logs/trainable_selection.json
│   └── inference/
│       ├── before_training.jsonl
│       └── after_epoch_<n>_checkpoint-<step>.jsonl
└── registered_models.json
```

تنها پوشهٔ `final_model` (مدل مستقل و کامل، دارای `config.json`) با نام `<SAVE_MODEL_NAME>_thinking_true|false` در رجیستری مدل ذخیره می‌شود؛ چک‌پوینت‌ها و لاگ‌ها در رجیستری بارگذاری نمی‌شوند. مدل ثبت‌شده قابل سرو مستقیم و قابل استفادهٔ مجدد به‌عنوان مدل پایه است.

## اجرا

```bash
pip install -r requirements.txt
python train.py
```

این آموزش به GPU انویدیا، توکن پلتفرم و مدل پایهٔ ثبت‌شده نیاز دارد. entry point اختیاری `train_hyperparameter.py` همان pipeline را اجرا می‌کند و چند مقدار اصلی (از جمله epochs، batch_size و grad_accum) را به ClearML متصل می‌کند تا از طریق فیلد hyperparams در API قابل بازنویسی باشند.

## تست

```bash
python -m unittest discover -s tests
```
