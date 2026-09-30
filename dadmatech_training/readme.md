# آموزش سفارشی DadmaTech روی پلتفرم MLOps

این مخزن کد فاین‌تیون سفارشی کارفرما (DadmaTech) است که طبق قالب رسمی
`sample_training` پلتفرم نوشته شده و مدل پایه و دیتاست را از رجیستری‌های
پلتفرم می‌گیرد، با `unsloth` + LoRA آموزش می‌دهد، و مدل خروجی را در رجیستری
مدل ثبت می‌کند.

---

## این مخزن چه می‌کند؟

کل جریان در فایل [`train.py`](train.py) و در پنج گام ساده است:

| گام | کار | ابزار |
|-----|------|-------|
| ۱ | احراز هویت با توکن شما | `MLOpsManager` |
| ۲ | دریافت دیتاست روی دیسک | `s3_download` |
| ۳ | دریافت مدل پایه روی دیسک | `manager.get_model` |
| ۴ | آموزش (unsloth + LoRA + لایه‌های آزادشده) | `SFTTrainer` |
| ۵ | ذخیرهٔ مدل خروجی (merge‌شدهٔ کامل) در رجیستری | `manager.add_model` |

---

## متغیرهای محیطی

چهار متغیر زیر را باید **پر کنید** — ساده‌ترین راه: فایل `.env.example` را به
`.env` کپی کنید و مقادیر را بنویسید (`.env` در `.gitignore` است):

| متغیر | توضیح | الزامی؟ |
|-------|--------|---------|
| `BEARER_TOKEN` | توکن پلتفرم — از **عملیات یادگیری ماشین ‹ کلید اپلیکیشن** | ✅ |
| `DATASET_NAME` | نام دیتاست ثبت‌شده در رجیستری داده | ✅ |
| `BASE_MODEL_NAME` | نام مدل پایهٔ ثبت‌شده در رجیستری مدل | ✅ |
| `OUTPUT_MODEL_NAME` | نام مدل خروجی در رجیستری مدل | اختیاری (پیش‌فرض `dadmatech_finetuned`) |

سه متغیر زیر از قبل در محیط اجرای پلتفرم فراهم هستند؛ **دست نزنید**:

```python
os.getenv("CLEARML_API_HOST")
os.getenv("USER_MANAGEMENT_API")
os.getenv("CEPH_ENDPOINT_URL")
```

> ⚠️ توکن را در گیت کامیت نکنید. `.env` در `.gitignore` قرار دارد.

---

## هایپرپارامترها (درخواست کارفرما)

مقادیر از پیش‌فرض‌های داکیومنت پلتفرم گرفته شده‌اند، به‌جز موارد درخواستی
کارفرما:

| پارامتر | مقدار |
|---------|-------|
| `learning_rate` | `10e-5` (یعنی `1e-4`) |
| `lora_rank` | `32` |
| `lora_alpha` | `16` |
| `lora_dropout` | `0.05` (۵٪) |
| `lora_target_modules` | `q_proj, o_proj, k_proj, v_proj, gate_proj, up_proj, down_proj` |
| لایه‌های آزادشدهٔ اضافی | **دو لایهٔ اول مدل + لایهٔ آخر مدل** (علاوه بر LoRA، کامل آموزش می‌بینند) |
| `warmup_steps` | **۰/۱ ایپاک** — به‌صورت خودکار از روی اندازهٔ دیتاست محاسبه می‌شود: `0.1 × ceil(len(dataset) / (batch_size × grad_accum))` |
| `epochs` | `1` (پیش‌فرض داکیومنت) |
| `batch_size` | `2` (پیش‌فرض داکیومنت) |
| `grad_accum` | `4` (پیش‌فرض داکیومنت) |
| `max_seq_length` | `2048` (پیش‌فرض داکیومنت) |

### دو نکتهٔ فنی مهم دربارهٔ لایه‌های آزادشده

1. **مدل ۱۶بیت بارگذاری می‌شود** (`load_in_4bit=False`). وزن‌های کوانتیزهٔ
   ۴بیت قابل بهینه‌سازی نیستند، پس برای آموزش واقعیِ لایه‌های آزادشده، بارگذاری
   ۴بیت (پیش‌فرض داکیومنت) عمداً غیرفعال شده است. در نتیجه مصرف VRAM نسبت به
   حالت ۴بیت بیشتر است.
2. **خروجی مدل merge‌شدهٔ کامل است** (نه فقط آداپتور LoRA). اگر فقط آداپتور
   ذخیره می‌شد، تغییراتِ لایه‌های آزادشده (دو لایهٔ اول و لایهٔ آخر) در مدل
   خروجی از دست می‌رفت. خروجی نهایی یک مدل مستقل و کامل است که مستقیم قابل
   سرو کردن است.

---

## قالب دیتاست

فایل داده به‌صورت خودکار از بین `jsonl / json / parquet / csv` پیدا و با همان
قالب خوانده می‌شود. هر ردیف می‌تواند یکی از این شکل‌ها باشد:

- **messages آماده:** `{"messages": [{"role": "...", "content": "..."}]}`
- **alpaca:** `{"instruction": "...", "output": "..."}` (یا `response` به‌جای
  `output`؛ فیلد اختیاری `input` هم به سؤال کاربر افزوده می‌شود)
- **متن آماده:** `{"text": "..."}` — بدون تغییر استفاده می‌شود.

در حالت‌های messages و alpaca، `system_prompt` (پیش‌فرض:
`You are a helpful assistant.`) به ابتدای گفتگو اضافه می‌شود.

---

## تغییر هایپرپارامترها از طریق API (بدون تغییر کد)

مقادیر `learning_rate, epochs, batch_size, grad_accum, max_seq_length,
lora_rank, lora_alpha, lora_dropout` به‌عنوان هایپرپارامتر ClearML ثبت
می‌شوند و هنگام ساخت تسک با فیلد `hyperparams` قابل بازنویسی‌اند:

```json
{
  "repo_url": "https://<GIT_USERNAME>:<GIT_TOKEN>@gitpilot.aip.sharif.ir/<گروه>/<مخزن>.git",
  "entry_point": "train.py",
  "hyperparams": {
    "General": {
      "epochs":        { "section": "General", "name": "epochs",        "value": "3" },
      "learning_rate": { "section": "General", "name": "learning_rate", "value": "0.0001" }
    }
  }
}
```

---

## اجرا از طریق پلتفرم

مخزن را در گیت‌لب AIP قرار دهید، سپس درخواست بزنید:

**Endpoint (طبق API Doc):**

```
POST https://mlopsapipilot.aip.sharif.ir/train/task/start
```

```bash
curl -X POST "https://mlopsapipilot.aip.sharif.ir/train/task/start" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <BEARER_TOKEN>" \
  -d '{
    "repo_url": "https://<GIT_USERNAME>:<GIT_TOKEN>@gitpilot.aip.sharif.ir/<گروه>/<مخزن>.git",
    "entry_point": "train.py"
  }'
```

- `repo_url`: آدرس مخزن به همراه توکن گیت‌لب (برای مخزن خصوصی؛ توکن از
  **عملیات یادگیری ماشین ‹ فضای گیت**).
- `entry_point`: `train.py`
- هدر `Authorization`: توکن پلتفرم (`BEARER_TOKEN`).

پاسخ شامل `task_id` است؛ لاگ اجرا را می‌توانید با
`POST /train/task/log` و بدنهٔ `{"task_id": "...", "type": "log"}` دنبال کنید.
پیکربندی مؤثر آموزش (همهٔ مقادیر نهایی) در ابتدای لاگ چاپ می‌شود.

---

## نصب محلی (اختیاری)

```bash
pip install -r requirements.txt
cp .env.example .env   # مقادیر را پر کنید
python train.py
```

> نکته: اجرای واقعی به GPU (NVIDIA) نیاز دارد؛ `unsloth` روی CPU اجرا نمی‌شود.
