# نمونهٔ آموزش سفارشی روی پلتفرم MLOps

این مخزن یک **نمونهٔ کامل** است که نشان می‌دهد چطور کد آموزش دلخواه خودتان را بنویسید و روی پلتفرم اجرا کنید. کد شما آزاد است هر کاری بکند؛ فقط کافی است داده و مدل را از پلتفرم بگیرد و در انتها مدل خروجی را به پلتفرم برگرداند. این نمونه از `unsloth` برای فاین‌تیون یک مدل زبانی استفاده می‌کند، اما شما می‌توانید بدنهٔ آموزش را با هر چیزی جایگزین کنید.

---

## این مخزن چه می‌کند؟

کل جریان در فایل [`train.py`](train.py) و در پنج گام ساده است:

| گام | کار | ابزار |
|-----|------|-------|
| ۱ | احراز هویت با توکن شما | `MLOpsManager` |
| ۲ | دریافت دیتاست روی دیسک | `s3_download` |
| ۳ | دریافت مدل پایه روی دیسک | `manager.get_model` |
| ۴ | آموزش (منطق دلخواه شما) | `unsloth` |
| ۵ | ذخیرهٔ مدل خروجی در رجیستری | `manager.add_model` |

---

## پیش‌نیازها: توکن‌ها

### ۱) توکن پلتفرم (`BEARER_TOKEN`)

این توکن هویت شماست؛ SDKها با آن به‌جای شما به داده و مدل دسترسی می‌گیرند.

آن را از این مسیر بردارید:

> **عملیات یادگیری ماشین ‹ کلید اپلیکیشن**

سپس آن را در یک فایل `.env` کنار کد قرار دهید:

```env
BEARER_TOKEN=<توکن شما>
```

> ⚠️ توکن را در گیت کامیت نکنید. فایل `.env` را در `.gitignore` بگذارید.

### ۲) توکن گیت‌لب (برای مخزن خصوصی)

اگر مخزن شما خصوصی است، پلتفرم برای `clone` کردن آن به یک توکن نیاز دارد. یک توکن بسازید از مسیر:

> **عملیات یادگیری ماشین ‹ فضای گیت**

این توکن را داخل آدرس مخزن قرار می‌دهید (بخش «اجرا از طریق پلتفرم» را ببینید).

---

## متغیرهای محیطی

تنها متغیری که **شما** باید تنظیم کنید `BEARER_TOKEN` است.

سه متغیر زیر از قبل در محیط اجرا فراهم هستند و **نیازی به تنظیم آن‌ها ندارید**:

```python
os.getenv("CLEARML_API_HOST")
os.getenv("USER_MANAGEMENT_API")
os.getenv("CEPH_ENDPOINT_URL")
```

آن‌ها را دست نزنید؛ به‌صورت خودکار مقداردهی می‌شوند.

---

## توضیح گام‌به‌گام کد

### تنظیم آفلاین (بالای فایل)

قبل از هر import مربوط به مدل، این سه خط قرار دارند:

```python
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
```

چون مدل و توکنایزر از قبل به‌صورت محلی دریافت می‌شوند، این خطوط جلوی تلاش کتابخانه‌ها برای اتصال به اینترنت را می‌گیرند.

### گام ۱ — احراز هویت (مدل/داده)

```python
manager = MLOpsManager(user_token=cfg["token"])
```

یک بار با توکن شما احراز هویت می‌شود؛ همین شیء برای گرفتن و ذخیرهٔ مدل استفاده می‌شود.

### گام ۲ — دریافت دیتاست (داده)

```python
s3_download(
    dataset_name=cfg["dataset_name"],
    absolute_path="./dataset",
    token=cfg["token"],
    clearml_api_host=cfg["clearml_api_host"],
    user_management_url=cfg["user_management_api"],
    s3_endpoint_url=cfg["s3_endpoint_url"],
)
data_file = next(Path("./dataset").rglob("*.jsonl"))
```

دیتاست ثبت‌شده (اینجا `alpaca-mini-6`) در پوشهٔ `./dataset` دانلود می‌شود و فایل آموزش پیدا می‌شود.

### گام ۳ — دریافت مدل پایه (مدل)

```python
manager.get_model(model_name=cfg["base_model_name"], local_dest="./base_model")
base_model = str(next(Path("./base_model").rglob("config.json")).parent)
```

مدل پایه (اینجا `smoltest02`) دانلود و مسیر پوشهٔ مدل استخراج می‌شود.

### گام ۴ — آموزش

```python
save_dir = train(cfg, data_file, base_model)
```

تابع `train` منطق آموزش شماست: بارگذاری مدل با `unsloth`، افزودن LoRA، تبدیل داده به قالب گفتگو، و اجرای `SFTTrainer`. **این بخش را برای کار خودتان تغییر دهید.** پارامترهای آموزش (تعداد epoch، نرخ یادگیری، batch و…) در `CONFIG` هستند.

### گام ۵ — ذخیرهٔ مدل خروجی (مدل)

```python
model_id = manager.add_model(
    source_type="local",
    source_path=save_dir,
    model_name=cfg["save_model_name"],
)
```

مدل آموزش‌دیده با نام `smoltest02_custom_repo` در رجیستری ثبت می‌شود.

---

## اجرا از طریق پلتفرم

کد خود را در گیت‌لب قرار دهید، سپس یک درخواست به endpoint زیر بزنید تا پلتفرم مخزن شما را `clone` کند و `train.py` را اجرا کند.

**Endpoint:**

```
POST https://mlopsapipilot.aip.sharif.ir/train/task/start
```

**شکل درخواست:**

```json
{
  "repo_url": "https://oauth2:<توکن‌گیت‌لب>@gitpilot.aip.sharif.ir/<گروه>/<مخزن>.git",
  "entry_point": "train.py"
}
```

**نمونهٔ کامل با curl:**

```bash
curl -X POST "https://mlopsapipilot.aip.sharif.ir/train/task/start" \
  -H "Authorization: Bearer <BEARER_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://oauth2:<توکن‌گیت‌لب>@gitpilot.aip.sharif.ir/<گروه>/<مخزن>.git",
    "entry_point": "train.py"
  }'
```

- `repo_url`: آدرس مخزن شما به همراه توکن گیت‌لب (برای مخزن خصوصی).
- `entry_point`: فایلی که باید اجرا شود، نسبت به ریشهٔ مخزن (اینجا `train.py`).
- هدر `Authorization`: توکن پلتفرم شما (`BEARER_TOKEN`).

پاسخ شامل `task_id` و لینک آن است؛ با آن می‌توانید وضعیت و لاگ اجرا را دنبال کنید.

---

## نصب محلی (اختیاری)

```bash
pip install -r requirements.txt
```

> نکته: اجرای واقعی به GPU (NVIDIA) نیاز دارد؛ `unsloth` روی CPU اجرا نمی‌شود.
