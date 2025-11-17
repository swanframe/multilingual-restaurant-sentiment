# Multilingual Restaurant Sentiment Analysis

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.2.0](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-mBERT-yellow.svg)

A production-ready NLP system that classifies **English** and **Indonesian** restaurant reviews into **negative / neutral / positive**, returning **confidence scores**, **per-class probabilities**, and **low-confidence advisories** to support human-in-the-loop decisions. The project includes a complete data pipeline, training with mBERT, transparent evaluation, and a lightweight Flask API with a built-in demo UI.

## Table of Contents
- [Quick Start](#quick-start)
- [GPU / Colab Training (Optional)](#gpu-colab-training)
- [Features](#features)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)
- [Project Layout](#project-layout)
- [Using Your Own Data](#using-your-own-data)
- [Local Tips & Troubleshooting](#local-tips--troubleshooting)
- [License](#license)
- [Contact](#contact)

---

## 🚀 Quick Start <a name="quick-start"></a>

```bash
# 1) Get the code
# Option A: Clone from GitHub (recommended)
git clone https://github.com/swanfame/multilingual-restaurant-sentiment.git
cd multilingual-restaurant-sentiment

# Option B: If you have the project folder locally, cd into it:
cd multilingual-restaurant-sentiment

# Option C: If you're viewing this on GitHub, click “Code” → “Download ZIP”,
# unzip it, then:
cd multilingual-restaurant-sentiment

# 2) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Prepare data folders
# - Place 6 AI CSVs under data/ai/model_{A..F}/ (columns: text, language [en|id], sentiment)
# - Place real test CSV at data/real/real_reviews.csv (labels optional)
mkdir -p data/ai/model_A data/ai/model_B data/ai/model_C data/ai/model_D data/ai/model_E data/ai/model_F data/real

# 4) Build clean splits (80/10/10, stratified by language+sentiment)
python -m src.data.datasets --summary

# 5) Train (saves best checkpoint to saved_models/best/)
python -m src.training.train

# 6) Evaluate (held-out AI test + real data)
python -m src.evaluation.evaluate --which both

# 7) Serve API + minimal UI (IMPORTANT: run as a module)
python -m app.server
# Then open http://localhost:8000
```

> **Note:** If a CUDA GPU is available (e.g., on Google Colab), the model will automatically use it.
> For Colab-specific steps and CPU vs GPU timings, see
> [⚡ GPU / Colab Training (Optional)](#gpu-colab-training).

---

## ⚡ GPU / Colab Training (Optional) <a name="gpu-colab-training"></a>

This project is **GPU-ready**: it automatically uses a CUDA device when available
(via `torch.cuda.is_available()` and `model.to(device)`). On environments like
**Google Colab** with a Tesla T4 GPU, you can train the model much faster with
no code changes.

### Running on Google Colab

1. Open a new notebook on Google Colab.
2. Go to **Runtime → Change runtime type → Hardware accelerator → GPU → Save**.
3. In the first cell:

   ```bash
   !git clone https://github.com/swanframe/multilingual-restaurant-sentiment.git
   %cd multilingual-restaurant-sentiment

   !pip install -r requirements.txt
   ```

4. Build stratified splits and train (GPU will be used automatically if available):

   ```bash
   # In Colab cells, prefix shell commands with "!"
   !python -m src.data.datasets --summary
   !python -m src.training.train
   ```

5. (Optional) Verify that Colab sees the GPU:

   ```python
   import torch, platform
   print("Python:", platform.python_version())
   print("Torch:", torch.__version__)
   print("CUDA available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print("GPU:", torch.cuda.get_device_name(0))
   ```

   Example output from a Colab T4 runtime:

   ```text
   Python: 3.12.12
   Torch: 2.2.0+cu121
   CUDA available: True
   GPU: Tesla T4
   ```

### CPU vs GPU Training Time

Using the default configuration on the same dataset:

* **MacBook CPU (local):**

  * ~25–28 minutes per epoch (≈1600 seconds/epoch)
  * Best validation macro-F1: **0.9632**

* **Google Colab GPU (Tesla T4):**

  * ~30 seconds per epoch
  * Best validation macro-F1: **0.9530**

This is roughly a **50× speed-up** while keeping performance in the same band
(macro-F1 ≈ 0.95–0.96). Training on GPU makes it much more practical to iterate
on hyperparameters, data cleaning, and new experiments.

### Tips

* If you hit CUDA out-of-memory on a smaller GPU, reduce
  `train.batch_size` and/or `model.max_length` in `configs/config.yaml`.
* If you want the training logs to explicitly show the device, you can add:

  ```python
  # in src/training/train.py
  from src.training.utils import get_device

  device = get_device()
  print(f"Using device: {device}")
  model.to(device)
  ```

This will print `Using device: cuda` on Colab and `Using device: cpu` on a non-GPU machine.

---

## ✨ Features <a name="features"></a>

* **Multilingual mBERT classifier**
  One model for English & Indonesian; robust tokenization and conservative multilingual text cleaning (URLs/emails redacted, emojis demojized).

* **Confidence-aware predictions**
  Per-class and general thresholds surface **low-confidence** outputs with actionable advisories—especially useful for minority classes (neutral/negative).

* **Human-in-the-loop feedback**
  `/feedback` endpoint logs human-verified labels; telemetry enables drift tracking and targeted retraining.

* **Strict data separation**
  Real test data is **never** used in training. Pipeline uses only AI-generated datasets for training; evaluation covers held-out AI test + real data.

* **Reproducible configuration**
  All paths, hyperparameters, and thresholds live in `configs/config.yaml`; deterministic splits via seed.

* **Lightweight serving**
  Flask API with a simple JSON contract and a built-in UI for demos and manual validation.

* **Transparent evaluation**
  Macro-F1/accuracy, classification reports, confusion matrices, and error tables saved under `outputs/eval/`.

---

## 🏗️ Architecture <a name="architecture"></a>

**Tech Stack**

* **PyTorch** — model training & inference
* **Transformers (Hugging Face)** — `bert-base-multilingual-cased`
* **pandas** — data handling
* **scikit-learn** — stratified splits, metrics
* **Flask** — REST API + minimal UI

**Design Decisions**

* **mBERT (cased)** for broad multilingual coverage; conservative cleaning keeps casing and sentiment cues.
* **Stratified language+sentiment splits** preserve balance across both languages and all classes.
* **Confidence thresholds** enable routing low-confidence outputs to human review.
* **Feedback logging** powers continuous improvement and domain adaptation.

**Lifecycle**

1. **Data** → Validate & clean → Stratified splits
2. **Training** → mBERT + classifier head → Early stopping on macro-F1
3. **Evaluation** → AI test + real data → Reports & confusions
4. **Serving** → `/predict` (probs + advisories) → `/feedback` (human labels)
5. **Retraining** → Add feedback as new shard → Re-run pipeline

---

## 📖 API Documentation <a name="api-documentation"></a>

### `GET /health`

Returns service and model info.

**Response**

```json
{
  "status": "ok",
  "model_dir": "saved_models/best",
  "max_length": 192,
  "thresholds": {
    "general": 0.6,
    "per_class": { "negative": 0.6, "neutral": 0.6, "positive": 0.55 }
  }
}
```

---

### `POST /predict`

Accepts a single review or a batch. Language is optional.

**Request (single)**

```json
{ "text": "Pelayanannya lambat, tapi satenya enak.", "language": "id" }
```

**Request (batch)**

```json
[
  { "text": "Great pasta and friendly staff.", "language": "en" },
  { "text": "Makanan hambar dan mahal.", "language": "id" }
]
```

**Response**

```json
{
  "results": [
    {
      "text": "Pelayanannya lambat, tapi satenya enak.",
      "language": "id",
      "sentiment": "negative",
      "confidence": 0.71,
      "probs": { "negative": 0.71, "neutral": 0.08, "positive": 0.21 },
      "low_confidence": false,
      "advisory": ""
    }
  ],
  "disclaimer": "This model performs best on positive reviews. Neutral/negative predictions may be lower confidence due to class imbalance. Low-confidence responses include an advisory for human review."
}
```

**Notes**

* `low_confidence` is controlled via `serve.general_threshold` and `serve.per_class_thresholds` in `configs/config.yaml`.
* Per-class probabilities are always included for transparency.

---

### `POST /feedback`

Stores human-verified labels for continuous learning.

**Request**

```json
{
  "text": "Makanan hambar dan mahal.",
  "language": "id",
  "pred": "negative",
  "confidence": 0.71,
  "true_label": "negative",
  "notes": "Confirmed by QA"
}
```

**Response**

```json
{ "status": "ok" }
```

**Telemetry**

* Predictions → `outputs/api_logs/predictions.csv`
* Feedback → `outputs/api_logs/feedback.csv`

---

## 📂 Project Layout <a name="project-layout"></a>

```
├─ app/                    # Flask app (API + minimal UI)
│  └─ server.py
├─ configs/                # config.yaml (paths, thresholds, hyperparams)
├─ data/                   # ai/ (A..F CSVs), real/ (held-out real test)
│  ├─ ai/
│  │  ├─ model_A/
│  │  ├─ model_B/
│  │  ├─ model_C/
│  │  ├─ model_D/
│  │  ├─ model_E/
│  │  └─ model_F/
│  └─ real/
│     └─ real_reviews.csv
├─ outputs/                # splits, eval artifacts, api_logs
│  ├─ api_logs/
│  └─ eval/
├─ saved_models/           # best checkpoint + tokenizer
│  └─ best/
├─ src/
│  ├─ config.py
│  ├─ data/                # loading, cleaning, splits
│  ├─ evaluation/          # metrics, reports
│  ├─ inference/           # predict + serve wrapper
│  ├─ models/              # BertForSentiment
│  └─ training/            # train loop, custom checkpointing
├─ scripts/
│  ├─ run_api.sh
│  ├─ run_eval.sh
│  └─ run_train.sh
├─ tests/
│  └─ test_smoke.py
├─ .env.example
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

## 🔄 Using Your Own Data <a name="using-your-own-data"></a>

This section shows how to train the system **from scratch with your datasets**, keep it improving with **continuous learning**, and **customize** it for new domains, languages, or label schemes.

---

### Starting from Scratch

#### 1) Required CSV Format

Your training CSVs must contain at least these columns:

| column      | type   | allowed values / notes                       |
| ----------- | ------ | -------------------------------------------- |
| `text`      | string | The raw review text                          |
| `language`  | string | Language code (e.g., `en`, `id`, `es`, `ms`) |
| `sentiment` | string | One of `negative`, `neutral`, `positive`     |

> Optional columns (ignored by training): `number`, `restaurant_type`, `rating`, etc.

#### 2) Folder Structure for Training Data

Create one or more **training shards** under `data/ai/` (the pipeline globs all CSVs):

```
data/
└── ai/
    ├── model_A/your_dataset_a.csv
    ├── model_B/your_dataset_b.csv
    ├── model_C/your_dataset_c.csv
    ├── model_D/...
    ├── model_E/...
    └── model_F/...
```

Then build clean splits and train:

```bash
python -m src.data.datasets --summary
python -m src.training.train
python -m src.evaluation.evaluate --which both
```

> Keep **real-world test data** separate in `data/real/real_reviews.csv` (labels optional). The pipeline **never** uses real data for training.

#### 3) Adding New Languages (Beyond English/Indonesian)

The model uses **mBERT (cased)**, which supports 100+ languages out-of-the-box.

To enable new language codes:

1. **Update validation** list in `src/data/datasets.py`:

   ```python
   # ALLOWED_LANGS = {"en", "id"}  # old
   ALLOWED_LANGS = {"en", "id", "es", "ms"}  # add your codes
   ```

   *Tip:* You can also set `ALLOWED_LANGS = None` and skip filtering by language if you trust your data (advanced).
2. Ensure your CSVs use the same codes (e.g., `es` for Spanish).
3. Re-run the pipeline commands above.

> **Stratification:** We preserve both **language** and **sentiment** balance by stratifying on the combo `language__sentiment`. Adding languages works seamlessly if each language has examples for every class.

#### 4) Data Cleaning & Validation Requirements

The loader will:

* Drop rows missing `text`, `language`, or `sentiment`
* Normalize whitespace, redact URLs/emails (`<URL>`, `<EMAIL>`), and **demojize** emojis (e.g., `:thumbs_up:`)
* De-duplicate on `(text_clean, language, sentiment)`

**You should:**

* Keep texts in natural casing (we use a **cased** model)
* Maintain **class balance** per language when possible (target ~60/20/20 is fine)
* Avoid leaking **real test** samples into training folders
* If mixing multiple sources, prefer one CSV per source (traceability via `__source_file`)

---

### Continuous Learning Cycle

Use model telemetry to decide **what to add** and **when to retrain**.

#### 1) Incorporate Feedback into Retraining

1. Collect feedback via the UI or API:

   * Predictions → `outputs/api_logs/predictions.csv`
   * Human labels → `outputs/api_logs/feedback.csv`
2. Convert *feedback* to the standard training schema:

   * Create `data/ai/model_G/feedback_labeled.csv` with columns
     `text,language,sentiment`
3. Rebuild splits & retrain:

   ```bash
   python -m src.data.datasets --summary
   python -m src.training.train
   python -m src.evaluation.evaluate --which both
   ```
4. Compare new vs previous metrics (`outputs/eval/*_metrics.json`) and confusion matrices.

#### 2) Monitor & Decide When to Retrain

* **Macro-F1 on real data**: drop of >2–3 points vs previous release
* **Low-confidence rate** (from `predictions.csv`): sustained increase >10–15%
* **Error patterns**: recurring high-confidence mistakes in minority classes
* **Domain shifts**: menu changes, new slang/expressions, seasonal reviews

Recommended cadence: review weekly; retrain when two or more indicators are triggered.

#### 3) Data Quality Best Practices

* **Balance by class** (especially **neutral**/**negative**) and by **language**
* **De-duplicate** across all shards before splitting
* **Label consistency**: set a short guideline (what counts as neutral vs. mild positive/negative)
* **Leakage checks**: real data must remain out of training
* **Document provenance**: keep a simple CHANGELOG of added shards (A..G..H)

---

### Customization Options

#### 1) Configuration for Different Use Cases

Most settings live in `configs/config.yaml`. Common changes:

```yaml
model:
  pretrained_name: "bert-base-multilingual-cased"
  max_length: 192
  dropout: 0.1

train:
  epochs: 5
  batch_size: 16
  lr: 2e-5
  weight_decay: 0.01
  seed: 42

serve:
  general_threshold: 0.60
  per_class_thresholds:
    negative: 0.60
    neutral: 0.60
    positive: 0.55
  max_batch_size: 64
```

* **Latency vs. accuracy**: Decrease `max_length` to 160 for faster inference; increase for long reviews.
* **Throughput**: Adjust `max_batch_size` for your environment.

#### 2) Adding New Sentiment Classes (Advanced)

If you want a 4-class scheme (e.g., **very_negative**, **negative**, **neutral**, **positive**):

1. Update label definitions in `src/evaluation/metrics.py`:

   ```python
   LABELS = ["very_negative", "negative", "neutral", "positive"]
   ```
2. Update validation in `src/data/datasets.py`:

   ```python
   ALLOWED_LABELS = {"very_negative", "negative", "neutral", "positive"}
   ```
3. Update `configs/config.yaml` to set:

   ```yaml
   model:
     num_labels: 4
   serve:
     per_class_thresholds:
       very_negative: 0.60
       negative: 0.60
       neutral: 0.60
       positive: 0.55
   ```
4. **Retrain from scratch** (old checkpoints won’t match new label dimensions):

   ```bash
   python -m src.data.datasets --summary
   python -m src.training.train
   python -m src.evaluation.evaluate --which both
   ```

#### 3) Adjusting Confidence Thresholds per Domain

* Edit `serve.general_threshold` and `serve.per_class_thresholds` in `configs/config.yaml`.
* For domains with **risky negatives** (e.g., safety or compliance), **raise** `negative` threshold (e.g., 0.7–0.8) to route more cases to human review.
* Monitor the **low-confidence rate** in `outputs/api_logs/predictions.csv` to tune thresholds iteratively.

---

**Summary**

* Drop your CSVs into `data/ai/model_*`, ensure columns and language codes are correct, and run the pipeline.
* Use `/feedback` to capture human corrections and periodically retrain to improve minority classes.
* Customize labels, thresholds, and hyperparameters in `configs/config.yaml` to fit your domain and risk profile.

---

## 🧪 Local Tips & Troubleshooting <a name="local-tips--troubleshooting"></a>

- **Start the server as a module** 

  Always use:
  ```bash
  python -m app.server
  ```

This ensures the project root is on `sys.path` so `src` imports resolve correctly.

* **Alternative runners (e.g., Gunicorn)**

  ```bash
  export PYTHONPATH="$(pwd)"
  gunicorn -w 2 -b 0.0.0.0:8000 app.server:app
  ```

* **Performance on CPU**
  Reduce `train.batch_size` and/or `model.max_length` in `configs/config.yaml`. For faster evaluation, lower `serve.max_batch_size`.

* **Neutral underperforming**
  Collect more neutral examples via `/feedback`, increase epochs slightly (+1–2), and/or increase `model.max_length` if reviews are long.

* **Data leakage checks**
  Ensure real data remains in `data/real/` only. Training data must come from `data/ai/` shards.

---

## 📄 License <a name="license"></a>

**MIT** — see the [`LICENSE`](LICENSE) file for details.

---

## 👨‍💻 Contact <a name="contact"></a>

Maintainer: **Rahman**

* Email: **[arahmanwahid@outlook.com](mailto:arahmanwahid@outlook.com)**
* GitHub: **@swanframe**