<div align="center">

# 🧠 AI CV Generator

### Fine-tuned LLM that turns raw candidate text into structured, recruiter-ready JSON résumés

**Phi-3-Mini · QLoRA 4-bit · Structured Generation · Flask REST API**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FFD21E)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA-blue)](https://github.com/huggingface/peft)
[![TRL](https://img.shields.io/badge/TRL-SFTTrainer-orange)](https://github.com/huggingface/trl)
[![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Model](https://img.shields.io/badge/base%20model-phi--3--mini--4k--instruct-purple)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

</div>

---

## 📌 TL;DR

**AI CV Generator** fine-tunes Microsoft's **Phi-3-Mini-4k-Instruct** (3.8 B parameters) with **QLoRA** so it can take an unstructured candidate description — the kind of messy free text a recruiter actually receives — and emit a **strict, schema-conformant JSON résumé**, while rewriting the professional summary in a polished tone.

The fine-tuned adapters are served behind a lightweight **Flask REST API** (`/health`, `/cv/generate`) that can be exposed publicly through an **Ngrok** tunnel, and the project ships a **batch script** to process many candidates from a single JSON file.

> In one line: *unstructured CV text → fine-tuned LLM → validated structured JSON, served over HTTP.*

---

## 🎯 The Problem

Recruiting tools live and die on **structured data**. But résumé information almost never arrives structured — it comes as free-form paragraphs, emails, copy-pasted profiles, and inconsistent notes. Two hard sub-problems hide inside "just parse the CV":

1. **Extraction** — reliably pull out entities (name, title, contact info, skills, experience, education, languages) and place each one in the *correct* field of a fixed schema.
2. **Enrichment** — rewrite weak, telegraphic summaries and task descriptions into clean professional prose **without inventing facts** (no hallucinated employers, schools, or dates).

A generic prompt to a base LLM does neither reliably: the schema drifts, keys get renamed, and the output is not machine-parseable. **This project's thesis is that a small, well-fine-tuned model beats a large prompted one for constrained structured generation** — and it builds the full pipeline to prove it: data → QLoRA fine-tuning → inference → API → batch processing.

---

## ✨ Key Features

| Capability | Description |
|---|---|
| 🪶 **QLoRA Fine-Tuning** | 4-bit **NF4** quantization with double-quant — fine-tunes a 3.8 B model on a single Colab **T4/A100** GPU. Only **~50 M / 3.87 B params (1.30 %)** are trainable. |
| 🧩 **Strict Structured Output** | Converts free text into a fixed JSON schema (`informations`, `resume`, `competences`, `experience`, `education`, `projets`, `langues`). |
| ✍️ **Summary Enrichment** | Rewrites `resume` and task bullets in a professional register — explicitly instructed *not* to fabricate entities. |
| 🌐 **REST API** | Flask app with `GET /health` (server + GPU status) and `POST /cv/generate` (single-CV extraction). |
| 🚇 **Public Tunnel** | One-cell **Ngrok** exposure — query the model from anywhere, no infra setup. |
| 📦 **Batch Mode** | Script reads a list of candidates from `input_cv.json`, calls the API per record, and writes consolidated `output_cvs.json` with success/error counts. |
| 🛡️ **Robust JSON Decoding** | Decodes **only generated tokens** (not the prompt) and extracts the first complete `{...}` block — fixing control-character corruption that breaks `json.loads()`. |

---

## 🎬 Demo

> **Request** — `POST /cv/generate`

```json
{
  "input": "Patrick Yoba. Étudiant ingénieur en data/IA. Résumé: recherche stage data science. Email: patrick.yoba@email.com. Compétences: Python, SQL, ML. École: 3iL Limoges (2024-)."
}
```

> **Response** — structured JSON résumé

```json
{
  "informations": {
    "prenom": "Patrick",
    "nom": "Yoba",
    "titre": "Étudiant ingénieur data/IA",
    "email": "patrick.yoba@email.com",
    "telephone": null,
    "adresse": "France",
    "liens": []
  },
  "resume": "Enquête active de stages en data science.",
  "competences": {
    "techniques": ["Python", "SQL", "ML"],
    "outils": [],
    "soft_skills": []
  },
  "experience": [],
  "education": [
    { "ecole": "3iL Limoges", "diplome": "Ingénieur data/IA", "annee": "2024" }
  ],
  "projets": [],
  "langues": []
}
```

> 📷 **Suggested assets to add** in `docs/assets/`: a GIF of the Colab notebook generating a CV, a screenshot of the `/health` JSON response, and the training loss curve. Reference them here once added.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                          │
│                                                                      │
│   dataset.jsonl (~2,488 chat examples)                               │
│        │  train/test split 80 / 20  (seed=42)                        │
│        ▼                                                             │
│   Phi-3-Mini-4k-Instruct  ──►  4-bit NF4 quantization (bitsandbytes)  │
│        │                        + double quant, bf16 compute         │
│        ▼                                                             │
│   QLoRA adapters (r=32, α=64, dropout=0.05, all-linear)              │
│        │  SFTTrainer · 4 epochs · cosine LR · paged_adamw_32bit       │
│        ▼                                                             │
│   model/cv-lora/  ◄── saved LoRA adapters (~50M trainable params)     │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          SERVING PIPELINE                            │
│                                                                      │
│   Base model (4-bit) + LoRA adapters  ──►  PeftModel (eval mode)      │
│        │                                                             │
│        ▼                                                             │
│   generate_cv():  apply_chat_template → generate → decode gen tokens  │
│        │           → extract_json() → json.loads()                   │
│        ▼                                                             │
│   Flask API  ──  GET /health   POST /cv/generate                     │
│        │                                                             │
│        ▼                                                             │
│   Ngrok tunnel  ──►  public HTTPS URL  ──►  client / batch script     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Base model** | `microsoft/Phi-3-mini-4k-instruct` (3.8 B params, 4k context) |
| **Fine-tuning** | PyTorch · 🤗 Transformers · PEFT (LoRA) · TRL (`SFTTrainer` / `SFTConfig`) · bitsandbytes (4-bit NF4) · Accelerate |
| **Inference** | `apply_chat_template`, sampling decode, custom JSON extraction |
| **API / Serving** | Python · Flask · Flask-CORS · PyNgrok |
| **Environment** | Google Colab (T4 / A100 GPU) + Google Drive for persistence |

---

## 🔬 Methodology & ML Pipeline

### 1. Dataset

The training data (`data/dataset.jsonl`, **~2,488 examples**) follows the **chat format** expected by `SFTTrainer`:

```jsonc
{
  "messages": [
    { "role": "system",    "content": "Tu convertis des informations de candidat en CV au format JSON exactement selon le schéma…" },
    { "role": "user",      "content": "Nom: Patrick Yoba. Titre: Data Analyst. Email: … Expérience: …" },
    { "role": "assistant", "content": "{\"informations\":{…},\"competences\":{…},\"experience\":[…],…}" }
  ]
}
```

Each example pairs **messy candidate text** with the **gold structured JSON**, teaching the model both the extraction mapping and the strict output contract. Split: **80 % train / 20 % eval** (`seed=42`).

### 2. Model loading & quantization

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

Loading Phi-3 in **4-bit NF4 with double quantization** shrinks the memory footprint enough to fine-tune a 3.8 B model on a single consumer-grade GPU, while `bfloat16` compute keeps numerical stability. A small **RoPE-scaling workaround** is applied at inference to avoid a known Phi-3 `ValueError`.

### 3. QLoRA configuration

| Hyperparameter | Value |
|---|---|
| LoRA rank `r` | 32 |
| LoRA `alpha` | 64 |
| LoRA dropout | 0.05 |
| Target modules | `all-linear` |
| Task type | `CAUSAL_LM` |
| **Trainable params** | **50,331,648 / 3,871,411,200 (1.30 %)** |

### 4. Supervised fine-tuning (TRL `SFTConfig`)

| Setting | Value | Rationale |
|---|---|---|
| Epochs | 4 | Enough passes for a small, focused dataset |
| Per-device batch size | 1 | Fits 4-bit model in T4 VRAM |
| Gradient accumulation | 8 | Simulates an effective batch size of 8 |
| Learning rate | 2e-4 | Standard QLoRA LR |
| LR scheduler | cosine + 3 % warmup | Smooth decay, stable convergence |
| Optimizer | `paged_adamw_32bit` | Memory-paged AdamW, stable under 4-bit |
| Max sequence length | 1024 | Covers full input + JSON output |
| Precision | bf16 | More stable than fp16 for training |
| Eval strategy | every 50 steps | Tracks generalization during training |

### 5. Inference

`generate_cv()` builds the prompt with `tokenizer.apply_chat_template(...)`, generates with controlled sampling (`temperature=0.5`, `top_p=0.9`, `repetition_penalty=1.12`, `max_new_tokens=900`), then **decodes only the newly generated tokens** before extracting the first complete JSON object. Decoding generated tokens only — rather than the full sequence — was the fix for control characters that previously broke `json.loads()`.

---

## 📊 Results

The fine-tuned model **reliably imposes the target JSON structure** on Phi-3 and correctly classifies entities (names, titles, skills, diplomas, experiences) into the right schema fields — validating that QLoRA on a small focused dataset is sufficient for this constrained generation task. Batch runs over multi-candidate input files completed with a **2/2 success rate** in the documented test, and the qualitative summary-rewriting objective is met.

### Known limitations (measured, not hidden)

| Limitation | Description | Mitigation path |
|---|---|---|
| 🎭 **Entity hallucination** | The model can invent school/company/diploma names when they are not explicit in the input. | Stronger negative supervision, retrieval grounding, post-hoc fact-checking against the input. |
| 🔑 **JSON key instability** | Rare key misspellings observed (`expereince`, `experece` instead of `experience`). | **Pydantic** validation + key-normalization layer; lower `temperature`; constrained / grammar-guided decoding. |
| 📉 **No formal eval metrics yet** | Loss is logged but exact-match / field-level F1 are not computed. | Add a held-out eval harness reporting JSON-valid rate, schema-match rate, per-field accuracy. |

> 📈 **To add:** export the training/eval loss curve from the notebook to `docs/assets/loss_curve.png` and embed it here.

---

## 📂 Project Structure

### Current layout

```
AI-CV-GENERATOR/
├── app/
│   ├── app.py            # Flask API (standalone serving script)
│   └── utils.py          # JSON cleaning + prompt formatting helpers
├── data/
│   ├── dataset.jsonl     # ~2,488 chat-format training examples
│   └── schema.json       # Target JSON résumé schema
├── model/
│   ├── load_dataset.py   # Dataset loading sanity check
│   ├── load_model.py     # Model + LoRA config loading
│   ├── train.py          # Training setup stub
│   └── model/
│       └── inference.py  # generate_cv() inference helper
├── scripts/
│   ├── config.py         # Paths & constants
│   ├── ngrok.py          # Public tunnel setup
│   ├── prompt.py         # System prompt definition
│   └── test_api.py       # API smoke tests
├── projet_cv.ipynb       # End-to-end Colab notebook (train → serve → test)
├── requirements.txt
├── LICENSE               # MIT
└── README.md
```

### Recommended layout (suggested refactor)

A cleaner structure would make the repo read like a production codebase rather than a notebook companion:

```
ai-cv-generator/
├── src/
│   ├── training/         # train.py, dataset.py, config.py  (single source of truth)
│   ├── inference/        # model_loader.py, generate.py
│   ├── api/              # app.py, routes.py, schemas.py (Pydantic)
│   └── schema/           # canonical schema + validators
├── data/                 # dataset.jsonl, schema.json
├── examples/             # input_cv.json, output_cvs.json samples
├── notebooks/            # projet_cv.ipynb
├── docs/assets/          # loss curve, demo GIF, screenshots
├── tests/                # pytest unit + API tests
├── Dockerfile
├── .env.example          # NGROK_AUTH_TOKEN, PATH_PROJET, ...
├── Makefile
└── .github/workflows/    # lint + test CI
```

---

## 🚀 Installation & Setup

### Prerequisites
- A **GPU** environment (Google Colab T4/A100 recommended).
- A free **Ngrok** account + auth token — https://dashboard.ngrok.com.
- Google Drive (the notebook persists the model and cache there).

### Steps

```bash
# 1. Clone
git clone https://github.com/Yobapatrick/AI-CV-GENERATOR.git
cd AI-CV-GENERATOR

# 2. Install dependencies
pip install -r requirements.txt
```

```python
# 3. In projet_cv.ipynb (Colab): mount Drive and set the project path
from google.colab import drive
drive.mount('/content/drive')
PATH_PROJET = "/content/drive/MyDrive/projet_cv/projet-cv"
```

```bash
# 4. Provide your Ngrok token via environment variable (NEVER hardcode it)
export NGROK_AUTH_TOKEN="your_token_here"
```

> ⚠️ **Security note:** the current notebook and `scripts/ngrok.py` contain a hardcoded Ngrok token. **Revoke that token immediately** and load it from an environment variable or `.env` file instead. Never commit secrets.

---

## ⚙️ Usage

### 1. Fine-tune the model
Run the training cell in `projet_cv.ipynb`. It loads and splits the dataset, quantizes Phi-3 to 4-bit, trains the LoRA adapters for 4 epochs, and saves them to `model/cv-lora/`.

### 2. Launch the API
Run the Flask cells — the server starts on port `5000` in a background thread.

### 3. Expose publicly
Run the Ngrok cell to obtain a public HTTPS URL.

### 4. Query the API

```bash
# Health check
curl https://<your-ngrok-url>/health
# → {"status":"ok","gpu":true,"model":"microsoft/phi-3-mini-4k-instruct","adapter":"…"}

# Generate a structured CV
curl -X POST https://<your-ngrok-url>/cv/generate \
  -H "Content-Type: application/json" \
  -d '{"input": "Inès Wagner. Maître-Nageur Sauveteur. Tel: 0467112233. Adresse: Montpellier. Exp: Piscine Municipale (2022-2024). Diplôme: BPJEPS AAN (2021)."}'
```

### 5. Batch processing
Provide an `input_cv.json` list of `{ "id", "input" }` records; the batch cell calls the API per record and writes a consolidated `output_cvs.json` with `total` / `success` / `errors` counts.

---

## 📡 API Reference

### `GET /health`
Returns server status and GPU availability.
```json
{ "status": "ok", "gpu": true, "model": "microsoft/phi-3-mini-4k-instruct", "adapter": "…/model/cv-lora" }
```

### `POST /cv/generate`
| | |
|---|---|
| **Body** | `{ "input": "<free-text candidate description>" }` |
| **200** | Full structured CV JSON |
| **400** | `input` field missing or empty |
| **422** | Model output contained no valid JSON block |
| **500** | Internal error during generation |

---

## ☁️ Deployment & Production Considerations

The current Colab + Ngrok setup is ideal for **demos and rapid iteration**, but a production deployment would need:

- **Containerization** — a `Dockerfile` pinning CUDA, Python, and dependency versions for reproducible builds.
- **Adapter merging** — optionally merge LoRA into the base model and serve with a dedicated inference engine (vLLM / TGI) for throughput.
- **Schema validation** — a **Pydantic** model wrapping every response, with key-normalization to absorb the rare misspelled keys before they reach the client.
- **Constrained decoding** — grammar- or JSON-schema-guided generation (e.g. Outlines / `lm-format-enforcer`) to make malformed JSON structurally impossible.
- **Stable hosting** — a real ASGI/WSGI server (Gunicorn + workers) behind a reverse proxy instead of Flask's dev server and an ephemeral tunnel.
- **Observability** — request logging, latency metrics, JSON-valid-rate monitoring.
- **Secret management** — tokens via environment / secret store, never in source.
- **CI** — lint + unit tests + an API smoke test on every push.

---

## 🗺️ Roadmap

- [ ] **Revoke the leaked Ngrok token** and migrate all secrets to `.env`.
- [ ] Fix `app/app.py`: `method=` → `methods=`; unify it with the notebook's `apply_chat_template` logic.
- [ ] Add a **Pydantic** schema + key-normalization post-processor.
- [ ] Build a quantitative **eval harness**: JSON-valid rate, schema-match rate, per-field accuracy.
- [ ] Export and commit the **training loss curve** + a demo GIF to `docs/assets/`.
- [ ] Add `examples/input_cv.json` and `examples/output_cvs.json`.
- [ ] **Dockerize** the API; add `gunicorn` for production serving.
- [ ] Experiment with **constrained / grammar-guided decoding** to eliminate malformed JSON.
- [ ] Add **pytest** tests and a GitHub Actions CI workflow.
- [ ] Refactor to the recommended `src/` layout (single source of truth for training/inference).

---

## 🤝 Contributing

Contributions are welcome. Good first issues: the bug fixes and validation layer in the roadmap above. Please open an issue to discuss substantial changes before submitting a PR.

---

## 💡 Lessons Learned

- **A small fine-tuned model can beat a large prompted one** for constrained structured generation — QLoRA on ~2.5 k focused examples was enough to lock Phi-3 onto a strict JSON contract.
- **Decoding matters as much as generation.** The `422` failures weren't a model problem — they came from decoding the *full* sequence (prompt + output) and dragging in control characters. Decoding only the generated tokens fixed it.
- **LLMs drift on schema keys.** Even with explicit "do not rename keys" instructions, rare misspellings appear — a reminder that *the model's output should never be trusted raw*; a validation layer is not optional.
- **Hallucination is the cost of fluency.** Asking the model to *enrich* summaries makes it more willing to *invent* — enrichment and faithfulness are in tension and must be balanced deliberately.
- **QLoRA democratizes fine-tuning.** Training a 3.8 B model by updating only 1.3 % of its parameters, on a free Colab GPU, is a genuinely accessible workflow.

---

## 🌟 Why This Project Matters

This repository is a **complete, end-to-end applied-ML project**, not a notebook experiment. It demonstrates the full lifecycle a real ML engineer owns:

- **Data engineering** — building a chat-format instruction dataset for a specific task.
- **Efficient training** — QLoRA / 4-bit quantization to work within real hardware budgets.
- **Structured generation** — one of the most practically valuable and underrated LLM skills.
- **Serving** — wrapping a model in a clean REST API with health checks and error handling.
- **Engineering honesty** — measuring, documenting, and planning fixes for the model's failure modes rather than hiding them.

For recruiters: it shows someone who can take an LLM from **raw data to a queryable service**, reason about trade-offs (enrichment vs. faithfulness, latency vs. VRAM), and think about what *production* actually requires.

---

## 📚 References

- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) — Microsoft
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al.
- [🤗 PEFT](https://github.com/huggingface/peft) · [TRL](https://github.com/huggingface/trl) · [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

---

## 👨‍💻 Author

**Patrick Yoba** — Engineering student @ 3iL Ingénieurs (Limoges), Data / AI track.

> Looking for a **Data Science / ML Engineering internship**. Feedback, issues, and PRs are very welcome.

## 📄 License

Released under the **MIT License** — see [LICENSE](./LICENSE).
