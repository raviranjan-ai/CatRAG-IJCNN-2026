# Llama‑3 BBQ Gender: Accuracy & Bias Score (CatRAG-style)

This repository contains a single script, **`Llama-3.py`**, that evaluates a small Llama‑3 style model on the BBQ‑Gender subset and reports **Accuracy** and a **BBQ‑style Bias Score** under four settings:
- Retrieval‑Augmented (**RAG**) ON/OFF
- Functor‑style **projection** (debiasing of the input embedding layer) ON/OFF

It mirrors the metric and evaluation style used in the *A Multi‑LLM Debiasing Framework* and BBQ papers.

---

## 1) What it does

- Loads an instruction‑tuned model (default: `meta-llama/Llama-3.2-1B-Instruct`) from Hugging Face.
- Optionally applies a **linear projection** to the input embedding layer to reduce demographic directions while preserving occupational semantics.
- Optionally uses a **tiny TF‑IDF RAG** over a text file to attach 2–3 most relevant context chunks to each question.
- Runs through a BBQ‑Gender JSON file of `{ "prompt": ..., "answer": ... }` items.
- Prints a small table of **accuracy** and **bias score** for each (RAG, Projection) setting.

---

## 2) Requirements

> Tested with Python 3.10+

Install the pinned dependencies (CUDA 12.1 wheels shown; use CPU wheels if you don’t have CUDA):

```bash
pip install "torch==2.3.1" --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.44.2" "accelerate==0.34.2" "scikit-learn==1.5.1" "sentencepiece==0.2.0"
```

### Hugging Face access (Llama weights)
- You need a Hugging Face token with access to Meta Llama models.
- Log in once on the machine where you run the script:
  ```bash
  huggingface-cli login
  ```
- Make sure you’ve accepted the model license on the model card page before running.

> **CPU only?** It works on CPU too (slower). The script auto‑selects `cuda` if available, otherwise `cpu`.

---

## 3) Files & expected layout

By default the script expects **two data files**. You have two ways to provide them.

### Option A — Use the original online links
Edit the top of `Llama-3.py` and set:
```python
DATA_URL = "/data/bias/bbq_gender.json"
RAG_URL  = "/data/bbq_rag_content.txt"
```

### Option B — Local files
1) Put the files on disk, e.g.:
```
/data/bbq_gender.json
/data/bbq_rag_content.txt
```
2) Then set **file URLs** (note the `file://` prefix is required by `urllib`):
```python
DATA_URL = "file:///data/bbq_gender.json"
RAG_URL  = "file:///data/bbq_rag_content.txt"
```
> If you set plain paths like `/data/bbq_gender.json` without `file://`, `urllib.request.urlopen` will raise an error. Use `file://` or switch to opening those files with standard Python I/O.

**Dataset format**
Each line/item should be a dict with the keys:
```json
{ "prompt": "Question text with (A) ... (B) ... (C) ...", "answer": "the correct option text" }
```

---

## 4) Configuration knobs (edit at the top of the script)

```python
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATA_URL = "..."
RAG_URL  = "..."

LAMBDA = 0.2     # projection trade-off (demographic invariance vs. occupation preservation)
DU_FRAC = 1.0    # fraction of eigenvectors kept for the projector (1.0 = full-dim reweighting)
APPLY_PROJECTION = True   # turn debias projection ON/OFF
USE_RAG = True            # turn RAG ON/OFF
MAX_EXAMPLES = 200        # limit examples for quick smoke tests; set None for all
SEED = 42                 # reproducibility
```

- **APPLY_PROJECTION** controls the functor‑style debiasing projector on the input embeddings.
- **USE_RAG** toggles retrieval of 2–3 small evidence chunks per query from `RAG_URL`.
- **LAMBDA** controls the fairness–utility trade‑off in the projection objective.
- **DU_FRAC** sets how many eigenvectors span the projection subspace (1.0 ≈ no dimensionality drop).

---

## 5) How to run

From the folder containing `Llama-3.py`:

```bash
python Llama-3.py
```

You will see output like:

```
==== Accuracy & Bias Score (BBQ-style) ====
RAG=ON  | Proj=OFF | acc=0.xxx | bias=0.xxx | m=... (biased=..., anti=...)
RAG=ON  | Proj=ON  | acc=0.xxx | bias=0.xxx | m=... (biased=..., anti=...)
RAG=OFF | Proj=OFF | acc=0.xxx | bias=0.xxx | m=... (biased=..., anti=...)
RAG=OFF | Proj=ON  | acc=0.xxx | bias=0.xxx | m=... (biased=..., anti=...)
```

Interpretation:
- **acc** = proportion of exact matches to the gold **answer**.
- **bias** = BBQ‑style score based on the fraction of non‑unknown predictions that align with stereotype vs. anti‑stereotype (closer to **0** is better).
- **m** = number of non‑unknown predictions used for bias calculation; also shows counts of **biased** and **anti** selections.

---

## 6) Tips & troubleshooting

- **Model access denied**: accept the model license on Hugging Face and run `huggingface-cli login`.
- **CUDA out of memory**: run on CPU by setting `CUDA_VISIBLE_DEVICES=""` or using a smaller model ID.
- **`urllib` file errors**: when using local files, ensure the `file:///absolute/path` scheme.
- **Slow generation**: increase `MAX_EXAMPLES` gradually; start with `50` or `100` to sanity check.
- **Projection off for ablations**: set `APPLY_PROJECTION=False` (keeps all other steps the same).
- **RAG off for ablations**: set `USE_RAG=False` (measures the pure effect of projection).

---

## 7) Reproducibility

- The script fixes NumPy and PyTorch seeds (`SEED = 42`).
- Inference still has minor non‑determinism on GPU; for strict determinism, set:
  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
  (Add near the top of the script if needed.)

---

## 8) Citing BBQ and related work

- Parrish et al., 2022 — BBQ: A hand‑built bias benchmark for QA.
- Owens et al., 2024 — A Multi‑LLM Debiasing Framework (bias metric flavor).
- The script also implements a functor‑style projection similar in spirit to contextual embedding debiasing.

---

## 9) Quick checklist

- [ ] Dependencies installed
- [ ] Hugging Face logged in (and license accepted)
- [ ] `DATA_URL` and `RAG_URL` point to valid **http(s)** or **file://** locations
- [ ] Optional: set `MAX_EXAMPLES=None` for full runs
- [ ] Run `python Llama-3.py` and record the 4‑row table
