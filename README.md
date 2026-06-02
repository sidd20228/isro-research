# AI-Native WAF Research Pipeline

Research-grade V1 pipeline for testing whether self-supervised HTTP security embeddings trained only on benign traffic improve zero-day web attack detection over classical anomaly-detection baselines.

The system deliberately avoids attack classification, signatures, session modeling, online learning, GNNs, Deep SVDD, and model fusion. Its core path is:

```text
HTTP request -> normalization -> field-aware text -> security embedding -> open-set detector -> anomaly score
```

## Repository Layout

```text
configs/                Experiment configuration
data/processed/         Generated unified splits and metadata
src/datasets/           Schema detection and CSV unification
src/normalization/      Deterministic request canonicalization
src/preprocessing/      Field-aware request representation
src/models/             TF-IDF, character CNN, and DistilBERT encoder
src/training/           Contrastive augmentations and training
src/detectors/          Mahalanobis, Isolation Forest, OCSVM, and flow detectors
src/evaluation/         Metrics, experiments, ablations, and latency suite
src/visualization/      ROC, PR, score-distribution, UMAP, and t-SNE figures
tests/                  Focused unit tests
```

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducible Workflow

Unify the heterogeneous raw CSV files into benign train/test splits and an attack-only test set:

```bash
python -m src.preprocessing.prepare_data --config configs/default.yaml
```

This writes:

```text
data/processed/train_benign.csv
data/processed/test_benign.csv
data/processed/test_attack.csv
data/processed/metadata.json
```

Normalize the unified splits and write auditable transformation statistics:

```bash
python -m src.normalization.normalize_data --config configs/default.yaml
```

This writes normalized CSV files and `data/normalized/metadata.json`. URL decoding is bounded, escaped characters are decoded, Unicode is normalized with NFKC, paths are lowercased, variable identifiers are replaced, query parameters are sorted deterministically, and excessive whitespace is collapsed. Query components are decoded independently so encoded delimiters do not alter parameter boundaries.

Render deterministic field-aware representations:

```bash
python -m src.preprocessing.render_representations --config configs/default.yaml
```

This writes streamed JSONL artifacts and metadata under:

```text
data/representations/request_time/
data/representations/offline_ablation/
```

The default `request_time` profile contains `METHOD`, `PATH`, and `BODY`. It excludes response-derived fields because a production WAF must score a request before upstream execution. The `offline_ablation` profile additionally contains `STATUS` and `RESPONSE_SIZE`; use it only to measure the effect of post-response context in offline experiments.

Run classical cross-application baselines and per-family zero-day slices:

```bash
python -m src.evaluation.run_baselines --config configs/default.yaml
```

Phase 4 consumes the materialized `request_time` JSONL profile, fits detectors only on benign training rows, evaluates only on held-out benign rows plus attack rows, calibrates thresholds from benign fit scores, saves model artifacts, and records CPU latency. TF-IDF Isolation Forest uses at most `100,000` fit rows and TF-IDF One-Class SVM uses at most `20,000` by default because kernel OCSVM scaling is substantially more expensive. The deterministic caps are configurable under `baselines.models`.

The command writes detailed JSON, flattened CSV, a readable Markdown summary, and persisted model artifacts under `reports/artifacts/phase4_baselines/`.

The character CNN implementation is available but disabled by default because it requires a longer CPU training run. Enable it explicitly with:

```bash
python -m src.evaluation.run_baselines \
  --config configs/default.yaml \
  --models character_cnn
```

Train the benign-only contrastive DistilBERT encoder:

```bash
python -m src.training.contrastive --config configs/default.yaml
```

Phase 5 reads the materialized `request_time` profile, trains only on benign requests, validates on held-out benign requests, and writes a resumable checkpoint after each epoch. CUDA mixed precision is enabled automatically when `training.device` is `cuda` and `training.mixed_precision` is `true`.

For a short Colab pilot:

```bash
python -m src.training.contrastive \
  --config configs/default.yaml \
  --output /content/drive/MyDrive/waf-results/distilbert_encoder_pilot \
  --device cuda \
  --epochs 1 \
  --max-train-samples 10000
```

For a full run, use `--max-train-samples 0`:

```bash
python -m src.training.contrastive \
  --config configs/default.yaml \
  --output /content/drive/MyDrive/waf-results/distilbert_encoder \
  --device cuda \
  --max-train-samples 0
```

Writing directly to mounted Google Drive ensures completed epochs survive a Colab disconnect. Resume from the latest checkpoint:

```bash
python -m src.training.contrastive \
  --config configs/default.yaml \
  --output /content/drive/MyDrive/waf-results/distilbert_encoder \
  --resume-from /content/drive/MyDrive/waf-results/distilbert_encoder/checkpoints/epoch_001.pt \
  --device cuda \
  --max-train-samples 0
```

Each output directory contains:

```text
backbone/                  Exportable HuggingFace encoder
projection.pt              Security-embedding projection
checkpoints/epoch_NNN.pt   Resumable training state
training_metadata.json     Losses, elapsed time, sample counts, and configuration
```

Evaluate embeddings with the configured open-set detector and generate figures:

```bash
python -m src.evaluation.run_embeddings --config configs/default.yaml
```

On a Colab T4, use a larger inference-only batch size and CUDA mixed precision:

```bash
python -m src.evaluation.run_embeddings \
  --config configs/default.yaml \
  --encoder /content/drive/MyDrive/waf-results/distilbert_encoder \
  --output /content/drive/MyDrive/waf-results/embedding_evaluation \
  --device cuda \
  --batch-size 128
```

Embedding evaluation logs per-split progress, throughput, detector fit timing, scoring timing, and plot timing. Increasing the inference batch size does not alter training or detector semantics. If GPU memory remains low, retry with `--batch-size 256`; reduce the value if CUDA reports an out-of-memory error.

Run CPU latency benchmarks. The TF-IDF baseline is always measured; trained CNN and DistilBERT artifacts are optional:

```bash
python -m src.evaluation.run_benchmarks \
  --config configs/default.yaml \
  --encoder reports/artifacts/distilbert_encoder
```

Materialize the controlled ablation matrix for scheduled training runs:

```bash
python -m src.evaluation.run_ablations
```

Evaluation commands write both detailed JSON records and flattened CSV comparison tables.

## Experiment Design

The baseline runner implements three application-level experiments:

| Experiment | Benign fit traffic | Attack evaluation traffic |
| --- | --- | --- |
| A | DVWA + Juice Shop | WebGoat |
| B | DVWA + WebGoat | Juice Shop |
| C | All applications | All attacks |

Attack-family evaluation is a test-time slice because representation learning and detector fitting are benign-only. When raw CSV files do not provide an attack-family column, the evaluator records a conservative heuristic family assignment and retains an `unknown` group. For publication results, manually curated family annotations should replace heuristic assignments.

## Research Notes

- `status_code` and `response_size` are standardized for analysis but excluded from the default request-time representation because they are unavailable before upstream execution.
- DistilBERT is intentionally encoder-only. The learned projection creates compact normalized embeddings for detector fitting.
- The normalizing-flow detector is a lightweight diagonal affine flow. It provides an exact likelihood baseline without adding architectural complexity before the core hypothesis is validated.
- Generated metrics are local JSON records under `reports/artifacts/`; no external tracking service is required.

## Tests

```bash
pytest
```
