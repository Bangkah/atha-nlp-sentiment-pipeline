# Atha NLP Sentiment Pipeline

Pipeline NLP end-to-end untuk klasifikasi sentimen Bahasa Indonesia (negative, neutral, positive), dari data preparation, training, evaluasi, hingga publikasi model dan demo.

## Result Highlight

- 3 kelas sentimen: `negative`, `neutral`, `positive`
- Model base: `indobenchmark/indobert-base-p1`
- Dataset publik: 1800 sampel (1500 train, 300 validation)
- Metrik validasi terbaru (synthetic dataset): Accuracy `1.0000`, Macro F1 `1.0000`
- Delivery saat ini: Model Hub + Dataset Hub + Space Demo + API lokal

## Tech Stack

- Model training: PyTorch, Hugging Face Transformers, Datasets, Evaluate
- Data processing: Pandas, NumPy
- API serving: FastAPI, Uvicorn
- Interactive demo: Gradio
- MLOps and hosting: Hugging Face Hub (Model, Dataset, Spaces)
- CI workflow: GitHub Actions

## Architecture

```text
data/raw/*.csv (train, valid)
	|
	v
scripts/train.py
  - tokenize
  - fine-tune IndoBERT
  - evaluate (accuracy, macro-f1)
  - generate metrics + error analysis
	|
	v
artifacts/model/
  - model weights + tokenizer
  - metrics_summary.txt
  - error_analysis.{csv,md}
	|
	+--> Hugging Face Model Hub
	+--> FastAPI (api.py)
	+--> Gradio Demo (app.py / space_demo/)

data/published/atha_text_dataset.csv
	|
	v
Hugging Face Dataset Hub
```

## Struktur

- `scripts/create_dataset.py`: Membuat dataset CSV contoh.
- `scripts/train.py`: Training model klasifikasi teks.
- `scripts/predict.py`: Inference dari model yang sudah dilatih.
- `app.py`: Demo Gradio lokal.
- `api.py`: API backend FastAPI.
- `space_demo/`: Source code demo untuk Hugging Face Spaces.
- `tests/`: Unit test ringan untuk kontrak repository.
- `.github/workflows/ci.yml`: Pipeline CI (lint + test) untuk push/PR.
- `Makefile`: Shortcut command untuk local workflow.
- `requirements.txt`: Dependency Python.

## Quick Start

1. Install dependency:

```bash
pip install -r requirements.txt
```

2. Buat dataset:

```bash
python scripts/create_dataset.py
```

3. Training model:

```bash
python scripts/train.py
```

4. Prediksi contoh:

```bash
python scripts/predict.py --text "produk ini bagus sekali"
```

Output label sekarang: `negative`, `neutral`, atau `positive`.

5. Login ke Hugging Face (sekali saja):

```bash
hf auth login
```

6. Training + push ke Hugging Face Hub:

```bash
python scripts/train.py --push-to-hub --repo-id Bangkah/atha-text-classifier
```

Training dari CSV real (custom path):

```bash
python scripts/train.py --train-file data/my_train.csv --valid-file data/my_valid.csv --epochs 2
```

Opsional override base model:

```bash
python scripts/train.py --model-name indobenchmark/indobert-base-p1 --epochs 2
```

Format CSV wajib:

```csv
text,label
produk ini bagus banget,2
biasa aja sih,1
gak worth it,0
```

Keterangan label: `0=negative`, `1=neutral`, `2=positive`.

Opsional private repo:

```bash
python scripts/train.py --push-to-hub --repo-id Bangkah/atha-text-classifier --private-repo
```

Rekomendasi:

- Untuk portfolio, gunakan mode public (tanpa `--private-repo`).
- Gunakan `--private-repo` jika model belum siap dipublikasikan atau data sensitif.

7. Jalankan demo Gradio lokal:

```bash
python app.py
```

8. Jalankan API backend (FastAPI, local):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Konfigurasi API key dan rate limit:

- `API_KEYS_JSON`: konfigurasi multi-key dalam format JSON.
- `API_KEY`: fallback legacy single-key (opsional).
- `RATE_LIMIT_MAX_REQUESTS`: default limit per menit jika `rpm` tidak ditentukan.
- `ADMIN_API_KEY`: key admin untuk endpoint rekap penggunaan.
- `USAGE_DB_PATH`: path SQLite untuk usage log.

Contoh konfigurasi (PowerShell):

```powershell
$env:API_KEYS_JSON='{"demo-key":{"owner":"demo","rpm":30}}'
$env:ADMIN_API_KEY='admin-secret'
$env:USAGE_DB_PATH='artifacts/usage/usage.db'
uvicorn api:app --host 0.0.0.0 --port 8000
```

Contoh request prediksi:

```bash
curl -X POST http://127.0.0.1:8000/predict \
	-H "Content-Type: application/json" \
	-H "X-API-Key: demo-key" \
	-d "{\"text\":\"produk ini biasa saja\"}"
```

Contoh request usage summary (admin):

```bash
curl "http://127.0.0.1:8000/usage/summary" \
  -H "X-Admin-Key: admin-secret"
```

## API Reference (Ringkas)

- `GET /health` -> cek status service.
- `POST /predict` -> prediksi sentimen, butuh `X-API-Key` jika key config aktif.
- `GET /usage/summary` -> ringkasan usage per owner, butuh `X-Admin-Key`.

Status code utama:

- `200`: request sukses.
- `400`: input tidak valid (misalnya teks kosong).
- `401`: API key/admin key tidak valid.
- `429`: rate limit terlampaui.
- `503`: model artifact belum siap.
- `500`: error internal saat inferensi.

## Docker (Opsional)

Build image:

```bash
docker build -t atha-text-classifier:latest .
```

Run container:

```bash
docker run --rm -p 8000:8000 \
	-v "$(pwd)/artifacts:/app/artifacts" \
	-e API_KEYS_JSON='{"demo-key":{"owner":"demo","rpm":30}}' \
	-e ADMIN_API_KEY='admin-secret' \
	-e USAGE_DB_PATH='artifacts/usage/usage.db' \
	atha-text-classifier:latest
```

Untuk PowerShell (Windows):

```powershell
docker run --rm -p 8000:8000 `
  -v "${PWD}/artifacts:/app/artifacts" `
  -e API_KEYS_JSON='{"demo-key":{"owner":"demo","rpm":30}}' `
  -e ADMIN_API_KEY='admin-secret' `
  -e USAGE_DB_PATH='artifacts/usage/usage.db' `
  atha-text-classifier:latest
```

Gunakan secret acak yang kuat; jangan pakai contoh key di atas untuk production.

Alternatif dengan Docker Compose:

```bash
cp .env.example .env
docker compose up --build
```

Cek health container:

```bash
docker compose ps
```

Edit nilai environment di `docker-compose.yml` sebelum dipakai di environment selain local.
Edit nilai secret di `.env` sebelum menjalankan service.

Preflight sebelum run:

- Pastikan model sudah ada di `artifacts/model` (jalankan training minimal sekali di host).
- Pastikan volume mount `artifacts:/app/artifacts` aktif agar API bisa membaca model.

## Next Phase

- Deploy API ke platform cloud (Render/Railway/Fly) setelah local smoke test stabil.
- Tambah observability (structured log, dashboard latency, alert error-rate).
- Tambah test API otomatis di CI untuk auth, rate-limit, dan error-path.
- Dockerization sebagai opsi packaging saat deployment target sudah dipilih.

## Catatan

- Dataset default disimpan di `data/raw/train.csv` dan `data/raw/valid.csv`.
- Model hasil training disimpan di `artifacts/model`.
- Default base model: `indobenchmark/indobert-base-p1`.
- Label mapping: `0=negative`, `1=neutral`, `2=positive`.
- Default ukuran data sintetis: 1500 train + 300 validation.
- Data ini masih sintetis (untuk portfolio pipeline), disarankan lanjut fine-tune dengan data real.
- Ringkasan evaluasi tambahan otomatis di `artifacts/model/metrics_summary.txt`.
- Error analysis otomatis di `artifacts/model/error_analysis.csv` dan `artifacts/model/error_analysis.md`.
- Metrik saat ini sangat tinggi karena data masih sintetis; gunakan data real untuk evaluasi produksi.
- Usage logs API disimpan di `artifacts/usage/usage.db` (jika fitur logging diaktifkan).

## Output Artifacts

- `artifacts/model/metrics_summary.txt`: ringkasan metrik + confusion matrix + classification report.
- `artifacts/model/error_analysis.csv`: daftar sampel validation yang salah prediksi.
- `artifacts/model/error_analysis.md`: ringkasan human-readable untuk debugging model.

## Development

- Panduan kontribusi: `CONTRIBUTING.md`
- Lisensi repository: `LICENSE`
- Quality check lokal:

```bash
make lint
make test
```

## Published Links

- Model: https://huggingface.co/Bangkah/atha-text-classifier
- Dataset: https://huggingface.co/datasets/Bangkah/atha-text-dataset
- Space Demo: https://huggingface.co/spaces/Bangkah/atha-text-demo
