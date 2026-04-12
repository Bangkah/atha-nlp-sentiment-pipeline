# AI Model Starter (Dataset -> Training -> Inference)

Project ini menyiapkan alur sederhana untuk:
1. Membuat dataset teks klasifikasi otomatis.
2. Training model Transformer (Hugging Face + PyTorch).
3. Menjalankan prediksi dari model hasil training.
4. Menjalankan demo web dengan Gradio.
5. Menyediakan API inference dengan FastAPI.

## Struktur

- `scripts/create_dataset.py`: Membuat dataset CSV contoh.
- `scripts/train.py`: Training model klasifikasi teks.
- `scripts/predict.py`: Inference dari model yang sudah dilatih.
- `app.py`: Demo Gradio lokal.
- `api.py`: API backend FastAPI.
- `space_demo/`: Source code demo untuk Hugging Face Spaces.
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
python scripts/train.py --push-to-hub --repo-id USERNAME/atha-text-classifier
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
python scripts/train.py --push-to-hub --repo-id USERNAME/atha-text-classifier --private-repo
```

7. Jalankan demo Gradio lokal:

```bash
python app.py
```

8. Jalankan API backend (FastAPI):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Opsional keamanan API:

- Set environment variable `API_KEY` untuk mewajibkan header `X-API-Key`.
- Rate limit default: 30 request per 60 detik per IP.

Contoh request prediksi:

```bash
curl -X POST http://127.0.0.1:8000/predict \
	-H "Content-Type: application/json" \
	-H "X-API-Key: YOUR_API_KEY" \
	-d "{\"text\":\"produk ini biasa saja\"}"
```

## Catatan

- Dataset default disimpan di `data/raw/train.csv` dan `data/raw/valid.csv`.
- Model hasil training disimpan di `artifacts/model`.
- Default base model: `indobenchmark/indobert-base-p1`.
- Label mapping: `0=negative`, `1=neutral`, `2=positive`.
- Default ukuran data sintetis: 1500 train + 300 validation.
- Data ini masih sintetis (untuk portfolio pipeline), disarankan lanjut fine-tune dengan data real.
- Ringkasan evaluasi tambahan otomatis di `artifacts/model/metrics_summary.txt`.
- Error analysis otomatis di `artifacts/model/error_analysis.csv` dan `artifacts/model/error_analysis.md`.

## Output Artifacts

- `artifacts/model/metrics_summary.txt`: ringkasan metrik + confusion matrix + classification report.
- `artifacts/model/error_analysis.csv`: daftar sampel validation yang salah prediksi.
- `artifacts/model/error_analysis.md`: ringkasan human-readable untuk debugging model.

## Published Links

- Model: https://huggingface.co/Bangkah/atha-text-classifier
- Dataset: https://huggingface.co/datasets/Bangkah/atha-text-dataset
- Space Demo: https://huggingface.co/spaces/Bangkah/atha-text-demo
