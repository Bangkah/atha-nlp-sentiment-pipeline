---
language:
- id
tags:
- sentiment-analysis
- indonesian
- text-classification
pretty_name: Atha Text Dataset
size_categories:
- 1K<n<10K
task_categories:
- text-classification
---

# Atha Text Dataset

Dataset klasifikasi sentimen Bahasa Indonesia untuk eksperimen NLP dengan 3 kelas sentimen.

Dataset ini ditujukan untuk pembelajaran pipeline dan baseline eksperimen, bukan benchmark produksi.

Repository dataset: https://huggingface.co/datasets/Bangkah/atha-text-dataset

## Struktur Kolom

- text: Kalimat ulasan
- label: Label numerik (0 = negative, 1 = neutral, 2 = positive)
- split: train atau validation
- label_text: Label teks (negative/neutral/positive)

## Distribusi Data

- Total: 1800 baris
- Train: 1500 baris
- Validation: 300 baris

Setiap split memiliki distribusi kelas seimbang.

## Example Row

```csv
text,label,split,label_text
"produk ini bagus banget",2,train,positive
```

## Sumber

Dataset sintetis yang dibuat untuk latihan pipeline AI end-to-end.

## Catatan Penggunaan

- Gunakan dataset ini sebagai seed dataset untuk prototyping.
- Untuk hasil produksi, gabungkan dengan data real dari domain aplikasi.
- Lakukan evaluasi ulang di data out-of-distribution sebelum deployment.
