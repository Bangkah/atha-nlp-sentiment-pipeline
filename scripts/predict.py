from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@lru_cache(maxsize=4)
def _load_model(model_dir: str) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict(text: str, model_dir: str) -> tuple[str, float]:
    if not Path(model_dir).exists():
        raise FileNotFoundError(
            "Model belum ada. Jalankan dulu: python scripts/train.py"
        )

    tokenizer, model = _load_model(model_dir)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        label_id = int(torch.argmax(probs).item())
        score = float(probs[label_id].item())

    label_name = model.config.id2label.get(label_id, str(label_id)).lower()
    return label_name, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Kalimat yang ingin diprediksi")
    parser.add_argument("--model-dir", default="artifacts/model")
    args = parser.parse_args()

    label, confidence = predict(args.text, args.model_dir)
    print({"label": label, "confidence": round(confidence, 4)})


if __name__ == "__main__":
    main()
