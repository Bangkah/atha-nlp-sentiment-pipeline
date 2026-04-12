from __future__ import annotations

import gradio as gr
from transformers import pipeline

MODEL_ID = "Bangkah/atha-text-classifier"
classifier = pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID)

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


def infer(text: str) -> dict[str, float]:
    if not text.strip():
        return {"empty": 1.0}
    result = classifier(text, truncation=True, max_length=128)[0]
    label = LABEL_MAP.get(result["label"], result["label"].lower())
    score = float(result["score"])
    return {label: score}


demo = gr.Interface(
    fn=infer,
    inputs=gr.Textbox(lines=3, label="Masukkan teks ulasan"),
    outputs=gr.Label(label="Prediksi sentimen"),
    title="Atha Text Classifier Demo",
    description="Demo model sentimen Bahasa Indonesia: negative, neutral, positive.",
)


if __name__ == "__main__":
    demo.launch()
