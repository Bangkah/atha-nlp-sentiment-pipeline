from __future__ import annotations

import gradio as gr

from scripts.predict import predict


def classify_text(text: str) -> tuple[str, float]:
    if not text.strip():
        return "empty", 0.0
    try:
        label, confidence = predict(text=text, model_dir="artifacts/model")
    except FileNotFoundError:
        return "model_not_found", 0.0
    return label, round(confidence, 4)


demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=3, label="Masukkan teks ulasan"),
    outputs=[gr.Label(label="Label"), gr.Number(label="Confidence")],
    title="Atha Text Classifier",
    description="Demo klasifikasi sentimen Bahasa Indonesia (negative / neutral / positive)",
)


if __name__ == "__main__":
    demo.launch()
