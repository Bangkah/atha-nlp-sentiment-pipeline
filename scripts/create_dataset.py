from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    train_size: int = 1500
    valid_size: int = 300
    seed: int = 42


POSITIVE_TEMPLATES = (
    "produk ini sangat bagus",
    "pelayanan cepat dan memuaskan",
    "kualitasnya mantap",
    "saya suka hasilnya",
    "pengiriman tepat waktu",
    "recommended banget",
    "fiturnya sangat membantu",
    "harga sesuai kualitas",
    "worth it banget buat dibeli",
    "adminnya responsif dan ramah",
    "packaging rapi dan aman",
    "dipakai harian tetap nyaman",
)

NEGATIVE_TEMPLATES = (
    "produk ini mengecewakan",
    "pelayanan sangat lambat",
    "kualitasnya buruk",
    "saya tidak puas",
    "pengiriman terlambat",
    "tidak recommended",
    "fiturnya sering error",
    "harga tidak sebanding",
    "barang datang dalam kondisi jelek",
    "adminnya susah dihubungi",
    "packaging rusak waktu sampai",
    "dipakai sebentar langsung bermasalah",
)

NEUTRAL_TEMPLATES = (
    "produk ini standar",
    "pelayanan cukup biasa",
    "kualitasnya lumayan",
    "hasilnya tidak buruk tidak istimewa",
    "pengiriman sesuai estimasi",
    "fiturnya cukup membantu",
    "harga masih oke",
    "pengalaman saya biasa saja",
    "sejauh ini aman tapi belum spesial",
    "fiturnya jalan sesuai deskripsi",
    "pengiriman normal tidak cepat tidak lambat",
    "overall masih oke untuk harga segini",
)

OPENERS = (
    "",
    "menurutku ",
    "buatku ",
    "jujur ya ",
    "review singkat ",
    "kalau dari pengalamanku ",
)

CLOSERS = (
    "",
    " sih",
    " banget",
    " ya",
    " buatku",
    " menurutku",
)

EXTRAS = (
    "",
    " dari segi fitur",
    " buat pemakaian harian",
    " untuk harga segini",
    " dari pengalaman pertama",
)

SLANG_REPLACEMENTS: dict[str, tuple[str, ...]] = {
    "tidak": ("ga", "nggak", "gak"),
    "banget": ("banget", "bgt"),
    "sangat": ("sangat", "beneran"),
    "produk": ("produk", "barang"),
    "pelayanan": ("pelayanan", "servis"),
    "memuaskan": ("memuaskan", "ok"),
    "mengecewakan": ("zonk", "mengecewakan"),
    "recommended": ("recommended", "rekomen"),
    "pengiriman": ("pengiriman", "delivery"),
}

TYPO_REPLACEMENTS: dict[str, str] = {
    "banget": "bnget",
    "kualitas": "kwalitas",
    "pelayanan": "pelayanannya",
    "pengiriman": "pengiriman",
}


def build_sentence(base: str, rng: random.Random) -> str:
    text = f"{rng.choice(OPENERS)}{base}{rng.choice(EXTRAS)}{rng.choice(CLOSERS)}".strip()

    for key, choices in SLANG_REPLACEMENTS.items():
        if key in text and rng.random() < 0.35:
            text = text.replace(key, rng.choice(choices), 1)

    for key, typo in TYPO_REPLACEMENTS.items():
        if key in text and rng.random() < 0.12:
            text = text.replace(key, typo, 1)

    if rng.random() < 0.2:
        text = text.replace("  ", " ").strip() + rng.choice(("!", "...", "."))

    return text.strip()


def create_samples(size: int, label: int, rng: random.Random) -> list[dict[str, int | str]]:
    if label == 2:
        templates = POSITIVE_TEMPLATES
    elif label == 1:
        templates = NEUTRAL_TEMPLATES
    else:
        templates = NEGATIVE_TEMPLATES
    rows: list[dict[str, int | str]] = []
    for _ in range(size):
        base = rng.choice(templates)
        rows.append({"text": build_sentence(base, rng), "label": label})
    return rows


def main() -> None:
    cfg = DatasetConfig()
    rng = random.Random(cfg.seed)

    if cfg.train_size % 3 != 0 or cfg.valid_size % 3 != 0:
        raise ValueError("train_size dan valid_size harus habis dibagi 3 untuk dataset 3 kelas")

    train_each_class = cfg.train_size // 3
    valid_each_class = cfg.valid_size // 3

    train_rows = (
        create_samples(train_each_class, 2, rng)
        + create_samples(train_each_class, 1, rng)
        + create_samples(train_each_class, 0, rng)
    )
    valid_rows = (
        create_samples(valid_each_class, 2, rng)
        + create_samples(valid_each_class, 1, rng)
        + create_samples(valid_each_class, 0, rng)
    )

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(train_rows)
    valid_df = pd.DataFrame(valid_rows)

    train_path = output_dir / "train.csv"
    valid_path = output_dir / "valid.csv"

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)

    print(f"train -> {train_path} ({len(train_df)} rows)")
    print(f"valid -> {valid_path} ({len(valid_df)} rows)")


if __name__ == "__main__":
    main()
