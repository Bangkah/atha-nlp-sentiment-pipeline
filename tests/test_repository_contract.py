from __future__ import annotations

import csv
import unittest
from pathlib import Path


class RepositoryContractTests(unittest.TestCase):
    def test_required_files_exist(self) -> None:
        required = [
            Path("README.md"),
            Path("Makefile"),
            Path("LICENSE"),
            Path("CONTRIBUTING.md"),
            Path("scripts/train.py"),
            Path("scripts/predict.py"),
            Path("api.py"),
        ]
        missing = [str(path) for path in required if not path.exists()]
        self.assertEqual(missing, [])

    def test_training_csv_schema(self) -> None:
        csv_path = Path("data/raw/train.csv")
        self.assertTrue(csv_path.exists())

        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            self.assertEqual(reader.fieldnames, ["text", "label"])
            sample = [row for _, row in zip(range(30), reader)]

        self.assertGreater(len(sample), 0)
        for row in sample:
            self.assertIn(row["label"], {"0", "1", "2"})
            self.assertTrue(row["text"].strip())


if __name__ == "__main__":
    unittest.main()
