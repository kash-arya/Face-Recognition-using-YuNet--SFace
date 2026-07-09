import os
import tempfile
import numpy as np
import pytest

from src.utils import save_encodings, load_encodings


class TestEncodingsRoundtrip:
    def test_save_and_load(self):
        embeddings_db = {
            "Alice": [np.random.randn(1, 128).astype(np.float32) for _ in range(3)],
            "Bob": [np.random.randn(1, 128).astype(np.float32) for _ in range(2)],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            tmp_path = f.name

        try:
            save_encodings(embeddings_db, tmp_path)
            loaded = load_encodings(tmp_path)

            assert set(loaded.keys()) == {"Alice", "Bob"}
            assert len(loaded["Alice"]) == 3
            assert len(loaded["Bob"]) == 2
            for emb in loaded["Alice"]:
                assert emb.shape == (1, 128)
                assert emb.dtype == np.float32
        finally:
            os.unlink(tmp_path)

    def test_sanitized_names(self):
        embeddings_db = {
            "stu#1": [np.random.randn(1, 128).astype(np.float32)],
            "multi\nline": [np.random.randn(1, 128).astype(np.float32)],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            tmp_path = f.name

        try:
            save_encodings(embeddings_db, tmp_path)
            loaded = load_encodings(tmp_path)

            assert "stu_1" in loaded
            assert len(loaded["stu_1"]) == 1
            assert "multi" in "".join(loaded.keys())
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent(self):
        result = load_encodings("/nonexistent/path/encodings.txt")
        assert result == {}

    def test_malformed_line_skipped_with_warning(self, capsys):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("#ValidStudent\n")
            f.write(" ".join(["0.1"] * 128) + "\n")
            f.write("short line with only 3 values\n")
            f.write(" ".join(["0.2"] * 128) + "\n")
            tmp_path = f.name

        try:
            loaded = load_encodings(tmp_path)
            captured = capsys.readouterr()
            assert len(loaded["ValidStudent"]) == 2
            assert "Skipping non-numeric line" in captured.out
        finally:
            os.unlink(tmp_path)
