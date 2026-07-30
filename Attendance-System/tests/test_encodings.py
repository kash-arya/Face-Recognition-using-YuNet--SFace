import os
import tempfile
import numpy as np
import pytest

from src.utils import save_encodings, load_encodings


class TestEncodingsRoundtrip:
    def test_save_and_load(self):
        embeddings_db = {
            "101": [np.random.randn(1, 128).astype(np.float32) for _ in range(3)],
            "102": [np.random.randn(1, 128).astype(np.float32) for _ in range(2)],
        }
        names_map = {"101": "Alice", "102": "Bob"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            tmp_path = f.name

        try:
            save_encodings(embeddings_db, names_map, tmp_path)
            loaded, loaded_names = load_encodings(tmp_path)

            assert set(loaded.keys()) == {"101", "102"}
            assert loaded_names["101"] == "Alice"
            assert loaded_names["102"] == "Bob"
            assert len(loaded["101"]) == 3
            assert len(loaded["102"]) == 2
            for emb in loaded["101"]:
                assert emb.shape == (1, 128)
                assert emb.dtype == np.float32
        finally:
            os.unlink(tmp_path)

    def test_legacy_format_rejected(self, capsys):
        """Encodings files without the pipe separator should fail with a clear error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("#Alice\n")
            f.write(" ".join(["0.1"] * 128) + "\n")
            tmp_path = f.name

        try:
            loaded, loaded_names = load_encodings(tmp_path)
            captured = capsys.readouterr()
            assert loaded == {}
            assert loaded_names == {}
            assert "legacy" in captured.out.lower()
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent(self):
        result, names = load_encodings("/nonexistent/path/encodings.txt")
        assert result == {}
        assert names == {}

    def test_malformed_line_skipped_with_warning(self, capsys):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("#101|ValidStudent\n")
            f.write(" ".join(["0.1"] * 128) + "\n")
            f.write("short line with only 3 values\n")
            f.write(" ".join(["0.2"] * 128) + "\n")
            tmp_path = f.name

        try:
            loaded, _ = load_encodings(tmp_path)
            captured = capsys.readouterr()
            assert len(loaded["101"]) == 2
            assert "Skipping non-numeric line" in captured.out
        finally:
            os.unlink(tmp_path)
