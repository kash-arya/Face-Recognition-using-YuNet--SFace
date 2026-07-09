import os
import tempfile
import csv
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from src.utils import save_encodings, load_encodings, save_attendance


class TestAttendanceCsv:
    def _make_encodings(self, filepath, students):
        embeddings_db = {s: [np.random.randn(1, 128).astype(np.float32)] for s in students}
        save_encodings(embeddings_db, filepath)

    def test_first_run_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            enc_file = os.path.join(tmpdir, "encodings.txt")
            csv_file = os.path.join(tmpdir, "attendance.csv")

            self._make_encodings(enc_file, ["Alice", "Bob", "Charlie"])
            save_attendance({"Alice"}, csv_file, enc_file)

            with open(csv_file, 'r') as f:
                content = f.read()

            assert "Names" in content
            assert "Alice" in content
            assert "Bob" in content
            assert "Charlie" in content
            assert "P" in content
            assert "A" in content
            assert "100.0%" in content
            assert "0.0%" in content

    def test_subsequent_run_appends_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            enc_file = os.path.join(tmpdir, "encodings.txt")
            csv_file = os.path.join(tmpdir, "attendance.csv")

            self._make_encodings(enc_file, ["Alice", "Bob"])

            # first run (day 1): only Alice present
            with patch('src.utils.datetime') as mock_dt1:
                mock_dt1.now.return_value = MagicMock(strftime=lambda fmt: "01-01-2026" if fmt == "%d-%m-%Y" else "2026-01-01")
                save_attendance({"Alice"}, csv_file, enc_file)

            with open(csv_file, 'r') as f:
                first_header = f.readline().strip().split(",")
            first_date_count = len(first_header) - 2

            # second run (day 2): only Bob present
            with patch('src.utils.datetime') as mock_dt2:
                mock_dt2.now.return_value = MagicMock(strftime=lambda fmt: "02-01-2026" if fmt == "%d-%m-%Y" else "2026-01-02")
                save_attendance({"Bob"}, csv_file, enc_file)

            with open(csv_file, 'r') as f:
                second_header = f.readline().strip().split(",")
            second_date_count = len(second_header) - 2

            assert second_date_count == 2
            assert first_date_count == 1

    def test_empty_csv_warns_and_reinitializes(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            enc_file = os.path.join(tmpdir, "encodings.txt")
            csv_file = os.path.join(tmpdir, "attendance.csv")

            self._make_encodings(enc_file, ["Alice"])
            # Create an empty CSV file
            open(csv_file, 'w').close()

            save_attendance({"Alice"}, csv_file, enc_file)

            captured = capsys.readouterr()
            assert "empty" in captured.out.lower()
            assert "reinitializing" in captured.out.lower()

            with open(csv_file, 'r') as f:
                content = f.read()
            assert "Alice" in content
