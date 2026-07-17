import os
import time
import hashlib
import urllib.request
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

_SCRIPT_DIR = Path(__file__).resolve().parent.parent

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

MODEL_CHECKSUMS = {
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
}


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        expected_checksum = MODEL_CHECKSUMS.get(os.path.basename(dest_path))
        if expected_checksum:
            actual = _sha256(dest_path)
            if actual == expected_checksum:
                print(f"[INFO] File already exists and checksum verified: {os.path.basename(dest_path)}")
                return
            else:
                print(f"[WARNING] Existing file checksum mismatch: {os.path.basename(dest_path)}. Re-downloading...")
                os.remove(dest_path)
        else:
            print(f"[INFO] File already exists: {os.path.basename(dest_path)}")
            return

    print(f"[INFO] Downloading {os.path.basename(dest_path)} ...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    for attempt in range(3):
        try:
            _do_download(url, dest_path)
            break
        except Exception:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if attempt < 2:
                wait = 2 * (attempt + 1)
                print(f"\n[WARNING] Download failed, retrying in {wait}s... (attempt {attempt + 2}/3)")
                time.sleep(wait)
            else:
                raise

    expected = MODEL_CHECKSUMS.get(os.path.basename(dest_path))
    if expected:
        actual = _sha256(dest_path)
        if actual != expected:
            os.remove(dest_path)
            raise RuntimeError(f"Checksum mismatch for {os.path.basename(dest_path)}. Expected {expected[:8]}..., got {actual[:8]}...")

    print(f"[INFO] Download complete: {os.path.basename(dest_path)}")


def _do_download(url, dest_path):
    with urllib.request.urlopen(url) as response:
        total_size = response.headers.get("Content-Length")
        if total_size is not None:
            total_size = int(total_size)
        else:
            total_size = 0

        downloaded = 0
        with open(dest_path, 'wb') as out_file:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = min(100, int(downloaded * 100 / total_size))
                    print(f"\rDownloading: {pct}%", end="")


def _sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_models(models_dir=str(_SCRIPT_DIR / "models")):
    os.makedirs(models_dir, exist_ok=True)
    yunet_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")

    download_file(YUNET_URL, yunet_path)
    download_file(SFACE_URL, sface_path)

    return yunet_path, sface_path


def save_encodings(encodings, filepath=str(_SCRIPT_DIR / "data" / "encodings.txt")):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for student_name, embeddings in encodings.items():
            safe_name = student_name.replace("#", "_").replace("\n", " ").replace("\r", "")
            f.write(f"#{safe_name}\n")
            for embedding in embeddings:
                line = " ".join(str(v) for v in embedding.flatten())
                f.write(line + "\n")
    print(f"[INFO] Saved {len(encodings)} student encodings to {filepath}")


def load_encodings(filepath=str(_SCRIPT_DIR / "data" / "encodings.txt")):
    if not os.path.exists(filepath):
        print(f"[WARNING] Encodings file {filepath} not found. Please register students first.")
        return {}
    encodings = {}
    current_name = None
    line_number = 0
    skipped = 0
    with open(filepath, 'r') as f:
        for line in f:
            line_number += 1
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                current_name = line[1:]
                if current_name in encodings:
                    print(f"[WARNING] Duplicate student name '{current_name}' at line {line_number} — overwriting previous embeddings.")
                encodings[current_name] = []
            elif current_name is not None:
                try:
                    values = np.array([float(x) for x in line.split()], dtype=np.float32)
                except ValueError:
                    skipped += 1
                    print(f"[WARNING] Skipping non-numeric line at line {line_number} for '{current_name}'.")
                    continue
                if len(values) == 128:
                    encodings[current_name].append(values.reshape(1, 128))
                else:
                    skipped += 1
                    print(f"[WARNING] Skipping malformed embedding at line {line_number} for '{current_name}': expected 128 values, got {len(values)}.")
    if skipped > 0:
        print(f"[WARNING] {skipped} embedding line(s) were skipped due to unexpected length.")
    return encodings


def save_attendance(present_students, filepath=str(_SCRIPT_DIR / "data" / "attendance.csv"), database_filepath=str(_SCRIPT_DIR / "data" / "encodings.txt")):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    database = load_encodings(database_filepath)
    if not database:
        print("[WARNING] No registered students found in encodings database. Skipping CSV export.")
        return

    all_students = sorted(list(database.keys()))

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")

    rows = []
    header = []

    if not os.path.exists(filepath):
        header = ["Names", date_str, "Percentage of total lectures attended"]
        rows.append(header)
        for student in all_students:
            status = "P" if student in present_students else "A"
            pct_str = "100.0%" if status == "P" else "0.0%"
            rows.append([student, status, pct_str])
    else:
        with open(filepath, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            print("[WARNING] Existing attendance CSV is empty — reinitializing with fresh header. Prior history will be lost.")
            header = ["Names", date_str, "Percentage of total lectures attended"]
            rows.append(header)
        else:
            header = rows[0]

        pct_col_name = "Percentage of total lectures attended"
        if pct_col_name in header:
            pct_col_idx = header.index(pct_col_name)
        else:
            pct_col_idx = len(header)
            header.append(pct_col_name)

        if date_str in header:
            date_col_idx = header.index(date_str)
            new_col_inserted = False
        else:
            header.insert(pct_col_idx, date_str)
            date_col_idx = pct_col_idx
            new_col_inserted = True

        existing_names = set()
        for row in rows[1:]:
            if not row or len(row) == 0:
                continue
            student_name = row[0]
            existing_names.add(student_name)

            if new_col_inserted:
                row.insert(date_col_idx, "A")

            if student_name in present_students:
                row[date_col_idx] = "P"
            else:
                row[date_col_idx] = "A"

        for student in all_students:
            if student not in existing_names:
                new_row = [student]
                for col_idx in range(1, len(header) - 1):
                    if col_idx == date_col_idx:
                        new_row.append("P" if student in present_students else "A")
                    else:
                        new_row.append("A")
                new_row.append("")
                rows.append(new_row)

        for row in rows[1:]:
            if not row or len(row) == 0:
                continue
            total_days = len(header) - 2
            p_count = row[1:-1].count("P")
            pct = (p_count / total_days) * 100 if total_days > 0 else 0.0
            row[-1] = f"{pct:.1f}%"

        _validate_csv_rows(header, rows[1:])

    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[INFO] Exported matrix attendance to {filepath}")


def _validate_csv_rows(header, data_rows):
    expected_len = len(header)
    for i, row in enumerate(data_rows):
        if not row:
            continue
        if len(row) != expected_len:
            print(f"[WARNING] Row {i + 2} has {len(row)} columns but header has {expected_len}. CSV may be malformed.")


def draw_face_annotations(image, faces, match_names, distances=None):
    annotated_img = image.copy()

    for idx, face in enumerate(faces):
        bbox = face[0:4].astype(int)
        x, y, w, h = bbox

        name = match_names[idx] if idx < len(match_names) else "Unknown"
        is_known = name != "Unknown"

        color = (46, 204, 113) if is_known else (60, 76, 231)
        thickness = 2

        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, thickness)

        label = name
        if distances and idx < len(distances) and is_known:
            label += f" (dist: {distances[idx]:.2f})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        label_y = y - 10 if y - 10 > text_h else y + text_h + 10
        cv2.rectangle(annotated_img, (x, label_y - text_h - baseline), (x + text_w, label_y + baseline), color, cv2.FILLED)
        cv2.putText(annotated_img, label, (x, label_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return annotated_img
