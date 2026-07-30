# AI-Based Face Recognition Attendance System

An AI/ML image-processing system built with Python and OpenCV DNN that automatically marks classroom attendance from face photos.

The system uses:
1. **YuNet (Face Detection):** A fast, lightweight CNN face detector optimised for CPU.
2. **SFace (Face Recognition):** An ONNX-based deep learning face recognition model that extracts 128-dimensional face embedding vectors.

---

## System Architecture & Process Flow

```
                      +-------------------+
                      | Student Database  | (dataset/{roll}_{Name}/*.jpg)
                      +---------+---------+
                                |
                                v (python main.py register)
                      +-------------------+
                      | Face Embeddings   | (encodings.txt)
                      +---------+---------+
                                |
                                | (Comparison via L2 Norm Distance)
                                v
+------------------+  +-------------------+  +-----------------------+
|  Classroom Photo |->| Face Recognition  |->|   Attendance Report   |
| (Selfie/Group)   |  |   Engine (SFace)  |  |  (CSV & Visual Photo) |
+------------------+  +-------------------+  +-----------------------+
```

---

## Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) as the Python version and package manager to guarantee fast, reproducible environments across platforms.

### 1. Install `uv`

- **Linux & macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows (PowerShell)**:
  ```powershell
  powershell -c "irm https://optimised/uv/install.ps1 | iex"
  ```

### 2. Set Up Virtual Environment

Set up the project virtual environment with Python 3.14 (or any version 3.12+):

- **Linux & macOS**:
  ```bash
  uv venv --python 3.14
  source .venv/bin/activate
  ```
- **Windows (PowerShell/CMD)**:
  ```powershell
  uv venv --python 3.14
  .venv\Scripts\activate
  ```

### 3. Install Dependencies

Install the required deep learning and image processing dependencies:

- **Linux, macOS, and Windows**:
  ```bash
  uv sync
  ```

The system will automatically download the required model weights (`face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`) during the first run.

---

### Project Structure

```
├── pyproject.toml                     # Project config with pinned dependencies
├── uv.lock                            # Locked dependency versions
├── README.md                          # Project documentation
└── Attendance-System
    ├── main.py                        # CLI controller (register, recognize-single, attendance, list, unregister)
    ├── evaluate.py                    # Threshold accuracy evaluation with recognise-single-support
    ├── split_dataset.py               # Splits photos 60/40 into enrollment/test for honest evaluation
    ├── setup_lfw_dataset.py           # Downloads LFW funnelled dataset for expanded testing
    ├── dataset/                       # Reference face photos (enrollment) — {roll}_{Name} folders, 15 people from LFW benchmark
    ├── test/                          # Held-out test photos (not used during registration)
    ├── models/                        # ONNX model weights (downloaded automatically with SHA256 verification)
    ├── data/                          # Runtime artifacts
    │   ├── encodings.txt              # Plain text student face encodings
    │   └── attendance.csv             # Cumulative attendance spreadsheet
    ├── output/                        # Visual verification proofs (timestamped)
    │   └── annotated_attendance_DD-MM-YYYY.jpeg
    ├── tests/                         # Unit tests (encodings, attendance CSV, face engine)
    │   ├── conftest.py
    │   ├── test_encodings.py
    │   ├── test_attendance.py
    │   └── test_face_engine.py
    └── src/
        ├── __init__.py                # Package marker
        ├── face_engine.py             # YuNet & SFace core wrappers
        └── utils.py                   # Downloads, serialisation, CSV logging, and drawing
```

---

## Usage Guide

Run all commands using `uv run`. All file paths default relative to the script location, so commands work from any directory.

### Step 1: Student Registration (Enrollment)
Extract face embeddings for each student located in the `dataset/` subfolders:
```bash
uv run python Attendance-System/main.py register
```
*Creates `encodings.txt` containing 128-dimensional face embedding vectors for each student. Registration automatically generates Gaussian-blurred variants of each photo, ensuring that close-up selfie registrations still match distant lecture-hall faces during attendance. A face quality gate rejects tilted (>30°), too-small (<60px), or downward-looking enrollment photos — use `--no-quality-check` to bypass, or `--min-pitch-ratio` to tune sensitivity. Re-running registration refreshes embeddings for current students while preserving encodings for students not currently in the dataset folder. Use `--no-augmentation` to skip blur augmentation for faster iterative testing.*

### Step 2: Test Face Recognition (Single Image)
Test the engine against a single image to verify identity matching:
```bash
uv run python Attendance-System/main.py recognise-single Attendance-System/dataset/114_Tony_Blair/01.jpg
```
*Outputs: Identified student (roll number and name), Euclidean distance score, and face detection confidence. Use `--output-json` for structured results and `--output-image` for an annotated verification image.*

### Step 3: Run Classroom Attendance
Process a classroom group photo to headcount students and log attendance:
```bash
uv run python Attendance-System/main.py attendance <path_to_group_photo>
```
**Example (WSL / Windows)**:
```bash
uv run python Attendance-System/main.py attendance /mnt/c/Users/ASUS/Downloads/FriendsLead.webp
```

This command automatically:
1. **Detects** all faces present (prints total headcount to console).
2. **Matches** faces against `encodings.txt` database (using L2/Euclidean distance threshold of `1.10`).
3. **Logs** results to `Attendance-System/data/attendance.csv` (inserts a new column for today's date formatted as `DD-MM-YYYY`, marks `P`/`A` for all registered students, and updates cumulative attendance percentage).
4. **Saves** annotated visual proof to `Attendance-System/output/annotated_attendance_DD-MM-YYYY.jpeg` showing green bounding boxes for recognised students and red boxes for unknowns.
5. **Optionally outputs JSON** for app integration: add `--output-json Attendance-System/output/attendance.json` to get structured results (date, headcount, present students, per-face distances).

### Step 4: Evaluate Recognition Accuracy

**First, split your dataset** into enrollment and held-out test photos:
```bash
uv run python Attendance-System/split_dataset.py
```
*Splits each person's photos 60/40 into `dataset/` (for registration) and `test/` (for evaluation). The current dataset is backed up to `data/.dataset_backup/` before splitting.*

Then evaluate with held-out test data for honest accuracy measurements:
```bash
uv run python Attendance-System/evaluate.py --test-dir Attendance-System/test
```
*Outputs: same-person vs different-person distance statistics and the optimal threshold value. Using `--test-dir` prevents data leakage — testing on photos the model has never seen during registration. Without it, evaluate.py warns about artificially optimistic results.*

### Step 5: List Registered Students
View all currently enrolled students:
```bash
uv run python Attendance-System/main.py list
```
*Outputs: Each registered student's roll number, name, and the number of stored face embeddings in table format.*

### Step 6: Unregister a Student
Remove a student from the encodings database:
```bash
uv run python Attendance-System/main.py unregister 114
```
*Removes all embeddings for the given roll number from `encodings.txt`. Non-destructive — the student's dataset folder and attendance CSV history are left untouched. Re-register the student later by running `register` again.*

---

## Key Design Decisions & Portability

* **Precomputed embeddings, not runtime scans.** Face embeddings are calculated once at registration and stored as plain text in `encodings.txt`. At attendance time, matching reduces to simple vector math — milliseconds per face instead of running a neural network against every reference photo.
* **Cumulative CSV matrix, not a transaction log.** Attendance is a spreadsheet grid where each row is a student and each column is a date. The percentage column stays current without needing to scan the entire history.
* **OpenCV DNN, not PyTorch or TensorFlow.** The pipeline uses ONNX weights natively through OpenCV's DNN module with multi-threaded inference (`cv2.setNumThreads(4)`). The same YuNet/SFace models can be loaded on Android via OpenCV Android SDK or ONNX Runtime Mobile — no retraining, no conversion. The backend drops into a FastAPI endpoint without touching the engine layer.
