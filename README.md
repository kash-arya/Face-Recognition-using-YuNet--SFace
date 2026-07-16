# AI-Based Face Recognition Attendance System

An AI/ML image-processing system built with Python and OpenCV DNN that automatically marks classroom attendance from face photos.

The system uses:
1. **YuNet (Face Detection):** A fast, lightweight CNN face detector optimized for CPU.
2. **SFace (Face Recognition):** An ONNX-based deep learning face recognition model that extracts 128-dimensional face embedding vectors.

---

## System Architecture & Process Flow

```
                      +-------------------+
                      | Student Database  | (dataset/PersonName/*.jpg)
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
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
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
├── docs/
│   └── study_guide.md                 # In-depth technical guide for understanding the system
└── Attendance-System
    ├── main.py                        # CLI controller (register, recognize-single, attendance, list)
    ├── evaluate.py                    # Threshold accuracy evaluation with held-out test support
    ├── split_dataset.py               # Splits photos 60/40 into enrollment/test for honest evaluation
    ├── setup_lfw_dataset.py           # Downloads LFW funneled dataset for expanded testing
    ├── dataset/                       # Reference face photos by student name (enrollment)
    │   ├── Monica/
    │   ├── Chandler/
    │   └── Ross/
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
        └── utils.py                   # Downloads, serialization, CSV logging, and drawing
```

---

## Usage Guide

Run all commands from the workspace root directory using `uv run`.

### Step 1: Student Registration (Enrollment)
Extract face embeddings for each student located in the `dataset/` subfolders:
```bash
uv run python Attendance-System/main.py register
```
*Creates `encodings.txt` containing 128-dimensional face embedding vectors for each student. Registration automatically generates Gaussian-blurred variants of each photo so that close-up selfie registrations still match distant lecture hall faces during attendance. A face quality gate rejects tilted (>30°) or too-small (<60px) enrollment photos — use `--no-quality-check` to bypass. Re-running registration refreshes embeddings for present students while preserving encodings for any student not currently in the dataset folder. Use `--no-augmentation` to skip blur augmentation for faster iterative testing.*

### Step 2: Test Face Recognition (Single Image)
Test the engine against a single image to verify identity matching:
```bash
uv run python Attendance-System/main.py recognize-single Attendance-System/dataset/Monica/Monica1.jpeg
```
*Outputs: Identified name, Euclidean distance score, and face detection confidence. Use `--output-json` for structured results and `--output-image` for an annotated verification image.*

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
2. **Matches** faces against `encodings.txt` database (using L2/Euclidean distance threshold of `1.19`).
3. **Logs** results to `Attendance-System/data/attendance.csv` (inserts a new column for today's date formatted as `DD-MM-YYYY`, marks `P`/`A` for all registered students, and updates cumulative attendance percentage).
4. **Saves** annotated visual proof to `Attendance-System/output/annotated_attendance_DD-MM-YYYY.jpeg` showing green bounding boxes for recognized students and red boxes for unknowns.
5. **Optionally outputs JSON** for app integration: add `--output-json Attendance-System/output/attendance.json` to get structured results (date, headcount, present students, per-face distances).

### Step 4: Evaluate Recognition Accuracy

**First, split your dataset** into enrollment and held-out test photos:
```bash
uv run python Attendance-System/split_dataset.py
```
*Splits each person's photos 60/40 into `dataset/` (for registration) and `test/` (for evaluation). Original full dataset is backed up.*

Then evaluate with held-out test data for honest accuracy measurements:
```bash
uv run python Attendance-System/evaluate.py --test-dir Attendance-System/test
```
*Outputs: same-person vs different-person distance statistics and the optimal threshold value. Using `--test-dir` prevents data leakage — testing on photos the model has never seen during registration. Without it, evaluate.py warns about artificially optimistic results.*

### Step 5: List Registered Students
View all currently enrolled students and their embedding counts:
```bash
uv run python Attendance-System/main.py list
```
*Outputs: Each registered student name and the number of stored face embeddings.*

---

## Key Design Decisions & Portability

* **Embedded Vector Serialization**: Instead of performing linear deep learning scans on reference images at runtime, embeddings are calculated once and stored in `encodings.txt` as plain text. This speeds up matching into simple vector math executing in milliseconds and keeps the data portable across platforms.
* **Cumulative Grid Export**: Attendance logs are organized as a spreadsheet matrix rather than a transaction log, keeping track of every registered student and their total attendance percentage over the term.
* **OpenCV DNN Portability**: The system uses native ONNX weights supported directly by OpenCV DNN. This enables seamless porting to other platforms, such as integrating the backend into a Flask/FastAPI REST API or loading models directly on-device using the OpenCV Android SDK or ONNX Runtime Mobile, using the exact same YuNet/SFace pipeline logic.
