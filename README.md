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
                      | Face Embeddings   | (encodings.pkl)
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

Set up the project virtual environment with Python 3.14 (or any version 3.8+):

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
  uv pip install opencv-contrib-python numpy pillow scikit-learn
  ```

The system will automatically download the required model weights (`face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`) during the first run.

---

### Project Structure

```
├── README.md                           # Project documentation
└── Attendance-System
    ├── data/                            # Runtime artifacts
    │   ├── encodings.pkl               # Pickled student encodings database
    │   └── attendance.csv              # Cumulative attendance spreadsheet
    ├── main.py                         # CLI controller
    ├── dataset/                        # Reference photos directory
    │   ├── Monica/                     # Monica reference photos
    │   ├── Chandler/                   # Chandler reference photos
    │   └── Ross/                       # Ross reference photos
    ├── models/                         # ONNX model weights (downloaded automatically)
    ├── output/                         # Visual verification proofs (timestamped)
    │   └── annotated_attendance_DD-MM-YYYY.jpeg
    └── src/
        ├── face_engine.py              # YuNet & SFace core wrappers
        └── utils.py                    # Downloads, serialization, and drawing helpers
```

---

## Usage Guide

Run all commands from the workspace root directory using `uv run`.

### Step 1: Student Registration (Enrollment)
Extract face coordinates for each student located in the `dataset/` subfolders:
```bash
uv run python Attendance-System/main.py register
```
*Creates `encodings.pkl` containing the serialized face representations.*

### Step 2: Test Face Recognition (Single Image)
Test the engine against a single image to verify identity matching:
```bash
uv run python Attendance-System/main.py recognize-single Attendance-System/dataset/Monica/Monica1.jpeg
```
*Outputs: Identified name, Euclidean distance score, and face detection confidence.*

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
2. **Matches** faces against `encodings.pkl` database (using L2/Euclidean distance threshold of `1.128`).
3. **Logs** results to `Attendance-System/data/attendance.csv` (inserts a new column for today's date formatted as `DD-MM-YYYY`, marks `P`/`A` for all registered students, and updates cumulative attendance percentage).
4. **Saves** annotated visual proof to `Attendance-System/output/annotated_attendance_DD-MM-YYYY.jpeg` showing green bounding boxes for recognized students and red boxes for unknowns.

---

## Key Design Decisions & Portability

* **Embedded Vector Pickling**: Instead of performing linear deep learning scans on reference images at runtime, embeddings are calculated once and stored in `encodings.pkl`. This speeds up matching into simple vector math executing in milliseconds.
* **Cumulative Grid Export**: Attendance logs are organized as a spreadsheet matrix rather than a transaction log, keeping track of every registered student and their total attendance percentage over the term.
* **OpenCV DNN Portability**: The system uses native ONNX weights supported directly by OpenCV DNN. This enables seamless porting to other platforms, such as integrating the backend into a Flask/FastAPI REST API or loading models directly on-device using the OpenCV Android SDK or ONNX Runtime Mobile, using the exact same YuNet/SFace pipeline logic.
