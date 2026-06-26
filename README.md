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

1. **Python version:** Ensure you are running Python 3.8+ (tested on Python 3.14).
2. **Install dependencies:**
   ```bash
   pip install opencv-contrib-python numpy pillow scikit-learn
   ```

The system will automatically download the required model weights (`face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`) during the first run.

---

## Project Structure

```
├── README.md                           # Project documentation
├── basicFunctionsOpenCV.py             # OpenCV sanity check script
└── Attendance-System
    ├── main.py                         # CLI entry point (register, recognize-single, attendance)
    ├── encodings.pkl                   # Generated student encodings database
    ├── attendance.csv                  # Attendance log spreadsheet
    ├── annotated_attendance.jpeg       # Annotated verification output photo
    ├── dataset/                        # Reference photos directory
    │   ├── Person1/                    # Monica photos
    │   ├── Person2/                    # Chandler photos
    │   └── Person3/                    # Ross photos
    ├── models/                         # ONNX model files (downloaded automatically)
    └── src/
        ├── face_engine.py              # YuNet & SFace core wrappers
        └── utils.py                    # Downloads, serialisation, and drawing helpers
```

---

## Usage Guide

Run all commands from the workspace root directory.

### Phase 1: Student Registration (Enrollment)
Build the face database by extracting face embeddings for each student in the `dataset` folder:
```bash
python Attendance-System/main.py register
```
*This processes each directory in `dataset/` (e.g. `Person1`, `Person2`) and generates a serialized `encodings.pkl` database of 128-dimensional vectors.*

### Phase 2: Identify a Single Face
Test identification on a single face photograph:
```bash
python Attendance-System/main.py recognize-single Attendance-System/dataset/Person1/Monica1.jpeg
```
*Outputs the student's name, Euclidean distance score, and face detection confidence.*

### Phase 3: Mark Classroom Attendance
Process a classroom selfie/group photo to headcount and identify present students:
```bash
python Attendance-System/main.py attendance <path_to_group_photo>
```
Example command:
```bash
python Attendance-System/main.py attendance assets/test1.jpeg
```
This command will:
1. Detect all faces and print the total headcount.
2. Cross-reference detected faces against `encodings.pkl`.
3. Print the list of identified students (e.g., Monica, Chandler, Ross).
4. Save the names to `Attendance-System/attendance.csv`.
5. Export a visual verification photo (`Attendance-System/annotated_attendance.jpeg`) showing named bounding boxes (green for recognized students, red for unknowns).

---

## Key Design Decisions & Future Android Porting

* **Embedded Vector Pickling:** Instead of performing linear deep-learning scans on reference images at attendance runtime (which gets slower as database size increases), reference face embeddings are computed *once* during registration and stored in `encodings.pkl`. Live comparisons are fast vector math ($O(1)$ database load), executing in milliseconds.
* **OpenCV DNN Portability:** We use standard ONNX weights natively supported by OpenCV DNN. This enables trivial migration to mobile environments:
  * **Option A (API Backend):** Wrap the engine in a Flask/FastAPI REST API. The Android app takes a photo, uploads it via HTTP, and receives attendance results in JSON format.
  * **Option B (On-Device Mobile Running):** The `.onnx` models can be loaded directly in Java/Kotlin via the OpenCV Android SDK or ONNX Runtime Mobile, using the exact same YuNet/SFace pipeline logic.
