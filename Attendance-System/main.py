import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from src.face_engine import FaceRecognitionEngine
from src.utils import load_models, save_encodings, load_encodings, save_attendance, draw_face_annotations

_SCRIPT_DIR = Path(__file__).resolve().parent


def _setup_engine_and_db(args):
    yunet_path, sface_path = load_models(args.models_dir)
    engine = FaceRecognitionEngine(
        yunet_path, sface_path,
        distance_threshold=args.threshold,
        min_pitch_ratio=getattr(args, 'min_pitch_ratio', 0.05),
    )
    encodings, names_map = load_encodings(args.encodings_file)
    return engine, encodings, names_map


def _read_image(path):
    if not os.path.exists(path):
        print(f"[ERROR] Image not found: {path}")
        return None
    image = cv2.imread(path)
    if image is None:
        print(f"[ERROR] Could not read image: {path}")
    return image


def _filter_faces(faces, min_confidence):
    """Keeps only faces with confidence >= min_confidence. Returns filtered list."""
    filtered = [f for f in faces if f[14] >= min_confidence]
    if len(faces) > 0 and len(filtered) == 0:
        print(f"[WARNING] All {len(faces)} detected faces have confidence below {min_confidence}. None will be recognized.")
    elif len(faces) != len(filtered):
        print(f"[INFO] Filtered out {len(faces) - len(filtered)} low-confidence face(s) (threshold: {min_confidence}).")
    return filtered


def _timestamped_path(filepath):
    """Inserts a date stamp before the file extension: 'out.jpg' -> 'out_01-01-2026.jpg'"""
    base, ext = os.path.splitext(filepath)
    return f"{base}_{datetime.now():%d-%m-%Y}{ext}"


def cmd_register(args):
    yunet_path, sface_path = load_models(args.models_dir)
    engine = FaceRecognitionEngine(
        yunet_path, sface_path,
        min_pitch_ratio=args.min_pitch_ratio,
    )

    if not os.path.exists(args.dataset_dir):
        print(f"[ERROR] Dataset directory not found: {args.dataset_dir}")
        return

    if os.path.exists(args.encodings_file):
        encodings_db, names_map = load_encodings(args.encodings_file)
        print(f"[INFO] Loaded {len(encodings_db)} existing student(s) from {args.encodings_file}")
    else:
        encodings_db = {}
        names_map = {}

    for item in os.listdir(args.dataset_dir):
        student_path = os.path.join(args.dataset_dir, item)
        if not os.path.isdir(student_path):
            continue

        # Parse folder name: "{roll}_{display_name}" — split on first underscore
        if '_' not in item:
            print(f"[WARNING] Folder '{item}' does not follow '{{roll}}_{{name}}' convention. Skipping.")
            continue
        roll, display_name = item.split('_', 1)
        if not roll.isdigit():
            print(f"[WARNING] Folder '{item}' has non-numeric roll '{roll}'. Roll must be digits only. Skipping.")
            continue
        display_name = display_name.replace('_', ' ')  # restore spaces for display

        print(f"\n[INFO] Enrolling: {roll} - {display_name}")
        embeddings = []

        for file_name in os.listdir(student_path):
            img_path = os.path.join(student_path, file_name)
            if not os.path.isfile(img_path) or not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue

            print(f"  Processing image: {file_name}")
            image = cv2.imread(img_path)
            if image is None:
                print(f"  [WARNING] Could not read image: {img_path}")
                continue

            success, faces = engine.detect_faces(image)
            if not success or len(faces) == 0:
                print(f"  [WARNING] No face detected in {file_name}. Skipping image.")
                continue
            elif len(faces) > 1:
                print(f"  [WARNING] Multiple faces ({len(faces)}) detected in {file_name}. Using the highest-confidence face.")

            best_face = faces[0] if len(faces) == 1 else max(faces, key=lambda f: f[14])
            confidence = best_face[14]

            if confidence < args.min_confidence:
                print(f"  [WARNING] Face confidence too low ({confidence:.2f} < {args.min_confidence}). Skipping.")
                continue

            if not args.no_quality_check:
                ok, reason = engine.check_face_quality(best_face)
                if not ok:
                    print(f"  [WARNING] Face quality check failed: {reason}. Skipping.")
                    continue

            before = len(embeddings)
            embedding = engine.extract_embedding(image, best_face)
            embeddings.append(embedding)

            if not args.no_augmentation:
                for ksize in [(5, 5), (11, 11), (17, 17)]:
                    blurred = cv2.GaussianBlur(image, ksize, 0)
                    s, aug_faces = engine.detect_faces(blurred)
                    if s and len(aug_faces) > 0:
                        af = max(aug_faces, key=lambda f: f[14]) if len(aug_faces) > 1 else aug_faces[0]
                        if af[14] >= args.min_confidence:
                            embeddings.append(engine.extract_embedding(blurred, af))
            num_augmented = len(embeddings) - before
            if num_augmented > 0:
                print(f"    Added {num_augmented} embeddings (original + {num_augmented - 1} augmented)")

        if embeddings:
            encodings_db[roll] = embeddings
            names_map[roll] = display_name
            print(f"[SUCCESS] Registered {roll} - {display_name} with {len(embeddings)} faces.")
        else:
            print(f"[WARNING] No valid face embeddings found for {roll} - {display_name}.")

    if encodings_db:
        save_encodings(encodings_db, names_map, args.encodings_file)
    else:
        print("[ERROR] No students registered. Database is empty.")


def cmd_recognize_single(args):
    engine, encodings, names_map = _setup_engine_and_db(args)
    if not encodings:
        print("[ERROR] No student encodings found. Please run the 'register' command first.")
        return

    image = _read_image(args.image_path)
    if image is None:
        return

    success, faces = engine.detect_faces(image)
    if not success or len(faces) == 0:
        print("[RESULT] No face detected in the image.")
        return

    faces = _filter_faces(faces, args.min_confidence)
    if len(faces) == 0:
        print("[RESULT] No faces passed the confidence threshold.")
        return

    if len(faces) > 1:
        print(f"[INFO] {len(faces)} faces detected but this is single-face mode. Using the highest-confidence face only.")
        print(f"       Use the 'attendance' command for group photos.")

    # Pick the single best face by detection confidence
    best_face = max(faces, key=lambda f: f[14])
    embedding = engine.extract_embedding(image, best_face)
    roll, distance = engine.match_face(embedding, encodings)
    display = f"{roll} - {names_map[roll]}" if roll != "Unknown" else "Unknown"

    print(f"[RESULT] {display}  (Distance: {distance:.3f}, Detection Confidence: {best_face[14]:.3f})")

    if args.output_json:
        result = {
            "image": args.image_path,
            "result": {
                "roll": roll,
                "name": names_map.get(roll, ""),
                "display": display,
                "distance": round(float(distance), 3),
                "detection_confidence": round(float(best_face[14]), 3),
            }
        }
        json_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved JSON result to: {json_path}")

    if args.output_image:
        output_path = _timestamped_path(args.output_image)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        annotated = draw_face_annotations(image, np.array([best_face]), [display], [distance])
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Saved annotated verification image to: {output_path}")


def cmd_attendance(args):
    engine, encodings, names_map = _setup_engine_and_db(args)
    if not encodings:
        print("[ERROR] No student encodings found. Please run the 'register' command first.")
        return

    image = _read_image(args.image_path)
    if image is None:
        return

    success, faces = engine.detect_faces(image)
    if not success or faces is None:
        faces = np.array([])

    faces = _filter_faces(faces, args.min_confidence)
    total_headcount = len(faces)

    # present: dict {roll: display_name}
    present = {}
    match_labels = []
    distances = []

    if total_headcount > 0:
        for face in faces:
            embedding = engine.extract_embedding(image, face)
            roll, distance = engine.match_face(embedding, encodings)
            if roll != "Unknown":
                display = f"{roll} - {names_map[roll]}"
                present[roll] = names_map[roll]
            else:
                display = "Unknown"
            match_labels.append(display)
            distances.append(distance)

    print("\n--- ATTENDANCE REPORT ---")
    print(f"Total Headcount: {total_headcount}")
    print("Present Students:")
    if present:
        for roll in sorted(present):
            print(f" - {roll} - {present[roll]}")
    else:
        print("  None")
        if total_headcount > 0:
            print("[TIP] No faces matched. You may want to tune the threshold with evaluate.py")
    print("-------------------------\n")

    save_attendance(present, args.attendance_file, args.encodings_file)

    if args.output_json:
        result = {
            "date": datetime.now().strftime("%d-%m-%Y"),
            "total_headcount": total_headcount,
            "present": [
                {"roll": r, "name": n} for r, n in sorted(present.items())
            ],
            "detections": [
                {"label": lbl, "distance": round(float(d), 3)}
                for lbl, d in zip(match_labels, distances)
            ]
        }
        json_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved JSON attendance data to: {json_path}")

    if total_headcount > 0 and args.output_image:
        output_path = _timestamped_path(args.output_image)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        annotated = draw_face_annotations(image, faces, match_labels, distances)
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Saved annotated verification image to: {output_path}")


def cmd_list_students(args):
    encodings, names_map = load_encodings(args.encodings_file)
    if not encodings:
        print("[INFO] No students currently registered.")
        return

    print(f"\n{len(encodings)} registered student(s):\n")
    print(f"  {'Roll':<8}  {'Name':<35}  Embeddings")
    print(f"  {'-'*8}  {'-'*35}  ----------")
    for roll in sorted(encodings.keys()):
        name = names_map.get(roll, roll)
        count = len(encodings[roll])
        print(f"  {roll:<8}  {name:<35}  {count}")
    print()


def cmd_unregister(args):
    if not os.path.exists(args.encodings_file):
        print(f"[ERROR] Encodings file not found: {args.encodings_file}")
        return

    encodings, names_map = load_encodings(args.encodings_file)
    if not encodings:
        print("[INFO] No students currently registered.")
        return

    roll = args.roll_number
    if roll not in encodings:
        print(f"[WARNING] Roll number '{roll}' is not registered. No action taken.")
        return

    display = f"{roll} - {names_map.get(roll, roll)}"
    del encodings[roll]
    del names_map[roll]
    save_encodings(encodings, names_map, args.encodings_file)
    print(f"[SUCCESS] Removed '{display}' from {args.encodings_file}")
    print(f"[INFO] {len(encodings)} student(s) remain registered.")


def main():
    parser = argparse.ArgumentParser(description="AI-Based Face Recognition Attendance System")
    subparsers = parser.add_subparsers(dest="command")

    reg = subparsers.add_parser("register", help="Scan student dataset and save face embeddings")
    reg.add_argument("--dataset-dir", default=str(_SCRIPT_DIR / "dataset"))
    reg.add_argument("--models-dir", default=str(_SCRIPT_DIR / "models"))
    reg.add_argument("--encodings-file", default=str(_SCRIPT_DIR / "data" / "encodings.txt"))
    reg.add_argument("--min-confidence", type=float, default=0.85, help="Minimum face detection confidence to accept")
    reg.add_argument("--no-augmentation", action="store_true", help="Skip Gaussian blur augmentation (faster, less robust)")
    reg.add_argument("--no-quality-check", action="store_true", help="Skip face quality gate (tilt, size, pitch) during registration")
    reg.add_argument("--min-pitch-ratio", type=float, default=0.05,
                     help="Minimum nose-to-eye vertical ratio for pitch quality check (default: 0.05)")

    rec = subparsers.add_parser("recognize-single", help="Recognize a single face image")
    rec.add_argument("image_path")
    rec.add_argument("--models-dir", default=str(_SCRIPT_DIR / "models"))
    rec.add_argument("--encodings-file", default=str(_SCRIPT_DIR / "data" / "encodings.txt"))
    rec.add_argument("--threshold", type=float, default=1.10)
    rec.add_argument("--min-confidence", type=float, default=0.85, help="Minimum face detection confidence to accept")
    rec.add_argument("--output-json", default=None, help="Output path for JSON recognition result")
    rec.add_argument("--output-image", default=None, help="Output path for annotated verification image")

    att = subparsers.add_parser("attendance", help="Mark classroom attendance from a group photo")
    att.add_argument("image_path")
    att.add_argument("--models-dir", default=str(_SCRIPT_DIR / "models"))
    att.add_argument("--encodings-file", default=str(_SCRIPT_DIR / "data" / "encodings.txt"))
    att.add_argument("--attendance-file", default=str(_SCRIPT_DIR / "data" / "attendance.csv"))
    att.add_argument("--output-image", default=str(_SCRIPT_DIR / "output" / "annotated_attendance.jpeg"), help="Output path for annotated verification image")
    att.add_argument("--output-json", default=None, help="Output path for JSON attendance results")
    att.add_argument("--threshold", type=float, default=1.10)
    att.add_argument("--min-confidence", type=float, default=0.85, help="Minimum face detection confidence to accept")

    lst = subparsers.add_parser("list", help="List all currently registered students")
    lst.add_argument("--encodings-file", default=str(_SCRIPT_DIR / "data" / "encodings.txt"))

    unreg = subparsers.add_parser("unregister", help="Remove a student from the encodings database")
    unreg.add_argument("roll_number", help="Roll number of the student to remove (e.g. 101)")
    unreg.add_argument("--encodings-file", default=str(_SCRIPT_DIR / "data" / "encodings.txt"))

    args = parser.parse_args()

    if args.command == "register":
        cmd_register(args)
    elif args.command == "recognize-single":
        cmd_recognize_single(args)
    elif args.command == "attendance":
        cmd_attendance(args)
    elif args.command == "list":
        cmd_list_students(args)
    elif args.command == "unregister":
        cmd_unregister(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
