import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from src.face_engine import FaceRecognitionEngine
from src.utils import load_models, save_encodings, load_encodings, save_attendance, draw_face_annotations


def _setup_engine_and_db(args):
    yunet_path, sface_path = load_models(args.models_dir)
    engine = FaceRecognitionEngine(yunet_path, sface_path, distance_threshold=args.threshold)
    database = load_encodings(args.encodings_file)
    return engine, database


def _read_image(path):
    if not os.path.exists(path):
        print(f"[ERROR] Image not found: {path}")
        return None
    image = cv2.imread(path)
    if image is None:
        print(f"[ERROR] Could not read image: {path}")
    return image


def cmd_register(args):
    yunet_path, sface_path = load_models(args.models_dir)
    engine = FaceRecognitionEngine(yunet_path, sface_path)

    if not os.path.exists(args.dataset_dir):
        print(f"[ERROR] Dataset directory not found: {args.dataset_dir}")
        return

    encodings_db = {}

    for item in os.listdir(args.dataset_dir):
        student_path = os.path.join(args.dataset_dir, item)
        if not os.path.isdir(student_path):
            continue

        print(f"\n[INFO] Enrolling student: {item}")
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

            before = len(embeddings)
            embedding = engine.extract_embedding(image, best_face)
            embeddings.append(embedding)

            for ksize in [(5, 5), (11, 11), (17, 17)]:
                blurred = cv2.GaussianBlur(image, ksize, 0)
                s, aug_faces = engine.detect_faces(blurred)
                if s and len(aug_faces) > 0:
                    af = max(aug_faces, key=lambda f: f[14]) if len(aug_faces) > 1 else aug_faces[0]
                    if af[14] >= args.min_confidence:
                        embeddings.append(engine.extract_embedding(blurred, af))
            print(f"    Added {len(embeddings) - before} embeddings (original + augmented)")

        if embeddings:
            encodings_db[item] = embeddings
            print(f"[SUCCESS] Registered {item} with {len(embeddings)} faces.")
        else:
            print(f"[WARNING] No valid face embeddings found for {item}.")

    if encodings_db:
        save_encodings(encodings_db, args.encodings_file)
    else:
        print("[ERROR] No students registered. Database is empty.")


def cmd_recognize_single(args):
    engine, database = _setup_engine_and_db(args)
    if not database:
        print("[ERROR] No student encodings found. Please run the 'register' command first.")
        return

    image = _read_image(args.image_path)
    if image is None:
        return

    success, faces = engine.detect_faces(image)
    if not success or len(faces) == 0:
        print("[RESULT] No face detected in the image.")
        return

    if len(faces) > 1:
        print(f"[WARNING] {len(faces)} faces detected but this is single-face mode. Recognizing the highest-confidence face only.")
        print(f"         Use the 'attendance' command for group photos.")

    print(f"[INFO] Detected {len(faces)} face(s) in {args.image_path}.")

    for idx, face in enumerate(faces):
        embedding = engine.extract_embedding(image, face)
        name, distance = engine.match_face(embedding, database)
        print(f"Face #{idx+1}: {name}  (Distance: {distance:.3f}, Confidence: {face[14]:.3f})")


def cmd_attendance(args):
    engine, database = _setup_engine_and_db(args)
    if not database:
        print("[ERROR] No student encodings found. Please run the 'register' command first.")
        return

    image = _read_image(args.image_path)
    if image is None:
        return

    success, faces = engine.detect_faces(image)
    total_headcount = len(faces) if (success and faces is not None) else 0

    present = set()
    match_names = []
    distances = []

    if total_headcount > 0:
        for face in faces:
            embedding = engine.extract_embedding(image, face)
            name, distance = engine.match_face(embedding, database)
            match_names.append(name)
            distances.append(distance)
            if name != "Unknown":
                present.add(name)

    print("\n--- ATTENDANCE REPORT ---")
    print(f"Total Headcount: {total_headcount}")
    print("Present Students:")
    if present:
        for name in sorted(present):
            print(f" - {name}")
    else:
        print("  None")
    print("-------------------------\n")

    save_attendance(present, args.attendance_file, args.encodings_file)

    if args.output_json:
        result = {
            "date": datetime.now().strftime("%d-%m-%Y"),
            "total_headcount": total_headcount,
            "present": sorted(present),
            "detections": [
                {"name": n, "distance": round(float(d), 3)} for n, d in zip(match_names, distances)
            ]
        }
        json_path = args.output_json
        os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else "Attendance-System/output", exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved JSON attendance data to: {json_path}")

    if total_headcount > 0 and args.output_image:
        output_path = args.output_image.replace("annotated_attendance.jpeg",
                                                f"annotated_attendance_{datetime.now():%d-%m-%Y}.jpeg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        annotated = draw_face_annotations(image, faces, match_names, distances)
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Saved annotated verification image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AI-Based Face Recognition Attendance System")
    subparsers = parser.add_subparsers(dest="command")

    reg = subparsers.add_parser("register", help="Scan student dataset and save face embeddings")
    reg.add_argument("--dataset-dir", default="Attendance-System/dataset")
    reg.add_argument("--models-dir", default="Attendance-System/models")
    reg.add_argument("--encodings-file", default="Attendance-System/data/encodings.txt")
    reg.add_argument("--min-confidence", type=float, default=0.85, help="Minimum face detection confidence to accept")

    rec = subparsers.add_parser("recognize-single", help="Recognize a single face image")
    rec.add_argument("image_path")
    rec.add_argument("--models-dir", default="Attendance-System/models")
    rec.add_argument("--encodings-file", default="Attendance-System/data/encodings.txt")
    rec.add_argument("--threshold", type=float, default=1.19)

    att = subparsers.add_parser("attendance", help="Mark classroom attendance from a group photo")
    att.add_argument("image_path")
    att.add_argument("--models-dir", default="Attendance-System/models")
    att.add_argument("--encodings-file", default="Attendance-System/data/encodings.txt")
    att.add_argument("--attendance-file", default="Attendance-System/data/attendance.csv")
    att.add_argument("--output-image", default="Attendance-System/output/annotated_attendance.jpeg")
    att.add_argument("--output-json", default=None, help="Output path for JSON attendance results")
    att.add_argument("--threshold", type=float, default=1.19)

    args = parser.parse_args()

    if args.command == "register":
        cmd_register(args)
    elif args.command == "recognize-single":
        cmd_recognize_single(args)
    elif args.command == "attendance":
        cmd_attendance(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
