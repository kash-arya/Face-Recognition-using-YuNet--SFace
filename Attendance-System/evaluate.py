import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
from src.face_engine import FaceRecognitionEngine
from src.utils import load_models, load_encodings

DEFAULT_THRESHOLD = 1.19


def evaluate_distance_distribution(args):
    yunet_path, sface_path = load_models(args.models_dir)
    engine = FaceRecognitionEngine(yunet_path, sface_path)

    database = load_encodings(args.encodings_file)
    if not database:
        print("[ERROR] No encodings found. Run 'register' first.")
        return

    known_students = sorted(database.keys())

    same_person = []
    diff_person = []

    for name in known_students:
        student_folder = os.path.join(args.dataset_dir, name)
        if not os.path.isdir(student_folder):
            continue

        for file_name in os.listdir(student_folder):
            img_path = os.path.join(student_folder, file_name)
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            success, faces = engine.detect_faces(image)
            if not success or len(faces) == 0:
                continue

            best_face = max(faces, key=lambda f: f[14]) if len(faces) > 1 else faces[0]
            query_embedding = engine.extract_embedding(image, best_face)

            for student_name, embeddings in database.items():
                for ref_embedding in embeddings:
                    dist = engine.compute_distance(query_embedding, ref_embedding)
                    if student_name == name:
                        same_person.append(dist)
                    else:
                        diff_person.append(dist)

    if not same_person or not diff_person:
        print("[ERROR] Could not compute distances. Check dataset and encodings.")
        return

    same_arr = np.array(same_person)
    diff_arr = np.array(diff_person)

    print("\n=== SAME PERSON (should match) ===")
    print(f"  Count : {len(same_arr)}")
    print(f"  Min   : {same_arr.min():.4f}")
    print(f"  Max   : {same_arr.max():.4f}")
    print(f"  Mean  : {same_arr.mean():.4f}")
    print(f"  Std   : {same_arr.std():.4f}")

    print("\n=== DIFFERENT PERSON (should not match) ===")
    print(f"  Count : {len(diff_arr)}")
    print(f"  Min   : {diff_arr.min():.4f}")
    print(f"  Max   : {diff_arr.max():.4f}")
    print(f"  Mean  : {diff_arr.mean():.4f}")
    print(f"  Std   : {diff_arr.std():.4f}")

    print("\n=== THRESHOLD SUGGESTIONS ===")
    step_count = 20
    candidates = np.linspace(same_arr.max(), diff_arr.min(), step_count)
    best_threshold = None
    best_accuracy = 0

    for t in candidates:
        same_correct = (same_arr <= t).sum()
        diff_correct = (diff_arr > t).sum()
        total = len(same_arr) + len(diff_arr)
        accuracy = 100 * (same_correct + diff_correct) / total
        fp = (diff_arr <= t).sum()
        fn = (same_arr > t).sum()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t

        print(f"  Threshold {t:.4f}: Accuracy {accuracy:.1f}%  (FP={fp}, FN={fn})")

    print(f"\n>>> Best threshold: {best_threshold:.4f} ({best_accuracy:.1f}% accuracy)")
    print(f"    Current default threshold: {DEFAULT_THRESHOLD}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate face matching threshold accuracy")
    parser.add_argument("--dataset-dir", default="Attendance-System/dataset")
    parser.add_argument("--models-dir", default="Attendance-System/models")
    parser.add_argument("--encodings-file", default="Attendance-System/data/encodings.txt")
    args = parser.parse_args()
    evaluate_distance_distribution(args)


if __name__ == "__main__":
    main()
