import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
from src.face_engine import FaceRecognitionEngine
from src.utils import load_models, load_encodings

DEFAULT_THRESHOLD = 1.10
_SCRIPT_DIR = Path(__file__).resolve().parent


def _extract_query_embedding(engine, img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None
    success, faces = engine.detect_faces(image)
    if not success or len(faces) == 0:
        return None
    best_face = max(faces, key=lambda f: f[14]) if len(faces) > 1 else faces[0]
    return engine.extract_embedding(image, best_face)


def _process_query_photos(engine, database, photo_dir, query_student_name):
    same_person = []
    diff_person = []

    student_folder = os.path.join(photo_dir, query_student_name)
    if not os.path.isdir(student_folder):
        return same_person, diff_person

    for file_name in os.listdir(student_folder):
        img_path = os.path.join(student_folder, file_name)
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue

        query_embedding = _extract_query_embedding(engine, img_path)
        if query_embedding is None:
            continue

        for student_name, embeddings in database.items():
            for ref_embedding in embeddings:
                dist = engine.compute_distance(query_embedding, ref_embedding)
                if student_name == query_student_name:
                    same_person.append(dist)
                else:
                    diff_person.append(dist)

    return same_person, diff_person


def _print_threshold_sweep(same_arr, diff_arr):
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

    if same_arr.max() > diff_arr.min():
        print("\n[WARNING] Same-person and different-person distributions overlap!")
        print("  No threshold perfectly separates them. Consider:")
        print("  - Adding more varied enrollment photos (different angles, lighting)")
        print("  - Reviewing the evaluation split for data leakage")


def evaluate_distance_distribution(args):
    yunet_path, sface_path = load_models(args.models_dir)
    engine = FaceRecognitionEngine(yunet_path, sface_path)

    database = load_encodings(args.encodings_file)
    if not database:
        print("[ERROR] No encodings found. Run 'register' first.")
        return

    known_students = sorted(database.keys())

    if args.test_dir is not None:
        test_dir_path = Path(args.test_dir)
        if not test_dir_path.is_dir():
            print(f"[ERROR] Test directory not found: {args.test_dir}")
            return

        print(f"[INFO] Using held-out test photos from {args.test_dir} (no data leakage)")
        photo_dir = str(test_dir_path)
        skipped = []
        for name in known_students:
            student_test_folder = test_dir_path / name
            if not student_test_folder.is_dir():
                skipped.append(name)

        if skipped:
            print(f"[WARNING] No test photos found for: {', '.join(skipped)}")
            print("  These students will be excluded from same-person evaluation.")
            print("  Different-person distances still include them.")
            known_students = [n for n in known_students if n not in skipped]

        if not known_students:
            print("[ERROR] No students have test photos. Add held-out photos to the test directory.")
            return
    else:
        print("[WARNING] No --test-dir provided. Using registration photos for evaluation.")
        print("  WARNING: This causes data leakage — same-person distances will be")
        print("  artificially low because evaluation photos overlap with registration.")
        print("  For honest evaluation, create a test directory with held-out photos")
        print("  (not used during registration) and pass --test-dir.")
        photo_dir = args.dataset_dir

    same_person = []
    diff_person = []

    for name in known_students:
        s, d = _process_query_photos(engine, database, photo_dir, name)
        same_person.extend(s)
        diff_person.extend(d)
        if not s:
            print(f"  [WARNING] No valid test faces found for '{name}'")
        else:
            print(f"  {name}: {len(s)} same-person, {len(d)} different-person comparisons")

    if not same_person:
        print("\n[ERROR] No same-person distances computed. Check test photos.")
        return

    if not diff_person:
        print("[ERROR] No different-person distances computed. Need at least 2 students.")
        return

    same_arr = np.array(same_person)
    diff_arr = np.array(diff_person)
    _print_threshold_sweep(same_arr, diff_arr)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate face matching threshold accuracy using held-out test photos"
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(_SCRIPT_DIR / "dataset"),
        help="Directory with registration photos (fallback if --test-dir not provided)"
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Directory with held-out test photos NOT used during registration "
             "(format: test/{PersonName}/*.jpg). "
             "Using this avoids data leakage."
    )
    parser.add_argument("--models-dir", default=str(_SCRIPT_DIR / "models"))
    parser.add_argument("--encodings-file", default=str(_SCRIPT_DIR / "data" / "encodings.txt"))
    args = parser.parse_args()
    evaluate_distance_distribution(args)


if __name__ == "__main__":
    main()
