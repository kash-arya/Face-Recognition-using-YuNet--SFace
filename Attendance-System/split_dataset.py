"""
Splits face photos from the dataset directory into enrollment (for registration)
and test (for held-out evaluation) directories.

Usage:
    uv run python Attendance-System/split_dataset.py

This creates:
    dataset/   — enrollment photos (used by 'register')
    test/      — held-out test photos (used by 'evaluate.py --test-dir')

Each person's photos are split 60/40 enrollment/test (minimum 1 test photo).
If a person has only 1 photo, they stay enrollment-only (no held-out evaluation).

Folder naming convention: {roll}_{display_name}  (e.g. 101_Ariel_Sharon/)
To add more people: place their photos in dataset/{roll}_{Name}/ and re-run this script.
"""

import sys
import random
import shutil
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_SRC = _SCRIPT_DIR / "dataset"
BACKUP_DIR = _SCRIPT_DIR / "data" / ".dataset_backup"
TEST_DIR = _SCRIPT_DIR / "test"

TRAIN_FRACTION = 0.6
SEED = 42


def is_image(p):
    return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


def main():
    if not DATASET_SRC.is_dir():
        print(f"[ERROR] Dataset directory not found: {DATASET_SRC}")
        sys.exit(1)

    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    print(f"[INFO] Backing up current dataset to {BACKUP_DIR}")
    shutil.copytree(DATASET_SRC, BACKUP_DIR)

    people = {}
    for person_dir in sorted(BACKUP_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        photos = sorted([p for p in person_dir.iterdir() if is_image(p)])
        if photos:
            people[person_dir.name] = photos

    if not people:
        print("[ERROR] No people with photos found.")
        sys.exit(1)

    random.seed(SEED)
    split_map = {}

    for name, photos in people.items():
        n_total = len(photos)
        n_train = max(1, int(n_total * TRAIN_FRACTION))
        n_test = n_total - n_train

        if n_test == 0:
            print(f"  {name}: {n_total} photos → {n_train} enrollment, 0 test (need >= 2 photos for held-out eval)")
            split_map[name] = (photos, [])
        else:
            shuffled = photos[:]
            random.shuffle(shuffled)
            train = shuffled[:n_train]
            test = shuffled[n_train:]
            split_map[name] = (train, test)
            print(f"  {name}: {n_total} photos → {n_train} enrollment, {n_test} test")

    # Rebuild dataset/ from scratch
    shutil.rmtree(DATASET_SRC)
    DATASET_SRC.mkdir(parents=True, exist_ok=True)

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    for name, (train_photos, test_photos) in split_map.items():
        person_dir = DATASET_SRC / name
        person_dir.mkdir(parents=True, exist_ok=True)

        for photo in train_photos:
            shutil.copy2(photo, person_dir / photo.name)

        if test_photos:
            test_person_dir = TEST_DIR / name
            test_person_dir.mkdir(parents=True, exist_ok=True)
            for photo in test_photos:
                shutil.copy2(photo, test_person_dir / photo.name)

    print(f"\n[SUCCESS] Dataset split complete.")
    print(f"  Enrollment: {DATASET_SRC}/ (use with 'register')")
    print(f"  Test:       {TEST_DIR}/ (use with 'evaluate.py --test-dir')")
    print(f"  Backup:     {BACKUP_DIR}/ (original full dataset)")
    print(f"\nNext steps:")
    print(f"  uv run python Attendance-System/main.py register")
    print(f"  uv run python Attendance-System/evaluate.py --test-dir {TEST_DIR}")
    print(f"\nTo add more people:")
    print(f"  Place photos in {DATASET_SRC}/{{roll}}_{{Name}}/ and re-run this script.")


if __name__ == "__main__":
    main()
