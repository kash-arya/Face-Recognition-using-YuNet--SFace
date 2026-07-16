"""
Downloads the LFW (Labeled Faces in the Wild) funneled dataset, selects
the 15 people with the most photos, and splits each person's images into
enrollment (for registration) and test (for held-out evaluation).

This eliminates data leakage: evaluate.py can use --test-dir with photos
the model has never seen during registration, giving honest accuracy figures.

Usage:
    uv run python Attendance-System/setup_lfw_dataset.py
"""

import os
import sys
import tarfile
import shutil
import urllib.request
import hashlib
import tempfile
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
_LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
_LFW_MD5 = "1b42dfed7d15c9b2dd63d5e5840c86ad"
_LFW_FILENAME = "lfw-funneled.tgz"
_TARGET_PEOPLE = 15
_MIN_PHOTOS = 3

ENROLLMENT_FRACTION = 0.6


def _download(url, dest):
    print(f"[INFO] Downloading {dest.name} ({_LFW_URL})...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print(f"  You can manually download from: {_LFW_URL}")
        sys.exit(1)

    actual = hashlib.md5(dest.read_bytes()).hexdigest()
    if actual != _LFW_MD5:
        dest.unlink()
        print(f"[ERROR] MD5 mismatch. Expected {_LFW_MD5[:16]}..., got {actual[:16]}...")
        sys.exit(1)
    print(f"[INFO] Download verified (MD5 OK)")


def _extract(archive_path, dest_dir):
    print(f"[INFO] Extracting {archive_path.name}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir, filter="data")
    print(f"[INFO] Extraction complete")


def _find_extracted_root(extract_dir):
    for child in extract_dir.iterdir():
        if child.is_dir():
            return child
    return extract_dir


def main():
    cache_dir = _SCRIPT_DIR / "data" / ".lfw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / _LFW_FILENAME

    if not archive_path.exists():
        _download(_LFW_URL, archive_path)
    else:
        print(f"[INFO] Using cached {archive_path.name}")

    extract_dir = cache_dir / "extracted"
    if not extract_dir.is_dir():
        _extract(archive_path, cache_dir)
        lfw_root = _find_extracted_root(cache_dir)
        lfw_root.rename(extract_dir)
    lfw_root = extract_dir

    person_photos = defaultdict(list)
    for person_dir in sorted(lfw_root.iterdir()):
        if person_dir.is_dir():
            photos = sorted([p for p in person_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])
            if len(photos) >= _MIN_PHOTOS:
                person_photos[person_dir.name] = photos

    selected = sorted(person_photos.items(), key=lambda x: len(x[1]), reverse=True)[:_TARGET_PEOPLE]
    print(f"\n[INFO] Selected {len(selected)} people (each with >= {_MIN_PHOTOS} photos):")
    for name, photos in selected:
        n_enroll = max(1, int(len(photos) * ENROLLMENT_FRACTION))
        n_test = len(photos) - n_enroll
        print(f"  {name}: {len(photos)} photos → {n_enroll} enrollment, {n_test} test")

    dataset_dir = _SCRIPT_DIR / "dataset"
    test_dir = _SCRIPT_DIR / "test"

    if dataset_dir.exists():
        print(f"\n[WARNING] Dataset directory already exists: {dataset_dir}")
        answer = input("  Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("  Aborting.")
            sys.exit(0)
        shutil.rmtree(dataset_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for name, photos in selected:
        n_enroll = max(1, int(len(photos) * ENROLLMENT_FRACTION))
        enroll_photos = photos[:n_enroll]
        test_photos = photos[n_enroll:]

        (dataset_dir / name).mkdir(parents=True, exist_ok=True)
        for i, photo in enumerate(enroll_photos):
            dest = dataset_dir / name / f"{i+1:02d}.jpg"
            shutil.copy2(photo, dest)

        (test_dir / name).mkdir(parents=True, exist_ok=True)
        for i, photo in enumerate(test_photos):
            dest = test_dir / name / f"{i+1:02d}.jpg"
            shutil.copy2(photo, dest)

        print(f"  ✓ {name}: {len(enroll_photos)} enrollment, {len(test_photos)} test photos copied")

    print(f"\n[SUCCESS] Dataset ready:")
    print(f"  Enrollment: {dataset_dir}/ (for 'register')")
    print(f"  Test:       {test_dir}/ (for 'evaluate.py --test-dir')")
    print(f"\nNext steps:")
    print(f"  1. uv run python Attendance-System/main.py register --dataset-dir {dataset_dir}")
    print(f"  2. uv run python Attendance-System/evaluate.py --test-dir {test_dir}")


if __name__ == "__main__":
    main()
