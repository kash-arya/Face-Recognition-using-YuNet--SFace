import os
import urllib.request
import pickle
import csv
from datetime import datetime
import cv2

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

def download_file(url, dest_path):
    """Downloads a file from a URL to a local destination if it doesn't already exist."""
    if os.path.exists(dest_path):
        print(f"[INFO] File already exists: {dest_path}")
        return
    
    print(f"[INFO] Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Simple progress callback
    def progress_callback(blocks, block_size, total_size):
        percent = min(100, int(blocks * block_size * 100 / total_size))
        print(f"\rDownloading: {percent}% completed", end="")
        
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_callback)
        print("\n[INFO] Download finished successfully.")
    except Exception as e:
        print(f"\n[ERROR] Failed to download from {url}: {e}")
        # Clean up partial download if it exists
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise e

def load_models(models_dir="Attendance-System/models"):
    """Ensures models are downloaded and returns their absolute paths."""
    os.makedirs(models_dir, exist_ok=True)
    yunet_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")
    
    download_file(YUNET_URL, yunet_path)
    download_file(SFACE_URL, sface_path)
    
    return yunet_path, sface_path

def save_encodings(encodings, filepath="Attendance-System/encodings.pkl"):
    """Saves the database of student face encodings to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(encodings, f)
    print(f"[INFO] Saved {len(encodings)} student encodings to {filepath}")

def load_encodings(filepath="Attendance-System/encodings.pkl"):
    """Loads the database of student face encodings from a pickle file."""
    if not os.path.exists(filepath):
        print(f"[WARNING] Encodings file {filepath} not found. Please register students first.")
        return {}
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_attendance(present_students, filepath="Attendance-System/attendance.csv"):
    """Exports a list of present students to a CSV file with date and timestamp."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Time", "Student Name"])
        for student in present_students:
            writer.writerow([date_str, time_str, student])
            
    print(f"[INFO] Exported attendance for {len(present_students)} students to {filepath}")

def draw_face_annotations(image, faces, match_names, confidences=None):
    """Draws custom bounding boxes and name labels on the image for manual verification.
    
    A green box is drawn for matched faces.
    A red box is drawn for "Unknown" faces.
    """
    annotated_img = image.copy()
    
    for idx, face in enumerate(faces):
        # YuNet face coordinates are: [x, y, w, h, right_eye_x, right_eye_y, ...]
        bbox = face[0:4].astype(int)
        x, y, w, h = bbox
        
        name = match_names[idx] if idx < len(match_names) else "Unknown"
        is_known = name != "Unknown"
        
        # Sleek dark/vibrant colors
        # Green for known, Red for unknown
        color = (46, 204, 113) if is_known else (231, 76, 60) 
        thickness = 2
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, thickness)
        
        # Label formatting
        label = name
        if confidences and idx < len(confidences) and is_known:
            # SFace distance metric: lower distance = higher match confidence
            label += f" ({confidences[idx]:.2f})"
            
        # Draw a translucent or solid name background bar
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Position label above or inside box depending on space
        label_y = y - 10 if y - 10 > text_h else y + text_h + 10
        cv2.rectangle(annotated_img, (x, label_y - text_h - baseline), (x + text_w, label_y + baseline), color, cv2.FILLED)
        cv2.putText(annotated_img, label, (x, label_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return annotated_img
