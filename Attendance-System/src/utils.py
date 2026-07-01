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

def save_attendance(present_students, filepath="Attendance-System/attendance.csv", database_filepath="Attendance-System/encodings.pkl"):
    """Exports attendance to a cumulative CSV matrix format with P/A markers and percentages."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 1. Load all registered students
    database = load_encodings(database_filepath)
    if not database:
        print("[WARNING] No registered students found in encodings database. Skipping CSV export.")
        return
        
    all_students = sorted(list(database.keys()))
    
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    
    rows = []
    header = []
    
    if not os.path.exists(filepath):
        # Initialize file
        header = ["Names", date_str, "Percentage of total lectures attended"]
        rows.append(header)
        for student in all_students:
            status = "P" if student in present_students else "A"
            pct_str = "100.0%" if status == "P" else "0.0%"
            rows.append([student, status, pct_str])
    else:
        # Load existing file
        with open(filepath, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        if not rows:
            header = ["Names", date_str, "Percentage of total lectures attended"]
            rows.append(header)
        else:
            header = rows[0]
            
        # Determine index of the percentage column
        pct_col_name = "Percentage of total lectures attended"
        if pct_col_name in header:
            pct_col_idx = header.index(pct_col_name)
        else:
            pct_col_idx = len(header)
            header.append(pct_col_name)
            
        # Determine index of today's date column
        if date_str in header:
            date_col_idx = header.index(date_str)
            new_col_inserted = False
        else:
            # Insert date column right before the percentage column
            header.insert(pct_col_idx, date_str)
            date_col_idx = pct_col_idx
            new_col_inserted = True
            
        # Update existing student rows
        existing_names = set()
        for row in rows[1:]:
            if not row or len(row) == 0:
                continue
            student_name = row[0]
            existing_names.add(student_name)
            
            # If a new date column was inserted, insert placeholder cell
            if new_col_inserted:
                row.insert(date_col_idx, "A")
                
            # Set today's presence status
            if student_name in present_students:
                row[date_col_idx] = "P"
            else:
                row[date_col_idx] = "A"
                
        # Append rows for new registered students who are not in the CSV yet
        for student in all_students:
            if student not in existing_names:
                new_row = [student]
                # Populate each date column up to the percentage column
                for col_idx in range(1, len(header) - 1):
                    if col_idx == date_col_idx:
                        new_row.append("P" if student in present_students else "A")
                    else:
                        new_row.append("A") # Absent for past dates
                new_row.append("") # Placeholder for percentage
                rows.append(new_row)
                
        # Recalculate percentages for all rows
        for row in rows[1:]:
            if not row or len(row) == 0:
                continue
            # Total date columns is header length minus 2 (excluding Name and Pct columns)
            total_days = len(header) - 2
            p_count = row[1:-1].count("P")
            pct = (p_count / total_days) * 100 if total_days > 0 else 0.0
            row[-1] = f"{pct:.1f}%"
            
    # Write the entire grid back to the CSV file
    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        
    print(f"[INFO] Exported matrix attendance to {filepath}")


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
