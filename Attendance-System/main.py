import os
import sys
import argparse
import cv2
import numpy as np

# Ensure src directory is in the import path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from face_engine import FaceRecognitionEngine
from utils import load_models, save_encodings, load_encodings, save_attendance, draw_face_annotations

def cmd_register(args):
    # Load model paths (downloads if not present)
    yunet_path, sface_path = load_models(args.models_dir)
    
    # Initialize engine
    engine = FaceRecognitionEngine(yunet_path, sface_path)
    
    # Scan dataset directory
    dataset_dir = args.dataset_dir
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return
        
    encodings_db = {}
    
    # Walk through dataset subdirectories
    for item in os.listdir(dataset_dir):
        student_path = os.path.join(dataset_dir, item)
        if os.path.isdir(student_path):
            student_name = item
            print(f"\n[INFO] Enrolling student: {student_name}")
            student_embeddings = []
            
            for file_name in os.listdir(student_path):
                img_path = os.path.join(student_path, file_name)
                if not os.path.isfile(img_path) or not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    continue
                    
                print(f"  Processing image: {file_name}")
                image = cv2.imread(img_path)
                if image is None:
                    print(f"  [WARNING] Could not read image: {img_path}")
                    continue
                    
                # Detect faces
                success, faces = engine.detect_faces(image)
                if not success or len(faces) == 0:
                    print(f"  [WARNING] No face detected in {file_name}. Skipping image.")
                    continue
                elif len(faces) > 1:
                    print(f"  [WARNING] Multiple faces ({len(faces)}) detected in {file_name}. Using the highest-confidence face.")
                
                # SFace extracts feature from the first face
                best_face = faces[0]
                # Find face with maximum score if multiple
                if len(faces) > 1:
                    # Face coordinate row contains score at index 14
                    best_face = max(faces, key=lambda f: f[14])
                    
                embedding = engine.extract_embedding(image, best_face)
                student_embeddings.append(embedding)
                
            if len(student_embeddings) > 0:
                encodings_db[student_name] = student_embeddings
                print(f"[SUCCESS] Registered {student_name} with {len(student_embeddings)} faces.")
            else:
                print(f"[WARNING] No valid face embeddings found for {student_name}.")
                
    # Save database to pickle file
    if encodings_db:
        save_encodings(encodings_db, args.encodings_file)
    else:
        print("[ERROR] No students registered. Database is empty.")

def cmd_recognize_single(args):
    # Load model paths
    yunet_path, sface_path = load_models(args.models_dir)
    
    # Initialize engine
    engine = FaceRecognitionEngine(
        yunet_path, 
        sface_path, 
        distance_threshold=args.threshold
    )
    
    # Load database
    database = load_encodings(args.encodings_file)
    if not database:
        print("[ERROR] No student encodings found. Please run the 'register' command first.")
        return
        
    # Read image
    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image not found: {args.image_path}")
        return
        
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {args.image_path}")
        return
        
    # Detect faces
    success, faces = engine.detect_faces(image)
    if not success or len(faces) == 0:
        print("[RESULT] No face detected in the image.")
        return
        
    print(f"[INFO] Detected {len(faces)} face(s) in {args.image_path}.")
    
    # Recognize face(s)
    for idx, face in enumerate(faces):
        embedding = engine.extract_embedding(image, face)
        name, distance = engine.match_face(embedding, database)
        score = face[14] # Detection confidence
        print(f"Face #{idx+1}: Result = {name} (Euclidean Distance: {distance:.3f}, Det Conf: {score:.3f})")

def cmd_attendance(args):
    # Load model paths
    yunet_path, sface_path = load_models(args.models_dir)
    
    # Initialize engine
    engine = FaceRecognitionEngine(
        yunet_path, 
        sface_path, 
        distance_threshold=args.threshold
    )
    
    # Load database
    database = load_encodings(args.encodings_file)
    if not database:
        print("[ERROR] No student encodings found. Please run the 'register' command first.")
        return
        
    # Read classroom image
    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image not found: {args.image_path}")
        return
        
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {args.image_path}")
        return
        
    # Detect all faces
    success, faces = engine.detect_faces(image)
    total_headcount = len(faces) if (success and faces is not None) else 0
    
    present_students = set()
    match_names = []
    confidences = []
    
    if total_headcount > 0:
        for face in faces:
            embedding = engine.extract_embedding(image, face)
            name, distance = engine.match_face(embedding, database)
            
            match_names.append(name)
            confidences.append(distance)
            
            if name != "Unknown":
                present_students.add(name)
                
    # Sort for consistent output
    present_list = sorted(list(present_students))
    
    # Output results in requested format
    print("\n--- ATTENDANCE REPORT ---")
    print(f"Total Headcount: {total_headcount}")
    print("Present Students:")
    if present_list:
        for name in present_list:
            print(f" - {name}")
    else:
        print("  None")
    print("-------------------------\n")
    
    # Log to CSV (always runs to log absences)
    save_attendance(present_list, args.attendance_file, args.encodings_file)
        
    # Save visual verification image
    if total_headcount > 0 and args.output_image:
        output_path = args.output_image
        # If it is the default path, make it timestamped in the verification directory
        if output_path == "Attendance-System/output/annotated_attendance.jpeg":
            from datetime import datetime
            date_str = datetime.now().strftime("%d-%m-%Y")
            output_path = f"Attendance-System/output/annotated_attendance_{date_str}.jpeg"
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        annotated_img = draw_face_annotations(image, faces, match_names, confidences)
        cv2.imwrite(output_path, annotated_img)
        print(f"[INFO] Saved annotated verification image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Based Face Recognition Attendance System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Register subcommand
    reg_parser = subparsers.add_parser("register", help="Scan student dataset and save face embeddings")
    reg_parser.add_argument("--dataset-dir", default="Attendance-System/dataset", help="Path to student face photos directory")
    reg_parser.add_argument("--models-dir", default="Attendance-System/models", help="Directory to save/load model ONNX weights")
    reg_parser.add_argument("--encodings-file", default="Attendance-System/data/encodings.pkl", help="Output path for serialized embeddings database")
    
    # Recognize Single subcommand
    rec_parser = subparsers.add_parser("recognize-single", help="Recognize a single face image")
    rec_parser.add_argument("image_path", help="Path to face image to identify")
    rec_parser.add_argument("--models-dir", default="Attendance-System/models", help="Directory to load model weights")
    rec_parser.add_argument("--encodings-file", default="Attendance-System/data/encodings.pkl", help="Path to serialized embeddings database")
    rec_parser.add_argument("--threshold", type=float, default=1.128, help="Euclidean distance threshold (lower = stricter match)")
    
    # Attendance subcommand
    att_parser = subparsers.add_parser("attendance", help="Mark classroom attendance from group/selfie photo")
    att_parser.add_argument("image_path", help="Path to classroom group photo")
    att_parser.add_argument("--models-dir", default="Attendance-System/models", help="Directory to load model weights")
    att_parser.add_argument("--encodings-file", default="Attendance-System/data/encodings.pkl", help="Path to serialized embeddings database")
    att_parser.add_argument("--attendance-file", default="Attendance-System/data/attendance.csv", help="Output path for attendance spreadsheet")
    att_parser.add_argument("--output-image", default="Attendance-System/output/annotated_attendance.jpeg", help="Output path for marked verification photo")
    att_parser.add_argument("--threshold", type=float, default=1.128, help="Euclidean distance threshold (lower = stricter match)")
    
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
