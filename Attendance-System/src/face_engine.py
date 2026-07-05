import cv2
import numpy as np

class FaceRecognitionEngine:
    def __init__(self, yunet_path, sface_path, score_threshold=0.9, nms_threshold=0.3, distance_threshold=1.17):
        self.yunet_path = yunet_path
        self.sface_path = sface_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.distance_threshold = distance_threshold
        
        # Instantiate detector placeholder. Input size needs to be set dynamically.
        self.detector = None
        self.detector_input_size = (0, 0)
        
        # Instantiate recognizer
        print("[INFO] Initializing SFace face recognizer...")
        self.recognizer = cv2.FaceRecognizerSF.create(model=sface_path, config="")
        
    def _get_detector(self, width, height):
        """Returns the FaceDetectorYN detector, updating the input size if necessary."""
        if self.detector is None or self.detector_input_size != (width, height):
            print(f"[INFO] Setting YuNet input size to: {width}x{height}")
            self.detector = cv2.FaceDetectorYN.create(
                model=self.yunet_path,
                config="",
                input_size=(width, height),
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold
            )
            self.detector_input_size = (width, height)
        return self.detector

    def detect_faces(self, image):
        """Detects faces in the image. Returns a tuple: (success_bool, faces_array)."""
        h, w = image.shape[:2]
        detector = self._get_detector(w, h)
        retval, faces = detector.detect(image)
        if retval and faces is not None:
            return True, faces
        return False, np.array([])

    def extract_embedding(self, image, face):
        """Aligns and crops the face, then extracts the 128-dimensional embedding vector."""
        aligned_face = self.recognizer.alignCrop(image, face)
        embedding = self.recognizer.feature(aligned_face)
        return embedding

    def compute_distance(self, embedding1, embedding2):
        """Computes the L2 (Euclidean) distance between two face embeddings.
        
        Lower distance means the faces are more similar.
        """
        return self.recognizer.match(embedding1, embedding2, cv2.FaceRecognizerSF_FR_NORM_L2)

    def match_face(self, query_embedding, database_encodings):
        """Compares the query embedding against the database of enrolled student encodings.
        
        database_encodings is a dict: {student_name: [embedding1, embedding2, ...]}
        Returns the best matched student name and the matching distance score.
        If no match is within the distance threshold, returns ("Unknown", best_distance).
        """
        best_name = "Unknown"
        best_distance = float('inf')
        
        for student_name, student_embeddings in database_encodings.items():
            for ref_embedding in student_embeddings:
                dist = self.compute_distance(query_embedding, ref_embedding)
                if dist < best_distance:
                    best_distance = dist
                    if dist <= self.distance_threshold:
                        best_name = student_name
                        
        return best_name, best_distance
