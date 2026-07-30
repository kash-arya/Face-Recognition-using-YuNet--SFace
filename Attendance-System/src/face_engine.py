import cv2
import numpy as np

class FaceRecognitionEngine:
    def __init__(self, yunet_path, sface_path, score_threshold=0.9, nms_threshold=0.3,
                 distance_threshold=1.10, min_pitch_ratio=0.05):
        self.yunet_path = yunet_path
        self.sface_path = sface_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.distance_threshold = distance_threshold

        self.min_face_size = 60
        self.max_tilt_degrees = 30
        self.min_pitch_ratio = min_pitch_ratio

        cv2.setNumThreads(4)

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

    def _check_tilt(self, face):
        """Checks head roll (in-plane rotation) using eye landmark positions.

        Returns (ok_bool, reason_string).
        Rejects faces where the inter-eye angle exceeds max_tilt_degrees.
        """
        re_x, re_y = face[4], face[5]  # right eye
        le_x, le_y = face[6], face[7]  # left eye

        dx = le_x - re_x
        dy = le_y - re_y
        angle_rad = np.arctan2(abs(dy), abs(dx))
        angle_deg = np.degrees(angle_rad)

        if angle_deg > self.max_tilt_degrees:
            return False, (
                f"Head tilt too high ({angle_deg:.1f}°, max {self.max_tilt_degrees}°) "
                "— use a front-facing photo"
            )
        return True, ""

    def _check_pitch(self, face):
        """Checks head pitch (nodding up/down) using nose and eye landmark positions.

        In image coordinates Y increases downward, so for a frontal face the nose sits
        below the eyes and (nose_y - avg_eye_y) / face_height is a positive ratio
        (~0.2–0.4).  When a person looks down the nose migrates toward the eye line
        and this ratio drops.  Reject if the ratio falls below min_pitch_ratio.

        YuNet landmark layout (face[4:10]):
            face[4], face[5]  — right eye (x, y)
            face[6], face[7]  — left eye  (x, y)
            face[8], face[9]  — nose tip  (x, y)

        Returns (ok_bool, reason_string).
        """
        face_height = face[3]
        if face_height <= 0:
            return False, "Invalid face bounding box (zero height)"

        avg_eye_y = (face[5] + face[7]) / 2.0
        nose_y = face[9]
        ratio = (nose_y - avg_eye_y) / face_height

        if ratio < self.min_pitch_ratio:
            direction = "looking up" if ratio < 0 else "looking down"
            return False, (
                f"Head pitch too extreme ({direction}, nose-to-eye ratio {ratio:.3f} < {self.min_pitch_ratio}) "
                "— use a front-facing photo"
            )
        return True, ""

    def check_face_quality(self, face):
        """Validates face quality for enrollment. Returns (ok_bool, reason_string).

        Checks (in order, short-circuiting on first failure):
        1. Face crop size  — too small faces produce unreliable embeddings
        2. Head tilt       — in-plane roll; tilted faces confuse alignment
        3. Head pitch      — up/down nod; downward-looking faces miss facial features

        The engine always performs these checks when this method is called.
        Callers (e.g. cmd_register) decide whether to invoke it; pass
        --no-quality-check at the CLI to skip the call entirely.
        """
        x, y, w, h = face[0:4].astype(int)

        if w < self.min_face_size or h < self.min_face_size:
            return False, f"Face too small ({w}x{h}px, need >= {self.min_face_size}x{self.min_face_size})"

        ok, reason = self._check_tilt(face)
        if not ok:
            return False, reason

        ok, reason = self._check_pitch(face)
        if not ok:
            return False, reason

        return True, ""

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

        database_encodings is a dict: {roll_number: [embedding1, embedding2, ...]}
        Returns the best matched roll number and the matching distance score.
        If no match is within the distance threshold, returns ("Unknown", best_distance).
        """
        best_key = "Unknown"
        best_distance = float('inf')

        for roll, student_embeddings in database_encodings.items():
            for ref_embedding in student_embeddings:
                dist = self.compute_distance(query_embedding, ref_embedding)
                if dist < best_distance:
                    best_distance = dist
                    if dist <= self.distance_threshold:
                        best_key = roll

        return best_key, best_distance
