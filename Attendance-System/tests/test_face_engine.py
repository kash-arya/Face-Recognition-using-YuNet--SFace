import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.face_engine import FaceRecognitionEngine


class TestFaceEngine:
    @pytest.fixture
    def engine(self):
        with patch('src.face_engine.cv2.FaceRecognizerSF') as mock_recognizer:
            mock_recognizer.create.return_value = MagicMock()
            engine = FaceRecognitionEngine("fake_yunet.onnx", "fake_sface.onnx")
            engine.detector = MagicMock()
            return engine

    def test_match_face_exact_match(self, engine):
        engine.distance_threshold = 1.10
        engine.compute_distance = MagicMock(return_value=0.5)

        database = {
            "101": [np.random.randn(1, 128).astype(np.float32)],
            "102": [np.random.randn(1, 128).astype(np.float32)],
        }

        query = np.random.randn(1, 128).astype(np.float32)
        roll, distance = engine.match_face(query, database)

        assert roll == "101"
        assert distance == 0.5

    def test_match_face_below_threshold_returns_unknown(self, engine):
        engine.distance_threshold = 1.10
        engine.compute_distance = MagicMock(return_value=2.5)

        database = {
            "101": [np.random.randn(1, 128).astype(np.float32)],
        }

        query = np.random.randn(1, 128).astype(np.float32)
        roll, distance = engine.match_face(query, database)

        assert roll == "Unknown"
        assert distance == 2.5

    def test_match_face_empty_database(self, engine):
        query = np.random.randn(1, 128).astype(np.float32)
        name, distance = engine.match_face(query, {})

        assert name == "Unknown"
        assert distance == float('inf')

    def test_match_face_multiple_embeddings_per_student(self, engine):
        engine.distance_threshold = 1.10

        call_count = [0]
        def distance_side_effect(e1, e2):
            call_count[0] += 1
            # Return decreasing distances: first 1.5, then 0.8, then 1.0
            return {1: 1.5, 2: 0.8, 3: 1.0}.get(call_count[0], 0.9)

        engine.compute_distance = MagicMock(side_effect=distance_side_effect)

        database = {
            "101": [
                np.random.randn(1, 128).astype(np.float32),
                np.random.randn(1, 128).astype(np.float32),
                np.random.randn(1, 128).astype(np.float32),
            ],
        }

        query = np.random.randn(1, 128).astype(np.float32)
        roll, distance = engine.match_face(query, database)

        assert roll == "101"
        assert distance == 0.8  # best (lowest) among the three


def _make_face(x=50, y=50, w=100, h=100,
               re=(80, 70), le=(120, 70),
               nose=(100, 100),
               confidence=0.95):
    """Build a synthetic YuNet face row (15 floats).

    Layout: [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y,
             mouth_right_x, mouth_right_y, mouth_left_x, mouth_left_y, score]
    """
    row = np.array([
        x, y, w, h,
        re[0], re[1], le[0], le[1],
        nose[0], nose[1],
        90, 130, 110, 130,  # mouth landmarks (unused by quality checks)
        confidence,
    ], dtype=np.float32)
    return row


class TestFaceQuality:
    @pytest.fixture
    def engine(self):
        with patch('src.face_engine.cv2.FaceRecognizerSF') as mock_recognizer:
            mock_recognizer.create.return_value = MagicMock()
            eng = FaceRecognitionEngine("fake_yunet.onnx", "fake_sface.onnx", min_pitch_ratio=0.05)
            eng.detector = MagicMock()
            return eng

    def test_good_face_passes(self, engine):
        """Frontal face: eyes level, nose clearly below eye line."""
        # avg_eye_y = 70, nose_y = 110, face_height = 100
        # pitch ratio = (110 - 70) / 100 = 0.40 → passes
        face = _make_face(re=(80, 70), le=(120, 70), nose=(100, 110))
        ok, reason = engine.check_face_quality(face)
        assert ok, f"Expected pass, got: {reason}"

    def test_small_face_rejected(self, engine):
        """Face bounding box smaller than min_face_size (60px) must fail."""
        face = _make_face(w=40, h=40)
        ok, reason = engine.check_face_quality(face)
        assert not ok
        assert "too small" in reason

    def test_tilt_rejected(self, engine):
        """Large inter-eye vertical offset → tilt exceeds 30°."""
        # dy=100, dx=40 → angle ≈ 68° → rejected
        face = _make_face(re=(80, 50), le=(120, 150))
        ok, reason = engine.check_face_quality(face)
        assert not ok
        assert "tilt" in reason.lower()

    def test_pitch_rejected_nose_too_close_to_eyes(self, engine):
        """Nose nearly at eye level → pitch ratio < 0.05 → rejected."""
        # avg_eye_y = 70, nose_y = 73, face_height = 100
        # ratio = (73 - 70) / 100 = 0.03 < 0.05 → rejected
        face = _make_face(re=(80, 70), le=(120, 70), nose=(100, 73))
        ok, reason = engine.check_face_quality(face)
        assert not ok
        assert "pitch" in reason.lower()

    def test_pitch_borderline_at_threshold_rejected(self, engine):
        """Ratio exactly equal to threshold (0.05 - epsilon) must be rejected."""
        # avg_eye_y = 70, face_height = 100, nose_y giving ratio=0.049
        nose_y = 70 + 0.049 * 100  # = 74.9
        face = _make_face(re=(80, 70), le=(120, 70), nose=(100, nose_y))
        ok, reason = engine.check_face_quality(face)
        assert not ok
        assert "pitch" in reason.lower()

    def test_check_pitch_passes_with_good_ratio(self, engine):
        """_check_pitch helper independently accepts a good frontal face."""
        face = _make_face(re=(80, 70), le=(120, 70), nose=(100, 120))
        ok, reason = engine._check_pitch(face)
        assert ok
        assert reason == ""

    def test_check_tilt_passes_level_eyes(self, engine):
        """_check_tilt helper passes when eyes are perfectly horizontal."""
        face = _make_face(re=(80, 70), le=(120, 70))
        ok, reason = engine._check_tilt(face)
        assert ok
        assert reason == ""

    def test_composition_order_size_before_tilt(self, engine):
        """check_face_quality must report 'too small' even when tilt also fails.

        Verifies the gate order: size → tilt → pitch. The first failing check
        determines the returned reason; later checks are not reached.
        """
        # Small face (40x40) AND heavily tilted eyes — size must win
        face = _make_face(w=40, h=40, re=(80, 50), le=(120, 150))
        ok, reason = engine.check_face_quality(face)
        assert not ok
        assert "too small" in reason, f"Expected size reason first, got: {reason}"

    def test_composition_order_tilt_before_pitch(self, engine):
        """check_face_quality must report tilt even when pitch also fails.

        Verifies tilt is evaluated before pitch in the gate chain.
        """
        # Heavily tilted (dy=100) AND nose at eye level (pitch fails too)
        face = _make_face(re=(80, 50), le=(120, 150), nose=(100, 73))
        ok, reason = engine.check_face_quality(face)
        assert not ok
        assert "tilt" in reason.lower(), f"Expected tilt reason before pitch, got: {reason}"
