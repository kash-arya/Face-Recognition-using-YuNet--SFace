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
        engine.distance_threshold = 1.19
        engine.compute_distance = MagicMock(return_value=0.5)

        database = {
            "Alice": [np.random.randn(1, 128).astype(np.float32)],
            "Bob": [np.random.randn(1, 128).astype(np.float32)],
        }

        query = np.random.randn(1, 128).astype(np.float32)
        name, distance = engine.match_face(query, database)

        assert name == "Alice"
        assert distance == 0.5

    def test_match_face_below_threshold_returns_unknown(self, engine):
        engine.distance_threshold = 1.19
        engine.compute_distance = MagicMock(return_value=2.5)

        database = {
            "Alice": [np.random.randn(1, 128).astype(np.float32)],
        }

        query = np.random.randn(1, 128).astype(np.float32)
        name, distance = engine.match_face(query, database)

        assert name == "Unknown"
        assert distance == 2.5

    def test_match_face_empty_database(self, engine):
        query = np.random.randn(1, 128).astype(np.float32)
        name, distance = engine.match_face(query, {})

        assert name == "Unknown"
        assert distance == float('inf')

    def test_match_face_multiple_embeddings_per_student(self, engine):
        engine.distance_threshold = 1.19

        call_count = [0]
        def distance_side_effect(e1, e2):
            call_count[0] += 1
            # Return decreasing distances: first 1.5, then 0.8, then 1.0
            return {1: 1.5, 2: 0.8, 3: 1.0}.get(call_count[0], 0.9)

        engine.compute_distance = MagicMock(side_effect=distance_side_effect)

        database = {
            "Alice": [
                np.random.randn(1, 128).astype(np.float32),
                np.random.randn(1, 128).astype(np.float32),
                np.random.randn(1, 128).astype(np.float32),
            ],
        }

        query = np.random.randn(1, 128).astype(np.float32)
        name, distance = engine.match_face(query, database)

        assert name == "Alice"
        assert distance == 0.8  # best (lowest) among the three
