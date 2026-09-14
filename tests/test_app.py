"""Exercise HTTP, WebSocket, JPEG and filtering without loading YOLO weights."""

import base64
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
from fastapi.testclient import TestClient
import numpy as np

import app as application
from config import FOOD_CLASSES, MODEL_HF_REPO, MODEL_HF_REVISION, NMS_IOU_THRESHOLD
from vision import FrameProcessor


class Array:
    """Minimal tensor interface used by the YOLO result adapter."""
    def __init__(self, values):
        self.values = values

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    # Deliberately different IDs from COCO/OIV7: the adapter must use model.names.
    names = {1000 + index: name for index, name in enumerate(sorted(FOOD_CLASSES))}
    names.update({0: "Person", 63: "Laptop"})

    def __init__(self):
        self.frames = []
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        rows = self.frames.pop(0) if self.frames else []
        boxes = SimpleNamespace(
            xyxy=Array([r[0] for r in rows]),
            cls=Array([r[1] for r in rows]),
            id=Array([r[2] for r in rows]) if rows else None,
        )
        return [SimpleNamespace(boxes=boxes, names=self.names)]


def jpeg():
    return cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()


OUTSIDE = (80, 180, 120, 220)
INSIDE = (260, 180, 300, 220)
CLASS_IDS = {name: class_id for class_id, name in FakeModel.names.items()}


class ModelLoadingTests(unittest.TestCase):
    def test_downloads_pinned_hf_checkpoint_when_local_file_is_missing(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "yolov8n-oiv7.pt"
            download = Mock(return_value=str(model_path))
            model_factory = Mock()
            with patch.object(application, "MODEL_PATH", str(model_path)), patch.dict(
                sys.modules, {
                    "huggingface_hub": SimpleNamespace(hf_hub_download=download),
                    "ultralytics": SimpleNamespace(YOLO=model_factory),
                },
            ):
                application.load_model()
            download.assert_called_once_with(
                repo_id=MODEL_HF_REPO, filename="yolov8n-oiv7.pt",
                revision=MODEL_HF_REVISION, local_dir=model_path.parent, token=False,
            )
            model_factory.assert_called_once_with(str(model_path))

    def test_existing_weights_load_without_network(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "yolov8n-oiv7.pt"
            model_path.touch()
            download = Mock(side_effect=AssertionError("Unexpected network access"))
            model_factory = Mock()
            with patch.object(application, "MODEL_PATH", str(model_path)), patch.dict(
                sys.modules, {
                    "huggingface_hub": SimpleNamespace(hf_hub_download=download),
                    "ultralytics": SimpleNamespace(YOLO=model_factory),
                },
            ):
                application.load_model()
            download.assert_not_called()
            model_factory.assert_called_once_with(str(model_path))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.model_patch = patch.object(application, "load_model", return_value=self.model)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.client = TestClient(application.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_home_and_script(self):
        self.assertIn("LightStore", self.client.get("/").text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)

    def test_websocket_filtering_tracking_packing_and_reset(self):
        self.model.frames = [
            [(bbox, CLASS_IDS["Apple"], 7), (bbox, 0, 1), (bbox, 63, 2)]
            for bbox in [OUTSIDE, INSIDE, INSIDE, INSIDE]
        ]
        with self.client.websocket_connect("/ws/detect") as ws:
            for expected_total in [0, 0, 1, 1]:
                ws.send_bytes(jpeg())
                result = ws.receive_json()
                self.assertEqual(result["visible_counts"], {"Apple": 1})
                self.assertEqual(result["packed_total"], expected_total)
                self.assertEqual(result["session_version"], 0)
            self.assertEqual(result["events"], [])
            self.assertEqual(result["last_event"]["track_id"], 7)
            image = cv2.imdecode(np.frombuffer(base64.b64decode(result["image"]), np.uint8), 1)
            self.assertEqual(image.shape, (480, 640, 3))
            # The yellow ROI is present in the encoded frame.
            self.assertGreater(int(image[250, 220, 1]), 150)
            self.assertEqual(self.client.get("/api/session").json()["packed_counts"], {"Apple": 1})
            reset = self.client.post("/api/reset").json()
            self.assertEqual(reset["packed_counts"], {})
            self.assertEqual(reset["event_count"], 0)
            self.assertIsNone(reset["last_event"])
            self.assertEqual(reset["session_version"], 1)
            ws.send_bytes(jpeg())
            self.assertEqual(ws.receive_json()["session_version"], 1)
        for call in self.model.calls:
            self.assertEqual(
                {self.model.names[class_id] for class_id in call["classes"]}, FOOD_CLASSES,
            )
            self.assertTrue(call["persist"])
            self.assertEqual(call["tracker"], "bytetrack.yaml")
            self.assertTrue(call["agnostic_nms"])
            self.assertEqual(call["iou"], NMS_IOU_THRESHOLD)

    def test_all_enabled_product_and_packaging_classes_reach_websocket_counts(self):
        rows = [(name, CLASS_IDS[name]) for name in sorted(FOOD_CLASSES)]
        self.model.frames = [
            [(bbox, class_id, class_id) for _, class_id in rows]
            for bbox in [OUTSIDE, INSIDE, INSIDE, INSIDE]
        ]
        with self.client.websocket_connect("/ws/detect") as ws:
            for total in [0, 0, len(rows), len(rows)]:
                ws.send_bytes(jpeg())
                result = ws.receive_json()
                self.assertEqual(result["visible_counts"], {name: 1 for name, _ in rows})
                self.assertEqual(result["packed_total"], total)
            self.assertEqual(result["packed_counts"], {name: 1 for name, _ in rows})
            self.assertEqual(result["events"], [])

    def test_partial_class_match_fails_instead_of_silently_dropping_products(self):
        names = {key: value for key, value in FakeModel.names.items() if value != "Milk"}
        with self.assertRaisesRegex(ValueError, "missing configured FOOD_CLASSES: Milk"):
            FrameProcessor(SimpleNamespace(names=names))

    def test_malformed_frame_does_not_kill_stream(self):
        with self.client.websocket_connect("/ws/detect") as ws:
            ws.send_bytes(b"not a jpeg")
            self.assertIn("error", ws.receive_json())
            ws.send_bytes(jpeg())
            self.assertEqual(ws.receive_json()["visible_counts"], {})

    def test_concurrent_camera_rejected_and_slot_released(self):
        with self.client.websocket_connect("/ws/detect") as first:
            # Receive a response before connecting the second camera.
            first.send_bytes(jpeg())
            first.receive_json()
            with self.client.websocket_connect("/ws/detect") as second:
                self.assertIn("Another camera", second.receive_json()["error"])
            first.send_bytes(jpeg())
            self.assertIn("image", first.receive_json())
        with self.client.websocket_connect("/ws/detect") as third:
            third.send_bytes(jpeg())
            self.assertIn("image", third.receive_json())

    def test_no_matching_model_classes_fails_instead_of_detecting_everything(self):
        with self.assertRaises(ValueError):
            FrameProcessor(SimpleNamespace(names={0: "Person"}))


if __name__ == "__main__":
    unittest.main()
