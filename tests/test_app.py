"""Exercise HTTP, WebSocket, JPEG and filtering without loading YOLO weights."""

import base64
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
from fastapi.testclient import TestClient
import numpy as np

import app as application
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
    names = {0: "person", 46: "banana", 47: "apple", 63: "laptop"}

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
            [(bbox, 47, 7), (bbox, 0, 1), (bbox, 63, 2)]
            for bbox in [OUTSIDE, INSIDE, INSIDE, INSIDE]
        ]
        with self.client.websocket_connect("/ws/detect") as ws:
            for expected_total in [0, 0, 1, 1]:
                ws.send_bytes(jpeg())
                result = ws.receive_json()
                self.assertEqual(result["visible_counts"], {"apple": 1})
                self.assertEqual(result["packed_total"], expected_total)
                self.assertEqual(result["session_version"], 0)
            self.assertEqual(result["events"], [])
            self.assertEqual(result["last_event"]["track_id"], 7)
            image = cv2.imdecode(np.frombuffer(base64.b64decode(result["image"]), np.uint8), 1)
            self.assertEqual(image.shape, (480, 640, 3))
            # The yellow ROI is present in the encoded frame.
            self.assertGreater(int(image[250, 220, 1]), 150)
            self.assertEqual(self.client.get("/api/session").json()["packed_counts"], {"apple": 1})
            reset = self.client.post("/api/reset").json()
            self.assertEqual(reset["packed_counts"], {})
            self.assertEqual(reset["event_count"], 0)
            self.assertIsNone(reset["last_event"])
            self.assertEqual(reset["session_version"], 1)
            ws.send_bytes(jpeg())
            self.assertEqual(ws.receive_json()["session_version"], 1)
        for call in self.model.calls:
            self.assertEqual(call["classes"], [46, 47])
            self.assertTrue(call["persist"])
            self.assertEqual(call["tracker"], "bytetrack.yaml")

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
            FrameProcessor(SimpleNamespace(names={0: "person"}))


if __name__ == "__main__":
    unittest.main()
