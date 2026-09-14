"""MVP settings. ROI coordinates refer to the resized, processed image."""

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_PATH = "yolo11n.pt"
TRACKER_CONFIG = "bytetrack.yaml"
CONF_THRESHOLD = 0.35

FOOD_CLASSES = {
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
}

# (left, top, right, bottom), in pixels, with origin at the top left.
BAG_ROI = (220, 140, 420, 380)
BAG_OVERLAP_THRESHOLD = 0.30
MIN_INSIDE_FRAMES = 2
TRACK_TTL_FRAMES = 60
PACKED_BANNER_FRAMES = 30
MAX_FRAME_BYTES = 2_000_000
