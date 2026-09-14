"""MVP settings. ROI coordinates refer to the resized, processed image."""

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_PATH = "yolov8n-oiv7.pt"
MODEL_HF_REPO = "Blue2020Panda/YOLOv8nOIV7"
MODEL_HF_REVISION = "6085b73632b3e7e56ab64d60b1623d38679bf120"
TRACKER_CONFIG = "bytetrack.yaml"
CONF_THRESHOLD = 0.35
# A product may have both a food label and a packaging label in Open Images.
# Suppress highly overlapping boxes across classes before assigning track IDs.
AGNOSTIC_NMS = True
NMS_IOU_THRESHOLD = 0.70

FOOD_CLASS_GROUPS = {
    "Fruit and berries": {
        "Apple", "Banana", "Orange", "Lemon", "Pear", "Peach", "Grape",
        "Grapefruit", "Mango", "Pineapple", "Pomegranate", "Strawberry",
        "Watermelon", "Cantaloupe", "Coconut", "Common fig",
    },
    "Vegetables and mushrooms": {
        "Tomato", "Cucumber", "Potato", "Carrot", "Bell pepper", "Broccoli",
        "Cabbage", "Pumpkin", "Radish", "Zucchini", "Garden Asparagus",
        "Artichoke", "Mushroom",
    },
    "Bread and bakery": {
        "Bread", "Croissant", "Bagel", "Muffin", "Cookie", "Pretzel",
        "Waffle", "Pancake", "Cake", "Tart",
    },
    "Prepared food": {
        "Pizza", "Sandwich", "Hamburger", "Hot dog", "Sushi", "Pasta",
        "Salad", "Burrito", "Taco", "French fries",
    },
    "Dairy and eggs": {"Milk", "Cheese", "Cream", "Egg (Food)", "Dairy Product"},
    "Sweets and drinks": {"Candy", "Ice cream", "Popcorn", "Juice", "Tea", "Coffee"},
    "Packaging": {"Bottle", "Box", "Tin can", "Plastic bag", "Container"},
}
# Keep the existing setting name; it now includes the requested packaging classes.
FOOD_CLASSES = frozenset(name for group in FOOD_CLASS_GROUPS.values() for name in group)

# (left, top, right, bottom), in pixels, with origin at the top left.
BAG_ROI = (220, 140, 420, 380)
BAG_OVERLAP_THRESHOLD = 0.30
MIN_INSIDE_FRAMES = 2
TRACK_TTL_FRAMES = 60
PACKED_BANNER_FRAMES = 30
MAX_PACKED_OVERLAY_ROWS = 8
MAX_FRAME_BYTES = 2_000_000
