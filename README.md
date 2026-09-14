# LightStore

An overhead-camera packing MVP built on the original FastAPI + browser WebSocket
prototype. YOLOv8n pretrained on Open Images V7 detects 65 selected food and
packaging classes, ByteTrack gives each visible product a persistent ID, and a
small state machine counts transfers into a fixed bag zone. Annotated
JPEG frames and counters are returned over the existing `/ws/detect` connection.

## Setup and launch

Use Python 3.12 or newer (verified locally with Python 3.14 on macOS).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

On Windows, use WSL and the same commands above: the original frozen requirements
include `uvloop`, which is Unix-only.
The first launch uses `huggingface_hub` to download `yolov8n-oiv7.pt` (~7.2 MB)
from the community HF repository
[Blue2020Panda/YOLOv8nOIV7](https://huggingface.co/Blue2020Panda/YOLOv8nOIV7), pinned
to revision `6085b73632b3e7e56ab64d60b1623d38679bf120`. The checkpoint is saved
beside `app.py` and reused offline on subsequent starts. Internet access is needed
for installation and the initial download; no HF token or dataset download is
required. Model weights and download metadata are ignored by Git. ByteTrack's
`lap` dependency is installed explicitly with the other requirements.

Open <http://127.0.0.1:8000>, allow camera access, and click **Start camera**.
Browsers require localhost or HTTPS for camera capture. Point the camera down at
the packing table and align the physical bag opening with the yellow rectangle.
Start with products **outside** the rectangle, move one inside, and wait for the
green `PACKED` label before hiding it inside an opaque bag.

- **Stop** releases the camera and preserves session counts and track state.
- **Reset session** clears counts, events, packed-ID history and per-track packing
  state. ByteTrack IDs remain continuous. Products currently inside are not
  counted again until they have been observed outside and then enter the zone.
- The MVP supports **one active camera connection and one shared packing session
  per server process**. A second camera is rejected with a clear message. Use one
  Uvicorn worker. Reset affects this shared session, including from another tab.
- A server restart or development reload clears the in-memory session. Use Reset
  before changing the camera/table or starting a new packing demonstration.

## Configuration

All tuning lives in `config.py`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | `640`, `480` | Size after every incoming frame is resized |
| `BAG_ROI` | `(220, 140, 420, 380)` | Rectangle `(x1, y1, x2, y2)` in processed-image pixels |
| `BAG_OVERLAP_THRESHOLD` | `0.30` | Minimum fraction of the **object's** box inside the ROI |
| `MIN_INSIDE_FRAMES` | `2` | Consecutive observed inside frames needed to confirm packing |
| `TRACK_TTL_FRAMES` | `60` | Remove geometry after more than this many unseen processed frames |
| `CONF_THRESHOLD` | `0.35` | YOLO detection confidence |
| `AGNOSTIC_NMS` | `True` | Suppress overlapping detections across different classes |
| `NMS_IOU_THRESHOLD` | `0.70` | Box IoU above which the lower-confidence detection is suppressed |
| `PACKED_BANNER_FRAMES` | `30` | How long the latest event stays on the video |
| `MAX_PACKED_OVERLAY_ROWS` | `8` | Maximum class rows on the video; the sidebar keeps all counts |
| `MODEL_PATH` | `yolov8n-oiv7.pt` | Open Images V7 checkpoint, relative to `app.py` or absolute |
| `MODEL_HF_REPO`, `MODEL_HF_REVISION` | See `config.py` | Pinned source used when the local checkpoint is absent |
| `TRACKER_CONFIG` | `bytetrack.yaml` | Ultralytics ByteTrack configuration |

Coordinates start at the top-left corner: x increases to the right, y downward.
For the default ROI, the bag zone is 200 pixels wide and 240 pixels tall. Keep
`0 <= x1 < x2 <= FRAME_WIDTH` and `0 <= y1 < y2 <= FRAME_HEIGHT`. These coordinates
apply after resizing, independently of camera capture resolution. Adjust the
rectangle to match the real bag opening. Changing settings requires a reload.

`FOOD_CLASSES` is built from `FOOD_CLASS_GROUPS` and includes **all 65 requested
classes**, including packaging. Names are case-sensitive and match the checkpoint:

| Group | Enabled Open Images V7 labels |
| --- | --- |
| Fruit and berries (16) | Apple, Banana, Orange, Lemon, Pear, Peach, Grape, Grapefruit, Mango, Pineapple, Pomegranate, Strawberry, Watermelon, Cantaloupe, Coconut, Common fig |
| Vegetables and mushrooms (13) | Tomato, Cucumber, Potato, Carrot, Bell pepper, Broccoli, Cabbage, Pumpkin, Radish, Zucchini, Garden Asparagus, Artichoke, Mushroom |
| Bread and bakery (10) | Bread, Croissant, Bagel, Muffin, Cookie, Pretzel, Waffle, Pancake, Cake, Tart |
| Prepared food (10) | Pizza, Sandwich, Hamburger, Hot dog, Sushi, Pasta, Salad, Burrito, Taco, French fries |
| Dairy and eggs (5) | Milk, Cheese, Cream, Egg (Food), Dairy Product |
| Sweets and drinks (6) | Candy, Ice cream, Popcorn, Juice, Tea, Coffee |
| Packaging (5) | Bottle, Box, Tin can, Plastic bag, Container |

Edit the groups to change the allowed classes. Class IDs are resolved from the
model's names rather than hardcoded dataset IDs. Startup rejects a checkpoint
missing any enabled class, so an accidental switch back to COCO cannot silently
disable products. See the [full upstream class list](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/open-images-v7.yaml).

Filtering happens both at inference and before tracking-state updates/drawing;
people, furniture, phones and all other unselected classes are ignored. Detection
boxes without a track ID are withheld until ByteTrack confirms them.
`visible_counts` counts displayed tracked objects, including packaging. A track
keeps its first observed enabled class to reduce label flicker. Labels retain
their Open Images capitalization in the video and JSON responses.

Open Images includes overlapping meanings such as `Milk`, `Dairy Product` and
`Bottle`. Class-agnostic NMS suppresses highly overlapping boxes before ByteTrack,
keeping the highest-confidence label to reduce duplicate tracks for one object.
This is a heuristic: nearby overlapping products may be suppressed, and a larger
package enclosing a smaller visible product can still receive a separate ID.
Packaging follows exactly the same outside → inside packing rule as food.

For a custom checkpoint, place it locally and update `MODEL_PATH` and the class
groups together; update the HF source as well if it should be downloaded remotely.

## Packing state machine

A valid box is **inside** if its center is inside/on the bag rectangle **OR** its
intersection area with the ROI is at least 30% of its own area. This is not IoU.
Zero-area, inverted or non-finite boxes are ignored.

1. **Unarmed:** a new track seen inside has no evidence of transfer. It is not
   counted, however long it stays inside.
2. **Outside / armed:** once the track is observed outside, it can start a transfer.
3. **Entering:** each consecutive observed inside frame advances confirmation.
   Returning outside or missing a frame clears this streak. A missing frame does
   not erase the previously observed outside position unless the track expires.
4. **Packed:** after `MIN_INSIDE_FRAMES`, emit one timestamped event for this ID,
   increment its class count and mark it packed. Staying inside, disappearing or
   exiting/re-entering with the same ID does not emit another event.

Stale track geometry is pruned. Already packed IDs are retained separately until
Reset, so pruning cannot accidentally allow a duplicate for the same ID. An
expired **unpacked** ID must be observed outside again. All counters, model
inference and reset operations are serialized to prevent concurrent updates.
Frame counts refer to processed frames, not elapsed seconds.

## API

- `GET /` — existing browser camera frontend.
- `GET /api/session` — packed counts, total, event count, latest event and session version.
- `POST /api/reset` — reset and return the cleared session.
- `WS /ws/detect` — send binary camera JPEGs, receive JSON with annotated JPEGs.

Example WebSocket response (image omitted):

```json
{
  "image": "<base64 JPEG>",
  "visible_counts": {"Apple": 1},
  "counts": {"Apple": 1},
  "packed_counts": {"Apple": 2, "Banana": 1},
  "packed_total": 3,
  "event_count": 3,
  "session_version": 0,
  "last_event": {
    "track_id": 7,
    "class_name": "Apple",
    "event": "packed",
    "timestamp": "2026-09-14T19:00:00+00:00",
    "frame_number": 42
  },
  "events": []
}
```

`counts` is a compatibility alias for `visible_counts`. `events` contains only
new events from this frame, including simultaneous transfers; `last_event` is
the latest session event or `null`. The full chronological log is retained as
`PackingTracker.packed_events` in memory. `session_version` increments on Reset
so the browser can ignore late responses from before that reset. The client
keeps one frame in flight to avoid building up a video-processing queue.

## Tests

The geometry/event layer uses only Python's standard library and never imports
YOLO or OpenCV:

```bash
python -m unittest discover -s tests -p 'test_tracking.py' -v
```

After installing requirements, run all Python tests. The API tests use a fake
model and real JPEG encoding/decoding; they do not download or load weights.

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py config.py tracking.py vision.py tests
```

The optional browser-controller tests use Node's built-in runner, with no npm
packages or physical camera required:

```bash
node --test tests/frontend.test.cjs
node --check static/app.js
```

For a real-camera acceptance check, move one apple outside → inside, hold it for
several frames, and verify one event. Move it out and back with the same displayed
ID and verify no duplicate. Repeat with two products, a bottle or box, an excluded
object such as a phone, and Reset. Automated geometry tests cannot establish real-world detection
or tracking accuracy.

## Files

- `app.py`: HF checkpoint loading, FastAPI lifecycle, WebSocket and session/reset endpoints.
- `config.py`: grouped product/packaging labels, checkpoint source, ROI and thresholds.
- `tracking.py`: dependency-free geometry, per-ID state, counts and event log.
- `vision.py`: YOLO tracking adapter, class validation/filtering, NMS, annotation and session lock.
- `static/index.html`, `static/app.js`: original camera UI, capture loop, packed
  counters and reset controls.
- `tests/`: geometry, event, API and browser-controller regression tests.

## MVP limitations

- An ROI crossing is a proxy for packing; this does not verify that an item is
  actually inside a physical bag. Brief passes through the ROI can count once
  they meet the configured confirmation length.
- The 65 enabled Open Images classes are generic categories, not store SKUs.
  Brand, size and flavour are not distinguished. A food label does not guarantee
  recognition through a closed package; packaging labels describe its shape,
  not its contents. Lighting, occlusion,
  fast motion and overlapping products can cause missed detections or ID swaps.
  A physical product that acquires a new ID can be counted again after a new
  outside → inside transition; exact-once applies to a track ID within a session.
- If a product disappears before enough inside frames are observed, it is not
  counted. No event is inferred from disappearance alone.
- This is a local, in-memory demonstration, with no authentication, persistence,
  multiple workers, order database, segmentation, hand/pose detection, scales,
  barcodes or custom training. It does not subtract items when they leave the bag.
- Follow the existing Ultralytics/PyTorch platform requirements. GPU acceleration
  is optional; achievable frame rate depends on the computer.
