# LightStore

An overhead-camera packing MVP built on the original FastAPI + browser WebSocket
prototype. YOLO11n detects food, ByteTrack gives each visible product a persistent
ID, and a small state machine counts transfers into a fixed bag zone. Annotated
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
The first launch downloads the official `yolo11n.pt` weights if absent. Internet
access is needed for installation and that initial download; model weights are
ignored by Git. Subsequent inference runs locally. ByteTrack's `lap` dependency
is installed explicitly with the other requirements.

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
| `PACKED_BANNER_FRAMES` | `30` | How long the latest event stays on the video |
| `MODEL_PATH` | `yolo11n.pt` | Existing YOLO11n model |
| `TRACKER_CONFIG` | `bytetrack.yaml` | Ultralytics ByteTrack configuration |

Coordinates start at the top-left corner: x increases to the right, y downward.
For the default ROI, the bag zone is 200 pixels wide and 240 pixels tall. Keep
`0 <= x1 < x2 <= FRAME_WIDTH` and `0 <= y1 < y2 <= FRAME_HEIGHT`. These coordinates
apply after resizing, independently of camera capture resolution. Adjust the
rectangle to match the real bag opening. Changing settings requires a reload.

`FOOD_CLASSES` contains exactly these COCO labels:

```python
{
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
}
```

Edit that set to change the allowed classes. Class IDs are resolved from the
model's names rather than hardcoded COCO IDs. Filtering happens both at inference
and before tracking-state updates/drawing; people, furniture, phones and other
classes are ignored. Detection boxes without an assigned track ID are withheld
until ByteTrack confirms them. `visible_counts` counts the displayed tracked
food objects. A track keeps its first observed food class to reduce label flicker.
For a future custom model, update `MODEL_PATH` and `FOOD_CLASSES` together.

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
  "visible_counts": {"apple": 1},
  "counts": {"apple": 1},
  "packed_counts": {"apple": 2, "banana": 1},
  "packed_total": 3,
  "event_count": 3,
  "session_version": 0,
  "last_event": {
    "track_id": 7,
    "class_name": "apple",
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
ID and verify no duplicate. Repeat with two products, test a non-food object,
and test Reset. Automated geometry tests cannot establish real-world detection
or tracking accuracy.

## Files

- `app.py`: FastAPI lifecycle, existing WebSocket route, session/reset endpoints.
- `config.py`: food labels, ROI and thresholds.
- `tracking.py`: dependency-free geometry, per-ID state, counts and event log.
- `vision.py`: YOLO tracking adapter, food filtering, annotation and session lock.
- `static/index.html`, `static/app.js`: original camera UI, capture loop, packed
  counters and reset controls.
- `tests/`: geometry, event, API and browser-controller regression tests.

## MVP limitations

- An ROI crossing is a proxy for packing; this does not verify that an item is
  actually inside a physical bag. Brief passes through the ROI can count once
  they meet the configured confirmation length.
- COCO has ten generic food labels here, not store SKUs. Lighting, occlusion,
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
