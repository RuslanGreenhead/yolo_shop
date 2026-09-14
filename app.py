import asyncio
import base64

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Smallest YOLO11 detection model: a good default for live camera inference.
model = YOLO("yolo11n.pt")


@app.get("/")
async def home():
    return FileResponse("static/index.html")


def detect_encode_and_count(image_bytes: bytes) -> dict:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode camera frame.")

    # Keep processing resolution reasonable for real-time use.
    frame = cv2.resize(frame, (640, 480))

    result = model(
        frame,
        conf=0.35,
        imgsz=640,
        verbose=False
    )[0]

    annotated_frame = result.plot()
    counts = {}

    # result.boxes.cls contains the detected class IDs.
    if result.boxes is not None and len(result.boxes) > 0:
        class_ids = result.boxes.cls.int().cpu().tolist()

        for class_id in class_ids:
            class_name = result.names[class_id]
            counts[class_name] = counts.get(class_name, 0) + 1

    success, encoded_image = cv2.imencode(
        ".jpg",
        annotated_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )

    if not success:
        raise ValueError("Could not encode annotated frame.")

    return {
        "image": base64.b64encode(encoded_image.tobytes()).decode("ascii"),
        "counts": counts
    }


@app.websocket("/ws/detect")
async def detect_websocket(websocket: WebSocket):
    await websocket.accept()
    print("Browser connected.")

    try:
        while True:
            image_bytes = await websocket.receive_bytes()

            response = await asyncio.to_thread(
                detect_encode_and_count,
                image_bytes
            )

            # Sent as a normal JSON TEXT WebSocket message.
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("Browser disconnected.")

    except Exception as error:
        print(f"WebSocket error: {error}")

        try:
            await websocket.send_json({
                "error": str(error)
            })
        except Exception:
            pass