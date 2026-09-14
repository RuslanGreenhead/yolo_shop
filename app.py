import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import MODEL_HF_REPO, MODEL_HF_REVISION, MODEL_PATH
from vision import FrameProcessor

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


def load_model():
    from ultralytics import YOLO

    # Reuse project-local weights offline; download the pinned HF file on first run.
    model_path = Path(__file__).resolve().parent / MODEL_PATH
    if not model_path.is_file():
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=MODEL_HF_REPO, filename=model_path.name,
            revision=MODEL_HF_REVISION, local_dir=model_path.parent, token=False,
        )
    return YOLO(str(model_path))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading at startup keeps imports and API tests independent of model files.
    app.state.processor = await asyncio.to_thread(lambda: FrameProcessor(load_model()))
    app.state.active_camera = None
    yield


app = FastAPI(title="LightStore", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/session")
def get_session():
    return app.state.processor.snapshot()


@app.post("/api/reset")
def reset_session():
    return app.state.processor.reset()


@app.websocket("/ws/detect")
async def detect_websocket(websocket: WebSocket):
    await websocket.accept()
    # A persistent YOLO tracker cannot mix frames from different cameras.
    if app.state.active_camera is not None:
        await websocket.send_json({"error": "Another camera is already connected. Stop it first."})
        await websocket.close(code=1008)
        return
    app.state.active_camera = websocket
    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            try:
                response = await asyncio.to_thread(app.state.processor.process, image_bytes)
            except ValueError as error:
                await websocket.send_json({"error": str(error)})
                continue
            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Could not process camera stream")
        try:
            await websocket.send_json({"error": "Camera processing failed. See the server log."})
            await websocket.close(code=1011)
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        if app.state.active_camera is websocket:
            app.state.active_camera = None
