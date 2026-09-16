"""Fetch pinned official PaddleDetection source, Objects365 weights and labels."""

from pathlib import Path
import subprocess

from ppyoloe_assets import (CHECKPOINT_URL, CHECKPOINT_SHA256, LABELS_URL,
                           LABELS_SHA256, PADDLEDETECTION_COMMIT)
from yoloe_model import download_verified


def main():
    root = Path(__file__).resolve().parent
    source = root / ".cache/PaddleDetection"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                        "https://github.com/PaddlePaddle/PaddleDetection.git", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", PADDLEDETECTION_COMMIT], check=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != PADDLEDETECTION_COMMIT:
        raise ValueError(f"Existing PaddleDetection checkout must be at {PADDLEDETECTION_COMMIT}; it was not changed.")
    assets = root / "weights/ppyoloe-objects365"
    download_verified(CHECKPOINT_URL, assets / "ppyoloe_crn_s_obj365_pretrained.pdparams", CHECKPOINT_SHA256)
    download_verified(LABELS_URL, assets / "objects365_labels.txt", LABELS_SHA256)
    print("PP-YOLOE+ Small source, weights and labels verified. No ONNX export is needed.")


if __name__ == "__main__":
    main()
