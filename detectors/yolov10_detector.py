import torch
from ultralytics import YOLO
from detectors.base import analyze_video_with_detector

PERSON_CLASS_ID = 0
CONFIDENCE = 0.6
MODEL_WEIGHTS = "yolov10n.pt"

_model = None


def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_WEIGHTS)
        _model.to(get_device())
    return _model


def detect_people(frame):
    model = load_model()
    results = model.predict(
        source=frame,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE,
        verbose=False,
    )

    boxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == PERSON_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))

    return len(boxes), boxes


def analyze_video(video_path, processing_trace_path=None):
    return analyze_video_with_detector(
        video_path,
        detect_people,
        processing_trace_path=processing_trace_path,
    )
