import cv2
from detectors.base import analyze_video_with_detector, apply_nms

WIN_STRIDE = (8, 8)
PADDING = (4, 4)
SCALE = 1.05
HIT_THRESH = 0.0
NMS_THRESH = 0.65

_hog = None


def load_model():
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog


def detect_people(frame):
    hog = load_model()

    h, w = frame.shape[:2]
    target_width = 640
    scale_ratio = target_width / w
    resized = cv2.resize(frame, (target_width, int(h * scale_ratio)))

    rects, _ = hog.detectMultiScale(
        resized,
        winStride=WIN_STRIDE,
        padding=PADDING,
        scale=SCALE,
        hitThreshold=HIT_THRESH,
    )

    if len(rects) == 0:
        return 0, []

    boxes_xyxy = [(x, y, x + rw, y + rh) for (x, y, rw, rh) in rects]
    boxes_nms = apply_nms(boxes_xyxy, overlap_thresh=NMS_THRESH)

    inv_scale = 1.0 / scale_ratio
    final_boxes = [
        (
            int(x1 * inv_scale),
            int(y1 * inv_scale),
            int(x2 * inv_scale),
            int(y2 * inv_scale),
        )
        for (x1, y1, x2, y2) in boxes_nms
    ]

    return len(final_boxes), final_boxes


def analyze_video(video_path, processing_trace_path=None):
    return analyze_video_with_detector(
        video_path,
        detect_people,
        processing_trace_path=processing_trace_path,
    )
