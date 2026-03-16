import csv
import json
import os

import cv2
import numpy as np


def apply_nms(boxes, overlap_thresh=0.65):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(y2)

    keep = []
    while order.size > 0:
        i = order[-1]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[order[:-1]]

        order = order[np.where(overlap <= overlap_thresh)[0]]

    return boxes[keep].astype(int).tolist()


def analyze_video_with_detector(
    video_path,
    detector_fn,
    sample_fps=1,
    min_frames_for_multi=2,
    processing_trace_path=None,
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(total_frames / fps, 2) if fps and fps > 0 else 0
    frame_interval = max(1, int(fps / sample_fps)) if fps and fps > 0 else 1

    current_frame_idx = 0
    frames_processed = 0
    max_person_count = 0
    frames_with_multiple_people = 0

    trace_file = None
    trace_writer = None

    if processing_trace_path:
        trace_dir = os.path.dirname(processing_trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)

        trace_file = open(processing_trace_path, "w", newline="", encoding="utf-8")
        trace_writer = csv.DictWriter(
            trace_file,
            fieldnames=[
                "frame_index",
                "timestamp_sec",
                "person_count",
                "multiple_people",
                "boxes",
            ],
        )
        trace_writer.writeheader()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx % frame_interval == 0:
                person_count, boxes = detector_fn(frame)

                frames_processed += 1
                max_person_count = max(max_person_count, person_count)

                multiple_people = person_count >= 2
                if multiple_people:
                    frames_with_multiple_people += 1

                if trace_writer:
                    timestamp_sec = round(current_frame_idx / fps, 3) if fps and fps > 0 else 0
                    trace_writer.writerow(
                        {
                            "frame_index": current_frame_idx,
                            "timestamp_sec": timestamp_sec,
                            "person_count": person_count,
                            "multiple_people": multiple_people,
                            "boxes": json.dumps(boxes),
                        }
                    )

            current_frame_idx += 1
    finally:
        cap.release()
        if trace_file:
            trace_file.close()

    classification = (
        "Multiple People"
        if frames_with_multiple_people >= min_frames_for_multi
        else "Single Person"
    )

    return {
        "classification": classification,
        "duration_sec": duration_sec,
        "frames_processed": frames_processed,
        "max_person_count": max_person_count,
        "frames_with_multiple_people": frames_with_multiple_people,
        "processing_trace_path": processing_trace_path,
    }
