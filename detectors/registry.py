from detectors.yolov8_detector import analyze_video as analyze_yolov8
from detectors.yolov10_detector import analyze_video as analyze_yolov10
from detectors.hog_svm_detector import analyze_video as analyze_hog_svm

DETECTORS = {
    "yolov8": analyze_yolov8,
    "yolov10": analyze_yolov10,
    "hog_svm": analyze_hog_svm,
}

DISPLAY_NAMES = {
    "yolov8": "YOLOv8",
    "yolov10": "YOLOv10",
    "hog_svm": "HOG + SVM",
}
