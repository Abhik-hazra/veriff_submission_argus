import csv
import os
import uuid
import zipfile
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from detectors.registry import DETECTORS, DISPLAY_NAMES

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
VIDEOS_DIR = os.path.join(UPLOAD_FOLDER, "videos")
PROCESSING_DIR = os.path.join(UPLOAD_FOLDER, "processing")
GENERATED_DIR = os.path.join(UPLOAD_FOLDER, "generated")

for folder in (UPLOAD_FOLDER, VIDEOS_DIR, PROCESSING_DIR, GENERATED_DIR):
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "mpeg", "mpg", "webm"}

CALCULATOR_DEFAULTS = {
    "item_type": "video",
    "num_items": 10000,
    "duration_sec": 15,
    "sample_fps": 1,
    "fraud_rate_pct": 5,
    "fraud_cost_usd": 500,
    "selected_models": ["yolov10", "yolov8", "hog_svm"],
}

CALCULATOR_MODEL_DEFAULTS = {
    "yolov10": {
        "recall_multi": 0.95,
        "throughput_fps": 35.0,
        "hourly_cost_usd": 0.45,
        "annual_license_usd": 10000.0,
    },
    "yolov8": {
        "recall_multi": 0.92,
        "throughput_fps": 42.0,
        "hourly_cost_usd": 0.40,
        "annual_license_usd": 5000.0,
    },
    "hog_svm": {
        "recall_multi": 0.78,
        "throughput_fps": 3.0,
        "hourly_cost_usd": 0.10,
        "annual_license_usd": 0.0,
    },
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_unique_filename(filename):
    safe_name = secure_filename(filename)
    stem, ext = os.path.splitext(safe_name)
    return f"{stem}_{uuid.uuid4().hex[:8]}{ext.lower()}"


def resolve_methods(method_key):
    if method_key == "all":
        return list(DETECTORS.keys())
    if method_key in DETECTORS:
        return [method_key]
    raise ValueError("Invalid detection method selected.")


def create_log_file(results):
    logfile_name = f"analysis_log_{uuid.uuid4().hex}.csv"
    logfile_path = os.path.join(GENERATED_DIR, logfile_name)

    fieldnames = [
        "filename",
        "method",
        "classification",
        "max_person_count",
        "frames_with_multiple_people",
        "frames_processed",
        "duration_sec",
        "status",
        "error",
    ]

    with open(logfile_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(
                {
                    "filename": row.get("filename", ""),
                    "method": row.get("method", ""),
                    "classification": row.get("classification", ""),
                    "max_person_count": row.get("max_person_count", ""),
                    "frames_with_multiple_people": row.get("frames_with_multiple_people", ""),
                    "frames_processed": row.get("frames_processed", ""),
                    "duration_sec": row.get("duration_sec", ""),
                    "status": "Error" if row.get("error") else "Done",
                    "error": row.get("error", ""),
                }
            )

    return f"generated/{logfile_name}"


def create_processing_zip(trace_paths):
    valid_paths = [
        path for path in trace_paths
        if os.path.exists(path) and os.path.getsize(path) > 0
    ]
    if not valid_paths:
        return None

    zip_name = f"processing_steps_{uuid.uuid4().hex}.zip"
    zip_path = os.path.join(GENERATED_DIR, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for trace_path in valid_paths:
            zipf.write(trace_path, arcname=os.path.basename(trace_path))

    return f"generated/{zip_name}"


def safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template(
            "index.html",
            results=None,
            error=None,
            logfile=None,
            processingfile=None,
            selected_method="yolov10",
        )

    selected_method = request.form.get("method", "yolov10")
    uploaded_files = request.files.getlist("videos")
    non_empty_files = [f for f in uploaded_files if f and f.filename and f.filename.strip()]

    if not non_empty_files:
        return render_template(
            "index.html",
            results=None,
            error="Please upload at least one video file.",
            logfile=None,
            processingfile=None,
            selected_method=selected_method,
        )

    try:
        methods_to_run = resolve_methods(selected_method)
    except ValueError as exc:
        return render_template(
            "index.html",
            results=None,
            error=str(exc),
            logfile=None,
            processingfile=None,
            selected_method="yolov10",
        )

    results = []
    trace_paths = []

    for file_storage in non_empty_files:
        original_filename = secure_filename(file_storage.filename) or "uploaded_video"

        if not allowed_file(original_filename):
            method_labels = (
                [DISPLAY_NAMES.get(key, key) for key in DETECTORS.keys()]
                if selected_method == "all"
                else [DISPLAY_NAMES.get(selected_method, selected_method)]
            )

            for method_label in method_labels:
                results.append(
                    {
                        "filename": original_filename,
                        "method": method_label,
                        "classification": "-",
                        "max_person_count": "-",
                        "frames_with_multiple_people": "-",
                        "frames_processed": "-",
                        "duration_sec": "-",
                        "error": "Unsupported video format.",
                    }
                )
            continue

        saved_filename = make_unique_filename(original_filename)
        saved_video_path = os.path.join(VIDEOS_DIR, saved_filename)
        file_storage.save(saved_video_path)

        try:
            for method_key in methods_to_run:
                detector_fn = DETECTORS[method_key]
                method_label = DISPLAY_NAMES.get(method_key, method_key)

                trace_filename = (
                    f"{Path(original_filename).stem}__{method_key}__{uuid.uuid4().hex[:8]}.csv"
                )
                trace_path = os.path.join(PROCESSING_DIR, trace_filename)

                try:
                    result = detector_fn(
                        saved_video_path,
                        processing_trace_path=trace_path,
                    )

                    if os.path.exists(trace_path) and os.path.getsize(trace_path) > 0:
                        trace_paths.append(trace_path)

                    results.append(
                        {
                            "filename": original_filename,
                            "method": method_label,
                            "classification": result.get("classification", "-"),
                            "max_person_count": result.get("max_person_count", "-"),
                            "frames_with_multiple_people": result.get(
                                "frames_with_multiple_people", "-"
                            ),
                            "frames_processed": result.get("frames_processed", "-"),
                            "duration_sec": result.get("duration_sec", "-"),
                            "error": None,
                        }
                    )
                except Exception as exc:
                    if os.path.exists(trace_path) and os.path.getsize(trace_path) > 0:
                        trace_paths.append(trace_path)

                    results.append(
                        {
                            "filename": original_filename,
                            "method": method_label,
                            "classification": "-",
                            "max_person_count": "-",
                            "frames_with_multiple_people": "-",
                            "frames_processed": "-",
                            "duration_sec": "-",
                            "error": str(exc),
                        }
                    )
        finally:
            if os.path.exists(saved_video_path):
                os.remove(saved_video_path)

    logfile = create_log_file(results) if results else None
    processingfile = create_processing_zip(trace_paths)

    return render_template(
        "index.html",
        results=results,
        error=None,
        logfile=logfile,
        processingfile=processingfile,
        selected_method=selected_method,
    )


@app.route("/calculator", methods=["GET", "POST"])
def calculator():
    models = []
    for key in DETECTORS.keys():
        defaults = CALCULATOR_MODEL_DEFAULTS.get(key, {})
        models.append(
            {
                "key": key,
                "label": DISPLAY_NAMES.get(key, key),
                "recall_multi": defaults.get("recall_multi", 0.90),
                "throughput_fps": defaults.get("throughput_fps", 10.0),
                "hourly_cost_usd": defaults.get("hourly_cost_usd", 0.20),
                "annual_license_usd": defaults.get("annual_license_usd", 0.0),
            }
        )

    form = dict(CALCULATOR_DEFAULTS)
    results = None
    best_model_key = None

    if request.method == "POST":
        selected_models = [m for m in request.form.getlist("models") if m in DETECTORS]
        if not selected_models:
            selected_models = ["yolov10"]

        form = {
            "item_type": request.form.get("item_type", "video"),
            "num_items": safe_int(
                request.form.get("num_items"),
                CALCULATOR_DEFAULTS["num_items"],
            ),
            "duration_sec": safe_float(
                request.form.get("duration_sec"),
                CALCULATOR_DEFAULTS["duration_sec"],
            ),
            "sample_fps": safe_float(
                request.form.get("sample_fps"),
                CALCULATOR_DEFAULTS["sample_fps"],
            ),
            "fraud_rate_pct": safe_float(
                request.form.get("fraud_rate_pct"),
                CALCULATOR_DEFAULTS["fraud_rate_pct"],
            ),
            "fraud_cost_usd": safe_float(
                request.form.get("fraud_cost_usd"),
                CALCULATOR_DEFAULTS["fraud_cost_usd"],
            ),
            "selected_models": selected_models,
        }

        frames_per_item = 1 if form["item_type"] == "image" else form["duration_sec"] * form["sample_fps"]
        total_sampled_frames = form["num_items"] * frames_per_item
        fraud_rate = form["fraud_rate_pct"] / 100.0

        results = []
        for model in models:
            if model["key"] not in selected_models:
                continue

            recall_multi = min(
                max(
                    safe_float(
                        request.form.get(f"recall_{model['key']}"),
                        model["recall_multi"],
                    ),
                    0.0,
                ),
                1.0,
            )

            throughput_fps = max(
                safe_float(
                    request.form.get(f"throughput_{model['key']}"),
                    model["throughput_fps"],
                ),
                0.0001,
            )

            hourly_cost_usd = max(
                safe_float(
                    request.form.get(f"hourly_cost_{model['key']}"),
                    model["hourly_cost_usd"],
                ),
                0.0,
            )

            annual_license_usd = max(
                safe_float(
                    request.form.get(f"license_{model['key']}"),
                    model["annual_license_usd"],
                ),
                0.0,
            )

            compute_hours = total_sampled_frames / throughput_fps / 3600.0
            compute_cost_usd = compute_hours * hourly_cost_usd

            expected_fraud_cases = form["num_items"] * fraud_rate
            missed_fraud_cases = expected_fraud_cases * (1.0 - recall_multi)
            fraud_loss_usd = missed_fraud_cases * form["fraud_cost_usd"]

            total_cost_usd = compute_cost_usd + fraud_loss_usd + annual_license_usd

            results.append(
                {
                    "model_key": model["key"],
                    "model_label": model["label"],
                    "recall_multi": recall_multi,
                    "throughput_fps": throughput_fps,
                    "hourly_cost_usd": hourly_cost_usd,
                    "annual_license_usd": annual_license_usd,
                    "frames_per_item": frames_per_item,
                    "total_sampled_frames": total_sampled_frames,
                    "compute_hours": compute_hours,
                    "compute_cost_usd": compute_cost_usd,
                    "expected_fraud_cases": expected_fraud_cases,
                    "missed_fraud_cases": missed_fraud_cases,
                    "fraud_loss_usd": fraud_loss_usd,
                    "total_cost_usd": total_cost_usd,
                }
            )

        results.sort(key=lambda row: row["total_cost_usd"])
        if results:
            best_model_key = results[0]["model_key"]

        for model in models:
            model["recall_multi"] = safe_float(
                request.form.get(f"recall_{model['key']}"),
                model["recall_multi"],
            )
            model["throughput_fps"] = safe_float(
                request.form.get(f"throughput_{model['key']}"),
                model["throughput_fps"],
            )
            model["hourly_cost_usd"] = safe_float(
                request.form.get(f"hourly_cost_{model['key']}"),
                model["hourly_cost_usd"],
            )
            model["annual_license_usd"] = safe_float(
                request.form.get(f"license_{model['key']}"),
                model["annual_license_usd"],
            )

    return render_template(
        "calculator.html",
        models=models,
        form=form,
        results=results,
        best_model_key=best_model_key,
    )


@app.route("/download-log/<path:filename>")
def download_log(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)


@app.route("/download-processing/<path:filename>")
def download_processing(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
