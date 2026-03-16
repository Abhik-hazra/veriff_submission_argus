# Argus – Automated Recognition and Group Understanding System (Flask-based App for Veriff Take-Home test)

Argus is a small prototype application built as part of the Veriff Fraud Data Scientist take-home Assessment.

It demonstrates how a person-detection pipeline can be integrated into a simple operational tool to detect whether a verification video contains:

Single Person or Multiple People

The presence of multiple people in a verification video can be a potential fraud signal, and this tool shows how such detection can be automated and evaluated.

It is part of the integration solution and cost analysis of different Image Models. 

![Argus working](https://github.com/user-attachments/assets/bc52b4bc-c650-4d0b-ad40-ebcd03d0528c) 
<img src="https://github.com/user-attachments/assets/bc52b4bc-c650-4d0b-ad40-ebcd03d0528c" alt="Argus working" width="500">

# What Argus Can Do

Argus provides a simple interface for testing and evaluating different person-detection approaches.

## 1. Detect Single vs Multiple People in Videos

Upload one or more videos and Argus will:

Sample frames from the video

Detect people in each frame

Classify the video as:

Single Person
or
Multiple People

## 2. Bulk Video Analysis

Argus supports multiple video uploads and processes them sequentially.

For each video it reports:

Classification result

Maximum number of people detected

Frames containing multiple people

Frames processed

Video duration

The results can be downloaded as a CSV log file.

## 3. Frame-Level Processing Trace

For deeper inspection, Argus records frame-level processing traces including:

Frame index

Timestamp

Number of people detected

Bounding box coordinates

These traces can be downloaded as a ZIP file for debugging or auditing.

## 4. Model Comparison

Argus allows testing multiple detection approaches:

YOLOv10

YOLOv8

HOG + SVM

The architecture is modular, so additional detectors can easily be added.

## 5. Deployment Cost Calculator

Argus also includes a cost simulation tool that estimates the long-term cost of deploying different models.

The calculator considers:

Number of videos processed

Average video duration

Frame sampling rate

Fraud rate

Cost per fraud case

Model recall

Model throughput

Infrastructure cost

Licensing cost

Based on these inputs, the tool estimates:

Compute cost

Fraud loss cost

Total operational cost

and recommends the lowest total-cost model.

### Folder Structure

```
argus/
├── app.py
├── requirements.txt
├── detectors/
│   ├── base.py
│   ├── registry.py
│   ├── hog_svm_detector.py
│   ├── yolov8_detector.py
│   └── yolov10_detector.py
├── templates/
│   ├── index.html
│   └── calculator.html
├── static/
│   ├── style.css
│   ├── calculator.css
│   └── app.js
└── uploads/
    ├── videos/
    ├── processing/
    └── generated/
```
