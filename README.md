# GhostFaceNet Real-Time Facial Recognition System

A streamlined, single-backbone facial recognition system powered by **GhostFaceNetV1-1** with **Transfer Learning**.

---

## 1. System Architecture

```
                  [ Live Camera Frame ]
                            │
                            ▼
                OpenCV YuNet Face Detector
                (High-speed >60 FPS detection)
                            │
                            ▼
               Analytical Umeyama SVD Alignment
                    Canonical 112×112 Crop
                            │
                            ▼
          Frozen GhostFaceNet Feature Extractor
          (4.09M params, 215M FLOPs, 512D unit vector)
                            │
                            ▼
          PyTorch Cosine Classification Head
             (Cosine Similarity on Hypersphere)
                            │
           ┌────────────────┴────────────────┐
           ▼                                 ▼
   Similarity >= Threshold           Similarity < Threshold
  [ Enrolled Student Tag ]            [ unidentified ]
     (Green Box + Conf%)             (Red Box + Conf%)
```

---

## 2. Quick Start

### Launch Official Submission Entry Point:
```bash
py Group_x_1.py
```
*(Or `python Group_x_1.py` or `py app.py`)*

This launches the unified command-center CLI menu:
```text
====================================================================
      GHOSTFACENET REAL-TIME FACIAL RECOGNITION
====================================================================
Architecture: Pure-CNN GhostFaceNetV1-1 (Frozen 512D Extractor)
Classifier  : PyTorch Cosine Classification Head (W · x)
--------------------------------------------------------------------
Enrolled Identities:
   * khoosh             : 7 photos
Classification Head : TRAINED (1 class: ['khoosh'])
Confidence Threshold: 0.80
--------------------------------------------------------------------
Options:
  [1] Start Live Webcam Recognition (Desktop Window)
  [2] Enroll New Identity (Live Camera Capture with SPACEBAR)
  [3] Train / Retrain Classification Head
  [4] Manage Enrolled Identities (List / Delete)
  [5] Adjust Confidence Threshold
  [q] Quit
====================================================================
```

---

## 3. Workflow Guide

### Step 1: Enroll Identities (`[2]`)
1. Enter person's name (e.g., `khoosh` or `Gaurav`).
2. An interactive camera window opens at native aspect ratio.
3. Position your face in front of the camera and press **`[SPACEBAR]`** to capture 6–10 photos (tilt slightly, vary expressions).
4. Press **`[q]`** or **`[ESC]`** when finished. Photos are saved into `data/train/<name>/`.
5. Repeat for all group members.

### Step 2: Train the Classification Head (`[3]`)
1. Select option **`[3]`** (or confirm when prompted after enrollment).
2. The script extracts 512D embeddings using the frozen **GhostFaceNet** backbone.
3. Computes normalized class centroids on the ArcFace hypersphere (and optimizes inter-class separation if $K \ge 2$).
4. Completes in **~1 second on CPU**!
5. Model weights are saved to `models/classifier_head.pth` and `models/classes.json`.

### Step 3: Start Live Webcam Recognition (`[1]`)
1. Select option **`[1]`**.
2. Opens high-FPS webcam window at native 16:9 HD:
   - Green bounding box + Person's Name + Confidence % for recognized members.
   - Red bounding box + `"unidentified"` for strangers or unrecognized faces.
   - Multiple faces in the same room are detected and classified simultaneously.
   - Live HUD showing FPS, active classes, and threshold.
3. **Keyboard Hotkeys inside the Webcam Window:**
   - **`[d]`**: Toggle the **Neural Net Input Inspector** (shows Raw Crop vs. Aligned 112×112 side-by-side with live confidence boost delta, locked to primary face).
   - **`[+]` / `[-]`**: Increase or decrease confidence rejection threshold (default `0.80`).
   - **`[q]` / `[ESC]`**: Exit webcam window and return cleanly to the main menu.

---

## 4. Command-Line Arguments
```bash
# Run directly without menu
py Group_x_1.py --run-now

# Train directly from CLI
py Group_x_1.py --train-now

# Test on a video file or test image
py Group_x_1.py --source /path/to/test.mp4
py Group_x_1.py --source /path/to/test.jpg
```
