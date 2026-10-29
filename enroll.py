"""
enroll.py - Interactive CLI & Webcam Face Enrollment System for GhostFaceNet.
Captures live photos via webcam, aligns them using YuNet, and organizes into data/train and data/test splits.
"""

import os
import shutil
import cv2
import numpy as np
from face_detector import YuNetDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")

os.makedirs(TRAIN_DIR, exist_ok=True)


def list_identities():
    """Returns list of tuples: [(name, photo_count), ...]"""
    if not os.path.exists(TRAIN_DIR):
        return []

    names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    result = []
    for name in names:
        tr_dir = os.path.join(TRAIN_DIR, name)
        tr_cnt = len([f for f in os.listdir(tr_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.exists(tr_dir) else 0
        result.append((name, tr_cnt))
    return result


def enroll_person_webcam(name, camera_id=0):
    """
    Opens live webcam stream.
    User presses SPACEBAR to capture aligned face crops.
    Press [q] or [ESC] to complete enrollment.
    """
    clean_name = name.strip().replace(" ", "_")
    if not clean_name:
        print("Error: Name cannot be empty.")
        return 0

    person_train_dir = os.path.join(TRAIN_DIR, clean_name)
    os.makedirs(person_train_dir, exist_ok=True)

    # Initialize YuNet detector
    detector = YuNetDetector()

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        # Clean up empty folder
        if os.path.exists(person_train_dir) and len(os.listdir(person_train_dir)) == 0:
            os.rmdir(person_train_dir)
        return 0

    # Negotiate native 720p HD with MJPEG codec
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win_name = f"Enrollment Capture - {clean_name}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Automatically set window dimensions matching the exact camera aspect ratio
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if cam_w > 0 and cam_h > 0:
        win_w = min(1280, cam_w)
        win_h = int(win_w * (cam_h / cam_w))
        cv2.resizeWindow(win_name, win_w, win_h)
    else:
        cv2.resizeWindow(win_name, 1280, 720)

    print("\n" + "=" * 65)
    print(f"  LIVE ENROLLMENT CAPTURE: '{clean_name}'")
    print("=" * 65)
    print("  * Position your face in front of the camera.")
    print("  * Press [SPACEBAR] to capture a face photo.")
    print("    (Tilt slightly, change expressions, glasses on/off).")
    print("  * Aim for 6 to 10 captures.")
    print("  * Press [q] or [ESC] when finished.")
    print("=" * 65 + "\n")

    captured_count = len([f for f in os.listdir(person_train_dir) if f.lower().endswith(".jpg")])
    flash_frames = 0
    last_saved_crop = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam frame.")
            break

        # Detect face with YuNet
        results = detector.detect_and_align(frame)
        h, w = frame.shape[:2]

        display = frame.copy()

        # Visual flash feedback on capture
        if flash_frames > 0:
            white_overlay = np.full_like(display, 255)
            display = cv2.addWeighted(display, 0.4, white_overlay, 0.6, 0)
            flash_frames -= 1

        active_crop = None
        if results:
            # Pick highest confidence face
            best_face = max(results, key=lambda x: x["score"])
            bx, by, bw, bh = best_face["box"]
            active_crop = best_face["aligned_face"]

            # Draw green bounding box & corners
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 255, 128), 2)
            corner_len = min(bw, bh) // 5
            for cx, cy, dx, dy in [(bx, by, 1, 1), (bx + bw, by, -1, 1),
                                   (bx, by + bh, 1, -1), (bx + bw, by + bh, -1, -1)]:
                cv2.line(display, (cx, cy), (cx + dx * corner_len, cy), (0, 255, 128), 3)
                cv2.line(display, (cx, cy), (cx, cy + dy * corner_len), (0, 255, 128), 3)

            # Draw 5 landmark points
            for pt in best_face["landmarks"]:
                cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (0, 220, 255), -1)

            cv2.putText(display, f"Face Detected: {best_face['score']:.2f}",
                        (bx, max(25, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)

        # Draw HUD bar
        cv2.rectangle(display, (0, 0), (w, 50), (20, 20, 25), -1)
        cv2.putText(display, f"Enrolling: {clean_name} | Captured: {captured_count} photos",
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(display, (0, h - 45), (w, h), (20, 20, 25), -1)
        cv2.putText(display, "[SPACE] Capture photo   |   [q] / [ESC] Finish & Save",
                    (15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        # Thumbnail preview of last saved crop in top-right corner
        if last_saved_crop is not None:
            thumb = cv2.resize(last_saved_crop, (80, 80))
            display[10:90, w - 90:w - 10] = thumb
            cv2.rectangle(display, (w - 90, 10), (w - 10, 90), (0, 255, 128), 2)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):  # 'q' or ESC
            break
        elif key == 32:  # SPACEBAR
            if active_crop is not None:
                captured_count += 1
                flash_frames = 3
                last_saved_crop = active_crop.copy()

                out_path = os.path.join(person_train_dir, f"sample_{captured_count}.jpg")
                cv2.imwrite(out_path, active_crop)
                print(f"Captured #{captured_count} -> train/{clean_name}/sample_{captured_count}.jpg")
            else:
                print("No face detected! Look directly at the camera.")

    cap.release()
    cv2.destroyAllWindows()
    if captured_count == 0:
        if os.path.exists(person_train_dir) and len(os.listdir(person_train_dir)) == 0:
            os.rmdir(person_train_dir)
        print(f"\nEnrollment cancelled for '{clean_name}'. No photos captured.\n")
    else:
        print(f"\nEnrollment finished for '{clean_name}'. Total captured: {captured_count} photos.\n")
    return captured_count


def delete_person(name):
    """Deletes person's train folder."""
    clean_name = name.strip()
    tr = os.path.join(TRAIN_DIR, clean_name)
    deleted = False
    if os.path.exists(tr):
        shutil.rmtree(tr)
        deleted = True
    return deleted

