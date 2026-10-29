"""
app.py - Unified CLI Menu & Live Webcam Recognition for GhostFaceNet.

Features:
- GhostFaceNetV1-1 pure-CNN backbone (frozen 512D feature extractor).
- Trainable PyTorch linear classification head for face identification & unknown rejection.
- Live desktop webcam window with high-FPS YuNet detection, alignment, and HUD.
- Interactive CLI menu for enrollment, training, and configuration.
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import sys
import time
import argparse
import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass
import numpy as np
import torch

from face_detector import YuNetDetector
from ghostfacenet import GhostFaceNetExtractor
from classifier import (
    FaceClassifierHead,
    train_classifier,
    load_trained_classifier,
    MODELS_DIR,
    DEFAULT_WEIGHTS_PATH,
    DEFAULT_CLASSES_PATH,
    save_calibration,
    load_calibration
)
from enroll import (
    enroll_person_webcam,
    list_identities,
    delete_person,
    TRAIN_DIR
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset_samples(split_dir, detector):
    """
    Loads all images in split_dir/<class_name>/*.jpg, aligns them, and returns:
    samples: list of (aligned_crop_112, class_idx, class_name)
    class_names: list of class names
    """
    if not os.path.exists(split_dir):
        return [], []

    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    # "unknown" must NOT be a class: strangers are rejected by threshold + margin.
    skipped = [c for c in classes if c.strip().lower() in ("unknown", "unidentified")]
    if skipped:
        print(f"[i] Ignoring folder(s) {skipped}: unknown people are rejected by threshold, not learned as a class.")
    classes = [c for c in classes if c not in skipped]
    # Filter classes with at least 1 image
    valid_classes = []
    for c in classes:
        cdir = os.path.join(split_dir, c)
        imgs = [f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if imgs:
            valid_classes.append(c)

    samples = []
    for idx, cname in enumerate(valid_classes):
        cdir = os.path.join(split_dir, cname)
        imgs = sorted([f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for f in imgs:
            fpath = os.path.join(cdir, f)
            img = cv2.imread(fpath)
            if img is None:
                continue

            if img.shape[:2] == (112, 112):
                # Already aligned at enrollment time. Do NOT detect/align again,
                # otherwise training crops differ from live-inference crops.
                crop = img
            else:
                results = detector.detect_and_align(img)
                if results:
                    best = max(results, key=lambda r: r["box"][2] * r["box"][3])
                    crop = best["aligned_face"]
                else:
                    crop = cv2.resize(img, (112, 112))

            samples.append((crop, idx, cname))

    return samples, valid_classes


DEFAULT_THRESHOLD = 0.80   # threshold on the CALIBRATED match probability (0..1)
N_AUG = 5                  # augmented copies per training photo


def _split_indices(y, val_frac=0.2, min_per_class=5, seed=42):
    """Stratified train/validation split (classes with < min_per_class stay fully in train)."""
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for k in np.unique(y):
        idx = np.where(y == k)[0]
        rng.shuffle(idx)
        if len(idx) >= min_per_class:
            n_val = max(1, int(round(len(idx) * val_frac)))
            va += list(idx[:n_val])
            tr += list(idx[n_val:])
        else:
            tr += list(idx)
    return np.array(tr, dtype=int), np.array(va, dtype=int)


def augment_crop(crop, rng):
    """Random lighting / small pose / blur changes on an aligned 112x112 crop (training only)."""
    out = crop.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)
    out = np.clip(out, 0, 255).astype(np.uint8)
    M = cv2.getRotationMatrix2D((56, 56), rng.uniform(-8, 8), rng.uniform(0.95, 1.05))
    M[:, 2] += rng.uniform(-4, 4, 2)
    out = cv2.warpAffine(out, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)
    if rng.rand() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def _fit_calibration(head, X_val, y_val):
    """
    Fits  confidence = sigmoid(k * (cosine - c0))  from held-out genuine vs impostor similarities.
    c0 sits midway between the two groups; k makes a typical genuine match score ~0.95.
    """
    with torch.no_grad():
        sims = head(torch.tensor(X_val, dtype=torch.float32)).numpy()
    n, _ = sims.shape
    genuine = sims[np.arange(n), y_val]
    mask = np.ones_like(sims, dtype=bool)
    mask[np.arange(n), y_val] = False
    impostor = sims[mask]

    g_med, g_low = float(np.median(genuine)), float(np.percentile(genuine, 5))
    i_high = float(np.percentile(impostor, 95)) if impostor.size else 0.30
    i_mean = float(impostor.mean()) if impostor.size else 0.20
    print(f"  Genuine  cosine: median {g_med:.2f} | 5th pct {g_low:.2f}")
    print(f"  Impostor cosine: mean {i_mean:.2f} | 95th pct {i_high:.2f}")
    if g_low > i_high:
        c0 = (g_low + i_high) / 2.0
    else:
        print("  [!] Genuine and impostor ranges OVERLAP -> enroll more / better photos.")
        c0 = (g_med + i_mean) / 2.0
    c0 = float(min(0.75, max(0.20, c0)))
    k = 2.94 / max(0.05, g_med - c0)   # sigmoid(2.94) ~ 0.95 at the median genuine match
    return c0, float(k)


def _evaluate(head, X, y, threshold):
    """Returns (correct_accepted, wrong_person_accepted, rejected) fractions."""
    ok = wrong = rej = 0
    for x, t in zip(X, y):
        idx, _, rec = head.predict(x, threshold=threshold)
        if not rec:
            rej += 1
        elif idx == t:
            ok += 1
        else:
            wrong += 1
    n = max(1, len(y))
    return ok / n, wrong / n, rej / n


def _stranger_rejection(X_tr, y_tr, X_va, y_va, calib, threshold, num_classes):
    """
    Simulates strangers: for each person, build a recognizer WITHOUT them and check that
    their held-out photos are rejected. Needs >= 3 enrolled people.
    """
    if num_classes < 3:
        return None
    rates = []
    for k in range(num_classes):
        others = [j for j in range(num_classes) if j != k]
        h = FaceClassifierHead(512, len(others))
        for n, j in enumerate(others):
            m = X_tr[y_tr == j].mean(axis=0)
            h.fc.weight.data[n] = torch.nn.functional.normalize(torch.tensor(m, dtype=torch.float32), dim=0)
        h.calib = calib
        h.eval()
        probe = X_va[y_va == k]
        if len(probe) == 0:
            continue
        rates += [0.0 if h.predict(x, threshold=threshold)[2] else 1.0 for x in probe]
    return float(np.mean(rates)) if rates else None


def run_training_pipeline(epochs=100, lr=0.005):
    """Gathers data, augments, extracts GhostFaceNet embeddings, validates, calibrates and trains the head."""
    import tempfile
    print("\nPreparing dataset for training...")
    detector = YuNetDetector()
    extractor = GhostFaceNetExtractor()

    train_samples, class_names = load_dataset_samples(TRAIN_DIR, detector)
    if not class_names:
        print("\n[!] No training data found in data/train/.")
        print("    Please enroll identities first using option [2].\n")
        return None

    if len(class_names) < 2:
        print(f"\n[!] Notice: Only 1 class found ({class_names}).")
        print("    For multi-class classification, consider enrolling at least 2 people.")
        ans = input("    Proceed with training anyway? [y/N]: ").strip().lower()
        if ans != 'y':
            return None

    crops = [c for c, _, _ in train_samples]
    y = np.array([i for _, i, _ in train_samples], dtype=np.int64)
    counts = {c: int((y == i).sum()) for i, c in enumerate(class_names)}
    print(f"Photos per class: {counts}")
    low = [c for c, n in counts.items() if n < 15]
    if low:
        print(f"[!] Tip: {low} have < 15 photos. 20-30 varied photos per person is recommended.")

    print(f"Extracting embeddings for {len(crops)} photos (+{N_AUG} augmented copies each)...")
    X = np.array([extractor.extract_feature(c) for c in crops], dtype=np.float32)
    rng = np.random.RandomState(0)
    aug_X, aug_y, aug_src = [], [], []
    for i, c in enumerate(crops):
        for _ in range(N_AUG):
            aug_X.append(extractor.extract_feature(augment_crop(c, rng)))
            aug_y.append(y[i])
            aug_src.append(i)
    aug_X = np.array(aug_X, dtype=np.float32)
    aug_y = np.array(aug_y, dtype=np.int64)
    aug_src = np.array(aug_src, dtype=int)

    # ---- Validation run: held-out 20% of ORIGINAL photos (never augmented, never trained on) ----
    calib = None
    tr_idx, va_idx = _split_indices(y)
    if len(class_names) > 1 and len(va_idx) > 0:
        print("\n>>> VALIDATION RUN (80% train + augmentation / 20% held-out) <<<")
        keep = np.isin(aug_src, tr_idx)
        X_tr = np.concatenate([X[tr_idx], aug_X[keep]])
        y_tr = np.concatenate([y[tr_idx], aug_y[keep]])
        val_head, _ = train_classifier(X_tr, y_tr, class_names, X_val=X[va_idx], y_val=y[va_idx],
                                       epochs=epochs, lr=lr, save_dir=tempfile.mkdtemp())
        print("\nCalibration on held-out photos:")
        calib = _fit_calibration(val_head, X[va_idx], y[va_idx])
        val_head.calib = calib
        print(f"  Cosine boundary c0 = {calib[0]:.2f}, steepness k = {calib[1]:.1f}")

        print("\nResults on held-out photos (single images, before live frame-averaging):")
        print(f"  {'Threshold':<10} | {'Correct':>8} | {'Wrong person':>12} | {'Rejected':>9} | {'Strangers rejected':>18}")
        for th in (0.80, 0.90, 0.95, 0.99):
            ok, wrong, rej = _evaluate(val_head, X[va_idx], y[va_idx], th)
            st = _stranger_rejection(X_tr, y_tr, X[va_idx], y[va_idx], calib, th, len(class_names))
            st_s = f"{st*100:.1f}%" if st is not None else "n/a (need 3+ people)"
            print(f"  {th:<10.2f} | {ok*100:7.1f}% | {wrong*100:11.1f}% | {rej*100:8.1f}% | {st_s:>18}")
        print("  Higher threshold = fewer wrong IDs but more real people rejected. Pick the best balance.")
    elif len(class_names) > 1:
        print("\n[i] Not enough photos per class for a validation split (need >= 5). Skipping validation.")

    # ---- Final model on ALL photos + augmentation ----
    print("\n>>> FINAL TRAINING (all photos + augmentation) <<<")
    head, _ = train_classifier(
        X_train=np.concatenate([X, aug_X]),
        y_train=np.concatenate([y, aug_y]),
        class_names=class_names,
        epochs=epochs,
        lr=lr,
        save_dir=MODELS_DIR
    )
    if calib is None:
        # Single class / tiny data: assume a typical same-person vs stranger boundary
        with torch.no_grad():
            g = head(torch.tensor(X, dtype=torch.float32)).max(dim=-1).values.numpy()
        c0 = 0.35
        calib = (c0, 2.94 / max(0.05, float(np.median(g)) - c0))
    save_calibration(calib[0], calib[1], MODELS_DIR)
    print(f"Saved calibration -> models/calibration.json (c0={calib[0]:.2f}, k={calib[1]:.1f})")
    return head


def draw_hud(frame, fps, classes_count, threshold, toast=None):
    """Renders top and bottom HUD overlays."""
    h, w = frame.shape[:2]
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 42), (18, 18, 22), -1)
    title = f"GHOSTFACENET  |  FPS: {fps:.1f}  |  Enrolled: {classes_count} IDs  |  Threshold: {threshold:.2f}"
    cv2.putText(frame, title, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 35), (w, h), (18, 18, 22), -1)
    keys_help = "[d] Toggle Inspector  |  [+] / [-] Adjust Threshold  |  [q] / [ESC] Return to Menu"
    cv2.putText(frame, keys_help, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    if toast:
        (tw, th), _ = cv2.getTextSize(toast, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx, ty = (w - tw) // 2, 75
        cv2.rectangle(frame, (tx - 12, ty - th - 8), (tx + tw + 12, ty + 10), (0, 180, 240), -1)
        cv2.putText(frame, toast, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)


def draw_inspector(frame, raw_crop, aligned_crop, raw_pred=None, align_pred=None):
    """
    Renders picture-in-picture inspection panel in top-right corner.
    Shows the raw camera face crop vs 5-point affine aligned 112x112 ArcFace crop,
    with live prediction tags and alignment confidence boost comparison.
    """
    if raw_crop is None or aligned_crop is None:
        return

    h, w = frame.shape[:2]
    panel_w, panel_h = 240, 178
    if w < panel_w + 20 or h < panel_h + 20:
        return

    px = max(0, w - panel_w - 15)
    py = max(0, min(h - panel_h - 10, 52))

    # Draw semi-dark background card
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (20, 24, 30), -1)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (0, 200, 255), 1)

    # Header title
    cv2.putText(frame, "ALIGNMENT IMPACT [d]", (px + 30, py + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1)

    thumb_size = 88
    # 1. Raw Crop Thumbnail
    if raw_crop.size > 0:
        raw_thumb = cv2.resize(raw_crop, (thumb_size, thumb_size))
        frame[py + 36:py + 36 + thumb_size, px + 16:px + 16 + thumb_size] = raw_thumb
        cv2.rectangle(frame, (px + 16, py + 36), (px + 16 + thumb_size, py + 36 + thumb_size), (100, 100, 100), 1)
        cv2.putText(frame, "Raw Crop", (px + 30, py + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)

    # 2. Aligned 112x112 Thumbnail
    if aligned_crop.size > 0:
        align_thumb = cv2.resize(aligned_crop, (thumb_size, thumb_size))
        frame[py + 36:py + 36 + thumb_size, px + 136:px + 136 + thumb_size] = align_thumb
        cv2.rectangle(frame, (px + 136, py + 36), (px + 136 + thumb_size, py + 36 + thumb_size), (240, 140, 30), 2)
        cv2.putText(frame, "Aligned 112", (px + 144, py + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (250, 180, 70), 1)

    # Prediction Pills under thumbnails
    raw_conf = 0.0
    align_conf = 0.0

    if raw_pred:
        r_tag, r_conf, r_rec = raw_pred
        raw_conf = r_conf
        r_color = (0, 200, 50) if r_rec else (40, 40, 200)
        lbl = f"{r_tag[:6]} {int(r_conf*100)}%" if r_rec else f"unid {int(r_conf*100)}%"
        cv2.rectangle(frame, (px + 16, py + 128), (px + 16 + thumb_size, py + 146), r_color, -1)
        cv2.putText(frame, lbl, (px + 20, py + 141), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

    if align_pred:
        a_tag, a_conf, a_rec = align_pred
        align_conf = a_conf
        a_color = (0, 200, 50) if a_rec else (40, 40, 200)
        lbl = f"{a_tag[:6]} {int(a_conf*100)}%" if a_rec else f"unid {int(a_conf*100)}%"
        cv2.rectangle(frame, (px + 136, py + 128), (px + 136 + thumb_size, py + 146), a_color, -1)
        cv2.putText(frame, lbl, (px + 140, py + 141), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

    # Bottom Alignment Boost / Impact Indicator
    if align_pred and raw_pred:
        diff = int((align_conf - raw_conf) * 100)
        delta_str = f"Alignment Boost: +{diff}%" if diff >= 0 else f"Alignment Delta: {diff}%"
        color = (0, 255, 120) if diff > 0 else (180, 180, 180)
        cv2.putText(frame, delta_str, (px + 36, py + 166), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)


def draw_detection_box(frame, box, label, confidence, is_recognized):
    """Draws bounding box with stylized corners and classification tag."""
    x, y, w, h = box
    color = (0, 220, 50) if is_recognized else (40, 40, 240)  # Green if recognized, Red if unrecognised

    # Bounding rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Stylized corner accents
    corner_len = min(w, h) // 5
    t = 3
    # Top-left
    cv2.line(frame, (x, y), (x + corner_len, y), color, t)
    cv2.line(frame, (x, y), (x, y + corner_len), color, t)
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, t)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, t)
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, t)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, t)
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, t)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, t)

    # Badge text
    conf_pct = int(confidence * 100)
    if is_recognized:
        badge_text = f"{label} ({conf_pct}%)"
    else:
        badge_text = f"unidentified ({conf_pct}%)"

    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    badge_y1 = max(0, y - th - 12)
    badge_y2 = max(0, y)
    cv2.rectangle(frame, (x, badge_y1), (x + tw + 14, badge_y2), color, -1)
    cv2.putText(frame, badge_text, (x + 7, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


MIN_FACE_PX = 60        # faces narrower than this (pixels) are too small to identify reliably
MIN_DET_SCORE = 0.80    # YuNet detection confidence needed
MIN_SHARPNESS = 15.0    # Laplacian variance of the aligned crop (below = blurry)
TRACK_FRAMES = 6        # embeddings averaged per tracked face
MIN_TRACK_FRAMES = 3    # frames needed before a name is shown


class FaceTracker:
    """Minimal frame-to-frame face tracker (nearest box center) used to average embeddings over time."""
    def __init__(self, maxlen=TRACK_FRAMES, ttl=10):
        self.maxlen, self.ttl = maxlen, ttl
        self.tracks, self.next_id, self.frame, self.used = {}, 0, 0, set()

    def begin_frame(self):
        self.frame += 1
        self.used = set()
        self.tracks = {t: v for t, v in self.tracks.items() if self.frame - v["last"] <= self.ttl}

    def assign(self, box):
        bx, by, bw, bh = box
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        best, best_d = None, max(bw, bh) * 0.75
        for tid, t in self.tracks.items():
            if tid in self.used:
                continue
            d = ((t["c"][0] - cx) ** 2 + (t["c"][1] - cy) ** 2) ** 0.5
            if d < best_d:
                best, best_d = tid, d
        if best is None:
            best = self.next_id
            self.next_id += 1
            self.tracks[best] = {"embs": [], "c": (cx, cy), "last": self.frame}
        t = self.tracks[best]
        t["c"], t["last"] = (cx, cy), self.frame
        self.used.add(best)
        return best

    def add(self, tid, emb):
        e = self.tracks[tid]["embs"]
        e.append(emb)
        if len(e) > self.maxlen:
            e.pop(0)
        avg = np.mean(e, axis=0)
        n = np.linalg.norm(avg)
        return (avg / n if n > 1e-6 else avg).astype(np.float32), len(e)


def _face_quality(res):
    """Rejects faces that are too small, blurry, or turned away. Returns (ok, reason)."""
    bw, bh = res["box"][2], res["box"][3]
    if res["score"] < MIN_DET_SCORE:
        return False, "low detection score"
    if min(bw, bh) < MIN_FACE_PX:
        return False, "move closer"
    lm = res["landmarks"]
    eye_dist = float(lm[1][0] - lm[0][0])
    if eye_dist < 8:
        return False, "face the camera"
    yaw = float(lm[2][0] - lm[0][0]) / eye_dist
    if yaw < 0.25 or yaw > 0.75:
        return False, "face the camera"
    gray = cv2.cvtColor(res["aligned_face"], cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS:
        return False, "too blurry"
    return True, ""


def _draw_pending_box(frame, box, text):
    bx, by, bw, bh = box
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (150, 150, 150), 2)
    cv2.putText(frame, text, (bx, max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


def run_live_recognition(source=0, threshold=None):
    """
    Launches live webcam stream in an OpenCV desktop window with real-time recognition.
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    head, class_names = load_trained_classifier(MODELS_DIR)
    if head is None or not class_names:
        print("\n" + "!" * 65)
        print("  WARNING: No trained classification head found!")
        print("  The model must be trained first on enrolled faces.")
        print("!" * 65)
        ans = input("  Would you like to run training now? [Y/n]: ").strip().lower()
        if ans != 'n':
            head = run_training_pipeline()
            if head is None:
                return
            head, class_names = load_trained_classifier(MODELS_DIR)
        else:
            return

    print(f"\nInitializing YuNet detector and GhostFaceNet feature extractor...")
    detector = YuNetDetector()
    extractor = GhostFaceNetExtractor()

    # Open video capture
    is_image = False
    if isinstance(source, str) and not source.isdigit():
        if os.path.exists(source) and source.lower().endswith((".jpg", ".jpeg", ".png")):
            is_image = True
            cap = None
            static_frame = cv2.imread(source)
            if static_frame is None:
                print(f"Error: Could not read image at {source}")
                return
        else:
            cap = cv2.VideoCapture(source)
    else:
        cam_id = int(source) if str(source).isdigit() else 0
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            # Negotiate native 720p HD with MJPEG codec for razor-sharp feed
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not is_image and (cap is None or not cap.isOpened()):
        print(f"Error: Could not open camera/video source '{source}'.")
        return

    win_name = "GhostFaceNet Live Face Recognition"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Automatically set window dimensions matching the exact camera aspect ratio (prevents vertical stretching)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if not is_image else static_frame.shape[1]
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if not is_image else static_frame.shape[0]
    if cam_w > 0 and cam_h > 0:
        win_w = min(1280, cam_w)
        win_h = int(win_w * (cam_h / cam_w))
        cv2.resizeWindow(win_name, win_w, win_h)
    else:
        cv2.resizeWindow(win_name, 1280, 720)

    print("\n" + "=" * 65)
    print("  GHOSTFACENET LIVE RECOGNITION RUNNING")
    print("=" * 65)
    print(f"  Classes: {class_names}")
    print(f"  Initial Confidence Threshold: {threshold:.2f}")
    print("  Keyboard controls inside webcam window:")
    print("    [d] : Toggle Preprocessing Inspector (Raw vs Aligned)")
    print("    [+] : Increase confidence threshold (+0.05)")
    print("    [-] : Decrease confidence threshold (-0.05)")
    print("    [q] or [ESC] : Return to Main Menu")
    print("=" * 65 + "\n")

    fps = 30.0
    prev_time = time.time()
    toast_msg = None
    toast_expire = 0.0
    show_inspector = False  # Off by default per requirement; press [d] in window to toggle ON
    latest_raw_crop = None
    latest_aligned_crop = None
    latest_raw_pred = None
    latest_align_pred = None
    tracker = FaceTracker()

    while True:
        if is_image:
            frame = static_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Video stream ended.")
                break

        curr_time = time.time()
        fps = 0.85 * fps + 0.15 * (1.0 / max(1e-4, curr_time - prev_time))
        prev_time = curr_time

        active_toast = toast_msg if curr_time < toast_expire else None

        # Face detection & alignment via YuNet
        results = detector.detect_and_align(frame)

        max_face_area = 0  # Selects the primary (closest / largest) face for the inspector
        tracker.begin_frame()

        for res in results:
            box = res["box"]
            aligned_face = res["aligned_face"]
            bx, by, bw, bh = box
            face_area = bw * bh
            is_primary_face = (face_area > max_face_area)

            # Save crops for inspector from the primary face
            if is_primary_face:
                max_face_area = face_area
                rx1, ry1 = max(0, bx), max(0, by)
                rx2, ry2 = min(frame.shape[1], bx + bw), min(frame.shape[0], by + bh)
                if rx2 > rx1 and ry2 > ry1:
                    latest_raw_crop = frame[ry1:ry2, rx1:rx2].copy()
                latest_aligned_crop = aligned_face.copy()

            try:
                # Skip faces that are too small / blurry / turned away (they cause wrong IDs)
                q_ok, q_reason = _face_quality(res)
                if not q_ok:
                    _draw_pending_box(frame, box, q_reason)
                    continue

                # Extract 512D unit embedding with GhostFaceNet
                emb = extractor.extract_feature(aligned_face)

                # Average the embedding over the last few frames of this tracked face
                tid = tracker.assign(box)
                emb, n_frames = tracker.add(tid, emb)
                if n_frames < MIN_TRACK_FRAMES:
                    _draw_pending_box(frame, box, "verifying...")
                    continue

                # Predict with trained linear classification head
                pred_idx, conf, is_rec = head.predict(emb, threshold=threshold)
                tag = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
                if tag.strip().lower() in ("unknown", "unidentified"):
                    is_rec = False  # legacy "unknown" class from old models is never a match

                # Draw bounding box and tag on main screen
                draw_detection_box(frame, box, tag, conf, is_rec)

                # Update inspector predictions only for the primary face
                if is_primary_face:
                    latest_align_pred = (tag, conf, is_rec)
                    if show_inspector and latest_raw_crop is not None and latest_raw_crop.size > 0:
                        try:
                            raw_112 = cv2.resize(latest_raw_crop, (112, 112))
                            raw_emb = extractor.extract_feature(raw_112)
                            r_idx, r_conf, r_rec = head.predict(raw_emb, threshold=threshold)
                            r_tag = class_names[r_idx] if r_idx < len(class_names) else "Unknown"
                            latest_raw_pred = (r_tag, r_conf, r_rec)
                        except Exception:
                            pass
            except Exception as e:
                print(f"Inference error: {e}")

        # Draw Preprocessing Inspector if toggled ON
        if show_inspector:
            draw_inspector(frame, latest_raw_crop, latest_aligned_crop, latest_raw_pred, latest_align_pred)

        # Draw HUD overlays
        draw_hud(frame, fps, len(class_names), threshold, active_toast)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):  # 'q' or ESC
            break
        elif key in (ord('d'), ord('D')):
            show_inspector = not show_inspector
            toast_msg = f"Inspector: {'ON' if show_inspector else 'OFF'}"
            toast_expire = curr_time + 1.5
        elif key in (ord('+'), ord('=')):
            threshold = min(0.99, threshold + 0.05)
            toast_msg = f"Threshold: {threshold:.2f}"
            toast_expire = curr_time + 1.5
        elif key in (ord('-'), ord('_')):
            threshold = max(0.10, threshold - 0.05)
            toast_msg = f"Threshold: {threshold:.2f}"
            toast_expire = curr_time + 1.5

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("\nLive recognition closed. Returning to menu.\n")


def print_main_menu(threshold):
    identities = list_identities()
    head, classes = load_trained_classifier(MODELS_DIR)
    trained_status = f"TRAINED ({len(classes)} classes: {classes})" if head else "NOT TRAINED"

    print("\n" + "=" * 68)
    print("      GHOSTFACENET REAL-TIME FACIAL RECOGNITION")
    print("=" * 68)
    print("Architecture: Pure-CNN GhostFaceNetV1-1 (Frozen 512D Extractor)")
    print("Classifier  : PyTorch Cosine Classification Head (W · x)")
    print("-" * 68)
    print(f"Enrolled Identities ({len(identities)}):")
    if identities:
        for name, count in identities:
            print(f"   * {name:<16} : {count} photos")
    else:
        print("   (No identities enrolled yet)")
    print(f"Classification Head : {trained_status}")
    print(f"Confidence Threshold: {threshold:.2f}")
    print("-" * 68)
    print("Options:")
    print("  [1] Start Live Webcam Recognition (Desktop Window)")
    print("  [2] Enroll New Identity (Live Camera Capture with SPACEBAR)")
    print("  [3] Train / Retrain Classification Head")
    print("  [4] Manage Enrolled Identities (List / Delete)")
    print("  [5] Adjust Confidence Threshold")
    print("  [q] Quit")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="GhostFaceNet Real-Time Recognition")
    parser.add_argument("--source", default=0, help="Webcam index (e.g. 0) or video/image path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Match-probability threshold 0-1 (default 0.80)")
    parser.add_argument("--train-now", action="store_true", help="Directly train classification head and exit")
    parser.add_argument("--run-now", action="store_true", help="Directly start live recognition window and exit")
    args = parser.parse_args()

    threshold = args.threshold

    if args.train_now:
        run_training_pipeline()
        return

    if args.run_now:
        run_live_recognition(source=args.source, threshold=threshold)
        return

    # Interactive CLI Menu Loop
    while True:
        print_main_menu(threshold)
        choice = input("Enter selection [1-5 or q, default=1]: ").strip().lower()
        if not choice:
            choice = "1"

        if choice == "1":
            run_live_recognition(source=args.source, threshold=threshold)
        elif choice == "2":
            name = input("\nEnter name for new person (e.g. Gaurav): ").strip()
            if name:
                enroll_person_webcam(name, camera_id=args.source if str(args.source).isdigit() else 0)
                # Ask if user wants to retrain
                ans = input("Would you like to train/retrain the classification head now? [Y/n]: ").strip().lower()
                if ans != 'n':
                    run_training_pipeline()
        elif choice == "3":
            run_training_pipeline()
        elif choice == "4":
            identities = list_identities()
            if not identities:
                print("\nNo enrolled identities found.")
                continue
            print("\nEnrolled Identities:")
            for idx, (name, count) in enumerate(identities, 1):
                print(f"  [{idx}] {name} ({count} photos)")
            print("  [b] Back to Main Menu")
            del_choice = input("Enter identity number to delete (or 'b'): ").strip()
            if del_choice.isdigit():
                idx = int(del_choice) - 1
                if 0 <= idx < len(identities):
                    target_name = identities[idx][0]
                    confirm = input(f"Are you sure you want to delete '{target_name}'? [y/N]: ").strip().lower()
                    if confirm == 'y':
                        delete_person(target_name)
                        print(f"Deleted '{target_name}'. You should retrain the classifier head [Option 3].")
        elif choice == "5":
            try:
                new_th = float(input(f"\nEnter new confidence threshold [0.10 - 0.99] (current: {threshold:.2f}): "))
                if 0.10 <= new_th <= 0.99:
                    threshold = new_th
                    print(f"Threshold updated to {threshold:.2f}")
                else:
                    print("Value must be between 0.10 and 0.99.")
            except ValueError:
                print("Invalid number.")
        elif choice in ("q", "quit", "exit"):
            print("\nExiting. Goodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please select 1-5 or q.")


if __name__ == "__main__":
    main()
