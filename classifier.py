"""
classifier.py - Trainable PyTorch Classification Head and Transfer Learning Pipeline.
Adds a trainable linear layer on top of the frozen GhostFaceNet 512D feature extractor.
"""

import os
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DEFAULT_WEIGHTS_PATH = os.path.join(MODELS_DIR, "classifier_head.pth")
DEFAULT_CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
CALIB_PATH = os.path.join(MODELS_DIR, "calibration.json")

# Minimum gap between best and 2nd-best class similarity. Faces that look equally
# like two enrolled people are rejected as "unidentified" instead of guessed.
MIN_MARGIN = 0.05
COSINE_SCALE = 32.0


def save_calibration(c0, k, save_dir=MODELS_DIR):
    """
    Saves the mapping raw cosine similarity -> match probability:
        confidence = sigmoid(k * (cosine - c0))
    c0 = decision boundary between 'same person' and 'different person' (confidence 0.5)
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"c0": float(c0), "k": float(k)}, f)


def load_calibration(save_dir=MODELS_DIR):
    p = os.path.join(save_dir, "calibration.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return float(d["c0"]), float(d["k"])
    except Exception:
        return None


class FaceClassifierHead(nn.Module):
    """
    Linear classification layer mapping 512D GhostFaceNet embedding to K classes.
    Computes cosine similarity between unit-normalized face embeddings and class weight vectors.
    """
    def __init__(self, in_features=512, num_classes=1):
        super(FaceClassifierHead, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes, bias=False)
        self.calib = None  # (c0, k) set after training; maps cosine -> match probability

    def forward(self, x):
        # Cosine similarity: normalized weights dot normalized inputs
        w_norm = nn.functional.normalize(self.fc.weight, p=2, dim=1)
        x_norm = nn.functional.normalize(x, p=2, dim=-1)
        return nn.functional.linear(x_norm, w_norm)

    def predict(self, x, threshold=0.60, margin=MIN_MARGIN):
        """
        x: Tensor of shape (1, 512) or (512,) or numpy array
        Recognized only if best similarity >= threshold AND (for 2+ classes) the
        best class beats the runner-up by at least `margin`.
        Returns:
            predicted_idx (int)
            confidence (float 0.0 - 1.0)
            is_recognized (bool)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            sims = self.forward(x)[0]  # shape (num_classes,)
            k = min(2, sims.shape[0])
            top = torch.topk(sims, k=k)
            conf = float(top.values[0].item())
            pred_idx = int(top.indices[0].item())
            gap = conf - float(top.values[1].item()) if k > 1 else 1.0

        # Convert raw cosine similarity into a calibrated match probability (0..1)
        calib = getattr(self, "calib", None)
        if calib is not None:
            c0, k = calib
            conf = 1.0 / (1.0 + math.exp(-k * (conf - c0)))

        conf_clamped = max(0.0, min(1.0, conf))
        is_recognized = (conf_clamped >= threshold) and (gap >= margin)
        return pred_idx, conf_clamped, is_recognized


def train_classifier(X_train, y_train, class_names, X_val=None, y_val=None,
                     epochs=100, lr=0.005, save_dir=MODELS_DIR):
    """
    Trains the linear classification head on GhostFaceNet embeddings.
    Computes class centroids on the hypersphere and optimizes inter-class separation.
    """
    num_classes = len(class_names)
    if num_classes < 1:
        raise ValueError("Cannot train with 0 classes.")

    print("\n" + "=" * 65)
    print("  TRAINING CLASSIFICATION HEAD ON GHOSTFACENET EMBEDDINGS")
    print("=" * 65)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Training Samples: {len(X_train)}")
    if X_val is not None and len(X_val) > 0:
        print(f"Validation Samples: {len(X_val)}")
    print("-" * 65)

    head = FaceClassifierHead(in_features=512, num_classes=num_classes)

    # Convert to PyTorch tensors
    if not isinstance(X_train, torch.Tensor):
        X_t = torch.tensor(X_train, dtype=torch.float32)
    else:
        X_t = X_train.float()

    if not isinstance(y_train, torch.Tensor):
        y_t = torch.tensor(y_train, dtype=torch.long)
    else:
        y_t = y_train.long()

    # Initialize weights directly with normalized class centroids
    for k in range(num_classes):
        mask = (y_t == k)
        if mask.sum() > 0:
            centroid = X_t[mask].mean(dim=0)
            centroid = nn.functional.normalize(centroid, p=2, dim=0)
            head.fc.weight.data[k] = centroid

    start_time = time.time()

    if num_classes > 1:
        # Multi-class fine-tuning with Cosine Softmax
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)

        print(f"{'Epoch':<10} | {'Loss':<12} | {'Train Acc':<12} | {'Val Acc'}")
        print("-" * 65)

        scale = COSINE_SCALE  # Cosine scaling factor for softmax temperature
        for ep in range(1, epochs + 1):
            head.train()
            optimizer.zero_grad()
            cos_sim = head(X_t)
            logits = cos_sim * scale
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

            # Renormalize weights after each gradient step
            with torch.no_grad():
                head.fc.weight.data = nn.functional.normalize(head.fc.weight.data, p=2, dim=1)

            # Metrics
            head.eval()
            with torch.no_grad():
                preds = head(X_t).argmax(dim=-1)
                train_acc = (preds == y_t).float().mean().item() * 100.0

                val_str = "--"
                if X_val is not None and len(X_val) > 0:
                    X_v = torch.tensor(X_val, dtype=torch.float32) if not isinstance(X_val, torch.Tensor) else X_val.float()
                    y_v = torch.tensor(y_val, dtype=torch.long) if not isinstance(y_val, torch.Tensor) else y_val.long()
                    v_cos = head(X_v)
                    v_preds = v_cos.argmax(dim=-1)
                    val_acc = (v_preds == y_v).float().mean().item() * 100.0
                    val_str = f"{val_acc:.1f}%"

            if ep == 1 or ep % 5 == 0 or ep == epochs:
                print(f"Epoch {ep:02d}/{epochs:02d} | {loss.item():<12.4f} | {train_acc:5.1f}%       | {val_str}")
    else:
        # Single class: Centroid represents the optimal reference template
        head.eval()
        with torch.no_grad():
            sims = head(X_t).squeeze(-1)
            mean_sim = sims.mean().item()
            min_sim = sims.min().item()
            max_sim = sims.max().item()
        print(f"Single Class Mode: Hypersphere Centroid computed for '{class_names[0]}'.")
        print(f"Intra-class Similarity - Mean: {mean_sim*100:.1f}%, Min: {min_sim*100:.1f}%, Max: {max_sim*100:.1f}%")

    elapsed = time.time() - start_time
    print("-" * 65)
    print(f"Training completed successfully in {elapsed:.2f}s!")

    # Save checkpoint & class metadata
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "classifier_head.pth")
    classes_path = os.path.join(save_dir, "classes.json")

    torch.save({
        "state_dict": head.state_dict(),
        "in_features": 512,
        "num_classes": num_classes,
        "classes": class_names,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }, weights_path)

    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved weights: {weights_path}")
    print(f"Saved classes: {classes_path}")
    return head, class_names


def load_trained_classifier(save_dir=MODELS_DIR):
    """
    Loads saved FaceClassifierHead and classes list.
    Returns: (head_model, classes_list) or (None, []) if not trained.
    """
    weights_path = os.path.join(save_dir, "classifier_head.pth")
    classes_path = os.path.join(save_dir, "classes.json")

    if not os.path.exists(weights_path) or not os.path.exists(classes_path):
        return None, []

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    checkpoint = torch.load(weights_path, map_location="cpu")
    num_classes = len(classes)
    head = FaceClassifierHead(in_features=512, num_classes=num_classes)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    head.calib = load_calibration(save_dir)
    return head, classes
