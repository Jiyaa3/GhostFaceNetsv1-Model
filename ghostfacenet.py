"""
ghostfacenet.py - Pure PyTorch GhostFaceNetV1-1 Feature Extractor (Part-II).
Loads official pre-trained weights (4.01M params, MS1MV3 ArcFace) and extracts normalized 512D embeddings.
"""

import os
import math
import cv2
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "ghostfacenet_v1_1_model.pt")


class GhostModule(nn.Module):
    """Ghost module: Intrinsic feature maps + cheap linear depthwise transformations."""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.PReLU(init_channels) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.PReLU(new_channels) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """Ghost bottleneck with stride=1 or stride=2."""
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
            ) if stride == 2 else nn.Sequential(),
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostFaceNetExtractor:
    """100% Native PyTorch GhostFaceNetV1-1 Feature Extractor."""
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GhostFaceNet PyTorch model not found at {model_path}. "
                "Ensure ghostfacenet_v1_1_model.pt exists in models/."
            )

        # Optimize PyTorch CPU execution
        torch.set_num_threads(4)

        self.net = torch.load(model_path, map_location="cpu", weights_only=False)
        self.net.eval()

        self.name = "GhostFaceNetV1-1 (PyTorch)"
        self.embedding_dim = 512
        self.params = "4.09M"
        self.flops = "215.7M"
        self.training_dataset = "MS1MV3 (ArcFace Loss)"

    def _embed(self, bgr_112):
        """Single forward pass. Returns raw (un-normalized) 512D vector."""
        # ArcFace InsightFace normalization: (x - 127.5) / 128.0
        blob = cv2.cvtColor(bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (blob - 127.5) / 128.0
        # PyTorch tensor shape: (1, 112, 112, 3)
        tensor = torch.from_numpy(blob).unsqueeze(0)
        with torch.no_grad():
            return self.net(tensor)[0].numpy()

    def extract_feature(self, aligned_face, flip_tta=True):
        """
        aligned_face: (112, 112, 3) BGR uint8
        Returns: (512,) float32 normalized ArcFace embedding.
        With flip_tta=True the embeddings of the face and its mirror image are
        summed (standard ArcFace test-time augmentation) -> more stable matching.
        """
        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112))

        feat = self._embed(aligned_face)
        if flip_tta:
            feat = feat + self._embed(cv2.flip(aligned_face, 1))

        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm
        return feat.astype(np.float32)
