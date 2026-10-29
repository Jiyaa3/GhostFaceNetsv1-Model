"""
face_align.py - 5-point similarity transformation for canonical 112x112 face alignment.
Standard InsightFace / Umeyama analytical alignment for ArcFace-based backbones.
"""

import cv2
import numpy as np

# Canonical 5 landmarks for 112x112 aligned face (InsightFace / ArcFace standard)
REFERENCE_5PTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041]   # right mouth corner
], dtype=np.float32)


def umeyama(src, dst, estimate_scale=True):
    """
    Estimate similarity transform between two sets of 2D points using the Umeyama algorithm.
    Closed-form analytical SVD solution - deterministic, zero random sampling, never returns None.

    Args:
        src: source points of shape (N, 2)
        dst: destination points of shape (N, 2)
        estimate_scale: boolean flag to estimate scaling factor

    Returns:
        M: (2, 3) affine transformation matrix
    """
    num = src.shape[0]
    dim = src.shape[1]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = np.dot(dst_demean.T, src_demean) / num

    U, S, Vt = np.linalg.svd(A)
    V = Vt.T

    d = np.ones(dim, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        d[-1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    R = np.dot(U, np.dot(np.diag(d), V.T))
    T[:dim, :dim] = R

    if estimate_scale:
        src_var = (src_demean ** 2).sum() / num
        c = 1.0 if src_var == 0 else np.dot(S, d) / src_var
        T[:dim, :dim] *= c

    T[:dim, dim] = dst_mean - np.dot(T[:dim, :dim], src_mean)
    return T[:2, :]


def align_face_112(image, landmarks_5pts, output_size=(112, 112)):
    """
    Align face to canonical 112x112 using 5 landmarks via Umeyama analytical transform.
    
    Args:
        image: BGR image (H, W, 3)
        landmarks_5pts: array-like of shape (5, 2)
        output_size: tuple (width, height), default (112, 112)
        
    Returns:
        aligned_face: warped image of shape (112, 112, 3)
    """
    pts = np.array(landmarks_5pts, dtype=np.float32).reshape(5, 2)
    
    try:
        M = umeyama(pts, REFERENCE_5PTS, estimate_scale=True)
        aligned = cv2.warpAffine(image, M, output_size, flags=cv2.INTER_LINEAR, borderValue=0.0)
        return aligned
    except Exception:
        # Fallback to cropping around the landmarks bounding box if singular
        min_x = max(0, int(np.min(pts[:, 0])))
        max_x = min(image.shape[1], int(np.max(pts[:, 0])))
        min_y = max(0, int(np.min(pts[:, 1])))
        max_y = min(image.shape[0], int(np.max(pts[:, 1])))
        if max_x > min_x and max_y > min_y:
            crop = image[min_y:max_y, min_x:max_x]
            return cv2.resize(crop, output_size)
        return cv2.resize(image, output_size)

