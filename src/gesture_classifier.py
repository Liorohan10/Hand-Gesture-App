from typing import List, Tuple
import numpy as np

# MediaPipe hand landmark indices reference:
# 0: wrist
# Thumb: 1 (CMC), 2 (MCP), 3 (IP), 4 (TIP)
# Index: 5 (MCP), 6 (PIP), 7 (DIP), 8 (TIP)
# Middle: 9 (MCP), 10 (PIP), 11 (DIP), 12 (TIP)
# Ring: 13 (MCP), 14 (PIP), 15 (DIP), 16 (TIP)
# Pinky: 17 (MCP), 18 (PIP), 19 (DIP), 20 (TIP)

FINGER_TIPS = [8, 12, 16, 20]  # index, middle, ring, pinky tips
FINGER_PIPS = [6, 10, 14, 18]  # corresponding PIP joints
FINGER_MCPS = [5, 9, 13, 17]   # corresponding MCP joints
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
WRIST = 0


def _vector(a: Tuple[float, float], b: Tuple[float, float]) -> np.ndarray:
    return np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float32)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 180.0
    cosang = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _angle_at(center: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    v1 = np.array([p1[0] - center[0], p1[1] - center[1]], dtype=np.float32)
    v2 = np.array([p2[0] - center[0], p2[1] - center[1]], dtype=np.float32)
    return _angle_deg(v1, v2)


def _is_finger_extended(landmarks: List[Tuple[float, float]], tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    # Combine multiple cues for robustness across rotations:
    # 1) Vertical cue: tip above PIP (y smaller) by margin.
    tip_y = landmarks[tip_idx][1]
    pip_y = landmarks[pip_idx][1]
    vertical_extended = tip_y < pip_y - 0.015

    # 2) Radial cue: tip farther from wrist than PIP by margin.
    wrist = landmarks[WRIST]
    radial_extended = _dist(wrist, landmarks[tip_idx]) > _dist(wrist, landmarks[pip_idx]) + 0.02

    # 3) Joint angle cue: finger straighter at PIP when extended (angle ~180 deg)
    angle_pip = _angle_at(landmarks[pip_idx], landmarks[tip_idx], landmarks[mcp_idx])
    angle_straight = angle_pip > 150.0

    # Majority vote among cues
    cues_true = sum([vertical_extended, radial_extended, angle_straight])
    return cues_true >= 2


def _thumb_state(landmarks: List[Tuple[float, float]]) -> str:
    # Determine thumb orientation: up, down, left, right, or folded
    wrist = landmarks[WRIST]
    tip = landmarks[THUMB_TIP]
    mcp = landmarks[THUMB_MCP]
    ip = landmarks[THUMB_IP]

    # Folded if tip is close to palm center or aligned towards wrist
    palm_center = np.mean(np.array([landmarks[i] for i in [0, 5, 9, 13, 17]]), axis=0)
    dist_tip_palm = np.linalg.norm(np.array(tip) - palm_center)
    dist_mcp_palm = np.linalg.norm(np.array(mcp) - palm_center)
    if dist_tip_palm < dist_mcp_palm * 0.8:
        return "folded"

    # Use vector from MCP->TIP and global axes to decide orientation
    v = _vector(mcp, tip)
    if np.linalg.norm(v) < 1e-3:
        return "folded"

    # Compare angle with upward, downward, left, right unit vectors
    up = np.array([0, -1], dtype=np.float32)
    down = np.array([0, 1], dtype=np.float32)
    left = np.array([-1, 0], dtype=np.float32)
    right = np.array([1, 0], dtype=np.float32)

    ang_up = _angle_deg(v, up)
    ang_down = _angle_deg(v, down)
    ang_left = _angle_deg(v, left)
    ang_right = _angle_deg(v, right)

    best = min([(ang_up, "up"), (ang_down, "down"), (ang_left, "left"), (ang_right, "right")], key=lambda x: x[0])
    if best[0] <= 45:
        return best[1]
    return "folded"


def classify_gesture(landmarks_norm: List[Tuple[float, float]]) -> str:
    """
    landmarks_norm: list of (x, y) in image normalized coords [0,1]
    Returns one of: "Open Palm", "Fist", "Peace", "Thumbs Up", or "Unknown"
    """
    fingers = [
        _is_finger_extended(landmarks_norm, FINGER_TIPS[i], FINGER_PIPS[i], FINGER_MCPS[i])
        for i in range(4)
    ]
    index_ext, middle_ext, ring_ext, pinky_ext = fingers
    thumb_dir = _thumb_state(landmarks_norm)

    # Open Palm: all four fingers extended; thumb not strictly required to be extended
    if all(fingers):
        return "Open Palm"

    # Fist: all fingers folded and thumb not up (allow folded/left/right/down)
    if not any(fingers) and thumb_dir != "up":
        return "Fist"

    # Peace: index + middle extended; ring + pinky folded; thumb any
    if index_ext and middle_ext and (not ring_ext) and (not pinky_ext):
        return "Peace"

    # Thumbs Up: thumb up; others folded
    if thumb_dir == "up" and (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext):
        return "Thumbs Up"

    return "Unknown"


def landmarks_to_xy_norm(hand_landmarks, image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
    # MediaPipe provides normalized coords already; we simply extract (x, y)
    result = []
    for lm in hand_landmarks.landmark:
        result.append((float(lm.x), float(lm.y)))
    return result
