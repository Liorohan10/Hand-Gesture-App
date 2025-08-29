"""
Minimal sanity tests for the rules-based gesture classifier.

These tests use synthetic landmark coordinates to approximate finger states and
validate that the rules map to the expected labels for the four target gestures.
"""
from src.gesture_classifier import classify_gesture

# Synthetic landmarks: very rough, normalized coords where smaller y = higher on screen
def make_landmarks(tip_states, thumb_dir="folded"):
    # 21 points, fill with neutral center (0.5, 0.5)
    lm = [(0.5, 0.5) for _ in range(21)]
    # Tips, PIPs, MCPs indices
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    mcps = [5, 9, 13, 17]
    # Set PIPs slightly lower (bigger y) than base to emulate finger curvature
    for p in pips:
        lm[p] = (0.5, 0.55)
    # Extended tips above PIP, folded below
    for i, tip_idx in enumerate(tips):
        lm[tip_idx] = (0.5, 0.45 if tip_states[i] else 0.65)
    # MCPs nearer wrist
    for i, mcp_idx in enumerate(mcps):
        lm[mcp_idx] = (0.5, 0.6)

    # Thumb orientation via MCP(2) and TIP(4)
    if thumb_dir == "up":
        lm[2] = (0.4, 0.55); lm[4] = (0.4, 0.35)
    elif thumb_dir == "folded":
        lm[2] = (0.45, 0.55); lm[4] = (0.48, 0.53)

    return lm


def test_rules():
    """Smoke-test the core gestures using synthetic landmarks."""
    # Open palm: all extended
    lm = make_landmarks([True, True, True, True], thumb_dir="up")
    assert classify_gesture(lm) == "Open Palm"

    # Fist: all folded + thumb folded
    lm = make_landmarks([False, False, False, False], thumb_dir="folded")
    assert classify_gesture(lm) == "Fist"

    # Peace: index/middle extended
    lm = make_landmarks([True, True, False, False], thumb_dir="folded")
    assert classify_gesture(lm) == "Peace"

    # Thumbs Up: thumb up, others folded
    lm = make_landmarks([False, False, False, False], thumb_dir="up")
    assert classify_gesture(lm) == "Thumbs Up"

if __name__ == "__main__":
    test_rules()
    print("Classifier tests passed.")
