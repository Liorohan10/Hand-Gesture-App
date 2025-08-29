"""
Application entry point.

Captures webcam frames, uses MediaPipe Hands to detect hand landmarks, classifies
the gesture with a lightweight rules-based classifier, and visualizes the result.
Supports GUI display or headless logging, and optional recording to a video file.
"""
import os
import cv2
import numpy as np
import argparse
from typing import Tuple

# Reduce TensorFlow/TF-Lite logs before importing mediapipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import mediapipe as mp

from gesture_classifier import classify_gesture, landmarks_to_xy_norm

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def parse_args():
    """Parse CLI arguments for device, resolution, recording, and headless mode."""
    parser = argparse.ArgumentParser(description="Real-Time Hand Gesture Recognition")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--width", type=int, default=960, help="Capture width")
    parser.add_argument("--height", type=int, default=540, help="Capture height")
    parser.add_argument("--record", type=str, default="", help="Optional output video path (e.g., output.mp4)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window (use with --record)")
    return parser.parse_args()


def main():
    """Run the real-time loop: read frames, detect hands, classify, draw, and output."""
    args = parse_args()

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    writer = None
    if args.record:
        # Initialize an MP4 writer to save annotated frames to disk
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 20.0, (args.width, args.height))

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from webcam.")
                break

            # Mirror the preview for natural user experience (like a mirror)
            frame = cv2.flip(frame, 1)
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            gesture = "No Hand"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract normalized (x, y) and classify current gesture
                    landmarks_norm = landmarks_to_xy_norm(hand_landmarks, frame.shape[:2])
                    gesture = classify_gesture(landmarks_norm)

                    # Draw hand landmarks and connections for visual feedback
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            # Overlay a label with the current gesture on the frame
            cv2.rectangle(frame, (10, 10), (330, 70), (0, 0, 0), thickness=-1)
            cv2.putText(frame, f"Gesture: {gesture}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            if writer is not None:
                writer.write(frame)
            
            if args.headless:
                # In headless mode, print periodically to avoid flooding the console
                frame_count += 1
                if frame_count % 15 == 0:
                    print(f"Gesture: {gesture}")
            else:
                try:
                    cv2.imshow("Hand Gesture Recognition", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except cv2.error as e:
                    print("OpenCV GUI not available. Rerun with --headless (and optional --record).\n",
                          "Or reinstall opencv-python (non-headless). Error:", e)
                    break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
