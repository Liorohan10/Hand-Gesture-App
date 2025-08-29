# Real-Time Static Hand Gesture Recognition

Author: Your Name Here

## Overview
This Python app recognizes four static hand gestures from a webcam in real time:
- Open Palm
- Fist
- Peace Sign (V-sign)
- Thumbs Up

It uses MediaPipe Hands for robust, real-time hand landmark detection, and OpenCV for video capture and visualization. A lightweight, rules-based classifier interprets landmark geometry to label gestures.

## Why MediaPipe + OpenCV?
- MediaPipe Hands provides high-quality, fast hand landmark detection (21 keypoints) that works well on CPUs, enabling real-time inference without training a custom model. It’s battle-tested for real-time HCI tasks and includes tracking for temporal stability.
- OpenCV offers reliable webcam access, frame processing, and drawing utilities for high-FPS visualization on Windows.

## Gesture Logic
The classifier uses normalized landmark positions to determine finger states (extended vs folded) and thumb orientation.

Finger extension is decided via a robust majority vote of three cues:
- Vertical cue: fingertip y is above PIP y (with a small margin).
- Radial cue: fingertip is farther from the wrist than the PIP (normalized distance).
- Joint-angle cue: the PIP joint angle is near-straight (>150°) when extended.

Rules:
- Open Palm: index/middle/ring/pinky extended (thumb may be relaxed).
- Fist: all four fingers folded and thumb not pointing up (folded/sideways accepted).
- Peace (V): index and middle extended; ring and pinky folded; thumb neutral.
- Thumbs Up: thumb extended upward; other fingers folded.

## Install
Create a virtual environment (recommended) and install dependencies.

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bat
python src\app.py
```

Press `q` to quit the window.

Advanced options:

```bat
python src\app.py --device 0 --width 960 --height 540 --record assets\demo.mp4
```

- `--device`: webcam index if you have multiple cameras.
- `--width`, `--height`: capture resolution for performance tuning.
- `--record`: optional path to save annotated video.
- `--headless`: run without opening a window (useful on systems missing OpenCV GUI support; combine with `--record`).

## Demo
Add a short screen recording showing all four gestures working in real time to `assets/demo.mp4` or `assets/demo.gif`. If you prefer to host it elsewhere, paste the link here.

## Notes
- Tested on Windows with a standard integrated webcam. If you have multiple cameras, you can change the device index in `app.py`.
- If the frame rate is low, reduce resolution or set `max_num_hands=1`.

### Troubleshooting
- Error: `cv2.imshow ... The function is not implemented`.
  - You're likely on a headless OpenCV build. Options:
	 1) Reinstall non-headless OpenCV (already pinned in `requirements.txt`):
		 - Reactivate venv and reinstall requirements.
		 - Or `pip uninstall opencv-python-headless` then `pip install opencv-python==4.9.0.80`.
	 2) Run in headless mode and record to a file:
		 ```bat
		 python src\app.py --headless --record assets\demo.mp4
		 ```
  - If using WSL, run from Windows Python or enable X server.

## Project Structure

- `src/app.py`: Main entry; webcam loop, MediaPipe inference, drawing, overlay, CLI.
- `src/gesture_classifier.py`: Rules-based classifier using 2D landmarks.
- `tests/test_classifier.py`: Minimal sanity tests for rules.
- `requirements.txt`: Dependencies.
- `assets/`: Put your demo recording here.

## Submission Checklist
- [ ] Repo is public and contains source code and `requirements.txt`.
- [ ] `README.md` filled with your name, tech justification, gesture logic, and run steps.
- [ ] `assets/demo.mp4` or a link to a recording showing all four gestures.
- [ ] Email the repo link to: support@bhatiyaniai.com (Attention: Adhip Sarda) by 02/09/2025, 5pm IST.
