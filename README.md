# Real-Time Static Hand Gesture Recognition

Author: Rohan Ganguly

Email ID: rohan.ganguly2806@gmail.com

## Overview
This Python app recognizes four static hand gestures from a webcam in real time:
- Open Palm
- Fist
- Peace Sign (V-sign)
- Thumbs Up

It uses MediaPipe Hands for robust, real-time hand landmark detection, and OpenCV for video capture and visualization. A lightweight, rules-based classifier interprets landmark geometry to label gestures.

## Technology & Models Used
- MediaPipe Hands (Model + Tracking):
	- Pipeline: a lightweight palm detector followed by a hand landmark regressor that outputs 21 3D landmarks and handedness. It uses TensorFlow Lite with XNNPACK delegate for efficient CPU inference.
	- Why it fits: robust to lighting/background, works out-of-the-box without custom data collection or training, and maintains temporal consistency via tracking. This is ideal for a time-bound assessment and for static gestures.
	- Alternatives considered:
		- Classical CV (skin color thresholding + contours): fragile under varying lighting and skin tones; less robust in cluttered backgrounds.
		- Custom CNN classifier on raw images: requires a sizeable labeled dataset and training time; brittle to domain shifts; overkill for 4 static poses when landmarks suffice.
		- YOLO/pose models: can detect hands but lack fine-grained finger joint detail needed for rule-based static pose classification.
	- Trade-offs: MediaPipe abstracts model complexity and gives stable keypoints; the small downside is a dependency on TF Lite runtime and occasional model downloads on first run.
- OpenCV: mature, cross-platform capture and visualization with simple APIs and good performance.
- NumPy: vector math for angles, distances, and geometric rules.

## Gesture Logic
We classify gestures from MediaPipe’s 21 normalized hand landmarks using geometric rules:

- Landmarks used: tips (8,12,16,20), PIPs (6,10,14,18), MCPs (5,9,13,17), thumb (MCP=2, IP=3, tip=4), wrist=0.
- Finger state (extended vs folded): majority vote of three cues for robustness:
	- Vertical: tip y < PIP y (with a small margin).
	- Radial: dist(wrist, tip) > dist(wrist, PIP) + margin.
	- Joint angle: angle at PIP (MCP–PIP–TIP) > 150°.
- Thumb orientation: compare vector MCP→TIP to up/down/left/right; if none within 45°, or tip near the palm, treat as folded.
- Rules:
	- Open Palm: index/middle/ring/pinky extended (thumb may be relaxed).
	- Fist: all four folded and thumb not up (folded/sideways accepted).
	- Peace (V): index + middle extended; ring + pinky folded.
	- Thumbs Up: thumb up; others folded.
- Edge cases: when landmarks are missing or heavily occluded, return "Unknown"; cues mitigate rotation; thresholds are conservative to reduce false positives.
- Possible enhancements: temporal smoothing, handedness-aware thumb logic, adaptive thresholds by hand scale.

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
![Image](https://github.com/user-attachments/assets/7f3b7dba-fbe5-4e8d-9ec3-b3bc94c8323b)

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

## Performance Considerations
- Defaults target real-time CPU performance at 960x540 with `max_num_hands=1`.
- For higher FPS:
	- Lower resolution (e.g., 640x360) via `--width`/`--height`.
	- Keep `model_complexity=0` (already set) for faster inference.
	- Avoid unnecessary copies; we only convert BGR→RGB once per frame.

## Architecture at a Glance
1) Capture frame (OpenCV) and mirror for natural UX.
2) Run MediaPipe Hands to get 21 keypoints and connections.
3) Convert landmarks to normalized (x, y) pairs.
4) Compute finger states and thumb orientation via geometric rules.
5) Map to gesture label; draw landmarks and overlay the label.
6) Display (or headless print) and optionally record annotated frames.
