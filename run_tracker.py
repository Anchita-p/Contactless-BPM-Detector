"""Quick visual smoke test for the ROI tracker.

Opens the default webcam, draws green dots on forehead ROI points and blue
dots on cheek ROI points in real-time. Press 'q' to quit.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tracker import FaceROITracker

if __name__ == "__main__":
    tracker = FaceROITracker(camera_index=0)
    print("Starting webcam feed. Press 'q' in the video window to quit.")
    tracker.run()
    signal = tracker.get_signal()
    print(f"Captured {signal.shape[0] if signal.size else 0} frames of RGB signal.")
