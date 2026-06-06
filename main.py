"""
main.py
CLI entry point — runs the detector and shows the live window.
Press Q to quit.

Usage:
    python main.py
    python main.py --video traffic.mp4 --threshold 50
"""

import argparse
import cv2

from core.models   import load_models
from core.detector import Detector
import config


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Speed Detection")
    parser.add_argument("--video",     default=config.VIDEO_PATH if hasattr(config, "VIDEO_PATH") else "traffic.mp4")
    parser.add_argument("--threshold", type=int, default=config.SPEED_THRESHOLD)
    return parser.parse_args()


def main():
    args = parse_args()

    # Allow CLI override of threshold
    config.SPEED_THRESHOLD = args.threshold

    vehicle_model, plate_model, ocr_reader = load_models()

    det = Detector(vehicle_model, plate_model, ocr_reader, args.video)

    print(f"\nRunning on: {det.fps:.1f} FPS  |  {det.orig_w}x{det.orig_h}")
    print("Press Q in the video window to stop.\n")

    try:
        for annotated_frame, stats in det.run():
            cv2.imshow("Vehicle Speed Detection", annotated_frame)
            if cv2.waitKey(1) == ord("q"):
                break

            for r in stats["new_speeds"]:
                direction = "↓" if r["direction"] == "down" else "↑"
                flag = " [OVERSPEED]" if r["overspeed"] else ""
                print(
                    f"  Vehicle {r['track_id']} {direction} "
                    f"{r['speed_kmh']:.1f} km/h{flag}"
                )
    finally:
        det.release()
        cv2.destroyAllWindows()
        print(f"\nDone. CSV saved to: {config.CSV_PATH}")


if __name__ == "__main__":
    main()
