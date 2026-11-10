import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ml_detector import MLDetector

if __name__ == "__main__":
    print("[*] Starting model training script...", flush=True)
    t0 = time.time()
    detector = MLDetector()
    detector.train_from_dataset()
    print(f"[+] Done in {time.time() - t0:.2f} seconds!", flush=True)
