"""Trains the PhishGuard model from data/train.csv (calibrated on data/val.csv).

Usage:  python train_model.py
Writes: models/trained_model.joblib
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ml_detector import MODEL_SAVE_PATH, MLDetector

if __name__ == "__main__":
    t0 = time.time()
    if os.path.exists(MODEL_SAVE_PATH):
        os.remove(MODEL_SAVE_PATH)          # start clean so a stale artifact is never loaded first
    MLDetector().train_from_dataset()
    print(f"[+] Done in {time.time() - t0:.1f}s")
