"""
Application Runner for PhishGuard AI - Explainable Phishing Risk Checker.
Starts the FastAPI web server on http://127.0.0.1:8000.
"""

import os
import sys

# Configure UTF-8 stdout to avoid cp1252 encoding errors on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import uvicorn
from evaluation.evaluator import load_evaluation_results

if __name__ == "__main__":
    print("=" * 70)
    print("   [+] PHISHGUARD AI - EXPLAINABLE PHISHING RISK CHECKER")
    print("   [+] Dual Detection Engine: Heuristic Rules + Random Forest ML")
    print("=" * 70)
    print("[*] Pre-checking model artifacts and test benchmarks...")
    load_evaluation_results()
    print("[+] Model and test evaluations ready.")
    print("[+] Launching server on http://127.0.0.1:8000")
    print("[+] Open http://127.0.0.1:8000 in your browser.")
    print("=" * 70)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
