"""
CLI Evaluation Tool for PhishGuard AI - Explainable Phishing Risk Checker.
Evaluates both the Rule-Based Baseline and Random Forest ML detector on held-out test data
and outputs formatted metrics, confusion matrices, and error diagnoses to the terminal.
"""

import os
import sys

# Configure UTF-8 stdout to avoid cp1252 encoding errors on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluator import run_evaluation

def print_separator(char="=", length=75):
    print(char * length)

def main():
    print_separator("=")
    print("   [+] PHISHGUARD AI - MODEL BENCHMARK & EVALUATION CLI")
    print("   [+] Dataset: Held-Out Test Set (data/test.csv) | Seed = 42")
    print("   [+] Comparing: Transparent Rule Baseline vs Random Forest ML")
    print_separator("=")

    report = run_evaluation()
    ds = report["evaluation_dataset"]
    detectors = report["detectors"]

    print(f"\n[+] Test Set Size: {ds['total_test_urls']} URLs ({ds['phishing_count']} Phishing, {ds['legitimate_count']} Legitimate)")
    print_separator("-")

    # Metrics Summary
    print("\n[+] COMPARATIVE PERFORMANCE METRICS ON HELD-OUT TEST DATA:")
    headers = ["Model", "F1 Score", "Precision", "Recall", "Accuracy", "FPR", "FNR"]
    print(f"{headers[0]:<30} {headers[1]:<10} {headers[2]:<11} {headers[3]:<10} {headers[4]:<10} {headers[5]:<8} {headers[6]:<8}")
    print("-" * 88)

    for key in ["rule_based_baseline", "machine_learning", "hybrid_ensemble"]:
        d = detectors[key]
        m = d["metrics"]
        print(f"{d['name']:<30} {m['f1_score']:<10.4f} {m['precision']:<11.4f} {m['recall']:<10.4f} {m['accuracy']:<10.4f} {m['false_positive_rate']:<8.4f} {m['false_negative_rate']:<8.4f}")

    print_separator("-")

    # Confusion Matrices
    print("\n[+] CONFUSION MATRICES:")
    for key in ["rule_based_baseline", "machine_learning"]:
        d = detectors[key]
        cm = d["confusion_matrix"]
        print(f"\n--- {d['name']} ---")
        print(f"  True Positives (TP)  [Correctly Caught Phishing] : {cm['true_positives']}")
        print(f"  True Negatives (TN)  [Correctly Cleared Benign]  : {cm['true_negatives']}")
        print(f"  False Positives (FP) [Benign Misclassified]       : {cm['false_positives']}")
        print(f"  False Negatives (FN) [Phishing Missed]           : {cm['false_negatives']}")

    print_separator("-")

    # Error cases
    print("\n[+] ERROR CASE SAMPLES (FALSE POSITIVES & FALSE NEGATIVES):")
    ml_errors = detectors["machine_learning"]["error_cases"]

    print("\n[*] Sample False Positive (Legitimate URL flagged as risk):")
    if ml_errors["false_positives"]:
        fp = ml_errors["false_positives"][0]
        print(f"    URL:     {fp['url']}")
        print(f"    Verdict: {fp['predicted_verdict']}")
        print(f"    Cause:   {fp['root_cause_analysis']}")
    else:
        print("    None observed in sample window.")

    print("\n[*] Sample False Negative (Phishing URL that evaded detection):")
    if ml_errors["false_negatives"]:
        fn = ml_errors["false_negatives"][0]
        print(f"    URL:     {fn['url']}")
        print(f"    Verdict: {fn['predicted_verdict']}")
        print(f"    Cause:   {fn['root_cause_analysis']}")
    else:
        print("    None observed in sample window.")

    print_separator("=")
    print("\n[+] Full evaluation JSON saved at: evaluation/evaluation_results.json\n")

if __name__ == "__main__":
    main()
