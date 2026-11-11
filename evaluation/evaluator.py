"""
Benchmark engine.

Evaluates the rule baseline, the ML model and the hybrid detector on two held-out sets:

  test : same distribution as training, but with registered domains the model never saw;
  ood  : older phishing campaigns (a different feed file) plus hand-written, hard benign URLs.

Nothing here is tuned on these sets: model hyper-parameters, the rule flag threshold and the hybrid
weights were all selected on the validation split. Error cases carry causes computed from each URL's
own triggered rules and attributions, not canned text.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from features.extractor import URLExtractor
from models.explainer import FLAG_THRESHOLD, ML_WEIGHT, RULE_WEIGHT, combine_scores
from models.ml_detector import get_ml_detector
from models.rule_based import RULE_FLAG_THRESHOLD, RuleBasedDetector

RESULTS_PATH = os.path.join(BASE_DIR, "evaluation_results.json")
N_BOOT = 1000


# --------------------------------------------------------------------------- #
def _confusion(y: np.ndarray, p: np.ndarray) -> Dict[str, int]:
    return {
        "true_positives": int(((y == 1) & (p == 1)).sum()),
        "true_negatives": int(((y == 0) & (p == 0)).sum()),
        "false_positives": int(((y == 0) & (p == 1)).sum()),
        "false_negatives": int(((y == 1) & (p == 0)).sum()),
    }


def _rates(tp, tn, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision, "recall": recall, "f1_score": f1,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
    }


def compute_metrics(y_true, y_pred, scores=None) -> Dict[str, Any]:
    """Confusion matrix, rates, ranking metrics and 95% bootstrap confidence intervals."""
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    cm = _confusion(y, p)
    m = {k: round(v, 4) for k, v in _rates(cm["true_positives"], cm["true_negatives"],
                                           cm["false_positives"], cm["false_negatives"]).items()}
    if scores is not None and len(set(y.tolist())) == 2:
        m["roc_auc"] = round(float(roc_auc_score(y, scores)), 4)
        m["average_precision"] = round(float(average_precision_score(y, scores)), 4)

    rng = np.random.default_rng(42)
    boot = defaultdict(list)
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        c = _confusion(y[idx], p[idx])
        r = _rates(c["true_positives"], c["true_negatives"], c["false_positives"], c["false_negatives"])
        for k in ("f1_score", "recall", "false_positive_rate"):
            boot[k].append(r[k])
    ci = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in boot.items()}
    return {"sample_count": int(n), "confusion_matrix": cm, "metrics": m, "confidence_intervals_95": ci}


def _read(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [{"url": r["url"], "label": int(r["label"]), "source": r.get("source", "")} for r in csv.DictReader(f)]


# --------------------------------------------------------------------------- #
def _score_all(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    urls = [s["url"] for s in samples]
    ml = get_ml_detector()
    p_ml = np.clip(ml.predict_proba_batch(urls), 0.005, 0.995)
    rule_scores, hybrid_scores, rule_evals, breakdowns = [], [], [], []
    for u, p in zip(urls, p_ml):
        bd = URLExtractor.extract_detailed_breakdown(u)
        re_ = RuleBasedDetector.evaluate(bd["features"], bd)
        comb = combine_scores(re_, {"risk_score": int(round(p * 100))}, bd)
        rule_scores.append(re_["risk_score"])
        hybrid_scores.append(comb["score"])
        rule_evals.append(re_)
        breakdowns.append(bd)
    return {"p_ml": p_ml, "rule": np.array(rule_scores), "hybrid": np.array(hybrid_scores),
            "rule_evals": rule_evals, "breakdowns": breakdowns}


def _describe_error(kind: str, url: str, rule_eval: Dict[str, Any], ml_result: Dict[str, Any], score: int) -> str:
    """Cause analysis built from what the detectors actually saw for this URL."""
    triggers = [r["name"] for r in rule_eval["triggered_rules"]]
    drivers = [c for c in ml_result["top_contributing_features"]
               if c["direction"] == ("RISK_INCREASING" if kind == "fp" else "PROTECTIVE")][:2]
    driver_txt = "; ".join(f"{c['label']} ({c['impact']:+.2f})" for c in drivers) or "no single dominant factor"
    if kind == "fp":
        rules_txt = ("Rules fired: " + ", ".join(triggers) + ". ") if triggers else "No rule fired. "
        return f"{rules_txt}Main model drivers toward 'phishing': {driver_txt}."
    rules_txt = "No rule matched. " if not triggers else "Only low-weight rules matched (" + ", ".join(triggers) + "). "
    return f"{rules_txt}Model drivers that made it look normal: {driver_txt}."


def _error_cases(samples, scored, kind: str, field: str, limit: int = 6) -> List[Dict[str, Any]]:
    ml = get_ml_detector()
    cases = []
    for i, s in enumerate(samples):
        if field == "rule":
            pred = int(scored["rule"][i] >= RULE_FLAG_THRESHOLD)
            shown = int(scored["rule"][i])
        elif field == "ml":
            pred = int(scored["p_ml"][i] >= 0.5)
            shown = int(round(scored["p_ml"][i] * 100))
        else:
            pred = int(scored["hybrid"][i] >= FLAG_THRESHOLD)
            shown = int(scored["hybrid"][i])
        wrong = (s["label"] == 0 and pred == 1) if kind == "fp" else (s["label"] == 1 and pred == 0)
        if wrong:
            cases.append((i, shown))
    # most confident mistakes first: they are the most instructive
    cases.sort(key=lambda t: -t[1] if kind == "fp" else t[1])
    out = []
    for i, shown in cases[:limit]:
        s = samples[i]
        ml_result = ml.predict(s["url"])
        out.append({
            "url": s["url"], "source": s["source"],
            "true_label": "Legitimate" if kind == "fp" else "Phishing",
            "predicted_verdict": (f"Flagged (score {shown}/100)" if kind == "fp" else f"Missed (score {shown}/100)"),
            "root_cause_analysis": _describe_error(kind, s["url"], scored["rule_evals"][i], ml_result, shown),
        })
    return out


def _evaluate_split(name: str, path: str) -> Dict[str, Any]:
    samples = _read(path)
    y = np.array([s["label"] for s in samples])
    scored = _score_all(samples)

    preds = {
        "rule_based_baseline": (scored["rule"] >= RULE_FLAG_THRESHOLD).astype(int),
        "machine_learning": (scored["p_ml"] >= 0.5).astype(int),
        "hybrid_ensemble": (scored["hybrid"] >= FLAG_THRESHOLD).astype(int),
    }
    scores = {"rule_based_baseline": scored["rule"], "machine_learning": scored["p_ml"], "hybrid_ensemble": scored["hybrid"]}
    meta = {
        "rule_based_baseline": ("Transparent Rule-Based Baseline", "16 auditable rules",
                                f"Flags a URL when rule points reach {RULE_FLAG_THRESHOLD} (chosen on validation data)."),
        "machine_learning": ("ML Ensemble", "Gradient boosting + character patterns",
                             "41 engineered features plus character 3-5-grams, with calibrated probabilities."),
        "hybrid_ensemble": ("Hybrid Detector", "ML 80% + rules 20%",
                            f"What the app shows users: weighted score ({int(ML_WEIGHT * 100)}/{int(RULE_WEIGHT * 100)}), "
                            "critical-rule floor and trusted-domain cap, flagged at "
                            f"{FLAG_THRESHOLD}/100."),
    }
    detectors = {}
    for key, (nm, typ, desc) in meta.items():
        d = {"name": nm, "type": typ, "description": desc, **compute_metrics(y, preds[key], scores[key])}
        field = {"rule_based_baseline": "rule", "machine_learning": "ml", "hybrid_ensemble": "hybrid"}[key]
        fps, fns = _error_cases(samples, scored, "fp", field), _error_cases(samples, scored, "fn", field)
        d["error_cases"] = {
            "false_positives": fps, "false_negatives": fns,
            "total_fp_count": d["confusion_matrix"]["false_positives"],
            "total_fn_count": d["confusion_matrix"]["false_negatives"],
        }
        detectors[key] = d

    # Per-source behaviour of the hybrid detector (shows where benign URLs get flagged)
    by_source = {}
    hyb = preds["hybrid_ensemble"]
    for src in sorted({s["source"] for s in samples}):
        idx = [i for i, s in enumerate(samples) if s["source"] == src]
        lab = int(y[idx[0]])
        wrong = int((hyb[idx] != lab).sum())
        by_source[src] = {"count": len(idx), "label": "Phishing" if lab else "Legitimate",
                          "hybrid_errors": wrong, "hybrid_error_rate": round(wrong / len(idx), 4)}

    return {
        "dataset": {"name": name, "path": os.path.basename(path), "total": len(samples),
                    "phishing_count": int(y.sum()), "legitimate_count": int(len(y) - y.sum())},
        "detectors": detectors,
        "by_source": by_source,
    }


def run_evaluation(test_csv_path: str = None) -> Dict[str, Any]:
    test_path = test_csv_path or os.path.join(PROJECT_DIR, "data", "test.csv")
    ood_path = os.path.join(PROJECT_DIR, "data", "ood_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")
    get_ml_detector()

    test = _evaluate_split("Held-out domains (same distribution)", test_path)
    ood = _evaluate_split("Older campaigns + hard benign (out of distribution)", ood_path) if os.path.exists(ood_path) else None

    t = test["detectors"]
    report = {
        "evaluation_dataset": {
            "name": test["dataset"]["name"], "total_test_urls": test["dataset"]["total"],
            "phishing_count": test["dataset"]["phishing_count"], "legitimate_count": test["dataset"]["legitimate_count"],
            "class_ratio": f"{test['dataset']['phishing_count']} phishing / {test['dataset']['legitimate_count']} legitimate",
        },
        "detectors": t,                      # primary view (held-out test)
        "by_source": test["by_source"],
        "test": test,
        "ood": ood,
        "thresholds": {"rule_flag": RULE_FLAG_THRESHOLD, "hybrid_flag": FLAG_THRESHOLD,
                       "ml_weight": ML_WEIGHT, "rule_weight": RULE_WEIGHT},
        "training_summary": get_ml_detector().training_summary,
        "key_takeaways": _takeaways(test, ood),
        "limitations": [
            "The legitimate side is partly synthetic (well-known domains with generated paths) and partly an older "
            "public corpus, so false-positive rates on today's real web traffic may be higher than shown.",
            "URL-only analysis cannot see page content. Phishing hosted on a compromised legitimate site, or reached "
            "through an open redirect on a trusted domain, can look normal.",
            "Phishing infrastructure changes quickly. Retrain regularly with fresh feeds (python data/dataset_builder.py).",
        ],
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    def line(tag, d):
        m = d["metrics"]
        return (f"[+] {tag:<10} F1={m['f1_score']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
                f"FPR={m['false_positive_rate']:.3f} AUC={m.get('roc_auc', float('nan')):.3f}")
    for split_name, split in (("test", test), ("ood", ood)):
        if split:
            print(f"--- {split_name}: {split['dataset']['total']} URLs ---")
            for k, tag in (("rule_based_baseline", "Rules"), ("machine_learning", "ML"), ("hybrid_ensemble", "Hybrid")):
                print(line(tag, split["detectors"][k]))
    print(f"[+] Results saved to: {RESULTS_PATH}")
    return report


def _takeaways(test, ood) -> List[str]:
    t = test["detectors"]
    r, m, h = (t[k]["metrics"] for k in ("rule_based_baseline", "machine_learning", "hybrid_ensemble"))
    out = [
        f"On held-out domains the rules alone reach F1 {r['f1_score']:.2f} (recall {r['recall']:.0%}): transparent but "
        f"they miss many real phishing URLs because most attacks avoid obvious tricks.",
        f"The ML ensemble reaches F1 {m['f1_score']:.2f} and the hybrid F1 {h['f1_score']:.2f}; neither is perfect, "
        f"which is the realistic expectation for URL-only detection.",
    ]
    if ood:
        o = ood["detectors"]["hybrid_ensemble"]["metrics"]
        hard = ood["by_source"].get("Hand-written hard benign")
        out.append(f"On older campaigns the hybrid keeps F1 {o['f1_score']:.2f}, which suggests it is not just "
                   f"memorising the training feed.")
        if hard:
            out.append(f"Hard benign pages (real sign-in pages, repos and docs with phishing-like words): "
                       f"{hard['hybrid_errors']} of {hard['count']} wrongly flagged.")
    return out


def load_evaluation_results() -> Dict[str, Any]:
    """Loads precomputed results, recomputing when absent or from an older schema."""
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "test" in data and "thresholds" in data:
                return data
        except Exception:
            pass
    return run_evaluation()


if __name__ == "__main__":
    run_evaluation()
