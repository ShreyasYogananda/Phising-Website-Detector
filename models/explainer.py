"""
Hybrid risk synthesis and explanation.

`combine_scores` is the single source of truth for how the rule engine and the ML model are merged.
Both the live API and the offline evaluator call it, so the numbers shown in the benchmark are the
numbers users actually get.
"""

import os
import sys
from typing import Any, Dict, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from features.extractor import URLExtractor
from models.ml_detector import get_ml_detector
from models.rule_based import RuleBasedDetector

SECURITY_DISCLAIMER = (
    "This tool estimates risk from the web address (and optionally the page's visible structure). "
    "It cannot prove that a link is safe. Open important accounts by typing the address yourself, "
    "and never enter passwords or payment details on a page you were sent unexpectedly."
)

# Selected on the validation split (see evaluation/evaluator.py).
ML_WEIGHT = 0.8
RULE_WEIGHT = 1.0 - ML_WEIGHT
FLAG_THRESHOLD = 35
HIGH_THRESHOLD = 65
CRITICAL_FLOOR = 70            # a CRITICAL rule (raw IP host, brand impersonation) raises the score to at least this
TRUSTED_CAP = 25               # official brand domain with no red flags is capped here

VERDICT_BADGE = {"High risk": "danger", "Suspicious": "warning", "Likely safe": "success"}
SUMMARY = {
    "High risk": "Strong signs of phishing. Do not enter passwords or payment details here.",
    "Suspicious": "Some warning signs. Check the address carefully before you trust this page.",
    "Likely safe": "The address looks normal and no strong warning signs were found.",
}


def sanitize_to_native_types(data):
    """Recursively converts numpy values to plain Python so FastAPI can serialise them."""
    if isinstance(data, dict):
        return {str(k): sanitize_to_native_types(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [sanitize_to_native_types(v) for v in data]
    if hasattr(data, "item"):
        return data.item()
    return data


def verdict_for_score(score: int) -> str:
    return "High risk" if score >= HIGH_THRESHOLD else ("Suspicious" if score >= FLAG_THRESHOLD else "Likely safe")


def combine_scores(rule_result: Dict[str, Any], ml_result: Dict[str, Any], breakdown: Dict[str, Any]) -> Dict[str, Any]:
    """Merges the two detectors into one 0-100 score. Returns the score plus the reasons it was adjusted."""
    rule_score, ml_score = rule_result["risk_score"], ml_result["risk_score"]
    score = ML_WEIGHT * ml_score + RULE_WEIGHT * rule_score
    adjustments = []

    has_critical = any(r["severity"] == "CRITICAL" for r in rule_result["triggered_rules"])
    if has_critical and score < CRITICAL_FLOOR:
        score = float(CRITICAL_FLOOR)
        adjustments.append("critical_rule_floor")

    a = breakdown["anatomy"]
    feats = breakdown["features"]
    deceptive = feats["has_ip_address"] or feats["at_symbol_count"] or feats["has_punycode"] or a["userinfo"] != "(none)"
    if breakdown.get("official_brand") and not has_critical and not deceptive:
        if score > TRUSTED_CAP:
            score = float(TRUSTED_CAP)
        adjustments.append("trusted_domain_cap")

    return {"score": int(round(min(max(score, 0), 100))), "adjustments": adjustments}


class RiskExplainer:
    @classmethod
    def analyze_url(cls, url_str: str, page_analysis_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        breakdown = URLExtractor.extract_detailed_breakdown(url_str)
        rule_result = RuleBasedDetector.evaluate(breakdown["features"], breakdown)
        ml_result = get_ml_detector().predict(url_str)

        combined = combine_scores(rule_result, ml_result, breakdown)
        score = combined["score"]

        # ---- optional live-page evidence is applied BEFORE the verdict is derived -------------
        page_evidence = None
        page_signals = []
        if page_analysis_result and page_analysis_result.get("status") == "success":
            page_evidence = page_analysis_result
            for pf in page_analysis_result.get("findings", []):
                page_signals.append({
                    "source": "Live page check", "severity": pf["severity"], "title": pf["title"],
                    "detail": pf["detail"], "points": pf.get("risk_delta", 0),
                })
                score += pf.get("risk_delta", 0)
            score = int(min(max(score, 0), 100))
            if "trusted_domain_cap" in combined["adjustments"] and any(
                    s["severity"] == "CRITICAL" for s in page_signals):
                combined["adjustments"].append("page_overrides_trust")

        verdict = verdict_for_score(score)

        # ---- agreement between the two detectors ------------------------------------------------
        rule_score, ml_score = rule_result["risk_score"], ml_result["risk_score"]
        rule_flag, ml_flag = rule_result["prediction"], ml_result["prediction"]
        gap = abs(ml_score - min(rule_score * 2, 100))          # rule scores run lower by design (flag at 15)
        if rule_flag == ml_flag and gap <= 35:
            agreement, agreement_desc = "Detectors agree", "The rule engine and the ML model reach the same conclusion."
            confidence = min(max(ml_result["confidence"], 0.75), 0.97)
        elif rule_flag == ml_flag:
            agreement, agreement_desc = "Mostly agree", "Both point the same way, but the strength of the evidence differs."
            confidence = min(max(ml_result["confidence"], 0.6), 0.85)
        else:
            who = "The ML model flags this URL but no rule fired" if ml_flag else \
                "A rule fired but the ML model considers the URL normal"
            agreement, agreement_desc = "Detectors disagree", f"{who}. Treat this result as less certain."
            confidence = min(ml_result["confidence"], 0.55)
        if "trusted_domain_cap" in combined["adjustments"]:
            confidence = max(confidence, 0.8)

        # ---- signals shown to the user ----------------------------------------------------------
        signals = []
        for r in rule_result["triggered_rules"]:
            signals.append({"source": "Rule", "severity": r["severity"], "title": r["name"],
                            "detail": r["description"], "points": r["points"], "why": r["rationale"]})
        if not rule_result["triggered_rules"]:
            signals.append({"source": "Rule", "severity": "BENIGN", "title": "No warning rules triggered",
                            "detail": "None of the 16 security rules matched this address.", "points": 0})
        if "trusted_domain_cap" in combined["adjustments"]:
            signals.append({"source": "Trust check", "severity": "BENIGN",
                            "title": f"Official {breakdown['official_brand'].title()} domain",
                            "detail": "The registered domain matches a verified official domain for this brand. "
                                      "Attackers can still abuse open redirects or user content on trusted sites.",
                            "points": 0})
        for c in ml_result["top_contributing_features"]:
            if c["direction"] == "RISK_INCREASING":
                signals.append({"source": "ML model", "severity": "HIGH" if c["impact"] > 0.25 else "MEDIUM",
                                "title": c["label"], "detail": c["description"], "points": round(c["impact"] * 100)})
            elif c["direction"] == "PROTECTIVE":
                signals.append({"source": "ML model", "severity": "BENIGN", "title": c["label"],
                                "detail": c["description"], "points": round(c["impact"] * 100)})
        for p in ml_result.get("pattern_signals", []):
            signals.append({"source": "ML model", "severity": "LOW", "title": f"Phishing-style pattern \"{p['pattern']}\"",
                            "detail": "This character sequence is common in known phishing URLs.", "points": 0})
        signals.extend(page_signals)

        return sanitize_to_native_types({
            "url": url_str,
            "unified_verdict": verdict,
            "verdict_badge": VERDICT_BADGE[verdict],
            "summary_statement": SUMMARY[verdict],
            "hybrid_risk_score": score,
            "confidence": round(confidence, 3),
            "uncertainty": round(1.0 - confidence, 3),
            "agreement_level": agreement,
            "agreement_description": agreement_desc,
            "score_adjustments": combined["adjustments"],
            "models_comparison": {"rule_based": rule_result, "machine_learning": ml_result},
            "signals_summary": signals,
            "breakdown": breakdown,
            "optional_page_analysis": page_evidence,
            "disclaimer": SECURITY_DISCLAIMER,
        })
