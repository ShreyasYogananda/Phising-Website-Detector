"""
Explainability & Hybrid Risk Synthesis Engine
Combines Rule-Based and Machine Learning detectors into an explainable, unified assessment.
"""

import os
import sys
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from features.extractor import URLExtractor, extract_features
from models.rule_based import RuleBasedDetector
from models.ml_detector import get_ml_detector

SECURITY_DISCLAIMER = (
    "SECURITY NOTICE: This tool provides heuristic and machine-learning risk indicators "
    "for educational and investigative purposes. It is an analytical aid and CANNOT guarantee "
    "that any URL is completely safe. Always exercise caution, check domains directly, "
    "and never submit sensitive credentials or payment data on untrusted websites."
)

def sanitize_to_native_types(data):
    """Recursively converts numpy numbers and tuples to native Python types for JSON serialization."""
    if isinstance(data, dict):
        return {str(k): sanitize_to_native_types(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_to_native_types(v) for v in data]
    elif hasattr(data, "item"): # handles numpy float32, int64, etc.
        return data.item()
    return data

class RiskExplainer:
    """
    Synthesizes rule-based baseline and ML detector signals into a comprehensive,
    explainable risk evaluation.
    """

    @classmethod
    def analyze_url(cls, url_str: str, page_analysis_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Full end-to-end analysis of a submitted URL:
        1. Validates and extracts features and anatomy.
        2. Executes Rule-Based Baseline Detector.
        3. Executes Machine Learning Detector.
        4. Synthesizes combined risk score, verdict, confidence, and uncertainty.
        5. Gathers explainable signals (rules, ML drivers, optional page findings).
        """
        # Feature extraction & breakdown
        breakdown = URLExtractor.extract_detailed_breakdown(url_str)
        features = breakdown["features"]

        # Run Rule-Based Detector
        rule_result = RuleBasedDetector.evaluate(features, breakdown)

        # Run Machine Learning Detector
        ml_detector = get_ml_detector()
        ml_result = ml_detector.predict(url_str)

        # Compute Consensus and Hybrid Risk Score
        rule_score = rule_result["risk_score"]
        ml_score = ml_result["risk_score"]

        # If a critical rule triggers (e.g., brand spoofing or direct IP), ensure baseline guardrail
        critical_rule_active = any(r["severity"] == "CRITICAL" for r in rule_result["triggered_rules"])
        
        # Weighted hybrid risk score (55% ML, 45% Rules)
        raw_hybrid = (0.55 * ml_score) + (0.45 * rule_score)
        if critical_rule_active and raw_hybrid < 65:
            # Boost hybrid score if critical rule was positively identified
            raw_hybrid = max(raw_hybrid, 70.0)

        hybrid_risk_score = int(round(raw_hybrid))
        hybrid_risk_score = min(max(hybrid_risk_score, 0), 100)

        # Unified Verdict
        if hybrid_risk_score >= 70:
            unified_verdict = "High risk"
            verdict_badge = "danger"
            summary_statement = "Strong indicators of phishing or malicious deception detected. Do not proceed or enter credentials."
        elif hybrid_risk_score >= 35:
            unified_verdict = "Suspicious"
            verdict_badge = "warning"
            summary_statement = "Anomalous patterns or deceptive characteristics found. High degree of caution recommended."
        else:
            unified_verdict = "Likely safe"
            verdict_badge = "success"
            summary_statement = "URL conforms to normal legitimate structure with no significant risk flags detected."

        # Model Agreement & Uncertainty Analysis
        score_diff = abs(ml_score - rule_score)
        prediction_agreement = (rule_result["prediction"] == ml_result["prediction"])

        if score_diff <= 20:
            agreement_level = "High Agreement"
            agreement_description = "Both the rule-based baseline and the ML classifier strongly converge on this verdict."
            confidence = min(max(ml_result["confidence"], 0.75), 0.96)
        elif score_diff <= 40:
            agreement_level = "Moderate Agreement"
            agreement_description = "Both detectors arrive at similar conclusions with minor score variance."
            confidence = 0.70
        else:
            agreement_level = "Detector Divergence"
            agreement_description = (
                f"Discrepancy observed: Rule baseline assessed risk at {rule_score}/100, while ML assessed {ml_score}/100. "
                "This indicates borderline or novel feature combinations."
            )
            confidence = 0.52

        uncertainty = round(1.0 - confidence, 3)

        # Synthesize Key Contributing Signals
        signals_summary = []

        # 1. Rules signals
        if rule_result["triggered_rules"]:
            for r in rule_result["triggered_rules"]:
                signals_summary.append({
                    "source": "Rule Engine",
                    "severity": r["severity"],
                    "title": r["name"],
                    "detail": r["description"],
                    "points": r["points"]
                })
        else:
            signals_summary.append({
                "source": "Rule Engine",
                "severity": "BENIGN",
                "title": "Clean Rule Evaluation",
                "detail": "No heuristic security alarms or blacklist patterns were triggered.",
                "points": 0
            })

        # 2. Top ML signals
        for c in ml_result["top_contributing_features"]:
            if c["direction"] == "RISK_INCREASING":
                signals_summary.append({
                    "source": "ML Detector",
                    "severity": "HIGH" if abs(c["impact"]) > 0.3 else "MEDIUM",
                    "title": f"Feature Signal: {c['feature']}",
                    "detail": c["description"],
                    "points": round(c["impact"] * 25, 1)
                })
            elif c["direction"] == "PROTECTIVE":
                signals_summary.append({
                    "source": "ML Detector",
                    "severity": "BENIGN",
                    "title": f"Protective Signal: {c['feature']}",
                    "detail": f"{c['description']} (characteristic of legitimate domains)",
                    "points": round(c["impact"] * 25, 1)
                })

        # Incorporate optional page analysis if provided
        optional_page_evidence = None
        if page_analysis_result and page_analysis_result.get("status") == "success":
            optional_page_evidence = page_analysis_result
            # Check page signals that adjust risk
            page_findings = page_analysis_result.get("findings", [])
            for pf in page_findings:
                signals_summary.append({
                    "source": "Live Page Inspection (Isolated)",
                    "severity": pf["severity"],
                    "title": pf["title"],
                    "detail": pf["detail"],
                    "points": pf.get("risk_delta", 0)
                })
                # Add to hybrid score
                hybrid_risk_score = min(max(hybrid_risk_score + pf.get("risk_delta", 0), 0), 100)

        return sanitize_to_native_types({
            "url": url_str,
            "unified_verdict": unified_verdict,
            "verdict_badge": verdict_badge,
            "summary_statement": summary_statement,
            "hybrid_risk_score": hybrid_risk_score,
            "confidence": round(confidence, 3),
            "uncertainty": uncertainty,
            "agreement_level": agreement_level,
            "agreement_description": agreement_description,
            "models_comparison": {
                "rule_based": rule_result,
                "machine_learning": ml_result
            },
            "signals_summary": signals_summary,
            "breakdown": breakdown,
            "optional_page_analysis": optional_page_evidence,
            "disclaimer": SECURITY_DISCLAIMER
        })
