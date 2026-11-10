"""
Automated Pipeline Tests for Explainable Phishing Risk Checker.
Verifies feature extraction, rule baseline, ML model inference, explainer, and security sandboxes.
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.extractor import URLExtractor, extract_features, extract_feature_vector, get_feature_names
from models.rule_based import RuleBasedDetector
from models.ml_detector import MLDetector, get_ml_detector
from models.explainer import RiskExplainer
from analyzer.page_analyzer import is_ssrf_risk, inspect_live_page_isolated

class TestPhishingPipeline(unittest.TestCase):

    def test_feature_extractor_dimensions(self):
        url = "https://www.google.com/search?q=machine+learning"
        names = get_feature_names()
        self.assertEqual(len(names), 21, "Expected 21 engineered URL features")

        vec = extract_feature_vector(url)
        self.assertEqual(len(vec), 21, "Vector shape must match feature names count")

        feats = extract_features(url)
        self.assertEqual(feats["has_ip_address"], 0.0)
        self.assertEqual(feats["has_https"], 1.0)
        self.assertEqual(feats["brand_impersonation_score"], 0.0)

    def test_brand_impersonation_detection(self):
        # Misleading brand in subdomain on untrusted registered domain
        spoof_url = "http://paypal.com.verify-billing-center.xyz/index.html"
        feats = extract_features(spoof_url)
        self.assertGreater(feats["brand_impersonation_score"], 0, "Should detect PayPal impersonation")
        self.assertEqual(feats["suspicious_tld"], 1.0, "Should detect .xyz as suspicious TLD")

        # Legitimate brand on genuine domain
        legit_url = "https://www.paypal.com/signin"
        feats_legit = extract_features(legit_url)
        self.assertEqual(feats_legit["brand_impersonation_score"], 0.0, "Legitimate PayPal domain should not trigger spoofing")

    def test_ip_address_detection(self):
        ip_url = "http://192.168.1.100/login/webscr.php"
        feats = extract_features(ip_url)
        self.assertEqual(feats["has_ip_address"], 1.0, "Should identify raw IP host")

    def test_rule_based_detector(self):
        # Safe URL
        safe_url = "https://www.wikipedia.org/wiki/Phishing"
        breakdown_safe = URLExtractor.extract_detailed_breakdown(safe_url)
        res_safe = RuleBasedDetector.evaluate(breakdown_safe["features"], breakdown_safe)
        self.assertEqual(res_safe["verdict"], "Likely safe")
        self.assertLess(res_safe["risk_score"], 35)

        # High risk phishing URL
        phish_url = "http://185.220.101.5/secure/login/webscr.php"
        breakdown_phish = URLExtractor.extract_detailed_breakdown(phish_url)
        res_phish = RuleBasedDetector.evaluate(breakdown_phish["features"], breakdown_phish)
        self.assertGreaterEqual(res_phish["risk_score"], 35)
        self.assertTrue(len(res_phish["triggered_rules"]) > 0)

    def test_ml_detector_inference(self):
        ml = get_ml_detector()
        res = ml.predict("https://apple-account-security-alert-resolution.top/signin/auth")
        self.assertIn("phishing_probability", res)
        self.assertTrue(0.0 <= res["phishing_probability"] <= 1.0)
        self.assertIn("top_contributing_features", res)
        self.assertTrue(len(res["top_contributing_features"]) > 0)

    def test_risk_explainer_synthesis(self):
        url = "http://paypal.com.verify-billing-center.xyz/index.html"
        explanation = RiskExplainer.analyze_url(url)
        
        self.assertIn(explanation["unified_verdict"], ["Likely safe", "Suspicious", "High risk"])
        self.assertTrue(0 <= explanation["hybrid_risk_score"] <= 100)
        self.assertTrue(0.0 <= explanation["confidence"] <= 1.0)
        self.assertTrue(0.0 <= explanation["uncertainty"] <= 1.0)
        self.assertIn("signals_summary", explanation)
        self.assertIn("disclaimer", explanation)
        self.assertIn("models_comparison", explanation)

    def test_ssrf_safety_defense(self):
        # Should block internal and loopback IP hosts
        self.assertTrue(is_ssrf_risk("127.0.0.1"))
        self.assertTrue(is_ssrf_risk("localhost"))
        self.assertTrue(is_ssrf_risk("192.168.1.1"))
        self.assertTrue(is_ssrf_risk("10.0.0.5"))

        # Should block inspection of loopback
        res = inspect_live_page_isolated("http://127.0.0.1:8000/internal")
        self.assertEqual(res["status"], "blocked")

if __name__ == "__main__":
    unittest.main()
