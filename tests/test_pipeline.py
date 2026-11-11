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
        self.assertEqual(len(names), 41, "Expected 41 engineered URL features")
        self.assertEqual(len(set(names)), len(names), "Feature names must be unique")

        vec = extract_feature_vector(url)
        self.assertEqual(len(vec), len(names), "Vector shape must match feature names count")

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

    def test_ssrf_blocks_non_public_addresses(self):
        for host in ["0.0.0.0", "100.64.0.1", "169.254.169.254", "[::ffff:127.0.0.1]", "::1", "foo.internal"]:
            self.assertTrue(is_ssrf_risk(host), host)
        self.assertFalse(is_ssrf_risk("8.8.8.8"))

    def test_sandbox_rejects_bad_schemes_and_private_dns(self):
        from unittest import mock
        self.assertEqual(inspect_live_page_isolated("file:///etc/passwd")["status"], "blocked")
        with mock.patch("analyzer.page_analyzer.socket.getaddrinfo",
                        return_value=[(2, 1, 6, "", ("10.0.0.7", 0))]):
            res = inspect_live_page_isolated("http://innocent-looking.example/")
        self.assertEqual(res["status"], "blocked")


class TestDetectionQuality(unittest.TestCase):
    """Behavioural checks on realistic URLs (guards against regressions in either direction)."""

    SAFE = [
        "https://www.google.com/search?q=cybersecurity+phishing+research",
        "https://accounts.google.com/signin",
        "https://www.chase.com/personal/banking",
        "https://docs.python.org/3/library/urllib.html",
        "https://www.irctc.co.in/nget/train-search",
        "https://login.microsoftonline.com/common/oauth2/authorize",
    ]
    PHISH = [
        "http://paypal.com.verify-billing-center.xyz/index.html",
        "http://185.220.101.5/login/webscr.php?cmd=_login_submit",
        "https://apple-account-security-alert-resolution.top/signin/auth",
        "http://google.com@phishing-target-account-recovery.xyz/login",
        "http://xn--paypl-ora.com/webscr/login_submit.php",
        "https://sbi-netbanking-kyc-update.site/login",
        "http://amaz0n-secure-login.top/ap/signin",
    ]

    def test_safe_urls_not_flagged(self):
        for u in self.SAFE:
            r = RiskExplainer.analyze_url(u)
            self.assertEqual(r["unified_verdict"], "Likely safe", f"{u} -> {r['hybrid_risk_score']}")

    def test_phishing_urls_flagged(self):
        for u in self.PHISH:
            r = RiskExplainer.analyze_url(u)
            self.assertNotEqual(r["unified_verdict"], "Likely safe", f"{u} -> {r['hybrid_risk_score']}")

    def test_brand_substring_is_not_impersonation(self):
        for u in ["https://www.pineapple.com/", "https://www.purchase.com/checkout"]:
            self.assertEqual(extract_features(u)["brand_impersonation_score"], 0.0, u)

    def test_garbage_input_does_not_crash(self):
        for u in ["", "   ", "not a url", "http://", "://", "http://[::1", "a" * 5000, "http://\x00bad.com", "javascript:alert(1)"]:
            r = RiskExplainer.analyze_url(u)
            self.assertTrue(0 <= r["hybrid_risk_score"] <= 100, u[:30])


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import app as app_module
        cls.app_module = app_module
        cls.client = TestClient(app_module.app)

    def test_index_renders_with_samples_and_features(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Check a link", r.text)
        self.assertIn("url_entropy", r.text)

    def test_analyze_ok_and_validation(self):
        ok = self.client.post("/api/analyze", json={"url": "https://www.google.com"})
        self.assertEqual(ok.status_code, 200)
        self.assertIn("unified_verdict", ok.json())
        self.assertEqual(self.client.post("/api/analyze", json={"url": "   "}).status_code, 400)
        self.assertEqual(self.client.post("/api/analyze", json={}).status_code, 422)
        self.assertEqual(self.client.post("/api/analyze", json={"url": "http://a.com/" + "x" * 3000}).status_code, 400)

    def test_pasted_whitespace_is_cleaned(self):
        r = self.client.post("/api/analyze", json={"url": " https://www.google.com/\n "})
        self.assertEqual(r.status_code, 200)

    def test_batch(self):
        r = self.client.post("/api/batch", json={"urls": ["https://www.google.com", "http://paypal.com.verify-billing-center.xyz/"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["count"], 2)
        too_many = self.client.post("/api/batch", json={"urls": ["http://a.com"] * 26})
        self.assertEqual(too_many.status_code, 400)

    def test_feature_dictionary_covers_every_feature(self):
        names = {f["name"] for f in self.client.get("/api/features/dictionary").json()}
        self.assertEqual(names, set(get_feature_names()))

    def test_samples_match_expected_verdicts(self):
        for s in self.app_module.SAMPLE_URLS:
            r = RiskExplainer.analyze_url(s["url"])
            self.assertEqual(r["unified_verdict"], s["expected"], s["url"])

    def test_evaluation_and_health(self):
        ev = self.client.get("/api/evaluation").json()
        self.assertIn("test", ev)
        self.assertIn("ood", ev)
        h = self.client.get("/api/health").json()
        self.assertEqual(h["status"], "ok")
        self.assertEqual(h["feature_count"], 41)


if __name__ == "__main__":
    unittest.main()
