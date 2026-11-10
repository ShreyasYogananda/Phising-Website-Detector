"""
Transparent Rule-Based Baseline Detector
PhishGuard Auditable Security Heuristic Rule Engine.

Provides an auditable, deterministic security heuristic baseline with point-based risk scoring
and comprehensive explainability rationale for each triggered rule.
"""

from typing import Dict, Any, List

class RuleBasedDetector:
    """
    Transparent heuristic baseline detector for phishing URL risk.
    """

    RULES = [
        {
            "id": "RULE_01_IP_HOST",
            "name": "Direct IP Address Host",
            "points": 35,
            "severity": "CRITICAL",
            "description": "The URL host is a raw IPv4/IPv6 address instead of a domain name.",
            "rationale": "Legitimate enterprise services use registered domain names. Attackers use raw IPs to bypass DNS-based reputation scoring and domain take-downs."
        },
        {
            "id": "RULE_02_BRAND_SPOOF",
            "name": "Target Brand Impersonation",
            "points": 40,
            "severity": "CRITICAL",
            "description": "A recognized high-value brand is present in the subdomain or path, but the registered domain is unaffiliated.",
            "rationale": "Classic credential harvesting attack where attacker registers a misleading domain and puts the target brand name in the subdomain or path."
        },
        {
            "id": "RULE_03_AT_SYMBOL",
            "name": "Embedded '@' Obfuscation Character",
            "points": 30,
            "severity": "HIGH",
            "description": "URL contains an '@' symbol in the authority or path segment.",
            "rationale": "Under RFC 3986, characters preceding '@' are treated as user credentials and discarded by browsers, redirecting to the host that follows."
        },
        {
            "id": "RULE_04_DOUBLE_SLASH",
            "name": "Double Slash '//' Path Redirect",
            "points": 25,
            "severity": "HIGH",
            "description": "URL contains '//' after the protocol scheme.",
            "rationale": "Frequently weaponized in path-confusion and open-redirect vulnerabilities to deceive browser address parsing."
        },
        {
            "id": "RULE_05_PUNYCODE",
            "name": "Punycode / Homograph Domain",
            "points": 25,
            "severity": "HIGH",
            "description": "Domain contains 'xn--' Punycode representation of Internationalized Domain Names (IDN).",
            "rationale": "Attackers substitute visually identical Unicode glyphs (e.g., Cyrillic 'а' for Latin 'a') to forge deceptive counterfeit domains."
        },
        {
            "id": "RULE_06_EXCESSIVE_SUBDOMAINS",
            "name": "Excessive Subdomain Depth",
            "points": 20,
            "severity": "MEDIUM",
            "description": "URL contains 3 or more nested subdomains.",
            "rationale": "Deep subdomain hierarchies are commonly used to push the actual attacker-controlled domain out of view on mobile screens."
        },
        {
            "id": "RULE_07_SUSPICIOUS_KEYWORD_DENSITY",
            "name": "Credential Harvesting Keywords",
            "points": 20,
            "severity": "HIGH",
            "description": "Multiple high-risk credential keywords (e.g. login, verify, account, banking) found in unverified domain paths.",
            "rationale": "Phishing landing pages heavily concentrate deceptive action-inducing words to induce panic and urgency."
        },
        {
            "id": "RULE_08_SUSPICIOUS_TLD",
            "name": "High-Abuse Top Level Domain",
            "points": 20,
            "severity": "MEDIUM",
            "description": "The URL uses a Top Level Domain associated with high abuse rates or disposable registrations.",
            "rationale": "Certain budget or lax registrar TLDs (e.g. .xyz, .top, .buzz, .work) host disproportionate volumes of short-lived malicious infrastructure."
        },
        {
            "id": "RULE_09_HIGH_ENTROPY_AND_LENGTH",
            "name": "Anomalous Length & High Character Entropy",
            "points": 15,
            "severity": "MEDIUM",
            "description": "URL length exceeds 80 characters with high Shannon entropy (> 4.2).",
            "rationale": "Algorithmic obfuscation tokens, session injection parameters, or DGA patterns exhibit elevated Shannon randomness."
        },
        {
            "id": "RULE_10_NON_STANDARD_PORT",
            "name": "Non-Standard Web Port",
            "points": 15,
            "severity": "MEDIUM",
            "description": "Host specifies a custom port number other than standard 80/443.",
            "rationale": "Standard legitimate web applications operate on default ports 80/443; custom ports often indicate rogue tunnels or ad-hoc attack servers."
        },
        {
            "id": "RULE_11_URL_SHORTENER",
            "name": "URL Shortener Cloaking",
            "points": 15,
            "severity": "LOW",
            "description": "URL uses a URL shortening redirection service.",
            "rationale": "URL shorteners conceal the ultimate destination domain, preventing inspection of landing page reputation before click."
        }
    ]

    @classmethod
    def evaluate(cls, features: Dict[str, float], breakdown: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluates heuristic security rules against extracted URL features.
        Returns risk score (0-100), triggered rules list, prediction, and verdict.
        """
        raw_score = 0
        triggered_rules: List[Dict[str, Any]] = []

        # Rule 1: IP Host
        if features.get("has_ip_address", 0) > 0:
            rule = cls._get_rule("RULE_01_IP_HOST")
            raw_score += rule["points"]
            triggered_rules.append(rule)

        # Rule 2: Brand Impersonation
        if features.get("brand_impersonation_score", 0) > 0:
            rule = cls._get_rule("RULE_02_BRAND_SPOOF").copy()
            if breakdown and breakdown.get("impersonated_brands"):
                brands_str = ", ".join(breakdown["impersonated_brands"]).title()
                rule["description"] += f" Detected target brand(s): {brands_str}."
            raw_score += rule["points"]
            triggered_rules.append(rule)

        # Rule 3: '@' symbol
        if features.get("at_symbol_count", 0) > 0:
            rule = cls._get_rule("RULE_03_AT_SYMBOL")
            raw_score += rule["points"]
            triggered_rules.append(rule)

        # Rule 4: Double slash in path
        if features.get("double_slash_redirect", 0) > 0:
            rule = cls._get_rule("RULE_04_DOUBLE_SLASH")
            raw_score += rule["points"]
            triggered_rules.append(rule)

        # Rule 5: Punycode
        if features.get("has_punycode", 0) > 0:
            rule = cls._get_rule("RULE_05_PUNYCODE")
            raw_score += rule["points"]
            triggered_rules.append(rule)

        # Rule 6: Excessive subdomains
        if features.get("subdomain_count", 0) >= 3:
            rule = cls._get_rule("RULE_06_EXCESSIVE_SUBDOMAINS")
            raw_score += rule["points"]
            triggered_rules.append(rule)

        # Rule 7: Suspicious keyword density
        if features.get("suspicious_keyword_count", 0) >= 2 or (
            features.get("suspicious_keyword_count", 0) >= 1 and features.get("has_https", 1) == 0
        ):
            rule = cls._get_rule("RULE_07_SUSPICIOUS_KEYWORD_DENSITY").copy()
            if breakdown and breakdown.get("matched_keywords"):
                rule["description"] += f" Matched keywords: {', '.join(breakdown['matched_keywords'])}."
            raw_score += rule["points"]
            triggered_rules.append(rule)

        # Rule 8: Suspicious TLD
        if features.get("suspicious_tld", 0) > 0:
            rule = cls._get_rule("RULE_08_SUSPICIOUS_TLD")
            raw_score += rule["points"]
            triggered_rules.append(rule)

        # Rule 9: High entropy and length
        if features.get("url_length", 0) > 80 and features.get("url_entropy", 0) > 4.2:
            rule = cls._get_rule("RULE_09_HIGH_ENTROPY_AND_LENGTH")
            raw_score += rule["points"]
            triggered_rules.append(rule)

        # Rule 10: Non-standard port
        if features.get("has_port", 0) > 0:
            rule = cls._get_rule("RULE_10_NON_STANDARD_PORT")
            raw_score += rule["points"]
            triggered_rules.append(rule)

        # Rule 11: Shortener
        if features.get("is_shortened", 0) > 0:
            rule = cls._get_rule("RULE_11_URL_SHORTENER")
            raw_score += rule["points"]
            triggered_rules.append(rule)

        # Bound score between 0 and 100
        risk_score = min(max(raw_score, 0), 100)

        # Baseline decision threshold:
        # Binary prediction for evaluation: >= 35 considered flagged (phishing/suspicious)
        binary_prediction = 1 if risk_score >= 35 else 0

        if risk_score >= 65:
            verdict = "High risk"
            confidence = min(0.65 + (risk_score / 250), 0.98)
        elif risk_score >= 35:
            verdict = "Suspicious"
            confidence = 0.60 + (risk_score - 35) / 100
        else:
            verdict = "Likely safe"
            confidence = max(0.85 - (risk_score / 100), 0.55)

        return {
            "model_name": "Transparent Rule-Based Baseline",
            "risk_score": int(risk_score),
            "prediction": int(binary_prediction),
            "verdict": verdict,
            "confidence": round(float(confidence), 3),
            "triggered_rules_count": len(triggered_rules),
            "triggered_rules": triggered_rules,
            "total_rules_evaluated": len(cls.RULES)
        }

    @classmethod
    def _get_rule(cls, rule_id: str) -> Dict[str, Any]:
        for r in cls.RULES:
            if r["id"] == rule_id:
                return r
        return {"id": rule_id, "name": rule_id, "points": 10, "severity": "LOW", "description": "", "rationale": ""}
