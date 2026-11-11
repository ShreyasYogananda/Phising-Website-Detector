"""
Transparent rule-based detector.

Every rule is deterministic and carries a plain-language explanation, so a reader can audit exactly
why a URL scored the way it did. Points add up to a 0-100 risk score.

Rules deliberately avoid penalising normal behaviour of legitimate sites, for example a sign-in
page on a brand's own domain, a subdomain such as docs.example.com, or a brand name appearing as
a repository owner in a GitHub path.
"""

from typing import Any, Dict, List


# Chosen on the validation split: flag a URL when at least one meaningful rule fires.
RULE_FLAG_THRESHOLD = 15
RULE_HIGH_THRESHOLD = 50


class RuleBasedDetector:
    RULES: List[Dict[str, Any]] = [
        {"id": "RULE_01_IP_HOST", "name": "Raw IP address instead of a domain", "points": 35, "severity": "CRITICAL",
         "description": "The link points to a numeric IP address instead of a website name.",
         "rationale": "Real services use registered domain names. Attackers use bare IPs to dodge domain take-downs and reputation checks."},
        {"id": "RULE_02_BRAND_SPOOF", "name": "Imitates a well-known brand", "points": 40, "severity": "CRITICAL",
         "description": "The domain name uses or imitates a well-known brand, but the site does not belong to that brand.",
         "rationale": "The classic credential-theft trick: put the brand in the web address so the link looks official at a glance."},
        {"id": "RULE_03_AT_SYMBOL", "name": "'@' sign hides the real destination", "points": 30, "severity": "HIGH",
         "description": "The link contains an '@' sign before the real host name.",
         "rationale": "Browsers ignore everything before '@' and go to what follows it, so attackers put a trusted name in front."},
        {"id": "RULE_04_EMBEDDED_URL", "name": "Second address hidden inside the link", "points": 25, "severity": "HIGH",
         "description": "The path or query contains another web address or a '//' redirect.",
         "rationale": "Used to bounce victims through a trusted-looking address or to confuse people reading the link."},
        {"id": "RULE_05_PUNYCODE", "name": "Look-alike characters in the domain", "points": 25, "severity": "HIGH",
         "description": "The domain uses punycode (xn--), which can display foreign letters that look like English ones.",
         "rationale": "Homograph attacks swap letters (for example Cyrillic 'a' for Latin 'a') to fake a brand."},
        {"id": "RULE_06_EXCESSIVE_SUBDOMAINS", "name": "Unusually deep subdomains", "points": 15, "severity": "MEDIUM",
         "description": "The domain has four or more subdomain levels.",
         "rationale": "Long subdomain chains push the real domain out of view, for example login.secure.account.example.attacker.com."},
        {"id": "RULE_07_CREDENTIAL_KEYWORDS", "name": "Sign-in or payment words on an unverified site", "points": 20, "severity": "HIGH",
         "description": "Several words like login, verify, account or billing appear in the address of a site that is not an official brand domain.",
         "rationale": "Phishing pages try to look like account-security pages. Official domains use these words too, so this only counts on unverified domains."},
        {"id": "RULE_08_SUSPICIOUS_TLD", "name": "Domain ending often used for abuse", "points": 15, "severity": "MEDIUM",
         "description": "The domain ends in an extension (such as .xyz, .top or .icu) that is cheap and heavily abused.",
         "rationale": "Many abused extensions are nearly free to register, which makes them attractive for disposable phishing sites."},
        {"id": "RULE_09_HIGH_ENTROPY_AND_LENGTH", "name": "Long, random-looking address", "points": 15, "severity": "MEDIUM",
         "description": "The link is over 80 characters and its characters look machine-generated.",
         "rationale": "Phishing kits often add random tokens and tracking IDs to each victim's link."},
        {"id": "RULE_10_NON_STANDARD_PORT", "name": "Unusual network port", "points": 15, "severity": "MEDIUM",
         "description": "The link names a port other than 80 or 443.",
         "rationale": "Ordinary websites rarely need a custom port; temporary attack servers often do."},
        {"id": "RULE_11_URL_SHORTENER", "name": "Link shortener hides the destination", "points": 15, "severity": "LOW",
         "description": "The link goes through a URL-shortening service, so the real destination is not visible.",
         "rationale": "Shorteners are legitimate, but they stop you from checking where a link leads before you click."},
        {"id": "RULE_12_FREE_HOSTING", "name": "Built on a free website platform", "points": 15, "severity": "MEDIUM",
         "description": "The site is hosted on a platform where anyone can publish a page for free.",
         "rationale": "Phishing campaigns are frequently published on free platforms. Many legitimate projects are too, so this is a mild signal on its own."},
        {"id": "RULE_13_HYPHEN_STUFFED_DOMAIN", "name": "Domain stuffed with hyphens", "points": 15, "severity": "MEDIUM",
         "description": "The domain name contains three or more hyphens.",
         "rationale": "Phishers glue brand and security words together, e.g. chase-online-banking-security-update."},
        {"id": "RULE_14_MACHINE_GENERATED_DOMAIN", "name": "Domain looks machine-generated", "points": 15, "severity": "MEDIUM",
         "description": "The domain name is mostly digits or an unpronounceable run of letters.",
         "rationale": "Throwaway attack domains are often created in bulk by scripts."},
        {"id": "RULE_15_BRAND_IN_PATH", "name": "Brand name appears in the page path", "points": 10, "severity": "LOW",
         "description": "A well-known brand name appears in the path, on a domain that is not that brand's.",
         "rationale": "Weak on its own (many legitimate sites mention brands), but meaningful when combined with other warnings."},
        {"id": "RULE_16_SERVER_SCRIPT_LOGIN", "name": "Sign-in page served by a raw script file", "points": 10, "severity": "LOW",
         "description": "The address ends in a script file such as .php or .asp and contains sign-in words.",
         "rationale": "Stolen-credential pages are commonly simple scripts such as login.php."},
    ]

    _BY_ID = {r["id"]: r for r in RULES}

    @classmethod
    def _rule(cls, rule_id: str, extra: str = "") -> Dict[str, Any]:
        rule = dict(cls._BY_ID[rule_id])
        if extra:
            rule["description"] = f"{rule['description']} {extra}"
        return rule

    @classmethod
    def evaluate(cls, features: Dict[str, float], breakdown: Dict[str, Any] = None) -> Dict[str, Any]:
        f = lambda k: features.get(k, 0)  # noqa: E731
        breakdown = breakdown or {}
        official = breakdown.get("official_brand")
        evidence = breakdown.get("brand_evidence", {})
        triggered: List[Dict[str, Any]] = []

        if f("has_ip_address") > 0:
            triggered.append(cls._rule("RULE_01_IP_HOST"))

        host_brands = list(dict.fromkeys(evidence.get("host", []) + evidence.get("typosquat", [])))
        if host_brands and not official:
            triggered.append(cls._rule("RULE_02_BRAND_SPOOF", f"Brand(s) involved: {', '.join(b.title() for b in host_brands)}."))

        if f("at_symbol_count") > 0:
            triggered.append(cls._rule("RULE_03_AT_SYMBOL"))
        if f("embedded_url") > 0 or f("double_slash_redirect") > 0:
            triggered.append(cls._rule("RULE_04_EMBEDDED_URL"))
        if f("has_punycode") > 0:
            triggered.append(cls._rule("RULE_05_PUNYCODE"))
        if f("subdomain_count") >= 4:
            triggered.append(cls._rule("RULE_06_EXCESSIVE_SUBDOMAINS"))

        kw = f("suspicious_keyword_count")
        if not official and (kw >= 2 or (kw >= 1 and f("has_https") == 0)):
            words = breakdown.get("matched_keywords") or []
            triggered.append(cls._rule("RULE_07_CREDENTIAL_KEYWORDS", f"Words found: {', '.join(words)}." if words else ""))

        if f("suspicious_tld") > 0:
            triggered.append(cls._rule("RULE_08_SUSPICIOUS_TLD"))
        if f("url_length") > 80 and f("url_entropy") > 4.2:
            triggered.append(cls._rule("RULE_09_HIGH_ENTROPY_AND_LENGTH"))
        if f("has_port") > 0:
            triggered.append(cls._rule("RULE_10_NON_STANDARD_PORT"))
        if f("is_shortened") > 0:
            triggered.append(cls._rule("RULE_11_URL_SHORTENER"))
        if f("free_hosting") > 0 and not official:
            triggered.append(cls._rule("RULE_12_FREE_HOSTING"))
        if f("registered_hyphen_count") >= 3 and not f("has_ip_address"):
            triggered.append(cls._rule("RULE_13_HYPHEN_STUFFED_DOMAIN"))
        if not f("has_ip_address") and not official and f("registered_label_length") >= 8 and (
                f("domain_digit_ratio") >= 0.35 or f("max_consonant_run") >= 7):
            triggered.append(cls._rule("RULE_14_MACHINE_GENERATED_DOMAIN"))
        if evidence.get("path") and not official and not host_brands and (
                f("free_hosting") > 0 or f("suspicious_tld") > 0 or kw >= 1):
            triggered.append(cls._rule("RULE_15_BRAND_IN_PATH", f"Brand(s): {', '.join(b.title() for b in evidence['path'])}."))
        if f("risky_file_extension") > 0 and kw >= 1 and not official:
            triggered.append(cls._rule("RULE_16_SERVER_SCRIPT_LOGIN"))

        score = int(min(sum(r["points"] for r in triggered), 100))
        if score >= RULE_HIGH_THRESHOLD:
            verdict, confidence = "High risk", min(0.65 + score / 250, 0.98)
        elif score >= RULE_FLAG_THRESHOLD:
            verdict, confidence = "Suspicious", 0.55 + (score - RULE_FLAG_THRESHOLD) / 200
        else:
            verdict, confidence = "Likely safe", max(0.85 - score / 100, 0.55)

        return {
            "model_name": "Transparent Rule-Based Baseline",
            "risk_score": score,
            "prediction": int(score >= RULE_FLAG_THRESHOLD),
            "verdict": verdict,
            "confidence": round(float(confidence), 3),
            "triggered_rules_count": len(triggered),
            "triggered_rules": triggered,
            "total_rules_evaluated": len(cls.RULES),
        }

    @classmethod
    def _get_rule(cls, rule_id: str) -> Dict[str, Any]:
        return cls._rule(rule_id) if rule_id in cls._BY_ID else {
            "id": rule_id, "name": rule_id, "points": 10, "severity": "LOW", "description": "", "rationale": ""}
