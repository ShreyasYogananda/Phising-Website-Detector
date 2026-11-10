"""
URL Feature Extraction Engine
PhishGuard AI URL Feature Extraction & Shannon Entropy Engine.

Extracts lexical, structural, statistical, and semantic features from raw URLs.
Includes domain anatomy breakdown, brand impersonation detection, and entropy calculations.
"""

import ipaddress
import math
import re
from urllib.parse import urlparse
import numpy as np

def is_ip_host(hostname: str) -> bool:
    """Returns True if hostname is a valid IPv4 or IPv6 address."""
    if not hostname:
        return False
    clean_host = hostname.split(":")[0].strip("[]").strip()
    try:
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        return False
KNOWN_BRANDS = {
    "paypal": ["paypal.com", "paypal.me"],
    "apple": ["apple.com", "icloud.com"],
    "microsoft": ["microsoft.com", "live.com", "office.com", "office365.com", "outlook.com", "azure.com"],
    "google": ["google.com", "gmail.com", "google.co.in", "google.co.uk", "youtube.com"],
    "netflix": ["netflix.com"],
    "amazon": ["amazon.com", "amazon.co.uk", "amazon.de", "aws.amazon.com"],
    "chase": ["chase.com"],
    "bankofamerica": ["bankofamerica.com", "bofa.com"],
    "wellsfargo": ["wellsfargo.com"],
    "facebook": ["facebook.com", "fb.com", "meta.com"],
    "instagram": ["instagram.com"],
    "whatsapp": ["whatsapp.com"],
    "steam": ["steampowered.com", "steamcommunity.com"],
    "coinbase": ["coinbase.com"],
    "binance": ["binance.com"],
    "metamask": ["metamask.io"],
    "dhl": ["dhl.com"],
    "usps": ["usps.com"],
    "fedex": ["fedex.com"],
    "dropbox": ["dropbox.com"],
    "spotify": ["spotify.com"]
}

# Suspicious keywords commonly seen in phishing attacks
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "security", "account",
    "banking", "update", "confirm", "password", "credential", "wallet",
    "recover", "webscr", "ebayisapi", "auth", "support", "alert",
    "billing", "reactivation", "resolution", "suspended", "unlock"
]

# High-abuse or frequently weaponized Top-Level Domains (TLDs)
SUSPICIOUS_TLDS = {
    "xyz", "top", "icu", "buzz", "club", "tk", "ml", "ga", "cf", "gq",
    "work", "fit", "sbs", "online", "site", "ru", "cc", "monster", "rest", "surf"
}

# Known URL shortener domains
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "buff.ly",
    "cutt.ly", "goo.gl", "bit.do", "shorte.st", "adf.ly", "rb.gy"
}

IPV4_PATTERN = re.compile(
    r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]?)$'
)

class URLExtractor:
    """
    Extracts numerical feature vectors and structured explanations from submitted URLs.
    """

    FEATURE_KEYS = [
        "url_length",
        "domain_length",
        "path_length",
        "dot_count",
        "hyphen_count",
        "at_symbol_count",
        "double_slash_redirect",
        "subdomain_count",
        "has_ip_address",
        "url_entropy",
        "domain_entropy",
        "suspicious_char_count",
        "digit_count",
        "digit_ratio",
        "has_https",
        "has_port",
        "suspicious_tld",
        "has_punycode",
        "is_shortened",
        "suspicious_keyword_count",
        "brand_impersonation_score"
    ]

    @staticmethod
    def calculate_shannon_entropy(text: str) -> float:
        """Calculates Shannon entropy to detect algorithmic or obfuscated character sequences."""
        if not text:
            return 0.0
        entropy = 0.0
        length = len(text)
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        return round(entropy, 4)

    @staticmethod
    def parse_domain_components(hostname: str):
        """
        Splits hostname into subdomains, registered domain, and tld.
        Handles IPv4 addresses gracefully.
        """
        if not hostname:
            return "", "", ""

        # Remove port if present
        host = hostname.split(":")[0].lower()

        # Check if direct IP
        if is_ip_host(host):
            return "", host, ""

        parts = host.split(".")
        if len(parts) <= 1:
            return "", host, ""
        elif len(parts) == 2:
            return "", host, parts[-1]
        else:
            # Handle multi-part TLDs (e.g., .co.uk, .gov.in, .ac.uk)
            if parts[-2] in ["co", "gov", "ac", "edu", "org", "net", "com"] and len(parts) >= 3:
                registered_domain = ".".join(parts[-3:])
                subdomains = ".".join(parts[:-3])
                tld = ".".join(parts[-2:])
            else:
                registered_domain = ".".join(parts[-2:])
                subdomains = ".".join(parts[:-2])
                tld = parts[-1]
            return subdomains, registered_domain, tld

    @classmethod
    def check_brand_impersonation(cls, url: str, hostname: str, subdomains: str, path: str):
        """
        Detects brand impersonation attacks:
        Triggers when a target brand keyword appears in subdomains, path, or query
        while the base registered domain is NOT the brand's verified domain.
        """
        host_clean = hostname.split(":")[0].lower()
        sub_lower = subdomains.lower()
        path_lower = path.lower()
        full_lower = url.lower()

        impersonated_brands = []
        impersonation_score = 0

        for brand, legitimate_domains in KNOWN_BRANDS.items():
            # Does brand appear anywhere in host or path?
            if brand in host_clean or brand in path_lower:
                # Is the current host actually an authorized legitimate domain for this brand?
                is_authorized = any(
                    host_clean == legit or host_clean.endswith("." + legit)
                    for legit in legitimate_domains
                )
                if not is_authorized:
                    impersonated_brands.append(brand)
                    # Severe penalty if brand appears in host/subdomain
                    if brand in host_clean:
                        impersonation_score += 2
                    else:
                        impersonation_score += 1

        return min(impersonation_score, 3), impersonated_brands

    @classmethod
    def extract_features(cls, url_str: str) -> dict:
        """
        Extracts all 21 structured numerical features from a URL.
        """
        if not isinstance(url_str, str) or not url_str.strip():
            url_str = ""

        raw_url = url_str.strip()
        
        # Ensure scheme for parsing
        has_original_scheme = bool(re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', raw_url))
        if not has_original_scheme:
            parsed_url = "https://" + raw_url
        else:
            parsed_url = raw_url

        try:
            parsed = urlparse(parsed_url)
            scheme = parsed.scheme.lower() if parsed.scheme else "http"
            netloc = parsed.netloc.lower() if parsed.netloc else ""
            path = parsed.path if parsed.path else ""
            query = parsed.query if parsed.query else ""
        except Exception:
            scheme, netloc, path, query = "http", "", "", ""

        hostname = netloc.split(":")[0] if ":" in netloc else netloc
        subdomains, reg_domain, tld = cls.parse_domain_components(hostname)

        # 1. Lexical and Length Features
        url_len = len(raw_url)
        domain_len = len(hostname)
        path_len = len(path)
        dot_cnt = raw_url.count(".")
        hyphen_cnt = raw_url.count("-")
        at_cnt = raw_url.count("@")
        
        # Double slash after scheme
        after_scheme = raw_url.split("://", 1)[1] if "://" in raw_url else raw_url
        double_slash_cnt = 1 if "//" in after_scheme else 0

        # Subdomains count
        subdomain_cnt = len(subdomains.split(".")) if subdomains else 0

        # IP Address check
        has_ip = 1 if is_ip_host(hostname) else 0

        # Shannon Entropy
        url_ent = cls.calculate_shannon_entropy(raw_url)
        dom_ent = cls.calculate_shannon_entropy(hostname)

        # Suspicious Characters (_ ~ % & = ? ; + $)
        suspicious_chars = set("_~%&=?;+$!")
        suspicious_char_cnt = sum(1 for c in raw_url if c in suspicious_chars)

        # Digits
        num_digits = sum(1 for c in raw_url if c.isdigit())
        digit_ratio = round(num_digits / max(url_len, 1), 4)

        # Protocol & Port
        has_https_flag = 1 if scheme == "https" else 0
        has_port_flag = 1 if (":" in netloc and not is_ip_host(hostname)) or (":" in netloc and len(netloc.split(":")) > 1 and netloc.split(":")[1] not in ["80", "443"]) else 0

        # Suspicious TLD
        clean_tld = tld.lower().lstrip(".")
        suspicious_tld_flag = 1 if clean_tld in SUSPICIOUS_TLDS else 0

        # Punycode / IDN Homograph
        has_punycode_flag = 1 if "xn--" in hostname else 0

        # Shortener
        is_short = 1 if hostname in SHORTENER_DOMAINS else 0

        # Suspicious Keywords in path, query, or subdomains
        target_text = f"{subdomains} {path} {query}".lower()
        keyword_hits = [kw for kw in SUSPICIOUS_KEYWORDS if kw in target_text]
        keyword_cnt = len(keyword_hits)

        # Brand impersonation
        brand_score, impersonated_brands = cls.check_brand_impersonation(raw_url, hostname, subdomains, path)

        features = {
            "url_length": float(url_len),
            "domain_length": float(domain_len),
            "path_length": float(path_len),
            "dot_count": float(dot_cnt),
            "hyphen_count": float(hyphen_cnt),
            "at_symbol_count": float(at_cnt),
            "double_slash_redirect": float(double_slash_cnt),
            "subdomain_count": float(subdomain_cnt),
            "has_ip_address": float(has_ip),
            "url_entropy": float(url_ent),
            "domain_entropy": float(dom_ent),
            "suspicious_char_count": float(suspicious_char_cnt),
            "digit_count": float(num_digits),
            "digit_ratio": float(digit_ratio),
            "has_https": float(has_https_flag),
            "has_port": float(has_port_flag),
            "suspicious_tld": float(suspicious_tld_flag),
            "has_punycode": float(has_punycode_flag),
            "is_shortened": float(is_short),
            "suspicious_keyword_count": float(keyword_cnt),
            "brand_impersonation_score": float(brand_score)
        }

        return features

    @classmethod
    def extract_feature_vector(cls, url_str: str) -> np.ndarray:
        """
        Returns numeric 1D numpy array ordered exactly according to FEATURE_KEYS.
        """
        feats = cls.extract_features(url_str)
        return np.array([feats[k] for k in cls.FEATURE_KEYS], dtype=np.float32)

    @classmethod
    def extract_detailed_breakdown(cls, url_str: str) -> dict:
        """
        Returns user-facing explainable breakdown including anatomy,
        feature signals, triggered keywords, and impersonation detection.
        """
        raw_url = url_str.strip() if isinstance(url_str, str) else ""
        has_original_scheme = bool(re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', raw_url))
        parsed_url = raw_url if has_original_scheme else "https://" + raw_url

        try:
            parsed = urlparse(parsed_url)
            scheme = parsed.scheme.lower() if parsed.scheme else "http"
            netloc = parsed.netloc.lower() if parsed.netloc else ""
            path = parsed.path if parsed.path else "/"
            query = parsed.query if parsed.query else ""
        except Exception:
            scheme, netloc, path, query = "http", "", "/", ""

        hostname = netloc.split(":")[0] if ":" in netloc else netloc
        subdomains, reg_domain, tld = cls.parse_domain_components(hostname)
        
        target_text = f"{subdomains} {path} {query}".lower()
        matched_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in target_text]
        brand_score, impersonated_brands = cls.check_brand_impersonation(raw_url, hostname, subdomains, path)
        features = cls.extract_features(raw_url)

        return {
            "anatomy": {
                "raw_url": raw_url,
                "scheme": scheme,
                "hostname": hostname,
                "subdomains": subdomains if subdomains else "(none)",
                "registered_domain": reg_domain,
                "tld": tld if tld else "(none)",
                "path": path,
                "query": query if query else "(none)"
            },
            "features": features,
            "matched_keywords": matched_keywords,
            "impersonated_brands": impersonated_brands,
            "has_ip_address": bool(features["has_ip_address"]),
            "is_shortened": bool(features["is_shortened"]),
            "has_punycode": bool(features["has_punycode"]),
            "suspicious_tld": bool(features["suspicious_tld"])
        }

def extract_features(url_str: str) -> dict:
    return URLExtractor.extract_features(url_str)

def extract_feature_vector(url_str: str) -> np.ndarray:
    return URLExtractor.extract_feature_vector(url_str)

def get_feature_names() -> list:
    return list(URLExtractor.FEATURE_KEYS)
