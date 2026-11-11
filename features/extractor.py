"""
URL feature extraction for PhishGuard.

Turns a raw URL into (a) a numeric feature vector for the ML model and
(b) a human-readable breakdown (anatomy, matched keywords, impersonated brands)
that powers the explanations in the UI.

Design notes
------------
* Registered-domain parsing uses a compact public-suffix table that also treats
  multi-tenant hosting platforms (github.io, pages.dev, workers.dev ...) as
  suffixes, so "evil.pages.dev" is correctly seen as its own site rather than
  as a subdomain of pages.dev.
* Brand matching is token-based. "pineapple.com" and "purchase.com" no longer
  trigger Apple/Chase impersonation the way naive substring matching did.
* Features avoid data-collection artefacts: HTTPS is kept (the rules use it) but
  the dataset is built so the scheme carries no label signal.
"""

import ipaddress
import math
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import numpy as np

# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #

# Brand -> official registered domains. Used for impersonation detection and the
# trusted-domain guard.
KNOWN_BRANDS: Dict[str, List[str]] = {
    "paypal": ["paypal.com", "paypal.me"],
    "apple": ["apple.com", "icloud.com"],
    "microsoft": ["microsoft.com", "live.com", "office.com", "office365.com", "outlook.com",
                  "azure.com", "microsoftonline.com", "sharepoint.com", "onedrive.com", "bing.com"],
    "google": ["google.com", "gmail.com", "google.co.in", "google.co.uk", "youtube.com", "goo.gl", "gstatic.com"],
    "netflix": ["netflix.com"],
    "amazon": ["amazon.com", "amazon.in", "amazon.co.uk", "amazon.de", "amazonaws.com", "amazon.co.jp"],
    "chase": ["chase.com"],
    "bankofamerica": ["bankofamerica.com", "bofa.com"],
    "wellsfargo": ["wellsfargo.com"],
    "citibank": ["citi.com", "citibank.com"],
    "usbank": ["usbank.com"],
    "capitalone": ["capitalone.com"],
    "hsbc": ["hsbc.com", "hsbc.co.uk", "hsbc.co.in"],
    "facebook": ["facebook.com", "fb.com", "meta.com", "messenger.com"],
    "instagram": ["instagram.com"],
    "whatsapp": ["whatsapp.com", "whatsapp.net"],
    "telegram": ["telegram.org", "t.me"],
    "linkedin": ["linkedin.com"],
    "twitter": ["twitter.com", "x.com"],
    "tiktok": ["tiktok.com"],
    "discord": ["discord.com", "discord.gg"],
    "steam": ["steampowered.com", "steamcommunity.com"],
    "roblox": ["roblox.com"],
    "coinbase": ["coinbase.com"],
    "binance": ["binance.com"],
    "metamask": ["metamask.io"],
    "blockchain": ["blockchain.com"],
    "dhl": ["dhl.com", "dhl.de"],
    "usps": ["usps.com"],
    "fedex": ["fedex.com"],
    "ups": ["ups.com"],
    "dropbox": ["dropbox.com"],
    "docusign": ["docusign.com", "docusign.net"],
    "adobe": ["adobe.com"],
    "zoom": ["zoom.us"],
    "spotify": ["spotify.com"],
    "yahoo": ["yahoo.com", "yahoo.co.jp"],
    "ebay": ["ebay.com", "ebay.co.uk"],
    "walmart": ["walmart.com"],
    "att": ["att.com"],
    "verizon": ["verizon.com"],
    "irs": ["irs.gov"],
    # India-focused targets
    "sbi": ["sbi.co.in", "onlinesbi.sbi", "onlinesbi.com", "sbicard.com"],
    "hdfcbank": ["hdfcbank.com"],
    "icicibank": ["icicibank.com"],
    "axisbank": ["axisbank.com"],
    "kotak": ["kotak.com"],
    "paytm": ["paytm.com"],
    "phonepe": ["phonepe.com"],
    "flipkart": ["flipkart.com"],
    "irctc": ["irctc.co.in"],
    "uidai": ["uidai.gov.in"],
    "npci": ["npci.org.in"],
}

# Words attackers append to a brand name ("paypal-secure", "applesupport").
BRAND_SUFFIX_VOCAB = {
    "login", "signin", "secure", "security", "support", "verify", "verification", "account",
    "accounts", "id", "pay", "payment", "payments", "service", "services", "update", "help",
    "online", "billing", "wallet", "care", "customer", "web", "mail", "portal", "center",
    "centre", "alert", "auth", "recovery", "unlock", "official", "app", "bank", "banking",
    "refund", "claim", "reward", "rewards", "team", "team1", "store", "shop", "info",
}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "security", "secure", "account", "banking",
    "update", "confirm", "password", "credential", "wallet", "recover", "webscr", "ebayisapi",
    "auth", "support", "alert", "billing", "reactivation", "suspended", "unlock", "validate",
    "authenticate", "restore", "limited", "invoice", "refund",
]

# Matched on whole tokens so "authors" / "authority" do not match "auth".
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

SUSPICIOUS_TLDS = {
    "xyz", "top", "icu", "buzz", "club", "tk", "ml", "ga", "cf", "gq", "work", "fit", "sbs",
    "online", "site", "ru", "cc", "monster", "rest", "surf", "cyou", "cfd", "live", "click",
    "link", "support", "zip", "mov", "bond", "lol", "loan", "men", "stream", "vip", "shop",
}

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "buff.ly", "cutt.ly", "goo.gl", "bit.do",
    "shorte.st", "adf.ly", "rb.gy", "tiny.cc", "rebrand.ly", "shorturl.at", "t.ly", "lnkd.in",
    "s.id", "v.gd", "qr.ae", "l.ead.me",
}

# Multi-tenant platforms: anyone can publish a site under <name>.<platform>.
# Phishing campaigns abuse these constantly, but so do plenty of legitimate users,
# so this is a *moderate* signal, never a verdict on its own.
FREE_HOSTING_SUFFIXES = {
    "workers.dev", "pages.dev", "github.io", "gitlab.io", "web.app", "firebaseapp.com",
    "vercel.app", "netlify.app", "webflow.io", "weebly.com", "weeblysite.com", "wixsite.com",
    "glitch.me", "repl.co", "replit.app", "herokuapp.com", "000webhostapp.com", "blogspot.com",
    "wordpress.com", "r2.dev", "gitbook.io", "godaddysites.com", "square.site", "myshopify.com",
    "appspot.com", "azurewebsites.net", "cloudfront.net", "duckdns.org", "ngrok.io", "ngrok-free.app",
    "plesk.page", "fleek.co", "ewp.live", "framer.website", "carrd.co", "typedream.app",
    "webcindario.com", "sites.google.com", "forms.gle", "ipfs.io", "dweb.link", "nftstorage.link",
    "surge.sh", "onrender.com", "railway.app", "fly.dev", "zohosites.com", "hubspotpagebuilder.com",
    "b-cdn.net", "weebly.net", "jimdofree.com", "site123.me", "mystrikingly.com", "yolasite.com",
    "bubbleapps.io", "wixstudio.io", "framer.app", "my.canva.site", "canva.site", "notion.site",
    "storage.googleapis.com", "s3.amazonaws.com", "blob.core.windows.net",
    # Open-content hosts on otherwise trusted domains: attackers publish forms, files and sites here.
    "docs.google.com", "drive.google.com", "forms.office.com", "sway.office.com", "onedrive.live.com",
    "1drv.ms", "dropboxusercontent.com", "sharepoint.com", "amazonaws.com", "web.core.windows.net",
    "typeform.com", "jotform.com", "formstack.com", "wufoo.com",
}

# A few two-level public suffixes so "bank.co.uk" is parsed as bank + co.uk.
MULTI_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "com.br", "net.br", "org.br", "gov.br", "co.in", "net.in", "org.in", "ac.in", "gov.in",
    "res.in", "firm.in", "co.jp", "ne.jp", "or.jp", "ac.jp", "com.cn", "net.cn", "org.cn",
    "gov.cn", "co.za", "org.za", "com.mx", "com.ar", "com.co", "com.tr", "com.sg", "edu.sg",
    "com.my", "com.hk", "com.tw", "co.nz", "co.kr", "or.kr", "co.id", "my.id", "web.id",
    "or.id", "ac.id", "go.id", "sch.id", "biz.id", "com.ph", "com.vn", "com.ua", "com.pk",
    "com.bd", "com.ng", "com.eg", "co.th", "co.il", "com.sa", "com.pe", "com.ve", "com.ec",
    "com.pl", "com.ru", "com.es", "com.pt", "com.gr", "co.ke", "com.np", "com.lk",
}

SUSPICIOUS_FILE_EXTENSIONS = {
    "php", "asp", "aspx", "jsp", "cgi", "exe", "scr", "apk", "zip", "rar", "js", "jar", "bat", "msi",
}

_VOWELS = set("aeiou")
_HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s"})


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def is_ip_host(hostname: str) -> bool:
    """True if the hostname is an IPv4/IPv6 literal (also catches decimal / hex IPv4 forms)."""
    if not hostname:
        return False
    host = hostname.strip().strip("[]")
    if host.count(":") == 1 and "." in host:      # strip an attached port for IPv4
        host = host.split(":")[0]
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    # Obfuscated forms browsers still resolve: 3232235777, 0xC0A80101, 0300.0250.0.1
    if re.fullmatch(r"\d{8,10}", host) or re.fullmatch(r"0x[0-9a-f]{6,8}", host, re.I):
        return True
    if re.fullmatch(r"(0x[0-9a-f]+|\d+)(\.(0x[0-9a-f]+|\d+)){3}", host, re.I):
        return True
    return False


def levenshtein(a: str, b: str, limit: int = 3) -> int:
    """Edit distance with an early exit once it exceeds `limit`."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_SPLIT.split(text.lower()) if t]


def split_host(hostname: str) -> Tuple[str, str, str]:
    """Returns (subdomains, registered_domain, public_suffix)."""
    host = (hostname or "").split(":")[0].lower().strip(".")
    if not host:
        return "", "", ""
    if is_ip_host(host):
        return "", host, ""
    labels = host.split(".")
    if len(labels) == 1:
        return "", host, ""

    suffix_len = 1
    for n in (4, 3, 2):                      # longest known suffix wins
        if len(labels) > n - 1:
            cand = ".".join(labels[-n:])
            if cand in FREE_HOSTING_SUFFIXES or (n == 2 and cand in MULTI_PART_SUFFIXES):
                suffix_len = n
                break
    if len(labels) <= suffix_len:            # host *is* the suffix (e.g. "github.io")
        return "", host, ".".join(labels[-suffix_len:])
    suffix = ".".join(labels[-suffix_len:])
    registered = ".".join(labels[-(suffix_len + 1):])
    subdomains = ".".join(labels[:-(suffix_len + 1)])
    return subdomains, registered, suffix


def registered_domain(hostname: str) -> str:
    return split_host(hostname)[1]


def is_free_hosting(hostname: str) -> bool:
    host = (hostname or "").split(":")[0].lower().strip(".")
    return any(host == s or host.endswith("." + s) for s in FREE_HOSTING_SUFFIXES)


_OFFICIAL_INDEX = {d: b for b, ds in KNOWN_BRANDS.items() for d in ds}


def is_official_domain(hostname: str) -> Optional[str]:
    """Returns the brand name if `hostname` belongs to that brand's official domains."""
    host = (hostname or "").split(":")[0].lower().strip(".")
    if is_free_hosting(host):
        return None                           # user-content hosts are never "official"
    for dom, brand in _OFFICIAL_INDEX.items():
        if host == dom or host.endswith("." + dom):
            return brand
    return None


# --------------------------------------------------------------------------- #
# Brand masking for the character-pattern model
# --------------------------------------------------------------------------- #
_LONG_BRANDS = sorted((b for b in KNOWN_BRANDS if len(b) >= 5), key=len, reverse=True)
_SHORT_BRANDS = sorted((b for b in KNOWN_BRANDS if len(b) < 5), key=len, reverse=True)
_LONG_BRAND_RE = re.compile("|".join(map(re.escape, _LONG_BRANDS)))
_SHORT_BRAND_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(map(re.escape, _SHORT_BRANDS)) + r")(?![a-z0-9])")
BRAND_MASK = "brandx"


def mask_brands(url: str) -> str:
    """Lower-cases the URL and replaces brand names with a placeholder.

    The character-pattern model then learns the *shape* of impersonation (brandx.com.verify-...xyz) instead of
    memorising which brands appear in its training data. Without this it scored github.com/paypal/paypal-sdk
    as phishing simply because "paypal" was frequent in phishing URLs.
    """
    text = (url or "").lower()
    return _SHORT_BRAND_RE.sub(BRAND_MASK, _LONG_BRAND_RE.sub(BRAND_MASK, text))


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #

class URLExtractor:
    FEATURE_KEYS = [
        # size / shape
        "url_length", "domain_length", "path_length", "query_length", "path_depth", "query_param_count",
        # character counts
        "dot_count", "hyphen_count", "underscore_count", "at_symbol_count", "percent_encoded_count",
        "suspicious_char_count", "digit_count", "digit_ratio", "domain_digit_ratio",
        # host structure
        "subdomain_count", "has_ip_address", "has_port", "registered_label_length", "registered_hyphen_count",
        "tld_length", "suspicious_tld", "has_punycode", "is_shortened", "free_hosting",
        # randomness
        "url_entropy", "domain_entropy", "path_entropy", "longest_token_length", "max_consonant_run",
        # semantics
        "suspicious_keyword_count", "keyword_in_host", "brand_in_host", "brand_in_path",
        "brand_impersonation_score", "typosquat_brand", "https_token_in_host",
        # structure of the path / query
        "embedded_url", "double_slash_redirect", "risky_file_extension",
        # protocol
        "has_https",
    ]

    @staticmethod
    def calculate_shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        freq: Dict[str, int] = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        n = len(text)
        return round(-sum((c / n) * math.log2(c / n) for c in freq.values()), 4)

    @staticmethod
    def parse_domain_components(hostname: str) -> Tuple[str, str, str]:
        return split_host(hostname)

    # ---- normalisation --------------------------------------------------- #
    @staticmethod
    def _parse(url_str: str):
        raw = url_str.strip() if isinstance(url_str, str) else ""
        has_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", raw))
        parse_target = raw if has_scheme else "http://" + raw      # scheme-less input: don't assume https
        try:
            p = urlparse(parse_target)
            scheme = p.scheme.lower() if has_scheme else ""
            netloc = (p.netloc or "").lower()
            path, query = p.path or "", p.query or ""
        except ValueError:
            scheme, netloc, path, query = "", "", "", ""
        userinfo, _, hostport = netloc.rpartition("@")
        if hostport.startswith("["):
            host = hostport.split("]")[0].strip("[")
            port = hostport.split("]")[1].lstrip(":") if "]" in hostport else ""
        else:
            host, _, port = hostport.partition(":")
        return raw, scheme, userinfo, host.strip("."), port, path, query, has_scheme

    # ---- brand logic ----------------------------------------------------- #
    @classmethod
    def find_impersonated_brands(cls, hostname: str, path: str, query: str = "", userinfo: str = "") -> Dict[str, List[str]]:
        """Returns {'host': [...], 'path': [...], 'typosquat': [...]} for brands the URL pretends to be.

        `userinfo` is the part before '@' in the authority; browsers discard it, which is why
        "http://google.com@evil.xyz/" is a classic disguise, so its tokens count as host evidence.
        """
        host = hostname.lower()
        out = {"host": [], "path": [], "typosquat": []}
        if is_ip_host(host):
            host_tokens: List[str] = []
        else:
            host_tokens = tokenize(host)
        host_tokens = host_tokens + tokenize(userinfo)
        reg = registered_domain(host)
        reg_label = reg.split(".")[0] if reg else ""
        official_brand = is_official_domain(host)
        path_tokens = tokenize(unquote(path) + " " + unquote(query))

        for brand in KNOWN_BRANDS:
            if brand == official_brand:
                continue
            # --- brand in host: exact token, or token = brand + attacker vocabulary
            hit = False
            for tok in host_tokens:
                if tok == brand:
                    hit = True
                elif tok.startswith(brand) and tok[len(brand):] in BRAND_SUFFIX_VOCAB:
                    hit = True
                elif len(brand) >= 5 and tok.endswith(brand) and tok[: -len(brand)] in BRAND_SUFFIX_VOCAB:
                    hit = True
                if hit:
                    break
            # joined form inside hyphenated label, e.g. "secure-paypal-login" handled by tokenising on "-"
            if hit:
                out["host"].append(brand)
                continue
            # --- typosquat / homoglyph of the registered label (paypa1, paypaI, arnazon)
            if len(brand) >= 5 and reg_label and reg_label != brand and not is_free_hosting(host):
                norm = reg_label.translate(_HOMOGLYPHS).replace("rn", "m").replace("vv", "w")
                if norm == brand or levenshtein(reg_label, brand, 2) <= (1 if len(brand) <= 6 else 2):
                    out["typosquat"].append(brand)
                    continue
            # --- brand only in path/query (weaker)
            if brand in path_tokens:
                out["path"].append(brand)
        return out

    @classmethod
    def check_brand_impersonation(cls, url: str, hostname: str, subdomains: str, path: str):
        """Backwards-compatible wrapper: (score 0-3, brand list)."""
        found = cls.find_impersonated_brands(hostname, path)
        brands = list(dict.fromkeys(found["host"] + found["typosquat"] + found["path"]))
        score = 2 * bool(found["host"]) + 2 * bool(found["typosquat"]) + 1 * bool(found["path"])
        return min(score, 3), brands

    # ---- features -------------------------------------------------------- #
    @classmethod
    def extract_features(cls, url_str: str) -> Dict[str, float]:
        raw, scheme, userinfo, host, port, path, query, _ = cls._parse(url_str)
        subdomains, reg, suffix = split_host(host)
        reg_label = reg.split(".")[0] if (reg and not is_ip_host(host)) else ""

        url_len = len(raw)
        digits = sum(c.isdigit() for c in raw)
        host_digits = sum(c.isdigit() for c in host)
        after_scheme = raw.split("://", 1)[1] if "://" in raw else raw
        ip = is_ip_host(host)

        host_tokens = [] if ip else tokenize(host)
        official = is_official_domain(host)
        label_tokens = set() if (official or ip) else set(tokenize(reg_label))
        kw_text_tokens = set(tokenize(f"{subdomains} {unquote(path)} {unquote(query)}")) | label_tokens
        kw_hits = [k for k in SUSPICIOUS_KEYWORDS if k in kw_text_tokens]
        kw_host = [k for k in SUSPICIOUS_KEYWORDS if k in (label_tokens | set(tokenize(subdomains)))]

        found = cls.find_impersonated_brands(host, path, query, userinfo)
        brand_score = min(2 * bool(found["host"]) + 2 * bool(found["typosquat"]) + 1 * bool(found["path"]), 3)

        tokens_all = tokenize(raw.split("://", 1)[-1])
        longest_token = max((len(t) for t in tokens_all), default=0)
        cons_runs = re.findall(r"[bcdfghjklmnpqrstvwxyz]+", reg_label) if reg_label else []
        max_cons = max((len(r) for r in cons_runs), default=0)

        ext = ""
        last_seg = path.rsplit("/", 1)[-1]
        if "." in last_seg:
            ext = last_seg.rsplit(".", 1)[-1].lower()

        embedded = 1 if re.search(r"(https?%3a|https?:)(//|%2f%2f)", (path + "?" + query).lower()) else 0
        port_flag = 1 if port and port not in ("80", "443") else 0

        f = {
            "url_length": url_len,
            "domain_length": len(host),
            "path_length": len(path),
            "query_length": len(query),
            "path_depth": len([s for s in path.split("/") if s]),
            "query_param_count": len([p for p in query.split("&") if p]) if query else 0,
            "dot_count": raw.count("."),
            "hyphen_count": raw.count("-"),
            "underscore_count": raw.count("_"),
            "at_symbol_count": raw.count("@"),
            "percent_encoded_count": len(re.findall(r"%[0-9a-fA-F]{2}", raw)),
            "suspicious_char_count": sum(1 for c in raw if c in "_~%&=?;+$!"),
            "digit_count": digits,
            "digit_ratio": round(digits / max(url_len, 1), 4),
            "domain_digit_ratio": round(host_digits / max(len(host), 1), 4),
            "subdomain_count": len(subdomains.split(".")) if subdomains else 0,
            "has_ip_address": int(ip),
            "has_port": port_flag,
            "registered_label_length": len(reg_label),
            "registered_hyphen_count": reg_label.count("-"),
            "tld_length": len(suffix.split(".")[-1]) if suffix else 0,
            "suspicious_tld": int(suffix.split(".")[-1] in SUSPICIOUS_TLDS) if suffix else 0,
            "has_punycode": int("xn--" in host),
            "is_shortened": int(host.removeprefix("www.") in SHORTENER_DOMAINS),
            "free_hosting": int(is_free_hosting(host)),
            "url_entropy": cls.calculate_shannon_entropy(raw),
            "domain_entropy": cls.calculate_shannon_entropy(host),
            "path_entropy": cls.calculate_shannon_entropy(path),
            "longest_token_length": longest_token,
            "max_consonant_run": max_cons,
            "suspicious_keyword_count": len(kw_hits),
            "keyword_in_host": int(bool(kw_host)),
            "brand_in_host": len(found["host"]),
            "brand_in_path": len(found["path"]),
            "brand_impersonation_score": brand_score,
            "typosquat_brand": len(found["typosquat"]),
            "https_token_in_host": int("https" in host_tokens or "http" in host_tokens),
            "embedded_url": embedded,
            "double_slash_redirect": int("//" in after_scheme.split("?", 1)[0].split("/", 1)[-1]) if "/" in after_scheme else 0,
            "risky_file_extension": int(ext in SUSPICIOUS_FILE_EXTENSIONS),
            "has_https": int(scheme == "https"),
        }
        return {k: float(f[k]) for k in cls.FEATURE_KEYS}

    @classmethod
    def extract_feature_vector(cls, url_str: str) -> np.ndarray:
        feats = cls.extract_features(url_str)
        return np.array([feats[k] for k in cls.FEATURE_KEYS], dtype=np.float32)

    @classmethod
    def extract_detailed_breakdown(cls, url_str: str) -> dict:
        raw, scheme, userinfo, host, port, path, query, has_scheme = cls._parse(url_str)
        subdomains, reg, suffix = split_host(host)
        features = cls.extract_features(url_str)
        official = is_official_domain(host)
        label = reg.split(".")[0] if (reg and not is_ip_host(host)) else ""
        kw_tokens = set(tokenize(f"{subdomains} {unquote(path)} {unquote(query)}")) | (set() if official else set(tokenize(label)))
        matched = [k for k in SUSPICIOUS_KEYWORDS if k in kw_tokens]
        found = cls.find_impersonated_brands(host, path, query, userinfo)
        brands = list(dict.fromkeys(found["host"] + found["typosquat"] + found["path"]))

        return {
            "anatomy": {
                "raw_url": raw,
                "scheme": scheme or "(none given)",
                "userinfo": userinfo or "(none)",
                "hostname": host,
                "subdomains": subdomains or "(none)",
                "registered_domain": reg,
                "tld": suffix or "(none)",
                "port": port or "(default)",
                "path": path or "/",
                "query": query or "(none)",
            },
            "features": features,
            "matched_keywords": matched,
            "impersonated_brands": brands,
            "brand_evidence": found,
            "official_brand": official,
            "has_ip_address": bool(features["has_ip_address"]),
            "is_shortened": bool(features["is_shortened"]),
            "has_punycode": bool(features["has_punycode"]),
            "suspicious_tld": bool(features["suspicious_tld"]),
            "free_hosting": bool(features["free_hosting"]),
        }


def extract_features(url_str: str) -> dict:
    return URLExtractor.extract_features(url_str)


def extract_feature_vector(url_str: str) -> np.ndarray:
    return URLExtractor.extract_feature_vector(url_str)


def get_feature_names() -> list:
    return list(URLExtractor.FEATURE_KEYS)
