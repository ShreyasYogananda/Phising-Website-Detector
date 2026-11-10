"""
PhishGuard AI - Explainable Phishing Risk Checker
Web Application & REST API
"""

import os
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from features.extractor import URLExtractor, get_feature_names
from models.explainer import RiskExplainer
from evaluation.evaluator import load_evaluation_results, run_evaluation
from analyzer.page_analyzer import inspect_live_page_isolated

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="Explainable Phishing Risk Checker",
    description="URL Phishing Risk Detection comparing transparent rule baseline with Random Forest ML model",
    version="1.0.0"
)

# Static and Templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

class AnalyzeRequest(BaseModel):
    url: str
    enable_page_analysis: Optional[bool] = False

# Quick test demo samples
SAMPLE_URLS = [
    {
        "category": "Legitimate Portals",
        "title": "Google Search Query",
        "url": "https://www.google.com/search?q=cybersecurity+phishing+research",
        "expected": "Likely safe",
        "notes": "Authoritative search domain with standard query parameters"
    },
    {
        "category": "Legitimate Portals",
        "title": "Mozilla Developer Docs",
        "url": "https://developer.mozilla.org/en-US/docs/Web/Security/Practical_implementation_guides/TLS_cheat_sheet",
        "expected": "Likely safe",
        "notes": "Trusted technical documentation with deep paths and hyphens"
    },
    {
        "category": "Legitimate Portals",
        "title": "Chase Bank Personal Banking",
        "url": "https://www.chase.com/personal/banking",
        "expected": "Likely safe",
        "notes": "Official genuine banking domain (passes brand check)"
    },
    {
        "category": "Active Phishing Patterns",
        "title": "PayPal Subdomain Deception",
        "url": "http://paypal.com.verify-billing-center.xyz/index.html",
        "expected": "High risk",
        "notes": "Misleading brand in subdomain on untrusted .xyz registrar"
    },
    {
        "category": "Active Phishing Patterns",
        "title": "Raw IP Address Credential Harvest",
        "url": "http://185.220.101.5/login/webscr.php?cmd=_login_submit",
        "expected": "High risk",
        "notes": "Direct IP host without registered domain name"
    },
    {
        "category": "Active Phishing Patterns",
        "title": "Apple Security Alert Squat",
        "url": "https://apple-account-security-alert-resolution.top/signin/auth",
        "expected": "High risk",
        "notes": "Hyphenated brand impersonation on suspicious .top TLD"
    },
    {
        "category": "Obfuscation & Evasion",
        "title": "Embedded '@' Symbol Trick",
        "url": "http://google.com@phishing-target-account-recovery.xyz/login",
        "expected": "High risk",
        "notes": "Uses '@' delimiter to mislead user with recognized brand"
    },
    {
        "category": "Obfuscation & Evasion",
        "title": "Punycode Homograph Spoof",
        "url": "http://xn--paypl-ora.com/webscr/login_submit.php",
        "expected": "High risk",
        "notes": "Internationalized Cyrillic character spoof of 'paypal'"
    },
    {
        "category": "Edge Cases",
        "title": "URL Shortener Redirection",
        "url": "https://bit.ly/3xPh1shT4nk",
        "expected": "Suspicious",
        "notes": "Shortened link cloaking actual landing destination"
    }
]

# Feature dictionary definitions
FEATURE_DEFINITIONS = [
    {"name": "url_length", "category": "Lexical", "description": "Total character length of the URL. Phishing URLs frequently use lengthy strings to pack credentials, redirection tokens, or push malicious domains out of view."},
    {"name": "domain_length", "category": "Lexical", "description": "Character count of the hostname. Excessively long hostnames are common in disposable cybercrime infrastructure."},
    {"name": "path_length", "category": "Lexical", "description": "Length of the directory and file path following the hostname."},
    {"name": "has_ip_address", "category": "Structural", "description": "Binary indicator for whether the host is a raw IPv4/IPv6 address instead of a domain name."},
    {"name": "subdomain_count", "category": "Structural", "description": "Count of subdomain levels preceding the registered domain. Used to visually mimic legitimate brands (e.g. login.secure.target.com.attacker.com)."},
    {"name": "dot_count", "category": "Structural", "description": "Total dots '.' across URL string. Phishing URLs typically exhibit elevated dot density."},
    {"name": "hyphen_count", "category": "Structural", "description": "Hyphen '-' count. Attackers heavily utilize hyphens to combine brand names with security terms (e.g. chase-online-banking-security)."},
    {"name": "at_symbol_count", "category": "Obfuscation", "description": "Count of '@' symbols. RFC 3986 allows userinfo before '@', which browsers discard, navigating to the host following '@'."},
    {"name": "double_slash_redirect", "category": "Obfuscation", "description": "Presence of '//' occurring after the protocol scheme, often used in redirect confusion."},
    {"name": "suspicious_char_count", "category": "Obfuscation", "description": "Count of delimiter and encoding characters such as _, ~, %, &, =, ?, ;, +, $."},
    {"name": "url_entropy", "category": "Statistical", "description": "Shannon information entropy of the full URL string, measuring randomness and algorithmic token generation."},
    {"name": "domain_entropy", "category": "Statistical", "description": "Shannon entropy of the domain name characters."},
    {"name": "has_https", "category": "Protocol", "description": "Indicator of TLS encryption (1 for HTTPS, 0 for HTTP). Lack of HTTPS on auth pages is a risk factor."},
    {"name": "has_port", "category": "Protocol", "description": "Indicates non-standard port numbers (e.g. :8080, :8888) outside standard web ports 80/443."},
    {"name": "suspicious_keyword_count", "category": "Semantic", "description": "Count of high-risk credential harvesting keywords (e.g. login, verify, banking, update, confirm, wallet)."},
    {"name": "brand_impersonation_score", "category": "Semantic", "description": "Measures whether recognized corporate brands appear in subdomains or paths while registered domain is untrusted."},
    {"name": "suspicious_tld", "category": "Reputation", "description": "Flag for Top-Level Domains (.xyz, .top, .icu, .buzz, .work, etc.) with statistically high abuse rates."},
    {"name": "has_punycode", "category": "Obfuscation", "description": "Flag indicating Punycode 'xn--' homoglyph character substitutions (e.g. Cyrillic lookalikes)."},
    {"name": "is_shortened", "category": "Obfuscation", "description": "Flag for known URL shortening services (bit.ly, tinyurl, t.co) that conceal destination URLs."},
    {"name": "digit_count", "category": "Statistical", "description": "Total count of numerical digits in the URL."},
    {"name": "digit_ratio", "category": "Statistical", "description": "Ratio of numeric characters to total URL length."}
]

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Renders the main application dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "sample_urls": SAMPLE_URLS,
            "features": FEATURE_DEFINITIONS
        }
    )

@app.post("/api/analyze")
async def analyze_url_endpoint(req: AnalyzeRequest):
    """
    Analyzes submitted URL:
    - Runs rule-based baseline
    - Runs ML detector
    - Optional isolated page analyzer
    - Returns explainable synthesized evaluation
    """
    url = req.url.strip() if req.url else ""
    if not url:
        return JSONResponse(
            status_code=400,
            content={"error": "URL parameter cannot be empty."}
        )

    # Optional live page inspection
    page_result = None
    if req.enable_page_analysis:
        page_result = inspect_live_page_isolated(url)

    result = RiskExplainer.analyze_url(url, page_analysis_result=page_result)
    return result

@app.get("/api/evaluation")
async def get_evaluation_metrics():
    """Returns actual held-out test evaluation metrics and confusion matrices."""
    return load_evaluation_results()

@app.post("/api/evaluation/recalculate")
async def recalculate_evaluation():
    """Recalculates metrics directly from test dataset."""
    return run_evaluation()

@app.get("/api/samples")
async def get_samples():
    """Returns curated demo samples."""
    return SAMPLE_URLS

@app.get("/api/features/dictionary")
async def get_feature_dictionary():
    """Returns full feature dictionary."""
    return FEATURE_DEFINITIONS
