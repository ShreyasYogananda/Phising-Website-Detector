"""
PhishGuard AI - Explainable Phishing Risk Checker
Web application and REST API (FastAPI).

Endpoints
---------
GET  /                         dashboard
POST /api/analyze              analyse one URL (optionally inspect the live page in an isolated sandbox)
POST /api/batch                analyse up to 25 URLs, summary only
GET  /api/evaluation           stored benchmark (held-out domains + older-campaign test)
POST /api/evaluation/recalculate
GET  /api/samples              demo URLs
GET  /api/features/dictionary  the 41 engineered features
GET  /api/health               liveness + model info
"""

import logging
import os
import re
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from analyzer.page_analyzer import inspect_live_page_isolated
from evaluation.evaluator import load_evaluation_results, run_evaluation
from features.extractor import get_feature_names
from models.explainer import RiskExplainer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("phishguard")

APP_VERSION = "2.0.0"
MAX_URL_LENGTH = 2048
MAX_BATCH = 25

app = FastAPI(
    title="PhishGuard AI",
    description="Explainable phishing risk checker: transparent rules + gradient boosting / character-pattern ML ensemble.",
    version=APP_VERSION,
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# --------------------------------------------------------------------------- #
# Request models and input hygiene
# --------------------------------------------------------------------------- #
class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="URL or bare domain to check")
    enable_page_analysis: Optional[bool] = False


class BatchRequest(BaseModel):
    urls: List[str] = Field(..., description=f"Up to {MAX_BATCH} URLs")


_CONTROL = re.compile(r"[\x00-\x1f\x7f\s]+")


def clean_url(raw: str) -> str:
    """Trim, drop control characters and embedded whitespace (pasted URLs often carry line breaks)."""
    return _CONTROL.sub("", raw or "")


def validation_error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.exception_handler(RequestValidationError)
async def _on_validation_error(_: Request, exc: RequestValidationError):
    return validation_error("Invalid request. Send JSON like {\"url\": \"https://example.com\"}.", 422)


# --------------------------------------------------------------------------- #
# Demo samples
# --------------------------------------------------------------------------- #
SAMPLE_URLS = [
    {"category": "Legitimate", "title": "Google search",
     "url": "https://www.google.com/search?q=cybersecurity+phishing+research",
     "expected": "Likely safe", "notes": "Well-known domain with an ordinary search query."},
    {"category": "Legitimate", "title": "Mozilla docs",
     "url": "https://developer.mozilla.org/en-US/docs/Web/Security/Practical_implementation_guides/TLS_cheat_sheet",
     "expected": "Likely safe", "notes": "Deep path with hyphens and underscores, but a trusted documentation site."},
    {"category": "Legitimate", "title": "Real bank, real domain",
     "url": "https://www.chase.com/personal/banking",
     "expected": "Likely safe", "notes": "Contains a brand name, but it is on that brand's own domain."},
    {"category": "Legitimate", "title": "Real sign-in page",
     "url": "https://accounts.google.com/signin",
     "expected": "Likely safe", "notes": "'signin' is a risky word only on domains that do not belong to the brand."},
    {"category": "Phishing pattern", "title": "Brand hidden in subdomain",
     "url": "http://paypal.com.verify-billing-center.xyz/index.html",
     "expected": "High risk", "notes": "Looks like paypal.com at a glance; the real domain is verify-billing-center.xyz."},
    {"category": "Phishing pattern", "title": "Raw IP address",
     "url": "http://185.220.101.5/login/webscr.php?cmd=_login_submit",
     "expected": "High risk", "notes": "A sign-in page on a bare IP address, with no registered domain."},
    {"category": "Phishing pattern", "title": "Brand + urgency on cheap TLD",
     "url": "https://apple-account-security-alert-resolution.top/signin/auth",
     "expected": "High risk", "notes": "Hyphen-stuffed brand and security words on a heavily abused .top domain."},
    {"category": "Phishing pattern", "title": "Indian bank KYC lure",
     "url": "https://sbi-netbanking-kyc-update.site/login",
     "expected": "High risk", "notes": "Impersonates SBI with a KYC-update pretext."},
    {"category": "Phishing pattern", "title": "Look-alike spelling",
     "url": "http://amaz0n-secure-login.top/ap/signin",
     "expected": "High risk", "notes": "Zero instead of 'o' to imitate Amazon."},
    {"category": "Evasion", "title": "'@' disguise",
     "url": "http://google.com@phishing-target-account-recovery.xyz/login",
     "expected": "High risk", "notes": "Everything before '@' is ignored by the browser; the real host comes after it."},
    {"category": "Evasion", "title": "Punycode look-alike",
     "url": "http://xn--paypl-ora.com/webscr/login_submit.php",
     "expected": "High risk", "notes": "Internationalised domain that renders like 'paypal'."},
    {"category": "Evasion", "title": "Shortened link",
     "url": "https://bit.ly/3xPh1shT4nk",
     "expected": "High risk", "notes": "A shortener hides the real destination; treat unexpected ones with care."},
]


# --------------------------------------------------------------------------- #
# Feature dictionary (all 41 features the model sees)
# --------------------------------------------------------------------------- #
FEATURE_DEFINITIONS = [
    # Size and shape
    {"name": "url_length", "category": "Size & shape", "description": "Total characters in the address. Very long URLs can bury the real domain or carry tracking and redirect payloads."},
    {"name": "domain_length", "category": "Size & shape", "description": "Characters in the hostname. Unusually long hostnames are common on throwaway infrastructure."},
    {"name": "path_length", "category": "Size & shape", "description": "Characters in the path after the hostname."},
    {"name": "query_length", "category": "Size & shape", "description": "Characters in the query string after '?'."},
    {"name": "path_depth", "category": "Size & shape", "description": "Number of '/'-separated path segments."},
    {"name": "query_param_count", "category": "Size & shape", "description": "Number of key=value parameters in the query string."},
    # Characters
    {"name": "dot_count", "category": "Characters", "description": "Dots in the whole URL. Many dots usually mean stacked subdomains."},
    {"name": "hyphen_count", "category": "Characters", "description": "Hyphens in the URL. Attackers join brand and security words, e.g. secure-bank-login."},
    {"name": "underscore_count", "category": "Characters", "description": "Underscores in the URL."},
    {"name": "at_symbol_count", "category": "Characters", "description": "'@' symbols. Browsers ignore everything before '@', so it can disguise the real host."},
    {"name": "percent_encoded_count", "category": "Characters", "description": "%XX encodings, sometimes used to hide characters or keywords."},
    {"name": "suspicious_char_count", "category": "Characters", "description": "Count of _ ~ % & = ? ; + $ ! characters."},
    {"name": "digit_count", "category": "Characters", "description": "Digits in the URL."},
    {"name": "digit_ratio", "category": "Characters", "description": "Share of the URL that is digits."},
    {"name": "domain_digit_ratio", "category": "Characters", "description": "Share of the hostname that is digits. High values suggest generated or IP-like hosts."},
    # Host structure
    {"name": "subdomain_count", "category": "Host structure", "description": "Subdomain levels before the registered domain."},
    {"name": "has_ip_address", "category": "Host structure", "description": "Host is a raw IPv4/IPv6 address instead of a domain name."},
    {"name": "has_port", "category": "Host structure", "description": "A non-standard port (not 80 or 443) is specified."},
    {"name": "registered_label_length", "category": "Host structure", "description": "Length of the registered domain name without its extension."},
    {"name": "registered_hyphen_count", "category": "Host structure", "description": "Hyphens inside the registered domain name itself."},
    {"name": "tld_length", "category": "Host structure", "description": "Length of the top-level extension."},
    {"name": "suspicious_tld", "category": "Host structure", "description": "Extension is on a list of cheap, heavily abused TLDs (.xyz, .top, .icu ...)."},
    {"name": "has_punycode", "category": "Host structure", "description": "Host uses 'xn--' encoding, which can render as look-alike characters."},
    {"name": "is_shortened", "category": "Host structure", "description": "Host is a known URL shortener that hides the final destination."},
    {"name": "free_hosting", "category": "Host structure", "description": "Site sits on a free or multi-tenant platform (github.io, pages.dev, ...) that anyone can use."},
    # Randomness
    {"name": "url_entropy", "category": "Randomness", "description": "Shannon entropy of the whole URL. Higher means more random-looking."},
    {"name": "domain_entropy", "category": "Randomness", "description": "Shannon entropy of the hostname."},
    {"name": "path_entropy", "category": "Randomness", "description": "Shannon entropy of the path."},
    {"name": "longest_token_length", "category": "Randomness", "description": "Longest run of letters and digits. Long unbroken tokens are often generated."},
    {"name": "max_consonant_run", "category": "Randomness", "description": "Longest run of consonants in the domain name, a cue for machine-generated names."},
    # Meaning
    {"name": "suspicious_keyword_count", "category": "Meaning", "description": "Credential or payment words (login, verify, update, wallet ...) outside an official brand domain."},
    {"name": "keyword_in_host", "category": "Meaning", "description": "Such a word appears in the domain or subdomains rather than only in the path."},
    {"name": "brand_in_host", "category": "Meaning", "description": "A well-known brand appears as a word in the hostname of a site that is not the brand's own."},
    {"name": "brand_in_path", "category": "Meaning", "description": "A brand name appears only in the path or query (weaker evidence)."},
    {"name": "brand_impersonation_score", "category": "Meaning", "description": "Combined 0-3 brand-spoofing score from host, look-alike and path evidence."},
    {"name": "typosquat_brand", "category": "Meaning", "description": "The registered domain is one or two characters away from a known brand (paypa1, arnazon)."},
    {"name": "https_token_in_host", "category": "Meaning", "description": "The word 'https' sits inside the hostname to fake a secure padlock."},
    # Path / query tricks
    {"name": "embedded_url", "category": "Path & query tricks", "description": "Another http(s):// URL is embedded in the path or query, as in open-redirect abuse."},
    {"name": "double_slash_redirect", "category": "Path & query tricks", "description": "'//' appears after the scheme, which can confuse redirects."},
    {"name": "risky_file_extension", "category": "Path & query tricks", "description": "Path ends in an executable or script-like extension (.exe, .apk, .php ...)."},
    # Protocol
    {"name": "has_https", "category": "Protocol", "description": "URL uses HTTPS. Note: phishing sites use HTTPS too, so this is a weak signal either way."},
]

# Fail loudly in development if the dictionary and the extractor drift apart again.
_missing = set(get_feature_names()) - {f["name"] for f in FEATURE_DEFINITIONS}
if _missing:  # pragma: no cover
    log.warning("Feature dictionary is missing: %s", sorted(_missing))


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"sample_urls": SAMPLE_URLS, "features": FEATURE_DEFINITIONS, "version": APP_VERSION},
    )


@app.post("/api/analyze")
async def analyze_url_endpoint(req: AnalyzeRequest):
    url = clean_url(req.url)
    if not url:
        return validation_error("Please enter a URL.")
    if len(url) > MAX_URL_LENGTH:
        return validation_error(f"That URL is too long (limit {MAX_URL_LENGTH} characters).")

    page_result = None
    if req.enable_page_analysis:
        try:
            page_result = inspect_live_page_isolated(url)
        except Exception:  # the optional sandbox must never break the main verdict
            log.exception("page analysis failed")
            page_result = {"status": "error", "message": "Page inspection failed.", "findings": []}

    try:
        return RiskExplainer.analyze_url(url, page_analysis_result=page_result)
    except Exception:
        log.exception("analysis failed for %r", url[:200])
        return validation_error("Could not analyse that URL.", 500)


@app.post("/api/batch")
async def analyze_batch(req: BatchRequest):
    urls = [clean_url(u) for u in req.urls if clean_url(u)]
    if not urls:
        return validation_error("Send at least one URL.")
    if len(urls) > MAX_BATCH:
        return validation_error(f"Batch limit is {MAX_BATCH} URLs per request.")

    results = []
    for u in urls:
        if len(u) > MAX_URL_LENGTH:
            results.append({"url": u[:80] + "…", "error": "URL too long"})
            continue
        try:
            r = RiskExplainer.analyze_url(u)
            results.append({
                "url": u,
                "verdict": r["unified_verdict"],
                "risk_score": r["hybrid_risk_score"],
                "top_signals": [s["title"] for s in r["signals_summary"][:3]],
            })
        except Exception:
            log.exception("batch analysis failed for %r", u[:200])
            results.append({"url": u, "error": "Could not analyse"})
    return {"count": len(results), "results": results}


@app.get("/api/evaluation")
async def get_evaluation_metrics():
    return load_evaluation_results()


@app.post("/api/evaluation/recalculate")
async def recalculate_evaluation():
    return run_evaluation()


@app.get("/api/samples")
async def get_samples():
    return SAMPLE_URLS


@app.get("/api/features/dictionary")
async def get_feature_dictionary():
    return FEATURE_DEFINITIONS


@app.get("/api/health")
async def health():
    ev = load_evaluation_results()
    return {
        "status": "ok",
        "version": APP_VERSION,
        "feature_count": len(get_feature_names()),
        "train_rows": ev.get("training_summary", {}).get("train_rows"),
        "thresholds": ev.get("thresholds"),
    }
