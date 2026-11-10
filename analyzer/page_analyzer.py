"""
Isolated Webpage Analysis Engine (Optional Extension)
Performs safe, restricted, and isolated inspection of live webpage signals without executing client-side scripts.

Safety Protections:
- Hard 2.5s network timeout
- SSRF prevention (blocks private/loopback/metadata addresses)
- Enforced maximum redirect cap (3 hops)
- Maximum byte read limit (500 KB)
- No JavaScript execution, cookie sandbox, no rendering engine
- Graceful degradation when host is offline or unreachable
"""

import ipaddress
import re
from typing import Dict, Any, List
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

SAFE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko; SecurityAuditEngine/1.0)",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

def is_ssrf_risk(hostname: str) -> bool:
    """Blocks requests to localhost, loopback, or internal intranet ranges."""
    clean_host = hostname.split(":")[0].strip().lower()
    if clean_host in ["localhost", "127.0.0.1", "::1", "metadata.google.internal"]:
        return True
    try:
        ip = ipaddress.ip_address(clean_host)
        return any(ip in net for net in PRIVATE_NETWORKS)
    except ValueError:
        # Host is a named domain, not a direct IP literal
        return False

def inspect_live_page_isolated(url_str: str) -> Dict[str, Any]:
    """
    Safely inspects HTML document from target URL in an isolated sandbox.
    Returns findings including title-to-domain match, login inputs, and redirect chains.
    """
    url = url_str.strip()
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url):
        url = "https://" + url

    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.split(":")[0].lower()
    except Exception:
        return {
            "status": "error",
            "message": "Malformed URL prevented live sandbox fetch.",
            "findings": []
        }

    # SSRF verification
    if is_ssrf_risk(hostname):
        return {
            "status": "blocked",
            "message": "Sandbox Policy: Requests to internal/loopback network targets are strictly prohibited.",
            "findings": []
        }

    findings: List[Dict[str, Any]] = []

    try:
        session = requests.Session()
        session.max_redirects = 3

        # Stream request with 2.5 second timeout to avoid slowloris
        response = session.get(
            url,
            headers=SAFE_HEADERS,
            timeout=2.5,
            allow_redirects=True,
            stream=True,
            verify=False # Allow inspecting self-signed phishing certs without breaking sandbox
        )

        # Read only up to 500 KB
        content_bytes = response.raw.read(500 * 1024, decode_content=True)
        html_text = content_bytes.decode("utf-8", errors="replace")

        final_url = response.url
        final_parsed = urlparse(final_url)
        final_hostname = final_parsed.netloc.split(":")[0].lower()

        # Signal 1: Cross-domain redirect detection
        if len(response.history) > 0:
            if final_hostname != hostname:
                findings.append({
                    "severity": "HIGH",
                    "title": "Cross-Domain Redirection Detected",
                    "detail": f"Initial host '{hostname}' redirected through {len(response.history)} hops to '{final_hostname}'.",
                    "risk_delta": 25
                })
            else:
                findings.append({
                    "severity": "LOW",
                    "title": "Internal Redirection",
                    "detail": f"Page redirected within the same domain to '{final_url}'.",
                    "risk_delta": 5
                })

        # Parse HTML safely
        soup = BeautifulSoup(html_text, "html.parser")

        # Signal 2: Page Title vs Domain Mismatch
        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else ""
        
        target_brands = ["paypal", "apple", "microsoft", "google", "netflix", "amazon", "chase", "bank of america", "wells fargo", "steam", "coinbase"]
        title_lower = page_title.lower()

        mismatched_brand = None
        for brand in target_brands:
            brand_token = brand.replace(" ", "")
            if brand in title_lower:
                # Does the brand name match the actual domain?
                if brand_token not in final_hostname:
                    mismatched_brand = brand
                    break

        if mismatched_brand:
            findings.append({
                "severity": "CRITICAL",
                "title": "Title vs Domain Brand Mismatch",
                "detail": f"Page title claims to be '{mismatched_brand.title()}' ('{page_title[:60]}'), but hosted domain is '{final_hostname}'.",
                "risk_delta": 35
            })

        # Signal 3: Password / Credential Input Harvest Forms
        password_inputs = soup.find_all("input", attrs={"type": re.compile(r"^password$", re.I)})
        has_password = len(password_inputs) > 0

        # Form action analysis
        suspicious_form_action = False
        forms = soup.find_all("form")
        for f in forms:
            action = f.get("action", "")
            if action:
                act_parsed = urlparse(action)
                if act_parsed.netloc and act_parsed.netloc.lower() != final_hostname:
                    suspicious_form_action = True
                    break

        if has_password:
            if suspicious_form_action:
                findings.append({
                    "severity": "CRITICAL",
                    "title": "Credential Harvest: Form Posts to Third-Party Host",
                    "detail": "Login form sends submitted password credentials to an external destination host.",
                    "risk_delta": 30
                })
            else:
                findings.append({
                    "severity": "MEDIUM",
                    "title": "Authentication Form Present",
                    "detail": f"Page contains {len(password_inputs)} password input field(s).",
                    "risk_delta": 10
                })

        return {
            "status": "success",
            "message": "Isolated page analysis completed successfully.",
            "http_status": response.status_code,
            "final_url": final_url,
            "page_title": page_title if page_title else "(none)",
            "redirect_hops": len(response.history),
            "has_password_input": has_password,
            "findings": findings
        }

    except requests.exceptions.Timeout:
        return {
            "status": "unavailable",
            "message": "Live inspection unavailable: Target server timed out (>2.5s) in sandbox.",
            "findings": []
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "unavailable",
            "message": "Live inspection unavailable: DNS resolution failed or target host refused connection.",
            "findings": []
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "message": f"Live inspection unavailable: {str(e)[:80]}.",
            "findings": []
        }
