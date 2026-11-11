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
import socket
import warnings
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse
import requests
import urllib3
from bs4 import BeautifulSoup

# Certificate checking is deliberately off (phishing sites often have bad certs and we only read HTML),
# so silence the per-request warning.
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

MAX_REDIRECTS = 3

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

def _ip_is_blocked(ip: "ipaddress._BaseAddress") -> bool:
    """True for anything that is not a normal public address (loopback, private, link-local, CGNAT,
    multicast, unspecified, reserved, IPv4-mapped IPv6 of those, ...)."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if any(ip in net for net in PRIVATE_NETWORKS):
        return True
    return not ip.is_global


def is_ssrf_risk(hostname: str) -> bool:
    """Blocks localhost names and IP literals that are not public addresses."""
    clean_host = hostname.strip().strip("[]").lower().rstrip(".")
    if clean_host.count(":") <= 1:          # plain host or host:port (an IPv6 literal has several colons)
        clean_host = clean_host.split(":")[0]
    if clean_host in ("localhost", "metadata.google.internal", "metadata") or clean_host.endswith((".localhost", ".internal", ".local")):
        return True
    try:
        return _ip_is_blocked(ipaddress.ip_address(clean_host))
    except ValueError:
        return False                         # a named domain: checked after DNS resolution


def resolves_to_blocked_address(hostname: str) -> Optional[bool]:
    """Resolves the name and reports whether ANY answer is a non-public address.
    Returns None if the name cannot be resolved."""
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        return None
    for info in infos:
        try:
            if _ip_is_blocked(ipaddress.ip_address(info[4][0])):
                return True
        except ValueError:
            return True
    return False


def _blocked_result(message: str) -> Dict[str, Any]:
    return {"status": "blocked", "message": message, "findings": []}


def _fetch_with_safe_redirects(url: str):
    """GET with every hop validated (scheme, literal IP, DNS answers) before it is requested.
    Returns (response, hops). Raises ValueError with a user-facing message when a hop is not allowed.

    Residual risk: DNS can change between our lookup and the connection (rebinding). Run this service
    without access to internal networks if that matters for your deployment."""
    session = requests.Session()
    session.trust_env = False                # ignore proxy env vars for the sandbox fetch
    hops: List[str] = []
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only http and https addresses can be inspected.")
        host = (parsed.hostname or "").lower()
        if not host or is_ssrf_risk(host):
            raise ValueError("Sandbox policy: requests to internal or loopback network targets are prohibited.")
        blocked = resolves_to_blocked_address(host)
        if blocked:
            raise ValueError("Sandbox policy: that name resolves to an internal network address.")
        response = session.get(current, headers=SAFE_HEADERS, timeout=2.5, allow_redirects=False,
                               stream=True, verify=False)
        if response.is_redirect and response.headers.get("Location"):
            hops.append(current)
            response.close()
            current = urljoin(current, response.headers["Location"])
            continue
        return response, hops
    raise ValueError("Too many redirects (more than %d)." % MAX_REDIRECTS)


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

    if is_ssrf_risk(hostname):
        return _blocked_result("Sandbox Policy: Requests to internal/loopback network targets are strictly prohibited.")

    findings: List[Dict[str, Any]] = []

    try:
        try:
            response, hops = _fetch_with_safe_redirects(url)
        except ValueError as exc:
            return _blocked_result(str(exc))

        # Read only up to 500 KB
        content_bytes = response.raw.read(500 * 1024, decode_content=True)
        html_text = content_bytes.decode("utf-8", errors="replace")

        final_url = response.url
        response.close()
        final_parsed = urlparse(final_url)
        final_hostname = final_parsed.netloc.split(":")[0].lower()

        # Signal 1: Cross-domain redirect detection
        if len(hops) > 0:
            if final_hostname != hostname:
                findings.append({
                    "severity": "HIGH",
                    "title": "Cross-Domain Redirection Detected",
                    "detail": f"Initial host '{hostname}' redirected through {len(hops)} hops to '{final_hostname}'.",
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
            "redirect_hops": len(hops),
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
