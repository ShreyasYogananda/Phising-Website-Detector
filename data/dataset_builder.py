"""
Dataset builder for PhishGuard.

What changed compared with the first version
--------------------------------------------
The original corpus was generated from string templates but labelled as if it came
from PhishTank / OpenPhish / Tranco. Because the templates were trivially separable the
model scored a perfect 100% on its test set, which says nothing about real-world accuracy.

This builder uses REAL phishing URLs and holds the model to an honest standard:

  Phishing  : mitchellkrogza/Phishing.Database (a public aggregation of PhishTank, OpenPhish
              and other community feeds). `phishing-links-ACTIVE.txt` feeds train/val/test,
              `phishing-links-INACTIVE.txt` (older, different campaigns) feeds the OOD test.
  Legitimate: (a) real long-tail benign URLs from the public ISCX-style corpus in
                  faizann24/Using-machine-learning-to-detect-malicious-URLs (DMOZ-era, with real paths);
              (b) well-known domains combined with realistic path templates that deliberately include
                  words like login/account/verify (curated, SYNTHETIC - labelled as such);
              (c) a hand-written hard-benign set used only for the OOD test.

Bias controls
-------------
* per-hostname, per-registered-domain and per-hosting-platform caps, so one campaign or one
  platform cannot dominate;
* scheme (http/https) and "has a path" are decorrelated from the label by resampling, so the
  model cannot cheat on collection artefacts;
* train / validation / test are split BY REGISTERED DOMAIN (a site never appears in two splits);
* the OOD test uses older campaigns and hand-written benign URLs the model never saw.

Run:  python data/dataset_builder.py            (downloads feeds into data/cache/, ~75 MB)
"""

import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from features.extractor import FREE_HOSTING_SUFFIXES, KNOWN_BRANDS, is_free_hosting, split_host  # noqa: E402

SEED = 42
CACHE_DIR = os.path.join(BASE_DIR, "cache")

FEEDS = {
    "phish_active": "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "phish_inactive": "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-INACTIVE.txt",
    "benign_longtail": "https://raw.githubusercontent.com/faizann24/Using-machine-learning-to-detect-malicious-URLs/master/data/data.csv",
}

N_PHISH_MAIN = 6000
N_PHISH_OOD = 1200
N_BENIGN_LONGTAIL = 3300
MAX_PER_HOST = 2
MAX_PER_REGISTERED = 8
MAX_PLATFORM_SHARE = 0.035        # of the phishing sample, per hosting platform

# --------------------------------------------------------------------------- #
# Curated well-known domains (benign, synthetic paths)
# --------------------------------------------------------------------------- #
POPULAR_DOMAINS = """
google.com microsoft.com apple.com github.com amazon.com wikipedia.org youtube.com linkedin.com reddit.com
mozilla.org stackoverflow.com cloudflare.com netflix.com spotify.com chase.com bankofamerica.com wellsfargo.com
paypal.com stripe.com dropbox.com salesforce.com adobe.com zoom.us slack.com atlassian.com nytimes.com bbc.com
cnn.com theguardian.com reuters.com harvard.edu mit.edu stanford.edu berkeley.edu ox.ac.uk cam.ac.uk nih.gov
nasa.gov cdc.gov irs.gov who.int docker.com kubernetes.io python.org rust-lang.org golang.org npmjs.com pypi.org
apache.org medium.com quora.com airbnb.com booking.com uber.com lyft.com etsy.com ebay.com target.com walmart.com
bestbuy.com homedepot.com fidelity.com vanguard.com schwab.com capitalone.com citi.com usps.com fedex.com ups.com
dhl.com weather.com yahoo.com bing.com duckduckgo.com twitter.com facebook.com instagram.com whatsapp.com
telegram.org tiktok.com pinterest.com twitch.tv discord.com steampowered.com roblox.com epicgames.com ea.com
nintendo.com playstation.com xbox.com coinbase.com binance.com kraken.com blockchain.com metamask.io
hdfcbank.com icicibank.com axisbank.com kotak.com onlinesbi.sbi sbi.co.in paytm.com phonepe.com flipkart.com
irctc.co.in uidai.gov.in incometax.gov.in india.gov.in npci.org.in zomato.com swiggy.com myntra.com
timesofindia.indiatimes.com thehindu.com ndtv.com hindustantimes.com iisc.ac.in iitb.ac.in iitd.ac.in bmsce.ac.in
vtu.ac.in nptel.ac.in coursera.org edx.org udemy.com khanacademy.org kaggle.com huggingface.co arxiv.org
openai.com anthropic.com tensorflow.org pytorch.org scikit-learn.org numpy.org pandas.pydata.org jupyter.org
fastapi.tiangolo.com flask.palletsprojects.com djangoproject.com reactjs.org nodejs.org vuejs.org angular.io
typescriptlang.org php.net ruby-lang.org kotlinlang.org developer.android.com developer.apple.com flutter.dev
firebase.google.com cloud.google.com aws.amazon.com azure.microsoft.com digitalocean.com heroku.com vercel.com
netlify.com gitlab.com bitbucket.org notion.so figma.com canva.com trello.com asana.com monday.com zendesk.com
hubspot.com mailchimp.com shopify.com squarespace.com wix.com wordpress.org godaddy.com namecheap.com
verizon.com att.com t-mobile.com comcast.com vodafone.com airtel.in jio.com bsnl.co.in
hsbc.com santander.com barclays.co.uk lloydsbank.com natwest.com usbank.com pnc.com tdbank.com ally.com
americanexpress.com discover.com visa.com mastercard.com westernunion.com wise.com revolut.com
expedia.com tripadvisor.com kayak.com skyscanner.com makemytrip.com goibibo.com delta.com united.com
imdb.com rottentomatoes.com goodreads.com last.fm soundcloud.com vimeo.com dailymotion.com
espn.com nba.com fifa.com uefa.com icc-cricket.com cricbuzz.com espncricinfo.com
cisco.com ibm.com oracle.com intel.com amd.com nvidia.com dell.com hp.com lenovo.com samsung.com sony.com
bbc.co.uk gov.uk nhs.uk europa.eu un.org worldbank.org imf.org
""".split()

BENIGN_SUBDOMAINS = ["www", "www", "www", "docs", "support", "help", "blog", "developer", "api", "status", "m", "mail",
                     "accounts", "login", "secure", "my", "portal", "app", "cloud", "learn", "news", "shop", "store"]

# Paths used for the TRAIN/VAL/TEST benign pool. Many intentionally contain "risky" vocabulary so the
# model learns that login/account/support on a real domain is normal.
BENIGN_PATHS_MAIN = [
    "", "/", "/about", "/contact", "/privacy", "/terms", "/careers", "/pricing", "/products", "/download",
    "/login", "/signin", "/account", "/account/settings", "/account/security", "/my-account/orders",
    "/support", "/support/contact-us", "/help/article/48392", "/help/faq", "/billing", "/billing/invoices",
    "/security", "/security/overview", "/verify-email?token=9f8a7b6c", "/update-profile", "/password/reset",
    "/blog/2025/09/how-we-built-our-search-infrastructure", "/blog/post/introducing-version-4",
    "/docs/getting-started", "/docs/v2/api/reference#authentication", "/docs/en/latest/index.html",
    "/search?q=best+laptops+2026&sort=relevance", "/products/category/electronics?page=3&sort=price",
    "/news/world/2026/03/technology-trends-explained", "/article/12345678/market-update",
    "/store/apps/details?id=com.example.app&hl=en_US", "/watch?v=dQw4w9WgXcQ", "/wiki/Computer_security",
    "/en-us/products/software/overview", "/en/support/downloads/drivers", "/in/personal/banking/accounts",
    "/personal/credit-cards/rewards", "/customer-service/track-order?orderId=114-5589231-2210617",
    "/careers/jobs/software-engineer-bangalore", "/investor-relations/annual-report-2025.pdf",
    "/events/webinar/register", "/community/discussions/topic/78103?reply=true", "/oauth2/authorize?client_id=abc123&response_type=code",
    "/signin/v2/identifier?continue=https%3A%2F%2Fmail.google.com&service=mail",
    "/dashboard/overview?period=last_30_days", "/settings/notifications", "/checkout/cart", "/orders/114-558",
]

# Brand names appearing in benign paths (repo owners, wiki titles, subreddits, news). Without these the model
# learns "brand in path" as a phishing proxy and flags e.g. github.com/paypal/paypal-sdk.
BRAND_PATH_TEMPLATES = [
    "/{b}/{b}-sdk", "/{B}", "/wiki/{B}", "/r/{b}", "/company/{b}", "/questions/tagged/{b}",
    "/news/{b}-reports-quarterly-earnings", "/blog/{b}-announces-new-features", "/topics/{b}",
    "/{b}/{b}-python/issues/1042", "/tag/{b}", "/pages/{B}/about",
]

FREE_HOST_PLATFORMS = ["github.io", "pages.dev", "netlify.app", "vercel.app", "web.app", "firebaseapp.com",
                       "herokuapp.com", "onrender.com", "gitlab.io", "blogspot.com", "wordpress.com", "wixsite.com",
                       "weebly.com", "glitch.me", "repl.co", "workers.dev"]
FREE_HOST_OWNERS = ["alex", "priya", "rahul", "sam", "jordan", "maria", "chen", "kumar", "li", "omar", "nina", "ravi",
                    "sara", "dev", "tech", "code", "data", "ml", "web", "art", "anita", "vikram", "meera", "arjun",
                    "kavya", "rohit", "neha", "shreya", "karan", "isha", "tom", "emma", "luke", "zoe", "noah"]
FREE_HOST_WORDS = ["portfolio", "blog", "notes", "docs", "resume", "cv", "project", "demo", "app", "website", "personal",
                   "study", "lab", "course", "hackathon", "todo", "weather-app", "calculator", "landing", "gallery",
                   "recipes", "travel", "photography", "college-project", "minor-project", "events", "club"]
FREE_HOST_PATHS = ["", "/", "/about", "/projects", "/blog/first-post", "/docs/intro", "/contact", "/index.html",
                   "/projects/weather-app/", "/resume.pdf", "/gallery?page=2"]

# Held out for the OOD test: unseen path shapes on unseen domains.
BENIGN_PATHS_OOD = [
    "/internal/tools/status", "/en-gb/legal/cookie-policy", "/services/online-banking/overview",
    "/accounts/recover/options", "/secure/login?redirect=%2Fhome", "/support/tickets/open",
    "/products/2026/new-arrivals?utm_source=newsletter&utm_medium=email&utm_campaign=spring",
]

# Hand-written hard-benign URLs for the OOD set. Real, well-known pages with phishing-like vocabulary.
HARD_BENIGN = """
https://accounts.google.com/signin/v2/identifier?service=mail
https://myaccount.google.com/security
https://login.microsoftonline.com/common/oauth2/v2.0/authorize
https://login.live.com/login.srf
https://appleid.apple.com/sign-in
https://www.paypal.com/signin
https://www.paypal.com/myaccount/summary
https://www.amazon.com/gp/css/homepage.html
https://www.amazon.in/ap/signin
https://secure.chase.com/web/auth/dashboard
https://www.bankofamerica.com/online-banking/sign-in/
https://www.wellsfargo.com/help/security/
https://www.hdfcbank.com/personal/pay/cards
https://netbanking.hdfcbank.com/netbanking/
https://www.icicibank.com/personal-banking/accounts
https://onlinesbi.sbi/
https://retail.onlinesbi.sbi/retail/login.htm
https://www.irctc.co.in/nget/train-search
https://www.incometax.gov.in/iec/foportal/
https://uidai.gov.in/en/my-aadhaar
https://www.npci.org.in/what-we-do/upi/product-overview
https://paytm.com/recharge
https://www.flipkart.com/account/login?ret=/
https://github.com/login
https://github.com/torvalds/linux/blob/master/README
https://github.com/pallets/flask/issues/4821
https://gitlab.com/users/sign_in
https://stackoverflow.com/users/login?ssrc=head
https://developer.mozilla.org/en-US/docs/Web/Security/Practical_implementation_guides/TLS
https://en.wikipedia.org/wiki/Phishing
https://en.wikipedia.org/wiki/PayPal
https://en.wikipedia.org/wiki/Apple_Inc.
https://www.bbc.co.uk/news/technology-68123456
https://www.nytimes.com/2026/02/14/technology/ai-security.html
https://pandas.pydata.org/docs/user_guide/index.html
https://numpy.org/doc/stable/reference/generated/numpy.array.html
https://scikit-learn.org/stable/modules/ensemble.html#forest
https://fastapi.tiangolo.com/tutorial/bigger-applications/
https://docs.python.org/3/library/urllib.parse.html
https://kubernetes.io/docs/concepts/security/overview/
https://www.cloudflare.com/learning/security/what-is-phishing/
https://www.microsoft.com/en-us/security/business/security-101/what-is-phishing
https://support.apple.com/en-us/HT204053
https://support.google.com/accounts/answer/32040
https://help.netflix.com/en/node/24926
https://www.netflix.com/login
https://www.spotify.com/us/account/overview/
https://discord.com/login
https://steamcommunity.com/login/home/
https://www.coinbase.com/signin
https://www.binance.com/en/my/security
https://www.linkedin.com/login
https://www.facebook.com/login/
https://www.instagram.com/accounts/login/
https://web.whatsapp.com/
https://www.dropbox.com/login
https://zoom.us/signin
https://slack.com/signin
https://www.docusign.com/products/electronic-signature
https://www.fedex.com/en-us/tracking.html
https://www.ups.com/track?loc=en_US
https://www.usps.com/manage/welcome.htm
https://www.dhl.com/in-en/home/tracking.html
https://www.irs.gov/refunds
https://www.irs.gov/payments
https://www.usa.gov/
https://www.nhs.uk/conditions/
https://www.who.int/news-room/fact-sheets/detail/influenza
https://nptel.ac.in/courses/106/105/106105152/
https://bmsce.ac.in/home/Computer-Science-and-Engineering
https://www.iisc.ac.in/admissions/
https://vtu.ac.in/en/
https://www.coursera.org/learn/machine-learning
https://www.kaggle.com/datasets
https://huggingface.co/models?pipeline_tag=text-classification
https://arxiv.org/abs/1706.03762
https://pypi.org/project/scikit-learn/
https://www.npmjs.com/package/express
https://hub.docker.com/_/python
https://aws.amazon.com/security/
https://console.aws.amazon.com/console/home
https://portal.azure.com/
https://cloud.google.com/security
https://firebase.google.com/docs/auth
https://vercel.com/login
https://dashboard.stripe.com/login
https://www.shopify.com/login
https://wordpress.org/support/
https://torvalds.github.io/
https://pages.github.com/
https://pytorch.org/tutorials/beginner/basics/intro.html
https://tensorflow.org/tutorials/quickstart/beginner
https://jupyter.org/install
https://www.mozilla.org/en-US/firefox/new/
https://www.apple.com/shop/buy-iphone/iphone-16
https://www.samsung.com/in/smartphones/all-smartphones/
https://www.airtel.in/recharge-online
https://www.jio.com/selfcare/login/
https://www.zomato.com/bangalore/restaurants
https://www.swiggy.com/restaurants
https://www.makemytrip.com/flights/
https://www.timesofindia.indiatimes.com/india/
https://www.thehindu.com/news/national/karnataka/
https://www.ndtv.com/india-news
https://www.cricbuzz.com/cricket-match/live-scores
https://www.espncricinfo.com/live-cricket-score
https://www.imdb.com/title/tt0111161/
https://medium.com/@user/how-phishing-works-3a9f2c1b
https://www.reddit.com/r/cybersecurity/comments/1abc23/phishing_report/
https://www.quora.com/What-is-phishing
https://www.youtube.com/watch?v=jNQXAC9IVRw
https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/edit
https://drive.google.com/drive/my-drive
https://outlook.office.com/mail/
https://www.office.com/launch/word
https://teams.microsoft.com/_#/conversations
https://github.com/paypal/paypal-checkout-components
https://github.com/stripe/stripe-python
https://github.com/microsoft/vscode/issues/200123
https://github.com/google/guava
https://github.com/apple/swift
https://github.com/facebook/react/pull/27012
https://github.com/amzn/selfimproving-agents
https://www.reddit.com/r/Netflix/
https://www.reddit.com/r/paypal/comments/1abc/refund_question/
https://stackoverflow.com/questions/tagged/paypal-sandbox
https://en.wikipedia.org/wiki/Amazon_(company)
https://en.wikipedia.org/wiki/Netflix
https://www.linkedin.com/company/microsoft/
https://medium.com/@spotify/how-we-built-wrapped
https://pallets.github.io/flask/
https://pytorch.github.io/
https://microsoft.github.io/vscode-pull-request-github/
https://google.github.io/styleguide/pyguide.html
https://facebook.github.io/react-native/
https://huggingface.github.io/
https://jekyllrb.com/docs/
https://octocat.github.io/
https://developers.cloudflare.com/pages/
https://vercel.com/docs
https://docs.github.com/en/pages
""".split()


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def download(name: str) -> str:
    """Downloads a feed into data/cache/ (once) and returns the local path."""
    import requests

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.txt")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    print(f"[*] Downloading {name} ...", flush=True)
    r = requests.get(FEEDS[name], timeout=300, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return path


_URL_OK = re.compile(r"^https?://[^\s/?#]+", re.I)


def clean_url(u: str):
    u = (u or "").strip().strip('"').strip("'")
    if not u or len(u) > 2000 or any(c.isspace() for c in u):
        return None
    if not _URL_OK.match(u):
        return None
    try:
        p = urlparse(u)
        if not p.hostname or "." not in p.hostname:
            return None
    except ValueError:
        return None
    return u


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def has_path(url: str) -> bool:
    p = urlparse(url)
    return bool(p.path.strip("/") or p.query)


def stable_bucket(key: str, buckets: int = 100) -> int:
    return int(hashlib.md5(f"{SEED}:{key}".encode()).hexdigest(), 16) % buckets


# --------------------------------------------------------------------------- #
# Collectors
# --------------------------------------------------------------------------- #
def collect_phishing(path: str, n: int, rng: random.Random, exclude_regs=frozenset()):
    """Reads a feed, applies caps, and returns a diversified random sample of n URLs."""
    seen, per_host, per_reg = set(), Counter(), Counter()
    kept = []
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [ln for ln in f]
    rng.shuffle(lines)                                   # unbiased w.r.t. file order
    platform_cap = max(5, int(n * MAX_PLATFORM_SHARE))
    per_platform = Counter()
    for ln in lines:
        u = clean_url(ln)
        if not u or u in seen:
            continue
        seen.add(u)
        host = host_of(u)
        _, reg, suffix = split_host(host)
        if reg in exclude_regs:
            continue
        if per_host[host] >= MAX_PER_HOST or per_reg[reg] >= MAX_PER_REGISTERED:
            continue
        if is_free_hosting(host):
            plat = next((s for s in FREE_HOSTING_SUFFIXES if host == s or host.endswith("." + s)), suffix)
            if per_platform[plat] >= platform_cap:
                continue
            per_platform[plat] += 1
        per_host[host] += 1
        per_reg[reg] += 1
        kept.append(u)
        if len(kept) >= n:
            break
    return kept


def collect_benign_longtail(path: str, n: int, rng: random.Random):
    csv.field_size_limit(10 ** 7)
    out, per_reg = [], Counter()
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[-1].strip().lower() == "good":
                rows.append(row[0].strip())
    rng.shuffle(rows)
    for raw in rows:
        u = clean_url("http://" + raw if not re.match(r"^[a-z]+://", raw, re.I) else raw)
        if not u:
            continue
        reg = split_host(host_of(u))[1]
        if per_reg[reg] >= 2:
            continue
        per_reg[reg] += 1
        out.append(u)
        if len(out) >= n:
            break
    return out


def collect_benign_freehost(rng: random.Random, n: int):
    """Personal / student sites on free platforms (SYNTHETIC names). Teaches the model that a free host alone is normal."""
    out = []
    while len(out) < n:
        plat = rng.choice(FREE_HOST_PLATFORMS)
        owner, word = rng.choice(FREE_HOST_OWNERS), rng.choice(FREE_HOST_WORDS)
        name = rng.choice([f"{owner}-{word}", f"{owner}{word}", f"{word}-{owner}", f"{owner}", f"{owner}-{rng.randint(1, 99)}-{word}"])
        path = rng.choice(FREE_HOST_PATHS)
        if plat in ("github.io", "gitlab.io") and rng.random() < 0.6:
            url = f"https://{owner}.{plat}/{word}{path if path not in ('', '/') else '/'}"
        else:
            url = f"https://{name}.{plat}{path}"
        out.append(url)
    return list(dict.fromkeys(out))


def collect_benign_curated(rng: random.Random, domains, paths, per_domain):
    out = []
    brands = list(KNOWN_BRANDS)
    for dom in domains:
        for _ in range(per_domain):
            sub = rng.choice(BENIGN_SUBDOMAINS)
            host = dom if (sub == "www" and rng.random() < 0.4) else f"{sub}.{dom}"
            if host.startswith("www.") and dom.startswith("www."):
                host = dom
            path = rng.choice(paths)
            if rng.random() < 0.25:                       # brand mentioned in a harmless path
                b = rng.choice(brands)
                path = rng.choice(BRAND_PATH_TEMPLATES).format(b=b, B=b.title())
            out.append(f"https://{host}{path}")
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------- #
# Bias controls
# --------------------------------------------------------------------------- #
def set_scheme(url: str, https: bool) -> str:
    return re.sub(r"^https?://", "https://" if https else "http://", url, count=1, flags=re.I)


def decorrelate_scheme(benign, phish, rng):
    """Re-assigns the benign scheme to match the phishing https rate so the scheme carries no label signal."""
    p_https = sum(u.lower().startswith("https://") for u in phish) / max(len(phish), 1)
    return [set_scheme(u, rng.random() < p_https) for u in benign]


def match_path_presence(benign_rows, phish_rows, rng):
    """Subsamples benign so P(has path/query | benign) matches phishing, then trims to equal size."""
    pp = sum(has_path(r["url"]) for r in phish_rows) / max(len(phish_rows), 1)
    with_p = [r for r in benign_rows if has_path(r["url"])]
    without = [r for r in benign_rows if not has_path(r["url"])]
    rng.shuffle(with_p)
    rng.shuffle(without)
    n_total = min(len(phish_rows), int(min(len(with_p) / pp if pp else 1e9, len(without) / (1 - pp) if pp < 1 else 1e9)))
    n_with, n_without = int(n_total * pp), n_total - int(n_total * pp)
    return with_p[:n_with] + without[:n_without]


def split_by_domain(rows):
    """70/15/15 split keyed by registered domain: a site never crosses splits."""
    buckets = {"train": [], "val": [], "test": []}
    for r in rows:
        reg = split_by_domain.cache.setdefault(r["url"], split_host(host_of(r["url"]))[1])
        b = stable_bucket(reg)
        name = "train" if b < 70 else ("val" if b < 85 else "test")
        buckets[name].append({**r, "split": name})
    return buckets


split_by_domain.cache = {}


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "label", "source", "split"])
        for r in rows:
            w.writerow([r["url"], r["label"], r["source"], r["split"]])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build():
    rng = random.Random(SEED)

    phish_active = collect_phishing(download("phish_active"), N_PHISH_MAIN, rng)
    main_regs = {split_host(host_of(u))[1] for u in phish_active}
    phish_ood = collect_phishing(download("phish_inactive"), N_PHISH_OOD, rng, exclude_regs=main_regs)

    # --- benign pools -------------------------------------------------------
    longtail = collect_benign_longtail(download("benign_longtail"), N_BENIGN_LONGTAIL, rng)
    domains = sorted(set(POPULAR_DOMAINS))
    rng.shuffle(domains)
    ood_domains = domains[: len(domains) // 6]                      # never used for training
    main_domains = domains[len(domains) // 6:]
    curated = collect_benign_curated(rng, main_domains, BENIGN_PATHS_MAIN, per_domain=14)
    curated_ood = collect_benign_curated(rng, ood_domains, BENIGN_PATHS_OOD + BENIGN_PATHS_MAIN[:10], per_domain=6)

    freehost = collect_benign_freehost(rng, 700)
    longtail = decorrelate_scheme(longtail, phish_active, rng)
    curated = decorrelate_scheme(curated, phish_active, rng)
    freehost = decorrelate_scheme(freehost, phish_active, rng)

    benign_rows = (
        [{"url": u, "label": 0, "source": "Real long-tail benign (ISCX-style corpus)"} for u in longtail]
        + [{"url": u, "label": 0, "source": "Curated popular domains (synthetic paths)"} for u in curated]
        + [{"url": u, "label": 0, "source": "Personal sites on free platforms (synthetic names)"} for u in freehost]
    )
    phish_rows = [{"url": u, "label": 1, "source": "Phishing.Database ACTIVE (real)"} for u in phish_active]
    benign_rows = match_path_presence(benign_rows, phish_rows, rng)

    # Never let a domain appear in both classes (e.g. a benign sample from a domain also seen phishing)
    phish_regs = {split_host(host_of(r["url"]))[1] for r in phish_rows}
    benign_rows = [r for r in benign_rows if split_host(host_of(r["url"]))[1] not in phish_regs]
    n = min(len(benign_rows), len(phish_rows))
    rng.shuffle(benign_rows)
    rng.shuffle(phish_rows)
    benign_rows, phish_rows = benign_rows[:n], phish_rows[:n]

    parts = split_by_domain(benign_rows + phish_rows)

    # --- OOD test -----------------------------------------------------------
    ood_rows = (
        [{"url": u, "label": 1, "source": "Phishing.Database INACTIVE (real, older campaigns)", "split": "ood"} for u in phish_ood]
        + [{"url": u, "label": 0, "source": "Hand-written hard benign", "split": "ood"} for u in HARD_BENIGN]
        + [{"url": u, "label": 0, "source": "Curated unseen domains (synthetic paths)", "split": "ood"} for u in curated_ood[:300]]
    )
    rng.shuffle(ood_rows)

    os.makedirs(BASE_DIR, exist_ok=True)
    for name, rows in parts.items():
        rng.shuffle(rows)
        write_csv(os.path.join(BASE_DIR, f"{name}.csv"), rows)
    write_csv(os.path.join(BASE_DIR, "ood_test.csv"), ood_rows)

    def stats(rows):
        c = Counter(r["label"] for r in rows)
        return {"total": len(rows), "phishing": c[1], "legitimate": c[0]}

    meta = {
        "dataset_name": "PhishGuard URL corpus v2",
        "seed": SEED,
        "honesty_notes": [
            "Phishing URLs are real, taken from the public Phishing.Database aggregation of community feeds.",
            "Roughly half of the legitimate URLs are real long-tail URLs from an older public corpus; the rest are "
            "well-known domains combined with synthetic-but-realistic paths. Treat metrics as an optimistic bound for "
            "the benign side.",
            "Scheme and 'has path' were decorrelated from the label; splits are grouped by registered domain.",
            "The OOD test uses older phishing campaigns and hand-written benign URLs not used in training.",
        ],
        "sources": FEEDS,
        "caps": {"per_host": MAX_PER_HOST, "per_registered_domain": MAX_PER_REGISTERED, "platform_share": MAX_PLATFORM_SHARE},
        "splits": {k: stats(v) for k, v in parts.items()} | {"ood_test": stats(ood_rows)},
        "source_breakdown": dict(Counter(r["source"] for rows in list(parts.values()) + [ood_rows] for r in rows)),
    }
    with open(os.path.join(BASE_DIR, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["splits"], indent=2))
    return meta


if __name__ == "__main__":
    build()
