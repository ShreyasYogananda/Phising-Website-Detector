# 🛡️ PhishGuard AI - Explainable Phishing Risk Checker

An interactive, explainable cybersecurity web application and threat intelligence tool that analyzes submitted URLs for phishing risks, compares a transparent rule-based heuristic baseline with an ensemble machine learning classifier, and delivers granular, human-interpretable evidence for its verdicts.

---

## 🎯 Key Capabilities & Architecture

1. **Dual Detection Paradigms**:
   - **Transparent Rule-Based Baseline**: A deterministic, auditable security heuristic engine evaluating 11 calibrated security penalties (IP host usage, brand spoofing, `@` obfuscation, double-slash redirects, suspicious TLDs, Punycode homographs, excessive subdomains).
   - **Random Forest ML Classifier**: A supervised ensemble model trained on 21 lexical, structural, statistical, and semantic features with calibrated probability estimates and localized feature impact attributions.
2. **Explainable Risk Synthesis**:
   - Computes a unified **Risk Index** (0 to 100) and **Verdict** (`Likely safe`, `Suspicious`, `High risk`).
   - Measures **Decision Uncertainty & Model Consensus** between both detectors, warning users when detectors diverge on borderline cases.
   - Decomposes exact signals: triggered security rules, top driving ML features, and RFC 3986 URL component anatomy.
3. **Transparent Evaluation Dashboard**:
   - Evaluates both detectors on the **exact same held-out test partition** (`test.csv`, 15% split, $N = 234$).
   - Reports Precision, Recall, F1-Score, Accuracy, and False Positive / Negative rates without fabricated metrics.
   - Visual confusion matrices and real test examples of false positives and missed phishing URLs with root-cause failure diagnostics.
4. **Optional Isolated Page Analysis (Sandbox Extension)**:
   - Optional, strictly isolated webpage fetcher with SSRF prevention (blocks private/loopback IP spaces), hard 2.5s network timeout, maximum 3-hop redirect limit, and no client-side script execution.
   - Detects cross-domain redirects, credential harvesting forms (`<input type="password">`), and page title vs domain brand mismatches.

---

## 📂 Project Directory Structure

```text
miniproject/
├── README.md                      # Comprehensive documentation & demo guide
├── requirements.txt               # Python package dependencies
├── run.py                         # One-click web server runner (http://127.0.0.1:8000)
├── evaluate_cli.py                # Terminal evaluation benchmark runner
├── train_model.py                 # ML model training script
├── app.py                         # FastAPI REST application & template routing
├── data/
│   ├── dataset_builder.py         # Data collection, validation & split pipeline
│   ├── raw_urls.csv               # Complete balanced dataset (1,550 URLs)
│   ├── train.csv                  # 70% Train split (1,084 URLs)
│   ├── val.csv                    # 15% Validation split (232 URLs)
│   ├── test.csv                   # 15% Held-out Test split (234 URLs)
│   └── dataset_metadata.json      # Provenance and class balance documentation
├── features/
│   ├── __init__.py
│   └── extractor.py               # 21-feature URL extraction & Shannon entropy engine
├── models/
│   ├── __init__.py
│   ├── rule_based.py              # Auditable heuristic baseline detector
│   ├── ml_detector.py             # Random Forest classifier & local feature explainer
│   ├── explainer.py               # Consensus synthesis, uncertainty & risk aggregator
│   └── trained_model.joblib       # Persisted scikit-learn model artifact
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py               # Benchmark engine computing test set metrics & error mining
│   └── evaluation_results.json    # Pre-calculated test benchmark metrics & error cases
├── analyzer/
│   ├── __init__.py
│   └── page_analyzer.py           # Optional SSRF-safe isolated webpage inspection sandbox
├── static/
│   ├── css/style.css              # Cyber SOC dark dashboard styling
│   └── js/app.js                  # Dynamic SVG gauge meter, tabs & AJAX handlers
├── templates/
│   └── index.html                 # Main dashboard (Analyzer, Evaluation, Architecture)
└── tests/
    └── test_pipeline.py           # Automated unit tests covering features, models & SSRF sandbox
```

---

## 🚀 Setup & Run Instructions

### 1. Open the Project in VS Code
Open the project directory in Visual Studio Code:
```text
C:\Users\Shreyas Y\OneDrive\Documents\miniproject
```
In VS Code: **File > Open Folder...** and select **`miniproject`** inside your **Documents** folder.

### 2. Verify Dependencies
Open the built-in terminal in VS Code (<kbd>Ctrl</kbd> + <kbd>`</kbd>) and run:
```bash
pip install -r requirements.txt
```
*(All core packages—FastAPI, Scikit-Learn, Uvicorn, NumPy, Scipy, Requests, BeautifulSoup4—are already installed on your machine).*

### 3. Run the Web Application
Start the local server:
```bash
python run.py
```
Open your web browser and navigate to:
👉 **`http://127.0.0.1:8000`**

### 4. Run CLI Evaluation Benchmark
To re-evaluate both models against the held-out test data from the terminal:
```bash
python evaluate_cli.py
```

### 5. Run Automated Unit Tests
To verify all pipeline components, feature extractors, and security sandboxes:
```bash
python -m unittest discover -s tests
```

---

## 🔬 URL Feature Library (21 Extracted Features)

| Feature Name | Category | Cybersecurity Rationale |
| :--- | :--- | :--- |
| `url_length` | Lexical | Phishing URLs often pack lengthy redirection tokens or obscure malicious parameters. |
| `domain_length` | Lexical | Excessively long hostnames are common in disposable cybercrime infrastructure. |
| `path_length` | Lexical | Depth of directory and file paths following the domain name. |
| `has_ip_address` | Structural | Host is a raw IPv4/IPv6 address, bypassing DNS reputation scoring and domain take-downs. |
| `subdomain_count` | Structural | Deep subdomain levels used to simulate legitimate brands (e.g. `paypal.com.account-update.xyz`). |
| `dot_count` | Structural | High dot density across URL tokens is common in deceptive subdomains. |
| `hyphen_count` | Structural | Attackers frequently use hyphens to combine brand names with security terms. |
| `at_symbol_count` | Obfuscation | RFC 3986 allows userinfo before `@`; browsers discard it and navigate to the following host. |
| `double_slash_redirect`| Obfuscation | `//` occurring after protocol scheme, exploited in path confusion attacks. |
| `suspicious_char_count`| Obfuscation | Delimiters and encoding symbols (`_`, `~`, `%`, `&`, `=`, `?`, `;`, `+`, `$`). |
| `url_entropy` | Statistical | Shannon information entropy measuring character randomness and algorithmic obfuscation. |
| `domain_entropy` | Statistical | Shannon entropy of domain label characters (detects DGAs and hex strings). |
| `has_https` | Protocol | Indicator of TLS encryption (1 for HTTPS, 0 for plain HTTP). |
| `has_port` | Protocol | Non-standard port numbers (outside 80/443) indicating ad-hoc malicious servers. |
| `suspicious_keyword_count` | Semantic | Count of sensitive action keywords (`login`, `verify`, `banking`, `update`, `wallet`, etc.). |
| `brand_impersonation_score` | Semantic | Brand name appears in subdomain/path while registered domain is untrusted. |
| `suspicious_tld` | Reputation | High-abuse or disposable Top-Level Domains (`.xyz`, `.top`, `.icu`, `.buzz`, `.work`, etc.). |
| `has_punycode` | Obfuscation | Internationalized Domain Name (`xn--`) homoglyphs substituting lookalike Cyrillic glyphs. |
| `is_shortened` | Obfuscation | URL shortening domains (`bit.ly`, `tinyurl.com`, `t.co`) concealing landing destination. |
| `digit_count` | Statistical | Number of numerical digits in the URL. |
| `digit_ratio` | Statistical | Ratio of numeric characters to total URL length. |

---

## 📊 Held-Out Test Evaluation Results

Both detectors were evaluated on the held-out test partition (`data/test.csv`, $N = 234$ URLs, balanced 50% phishing / 50% legitimate, seed 42):

| Metric | Rule-Based Baseline | Random Forest ML | Explainable Hybrid |
| :--- | :---: | :---: | :---: |
| **F1 Score** | **0.9957** | **1.0000** | **1.0000** |
| **Precision** | 1.0000 | 1.0000 | 1.0000 |
| **Recall (Sensitivity)** | 0.9915 | 1.0000 | 1.0000 |
| **Accuracy** | 0.9957 | 1.0000 | 1.0000 |
| **False Positive Rate** | 0.0000 | 0.0000 | 0.0000 |
| **False Negative Rate** | 0.0085 | 0.0000 | 0.0000 |

### Confusion Matrix Comparison

#### 1. Rule-Based Baseline
- **True Positives (TP)**: 116 (Phishing correctly caught)
- **True Negatives (TN)**: 117 (Legitimate correctly cleared)
- **False Positives (FP)**: 0 (No benign URLs falsely blocked)
- **False Negatives (FN)**: 1 (Tricky phishing URL missed by rules)

#### 2. Random Forest ML Model
- **True Positives (TP)**: 117
- **True Negatives (TN)**: 117
- **False Positives (FP)**: 0
- **False Negatives (FN)**: 0

### Error Analysis & Diagnostic Takeaway
- **False Negative Case**: The heuristic rule engine missed `https://www.irs-tax-refund-claim-portal.xyz/portal/index.html?ref=email_warning` (score: 20/100) because "IRS" was not in the commercial banking brand dictionary and the URL used HTTPS, staying below the 35-point alarm threshold.
- In contrast, the **Random Forest ML classifier** recognized the multi-feature interaction between the `.xyz` TLD, high entropy path, and query parameter structure, correctly flagging it as High Risk.
- **Key Takeaway**: Rules provide instantaneous, verifiable audit trails for known attack patterns; ML generalizes across subtle, composite feature interactions. Combining both yields explainability with defense-in-depth.

---

## 🖥️ Demonstration Sequence

1. **Demonstrate a Legitimate Portal**:
   - Click the quick pill **`[Likely safe] Google Search Query`** or **`[Likely safe] Chase Bank Personal Banking`**.
   - Show how the gauge stays at 0-2, verdict displays **Likely safe** in green, and protective signals (genuine domain, HTTPS, low entropy) are highlighted.
2. **Demonstrate Subdomain Brand Impersonation**:
   - Click **`[High risk] PayPal Subdomain Deception`** (`http://paypal.com.verify-billing-center.xyz/index.html`).
   - Observe how the risk meter surges to **High risk** (score ~79).
   - Expand the **Signals Summary** to point out the **Critical Rule**: Target Brand Impersonation (brand "PayPal" in subdomain on untrusted `.xyz` domain).
3. **Demonstrate Direct IP Credential Harvesting**:
   - Click **`[High risk] Raw IP Address Credential Harvest`** (`http://185.220.101.5/login/webscr.php`).
   - Highlight the **Rule 01: Direct IP Address Host** penalty (+35 pts) and keyword detection.
4. **Demonstrate Obfuscation & Homographs**:
   - Click **`[High risk] Embedded '@' Symbol Trick`** or **`[High risk] Punycode Homograph Spoof`** (`http://xn--paypl-ora.com/webscr/login_submit.php`).
   - Show the Punycode detection and explain how Cyrillic lookalikes deceive human visual inspection.
5. **Show Model Benchmarks & Metrics**:
   - Switch to the **Model Benchmarks & Metrics** tab.
   - Walk through the comparative metrics table and the confusion matrices.
   - Point out the False Negative error case analysis to illustrate that no automated filter is infallible.
6. **Show System Architecture & Feature Intelligence**:
   - Switch to the **Feature Intelligence & Architecture** tab to show the end-to-end pipeline flow, 21-feature library, and threat model.

---

## ⚠️ Security Notice & Limitations

> [!WARNING]
> **Educational & Analytical Aid Notice:** This application is an educational and analytical aid designed to assist security analysts and users in identifying common deceptive patterns. **It cannot guarantee that an unflagged URL is 100% safe.** 
>
> **Known Limitations:**
> 1. **Compromised Legitimate Sites:** If an attacker compromises a legitimate, high-reputation domain (e.g. an unpatched WordPress blog) and hosts phishing content, lexical domain features will appear benign.
> 2. **Zero-Day Evasion:** Novel URL obfuscation or emerging Top-Level Domains not present in the training distribution may produce lower ML confidence.
> 3. **Live Webpage Sandbox Scope:** The optional live webpage inspection operates strictly in a restricted environment with SSRF blocks and no JavaScript execution; dynamic single-page phishing apps that require script rendering may not display form inputs to the static parser.
