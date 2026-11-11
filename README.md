# PhishGuard AI - Explainable Phishing Risk Checker

Paste a web address and PhishGuard tells you whether it looks like phishing, **why**, and **which part of the address you would really be visiting**. It combines a transparent rule engine with a machine-learning ensemble, and it reports honest accuracy numbers measured on data the model never saw.

## What it does

- **Two detectors, one verdict.** 16 auditable rules plus a gradient-boosting model (41 URL features) blended with a character n-gram model. Final score = 80% model + 20% rules, with a critical-rule floor and a trusted-brand-domain cap. Verdicts: *Likely safe* (< 35), *Suspicious* (35-64), *High risk* (65+).
- **Explains itself.** Every verdict lists the rules that fired, the features that pushed the model's score up or down, and a URL anatomy strip that highlights the registered domain, `@` disguises and brand-in-subdomain tricks.
- **Honest evaluation.** The benchmark tab shows precision, recall, F1, false-alarm rate (with 95% confidence intervals), confusion matrices, error rates per data source, and real mistakes with root causes, on two test sets: held-out *domains* and *older phishing campaigns*.
- **Optional live-page inspection** in an SSRF-hardened sandbox (see Security).

## Results (held-out test, 996 links)

| Detector | F1 | Precision | Recall | False-alarm rate |
| --- | --- | --- | --- | --- |
| Rule engine | 0.66 | 0.69 | 0.63 | 29.5% |
| ML ensemble | 0.96 | 0.97 | 0.96 | 3.3% |
| Hybrid (shown in app) | 0.96 | 0.96 | 0.96 | 4.5% |

On older campaigns the hybrid scores F1 0.97. See the *How well it works* tab or `evaluation/evaluation_results.json` for everything, including the limitations below.

## What changed in v2

| Area | Before | Now |
| --- | --- | --- |
| Data | ~1,550 template-generated URLs labelled as PhishTank/Tranco; model scored 100% (meaningless) | ~6,500 URLs: real phishing from Phishing.Database, real long-tail benign URLs, plus curated benign; splits grouped by registered domain; separate older-campaign test |
| Bias controls | none | per-host / per-domain / per-platform caps; scheme and "has path" decorrelated from the label |
| Features | 21 | 41 (public-suffix aware, token-based brand matching, typosquat/homoglyph, embedded URLs, entropy, ...) |
| Model | Random Forest | Gradient boosting + character n-gram model, calibrated on validation data |
| Rules | 11 | 16, thresholds tuned on validation data |
| Page sandbox | literal-IP check only | DNS-resolution check on every hop, manual redirect validation, scheme allow-list |
| API | single endpoint | input validation, `/api/batch`, `/api/health`, error handling |
| UI | dark SOC dashboard, out of sync with the model | rebuilt light/dark-aware "evidence report" UI with URL anatomy, accessible tabs, mobile layout |
| Tests | 7 | 20 (API, detection quality on safe and phishing URLs, garbage input, SSRF) |

## Limitations (please read)

- The legitimate side of the corpus is partly synthetic (well-known domains with generated paths) and partly an older public corpus, so false-positive rates on today's real traffic may be higher than reported.
- URL-only analysis cannot see page content. Phishing on a compromised legitimate site, or via an open redirect on a trusted domain, can look normal.
- Phishing infrastructure changes quickly. Rebuild the dataset and retrain regularly.

## Project layout

```text
app.py                 FastAPI app, API routes, feature dictionary
run.py                 Starts the server on http://127.0.0.1:8000
train_model.py         Trains models/trained_model.joblib from data/train.csv (calibrated on val.csv)
evaluate_cli.py        Re-runs the benchmark, writes evaluation/evaluation_results.json
data/dataset_builder.py  Downloads feeds into data/cache/ and builds train/val/test/ood_test.csv
features/extractor.py  41 features + URL anatomy
models/                rule_based.py, ml_detector.py, explainer.py (score blending), trained_model.joblib
evaluation/evaluator.py  Metrics, confidence intervals, per-source errors, error mining
analyzer/page_analyzer.py  Optional isolated page inspection
templates/, static/    UI
tests/test_pipeline.py 20 automated tests
```

## Run it

```bash
pip install -r requirements.txt
python run.py                      # open http://127.0.0.1:8000
```

Other commands:

```bash
python -m pytest -q tests          # or: python -m unittest discover -s tests
python evaluate_cli.py             # recompute the benchmark
python data/dataset_builder.py     # re-download feeds and rebuild the splits (~75 MB, needs internet)
python train_model.py              # retrain after rebuilding the data
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/analyze` | `{"url": "...", "enable_page_analysis": false}` returns verdict, score, evidence, anatomy |
| POST | `/api/batch` | `{"urls": [...]}` (max 25) returns verdict and score per URL |
| GET | `/api/evaluation` | Stored benchmark |
| POST | `/api/evaluation/recalculate` | Recompute the benchmark from the CSVs |
| GET | `/api/samples` | Demo URLs |
| GET | `/api/features/dictionary` | All 41 features |
| GET | `/api/health` | Status, version, model info |

Interactive docs: `/docs`.

## Security notes

The optional page inspection fetches a user-supplied URL, so it is restricted: http/https only; every hop (including redirects) is checked against loopback, private, link-local, carrier-grade NAT and reserved ranges, both as IP literals and after DNS resolution; 2.5 s timeout, 3 redirects maximum, 500 KB read cap; scripts are never executed. A DNS-rebinding race between lookup and connect remains theoretically possible, so deploy the service without access to sensitive internal networks. The UI builds all dynamic content with `textContent`, so URLs and API responses are never interpreted as HTML.

## Reference

Phishing detection through URL-based hybrid machine learning, IEEE Access, 2023.
