"""
Machine-learning phishing detector.

Architecture
------------
Two complementary views of the same URL are combined by soft voting:

  1. HistGradientBoosting on 41 engineered lexical / structural / semantic features
     (hyper-parameters chosen on the validation split);
  2. Logistic regression on character 3-5-grams of the URL (captures phishing-kit vocabulary
     that hand-made features miss).

The averaged probability is passed through Platt scaling fitted on the validation split, so
"P(phishing) = 0.8" really means roughly 80% of such URLs were phishing on held-out data.

Explanations are computed, not guessed:
  * feature groups are ablated one at a time (replaced by the benign-median value) and the change in
    P(phishing) is reported -> faithful, model-agnostic, sign-aware;
  * the character-gram model exposes the n-grams that pushed the score up the most.
"""

import csv
import os
import sys
import warnings
from typing import Dict, List, Optional

import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from features.extractor import BRAND_MASK, URLExtractor, get_feature_names, mask_brands

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "trained_model.joblib")
ARTIFACT_VERSION = 3

HGB_PARAMS = dict(max_depth=5, learning_rate=0.08, max_iter=500, min_samples_leaf=10,
                  l2_regularization=1.0, random_state=42)
TFIDF_PARAMS = dict(analyzer="char", ngram_range=(3, 5), min_df=4, sublinear_tf=True, max_features=60000)

# Feature groups used for explanations (single features are often redundant, e.g. several brand counters).
FEATURE_GROUPS: Dict[str, List[str]] = {
    "brand_impersonation": ["brand_in_host", "brand_in_path", "brand_impersonation_score", "typosquat_brand"],
    "credential_keywords": ["suspicious_keyword_count", "keyword_in_host"],
    "free_hosting_platform": ["free_hosting"],
    "domain_reputation_tld": ["suspicious_tld", "tld_length"],
    "ip_or_odd_port": ["has_ip_address", "has_port"],
    "obfuscation_tricks": ["at_symbol_count", "double_slash_redirect", "has_punycode", "is_shortened",
                           "https_token_in_host", "embedded_url"],
    "url_size": ["url_length", "domain_length", "path_length", "query_length", "longest_token_length"],
    "path_structure": ["path_depth", "query_param_count", "risky_file_extension"],
    "domain_structure": ["subdomain_count", "registered_label_length", "registered_hyphen_count", "hyphen_count",
                         "dot_count"],
    "randomness": ["url_entropy", "domain_entropy", "path_entropy", "max_consonant_run"],
    "digits": ["digit_count", "digit_ratio", "domain_digit_ratio"],
    "special_characters": ["underscore_count", "percent_encoded_count", "suspicious_char_count"],
    "protocol": ["has_https"],
}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def verdict_for(score: int) -> str:
    return "High risk" if score >= 65 else ("Suspicious" if score >= 35 else "Likely safe")


class MLDetector:
    """HistGradientBoosting + char-n-gram ensemble with calibrated probabilities and local explanations."""

    def __init__(self, model_path: str = MODEL_SAVE_PATH):
        self.model_path = model_path
        self.feature_names = get_feature_names()
        self.hgb: Optional[HistGradientBoostingClassifier] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.char_lr: Optional[LogisticRegression] = None
        self.platt = (1.0, 0.0)                 # slope, intercept on the logit scale
        self.benign_medians: Optional[np.ndarray] = None
        self.training_summary: Dict = {}
        self.is_trained = False
        if os.path.exists(self.model_path):
            self.load()

    # ------------------------------------------------------------------ io
    def load(self):
        try:
            data = joblib.load(self.model_path)
            if data.get("artifact_version") != ARTIFACT_VERSION or data.get("feature_names") != get_feature_names():
                raise ValueError("artifact does not match the current feature set")
            if data.get("sklearn_version") != sklearn.__version__:
                warnings.warn("scikit-learn version differs from the one used for training; retraining for safety.")
                raise ValueError("sklearn version mismatch")
        except Exception as exc:                # stale / incompatible artifact -> rebuild from the CSVs
            print(f"[!] Could not load trained model ({exc}). Retraining from data/train.csv ...", flush=True)
            self.train_from_dataset()
            return self
        self.hgb, self.tfidf, self.char_lr = data["hgb"], data["tfidf"], data["char_lr"]
        self.platt = tuple(data["platt"])
        self.benign_medians = data["benign_medians"]
        self.feature_names = data["feature_names"]
        self.training_summary = data.get("training_summary", {})
        self.is_trained = True
        return self

    def save(self):
        joblib.dump({
            "artifact_version": ARTIFACT_VERSION,
            "sklearn_version": sklearn.__version__,
            "hgb": self.hgb, "tfidf": self.tfidf, "char_lr": self.char_lr,
            "platt": self.platt, "benign_medians": self.benign_medians,
            "feature_names": self.feature_names, "training_summary": self.training_summary,
        }, self.model_path, compress=3)

    # ------------------------------------------------------------ training
    @staticmethod
    def _read(csv_path: str):
        urls, labels = [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                urls.append(row["url"])
                labels.append(int(row["label"]))
        X = np.array([URLExtractor.extract_feature_vector(u) for u in urls], dtype=np.float32)
        return urls, X, np.array(labels, dtype=np.int32)

    def _raw_probability(self, urls: List[str], X: np.ndarray) -> np.ndarray:
        p_hgb = self.hgb.predict_proba(X)[:, 1]
        p_chr = self.char_lr.predict_proba(self.tfidf.transform([mask_brands(u) for u in urls]))[:, 1]
        return (p_hgb + p_chr) / 2.0

    def _calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        a, b = self.platt
        return 1.0 / (1.0 + np.exp(-(a * _logit(p_raw) + b)))

    def train_from_dataset(self, train_csv: str = None, val_csv: str = None):
        train_csv = train_csv or os.path.join(PROJECT_DIR, "data", "train.csv")
        val_csv = val_csv or os.path.join(PROJECT_DIR, "data", "val.csv")
        print(f"[*] Training on {train_csv}", flush=True)

        urls_tr, X_tr, y_tr = self._read(train_csv)
        self.benign_medians = np.median(X_tr[y_tr == 0], axis=0)

        self.hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_tr, y_tr)
        self.tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        self.char_lr = LogisticRegression(C=4, max_iter=3000, class_weight="balanced").fit(
            self.tfidf.fit_transform([mask_brands(u) for u in urls_tr]), y_tr)
        self.is_trained = True

        summary = {"train_rows": int(len(y_tr)), "hgb_params": HGB_PARAMS}
        if os.path.exists(val_csv):
            urls_va, X_va, y_va = self._read(val_csv)
            p_raw = self._raw_probability(urls_va, X_va)
            # Platt scaling fitted on the validation split (test + OOD stay untouched)
            platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw).reshape(-1, 1), y_va)
            self.platt = (float(platt.coef_[0][0]), float(platt.intercept_[0]))
            p = self._calibrate(p_raw)
            summary["validation"] = {
                "roc_auc": round(float(roc_auc_score(y_va, p)), 4),
                "average_precision": round(float(average_precision_score(y_va, p)), 4),
                "f1": round(float(f1_score(y_va, p >= 0.5)), 4),
            }
            print(f"[+] Validation: {summary['validation']}", flush=True)
        self.training_summary = summary
        self.save()
        print(f"[+] Saved model to {self.model_path}", flush=True)
        return self

    # ----------------------------------------------------------- inference
    def _ensure(self):
        if not self.is_trained:
            self.train_from_dataset()

    def predict_proba_batch(self, urls: List[str]) -> np.ndarray:
        self._ensure()
        X = np.array([URLExtractor.extract_feature_vector(u) for u in urls], dtype=np.float32)
        return self._calibrate(self._raw_probability(urls, X))

    def predict(self, url_str: str) -> dict:
        self._ensure()
        feat_dict = URLExtractor.extract_features(url_str)
        x = np.array([feat_dict[k] for k in self.feature_names], dtype=np.float32).reshape(1, -1)
        p_hgb = float(self.hgb.predict_proba(x)[0, 1])
        vec = self.tfidf.transform([mask_brands(url_str)])
        p_chr = float(self.char_lr.predict_proba(vec)[0, 1])
        p_raw = (p_hgb + p_chr) / 2.0
        # Never claim certainty: a URL model cannot know a site's content, so keep the score inside 1..99.
        p_phish = float(np.clip(self._calibrate(np.array([p_raw]))[0], 0.005, 0.995))

        risk_score = int(round(p_phish * 100))
        confidence = float(min(max(abs(p_phish - 0.5) * 2, 0.10), 0.99))
        # The two sub-models disagreeing is real uncertainty, even when their average looks decisive.
        disagreement = abs(p_hgb - p_chr)
        confidence = max(0.10, confidence * (1.0 - 0.6 * disagreement))

        return {
            "model_name": "Gradient Boosting + Character-Pattern Ensemble",
            "risk_score": risk_score,
            "prediction": int(p_phish >= 0.5),
            "phishing_probability": round(p_phish, 4),
            "verdict": verdict_for(risk_score),
            "confidence": round(confidence, 3),
            "uncertainty": round(1.0 - confidence, 3),
            "submodels": {"feature_model": round(p_hgb, 4), "character_pattern_model": round(p_chr, 4)},
            "top_contributing_features": self._explain(x[0], feat_dict, p_phish, p_chr),
            "pattern_signals": self._pattern_signals(vec, p_chr),
            "all_features": feat_dict,
        }

    # --------------------------------------------------------- explanations
    def _explain(self, x: np.ndarray, feats: Dict[str, float], p_phish: float, p_chr: float) -> list:
        """Feature-group attribution for the feature model.

        For each group we measure two things and average them (a two-ordering Shapley approximation):
          knock-out : P(phishing) of the real URL minus P with this group reset to a typical benign value
          add-in    : P(phishing) with ONLY this group taken from the real URL (everything else benign)
                      minus P of a fully benign-looking URL
        Knock-out alone fails when several groups point the same way (the score stays saturated); add-in alone
        misses interactions. Their average is stable, sign-aware, and always computed from real model calls.
        """
        idx = {n: i for i, n in enumerate(self.feature_names)}
        groups = [(g, [idx[m] for m in members if m in idx]) for g, members in FEATURE_GROUPS.items()]
        groups = [(g, c) for g, c in groups if c]

        base = self.benign_medians.astype(np.float32)
        knock = np.repeat(x.reshape(1, -1), len(groups), axis=0)
        add = np.repeat(base.reshape(1, -1), len(groups), axis=0)
        for r, (_, cols) in enumerate(groups):
            knock[r, cols] = base[cols]
            add[r, cols] = x[cols]
        p_full = float(self.hgb.predict_proba(x.reshape(1, -1))[0, 1])
        p_base = float(self.hgb.predict_proba(base.reshape(1, -1))[0, 1])
        p_knock = self.hgb.predict_proba(knock)[:, 1]
        p_add = self.hgb.predict_proba(add)[:, 1]

        out = []
        for (gname, cols), pk, pa in zip(groups, p_knock, p_add):
            impact = float(0.5 * ((p_full - pk) + (pa - p_base)))     # > 0 : pushes towards phishing
            if abs(impact) < 0.02:
                continue
            out.append({
                "feature": gname,
                "label": GROUP_LABELS[gname],
                "value": float(round(float(np.mean([x[c] for c in cols])), 3)),
                "importance": round(abs(impact), 4),
                "impact": round(impact, 4),
                "description": describe_group(gname, feats),
                "direction": "RISK_INCREASING" if impact > 0.05 else ("PROTECTIVE" if impact < -0.05 else "NEUTRAL"),
            })
        out.sort(key=lambda r: abs(r["impact"]), reverse=True)
        return out[:6]

    def _pattern_signals(self, vec, p_chr: float, top_k: int = 4) -> list:
        """Character n-grams that pushed the character model towards 'phishing'. Shown only when it leans phishing."""
        if p_chr < 0.5:
            return []
        row = vec.tocoo()
        names = self.tfidf.get_feature_names_out()
        coefs = self.char_lr.coef_[0]
        contrib = sorted(((float(coefs[j] * v), names[j]) for j, v in zip(row.col, row.data)), reverse=True)
        picked: List[str] = []
        for c, gram in contrib:
            if c < 0.12 or len(picked) >= top_k:
                break
            g = gram.strip().replace(BRAND_MASK, "[brand]")
            if sum(ch.isalnum() for ch in g) >= 3 and g.isprintable() and not any(g in q or q in g for q in picked):
                picked.append(g)
        return [{"pattern": g, "kind": "risk"} for g in picked]


GROUP_LABELS = {
    "brand_impersonation": "Brand impersonation",
    "credential_keywords": "Credential-themed words",
    "free_hosting_platform": "Free hosting platform",
    "domain_reputation_tld": "Domain extension",
    "ip_or_odd_port": "IP address or odd port",
    "obfuscation_tricks": "Obfuscation tricks",
    "url_size": "URL size",
    "path_structure": "Path and query structure",
    "domain_structure": "Domain structure",
    "randomness": "Randomness of characters",
    "digits": "Digits",
    "special_characters": "Special characters",
    "protocol": "Connection type",
}


def describe_group(group: str, f: Dict[str, float]) -> str:
    g = lambda k: f.get(k, 0.0)  # noqa: E731
    if group == "brand_impersonation":
        if g("brand_in_host") or g("typosquat_brand"):
            return "The domain name mentions or imitates a well-known brand that does not own it."
        if g("brand_in_path"):
            return "A well-known brand name appears in the path of an unrelated domain."
        return "No brand impersonation detected."
    if group == "credential_keywords":
        n = int(g("suspicious_keyword_count"))
        return f"{n} sign-in or payment word(s) such as login, verify or account appear in the address." if n else \
            "No sign-in or payment themed words in the address."
    if group == "free_hosting_platform":
        return "Hosted on a free site-building platform that anyone can use, which phishing campaigns often abuse." \
            if g("free_hosting") else "Not hosted on a free site-building platform."
    if group == "domain_reputation_tld":
        return "Uses a domain extension that is heavily abused for throwaway sites." if g("suspicious_tld") else \
            "Uses an ordinary domain extension."
    if group == "ip_or_odd_port":
        if g("has_ip_address"):
            return "The address is a raw IP number instead of a domain name."
        return "Uses an unusual network port." if g("has_port") else "Normal host and port."
    if group == "obfuscation_tricks":
        tricks = [t for k, t in (("at_symbol_count", "an '@' sign"), ("double_slash_redirect", "a '//' redirect"),
                                 ("has_punycode", "look-alike (punycode) characters"), ("is_shortened", "a link shortener"),
                                 ("https_token_in_host", "the word 'https' inside the domain"),
                                 ("embedded_url", "another URL embedded in the path")) if g(k)]
        return "Contains " + ", ".join(tricks) + "." if tricks else "No obfuscation tricks found."
    if group == "url_size":
        return f"The address is {int(g('url_length'))} characters long (longest word: {int(g('longest_token_length'))})."
    if group == "path_structure":
        return f"Path depth {int(g('path_depth'))}, {int(g('query_param_count'))} query parameter(s)" + \
            (", risky file extension." if g("risky_file_extension") else ".")
    if group == "domain_structure":
        return f"{int(g('subdomain_count'))} subdomain level(s), {int(g('registered_hyphen_count'))} hyphen(s) in the domain name."
    if group == "randomness":
        return f"Character randomness is {g('url_entropy'):.2f} bits (longest run of consonants: {int(g('max_consonant_run'))})."
    if group == "digits":
        return f"Digits make up {g('digit_ratio') * 100:.0f}% of the address."
    if group == "special_characters":
        return f"{int(g('suspicious_char_count'))} special characters and {int(g('percent_encoded_count'))} percent-encoded sequences."
    if group == "protocol":
        return "Uses HTTPS (encryption only; many phishing sites have it too)." if g("has_https") else \
            "Uses plain HTTP, so the connection is not encrypted."
    return group


# Singleton
_model_instance = None


def get_ml_detector() -> MLDetector:
    global _model_instance
    if _model_instance is None:
        _model_instance = MLDetector()
    return _model_instance
