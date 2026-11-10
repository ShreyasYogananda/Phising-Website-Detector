"""
Machine Learning Phishing Detector & Feature Explainer
PhishGuard Supervised Random Forest Classifier & Local Feature Explainer.

Implements an ensemble Random Forest classifier trained on URL lexical, structural, and semantic features.
Provides calibrated probability estimation, decision uncertainty, and localized feature contribution explanations.
"""

import csv
import os
import sys
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from features.extractor import URLExtractor, extract_features, extract_feature_vector, get_feature_names

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "trained_model.joblib")

class MLDetector:
    """
    Random Forest Machine Learning model for phishing URL classification
    with localized feature-level explainability.
    """

    def __init__(self, model_path: str = MODEL_SAVE_PATH):
        self.model_path = model_path
        self.feature_names = get_feature_names()
        self.model = None
        self.feature_importances_ = None
        self.baseline_feature_means = None
        self.baseline_feature_stds = None
        self.is_trained = False

        if os.path.exists(self.model_path):
            self.load()

    def load(self):
        """Loads trained model and distribution metadata from disk."""
        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.feature_importances_ = data.get("feature_importances", None)
        self.baseline_feature_means = data.get("baseline_feature_means", None)
        self.baseline_feature_stds = data.get("baseline_feature_stds", None)
        self.is_trained = True
        return self

    def save(self):
        """Saves trained model and distribution metadata to disk."""
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances_,
            "baseline_feature_means": self.baseline_feature_means,
            "baseline_feature_stds": self.baseline_feature_stds
        }, self.model_path)

    def train_from_dataset(self, train_csv: str = None, val_csv: str = None):
        """
        Extracts features from train.csv and trains the Random Forest classifier.
        Validates against val.csv and saves the trained model artifact.
        """
        if train_csv is None:
            train_csv = os.path.join(PROJECT_DIR, "data", "train.csv")
        if val_csv is None:
            val_csv = os.path.join(PROJECT_DIR, "data", "val.csv")

        print(f"[*] Training ML model using training set: {train_csv}")

        # Load training data
        X_train_list, y_train_list = [], []
        with open(train_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vec = extract_feature_vector(row["url"])
                X_train_list.append(vec)
                y_train_list.append(int(row["label"]))

        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.int32)

        # Compute baselines for explainability
        self.baseline_feature_means = np.mean(X_train, axis=0)
        self.baseline_feature_stds = np.std(X_train, axis=0) + 1e-6

        # Initialize and fit Random Forest (single-process for Windows compatibility)
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=None
        )
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        self.is_trained = True

        # Validation set evaluation
        if os.path.exists(val_csv):
            X_val_list, y_val_list = [], []
            with open(val_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vec = extract_feature_vector(row["url"])
                    X_val_list.append(vec)
                    y_val_list.append(int(row["label"]))
            X_val = np.array(X_val_list, dtype=np.float32)
            y_val = np.array(y_val_list, dtype=np.int32)
            
            val_preds = self.model.predict(X_val)
            acc = accuracy_score(y_val, val_preds)
            f1 = f1_score(y_val, val_preds)
            prec = precision_score(y_val, val_preds)
            rec = recall_score(y_val, val_preds)
            print(f"[+] Validation Results: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

        self.save()
        print(f"[+] ML model successfully saved to: {self.model_path}")
        return self

    def predict(self, url_str: str) -> dict:
        """
        Runs ML prediction and generates localized feature attribution signals.
        """
        if not self.is_trained:
            if os.path.exists(self.model_path):
                self.load()
            else:
                self.train_from_dataset()

        feat_dict = extract_features(url_str)
        feat_vector = np.array([feat_dict[k] for k in self.feature_names], dtype=np.float32).reshape(1, -1)

        # Probabilities: [P(legitimate), P(phishing)]
        probs = self.model.predict_proba(feat_vector)[0]
        p_phish = float(probs[1])
        predicted_class = int(p_phish >= 0.5)

        # Risk score 0 to 100
        risk_score = int(round(p_phish * 100))

        # Verdict
        if risk_score >= 65:
            verdict = "High risk"
        elif risk_score >= 35:
            verdict = "Suspicious"
        else:
            verdict = "Likely safe"

        # Confidence & Uncertainty calculation:
        # Distance from the 0.5 decision boundary:
        # Near 0.5 => high uncertainty, low confidence.
        # Near 1.0 or 0.0 => low uncertainty, high confidence.
        confidence = float(min(max(abs(p_phish - 0.5) * 2, 0.10), 0.99))
        uncertainty = float(round(1.0 - confidence, 3))

        # Localized feature contributions
        feature_contributions = self._compute_feature_contributions(feat_vector[0], p_phish)

        return {
            "model_name": "Random Forest ML Classifier",
            "risk_score": risk_score,
            "prediction": predicted_class,
            "phishing_probability": round(p_phish, 4),
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "uncertainty": uncertainty,
            "top_contributing_features": feature_contributions,
            "all_features": feat_dict
        }

    def _compute_feature_contributions(self, feature_values: np.ndarray, p_phish: float) -> list:
        """
        Calculates localized feature attributions for a single URL:
        Measures how each feature's deviation from the baseline training distribution
        multiplied by model feature importance impacts the risk verdict.
        """
        contributions = []
        if self.feature_importances_ is None:
            return []

        means = self.baseline_feature_means if self.baseline_feature_means is not None else np.zeros_like(feature_values)
        stds = self.baseline_feature_stds if self.baseline_feature_stds is not None else np.ones_like(feature_values)

        for i, name in enumerate(self.feature_names):
            val = float(feature_values[i])
            importance = float(self.feature_importances_[i])
            z_score = float((val - float(means[i])) / float(stds[i]))

            # Contribution impact score
            impact = float(z_score * importance)

            # Human-friendly descriptions
            label_desc = self._get_feature_description(name, val)

            contributions.append({
                "feature": str(name),
                "value": float(round(val, 3)),
                "importance": float(round(importance, 4)),
                "impact": float(round(impact, 4)),
                "description": str(label_desc),
                "direction": "RISK_INCREASING" if impact > 0.05 else ("PROTECTIVE" if impact < -0.05 else "NEUTRAL")
            })

        # Sort by absolute impact descending
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions[:6]

    @staticmethod
    def _get_feature_description(feature_name: str, value: float) -> str:
        descs = {
            "brand_impersonation_score": f"Brand impersonation score is {int(value)} (presence of known corporate brand on untrusted domain)",
            "has_ip_address": "Host uses a raw numerical IP address instead of a domain name" if value > 0 else "Domain uses standard named hostname",
            "suspicious_keyword_count": f"Detected {int(value)} sensitive credential/banking action keywords in URL",
            "subdomain_count": f"Contains {int(value)} subdomain levels",
            "suspicious_tld": "Resides on a high-abuse/suspicious Top Level Domain" if value > 0 else "Uses standard reputable Top Level Domain",
            "url_entropy": f"URL character entropy is {value:.2f} (measure of randomness and obfuscation)",
            "url_length": f"Total URL length is {int(value)} characters",
            "domain_length": f"Domain name length is {int(value)} characters",
            "path_length": f"Path length is {int(value)} characters",
            "at_symbol_count": "Contains '@' character which can obscure actual destination host" if value > 0 else "No '@' character present",
            "double_slash_redirect": "Contains '//' redirect pattern in path" if value > 0 else "Standard path formatting",
            "has_https": "Uses encrypted HTTPS protocol" if value > 0 else "Insecure plain HTTP protocol (elevates risk)",
            "has_punycode": "Uses Punycode 'xn--' homoglyph encoding" if value > 0 else "Standard ASCII domain characters",
            "is_shortened": "URL shortener detected (cloaks true landing page)" if value > 0 else "Direct non-shortened URL",
            "hyphen_count": f"Contains {int(value)} hyphens (frequent in typosquatted domains)",
            "dot_count": f"Contains {int(value)} dots",
            "digit_count": f"Contains {int(value)} numeric digits",
            "digit_ratio": f"Numeric characters make up {value*100:.1f}% of URL"
        }
        return descs.get(feature_name, f"{feature_name} = {value}")

# Singleton model instance
_model_instance = None

def get_ml_detector() -> MLDetector:
    global _model_instance
    if _model_instance is None:
        _model_instance = MLDetector()
    return _model_instance
