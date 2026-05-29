"""
Phishing Email Detection Model
Uses Scikit-learn with TF-IDF + engineered URL/keyword features
"""

import re
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# 1. SAMPLE DATASET
# ─────────────────────────────────────────────

EMAILS = [
    # --- PHISHING (label=1) ---
    ("URGENT: Your account has been suspended! Click here http://secure-login.xyz/verify to restore access immediately.", 1),
    ("Congratulations! You've won $1,000,000. Claim now at http://prizes-win.ru/claim?id=abc123", 1),
    ("Dear Customer, verify your PayPal account http://paypal-secure.tk/login or it will be closed within 24hrs.", 1),
    ("Your Apple ID is locked. Visit http://appleid-verify.ml/unlock immediately to avoid permanent suspension.", 1),
    ("Bank Alert: Suspicious activity detected. Login at http://bank-secure.xyz/confirm to verify your identity.", 1),
    ("FREE iPhone 15 giveaway! Limited time offer. Click http://free-iphone-giveaway.net/win to claim yours!", 1),
    ("Your Netflix subscription has expired. Update payment at http://netflix-billing.ml/update NOW!", 1),
    ("IRS Notice: You owe $3,450 in unpaid taxes. Pay immediately at http://irs-payment.xyz/pay to avoid arrest.", 1),
    ("WINNER! Amazon selected you for a $500 gift card. Confirm at http://amazon-rewards.tk/claim?user=you", 1),
    ("Urgent security alert: Someone accessed your Google account from Russia. Verify at http://google-protect.ml", 1),
    ("Your package could not be delivered. Reschedule at http://fedex-track.xyz/reschedule?id=9283720", 1),
    ("Microsoft Support: Your computer has a virus! Call 1-800-555-0199 or click http://microsoftfix.net now!", 1),
    ("Verify your identity at http://chase-secure.ml/login or your account will be permanently disabled today.", 1),
    ("You have a pending refund of $892. Submit your bank details at http://refund-irs.xyz/claim?ref=TA29", 1),
    ("FINAL WARNING: Your email will be deleted unless you login at http://mail-verify.tk/keep within 12 hours!", 1),
    ("Dear user, your password expires today. Reset immediately http://password-reset.ml/new or lose access.", 1),
    ("Crypto alert: Double your Bitcoin! Send 0.1 BTC to http://bitcoin-double.xyz and get 0.2 back instantly!", 1),
    ("Your social security number has been suspended. Call 1-800-555-1234 or visit http://ssa-verify.ml/ssn", 1),
    ("HSBC: We've noticed unusual login. Secure your account now at http://hsbc-protect.xyz/login?token=a1b2", 1),
    ("Lucky draw: You are today's winner! Claim your prize at http://lucky-winners.ru/claim?session=xyz789", 1),

    # --- LEGITIMATE (label=0) ---
    ("Hi Sarah, just following up on the project timeline we discussed last Tuesday. Let me know if you need anything.", 0),
    ("Your Amazon order #123-456789 has shipped and will arrive by Thursday. Track your package in the app.", 0),
    ("Team meeting is scheduled for Wednesday at 2 PM in Conference Room B. Please confirm your attendance.", 0),
    ("Monthly newsletter: Check out our latest blog posts on Python tips, design patterns, and cloud best practices.", 0),
    ("Your GitHub pull request #247 has been reviewed. Two comments from @alice need your attention.", 0),
    ("Invoice #INV-2024-089 from Acme Corp is ready. Total due: $1,200. Payment due by January 15, 2025.", 0),
    ("Reminder: Your dentist appointment is tomorrow at 10:30 AM. Please call if you need to reschedule.", 0),
    ("Welcome to the team! Your onboarding documents are attached. HR will call you on Monday morning.", 0),
    ("Quarterly report attached. Revenue was up 12% compared to Q3. See the full analysis inside.", 0),
    ("Your flight to New York on Dec 20 is confirmed. Check-in opens 24 hours before departure.", 0),
    ("Hi, I wanted to share this interesting article about machine learning trends: https://arxiv.org/abs/2024.01234", 0),
    ("Please review the attached contract and let us know if you have any questions before signing.", 0),
    ("Happy birthday! Hope you have a wonderful day. Looking forward to seeing you at the celebration tonight.", 0),
    ("Your subscription renewal is coming up on February 1. Visit your account page to manage your plan.", 0),
    ("Thank you for your support ticket #78432. Our team will respond within 24 business hours.", 0),
    ("The weekly standup notes are posted in Confluence. Key action items highlighted in yellow.", 0),
    ("Your salary has been deposited to your account. Please check your pay stub for breakdown details.", 0),
    ("Conference registration is now open. Early bird pricing ends March 31. Use code SPRING2024.", 0),
    ("Heads up: the server maintenance window is Saturday 2–4 AM. Brief downtime expected for all services.", 0),
    ("Great meeting everyone today! I've shared the slides and recording in our shared Google Drive folder.", 0),
]


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

PHISHING_KEYWORDS = [
    "urgent", "suspended", "verify", "click here", "won", "winner", "prize",
    "claim", "immediately", "warning", "limited time", "expire", "locked",
    "confirm", "update payment", "unusual activity", "secure your", "access restored",
    "free", "giveaway", "arrest", "owe", "refund", "final warning", "deleted",
    "crypto", "bitcoin", "double your", "lucky", "congratulations"
]

class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts hand-crafted numerical features from email text."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(text) for text in X])

    def _extract(self, text: str) -> list:
        text_lower = text.lower()
        urls = re.findall(r'http[s]?://\S+', text_lower)

        # URL-based features
        has_url = int(len(urls) > 0)
        url_count = len(urls)
        suspicious_tld = int(any(
            re.search(r'\.(xyz|ml|tk|ru|cf|ga|gq|click|loan|win|download)(/|$)', u)
            for u in urls
        ))
        long_url = int(any(len(u) > 60 for u in urls))
        ip_in_url = int(any(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', u) for u in urls))
        url_has_at = int(any('@' in u for u in urls))
        url_has_dash_in_domain = int(any(
            re.search(r'://[^/]*-[^/]*/', u) for u in urls
        ))
        redirect_params = int(any(re.search(r'\?.*=', u) for u in urls))

        # Keyword features
        keyword_hits = sum(1 for kw in PHISHING_KEYWORDS if kw in text_lower)
        exclamation_count = text.count('!')
        all_caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
        has_dollar = int('$' in text)
        has_phone = int(bool(re.search(r'\b1-\d{3}-\d{3}-\d{4}\b', text)))

        # Structural features
        word_count = len(text.split())
        avg_word_len = np.mean([len(w) for w in text.split()]) if text.split() else 0
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)

        return [
            has_url, url_count, suspicious_tld, long_url, ip_in_url,
            url_has_at, url_has_dash_in_domain, redirect_params,
            keyword_hits, exclamation_count, all_caps_words,
            has_dollar, has_phone, word_count, avg_word_len, digit_ratio
        ]


# ─────────────────────────────────────────────
# 3. PIPELINE
# ─────────────────────────────────────────────

class TextSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X


def build_pipeline():
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
        sublinear_tf=True,
        stop_words='english'
    )

    features = FeatureUnion([
        ("tfidf", Pipeline([
            ("selector", TextSelector()),
            ("tfidf", tfidf),
        ])),
        ("engineered", Pipeline([
            ("selector", TextSelector()),
            ("features", EmailFeatureExtractor()),
            ("scaler", StandardScaler()),
        ])),
    ])

    pipeline = Pipeline([
        ("features", features),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight="balanced", random_state=42
        )),
    ])
    return pipeline


# ─────────────────────────────────────────────
# 4. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def train_and_evaluate():
    df = pd.DataFrame(EMAILS, columns=["text", "label"])
    X, y = df["text"].values, df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred,
                                   target_names=["Safe", "Phishing"],
                                   output_dict=True)

    # Cross-val
    cv_scores = cross_val_score(
        build_pipeline(), X, y, cv=5, scoring='accuracy'
    )

    results = {
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "confusion_matrix": cm,
        "report": report,
        "test_predictions": [
            {
                "text": t,
                "true_label": int(tl),
                "predicted_label": int(pl),
                "phishing_prob": round(pp, 4)
            }
            for t, tl, pl, pp in zip(X_test, y_test, y_pred, y_proba)
        ]
    }
    return model, results


if __name__ == "__main__":
    print("Training phishing detection model...")
    model, results = train_and_evaluate()

    print(f"\n{'='*50}")
    print(f"  PHISHING EMAIL DETECTION MODEL RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy :  {results['accuracy']*100:.1f}%")
    print(f"  ROC-AUC  :  {results['auc']:.4f}")
    print(f"  CV Score :  {results['cv_mean']*100:.1f}% ± {results['cv_std']*100:.1f}%")

    print(f"\n  Confusion Matrix:")
    cm = results['confusion_matrix']
    print(f"              Predicted Safe  Predicted Phishing")
    print(f"  Actual Safe      {cm[0][0]}               {cm[0][1]}")
    print(f"  Actual Phish     {cm[1][0]}               {cm[1][1]}")

    print(f"\n  Classification Report:")
    r = results['report']
    for cls in ['Safe', 'Phishing']:
        d = r[cls]
        print(f"  {cls:10s} | P={d['precision']:.2f}  R={d['recall']:.2f}  F1={d['f1-score']:.2f}")

    print(f"\n  Sample Predictions:")
    for p in results['test_predictions']:
        label = "🚨 PHISHING" if p['predicted_label'] == 1 else "✅ SAFE"
        correct = "✓" if p['predicted_label'] == p['true_label'] else "✗"
        print(f"  [{correct}] {label} ({p['phishing_prob']*100:.0f}%) — {p['text'][:70]}...")
    print()
    print(json.dumps(results, indent=2))
