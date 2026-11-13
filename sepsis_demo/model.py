"""A deliberately small model: standardise, then logistic regression.

Because the model is linear, each organ system's share of the score is just
the sum of (coefficient x standardised value) over that organ's features.
That is the simplest possible version of "which organ is driving the risk".
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import feature_groups


class RiskModel:
    def __init__(self):
        self.groups = feature_groups()
        self.columns = list(self.groups)
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=0.5, max_iter=2000)

    def fit(self, frame):
        x = self.scaler.fit_transform(frame[self.columns].to_numpy())
        self.clf.fit(x, frame["label"].to_numpy())
        return self

    def predict(self, frame):
        x = self.scaler.transform(frame[self.columns].to_numpy())
        return self.clf.predict_proba(x)[:, 1]

    def organ_contributions(self, frame):
        """Per-row contribution of each organ system to the log-odds."""
        x = self.scaler.transform(frame[self.columns].to_numpy())
        parts = x * self.clf.coef_[0]
        organs = sorted(set(self.groups.values()))
        out = {}
        for organ in organs:
            idx = [i for i, c in enumerate(self.columns) if self.groups[c] == organ]
            out[organ] = parts[:, idx].sum(axis=1)
        return out
