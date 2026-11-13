"""Hourly features that use only what was known at that hour.

`causal=True` (the default) carries the last measurement FORWARD.
`causal=False` reproduces a classic mistake on purpose: gaps are also filled
BACKWARD, so a lab result drawn at hour 20 is visible to the model at hour 15.
It is here only to show how much a leak flatters the numbers.
"""
import numpy as np
import pandas as pd

from .synthetic import VARIABLES

HORIZON_HOURS = 6


def feature_groups():
    groups = {}
    for name, spec in VARIABLES.items():
        organ = spec[0]
        groups[name] = organ
        groups[name + "_change6h"] = organ
        groups[name + "_measured"] = organ
    return groups


def build_features(cohort, causal=True):
    cohort = cohort.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    by_patient = cohort.groupby("patient_id", sort=False)
    out = cohort[["patient_id", "hour", "onset_hour", "primary_organ"]].copy()
    for name, (_, mu, *_rest) in VARIABLES.items():
        filled = by_patient[name].ffill()
        if not causal:
            filled = filled.groupby(cohort["patient_id"], sort=False).bfill()
        filled = filled.fillna(mu)  # nothing measured yet: population typical value
        out[name] = filled
        previous = filled.groupby(cohort["patient_id"], sort=False).shift(HORIZON_HOURS)
        out[name + "_change6h"] = (filled - previous).fillna(0.0)
        out[name + "_measured"] = cohort[name].notna().astype(float)
    return out


def add_labels(features):
    """Label = sepsis onset within the next 6 hours: onset in (t, t+6].

    Hours at or after onset are dropped: the tool is meant to warn before
    onset, so it is neither trained nor scored on hours after it.
    """
    onset = features["onset_hour"]
    hours_to_onset = onset - features["hour"]
    features = features.copy()
    features["label"] = ((hours_to_onset > 0) & (hours_to_onset <= HORIZON_HOURS)).astype(int)
    keep = onset.isna() | (hours_to_onset > 0)
    return features[keep].reset_index(drop=True)
