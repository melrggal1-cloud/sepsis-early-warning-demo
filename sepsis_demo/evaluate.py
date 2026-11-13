"""Alert-level evaluation: what a ward would actually experience."""
import numpy as np
from sklearn.metrics import roc_auc_score

SNOOZE_HOURS = 6  # after an alert, stay quiet for this long


def hourly_auroc(frame, score):
    return float(roc_auc_score(frame["label"], score))


def simulate_alerts(frame, score, threshold):
    """Fire when the score crosses the threshold, then snooze.

    An alert is 'true' if sepsis onset follows within 12 hours.
    Returns ward-level numbers.
    """
    frame = frame.assign(score=score)
    n_true = n_false = 0
    caught = 0
    septic = 0
    lead_times = []
    hours_total = 0
    for _, stay in frame.groupby("patient_id", sort=False):
        hours = stay["hour"].to_numpy()
        s = stay["score"].to_numpy()
        onset = stay["onset_hour"].iloc[0]
        hours_total += len(hours)
        is_septic = not np.isnan(onset)
        septic += int(is_septic)
        quiet_until = -1
        first_true = None
        for h, value in zip(hours, s):
            if h < quiet_until or value < threshold:
                continue
            quiet_until = h + SNOOZE_HOURS
            if is_septic and 0 < onset - h <= 12:
                n_true += 1
                if first_true is None:
                    first_true = onset - h
            else:
                n_false += 1
        if first_true is not None:
            caught += 1
            lead_times.append(first_true)
    days = hours_total / 24.0
    return {
        "threshold": float(threshold),
        "alerts_per_100_patient_days": 100 * (n_true + n_false) / days,
        "false_alerts_per_100_patient_days": 100 * n_false / days,
        "share_of_alerts_that_were_true": n_true / max(1, n_true + n_false),
        "share_of_sepsis_patients_warned_before_onset": caught / max(1, septic),
        "median_warning_hours": float(np.median(lead_times)) if lead_times else None,
        "sepsis_patients": int(septic),
        "patient_days": days,
    }


def pick_threshold(frame, score, max_false_per_100_days=30.0):
    """Lowest threshold whose false-alert rate fits the agreed alarm budget.

    Chosen on validation patients only, never on the test patients.
    """
    best = None
    for threshold in np.round(np.arange(0.95, 0.01, -0.01), 2):
        result = simulate_alerts(frame, score, threshold)
        if result["false_alerts_per_100_patient_days"] <= max_false_per_100_days:
            best = threshold
        else:
            break
    return float(best if best is not None else 0.95)
