"""Synthetic ICU stays. Every number here is invented by a random generator.

No patient data, and no data from MIMIC-IV, eICU or PhysioNet, is used or
needed. The generator is deliberately simple: it exists so the rest of the
pipeline can be run and inspected end to end, not to imitate real physiology.
"""
import numpy as np
import pandas as pd

# variable: (organ system, baseline mean, baseline sd, hourly noise sd,
#            shift at sepsis onset when this organ is involved,
#            probability the value is measured in a given hour)
VARIABLES = {
    "HR":         ("cardiovascular", 82, 10, 4.0, +22, 0.95),
    "MAP":        ("cardiovascular", 82, 8, 4.0, -16, 0.95),
    "Lactate":    ("cardiovascular", 1.3, 0.4, 0.15, +1.8, 0.12),
    "Resp":       ("respiratory", 17, 3, 1.5, +7, 0.95),
    "SpO2":       ("respiratory", 97, 1.5, 0.8, -4, 0.95),
    "Temp":       ("inflammatory", 37.0, 0.4, 0.15, +1.2, 0.50),
    "WBC":        ("inflammatory", 9.0, 2.5, 0.6, +6.0, 0.08),
    "Creatinine": ("renal", 0.95, 0.25, 0.05, +0.9, 0.08),
    "UrineOutput": ("renal", 75, 20, 15, -40, 0.90),
    "Platelets":  ("hematologic", 230, 60, 10, -80, 0.08),
    "Bilirubin":  ("hepatic", 0.8, 0.3, 0.05, +1.4, 0.06),
    "GCS":        ("neurologic", 14.6, 0.6, 0.3, -2.0, 0.30),
}
ORGANS = sorted({v[0] for v in VARIABLES.values()})
PRIMARY_ORGANS = ["cardiovascular", "respiratory", "renal", "inflammatory"]


def make_cohort(n_patients=1500, sepsis_rate=0.15, seed=7):
    """Return one row per patient-hour.

    Columns: patient_id, hour, one column per variable (NaN when not measured
    that hour), onset_hour (NaN for patients who never develop sepsis) and
    primary_organ (the organ system the generator made deteriorate first).
    """
    rng = np.random.default_rng(seed)
    frames = []
    for pid in range(n_patients):
        length = int(rng.integers(24, 97))
        septic = rng.random() < sepsis_rate
        onset = int(rng.integers(12, length - 2)) if septic else None
        primary = rng.choice(PRIMARY_ORGANS) if septic else None
        ramp_hours = float(rng.uniform(4, 12))
        # How clearly this patient's sepsis shows in the numbers. Some barely do.
        severity = float(rng.uniform(0.35, 1.0))
        # A non-septic "noisy" patient: unwell for another reason, which is
        # what produces false alarms.
        noisy = (not septic) and rng.random() < 0.20
        noisy_start = int(rng.integers(6, length - 4)) if noisy else None
        noisy_organ = rng.choice(PRIMARY_ORGANS) if noisy else None

        hours = np.arange(length)
        data = {"patient_id": pid, "hour": hours}
        for name, (organ, mu, sd, noise, shift, p_obs) in VARIABLES.items():
            base = rng.normal(mu, sd)
            walk = np.cumsum(rng.normal(0, noise * 0.25, length))
            value = base + walk + rng.normal(0, noise, length)
            if septic:
                weight = 1.0 if organ == primary else 0.35
                progress = np.clip((hours - (onset - ramp_hours)) / ramp_hours, 0, 1.3)
                value = value + shift * severity * weight * progress
            if noisy and organ == noisy_organ:
                bump = np.clip((hours - noisy_start) / 6.0, 0, 1) * np.clip((noisy_start + 18 - hours) / 6.0, 0, 1)
                value = value + shift * 0.7 * bump
            observed = rng.random(length) < p_obs
            data[name] = np.where(observed, value, np.nan)
        frame = pd.DataFrame(data)
        frame["onset_hour"] = float(onset) if septic else np.nan
        frame["primary_organ"] = primary if septic else "none"
        frames.append(frame)
    cohort = pd.concat(frames, ignore_index=True)
    cohort["GCS"] = cohort["GCS"].clip(3, 15).round()
    cohort["SpO2"] = cohort["SpO2"].clip(70, 100)
    cohort["UrineOutput"] = cohort["UrineOutput"].clip(lower=0)
    for col in ("Lactate", "WBC", "Creatinine", "Platelets", "Bilirubin"):
        cohort[col] = cohort[col].clip(lower=0.1)
    return cohort
