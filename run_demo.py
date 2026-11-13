"""Run the whole demonstration on synthetic data and write outputs/.

    python run_demo.py

Everything printed and saved comes from the random generator in
sepsis_demo/synthetic.py. None of it describes real patients or says
anything about how a model would perform in a hospital.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sepsis_demo.evaluate import hourly_auroc, pick_threshold, simulate_alerts
from sepsis_demo.features import add_labels, build_features
from sepsis_demo.model import RiskModel
from sepsis_demo.synthetic import VARIABLES, make_cohort

OUT = Path(__file__).parent / "outputs"
SEED = 7
UNITS = {"HR": "bpm", "MAP": "mmHg", "Lactate": "mmol/L", "Resp": "/min", "SpO2": "%",
         "Temp": "°C", "WBC": "10⁹/L", "Creatinine": "mg/dL", "UrineOutput": "mL/h",
         "Platelets": "10⁹/L", "Bilirubin": "mg/dL", "GCS": ""}


def split_patients(ids, seed):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)
    a, b = int(0.6 * len(ids)), int(0.8 * len(ids))
    return set(ids[:a]), set(ids[a:b]), set(ids[b:])


def run(cohort, causal, splits):
    train_ids, val_ids, test_ids = splits
    table = add_labels(build_features(cohort, causal=causal))
    train = table[table.patient_id.isin(train_ids)]
    val = table[table.patient_id.isin(val_ids)]
    test = table[table.patient_id.isin(test_ids)]
    model = RiskModel().fit(train)
    threshold = pick_threshold(val, model.predict(val))
    score = model.predict(test)
    result = simulate_alerts(test, score, threshold)
    result["hourly_auroc"] = hourly_auroc(test, score)
    return model, test, score, threshold, result


def main():
    OUT.mkdir(exist_ok=True)
    cohort = make_cohort(seed=SEED)
    splits = split_patients(cohort.patient_id.unique(), SEED)
    model, test, score, threshold, causal_result = run(cohort, True, splits)
    _, _, _, _, leaky_result = run(cohort, False, splits)

    n_patients = int(cohort.patient_id.nunique())
    n_septic = int(cohort.groupby("patient_id").onset_hour.first().notna().sum())
    metrics = {
        "WARNING": "SYNTHETIC DATA ONLY. These numbers describe a random generator, "
                   "not patients, and are not evidence of clinical performance.",
        "cohort": {"patients": n_patients, "with_sepsis": n_septic,
                   "patient_hours": int(len(cohort)), "test_patients": len(splits[2])},
        "model": "standardised logistic regression, 36 features (value, 6 h change, measured flag for 12 variables)",
        "prediction_target": "sepsis onset within the next 6 hours",
        "alarm_budget": "threshold = lowest value giving at most 30 false alerts per 100 patient-days on validation patients",
        "causal_features": causal_result,
        "leaky_features_for_comparison_only": leaky_result,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # ---- figure 1: leak vs no leak -------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    names = ["Only past values\n(correct)", "Future lab values visible\n(leak, shown on purpose)"]
    values = [causal_result["hourly_auroc"], leaky_result["hourly_auroc"]]
    bars = ax.bar(names, values, color=["#1f8fb3", "#b9bfc7"], width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Hourly AUROC")
    ax.set_title("Synthetic data: a leak makes a model look better than it is", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "leak_vs_causal.png")
    plt.close(fig)

    # ---- demo unit for the dashboard prototype -------------------------
    test = test.assign(score=score)
    contributions = model.organ_contributions(test)
    for organ, values in contributions.items():
        test[f"c_{organ}"] = values
    organs = list(contributions)
    rng = np.random.default_rng(SEED)
    stays = {pid: s for pid, s in test.groupby("patient_id")}
    septic = [p for p, s in stays.items() if s.onset_hour.notna().all() and len(s) >= 20]
    calm = [p for p, s in stays.items() if s.onset_hour.isna().all() and len(s) >= 30]
    def last(p, back=1):
        return stays[p].score.iloc[-back]
    used = set()
    def take(candidates, n, driver_variety=False):
        picked, seen = [], set()
        for p in candidates:
            if p in used:
                continue
            row = stays[p].iloc[-1]
            driver = max(organs, key=lambda o: row[f"c_{o}"])
            if driver_variety and driver in seen:
                continue
            seen.add(driver); used.add(p); picked.append(p)
            if len(picked) == n:
                break
        return picked
    high = take([p for p in septic if last(p) >= threshold], 3, driver_variety=True)
    rising = take([p for p in septic if last(p, 4) < threshold and last(p, 4) > last(p, 10) + 0.05], 1)
    false_alarm = take([p for p in calm if stays[p].score.iloc[24] >= threshold], 1)
    quiet = sorted(calm, key=last)
    chosen = high + rising + false_alarm + take(list(rng.permutation(quiet[: len(quiet) // 2])), 3)
    beds = []
    for bed, pid in enumerate(chosen, start=1):
        stay = stays[pid]
        if pid in false_alarm:
            stay = stay.iloc[:25]
        elif pid in rising:
            stay = stay.iloc[:-3]
        window = stay.tail(24)
        now = window.iloc[-1]
        contrib = {o: float(now[f"c_{o}"]) for o in organs}
        driver = max(contrib, key=contrib.get)
        raw = cohort[(cohort.patient_id == pid) & (cohort.hour <= now.hour)]
        latest = {}
        for name, spec in VARIABLES.items():
            if spec[0] != driver:
                continue
            latest[name] = {"value": round(float(now[name]), 1), "unit": UNITS[name],
                            "change_6h": round(float(now[name + "_change6h"]), 1),
                            "hours_since_measured": int(now.hour - raw[raw[name].notna()].hour.max())
                            if raw[name].notna().any() else None}
        beds.append({
            "bed": f"Bed {bed:02d}",
            "label": f"Synthetic patient S-{int(pid):04d}",
            "hours_in_unit": int(now.hour),
            "risk_now": round(float(now.score), 2),
            "risk_6h_ago": round(float(window.score.iloc[-7]), 2),
            "alert": bool(now.score >= threshold),
            "driving_organ": driver,
            "organ_contributions": {o: round(v, 2) for o, v in contrib.items()},
            "driver_values": latest,
            "risk_last_24h": [round(float(v), 3) for v in window.score],
        })
    beds.sort(key=lambda b: -b["risk_now"])
    unit = {"WARNING": metrics["WARNING"], "threshold": threshold, "beds": beds}
    (OUT / "demo_unit.json").write_text(json.dumps(unit, indent=2, ensure_ascii=False))

    # ---- figure 2: one synthetic patient -------------------------------
    top = beds[0]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=160)
    x = np.arange(-len(top["risk_last_24h"]) + 1, 1)
    ax.plot(x, top["risk_last_24h"], color="#1f8fb3", lw=2)
    ax.axhline(threshold, color="#c0392b", ls="--", lw=1)
    ax.text(x[0], threshold + 0.02, "alert threshold", color="#c0392b", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Hours before now")
    ax.set_ylabel("Risk of sepsis in next 6 h")
    ax.set_title(f"{top['label']}: driven by {top['driving_organ']} (synthetic)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "example_patient.png")
    plt.close(fig)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\nDemo unit:")
    for b in beds:
        print(f"  {b['bed']}  risk {b['risk_now']:.2f}  (6 h ago {b['risk_6h_ago']:.2f})  "
              f"{'ALERT' if b['alert'] else '     '}  driver: {b['driving_organ']}")


if __name__ == "__main__":
    main()
