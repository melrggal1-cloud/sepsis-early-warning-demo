# Sepsis early warning: a synthetic-data demonstration

> **Research demonstration only. Not a medical device. Not for clinical use.**
> Everything that runs here uses **synthetic data** made by a random number
> generator. No patient data is in this repository, and none is needed.
> The numbers below describe that generator. They are **not** evidence of how
> any model performs on real patients.

This repository shows, in a form anyone can run in a minute, the idea behind an
hourly early-warning tool for sepsis in the intensive care unit (ICU):

1. Every hour, look only at what was known **up to that hour**.
2. Give the risk that sepsis starts in the **next 6 hours**.
3. Say **which organ system** is pushing the risk up.
4. Judge the tool by what a ward would feel: **false alerts per day** and
   **hours of warning**, not by a single accuracy score.

## Run it

```bash
pip install -r requirements.txt
python run_demo.py                 # builds the synthetic cohort, trains, evaluates
streamlit run app/dashboard.py     # optional: small interactive dashboard
```

`run_demo.py` writes to `outputs/`:

| File | Contents |
|---|---|
| `metrics.json` | All results of the run |
| `demo_unit.json` | Eight synthetic "beds" used by the dashboard and the design prototype |
| `leak_vs_causal.png` | The leak comparison described below |
| `example_patient.png` | Risk over 24 hours for one synthetic patient |

## What the demo does

| Step | File | In plain words |
|---|---|---|
| Make patients | `sepsis_demo/synthetic.py` | 1,500 invented ICU stays with 12 hourly variables. About 15 % develop "sepsis", which shows up as a gradual drift in one organ system. Some patients without sepsis are unwell for other reasons, which is what causes false alerts. |
| Build features | `sepsis_demo/features.py` | For each hour: the latest value, its change over 6 hours, and whether it was measured that hour. Gaps are filled from the past only. |
| Model | `sepsis_demo/model.py` | Logistic regression. Because it is linear, each organ system's share of the score can be read off directly. |
| Evaluate | `sepsis_demo/evaluate.py` | Simulates alerts with a 6-hour snooze. The threshold is set on validation patients to fit an alarm budget, then applied once to test patients. |

## Results of the synthetic run

300 synthetic test patients (44 with sepsis), threshold chosen on separate
validation patients for at most 30 false alerts per 100 patient-days.

| Measure (synthetic data) | Value |
|---|---|
| False alerts per 100 patient-days | 27 |
| Sepsis patients warned before onset | 75 % |
| Median warning time | 4 hours |
| Share of alerts followed by sepsis within 12 hours | 16 % |
| Hourly AUROC | 0.82 |

### Why "only what was known at that hour" matters

`features.py` can also be run with a deliberate mistake: lab results are filled
backward as well as forward, so a result drawn at hour 20 is visible at hour 15.
On the same synthetic patients the hourly AUROC rises from 0.82 to 0.84 and the
share of patients warned rises from 75 % to 84 %. Nothing about the model got
better; it was shown the future. On real hospital data this effect can be much
larger, and it disappears the moment the model is used live.

## Preliminary research code

`preliminary/` holds two earlier files from the research project behind this
idea, including an earlier pipeline with a **known leak**. They are examples of
the work, not working tools, and no number they print is valid. Read
`preliminary/README.md` first.

## What this is not

- Not validated on any patient, in any hospital.
- Not the research project's current code, protocol or results.
- Not a product. Software that warns clinicians about sepsis is regulated as a
  medical device and needs clinical validation and regulatory review before use.

## Licence

MIT. See `LICENSE`.
