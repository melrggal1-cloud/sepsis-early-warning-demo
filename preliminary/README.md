# Preliminary research code (example only)

These two files are **earlier versions** of code from the research project this
demonstration grew out of. They are here so a reader can see the ideas, and for
no other purpose.

| File | What it is |
|---|---|
| `osta_earlier_pipeline_KNOWN_LEAK.py` | An earlier training pipeline for the organ-system-aware model (OSTA): one small transformer per organ system, attention between organ systems, then a transformer over the whole stay. |
| `handcrafted_features_early.py` | An early set of composite clinical features (qSOFA-style indices, ratios, short-window trends). |

## Read this before using anything here

- **This version has known faults.** It did not pass the project's own test for
  information from the future, and it scored alarms with a different rule from
  the one used to choose the alarm threshold. Both flatter the results.
  **No number produced by this code is valid.** It has been superseded.
- **It cannot run from this repository.** It expects a hospital data file. That
  file is not here and must never be added: hospital research datasets are
  shared under agreements that forbid passing them on.
- **No trained model is included**, for the same reason.
- The current research code, protocol and results are not in this repository.

If you want something that runs, use the synthetic demonstration in the
repository root (`python run_demo.py`).
