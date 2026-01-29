# ARTIFACT.md — EAPS Evaluation Framework

This document describes the **research artifact** accompanying the paper:

**The Energy Is All You Forget: Toward Standardized Energy-Aware Evaluation in Machine Learning**

The artifact implements a reproducible benchmarking framework for evaluating machine learning models under joint **accuracy, energy, and carbon** constraints.

---

## Artifact Contents

The artifact consists of:

- Source code for training and evaluating ML models
- Energy and carbon measurement utilities
- Scripts for computing Energy-Aware Performance Scores (EAPS)
- Scripts for generating Carbon–Accuracy Tradeoff Curves (CATC)
- Configuration files for reproducible experiments
- Logged results and generated plots

---

## Claims Supported by the Artifact

The artifact supports the following claims made in the paper:

1. Accuracy-only evaluation can favor models with substantially higher carbon cost.
2. Energy and carbon metrics reveal meaningful tradeoffs between model performance and environmental impact.
3. EAPS provides a practical scalar score that alters model rankings compared to accuracy alone.
4. Carbon–Accuracy Tradeoff Curves expose empirical Pareto frontiers across datasets and architectures.
5. Relative carbon rankings are robust across measurement backends.

---

## How to Use the Artifact

### Minimum Requirements
- Python 3.9+
- OR Docker (recommended for full reproducibility)
- CPU-only hardware is sufficient

### Basic Usage
1. Install dependencies or build the Docker image
2. Run benchmark experiments
3. Generate CATC plots and EAPS scores
4. Inspect CSV results and figures

Refer to `README.md` for detailed, step-by-step instructions.

---

## Expected Outputs

After successful execution, the artifact produces:

- `results/results.csv` — logged accuracy, energy, carbon, and EAPS metrics
- `plots/*.png` — Carbon–Accuracy Tradeoff Curves (dataset-wise and aggregated)

---

## Reproducibility Notes

- All experiments use fixed hyperparameters and controlled execution.
- No hyperparameter tuning or architecture search is performed.
- Energy-to-carbon conversion uses a fixed grid intensity (400 gCO₂/kWh).
- Results are deterministic under fixed seeds and hardware.

---

## Ethical and Environmental Considerations

This artifact explicitly measures and reports the environmental cost of machine learning models.
It is intended to promote transparency, responsible benchmarking, and sustainable AI research practices.

---

## Anonymity

This artifact is suitable for **double-blind review**.
All identifying information has been removed.
A de-anonymized version will be released upon acceptance.
