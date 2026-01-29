# AI-CARE Artifact Description

This document describes the software artifact accompanying the paper:

**AI-CARE: Carbon-Aware Reporting Evaluation Metric for AI Models**

The purpose of this artifact is to enable transparent and reproducible
reproduction of the experiments, metrics, tables, and figures reported
in the paper. The artifact is strictly scoped to the reported work and
does not include extensions beyond the paper.

---

## Artifact Purpose

AI-CARE is a **reporting-centric evaluation artifact** for machine learning
models. It measures and reports:

- Task performance (accuracy)
- Training energy consumption and derived carbon emissions
- Inference energy consumption and derived carbon emissions
- Carbon–Accuracy Tradeoff Curves (CATC)
- Scalar Carbon-Aware Scores (SCAS)
- Pareto-optimal model sets

The artifact does **not** perform training-time optimization, resource
control, scheduling, or adaptive configuration. All measurements are
reported under fixed experimental conditions.

---

## Reproducibility Scope

Running the artifact reproduces:

- Global and per-dataset Carbon–Accuracy Tradeoff Curves (Fig. 2)
- Per-dataset scalar carbon-aware score plots (Fig. 3)
- Pareto tables reported in the appendix
- CSV logs containing performance, energy, and carbon metrics

The default execution produces **only** the results described in the paper.

---

## Datasets

The following datasets are evaluated:

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- ImageNet-100

Dataset-specific properties (input resolution, number of channels, number
of classes) are inferred automatically by the evaluation pipeline.

---

## Models

The artifact evaluates representative model families consistent with the paper:

- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Transformer-based classifier
- MLP-Mixer
- MobileNetV2
- ResNet-18 (ImageNet-100 only)

All models are trained using identical optimization settings and a fixed
number of epochs to emphasize relative accuracy–energy–carbon tradeoffs.

---

## Experimental Setup

- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 64
- Training epochs: 10
- Hardware: CPU-only
- Grid carbon intensity: 400 gCO₂/kWh

A fixed carbon intensity is used to ensure comparable and reproducible
carbon reporting across datasets and models.

---

## Software Requirements

- Python 3.9 or later
- CPU-only environment
- Operating systems: Linux, macOS, or Windows

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## How to Run the Artifact

Execute the full evaluation pipeline:

```bash
python run_benchmark_experiments.py
python analyze_results.py
```

These commands generate all figures, tables, and CSV outputs reported
in the paper.

---

## Output Structure

- `results/`  
  CSV files containing performance, energy, and carbon metrics

- `plots/`  
  Carbon–Accuracy Tradeoff Curves and scalar carbon-aware score plots

- `tables/`  
  Pareto-optimal model tables

---

## Notes for Reviewers

- The artifact is intentionally minimal and paper-aligned.
- Optional validation utilities are present in the repository but are
  disabled by default and are not part of the reported results.
- No Docker environment or specialized hardware is required.

---

## License

This artifact is released for research and educational use only.
