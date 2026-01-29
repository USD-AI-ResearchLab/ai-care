# EAPS / CARE-AI
Energy-Aware Performance Scoring and Carbon-Aware Evaluation for Machine Learning

This repository provides a reproducible benchmarking framework for evaluating machine learning models under joint accuracy, energy, and carbon constraints. It accompanies the paper:

The Energy Is All You Forget: Mandating Carbon Reporting and Carbon–Accuracy Tradeoff Metrics.

The framework elevates energy and carbon to first-class evaluation dimensions and enables standardized, deployment-relevant model comparison.

---

## Overview

The pipeline trains representative models under identical conditions, measures empirical energy usage, converts it to carbon emissions using a fixed grid intensity, and evaluates models using CARE-AI efficiency metrics and Carbon–Accuracy Tradeoff Curves (CATC).

The goal is not to maximize accuracy, but to expose accuracy–carbon tradeoffs transparently and reproducibly.

---

## Models

- MLP (lightweight baseline)
- CNN (accuracy–efficiency tradeoff)
- Tiny Transformer (compact transformer)

All models use fixed hyperparameters across datasets.

---

## Datasets

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- ImageNet-100 (stress test)

Dataset properties are inferred dynamically.

---

## Experimental Setup

- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 64
- Epochs: 3
- Hardware: CPU-only
- Carbon intensity: 400 gCO2/kWh

---

## Metrics

- Accuracy
- Training energy and carbon
- Inference energy and carbon
- CARE-AI efficiency score
- Pareto-optimality via CATC

---

## How to Run

```bash
pip install -r requirements.txt
python run_benchmark_experiments.py
python analyze_results.py
```

Docker execution is also supported.

---

## Outputs

Results are logged to CSV files, with plots and Pareto tables generated post hoc.

---

## Reproducibility

CPU-only, fixed configs, deterministic logging, and external carbon validation support ensure artifact reproducibility.

---

## License

Research and educational use only.
