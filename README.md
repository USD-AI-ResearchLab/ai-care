# AI-CARE
Carbon-Aware Reporting Evaluation Tool for AI Models

This repository provides the official implementation of **AI-CARE**, a
reporting-centric evaluation tool for measuring and reporting predictive
performance, energy consumption, and carbon emissions of machine learning
models.

The repository accompanies the paper:

**AI-CARE: Carbon-Aware Reporting Evaluation Metric for AI Models**

AI-CARE elevates energy and carbon emissions to first-class evaluation
quantities and enables transparent, reproducible comparison of models under
fixed experimental conditions.

---

## Overview

AI-CARE evaluates machine learning models by executing training and inference
under identical conditions, measuring empirical energy consumption, converting
energy usage to carbon emissions using a fixed grid carbon intensity, and
reporting accuracy–carbon tradeoffs through standardized metrics and
visualizations.

The framework does **not** modify model architectures, optimization algorithms,
or training procedures. It reports empirically observed performance, energy, and
carbon metrics during both training and inference.

---

## Models

The following representative model families are evaluated:

- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Transformer-based classifier
- MLP-Mixer
- MobileNetV2
- ResNet-18 (ImageNet-100 only)

Model hyperparameters are held fixed across datasets to isolate architectural
and dataset effects.

---

## Datasets

Experiments are conducted on five vision benchmarks of increasing complexity:

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- ImageNet-100

Dataset properties (input resolution, number of channels, number of classes)
are inferred automatically by the evaluation pipeline.

---

## Experimental Setup

- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 64
- Training epochs: 10
- Hardware: CPU-only
- Grid carbon intensity: 400 gCO₂/kWh

A fixed number of epochs is used to emphasize relative accuracy–energy–carbon
tradeoffs rather than absolute peak accuracy.

---

## Reported Metrics

AI-CARE reports the following quantities:

- Task performance (accuracy)
- Training energy consumption and carbon emissions
- Inference energy consumption and carbon emissions
- Total carbon emissions (training + inference)
- Carbon–Accuracy Tradeoff Curves (CATC)
- Scalar Carbon-Aware Score (SCAS)

The scalar score integrates normalized task performance and total carbon
emissions to support comparative ranking when a single decision criterion is
required.

---

## How to Run

```bash
pip install -r requirements.txt
python run_benchmark_experiments.py
python analyze_results.py
```

---

## Outputs

The evaluation pipeline produces CSV files containing reported performance,
energy, and carbon metrics, along with publication-quality visualizations of
carbon–accuracy tradeoffs and per-dataset scalar carbon-aware scores.

---

## Reproducibility

All experiments are executed under fixed hardware and software conditions with
deterministic configurations and a fixed grid carbon intensity to ensure
reproducible and comparable results across models and datasets.

---

## License

This project is licensed under the Apache License 2.0 – see the LICENSE file for details.

## Patent Notice

This project may be subject to patent protection. 
Certain components described in the associated publication are pending patent review.


