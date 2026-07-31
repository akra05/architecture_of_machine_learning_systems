# AMLS – Real vs. AI-Generated Image Classification

University project for the **Advanced Machine Learning Systems (AMLS)** course at **TU Berlin**.

The goal of this project was to develop a machine learning pipeline capable of distinguishing real images from AI-generated images under strict resource constraints, including **CPU-only execution**, **fixed time budgets per training step**, and **no internet access during runtime**.

A detailed discussion of the motivation, experimental setup, and design decisions can be found in the accompanying **report (`report.pdf`)**. This README only explains how to run the project.

---

# Project Overview

## Task 1 – Data Exploration & Cleaning

- Analyze the provided dataset
  - Class distribution
  - Image resolutions
  - Differences between real and AI-generated images
- Develop a deterministic data-cleaning pipeline

## Task 2 – Modeling under Time Constraints

Compare different model families for classifying **real** versus **AI-generated** images.

Implemented approaches:

- Classical baseline model using handcrafted features
- Convolutional Neural Network (CNN)

Target performance:

- Recall<sub>AI</sub> ≥ 0.8
- False Positive Rate ≤ 20%

## Task 3 – Robustness Evaluation

Improve the model's robustness by training and evaluating it on intentionally degraded images.

Applied augmentations include:

- Rotation
- Scaling
- Reduced image quality
- Additional image transformations

The objective is to assess how well the classifier generalizes to images with varying quality and appearance while maintaining competitive performance.

---

## Disclaimer

- The dataset is **not included** in this repository. Download it from the official TU Berlin Cloud link above and verify that the full download completed successfully (e.g. check that the resulting image counts after `clean.py` match what is documented in this README) before running the pipeline — an incomplete or interrupted download can silently produce a smaller, differently balanced dataset without raising an error.
- All required models and artifacts are generated automatically during execution.
- The project was developed for a **CPU-only Docker environment without internet access**, following the requirements of the AMLS course.
- The exact composition of the dataset (and therefore exact recall/FPR values reported below) is outside of this project's control, since it is provided externally via the course cloud link. Minor variations between runs or download instances may lead to slightly different training results than those reported here.

# Dataset

The dataset was provided as part of the course and can be downloaded from:

**TU Berlin Cloud**

https://tubcloud.tu-berlin.de/s/4BF6KzyQ7k8F6Ls

The dataset is **not included in this repository** because it is too large for Git (see `.gitignore`) and must be downloaded manually.

---

# Project Structure

```text
solution/
├── Dockerfile
├── requirements.txt
├── clean.py                 # Task 1: Data cleaning
├── prepare.py               # Tensor preparation
├── train.py                 # Task 2: Model training
├── predict.py               # Task 2: Inference and threshold calibration
├── train_augmented.py       # Task 3: Train augmented Model
└── predict_augmented.py     # Task 3: Inference and threshold calibration

report.pdf                   # Complete report
```

During execution, the following directories are also expected (not included in the repository):

```text
solution/
├── data/        # Dataset (read-only)
└── artifacts/   # Generated automatically
```

---

# Running the Project

The entire pipeline is designed to run inside a Docker container.

## 1. Build the Docker image

```bash
cd solution

docker build -t amls-solution .
```

---

## 2. Download the dataset

Download the dataset and extract its contents into the `solution/data/` directory.

The final directory structure should look like:

```text
solution/
└── data/
    ├── train/
    ├── validation/
    ├── calibration/
    └── predict/
```

## 3. Execute the pipeline

Run the scripts in the following order.

### Task 1 – Data Cleaning

```bash
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/artifacts:/app/artifacts amls-solution python clean.py --timeout_seconds 600
```
---

### Tensor Preparation

```bash
docker run --rm \
-v ${PWD}/data:/app/data \
-v ${PWD}/artifacts:/app/artifacts \
amls-solution \
python prepare.py --timeout_seconds 600
```

---

### Task 2 – Model Training

```bash
docker run --rm \
-v ${PWD}/data:/app/data \
-v ${PWD}/artifacts:/app/artifacts \
amls-solution \
python train.py --timeout_seconds 1800
```

---

### Task 2 – Prediction

```bash
docker run --rm \
-v ${PWD}/data:/app/data \
-v ${PWD}/artifacts:/app/artifacts \
amls-solution \
python predict.py --timeout_seconds 600
```

---

### Task 3 – Model Training

```bash
docker run --rm \
-v ${PWD}/data:/app/data \
-v ${PWD}/artifacts:/app/artifacts \
amls-solution \
python train_augmented.py --timeout_seconds 1800
```

---

### Task 3 – Prediction

```bash
docker run --rm \
-v ${PWD}/data:/app/data \
-v ${PWD}/artifacts:/app/artifacts \
amls-solution \
python predict_augmented.py --timeout_seconds 600
```

---

# Results

The final prediction files are written to

```text
artifacts/task02/predictions.csv
```

and

```text
artifacts/task03/predictions.csv
```

A complete discussion of all experiments, ablation studies, and design decisions is available in the accompanying **report (`report.pdf`)**.

### Summary

| Task | Result |
|------|--------|
| Task 2 | CNN with image-size metadata achieved Recall<sub>AI</sub> ≈ 0.77 while maintaining an FPR ≤ 20% |
| Task 3 | After additional training on augmented data, the model achieved a Recall<sub>AI</sub> of approximately **0.80** while maintaining an FPR ≤ 20% on the original validation dataset. On the **validation_augmented** dataset, the model achieved a Recall<sub>AI</sub> of approximately **0.60** while maintaining the same FPR constraint. |

---

- The dataset is **not included** in this repository.
- All required models and artifacts are generated automatically during execution.
- The project was developed for a **CPU-only Docker environment without internet access**, following the requirements of the AMLS course.
