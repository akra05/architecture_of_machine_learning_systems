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

## Task 3

*Briefly describe Task 3 here.*

---

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
├── train_augmented.py       # Task 3
├── predict_augmented.py
└── ...

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

Download the dataset and place it under

```text
solution/data/
```

Expected directory structure:

```text
data/
├── train/
├── validation/
├── calibration/
└── predict/
```

---

## 3. Execute the pipeline

Run the scripts in the following order.

### Task 1 – Data Cleaning

```bash
docker run --rm \
-v ${PWD}/data:/app/data \
-v ${PWD}/artifacts:/app/artifacts \
amls-solution \
python clean.py --timeout_seconds 600
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
| Task 2 | CNN with image-size metadata achieved Recall<sub>AI</sub> ≈ X while maintaining an FPR ≤ 20% |
| Task 3 | *(Add results here.)* |

---

- The dataset is **not included** in this repository.
- All required models and artifacts are generated automatically during execution.
- The project was developed for a **CPU-only Docker environment without internet access**, following the requirements of the AMLS course.
