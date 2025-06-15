# Advanced Data Challenge - Final Project

## Overview

This repository contains the code for the final project of the **Advanced Data Challenge** subject.  
The project implements a custom version of **ResNet-18** with improvements, including attention-based global pooling, to enhance medical image classification performance and interpretability.

---

## Features

- **Custom ResNet-18 Architecture**  
  Built from scratch for research transparency and flexibility.
- **Attention-Based Global Pooling**  
  Integrates a learnable attention mechanism for feature aggregation, replacing traditional global pooling layers.
- **Flexible Training Pipeline**  
  - Supports mixed-precision training (AMP) for improved efficiency.
  - Implements a cosine learning rate schedule with warmup.
  - Checkpoint saving and resuming for long training sessions.
  - Custom optimizer parameter grouping for advanced regularization.
- **Advanced Evaluation Metrics**  
  - Computes accuracy, precision, recall, F1-score (per class and macro-averaged).
  - Includes confusion matrix and ROC-AUC visualization.
  - Grad-CAM visualization for model interpretability on medical images.

---

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- einops
- fsspec
- scikit-learn
- hydra-core
Install all dependencies with:
```bash
pip install -r requirements.txt

---

## Dataset

- **Not included in this repository.**
- Please download the Chest X-Ray14 dataset (or your chosen dataset) and place the images in the `images/` folder.
- See `data/README.md` or project instructions for the download link and usage details.
-here is the data i used in this project, it is just a subset of X-Ray 14: https://drive.google.com/drive/folders/1sbL2rDCGw3-3ECQ5lkSC64eY7a9mYNi2?usp=sharing

---

## Usage

### **Training**

```bash
python train.py --config configs/project.yaml
