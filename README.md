# COMSYS Hackathon 2025 – Task B: Face Recognition

This repository contains a PyTorch implementation for **Task B** of the COMSYS Hackathon-5, 2025. The objective is to build a robust face recognition system that can identify individuals from images captured under visually degraded conditions like fog, blur, overexposure, rain, and low light.

---

## 📁 Dataset Structure

The dataset is structured as follows:

Task_B/
├── train/
│ ├── person_1/
│ ├── person_2/
│ ├── ...
├── val/
│ ├── person_1/
│ ├── person_2/
│ ├── ...
└── distorted/
├── (contains distorted versions of face images)


Each person’s folder contains 1 or more images of that individual.

---

## 🧠 Model Overview

- **Backbone:** Pretrained **ResNet18**
- **Classifier:** 2-layer fully connected head (512 → Dropout → ReLU → Output)
- **Framework:** PyTorch
- **Image Size:** 112 × 112
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (LR = 1e-4)
- **Scheduler:** StepLR (decay every 5 epochs)
- **Metrics:** Accuracy, F1-score (macro average)

---

## 🧪 How It Works

The model is trained using a **custom dataset loader** (`RecursiveFaceDataset`) that assigns a unique `Kxxxx` label to each individual, even across nested folder structures. The model learns to classify the identity of a face from degraded input images.

---

## 🔧 Setup Instructions

### 1️⃣ Install Requirements

```bash
pip install torch torchvision scikit-learn tqdm pillow
