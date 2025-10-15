# 🧠 NeuroAssist: Explainable Brain Tumor Classification for Clinical Decision Support

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

An **explainable, lightweight deep learning system** for classifying brain tumors from MRI scans, designed to assist radiologists in low-resource settings.

> **"AI should augment clinicians — not replace them."**

## 🌍 Clinical Motivation
- Brain tumors affect ~300,000 people globally each year (WHO).
- In many regions, there is **<1 radiologist per 100,000 people**.
- Early and accurate diagnosis improves survival rates by up to 40%.
- **Goal**: Provide a **transparent, reliable screening tool** to prioritize urgent cases.

## ✨ Features
- ✅ 4-class classification: `glioma`, `meningioma`, `pituitary`, `no tumor`
- 🔍 **Grad-CAM visualizations** – see *where* the model looks
- 📊 **Comprehensive evaluation**: per-class F1, AUC, confusion matrix
- 🧪 **Uncertainty estimation** via Monte Carlo Dropout
- 🌐 **Web demo** with Gradio (try it yourself!)
- 🐳 **Dockerized** for reproducibility
- 📜 **Model card** with ethical considerations

## 📈 Results (ResNet50, 10 epochs)
| Class          | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Glioma         | 0.96      | 0.94   | 0.95     |
| Meningioma     | 0.97      | 0.98   | 0.97     |
| No Tumor       | 0.99      | 0.98   | 0.98     |
| Pituitary      | 0.95      | 0.96   | 0.95     |
| **Macro Avg**  | **0.97**  | **0.96** | **0.96** |

> Full results in `notebooks/exploration.ipynb`

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt