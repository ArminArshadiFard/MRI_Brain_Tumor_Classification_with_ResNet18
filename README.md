## MRI Brain Tumor classification using ResNet18

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

In this project I chose ResNet and trained it on a MRI Brain Tumor dataset availiable on kaggle.
## Clinical Motivation
- Brain tumors affect ~300,000 people globally each year (WHO).
- In many regions, there is **<1 radiologist per 100,000 people**.
- Early and accurate diagnosis improves survival rates by up to 40%.
- **Goal**: Provide a **transparent, reliable screening tool** to prioritize urgent cases.

## Features
-  4-class classification: `glioma`, `meningioma`, `pituitary`, `no tumor`
-  **the dataset has a total of 3264 images which is divided into 2,870 (Training) + 394 (Testing)
-  **Grad-CAM visualizations** – see *where* the model looks (must run app.py)
-  **Comprehensive evaluation**: using TTA evaluation
-  **Uncertainty estimation** via Monte Carlo Dropout
-  **Web demo** using Gradio

## 📈 Results (ResNet18, 15 epochs)
<img width="4470" height="1466" alt="training_curves" src="https://github.com/user-attachments/assets/5d56f7c7-ea90-4565-89b9-d71ab610ad6e" />

- there is a little bit of overfitting.
