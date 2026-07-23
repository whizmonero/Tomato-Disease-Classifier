# Tomato Disease Classifier

## Overview

This project is a deep learning application developed to classify tomato leaf images as either **Healthy** or **Tomato Early Blight** using Transfer Learning with MobileNetV2.

The project was implemented using TensorFlow/Keras and deployed using Streamlit.

---

## Project Structure

```
Tomato-Disease-Classifier/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── tomato_disease_classifier.keras
│   └── training_history.csv
│
├── Notebook/
│   ├── 01_data_preparation.ipynb
│   ├── 02_train_model.ipynb
│   └── 03_model_evaluation.ipynb
│
└── venv/
```

---

## Dataset

PlantVillage Dataset

Classes:

- Tomato_Early_blight
- Tomato_healthy

---

## Model

- MobileNetV2 (Transfer Learning)
- TensorFlow/Keras
- Image Size: 224 × 224
- Optimizer: Adam
- Loss Function: SparseCategoricalCrossentropy

---

## Features

- Image Classification
- Healthy vs Early Blight Detection
- Streamlit Web Application
- Prediction Confidence
- Training History Visualization
- Confusion Matrix
- Classification Report

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Tomato-Disease-Classifier.git
```

Navigate into the project

```bash
cd Tomato-Disease-Classifier
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV

---

## Author

John Monday

Department of Electrical  Engineering

GET 324 – Deep Learning Project