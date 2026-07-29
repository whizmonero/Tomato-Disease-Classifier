# 🍅 Tomato Disease Classifier

## Overview

This project is a deep learning application developed for detecting tomato leaf diseases using Transfer Learning with MobileNetV2.

The model classifies tomato leaf images into two categories:

- Tomato Healthy
- Tomato Early Blight

The web application is built using Streamlit and TensorFlow.

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
- Loss Function: Sparse Categorical Crossentropy

---

## Project Structure

```
Tomato-Disease-Classifier/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
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
└── dataset/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/whizmonero/Tomato-Disease-Classifier.git
```

Move into the project folder

```bash
cd Tomato-Disease-Classifier
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

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
- Pillow

---

## Features

- Tomato Disease Classification
- Healthy vs Early Blight Detection
- Confidence Score
- Image Upload
- MobileNetV2 Transfer Learning
- Streamlit Web Interface

---

## Author

John Nkpoikanke Monday 

## Collaborators

Effiong Elisha Emmanuel;
Obot Mmokutmfon Daniel;
Abiodun Godswill Alexander;
Udo Aniekpeno Nkereuwem;
Edward John David;
Jackson Hannah Udoh;
Emmanuel Donald Uwem;
Wobo, Prosper Nyeyiruchi;
John, Wisdom Akan; 
Archibong, Oyokunyi Uwe;
Praise Nkereuwem Esikhene;
Gideon Akuche ifeanyichukwu 

Department of Electrical  Engineering

GET 324 – Deep Learning Project Group EE17