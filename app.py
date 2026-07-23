# ==========================================================
# Tomato Disease Classifier
# Healthy Tomato vs Early Blight
# ==========================================================

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Tomato Disease Classifier",
    page_icon="🍅",
    layout="centered"
)

# ==========================================================
# Paths
# ==========================================================

PROJECT_DIR = Path(__file__).parent

MODEL_PATH = PROJECT_DIR / "models" / "tomato_disease_classifier.keras"

TEST_DIR = PROJECT_DIR / "dataset" / "test"

# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==========================================================
# Load Class Names
# ==========================================================

@st.cache_resource
def get_class_names():

    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(224, 224),
        batch_size=1,
        shuffle=False
    )

    return dataset.class_names

class_names = get_class_names()

# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((224,224))

    image = np.array(image).astype(np.float32)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# ==========================================================
# User Interface
# ==========================================================

st.title("🍅 Tomato Disease Classifier")

st.write(
    """
Upload a tomato leaf image and the trained MobileNetV2 model
will classify it as:

- Healthy Tomato
- Tomato Early Blight
"""
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        processed = preprocess_image(image)

        prediction = model.predict(processed, verbose=0)

        predicted_index = np.argmax(prediction)

        confidence = float(np.max(prediction))

        predicted_class = class_names[predicted_index]

        st.divider()

        st.subheader("Prediction")

        if predicted_class == "Tomato_healthy":

            st.success("Healthy Tomato Leaf")

        else:

            st.error("Tomato Early Blight Detected")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.subheader("Class Probabilities")

        st.progress(float(prediction[0][0]))
        st.write(
            f"Early Blight: {prediction[0][0]*100:.2f}%"
        )

        st.progress(float(prediction[0][1]))
        st.write(
            f"Healthy: {prediction[0][1]*100:.2f}%"
        )

st.divider()

st.caption(
    "GET 324 • Deep Learning for Tomato Disease Detection"
)