from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Tomato Disease Classifier",
    page_icon="🍅",
    layout="centered"
)

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / "tomato_disease_classifier.keras"

# ==========================================================
# CLASS NAMES
# ==========================================================

CLASS_NAMES = [
    "Tomato_Early_blight",
    "Tomato_healthy"
]

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32)
    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# ==========================================================
# USER INTERFACE
# ==========================================================

st.title("🍅 Tomato Disease Classifier")

st.write("""
Upload a tomato leaf image to determine whether it is:

- Healthy Tomato Leaf
- Tomato Early Blight
""")

uploaded_file = st.file_uploader(
    "Choose an image...",
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

        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image, verbose=0)

        predicted_index = np.argmax(prediction)

        confidence = float(np.max(prediction))

        predicted_class = CLASS_NAMES[predicted_index]

        st.divider()

        st.subheader("Prediction Result")

        if predicted_class == "Tomato_healthy":
            st.success("Healthy Tomato Leaf")
        else:
            st.error("Tomato Early Blight Detected")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}"
        )

        st.subheader("Prediction Probabilities")

        for label, probability in zip(CLASS_NAMES, prediction[0]):
            st.write(f"**{label}** : {probability:.2%}")
            st.progress(float(probability))

st.divider()

st.caption(
    "GET 324 • Deep Learning for Tomato Disease Detection Group EE17"
)