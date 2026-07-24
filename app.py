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

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model.\n\n{e}")
    st.stop()

# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img, dtype=np.float32)

    # IMPORTANT:
    # Do NOT divide by 255 here because the trained model
    # already contains a Rescaling(1./255) layer.

    img = np.expand_dims(img, axis=0)

    return img


# ==========================================================
# USER INTERFACE
# ==========================================================

st.title("🍅 Tomato Disease Classifier")

st.write(
    """
Upload a **tomato leaf image** to classify it as:

- Healthy Tomato Leaf
- Tomato Early Blight
"""
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    except Exception:
        st.error("Unable to read the uploaded image.")
        st.stop()

    if st.button("Predict"):

        processed_image = preprocess_image(image)

        prediction = model.predict(
            processed_image,
            verbose=0
        )

        prediction = prediction[0]

        predicted_index = np.argmax(prediction)

        confidence = float(prediction[predicted_index])

        # Reject uncertain predictions
        if confidence < 0.75:

            st.warning(
                """
This image does not closely match the training data.

Please upload a clear tomato leaf image.
"""
            )

            st.stop()

        predicted_class = CLASS_NAMES[predicted_index]

        st.divider()

        st.subheader("Prediction Result")

        if predicted_class == "Tomato_healthy":

            st.success("✅ Healthy Tomato Leaf")

        else:

            st.error("🍂 Tomato Early Blight Detected")

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        st.subheader("Prediction Probabilities")

        for class_name, probability in zip(CLASS_NAMES, prediction):

            st.write(f"**{class_name}** : {probability:.2%}")

            st.progress(float(probability))

st.divider()

st.caption(
    "GET 324 • Deep Learning for Tomato Disease Detection • Group EE17"
)