"""Web app to upload an image and predict its handwritten digit."""
import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

st.set_page_config(page_title="Handwritten Digit Classifier", page_icon="🔢")
st.title("🔢 Handwritten Digit Classifier")
st.write("Upload a clear image containing one handwritten digit from 0 to 9.")
st.caption("Educational demo only. Predictions can be incorrect, especially for unclear or off-center images.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.keras")

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found. Run this first in the terminal: python train_model.py")
    st.stop()

uploaded = st.file_uploader("Upload a PNG or JPG image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("L")
    st.image(image, caption="Uploaded image", width=220)

    # MNIST images use light digits on a dark background.
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28))
    pixels = np.asarray(image).astype("float32") / 255.0
    pixels = pixels.reshape(1, 28, 28)

    model = load_model()
    probabilities = model.predict(pixels, verbose=0)[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction]) * 100

    st.subheader(f"Predicted digit: {prediction}")
    st.write(f"Model confidence: {confidence:.1f}%")
    st.bar_chart({str(i): float(probabilities[i]) for i in range(10)})
