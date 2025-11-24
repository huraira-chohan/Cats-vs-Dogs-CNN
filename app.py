import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import urllib.request

# YOUR WORKING .h5 MODEL LINK
MODEL_URL = "https://drive.google.com/uc?export=download&id=1zBY72cLNpfVaAnFvKGkM7OVsWS92anwR"
MODEL_PATH = "cats-vs-dogs.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("First run: downloading model (~60 seconds)…"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    with st.spinner("Loading model into memory…"):
        return tf.keras.models.load_model(MODEL_PATH)

# Load the model (this line will now succeed!)
model = load_model()

st.set_page_config(page_title="Cat vs Dog", page_icon="🐶", layout="centered")
st.title("🐱 Cat vs Dog Classifier 🐶")
st.caption("Trained on 25k images • Works instantly after first load")

def predict(img):
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    prob = float(model.predict(x, verbose=0)[0][0])
    return ("DOG", prob * 100) if prob > 0.5 else ("CAT", (1 - prob) * 100)

uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_column_width=True)
    with st.spinner("Predicting..."):
        label, confidence = predict(image)
    st.success(f"It's a **{label}**!")
    st.progress(confidence / 100)
    st.write(f"**Confidence: {confidence:.1f}%**")
    
    if label == "DOG":
        st.balloons()
    else:
        st.snow()
else:
    st.info("↑ Upload a cat or dog photo to see the magic!")

st.markdown("---")
st.caption("Made with ❤️ by huraira-chohan")
