import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# DIRECT LINK FROM HUGGING FACE (never fails, never corrupts)
MODEL_URL = "https://huggingface.co/huraira-chohan/cats-vs-dogs-cnn/resolve/main/cats-vs-dogs.h5"
MODEL_PATH = "cats-vs-dogs.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Hugging Face (~60 sec first time)…"):
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    with st.spinner("Loading model…"):
        return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="Cat vs Dog", page_icon="🐶", layout="centered")
st.title("🐱 Cat vs Dog Classifier 🐶")
st.caption("Trained on 25k images • Hosted on Hugging Face")

def predict(img):
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    prob = float(model.predict(x, verbose=0)[0][0])
    return ("DOG", prob * 100) if prob > 0.5 else ("CAT", (1 - prob) * 100)

file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
if file:
    img = Image.open(file)
    st.image(img, use_column_width=True)
    label, conf = predict(img)
    st.success(f"It's a **{label}**!")
    st.progress(conf / 100)
    st.write(f"**Confidence: {conf:.1f}%**")
    if label == "DOG": st.balloons()
    else: st.snow()
else:
    st.info("Upload an image to start!")
