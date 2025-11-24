import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
import urllib.request
import ssl

# Allow unverified SSL (needed on Streamlit Cloud sometimes)
ssl._create_default_https_context = ssl._create_unverified_context

# Direct download link from your release (right-click the file in Releases → Copy link address)
MODEL_URL = "https://github.com/huraira-chohan/Cats-vs-Dogs-CNN/releases/download/v1.0/cats-vs-dogs.pkl"
MODEL_PATH = "cats-vs-dogs.pkl"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("First run: downloading 508 MB model… (takes ~90 seconds)"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return joblib.load(MODEL_PATH)

# Load model
model = download_model()

# UI
st.set_page_config(page_title="Cat vs Dog", page_icon="dog", layout="centered")
st.title("Cat vs Dog Classifier")
st.caption("Trained on 25k images · 508 MB model · Works instantly after first load")

def predict(img):
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    prob = model.predict(x, verbose=0)[0][0]
    return ("DOG", prob * 100) if prob > 0.5 else ("CAT", (1 - prob) * 100)

uploaded = st.file_uploader("Upload a cat or dog photo", type=["jpg", "jpeg", "png", "webp"])

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
    st.info("Upload an image to get started!")
