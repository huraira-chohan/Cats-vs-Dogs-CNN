import streamlit as st
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Page config & title
# -----------------------------
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload an image and I'll tell you if it's a cat or a dog (with confidence %)")

# -----------------------------
# Load the model (with caching so it loads only once)
# -----------------------------
@st.cache_resource
def load_model():
    # If you trained with Keras, joblib works fine for simple models
    model = joblib.load("cats-vs-dogs.pkl")
    return model

model = load_model()

# -----------------------------
# Prediction function
# -----------------------------
def predict_image(image: Image.Image):
    # Resize + normalize exactly like during training
    img = image.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, 224, 224, 3)   # batch dimension

    prob_dog = model.predict(arr, verbose=0)[0][0]   # sigmoid output (0=cat, 1=dog)

    if prob_dog > 0.5:
        label = "DOG"
        confidence = prob_dog * 100
    else:
        label = "CAT"
        confidence = (1 - prob_dog) * 100

    return label, confidence

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    with st.spinner("Thinking..."):
        label, confidence = predict_image(image)

    st.success(f"Prediction: **{label}**")
    st.metric(label="Confidence", value=f"{confidence:.1f}%")

    # Fun little bar
    if label == "DOG":
        st.progress(float(confidence / 100))
    else:
        st.progress(float(100 - confidence) / 100)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Model trained on Microsoft Cats & Dogs dataset (25k images) using a simple CNN.")
