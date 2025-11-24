import streamlit as st
import joblib
import numpy as np
from PIL import Image

# ------------------- Config -------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="dog",
    layout="centered"
)

# ------------------- Title -------------------
st.title("Cat vs Dog Classifier")
st.markdown("Upload a photo of a cat or dog and I'll tell you which one it is!")

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    return joblib.load("cats-vs-dogs.pkl")

model = load_model()

# ------------------- Prediction Function -------------------
def predict(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, 224, 224, 3)

    prob_dog = model.predict(arr, verbose=0)[0][0]

    if prob_dog > 0.5:
        return "DOG", prob_dog * 100
    else:
        return "CAT", (1 - prob_dog) * 100

# ------------------- UI -------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your image", use_column_width=True)

    with st.spinner("Analyzing..."):
        label, confidence = predict(image)

    st.markdown(f"## It's a **{label}**!")
    st.progress(confidence / 100)
    st.write(f"**Confidence: {confidence:.1f}%**")

    # Fun emoji
    if label == "DOG":
        st.balloons()
    else:
        st.snow()

# ------------------- Footer -------------------
st.markdown("---")
st.caption("Trained on 25,000 cat & dog images • Simple CNN • Made with ❤️ & Streamlit")
