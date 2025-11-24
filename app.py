import streamlit as st
import pickle
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

st.set_page_config(page_title="My Cats vs Dogs", page_icon="🐶", layout="centered")
st.title("🐱 My Custom Cats vs Dogs Classifier 🐶")
st.write("Powered by **your own 532 MB trained model**")

@st.cache_resource
def load_model():
    with open("cats_vs_dogs_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
st.success("✅ Your personal model loaded successfully!")

file = st.file_uploader("Upload a cat or dog photo", type=["jpg","jpeg","png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_column_width=True)
    
    img = img.resize((224, 224))   # ← change only if you trained with different size
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, 0)
    
    pred = model.predict(x, verbose=0)[0][0]
    
    if pred > 0.5:
        st.balloons()
        st.success(f"**DOG** – {pred:.1%} confident")
    else:
        st.balloons()
        st.success(f"**CAT** – {(1-pred):.1%} confident")
        
    st.bar_chart({"Cat": 1-pred, "Dog": pred})

st.caption("Your own trained model • Live on the internet!")