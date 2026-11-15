import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")
st.title("Pneumonia Detection using CNN")
st.write("Educational chest X-ray image classification demonstration.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/pneumonia_cnn.keras")

try:
    model = load_model()
except Exception:
    st.error("Trained model not found. Run `python train.py` first.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded X-ray", width=350)

    if st.button("Run model"):
        resized = image.resize((224, 224))
        array = np.asarray(resized, dtype=np.float32)
        array = np.expand_dims(array, axis=(0, -1))

        score = float(model.predict(array, verbose=0)[0][0])
        if score >= 0.5:
            st.warning("Model output: PNEUMONIA class")
        else:
            st.success("Model output: NORMAL class")
        st.write(f"Raw model score: {score:.4f}")

st.caption(
    "Educational prototype only. This model is not clinically validated "
    "and must not be used for diagnosis, treatment, or medical decisions."
)
