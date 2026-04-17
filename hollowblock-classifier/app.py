import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# =========================
# Load ONNX model safely
# =========================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_hollowblock.onnx")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# =========================
# Preprocessing function
# =========================
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # MobileNet standard size
    img_array = np.array(image).astype(np.float32) / 255.0

    # Ensure 3 channels
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# Streamlit UI
# =========================
st.title("Hollow Block Classifier (ONNX)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_data = preprocess_image(image)

    # Run inference
    outputs = session.run(None, {input_name: input_data})
    prediction = np.argmax(outputs[0])

    st.write("Prediction:", int(prediction))
