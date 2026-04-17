import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# =========================
# Labels (IMPORTANT)
# =========================
labels = ["Grade A", "Grade B", "Grade C"]

# =========================
# Load ONNX model safely
# =========================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_hollowblock.onnx")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# =========================
# Preprocessing
# =========================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    return img_array

# =========================
# Softmax (only if needed)
# =========================
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# =========================
# UI
# =========================
st.title("Hollow Block Classifier (ONNX)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_data = preprocess_image(image)

    # Run inference
    outputs = session.run(None, {input_name: input_data})[0][0]

    # Detect if model already outputs probabilities
    if np.sum(outputs) > 1.5:
        probs = softmax(outputs)  # logits case
    else:
        probs = outputs  # already softmax

    pred_class = int(np.argmax(probs))

    st.subheader("Prediction")
    st.success(labels[pred_class])

    st.subheader("Confidence Scores")
    for i, label in enumerate(labels):
        st.write(f"{label}: {probs[i] * 100:.2f}%")
