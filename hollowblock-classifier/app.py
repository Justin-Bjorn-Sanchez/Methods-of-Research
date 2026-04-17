import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# =========================
# Labels
# =========================
labels = ["A", "B", "C"]

# =========================
# Load ONNX model
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
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# =========================
# Softmax
# =========================
def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# =========================
# FINAL COMBINE RULE
# =========================
def combine_grades(dl, hw):
    if dl == "A" and hw == "A":
        return "A"
    if dl == "C" or hw == "C":
        if dl == "A" or hw == "A":
            return "B"
        return "C"
    if dl == "B" or hw == "B":
        if dl == "A" or hw == "A":
            return "B"
        return "B"
    return "C"

# =========================
# COMMENT SYSTEM
# =========================
dl_comments = {
    "A": "Visually Great!",
    "B": "Visually Okay",
    "C": "Not Okay"
}

hw_comments = {
    "A": "High Density",
    "B": "Average Density",
    "C": "Low Density"
}

# =========================
# UI
# =========================
st.title("Hybrid Hollow Block Grading System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
hardware_grade = st.selectbox("Hardware Grade", ["A", "B", "C"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    x = preprocess(image)

    outputs = session.run(None, {input_name: x})[0][0]
    probs = softmax(outputs)

    dl_index = int(np.argmax(probs))
    dl_grade = labels[dl_index]

    final_grade = combine_grades(dl_grade, hardware_grade)

    # =========================
    # OUTPUT
    # =========================
    st.subheader("Results")

    st.write("Deep Learning Grade:", dl_grade)
    st.caption(dl_comments[dl_grade])

    st.write("Hardware Grade:", hardware_grade)
    st.caption(hw_comments[hardware_grade])

    st.success(f"Final Combined Grade: {final_grade}")

    st.subheader("Confidence Scores")
    for i, label in enumerate(labels):
        st.write(f"{label}: {probs[i] * 100:.2f}%")
