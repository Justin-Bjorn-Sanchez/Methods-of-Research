import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# =========================
# Labels
# =========================
labels = ["Grade A", "Grade B", "Grade C"]

grade_to_num = {"A": 2, "B": 1, "C": 0}
num_to_grade = {2: "A", 1: "B", 0: "C"}

# =========================
# SAFE COMBINATION RULE
# (A+B=B, A+C=B, B+C=C, etc.)
# =========================
def combine_grades(dl, hw):
    dl = int(dl)
    hw = int(hw)

    # safe fallback (prevents crash)
    return num_to_grade[min(dl, hw)]

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
# UI
# =========================
st.title("Hybrid Hollow Block Grading System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

hardware_grade = st.selectbox("Hardware Grade", ["A", "B", "C"])
hw_num = grade_to_num[str(hardware_grade)]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    # preprocess
    x = preprocess(image)

    # ONNX inference
    outputs = session.run(None, {input_name: x})[0][0]
    probs = softmax(outputs)

    dl = int(np.argmax(probs))
    dl_label = labels[dl]

    # combine
    final_grade = combine_grades(dl, hw_num)

    # =========================
    # RESULTS
    # =========================
    st.subheader("Results")

    st.write("Deep Learning Grade:", dl_label)
    st.write("Hardware Grade:", f"Grade {hardware_grade}")

    st.success(f"Final Combined Grade: {final_grade}")

    st.subheader("Confidence Scores")
    for i, label in enumerate(labels):
        st.write(f"{label}: {probs[i] * 100:.2f}%")
