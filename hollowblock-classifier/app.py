import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# =========================
# Labels
# =========================
labels = ["Grade A", "Grade B", "Grade C"]

grade_to_num = {
    "A": 2,
    "B": 1,
    "C": 0
}

num_to_grade = {
    2: "Grade A",
    1: "Grade B",
    0: "Grade C"
}

# =========================
# Load model
# =========================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_hollowblock.onnx")

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# =========================
# Preprocess
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# =========================
# Softmax
# =========================
def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# =========================
# Combine logic
# =========================
def combine_grades(dl_grade, hw_grade):
    avg = round((dl_grade + hw_grade) / 2)
    return num_to_grade[avg]

# =========================
# UI
# =========================
st.title("Hybrid Hollow Block Grading System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

hardware_grade = st.selectbox("Select Hardware Grade", ["A", "B", "C"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_data = preprocess_image(image)

    outputs = session.run(None, {input_name: input_data})[0][0]
    probs = softmax(outputs)

    dl_pred = int(np.argmax(probs))
    dl_grade = num_to_grade[dl_pred]

    hw_num = grade_to_num[hardware_grade]

    final_grade = combine_grades(dl_pred, hw_num)

    st.subheader("Results")

    st.write("Deep Learning Grade:", dl_grade)
    st.write("Hardware Grade:", f"Grade {hardware_grade}")

    st.success(f"Final Combined Grade: {final_grade}")

    st.subheader("Confidence Scores")
    for i, label in enumerate(labels):
        st.write(f"{label}: {probs[i]*100:.2f}%")
