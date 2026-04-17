import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# Load model
@st.cache_resource
def load_my_model():
    return load_model("mobilenet_hollowblock.keras")

model = load_my_model()

# CHANGE THIS to match your training labels
class_names = ["Grade A", "Grade B", "Reject"]

st.set_page_config(page_title="Hollow Block Classifier", layout="centered")

st.title("🧱 Hollow Block Grading Classifier")
st.write("Upload an image or take a photo to classify the hollow block.")

# Upload OR Camera
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("Or take a photo")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # Output
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

    # Show all probabilities
    st.write("### Class Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.2f}")

    # Pass/fail logic
    if predicted_class.lower() == "reject":
        st.error("❌ Block failed quality check")
    else:
        st.success("✅ Block passed quality check")
