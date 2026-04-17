import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("mobilenet_hollowblock.onnx")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

st.title("MobileNet ONNX Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def preprocess(image):
    image = image.resize((224, 224))  # MobileNet default
    img = np.array(image).astype(np.float32)

    # Normalize (adjust if your training used different preprocessing)
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_data = preprocess(image)

    # Run inference
    outputs = session.run([output_name], {input_name: input_data})

    st.write("Raw Output:", outputs[0])
