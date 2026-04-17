import streamlit as st
import numpy as np
from PIL import Image
import tf_keras as keras
from tf_keras.models import load_model
from tf_keras.applications.mobilenet import preprocess_input

st.set_page_config(
    page_title="Hollow Block Grading Classifier",
    page_icon="🧱",
    layout="centered"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 680px; }
    .grade-a { color: #3B6D11; font-weight: 600; font-size: 1.4rem; }
    .grade-b { color: #BA7517; font-weight: 600; font-size: 1.4rem; }
    .grade-c { color: #A32D2D; font-weight: 600; font-size: 1.4rem; }
    .stProgress > div > div > div { border-radius: 99px; }
    div[data-testid="stImage"] img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_my_model():
    return load_model("mobilenet_hollowblock.keras")

model = load_my_model()

CLASS_NAMES = ["Grade A", "Grade B", "Grade C"]
CLASS_COLORS = {"Grade A": "grade-a", "Grade B": "grade-b", "Grade C": "grade-c"}
CLASS_DESCRIPTIONS = {
    "Grade A": ("Excellent quality", "Passes all standards. Suitable for structural use.", "#639922"),
    "Grade B": ("Acceptable quality", "Minor imperfections. Suitable for non-critical use.", "#EF9F27"),
    "Grade C": ("Below standard", "Does not meet quality specifications.", "#E24B4A"),
}
STATUS_FN = {
    "Grade A": st.success,
    "Grade B": st.warning,
    "Grade C": st.error,
}
STATUS_MSG = {
    "Grade A": "Block passed quality check",
    "Grade B": "Block has minor imperfections",
    "Grade C": "Block failed quality check",
}

st.markdown("## 🧱 Hollow block grading classifier")
st.caption("Upload a photo of a hollow block to classify its quality grade using a MobileNet model.")

with st.expander("Grade reference guide"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Grade A — Excellent**")
        st.caption("Passes all quality standards. Suitable for structural use.")
    with col2:
        st.markdown("**Grade B — Acceptable**")
        st.caption("Minor surface imperfections. Suitable for non-critical applications.")
    with col3:
        st.markdown("**Grade C — Below standard**")
        st.caption("Does not meet quality specifications. Not recommended for use.")

st.divider()

uploaded_file = st.file_uploader(
    "Upload block image",
    type=["jpg", "jpeg", "png"],
    help="For best results, photograph the block on a flat, well-lit surface."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with col_result:
        with st.spinner("Classifying..."):
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)

        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        css_class = CLASS_COLORS[predicted_class]
        desc_title, desc_body, bar_color = CLASS_DESCRIPTIONS[predicted_class]

        st.markdown("**Classification result**")
        st.markdown(f'<p class="{css_class}">{predicted_class}</p>', unsafe_allow_html=True)
        st.caption(f"{desc_title} — {desc_body}")

        st.markdown(f"**Confidence: {confidence:.0%}**")
        st.progress(confidence)

    st.divider()
    st.markdown("**Class probabilities**")

    prob_cols = st.columns(3)
    grade_colors = ["#639922", "#EF9F27", "#E24B4A"]

    for i, (col, name, prob) in enumerate(zip(prob_cols, CLASS_NAMES, prediction[0])):
        with col:
            st.metric(label=name, value=f"{prob:.0%}")

    for i, (name, prob) in enumerate(zip(CLASS_NAMES, prediction[0])):
        st.progress(float(prob), text=name)

    st.divider()
    STATUS_FN[predicted_class](STATUS_MSG[predicted_class])

else:
    st.info("Upload a hollow block image above to get started.")
    st.caption("Tip: Photograph the block on a flat, well-lit surface from directly above or at a slight angle. Avoid shadows across the face of the block.")
