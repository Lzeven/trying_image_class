import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# LOAD MODEL
model = tf.keras.models.load_model(
    "mobilNetModelPenorPcel.keras",
    compile=False
)

# CLASS LABELS
class_names = ['pen', 'pencil']

# PAGE SETTINGS
st.set_page_config(
    page_title="Pen vs Pencil Classifier",
    page_icon="✏️"
)

# TITLE
st.title("✏️ Pen vs Pencil Classifier")

st.write(
    "Upload an image and AI will classify it."
)

# UPLOAD IMAGE
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # OPEN IMAGE
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # PREPROCESS
    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # PREDICT
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = prediction[0][predicted_index] * 100

    # RESULTS
    st.subheader("Prediction")

    st.success(
        f"Class: {predicted_class.upper()}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )