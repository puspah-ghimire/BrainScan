import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model1.h5")

# Class names (training order)
classes = ["glioma", "meningioma", "pituitary", "notumor"]

# Set custom web page title
st.set_page_config(page_title="BrainScan")

# Streamlit app
st.title("BrainScan")
st.markdown(
    "A deep learning model to detect and classify brain tumors from MRI images."
)


uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    with st.spinner("Processing..."):
        # Show uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        # Preprocess image
        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)[0]

        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Show result
        st.success(f"Prediction: {predicted_class}")
        st.write(f"Confidence: **{confidence:.2f}%**")